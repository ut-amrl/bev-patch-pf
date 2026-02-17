import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.aer_patch_sampler import AerialPatchSampler
from model.backbone import DINO_FeatureExtractor, SwinV2_FeatureExtractor
from model.bev_encoder import BevEncoder
from model.bev_mapper import LiftSplatBEVMapper
from model.blocks import gn
from model.upernet import UPerNet

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BevPatchHead(nn.Module):
    def __init__(self, return_only_score: bool = False) -> None:
        super().__init__()
        self.return_only_score = return_only_score

    def forward(self, bev_feat: torch.Tensor, aer_patches_feat: torch.Tensor, bev_mask: torch.Tensor) -> tuple:
        """
        Args:
            bev_feat: (B, C + 2, H, W)
              - [:, :-2]  : BEV descriptor
              - [:, -2:-1]: pixel distinctiveness logits
              - [:, -1:]  : frame uncertainty evidence map
            aer_patches_feat: (B, N, C, H, W)
            bev_mask: (B, 1, H, W)

        Returns:
            score: (B, N) - weighted similarity score per patch
            uncertainty: (B,) - frame-level uncertainty (log variance)
            ---
            distinct_logits: (B, 1, H, W) - BEV pixel distinctiveness logits
            bev_desc: (B, C, H, W) - L2-normalized BEV feature
            aer_patches_desc: (B, N, C, H, W) - L2-normalized aerial patches feature
            uncertainty_map: (B, 1, H, W) - BEV frame uncertainty evidence map
        """
        # L2-normalization for descriptors
        bev_desc = F.normalize(bev_feat[:, :-2], p=2, dim=1, eps=1e-6)  # (B, C, H, W)
        aer_patches_desc = F.normalize(aer_patches_feat, p=2, dim=2, eps=1e-6)  # (B, N, C, H, W)

        # extract distinctiveness & uncertainty maps
        distinct_logits = bev_feat[:, -2:-1]  # (B, 1, H, W)
        uncertainty_map = bev_feat[:, -1:]  # (B, 1, H, W)

        # frame uncertainty
        uncertainty = (uncertainty_map * bev_mask).sum(dim=(1, 2, 3)) / bev_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)

        with torch.no_grad():
            w = torch.sigmoid(distinct_logits) * bev_mask  # (B, 1, H, W)
            score = torch.einsum("bchw,bnchw,bhw->bn", bev_desc, aer_patches_desc, w.squeeze(1))
            score /= w.sum(dim=(1, 2, 3)).clamp_min(1e-6)[:, None]

        if self.return_only_score:
            return score, uncertainty

        return score, uncertainty, distinct_logits, bev_desc, aer_patches_desc, uncertainty_map


class PatchProj(nn.Module):
    """Project aerial patch features into compact descriptors for contrastive/redundancy loss."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, pool_hw: int, dropout: float, min_frac: float = 0.5):
        super().__init__()
        self.pool_hw = int(pool_hw)
        self.min_frac = float(min_frac)

        flatten_dim = in_dim * pool_hw * pool_hw
        self.proj = nn.Sequential(
            nn.Conv2d(flatten_dim, hidden_dim, kernel_size=1),
            gn(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, N, C, H, W = x.shape
        P = self.pool_hw

        x_bn = x.reshape(B * N, C, H, W)
        pooled = F.adaptive_avg_pool2d(x_bn, (P, P))  # (BN,C,P,P)

        if mask is None:
            valid_mask = torch.ones(B * N, dtype=torch.bool, device=x.device)
        else:
            frac = mask.reshape(B * N, -1).float().mean(dim=1)  # (BN,)
            valid_mask = frac >= self.min_frac

        flattened = pooled.reshape(B * N, C * P * P, 1, 1)

        feat = self.proj(flattened).flatten(1)  # (BN, D)

        return feat, valid_mask


###
# Main Model
###


class BEVPatchNet(nn.Module):
    def __init__(
        self,
        gnd_backbone_name: str,
        aer_backbone_name: str,
        bev_shape: tuple[int, int],
        aer_img_size: int,
        bev_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        **kwargs: dict[str, any],
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.bev_dim = bev_dim

        # Ground Encoder
        self.gnd_encoder = DINO_FeatureExtractor(model_name=gnd_backbone_name)

        # BEV Mapper & Encoder
        self.bev_mapper = LiftSplatBEVMapper(feat_dim=self.gnd_encoder.out_dim, out_dim=bev_dim, bev_shape=bev_shape)
        self.bev_encoder = BevEncoder(in_dim=bev_dim, hidden_dim=hidden_dim, out_dim=out_dim + 2, dropout=dropout)

        # Aerial Encoder
        self.aer_backbone = SwinV2_FeatureExtractor(model_name=aer_backbone_name, img_size=aer_img_size)
        self.aer_upernet = UPerNet(in_channels=self.aer_backbone.feature_dims, out_dim=hidden_dim)
        self.aer_head = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(hidden_dim, out_dim, stride=1, kernel_size=1, padding=0, bias=False),
        )

        # Patch Sampler & Head
        self.patch_sampler = AerialPatchSampler(bev_shape[:2], scale=4.0)
        self.head = BevPatchHead()

        # (auxiliary) Patch feature projection for contrastive loss
        self.patch_proj = PatchProj(
            in_dim=out_dim, out_dim=128, hidden_dim=512, pool_hw=2, dropout=dropout, min_frac=0.5
        )

    def forward(
        self,
        ground_image: torch.Tensor,
        ground_depth: torch.Tensor,
        aerial_image: torch.Tensor,
        info: dict[str, torch.Tensor],
        pose_uvr: torch.Tensor,
        aerial_image2: torch.Tensor | None = None,
        pose_uvr2: torch.Tensor | None = None,
        **kwargs: dict[str, any],
    ):
        # 1) BEV feature from ground view
        gnd_emb = self.gnd_encoder(ground_image)
        bev_emb, bev_mask = self.bev_mapper(gnd_emb, ground_depth, info["K"], info["cam2enu"], info["resolution"])
        bev_feat = self.bev_encoder(bev_emb)

        # 2) Aerial feature
        aer_embs = self.aer_backbone(aerial_image)
        aer_emb = self.aer_upernet(aer_embs)
        aer_feat = self.aer_head(aer_emb)

        # 3) Patch sampling
        aer_patches_raw_feat, aer_patches_mask = self.patch_sampler(aer_feat, pose_uvr, return_mask=True)

        # 4) Head
        score, uncertainty, distinct_logits, bev_feat, aer_patches_feat, uncertainty_map = self.head(
            bev_feat, aer_patches_raw_feat, bev_mask
        )

        # similarity
        with torch.no_grad():
            similarity = torch.einsum("bchw,bnchw->bnhw", bev_feat, aer_patches_feat)  # (B, N, H, W)

        # similarity margin for distinctiveness loss
        with torch.no_grad():
            B, N, H, W = similarity.shape
            pos, neg = similarity[:, 0], similarity[:, 1:]
            k = min(5, N - 1)
            hard_idx = score[:, 1:].topk(k, dim=1).indices[..., None, None].expand(-1, -1, H, W)
            neg_hard = neg.gather(dim=1, index=hard_idx)
            similarity_margin = pos - neg_hard.mean(dim=1)

        output = {
            "score": score.detach(),  # (B, N)
            "uncertainty": uncertainty,  # (B, 1, H, W)
            "distinct_logits": distinct_logits,  # (B, 1, H, W)
            "bev_feat": bev_feat,  # (B, C, H, W)
            "bev_mask": bev_mask,  # (B, 1, H, W)
            "aer_patches_feat": aer_patches_feat,  # (B, N, C, H, W)
            "aer_patches_mask": aer_patches_mask,  # (B, N, H, W)
            "similarity": similarity.detach(),  # (B, N, H, W)
            "similarity_margin": similarity_margin.detach(),  # (B,)
            "aer_feat": aer_feat.detach(),  # (B, C, Ha, Wa)
            "uncertainty_map": uncertainty_map.detach(),  # (B, 1, H, W)
        }

        ## additional aerial view for VICReg loss (no need for inference)
        if aerial_image2 is not None and pose_uvr2 is not None:
            p1, v1 = self.patch_proj(aer_patches_raw_feat, aer_patches_mask)

            with torch.no_grad():
                offset_scale = info["resolution"] / info["resolution2"]
                aer_feat2 = self.aer_head(self.aer_upernet(self.aer_backbone(aerial_image2)))
                aer_patches_raw_feat2, aer_patches_mask2 = self.patch_sampler(
                    aer_feat2, pose_uvr2, offset_scale=offset_scale, return_mask=True
                )
                aer_patches_feat2 = F.normalize(aer_patches_raw_feat2, p=2, dim=2, eps=1e-6)  # (B, N, C, H, W)
                p2, v2 = self.patch_proj(aer_patches_raw_feat2, aer_patches_mask2)

            output.update(
                {
                    "p1": p1[v1 & v2],  # (M, D)
                    "p2": p2[v1 & v2],  # (M, D)
                    "aer_patches_feat2": aer_patches_feat2.detach(),  # (B, N, C, H, W) - for visualization only
                    "aer_feat2": aer_feat2.detach(),  # (B, C, Ha, Wa) - for visualization only
                }
            )

        return output
