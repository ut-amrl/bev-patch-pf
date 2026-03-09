import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.aer_decoder import AerialDecoder
from model.aer_patch_sampler import AerialPatchSampler
from model.backbone import DINOv3_ConvNeXt_FeatureExtractor
from model.bev_encoder import BevEncoder
from model.bev_mapper import LiftSplatBEVMapper
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
            score = score / bev_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)[:, None]

        if self.return_only_score:
            return score, uncertainty

        return score, uncertainty, distinct_logits, bev_desc, aer_patches_desc, uncertainty_map


###
# Main Model
###


class BEVPatchNet(nn.Module):
    def __init__(
        self,
        gnd_backbone_name: str,
        gnd_backbone_out_indices: list[int],
        aer_backbone_name: str,
        aer_backbone_out_indices: list[int],
        bev_shape: tuple[int, int],
        gnd_hidden_dim: int,
        aer_hidden_dims: list[int],
        out_dim: int,
        dropout: float,
        **kwargs: dict[str, any],
    ) -> None:
        super().__init__()
        self.gnd_hidden_dim = gnd_hidden_dim
        self.out_dim = out_dim

        # Ground Encoder & BEV Mapper/Encoder
        self.gnd_backbone = DINOv3_ConvNeXt_FeatureExtractor(
            model_name=gnd_backbone_name, out_indices=gnd_backbone_out_indices, frozen=True
        )
        self.gnd_upernet = UPerNet(in_channels=self.gnd_backbone.feature_dims, out_dim=gnd_hidden_dim, dropout=dropout)
        self.bev_mapper = LiftSplatBEVMapper(feat_dim=gnd_hidden_dim, bev_shape=bev_shape)
        self.bev_encoder = BevEncoder(
            in_dim=gnd_hidden_dim, hidden_dim=gnd_hidden_dim, out_dim=out_dim + 2, dropout=dropout
        )

        # Aerial Encoder
        self.aer_backbone = DINOv3_ConvNeXt_FeatureExtractor(
            model_name=aer_backbone_name, out_indices=aer_backbone_out_indices, frozen=True
        )
        self.aer_decoder = AerialDecoder(
            in_dims=self.aer_backbone.feature_dims, proj_dims=aer_hidden_dims, out_dim=out_dim
        )

        # Patch sampler & Head
        self.patch_sampler = AerialPatchSampler(patch_size=bev_shape[:2], feature_scale=0.25)
        self.head = BevPatchHead()

    def forward(
        self,
        ground_image: torch.Tensor,
        ground_depth: torch.Tensor,
        aerial_image: torch.Tensor,
        info: dict[str, torch.Tensor],
        pose_uvr: torch.Tensor,
        **kwargs: dict[str, any],
    ):
        # 1) BEV feature from ground view
        gnd_emb = self.gnd_upernet(self.gnd_backbone(ground_image))
        bev_emb, bev_mask = self.bev_mapper(gnd_emb, ground_depth, info["K"], info["cam2enu"], info["resolution"])
        bev_raw_feat = self.bev_encoder(bev_emb)

        # 2) Aerial feature
        aer_raw_feat = self.aer_decoder(self.aer_backbone(aerial_image))

        # 3) Patch sampling
        aer_patches_raw_feat = self.patch_sampler(aer_raw_feat, pose_uvr)

        # 4) Head
        score, uncertainty, distinct_logits, bev_feat, aer_patches_feat, uncertainty_map = self.head(
            bev_raw_feat, aer_patches_raw_feat, bev_mask
        )

        output = {
            "score": score.detach(),  # (B, N)
            "uncertainty": uncertainty,  # (B, 1, H, W)
            "distinct_logits": distinct_logits,  # (B, 1, H, W)
            "bev_feat": bev_feat,  # (B, C, H, W)
            "bev_mask": bev_mask,  # (B, 1, H, W)
            "aer_patches_feat": aer_patches_feat,  # (B, N, C, H, W)
            "aer_raw_feat": aer_raw_feat,  # (B, C, Ha, Wa)
            "uncertainty_map": uncertainty_map.detach(),  # (B, 1, H, W)
        }

        return output
