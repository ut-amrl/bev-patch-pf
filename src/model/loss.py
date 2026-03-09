import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoLocLoss(nn.Module):
    def __init__(
        self,
        init_temp: float,
        min_temp: float,
        lambda_distinct: float,
        distinct_pos_coeff: float,
        distinct_margin_coeff: float,
    ):
        super().__init__()
        # Matching loss
        self.logit_scale_param = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))
        self.min_temp = min_temp
        # Distinctiveness
        self.lambda_distinct = lambda_distinct
        self.distinct_pos_coeff = distinct_pos_coeff
        self.distinct_margin_coeff = distinct_margin_coeff

    def forward(
        self,
        score: torch.Tensor,
        uncertainty: torch.Tensor,
        distinct_logits: torch.Tensor,
        bev_feat: torch.Tensor,
        bev_mask: torch.Tensor,
        aer_patches_feat: torch.Tensor,
        **kwargs: dict[str, any],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            score: (B, N) - weighted similarity score (batch-wise)
            uncertainty: (B,) - frame-level uncertainty (log variance)
            distinct_logits: (B, 1, H, W)
            bev_feat: (B, C, H, W)
            bev_mask: (B, 1, H, W)
            aer_patches_feat: (B, N, C, H, W)
        """
        B, N, C, H, W = aer_patches_feat.shape
        logit_scale = torch.exp(self.logit_scale_param).clamp_max(1.0 / self.min_temp)

        # 1. Uncertainty-weighted matching loss
        w = (torch.sigmoid(distinct_logits) * bev_mask).detach()
        score_matrix = torch.einsum("bchw,knchw,bhw->bkn", bev_feat, aer_patches_feat, w.squeeze(1))  # (B, B, N)
        score_matrix = score_matrix / bev_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)[:, None, None]

        scores = score_matrix.reshape(B, B * N)
        pos_idx = torch.arange(B, device=scores.device) * N

        logp = F.log_softmax(scores * logit_scale, dim=1)  # (B, B * N)
        info_nce = -logp[torch.arange(B, device=scores.device), pos_idx]  # (B,)
        matching_loss = (torch.exp(-uncertainty) * info_nce + uncertainty).mean()

        # 2. Distinctiveness
        with torch.no_grad():
            similarity = torch.einsum("bchw,bnchw->bnhw", bev_feat, aer_patches_feat)  # (B, N, H, W)
            pos = similarity[:, 0]
            hardest_neg_idx = (score[:, 1:].argmax(dim=1) + 1)[:, None, None, None].expand(-1, 1, H, W)
            hardest_neg = torch.gather(similarity, dim=1, index=hardest_neg_idx).squeeze(1)
            margin = pos - hardest_neg
            d_target = torch.sigmoid(self.distinct_pos_coeff * pos) * torch.sigmoid(self.distinct_margin_coeff * margin)

        distinct_loss = F.binary_cross_entropy_with_logits(distinct_logits.squeeze(1), d_target, reduction="mean")

        # total loss
        loss = matching_loss + self.lambda_distinct * distinct_loss

        result = {
            "loss": loss,
            "matching_loss": matching_loss.detach().item(),
            "distinct_loss": distinct_loss.detach().item(),
            "temp": float(1.0 / logit_scale.detach().item()),
        }

        return result
