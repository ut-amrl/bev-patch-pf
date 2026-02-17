import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoLoc_Loss(nn.Module):
    def __init__(
        self,
        init_temp: float,
        min_temp: float,
        lambda_distinctiveness: float,
        d_alpha: float,
        lambda_aer_vicreg: float,
    ):
        super().__init__()
        # BEV-aerial similarity loss
        self.logit_scale_param = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))
        self.max_scale = 1.0 / min_temp
        # BEV feature distinctiveness loss
        self.lambda_distinctiveness = lambda_distinctiveness
        self.d_alpha = d_alpha
        # aerial patch feature regularization params
        self.lambda_aer_vicreg = lambda_aer_vicreg
        self.vicreg_loss = VICRegLoss()

    def forward(
        self,
        uncertainty: torch.Tensor,
        distinct_logits: torch.Tensor,
        bev_feat: torch.Tensor,
        bev_mask: torch.Tensor,
        aer_patches_feat: torch.Tensor,
        aer_patches_mask: torch.Tensor,
        similarity: torch.Tensor,
        similarity_margin: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            distinct_logits: (B, 1, H, W)
            uncertainty: (B,) - frame-level uncertainty (log variance)
            bev_feat: (B, C, H, W)
            bev_mask: (B, 1, H, W)
            aer_patches_feat: (B, N, C, H, W)
            aer_patches_mask: (B, N, H, W)
            similarity: (B, N, H, W) - BEV-aerial patch similarity map
            similarity_margin: (B,) - BEV-aerial patch similarity margin (pos - hard neg)
            p1: (M, D) - projected aerial patch descriptors (view 1)
            p2: (M, D) - projected aerial patch descriptors (view 2)
        """
        B, N, C, H, W = aer_patches_feat.shape
        logit_scale = self.logit_scale_param.exp().clamp(max=self.max_scale)

        # 1. similarity loss (local + batch-level)
        w = (torch.sigmoid(distinct_logits) * bev_mask).detach()
        score_matrix = torch.einsum("bchw,knchw,bhw->bkn", bev_feat, aer_patches_feat, w.squeeze(1))
        score_matrix /= w.sum(dim=(1, 2, 3)).clamp_min(1e-6)[:, None, None]

        scores = score_matrix.reshape(B, B * N)
        s_target = torch.arange(B, device=scores.device) * N
        ce_similarity = F.cross_entropy(scores * logit_scale, s_target, reduction="none")  # (B,)
        similarity_loss = (torch.exp(-uncertainty) * ce_similarity + uncertainty).mean()

        # 2. distinctiveness loss (margin-based)
        with torch.no_grad():
            d_target = torch.sigmoid(self.d_alpha * similarity_margin).clamp(1e-3, 1.0 - 1e-3)  # (B, H, W)

        bce_distinct = F.binary_cross_entropy_with_logits(distinct_logits.squeeze(1), d_target, reduction="none")
        valid_d = aer_patches_mask[:, 0].float()  # (B, H, W)
        distinct_loss = ((bce_distinct * valid_d).sum(dim=(1, 2)) / valid_d.sum(dim=(1, 2)).clamp_min(1.0)).mean()

        # 3. aerial patch feature VICReg loss
        aer_vicreg_loss = self.vicreg_loss(p1, p2)

        # total loss
        loss = similarity_loss + self.lambda_distinctiveness * distinct_loss + self.lambda_aer_vicreg * aer_vicreg_loss

        result = {
            "loss": loss,
            "similarity_loss": float(similarity_loss.detach().item()),
            "distinct_loss": float(distinct_loss.detach().item()),
            "aer_vicreg_loss": float(aer_vicreg_loss.detach().item()),
            "temp": float((1.0 / logit_scale).detach().item()),
        }

        return result


class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (M, D) - Projected features from Aerial View 1
            y: (M, D) - Projected features from Aerial View 2
        """
        M, D = x.shape

        # invariance loss
        repr_loss = F.mse_loss(x, y)

        # variance loss
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(self.gamma - std_x)) + torch.mean(F.relu(self.gamma - std_y))

        # covariance loss
        x_centered = x - x.mean(dim=0)
        y_centered = y - y.mean(dim=0)
        cov_x = (x_centered.T @ x_centered) / (M - 1)
        cov_y = (y_centered.T @ y_centered) / (M - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum() / D + self.off_diagonal(cov_y).pow_(2).sum() / D

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

    def off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
