import torch
import torch.nn as nn
import torch.nn.functional as F


class AerialPatchSampler(nn.Module):
    """Sample patches from aerial feature map based on the pose uvr"""

    def __init__(self, patch_size: tuple[int, int], feature_scale: float = 1.0):
        super().__init__()
        Hb, Wb = patch_size
        self.feature_scale = feature_scale  # feature scale factor compared to the original image

        grid_u, grid_v = torch.meshgrid(
            torch.arange(Hb, dtype=torch.float32).flip(0),
            torch.arange(Wb, dtype=torch.float32) - Wb // 2,
            indexing="ij",
        )
        self.register_buffer("grid_u", grid_u.view(1, 1, Hb, Wb), persistent=False)
        self.register_buffer("grid_v", grid_v.view(1, 1, Hb, Wb), persistent=False)

    def forward(self, aer_feat: torch.Tensor, pose_uvr: torch.Tensor):
        """
        Args:
            aer_feat: (B, C, Ha, Wa) aerial feature map
            pose_uvr: (B, N, 3) hypotheses pose - <u, v, r>
        Returns:
            aer_patches_feat: (B, N, C, Hb, Wb) sampled patches
        """
        B, C, Ha, Wa = aer_feat.shape
        _, N, _ = pose_uvr.shape
        Hb, Wb = self.grid_u.shape[2:]

        grid_u0 = self.grid_u.to(device=aer_feat.device)
        grid_v0 = self.grid_v.to(device=aer_feat.device)

        # per-pose sampling grid
        u_offset = pose_uvr[..., 0].reshape(B, N, 1, 1)
        v_offset = pose_uvr[..., 1].reshape(B, N, 1, 1)
        theta = pose_uvr[..., 2].reshape(B, N, 1, 1).float()
        cos_r = torch.cos(-theta)
        sin_r = torch.sin(-theta)

        grid_u = u_offset + cos_r * grid_u0 - sin_r * grid_v0
        grid_v = v_offset + sin_r * grid_u0 + cos_r * grid_v0

        # normalize the grid to [-1, 1]
        g_x = (grid_u * self.feature_scale + 0.5) * (2.0 / Wa) - 1.0
        g_y = (grid_v * self.feature_scale + 0.5) * (2.0 / Ha) - 1.0
        grid_norm = torch.stack((g_x, g_y), dim=-1)  # (B, N, Hb, Wb, 2)
        valid_mask = (grid_norm.abs() < 1).all(dim=-1)  # (B, N, Hb, Wb)
        grid_norm = grid_norm.masked_fill(~valid_mask.unsqueeze(-1), 2.0).to(aer_feat.dtype)  # avoid sampling

        # sample the patches
        patches = F.grid_sample(
            aer_feat.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, Ha, Wa),
            grid_norm.view(B * N, Hb, Wb, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(B, N, C, Hb, Wb)

        return patches
