import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import gn


class LiftSplatBEVMapper(nn.Module):
    """Lift-Splat BEV Mapper (pooling vertically with weighted average)

    NOTE: max-pooling with scatter_reduce(reduce="amax") is not supported in TensorRT
    """

    def __init__(self, feat_dim: int, bev_shape: tuple[int, int], **kwargs) -> None:
        super().__init__()
        self.bev_shape = bev_shape

        self.feat_attn = nn.Sequential(
            nn.Conv2d(feat_dim + 1, feat_dim // 2, kernel_size=3, padding=1, bias=False),
            gn(feat_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 1, kernel_size=1, bias=True),
        )
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @torch.no_grad()
    @torch.autocast("cuda", enabled=False)
    def compute_geometry(
        self, depth: torch.Tensor, K: torch.Tensor, cam2enu: torch.Tensor, resolution: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute geometric mappings from 2D pixels to voxel grid.
        NOTE: z coordinate is ignored for BEV mapping for platform invariance.
        """
        B, H, W = depth.shape
        nx, ny = self.bev_shape

        # Cast to fp32 for numerical stability
        depth = depth.float()
        K = K.float()
        cam2enu = cam2enu.float()
        resolution = resolution.unsqueeze(-1).float()  # (B, 1)

        # Create pixel coordinate grid
        us, vs = torch.meshgrid(
            torch.arange(W, device=depth.device, dtype=torch.float32),
            torch.arange(H, device=depth.device, dtype=torch.float32),
            indexing="xy",
        )
        us = us.unsqueeze(0).expand(B, -1, -1)
        vs = vs.unsqueeze(0).expand(B, -1, -1)

        # Back-project pixels to 3D camera coordinates using pinhole model
        xs = (us - K[:, 0, 2].view(B, 1, 1)) * depth / K[:, 0, 0].view(B, 1, 1)
        ys = (vs - K[:, 1, 2].view(B, 1, 1)) * depth / K[:, 1, 1].view(B, 1, 1)
        pts_cam = torch.stack([xs, ys, depth], dim=-1)
        pts_enu = pts_cam.view(B, -1, 3) @ cam2enu[:, :3, :3].transpose(-1, -2) + cam2enu[:, :3, 3].unsqueeze(1)

        # Compute voxel indices
        y_min = -ny * resolution / 2.0
        vx = torch.floor(pts_enu[..., 0] / resolution).to(torch.int64)
        vy = torch.floor((pts_enu[..., 1] - y_min) / resolution).to(torch.int64)
        valid = (vx >= 0) & (vx < nx) & (vy >= 0) & (vy < ny)

        vx = vx.reshape(-1)
        vy = vy.reshape(-1)
        valid = valid.reshape(-1)

        # Compute 2D BEV indices
        batch_off_2d = torch.arange(B, device=depth.device, dtype=torch.int64).unsqueeze(1) * (nx * ny)
        batch_off_2d = batch_off_2d.expand(B, H * W).reshape(-1)
        idx2d = vx * ny + vy + batch_off_2d
        idx2d = idx2d[valid]

        return valid, idx2d

    def forward(
        self, x: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, cam2enu: torch.Tensor, resolution: torch.Tensor
    ):
        """
        Args:
            x: (B, C, Hf, Wf) - input feature map to be lifted (e.g., from ground image backbone)
            depth: (B, H, W) - depth map corresponding to the input feature map
            K: (B, 3, 3) - camera intrinsic matrix
            cam2enu: (B, 4, 4) - camera-to-ENU transformation matrix
            resolution: (B,) - BEV grid resolution (m/px)
        Returns:
            bev_emb: (B, C, H_bev, W_bev) - lift and splatted BEV feature
            bev_mask: (B, 1, H_bev, W_bev) - BEV mask indicating valid regions with at least one projected point
        """
        C = x.shape[1]
        B, H, W = depth.shape
        nx, ny = self.bev_shape

        # lift/splat
        valid, idx2d = self.compute_geometry(depth, K, cam2enu, resolution)  # (B*H*W,), (num_valid,)
        x_up = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)  # (B, C, H, W)

        # compute attention weights for vertical pooling
        clean_depth = torch.nan_to_num(depth, nan=0.0, posinf=100.0, neginf=0.0)
        norm_depth = (clean_depth.unsqueeze(1).clamp(0, 100.0) / 100.0).to(x.dtype)
        log_w = self.feat_attn(torch.cat([x_up, norm_depth], dim=1)) / torch.exp(self.log_temp)  # (B, 1, H, W)

        # vertical pooling with weighted average (softmax)
        valid_feat = x_up.permute(0, 2, 3, 1).reshape(-1, C)[valid].float()
        valid_log_w = log_w.permute(0, 2, 3, 1).reshape(-1, 1)[valid].float()

        weights = torch.exp(valid_log_w - valid_log_w.max())

        weight_sum = torch.zeros(B * nx * ny, 1, device=x.device, dtype=torch.float32)
        weight_sum.scatter_add_(0, idx2d.unsqueeze(-1), weights)
        weighted_feat = valid_feat * weights / weight_sum[idx2d].clamp_min(1e-4)

        bev_emb = torch.zeros((B * nx * ny, C), device=x.device, dtype=torch.float32)
        bev_emb.scatter_add_(0, idx2d.unsqueeze(-1).expand(-1, C), weighted_feat)
        bev_emb = bev_emb.view(B, nx, ny, C).permute(0, 3, 1, 2).to(x.dtype)
        bev_emb = bev_emb.flip(2, 3).contiguous()

        bev_mask = (weight_sum > 1e-6).view(B, nx, ny, 1).permute(0, 3, 1, 2).to(x.dtype)
        bev_mask = bev_mask.flip(2, 3).contiguous()

        return bev_emb, bev_mask
