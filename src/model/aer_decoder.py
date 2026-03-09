import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv1x1, conv3x3, gn


class AerialDecoder(nn.Module):
    def __init__(self, in_dims: list[int], proj_dims: list[int], out_dim: int) -> None:
        super().__init__()
        if len(in_dims) != len(proj_dims):
            raise ValueError(f"Number of input and projection dims must match. {len(in_dims)} != {len(proj_dims)}")

        self.projs = nn.ModuleList([conv1x1(in_dim, proj_dim) for in_dim, proj_dim in zip(in_dims, proj_dims)])
        hidden_dim = sum(proj_dims)
        self.head = nn.Sequential(
            conv3x3(hidden_dim, hidden_dim),
            gn(hidden_dim),
            nn.SiLU(inplace=True),
            conv1x1(hidden_dim, out_dim, bias=True),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        target_hw = feats[0].shape[-2:]

        outs = [self.projs[0](feats[0])]
        for i, feat in enumerate(feats[1:], start=1):
            x = self.projs[i](feat)
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            outs.append(x)

        neck_feat = torch.cat(outs, dim=1)
        return self.head(neck_feat)
