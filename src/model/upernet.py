"""
reference: https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv1x1, conv3x3, gn


class UPerNet(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(
        self,
        in_channels: tuple[int, ...],
        out_dim: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ppm = PPM(in_channels[-1], out_dim, pool_scales=pool_scales, dropout=dropout)

        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    conv1x1(in_channel, out_dim),
                    gn(out_dim),
                    nn.SiLU(inplace=True),
                )
                for in_channel in in_channels[:-1]
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    conv3x3(out_dim, out_dim),
                    gn(out_dim),
                    nn.SiLU(inplace=True),
                )
                for _ in in_channels[:-1]
            ]
        )

        self.fpn_bottleneck = nn.Sequential(
            conv3x3(len(in_channels) * out_dim, out_dim),
            gn(out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # laterals
        laterals = [l_conv(feats[i]) for i, l_conv in enumerate(self.lateral_convs)]
        laterals.append(self.ppm(feats[-1]))

        # top-down path
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False)
            laterals[i - 1] += up

        # outputs
        fpn_outs = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        fpn_outs.append(laterals[-1])  # ppm output

        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        return output


class PPM(nn.Module):
    """Pooling Pyramid Module
    NOTE: This is a custom implementation that mimics nn.AdaptiveAvgPool2d
    to avoid issues with ONNX export and dynamic input sizes.
    """

    def __init__(
        self, in_dim: int, out_dim: int, pool_scales: tuple[int, ...] = (1, 2, 3, 6), dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    # AdaptiveAvgPool2d cannot be exported to ONNX with dynamic input shapes
                    conv1x1(in_dim, out_dim),
                    gn(out_dim),
                    nn.SiLU(inplace=True),
                )
                for _ in pool_scales
            ]
        )
        self.bottleneck = nn.Sequential(
            conv3x3(in_dim + len(pool_scales) * out_dim, out_dim),
            gn(out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )
        self._export_hw = None
        self._pools = None

    @staticmethod
    def _pool_params(H: int, W: int, s: int) -> tuple[tuple[int, int], tuple[int, int]]:
        sh, sw = H // s, W // s
        kh, kw = H - (s - 1) * sh, W - (s - 1) * sw
        return (kh, kw), (sh, sw)

    def freeze_pools(self, H: int, W: int) -> None:
        """Call this once BEFORE onnx.export with the spatial size"""
        pools = []
        for s in self.pool_scales:
            (kh, kw), (sh, sw) = self._pool_params(H, W, s)
            pools.append(nn.AvgPool2d(kernel_size=(kh, kw), stride=(sh, sw), ceil_mode=False))
        self._pools = nn.ModuleList(pools)
        self._export_hw = (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if self._export_hw is None:
            self._export_hw = (H, W)
        else:
            if (H, W) != self._export_hw:
                raise RuntimeError(f"PPM expected input size {(self._export_hw)}, but got {(H, W)}")

        if self._pools is None:
            self.freeze_pools(H, W)

        ppm_outs = [x]
        for pool, stage in zip(self._pools, self.stages):
            y = pool(x)
            y = stage(y)
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            ppm_outs.append(y)

        return self.bottleneck(torch.cat(ppm_outs, dim=1))
