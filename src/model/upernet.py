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
        self, in_channels: tuple[int, ...], out_dim: int, pool_scales: tuple[int, ...] = (1, 2, 3, 6), **kwargs
    ) -> None:
        super().__init__()

        self.ppm = PPM(in_channels[-1], out_dim, pool_scales)
        self.ppm_bottleneck = nn.Sequential(
            conv3x3(in_channels[-1] + len(pool_scales) * out_dim, out_dim),
            gn(out_dim),
            nn.SiLU(inplace=True),
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in in_channels[:-1]:
            l_conv = nn.Sequential(
                conv1x1(in_channel, out_dim),
                gn(out_dim),
                nn.SiLU(inplace=True),
            )
            fpn_conv = nn.Sequential(
                conv3x3(out_dim, out_dim),
                gn(out_dim),
                nn.SiLU(inplace=True),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            conv3x3(len(in_channels) * out_dim, out_dim),
            gn(out_dim),
            nn.SiLU(inplace=True),
        )

    def freeze_ppm_for_export(self, H: int, W: int) -> None:
        """Call this once BEFORE onnx.export with the spatial size"""
        self.ppm.freeze_pools(H, W)

    def psp_forward(self, x: torch.Tensor) -> torch.Tensor:
        psp_outs = [x]
        psp_outs.extend(self.ppm(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.ppm_bottleneck(psp_outs)
        return output

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # build laterals
        laterals = [lateral_conv(feats[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(feats[-1]))

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode="bilinear", align_corners=False)

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])

        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        return output


class PPM(nn.Module):
    """Pooling Pyramid Module
    NOTE: This is a custom implementation that mimics nn.AdaptiveAvgPool2d
    to avoid issues with ONNX export and dynamic input sizes.
    """

    def __init__(self, in_dim: int, out_dim: int, pool_scales: tuple[int, ...] = (1, 2, 3, 6)) -> None:
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        H, W = x.shape[-2:]
        if self._export_hw is None:
            self._export_hw = (H, W)
        else:
            if (H, W) != self._export_hw:
                raise RuntimeError(f"PPM expected input size {(self._export_hw)}, but got {(H, W)}")

        if self._pools is None:
            self.freeze_pools(H, W)

        outs = []
        for pool, stage in zip(self._pools, self.stages):
            y = pool(x)
            y = stage(y)
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(y)

        return outs
