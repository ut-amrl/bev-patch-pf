from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_ntuple(val: int | Iterable[int], n: int) -> tuple[int, ...]:
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    return n * (val,)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def gn(c: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2):
        if c % g == 0 and (c // g) >= 8:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)  # fallback (LayerNorm-style)


# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.SiLU,
        norm_layer: nn.Module | None = None,
        bias: bool | tuple[bool, bool] = False,
        drop: float | tuple[float, float] = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_ntuple(bias, 2)
        drop_probs = to_ntuple(drop, 2)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.gn1 = gn(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.gn2 = gn(out_channels)

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.skip(x)
        return out


class Attention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"{dim} not divisible by {n_heads}")

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        hidden_dim = (dim // n_heads) * self.n_heads

        self.Wq = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.Wkv = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(hidden_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        B, Nq, C = query.shape
        _, Ns, _ = source.shape

        q = self.Wq(query)
        k, v = self.Wkv(source).chunk(2, dim=-1)

        # multi-head attention
        q = q.reshape(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Ns, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Ns, self.n_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        out = out.transpose(1, 2).reshape(B, Nq, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_drop=drop, proj_drop=drop)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.attn(self.norm1(x), self.norm2(y))
        x = x + self.mlp(self.norm3(x))
        return x
