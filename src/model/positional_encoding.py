import math

import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for grid-structured features.
    Adds spatial positional information to feature embeddings.
    """

    def __init__(self, d_model: int, max_h: int = 256, max_w: int = 256) -> None:
        super().__init__()

        if d_model % 4 != 0:
            raise ValueError(f"{d_model} % 4 != 0")

        # Create position encoding for height dimension
        d_model_half = d_model // 2
        div_term_h = torch.exp(
            torch.arange(0, d_model_half, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model_half)
        )

        # Create position encoding for width dimension
        div_term_w = torch.exp(
            torch.arange(0, d_model_half, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model_half)
        )

        # Initialize positional encoding tensors
        pe_h = torch.zeros(max_h, max_w, d_model_half, dtype=torch.float32)
        pe_w = torch.zeros(max_h, max_w, d_model_half, dtype=torch.float32)

        # Fill positional encoding tensors
        pos_h = torch.arange(0, max_h, dtype=torch.float32)[:, None, None]
        pos_w = torch.arange(0, max_w, dtype=torch.float32)[None, :, None]

        # Apply sine and cosine to even and odd indices
        pe_h[:, :, 0::2] = torch.sin(pos_h * div_term_h)
        pe_h[:, :, 1::2] = torch.cos(pos_h * div_term_h)
        pe_w[:, :, 0::2] = torch.sin(pos_w * div_term_w)
        pe_w[:, :, 1::2] = torch.cos(pos_w * div_term_w)

        # Combine height and width positional encodings
        pe = torch.cat([pe_h, pe_w], dim=2).unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = x.size()
        pe = self.pe[:, :h, :w, :].to(dtype=x.dtype, device=x.device)
        return x + pe
