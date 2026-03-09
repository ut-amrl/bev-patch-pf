import torch
import torch.nn as nn

from .blocks import DecoderBlock, ResidualBlock, conv1x1, conv3x3, gn
from .positional_encoding import PositionalEncoding2D
from .upernet import UPerNet


class BevEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        # Initial Residual Blocks
        self.residual1 = ResidualBlock(in_channels=in_dim, out_channels=hidden_dim, stride=2)
        self.residual2 = ResidualBlock(in_channels=hidden_dim, out_channels=hidden_dim, stride=2)

        # Self-attention
        self.pos_enc = PositionalEncoding2D(hidden_dim, max_h=128, max_w=128)
        self.decoder = nn.ModuleList([DecoderBlock(hidden_dim, n_heads=8, drop=dropout) for _ in range(3)])

        # UPerNet
        self.upernet = UPerNet(in_channels=(in_dim, hidden_dim, hidden_dim), out_dim=hidden_dim, dropout=dropout)

        # Head
        self.head = nn.Sequential(
            conv3x3(hidden_dim, hidden_dim),
            gn(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout),
            conv1x1(hidden_dim, out_dim, bias=True),
        )

    def forward(self, bev_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_emb: (B, C, Hb, Wb) BEV embedding
        Returns:
            bev_feat: (B, out_dim, Hb, Wb) BEV feature
        """
        B, C, H, W = bev_emb.shape
        H4, W4 = H // 4, W // 4

        # residual blocks
        b1 = self.residual1(bev_emb)
        b2 = self.residual2(b1)

        # self-attention
        b2 = self.pos_enc(b2.permute(0, 2, 3, 1)).reshape(B, H4 * W4, -1)
        for block in self.decoder:
            b2 = block(b2, b2)
        b2 = b2.reshape(B, H4, W4, -1).permute(0, 3, 1, 2)  # (B, C, H/4, W/4)

        # UPerNet & head
        bev_neck_feat = self.upernet([bev_emb, b1, b2])
        bev_feat = self.head(bev_neck_feat)
        return bev_feat
