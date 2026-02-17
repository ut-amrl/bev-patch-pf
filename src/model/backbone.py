import timm
import torch
import torch.nn as nn
from transformers import AutoModel


class DINO_FeatureExtractor(nn.Module):
    """DINOv2/v3 model for feature extraction."""

    def __init__(self, model_name: str = "facebook/dinov2-small") -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, device_map=None)
        self.model.eval()

        self.out_dim = self.model.config.hidden_size
        self.offset = 1  # skip cls, register tokens
        if "num_register_tokens" in self.model.config.to_dict():
            self.offset += self.model.config.num_register_tokens

        # freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.model.config.patch_size
        D = self.model.config.hidden_size

        if (H % P) or (W % P):
            raise ValueError(f"Input {H}x{W} not divisible by patch size {P}")

        out = self.model(x)
        patch_tokens = out.last_hidden_state[:, self.offset :]  # drop cls and register tokens
        patch_features = patch_tokens.reshape(B, H // P, W // P, D).permute(0, 3, 1, 2).contiguous()
        return patch_features


class SwinV2_FeatureExtractor(nn.Module):
    """SwinV2 model for feature extraction."""

    def __init__(self, img_size: int, model_name: str = "swinv2_tiny_window8_256") -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            img_size=img_size,
            pretrained=True,
            features_only=True,
            global_pool="",
            num_classes=0,
            out_indices=(0, 1, 2, 3),  # 1/4, 1/8, 1/16, 1/32
        )
        self.model.eval()

        self.feature_dims = self.model.feature_info.channels()  # [96, 192, 384, 768]

        # freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feats = self.model(x)
        feats = tuple(f.permute(0, 3, 1, 2).contiguous() for f in feats)  # (B, C, H, W)
        return feats
