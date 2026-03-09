import timm
import torch
import torch.nn as nn


class DINOv3_ConvNeXt_FeatureExtractor(nn.Module):
    """DINOv3 ConvNeXt model for feature extraction."""

    def __init__(
        self,
        model_name: str,
        out_indices: list[int],
        frozen: bool = True,
        **kwargs: dict[str, any],
    ) -> None:
        super().__init__()
        self.frozen = frozen

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices,  # 1/4, 1/8, 1/16, 1/32
        )

        self.feature_dims = self.model.feature_info.channels()

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.frozen:
            with torch.no_grad():
                feats = self.model(x)
        else:
            feats = self.model(x)

        return [f.contiguous() for f in feats]
