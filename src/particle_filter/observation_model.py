from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from geotiff.handler import AerialReframer, GeoTiffHandler
from utils.io import image_to_tensor, normalize_image_tensor

if TYPE_CHECKING:
    from particle_filter import Particle


class ObservationModel:
    def __init__(
        self,
        model: nn.Module,
        aerial_image_resize: tuple[int, int],
        var_ema_alpha: float,
        var_temp: float,
        likelihood_temp: float,
        geo_handler: GeoTiffHandler | None = None,
        **kwargs: dict[str, any],
    ) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device

        self.aerial_image_resize = aerial_image_resize

        self.var_ema = 1.0
        self.var_ema_alpha = var_ema_alpha
        self.var_temp = var_temp
        self.likelihood_temp = likelihood_temp

        self.geo_handler = geo_handler

    def __call__(
        self,
        particles: list[Particle],
        ground_image: torch.Tensor,
        ground_depth: torch.Tensor,
        info: dict,
        est_xyr: np.ndarray,
        results: dict | None = None,
    ) -> np.ndarray:
        if self.geo_handler is None:
            raise RuntimeError("GeoTiffHandler is not initialized")

        # Get local aerial image based on estimated coordinates
        reframer = AerialReframer.from_query_coords(self.geo_handler, est_xyr, out_size=self.aerial_image_resize)

        # NOTE: need image normalization
        aer_image = image_to_tensor(reframer.crop_image(), device=self.device)
        aer_image = normalize_image_tensor(aer_image, image_type="aerial").unsqueeze(0)

        # Reframe particles to UVR coordinates
        particles_xyr = np.array([[p.x, p.y, p.theta] for p in particles])
        particles_uvr = reframer.to_uvr(particles_xyr)
        particles_uvr = torch.from_numpy(particles_uvr).to(self.device).unsqueeze(0)

        # load data to device
        ground_image = ground_image.to(self.device)
        ground_depth = ground_depth.to(self.device)
        info.update({"resolution": torch.tensor([self.geo_handler.resolution])})
        info = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in info.items()}

        # forward pass
        with torch.inference_mode():
            output = self.model(ground_image, ground_depth, aer_image, info, particles_uvr)
            scores = output["score"][0]  # (N,)
            var = torch.exp(output["uncertainty"])  # (1,)

        # update variance EMA and compute adaptive alpha
        self.var_ema += self.var_ema_alpha * (var.item() - self.var_ema)
        alpha = 1.0 / (1.0 + (self.var_ema / self.var_temp) ** 2.0)

        # compute likelihood
        log_likelihood = (alpha * scores / self.likelihood_temp).cpu().numpy()  # (N,)

        # (debug)
        if results is not None:
            results.update({k: v.detach() for k, v in output.items() if isinstance(v, torch.Tensor)})
            results.update(
                {
                    "particles_uvr": particles_uvr.detach(),
                    "log_likelihood": log_likelihood,
                    "var": float(var.item()),
                    "var_ema": float(self.var_ema),
                    "alpha": float(alpha),
                    "aerial_image": aer_image.detach(),
                    "reframer": reframer,
                },
            )

        return log_likelihood
