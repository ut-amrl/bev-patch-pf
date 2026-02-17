from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from geotiff.handler import AerialReframer
from geotiff.manager import get_geotiff_handler
from utils.io import image_to_tensor, normalize_image_tensor, read_image_tensor


class GeoLocDataset(Dataset, ABC):
    """Base class for localization datasets with ground and aerial imagery."""

    # These should be defined in subclasses
    INTRINSIC: torch.Tensor = None
    CAM2ENU: torch.Tensor = None  # camera pose in ENU coordinates
    IMAGE_SIZE: tuple[int, int] = None  # (H, W)

    def __init__(
        self,
        root: str,
        scene: str,
        ground_image_resize: tuple[int, int],
        aerial_image_resize: tuple[int, int],
        geo_tiff_path: str | None = None,
        augment: bool = True,
        sample_poses: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.name = f"{self.__class__.__name__}@{scene}"
        self.scene = scene
        self.ground_image_resize = ground_image_resize
        self.aerial_image_resize = aerial_image_resize
        self.geo_tiff_path = geo_tiff_path
        self.geo_handler = get_geotiff_handler(geo_tiff_path)
        self.augment = augment
        self.sample_poses = sample_poses

        # --- data augmentation ---
        self.center_jitter = kwargs.get("center_jitter", 0.0)
        if self.augment:
            self.scale_bounds = kwargs.get("scale_bounds", (0.0, 0.0))
            self.rotate_augment = kwargs.get("rotate_augment", False)
            self.depth_dropout = kwargs.get("depth_dropout", 0.0)
            color_augment = kwargs.get("color_augment", 0.0)
            if color_augment > 0.0:
                self.color_jitter = transforms.ColorJitter(
                    brightness=0.2 * color_augment,
                    contrast=0.2 * color_augment,
                    saturation=0.5 * color_augment,
                    hue=0.1 * color_augment,
                )

        # --- sampling parameters ---
        if self.sample_poses:
            self.n_poses = kwargs.get("n_poses", 128)
            self.dist_bounds = kwargs.get("dist_bounds", (0.0, 0.0))
            self.r_bounds = kwargs.get("r_bounds", (0.0, 0.0))
            self.radial_power = kwargs.get("radial_power", 1.0)

        # depth, intrinsics resizing
        self.depth_resize = np.array(ground_image_resize)
        self.intrinsic = self.INTRINSIC.clone()
        self.intrinsic[[0, 0], [0, 2]] *= self.depth_resize[1] / self.IMAGE_SIZE[1]
        self.intrinsic[[1, 1], [1, 2]] *= self.depth_resize[0] / self.IMAGE_SIZE[0]

        # Load data paths and poses
        self._load_data(Path(root))

    @abstractmethod
    def _load_data(self, root: Path):
        """Load image paths, depth paths, and poses. Should set:
        - self.image_paths
        - self.depth_paths
        - self.poses
        """
        raise NotImplementedError

    @abstractmethod
    def _load_depth(self, depth_path: Path) -> np.ndarray:
        """Load and preprocess depth data from file."""
        raise NotImplementedError

    def __len__(self):
        return len(self.image_paths)

    def get_ground_image(self, idx):
        ground_image = read_image_tensor(str(self.image_paths[idx]), resize=self.ground_image_resize)

        # color augmentation
        if hasattr(self, "color_jitter"):
            ground_image = self.color_jitter(ground_image)

        ground_image = normalize_image_tensor(ground_image, image_type="ground")
        return ground_image

    def get_ground_depth(self, idx):
        depth = self._load_depth(self.depth_paths[idx])
        depth = cv2.resize(depth, self.depth_resize[::-1], interpolation=cv2.INTER_NEAREST)

        # depth augmentation
        if hasattr(self, "depth_dropout") and self.depth_dropout > 0.0:
            drop_mask = np.random.random(depth.shape) < self.depth_dropout
            depth[drop_mask] = 0

        return torch.from_numpy(depth)

    def get_aerial_image(self, gt_xyr: np.ndarray):
        # scale/rotation augmentation
        scale, rotation = 1.0, 0
        if hasattr(self, "scale_bounds") and self.scale_bounds[0] < self.scale_bounds[1]:
            scale = np.random.uniform(*self.scale_bounds)
        if hasattr(self, "rotate_augment") and self.rotate_augment:
            rotation = np.random.choice([0, 90, 180, 270])

        # center jitter augmentation
        query_coords = gt_xyr[:2].copy()
        query_coords += np.random.uniform(-self.center_jitter, self.center_jitter, size=2) * scale

        # get aerial image
        reframer = AerialReframer.from_query_coords(
            self.geo_handler, query_coords, out_size=self.aerial_image_resize, scale=scale, rotation=rotation
        )
        aerial_image = image_to_tensor(reframer.crop_image())

        # color augmentation
        if hasattr(self, "color_jitter"):
            aerial_image = self.color_jitter(aerial_image)

        # convert to tensor
        aerial_image = normalize_image_tensor(aerial_image, image_type="aerial")
        gt_uvr = torch.tensor(reframer.to_uvr(gt_xyr), dtype=torch.float32)
        resolution = torch.tensor(reframer.resolution, dtype=torch.float32)
        rotation = torch.tensor(rotation, dtype=torch.int32)
        return aerial_image, reframer, gt_uvr, resolution, rotation

    def __getitem__(self, idx):
        gt_xyr = self.poses[idx, 1:4].astype(np.float64)  # (x, y, theta) ENU coordinates

        data = {
            "idx": idx,
            "name": f"{self.__class__.__name__}-{self.scene}-{self.image_paths[idx].stem}",
            "ground_image": self.get_ground_image(idx),
            "ground_depth": self.get_ground_depth(idx),
            "gt_xyr": torch.from_numpy(gt_xyr.copy()),
            "info": {"K": self.intrinsic, "cam2enu": self.CAM2ENU},
        }

        if self.geo_handler is None:
            return data

        aerial_image, reframer, gt_uvr, resolution, rotation = self.get_aerial_image(gt_xyr)
        aerial_image2, reframer2, _, resolution2, _ = self.get_aerial_image(gt_xyr)

        data.update(
            {
                "aerial_image": aerial_image,
                "reframer": reframer,
                "gt_uvr": gt_uvr,
                # (2nd view for vicreg loss)
                "aerial_image2": aerial_image2,
            }
        )
        data["info"].update({"resolution": resolution, "rotation": rotation, "resolution2": resolution2})

        if self.sample_poses:
            pose_xyr = reframer.sample_poses(
                gt_xyr, self.n_poses, self.dist_bounds, self.r_bounds, radial_power=self.radial_power
            )
            pose_uvr = reframer.to_uvr(pose_xyr)
            pose_uvr2 = reframer2.to_uvr(pose_xyr)
            data.update(
                {
                    "pose_xyr": torch.tensor(pose_xyr, dtype=torch.float32),
                    "pose_uvr": torch.tensor(pose_uvr, dtype=torch.float32),
                    "pose_uvr2": torch.tensor(pose_uvr2, dtype=torch.float32),
                }
            )

        return data
