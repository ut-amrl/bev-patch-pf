from pathlib import Path

import cv2
import numpy as np
import torch

from dataset.common import GeoLocDataset
from dataset.utils import compute_se3_actions, load_csv_columns, se3_poses_from_rows

FAST_LIO_POSE_COLUMNS = ("x", "y", "z", "qx", "qy", "qz", "qw")


class AlphaTruckDataset(GeoLocDataset):
    INTRINSIC = torch.tensor(
        [
            [646.88623046875, 0.0, 634.9051513671875],
            [0.0, 646.12158203125, 364.38299560546875],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    CAM2ENU = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    IMAGE_SIZE = (720, 1280)  # H, W

    def _load_data(self, root: Path):
        """Load ARL Jackal specific data paths."""
        image_dir = root / self.scene / "image"
        self.image_paths = sorted(image_dir.glob("*.png"))

        depth_dir = root / self.scene / "depth"
        self.depth_paths = sorted(depth_dir.glob("*.png"))

        pose_file = root / self.scene / "utm_pose.csv"
        self.poses = np.genfromtxt(pose_file, delimiter=",", skip_header=1)
        assert len(self.image_paths) == len(self.depth_paths) == len(self.poses), (
            f"{len(self.image_paths)} != {len(self.depth_paths)} != {len(self.poses)}"
        )

    def _load_depth(self, depth_path: Path) -> np.ndarray:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0  # convert to meters
        depth[(depth > 65.5) | (depth < 0.1) | np.isnan(depth)] = 0
        return depth

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data


class AlphaTruckSequence(AlphaTruckDataset):
    def __init__(self, root: str, scene: str, **kwargs):
        super().__init__(root, scene, **kwargs)

        odom_path = Path(root) / scene / "fast_lio.csv"
        odom_data = load_csv_columns(odom_path, FAST_LIO_POSE_COLUMNS)
        odom_poses = se3_poses_from_rows(odom_data, camera_frame=False)
        self.actions = compute_se3_actions(odom_poses)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.update({"timestamp": self.poses[idx, 0], "action": self.actions[idx]})
        return data
