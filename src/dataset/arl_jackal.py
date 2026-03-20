from pathlib import Path

import cv2
import numpy as np
import torch

from dataset.common import GeoLocDataset
from dataset.utils import compute_se3_actions, interpolate_poses, load_csv_columns, load_timestamps, se3_poses_from_rows

GT_POSE_COLUMNS = ("ts", "x", "y", "angle")
RAW_ODOM_COLUMNS = ("timestamp", "x", "y", "z", "qx", "qy", "qz", "qw")


class ARLJackalDataset(GeoLocDataset):
    INTRINSIC = torch.tensor(
        [
            [640.51513672, 0.0, 647.99377441],
            [0.0, 639.59344482, 359.21810913],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    CAM2ENU = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    IMAGE_SIZE = (720, 1280)  # H, W

    def _load_data(self, root: Path):
        """Load ARL Jackal specific data paths."""
        image_dir = root / self.scene / "image"
        self.image_paths = sorted([*image_dir.glob("*.jpg"), *image_dir.glob("*.jpeg"), *image_dir.glob("*.png")])

        depth_dir = root / self.scene / "depth"
        self.depth_paths = sorted(depth_dir.glob("*.png"))

        pose_file = root / self.scene / "utm_pose.csv"
        self.poses = load_csv_columns(pose_file, GT_POSE_COLUMNS)

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


class ARLJackalSequence(ARLJackalDataset):
    def __init__(self, root: str, scene: str, **kwargs):
        super().__init__(root, scene, **kwargs)

        # load timestamps
        timestamp_path = Path(root) / scene / "timestamps.txt"
        self.timestamps = load_timestamps(timestamp_path)
        assert len(self.timestamps) == len(self.image_paths), f"{len(self.timestamps)} != {len(self.image_paths)}"

        # motion updates from EKF Odometry
        odom_path = Path(root) / scene / "odom_platform.csv"
        odom_data = load_csv_columns(odom_path, RAW_ODOM_COLUMNS)
        odom_poses = se3_poses_from_rows(odom_data[:, 1:8], camera_frame=False)
        interpolated_poses = interpolate_poses(odom_data[:, 0], odom_poses, self.timestamps)
        self.actions = compute_se3_actions(interpolated_poses)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.update({"timestamp": self.timestamps[idx], "action": self.actions[idx]})
        return data
