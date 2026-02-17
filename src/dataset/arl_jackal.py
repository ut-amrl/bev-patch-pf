from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from manifpy import SE2

from dataset.common import GeoLocDataset


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
        # load RGB images
        image_dir = root / self.scene / "image"
        self.image_paths = sorted(image_dir.glob("*.png"))
        # load depth images
        depth_dir = root / self.scene / "depth"
        self.depth_paths = sorted(depth_dir.glob("*.png"))
        # load ground truth poses (UTM coordinates)
        pose_file = root / self.scene / "utm_pose.csv"
        self.poses = np.genfromtxt(pose_file, delimiter=",", skip_header=1)
        assert len(self.image_paths) == len(self.depth_paths) == len(self.poses)

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
        self.timestamps = np.loadtxt(timestamp_path)
        assert len(self.timestamps) == len(self.image_paths), f"{len(self.timestamps)} != {len(self.image_paths)}"

        # motion updates from EKF Odometry
        odom_path = Path(root) / scene / "odom_platform.csv"
        odom_df = pd.read_csv(odom_path)
        odom_SE2 = [SE2(*xyr) for xyr in odom_df[["x", "y", "angle"]].values]

        actions = [np.zeros(3)]
        for i in range(1, len(odom_df)):
            action = odom_SE2[i - 1].between(odom_SE2[i])
            dx, dy = action.translation()
            dtheta = action.angle()
            actions.append(np.array([dx, dy, dtheta]))
        self.actions = np.array(actions)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.update({"timestamp": self.timestamps[idx], "action": self.actions[idx]})
        return data
