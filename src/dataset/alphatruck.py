from pathlib import Path

import cv2
import numpy as np
import pandas as pd  # noqa F401
import torch
from manifpy import SE2, SE3  # noqa F401

from dataset.common import GeoLocDataset


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

        # # motion updates from GT odometry
        # odom_path = Path(root) / scene / "utm_pose.csv"
        # odom_df = pd.read_csv(odom_path)
        # odom_SE2 = [SE2(*xyr) for xyr in odom_df[["x", "y", "angle"]].values]

        # actions = [np.zeros(3)]
        # for i in range(1, len(odom_df)):
        #     action = odom_SE2[i - 1].between(odom_SE2[i])
        #     dx, dy = action.translation()
        #     dtheta = action.angle()
        #     actions.append(np.array([dx, dy, dtheta]))
        # self.actions = np.array(actions)

        odom_path = Path(root) / scene / "fast_lio.csv"
        odom_df = pd.read_csv(odom_path)
        odom_SE3 = [SE3(pose) for pose in odom_df[["x", "y", "z", "qx", "qy", "qz", "qw"]].values]

        actions = [np.zeros(3)]
        for i in range(1, len(odom_SE3)):
            # compute motion update
            action_SE3 = odom_SE3[i - 1].between(odom_SE3[i])  # SE3
            dx, dy, _ = action_SE3.translation()
            dtheta = self.quat_to_yaw(action_SE3.quat())
            actions.append(np.array([dx, dy, dtheta]))
        self.actions = np.array(actions)

    @staticmethod
    def quat_to_yaw(quat: np.ndarray) -> float:
        qx, qy, qz, qw = quat
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), -1.0 + 2.0 * (qw * qw + qx * qx))
        return yaw

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.update({"timestamp": self.poses[idx, 0], "action": self.actions[idx]})
        return data
