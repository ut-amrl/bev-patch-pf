from pathlib import Path

import cv2
import numpy as np
import torch
from manifpy import SE3

from dataset.common import GeoLocDataset


class TartanDriveDataset(GeoLocDataset):
    INTRINSIC = torch.tensor(
        [
            [477.60495, 0.0, 499.5],
            [0.0, 477.60495, 252.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    CAM2ENU = torch.tensor(
        [
            [0.00328796, -0.22472805, 0.97441597, 0.0],
            [-0.99971659, -0.02371374, -0.00209573, 0.0],
            [0.02357802, -0.97413293, -0.22474233, 1.8],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    IMAGE_SIZE = (544, 1024)

    def __init__(self, root: str, scene: str, **kwargs):
        super().__init__(root, scene, **kwargs)

    def _load_data(self, root: Path):
        """Load TartanDrive2.0 specific data paths."""
        image_dir = root / self.scene / "color"
        self.image_paths = sorted(image_dir.glob("*.png"))

        depth_dir = root / self.scene / "depth"
        self.depth_paths = sorted(depth_dir.glob("*.png"))

        pose_file = root / self.scene / "odom.csv"
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


class TartanDriveSequence(TartanDriveDataset):
    def __init__(self, root: str, scene: str, **kwargs):
        super().__init__(root, scene, **kwargs)

        ## motion updates from Visual Odometry
        # convert to SE3 (CAM -> ENU coordinates)
        odom_path = Path(root) / scene / "pycuvslam.csv"
        odom_data = np.genfromtxt(odom_path, delimiter=",", skip_header=1)
        CAM2ENU = SE3(np.array([0, 0, 0]), np.array([0.5, -0.5, 0.5, -0.5]))
        ENU2CAM = SE3(np.array([0, 0, 0]), np.array([0.5, -0.5, 0.5, 0.5]))
        odom_SE3 = [CAM2ENU * SE3(pose) * ENU2CAM for pose in odom_data[:, 1:8]]

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
