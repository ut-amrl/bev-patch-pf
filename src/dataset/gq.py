from pathlib import Path

import cv2
import numpy as np
import torch
from manifpy import SE2, SE3  # noqa F401

from dataset.common import GeoLocDataset


class GQDataset(GeoLocDataset):
    INTRINSIC = torch.tensor(
        [
            [577.4217795, 0.0, 379.497589],
            [0.0, 577.4217795, 295.068590],
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
    IMAGE_SIZE = (540, 720)

    def __init__(self, root: str, scene: str, **kwargs):
        super().__init__(root, scene, **kwargs)

    def _load_data(self, root: Path):
        """Load GQ specific data paths."""
        image_dir = root / self.scene / "cam_left_half"
        self.image_paths = sorted(image_dir.glob("*.png"))

        depth_dir = root / self.scene / "depth"
        self.depth_paths = sorted(depth_dir.glob("*.png"))

        pose_file = root / self.scene / "gt_odom.csv"
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


class GQSequence(GQDataset):
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

        # ## motion updates from Wheel Odometry
        # odom_path = Path(root) / scene / "wheel_odom.csv"
        # odom_data = np.genfromtxt(odom_path, delimiter=",", skip_header=1)
        # odom_SE2 = [SE2(*pose) for pose in odom_data[:, 1:]]

        # actions = [np.zeros(3)]
        # for i in range(1, len(odom_SE2)):
        #     # compute motion update
        #     action_SE2 = odom_SE2[i - 1].between(odom_SE2[i])  # SE2
        #     actions.append(np.array([*action_SE2.translation(), action_SE2.angle()]))
        # self.actions = np.array(actions)

    @staticmethod
    def quat_to_yaw(quat: np.ndarray) -> float:
        qx, qy, qz, qw = quat
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), -1.0 + 2.0 * (qw * qw + qx * qx))
        return yaw

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.update({"timestamp": self.poses[idx, 0], "action": self.actions[idx]})
        return data
