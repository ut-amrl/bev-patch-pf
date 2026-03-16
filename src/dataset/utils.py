from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from manifpy import SE2, SE3
from scipy.spatial.transform import Rotation
from torch.utils.data import Subset
from torch.utils.data._utils.collate import default_collate

###
# dataset utilities
###


class SubsetWithAttributes(Subset):
    def __getattr__(self, name):
        return getattr(self.dataset, name)


def safe_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of dictionaries, skipping fields that cannot be collated"""
    output = {}
    keys = batch[0].keys()
    for key in keys:
        values = [b[key] for b in batch]
        try:
            output[key] = default_collate(values)
        except Exception:
            output[key] = values
    return output


###
# functions for loading and processing pose data
###


def load_csv_columns(path: Path, required_columns: tuple[str, ...]) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")

    missing_columns = [column for column in required_columns if column not in header]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    column_indices = [header.index(column) for column in required_columns]
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    data = np.atleast_2d(data).astype(np.float64, copy=False)

    if data.size == 0:
        raise ValueError(f"{path} is empty")

    return data[:, column_indices]


def load_timestamps(path: Path) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path, dtype=np.float64))


def se3_poses_from_rows(
    pose_rows: np.ndarray,
    *,
    camera_frame: bool = False,
    cam2enu: torch.Tensor | np.ndarray | None = None,
) -> list[SE3]:
    pose_rows = np.atleast_2d(np.asarray(pose_rows, dtype=np.float64))
    if pose_rows.shape[1] != 7:
        raise ValueError(f"Expected pose rows with 7 columns, got shape {pose_rows.shape}")

    poses = [SE3(row) for row in pose_rows]
    if not camera_frame:
        return poses

    cam2enu_matrix = _as_rigid_transform(cam2enu)
    enu2cam_matrix = np.linalg.inv(cam2enu_matrix)
    cam2enu_se3 = _matrix_to_se3(cam2enu_matrix)
    enu2cam_se3 = _matrix_to_se3(enu2cam_matrix)
    return [cam2enu_se3 * pose * enu2cam_se3 for pose in poses]


def interpolate_se3_poses(
    pose_timestamps: np.ndarray, poses: Sequence[SE3], target_timestamps: np.ndarray
) -> list[SE3]:
    pose_timestamps = np.atleast_1d(np.asarray(pose_timestamps, dtype=np.float64))
    target_timestamps = np.atleast_1d(np.asarray(target_timestamps, dtype=np.float64))
    poses = list(poses)

    if len(pose_timestamps) != len(poses):
        raise ValueError(f"{len(pose_timestamps)} timestamps do not match {len(poses)} poses")
    if not poses:
        raise ValueError("Cannot interpolate an empty pose sequence")

    interpolated_poses = []
    for timestamp in target_timestamps:
        idx = np.searchsorted(pose_timestamps, timestamp, side="left")
        if idx == 0:
            interpolated_poses.append(poses[0])
            continue
        if idx == len(pose_timestamps):
            interpolated_poses.append(poses[-1])
            continue

        t0, t1 = pose_timestamps[idx - 1], pose_timestamps[idx]
        if t1 <= t0:
            interpolated_poses.append(poses[idx])
            continue

        p0, p1 = poses[idx - 1], poses[idx]
        alpha = (timestamp - t0) / (t1 - t0)
        interpolated_poses.append(p0 + alpha * (p1 - p0))

    return interpolated_poses


def compute_se3_actions(poses: Sequence[SE3]) -> np.ndarray:
    poses = list(poses)
    actions = np.zeros((len(poses), 3), dtype=np.float64)
    for i in range(1, len(poses)):
        action_se3 = poses[i - 1].between(poses[i])
        dx, dy, _ = action_se3.translation()
        qx, qy, qz, qw = action_se3.quat()
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), -1.0 + 2.0 * (qw * qw + qx * qx))
        actions[i] = np.array([dx, dy, yaw], dtype=np.float64)
    return actions


def se2_poses_from_rows(pose_rows: np.ndarray) -> list[SE2]:
    pose_rows = np.atleast_2d(np.asarray(pose_rows, dtype=np.float64))
    if pose_rows.shape[1] != 3:
        raise ValueError(f"Expected pose rows with 3 columns, got shape {pose_rows.shape}")
    return [SE2(*row) for row in pose_rows]


def compute_se2_actions(poses: Sequence[SE2]) -> np.ndarray:
    poses = list(poses)
    actions = np.zeros((len(poses), 3), dtype=np.float64)
    for i in range(1, len(poses)):
        action = poses[i - 1].between(poses[i])
        dx, dy = action.translation()
        actions[i] = np.array([dx, dy, action.angle()], dtype=np.float64)
    return actions


def _as_rigid_transform(transform: torch.Tensor | np.ndarray | None) -> np.ndarray:
    if transform is None:
        raise ValueError("cam2enu is required when camera_frame=True")

    if isinstance(transform, torch.Tensor):
        transform = transform.detach().cpu().numpy()

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got shape {transform.shape}")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError("cam2enu must be a homogeneous 4x4 transform")

    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("cam2enu rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("cam2enu rotation must have determinant 1")

    return transform


def _matrix_to_se3(transform: np.ndarray) -> SE3:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    quaternion = Rotation.from_matrix(rotation).as_quat()
    return SE3(translation, quaternion)
