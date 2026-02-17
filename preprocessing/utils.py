import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from manifpy import SE2, SE3
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

logger = logging.getLogger(__name__)


def redistribute_timestamps(df: pd.DataFrame, ts_col: str = "timestamp"):
    """Redistribute duplicate timestamps by spreading them evenly between neighbors."""
    original_ts = df[ts_col].to_numpy()

    ts_counts = pd.Series(original_ts).value_counts()
    duplicate_ts = ts_counts[ts_counts > 1]
    if duplicate_ts.empty:
        return

    unique_ts = np.sort(np.unique(original_ts))
    new_ts = original_ts.copy()
    rows_to_drop = []

    for t0, count in duplicate_ts.items():
        idxs = np.where(original_ts == t0)[0]
        pos = np.where(unique_ts == t0)[0][0]

        if pos == len(unique_ts) - 1:
            # For edge timestamps, mark duplicates for removal (keep first)
            logger.warning(f"Removing {count - 1} duplicate timestamps at end position {t0:.6f}")
            rows_to_drop.extend(idxs[1:].tolist())
        else:
            # For middle timestamps, redistribute between current and next timestamp
            lower = unique_ts[pos]
            upper = unique_ts[pos + 1]
            spaced = np.linspace(lower, upper, count + 1)[:-1]
            new_ts[idxs] = spaced

    # Drop rows with edge duplicates and update timestamps
    if rows_to_drop:
        df.drop(index=rows_to_drop, inplace=True)
        remaining_mask = np.ones(len(original_ts), dtype=bool)
        remaining_mask[rows_to_drop] = False
        df[ts_col] = new_ts[remaining_mask]
    else:
        df[ts_col] = new_ts

    # Sort by timestamp and reset index
    df.sort_values(ts_col, inplace=True)
    df.reset_index(drop=True, inplace=True)


def sync_pose(pose_df: pd.DataFrame, sync_ts: np.ndarray) -> pd.DataFrame:
    pose_ts = pose_df["timestamp"].values
    pose_SE2 = [SE2(*xyr) for xyr in pose_df[["x", "y", "angle"]].values]

    # interpolate odometry data to match image timestamps
    # NOTE: set nan values for timestamps that are outside the pose data range
    interp_pose = []
    for ts in sync_ts:
        idx = np.searchsorted(pose_ts, ts, side="left")
        if idx == 0 or idx == len(pose_ts):
            interp_pose.append(np.array([ts, np.nan, np.nan, np.nan]))
            continue

        t0, t1 = pose_ts[idx - 1], pose_ts[idx]
        p0, p1 = pose_SE2[idx - 1], pose_SE2[idx]
        alpha = (ts - t0) / (t1 - t0)
        p_alpha = p0 + alpha * (p1 - p0)
        interp_pose.append(np.array([ts, p_alpha.x(), p_alpha.y(), p_alpha.angle()]))

    interp_pose = np.array(interp_pose)
    synced_pose_df = pd.DataFrame(interp_pose, columns=["ts", "x", "y", "angle"])
    return synced_pose_df


def sync_lidar_odom(lidar_odom_df: pd.DataFrame, synced_ts: np.ndarray):
    redistribute_timestamps(lidar_odom_df, ts_col="timestamp")

    lidar_odom_ts = lidar_odom_df["timestamp"].values
    lidar_odom_SE3 = [
        SE3(*xyz_qxyzw) for xyz_qxyzw in zip(lidar_odom_df[["x", "y", "z", "qx", "qy", "qz", "qw"]].values)
    ]

    # interpolate odometry data to match image timestamps
    interp_ts = []
    interp_lidar_odom = []
    for ts in synced_ts:
        idx = np.searchsorted(lidar_odom_ts, ts)

        # edge case
        if idx == 0 or idx == len(lidar_odom_ts):
            continue

        t0, t1 = lidar_odom_ts[idx - 1], lidar_odom_ts[idx]

        p0, p1 = lidar_odom_SE3[idx - 1], lidar_odom_SE3[idx]
        alpha = (ts - t0) / (t1 - t0)
        p_alpha = p0 + alpha * (p1 - p0)

        interp_ts.append(ts)
        interp_lidar_odom.append(np.array(p_alpha.coeffs()))
    interp_ts = np.array(interp_ts)
    interp_lidar_odom = np.array(interp_lidar_odom)

    # concat tstamps and odometry data
    interp_lidar_odom = np.concatenate((interp_ts[:, None], interp_lidar_odom), axis=1)
    synced_lidar_odom = pd.DataFrame(interp_lidar_odom, columns=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
    return synced_lidar_odom


def sync_wheel_odom(odom_df: pd.DataFrame, sync_ts: np.ndarray) -> pd.DataFrame:
    odom_ts = odom_df["timestamp"].values
    odom_xy = odom_df[["x", "y"]].values
    odom_quat = odom_df[["qx", "qy", "qz", "qw"]].values
    odom_rot = R.from_quat(odom_quat).as_matrix()
    odom_yaw = np.arctan2(odom_rot[:, 1, 0], odom_rot[:, 0, 0])
    odom_SE2 = [SE2(*xy, yaw) for xy, yaw in zip(odom_xy, odom_yaw)]

    # interpolate odometry data to match image timestamps
    interp_odom = []
    for ts in sync_ts:
        idx = np.searchsorted(odom_ts, ts, side="left")
        if idx == 0:
            interp_odom.append(np.array([ts, odom_SE2[0].x(), odom_SE2[0].y(), odom_SE2[0].angle()]))
            continue
        if idx == len(odom_ts):
            interp_odom.append(np.array([ts, odom_SE2[-1].x(), odom_SE2[-1].y(), odom_SE2[-1].angle()]))
            continue

        t0, t1 = odom_ts[idx - 1], odom_ts[idx]
        p0, p1 = odom_SE2[idx - 1], odom_SE2[idx]
        alpha = (ts - t0) / (t1 - t0)
        p_alpha = p0 + alpha * (p1 - p0)
        interp_odom.append(np.array([ts, p_alpha.x(), p_alpha.y(), p_alpha.angle()]))

    interp_odom = np.array(interp_odom)
    synced_odom_df = pd.DataFrame(interp_odom, columns=["ts", "x", "y", "angle"])
    return synced_odom_df


def copy_rect_image_depth(synced_df: pd.DataFrame, K: np.ndarray, D: np.ndarray, IMAGE_SIZE: tuple, outdir: Path):
    image_outdir = outdir / "image"
    depth_outdir = outdir / "depth"
    image_outdir.mkdir(parents=True, exist_ok=True)
    depth_outdir.mkdir(parents=True, exist_ok=True)

    # Rectification Map
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, IMAGE_SIZE[::-1], cv2.CV_16SC2)

    # remove existing files in the output directories
    for path in image_outdir.glob("*.png"):
        path.unlink()
    for path in depth_outdir.glob("*.png"):
        path.unlink()

    # copy and rectify images and depth files
    synced_df = synced_df.reset_index(drop=True)
    for idx, row in tqdm(synced_df.iterrows(), desc="Copying and rectifying images"):
        image_path = Path(row["image_path"])
        depth_path = Path(row["depth_path"])
        image_dest = image_outdir / f"image_{idx:06d}.png"
        depth_dest = depth_outdir / f"depth_{idx:06d}.png"
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image.shape[:2] != IMAGE_SIZE:
                raise ValueError(f"Image {image_path} has unexpected shape {image.shape}")
            undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(str(image_dest), undistorted)
            depth_dest.write_bytes(depth_path.read_bytes())
        except Exception as e:
            logger.error(f"Failed to copy files for index {idx}: {e}")
            continue
