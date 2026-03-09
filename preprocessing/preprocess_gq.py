"""Preprocess the data from the ARL Jackal robot"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from manifpy import SE2
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SCENES = [
    "2024-08-13-11-28-57",
    "2024-08-13-16-48-47",
    "2024-08-15-13-23-03",
    "2024-08-13-16-56-52",
    "2024-08-15-12-40-46",
    "2024-08-15-13-08-57",
]


def preprocess_scene(root: Path, out_scene_dir: Path, sync_tol: float = 0.1):
    logger.info(f"Preprocessing {out_scene_dir.name}")

    timestamp = np.loadtxt(out_scene_dir / "timestamps.txt")

    # load pose data (UTM coordinates) and synchronize with image timestamps
    pose_df = pd.read_csv(root / "odom" / f"{out_scene_dir.name}.csv")
    synced_pose_df = sync_odom(pose_df, timestamp)

    before = len(synced_pose_df)
    synced_pose_df.dropna(inplace=True)
    after = len(synced_pose_df)
    logger.info(f"Merged pose data: {before} -> {after} frames after pose sync.")

    # save the synchronized pose data
    synced_pose_df.to_csv(out_scene_dir / "wheel_odom.csv", index=False)


def sync_odom(odom_df: pd.DataFrame, sync_ts: np.ndarray) -> pd.DataFrame:
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
            interp_odom.append(
                np.array([ts, odom_SE2[0].x(), odom_SE2[0].y(), odom_SE2[0].angle()])
            )
            continue
        if idx == len(odom_ts):
            interp_odom.append(
                np.array([ts, odom_SE2[-1].x(), odom_SE2[-1].y(), odom_SE2[-1].angle()])
            )
            continue

        t0, t1 = odom_ts[idx - 1], odom_ts[idx]
        p0, p1 = odom_SE2[idx - 1], odom_SE2[idx]
        alpha = (ts - t0) / (t1 - t0)
        p_alpha = p0 + alpha * (p1 - p0)
        interp_odom.append(np.array([ts, p_alpha.x(), p_alpha.y(), p_alpha.angle()]))

    interp_odom = np.array(interp_odom)
    synced_odom_df = pd.DataFrame(interp_odom, columns=["ts", "x", "y", "angle"])
    return synced_odom_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/robodata/dlee/Datasets/GQ-dataset")
    parser.add_argument("--outdir", type=str, default="/scratch/dlee/Datasets/GQ")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)

    for scene in SCENES:
        out_scene_dir = outdir / scene
        out_scene_dir.mkdir(parents=True, exist_ok=True)

        preprocess_scene(root, out_scene_dir)
