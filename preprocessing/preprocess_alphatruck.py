"""Preprocess the data from the ARL Jackal robot"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import copy_rect_image_depth, sync_lidar_odom, sync_pose  # noqa E402

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SCENES = [
    # "AHG_courtyard_slow",
    # "AHG_courtyard_fast",
    "AHG_Speedway",
    "Speedway_AHG",
]


IMAGE_SIZE = (720, 1280)  # H, W
K = np.array(
    [
        [646.88623046875, 0.0, 634.9051513671875],
        [0.0, 646.12158203125, 364.38299560546875],
        [0.0, 0.0, 1.0],
    ]
)
D = np.array(
    [
        -0.05455869436264038,
        0.06151237711310387,
        0.000513636798132211,
        0.00010752628440968692,
        -0.020269419997930527,
    ]
)


def preprocess_scene(scene_dir: Path, out_scene_dir: Path, sync_tol: float = 0.1):
    logger.info(f"Preprocessing {scene_dir}...")

    # load image and depth data
    image_paths = sorted((scene_dir / "image_raw").glob("*.png"))
    image_ts = np.loadtxt(scene_dir / "timestamp_image_raw.txt")
    assert len(image_paths) == len(image_ts), f"{len(image_paths)} != {len(image_ts)}"
    image_df = pd.DataFrame({"ts": image_ts, "image_path": image_paths})

    depth_paths = sorted((scene_dir / "depth").glob("*.png"))
    depth_ts = np.loadtxt(scene_dir / "timestamp_depth.txt")
    assert len(depth_paths) == len(depth_ts), f"{len(depth_paths)} != {len(depth_ts)}"
    depth_df = pd.DataFrame({"ts": depth_ts, "depth_path": depth_paths})

    # synchronize image and depth data
    synced_df = pd.merge_asof(image_df, depth_df, on="ts", direction="nearest", tolerance=sync_tol)
    synced_df.dropna(inplace=True)
    logger.info(f"Synced (image: {len(image_df)}, depth: {len(depth_df)}) to {len(synced_df)} entries")

    # 1. load pose data (UTM coordinates) and synchronize with image timestamps
    utm_pose_path = scene_dir / "fast_lio_aligned.csv"
    if utm_pose_path.is_file():
        pose_df = pd.read_csv(str(utm_pose_path))
        synced_pose_df = sync_pose(pose_df, synced_df["ts"].values)

        before = len(synced_df)
        synced_df = synced_df.merge(synced_pose_df, on="ts", how="inner")
        synced_df.dropna(inplace=True)
        after = len(synced_df)
        logger.info(f"Merged pose data: {before} -> {after} frames after pose sync.")

        # save the synced data
        timestamps = synced_df["ts"].values
        np.savetxt(out_scene_dir / "timestamps.txt", timestamps, fmt="%.6f")
        synced_pose = synced_df[["ts", "x", "y", "angle"]]
        synced_pose.to_csv(out_scene_dir / "utm_pose.csv", index=False, float_format="%.6f")
    else:
        logger.warning(f"Pose file {utm_pose_path} not found. Skipping pose synchronization.")

    # 2. load odometry data and synchronize with image timestamps
    lidar_odom_path = scene_dir / "fast_lio.csv"
    if lidar_odom_path.is_file():
        odom_df = pd.read_csv(str(lidar_odom_path))
        synced_odom_df = sync_lidar_odom(odom_df, synced_df["ts"].values)
        synced_odom_df.to_csv(out_scene_dir / "fast_lio.csv", index=False, float_format="%.6f")
    else:
        logger.warning(f"Lidar odom file {lidar_odom_path} not found. Skipping odom synchronization.")

    # 3. copy synced rectified image and depth data
    copy_rect_image_depth(synced_df, K, D, IMAGE_SIZE, out_scene_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/robodata/dlee/Datasets/AlphaTruck")
    parser.add_argument("--outdir", type=str, default="/scratch/dlee/Datasets/AlphaTruck")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)

    for scene in SCENES:
        scene_dir = root / scene
        if not scene_dir.is_dir():
            logger.warning(f"Scene directory {scene_dir} does not exist.")
            continue

        out_scene_dir = outdir / scene
        out_scene_dir.mkdir(parents=True, exist_ok=True)

        preprocess_scene(scene_dir, out_scene_dir)
