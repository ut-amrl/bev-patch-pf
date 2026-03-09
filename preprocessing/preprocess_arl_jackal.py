"""Preprocess the data from the ARL Jackal robot"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import copy_rect_image_depth, sync_wheel_odom, sync_pose  # noqa E402

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SCENES = [
    # "ahg_courtyard_2025_07_03-18_09_49",
    # "ahg_courtyard_2025_07_03-18_17_34",
    # "eastwood_2025_08_13-16_53_17",
    # "eastwood_2025_08_13-17_09_38",
    # "eastwood_2025_08_13-17_15_58",
    # "run_05_onioncreek_2025-11-30-18-28-52"
    # "pickle_north_2025-12-18-11-44-00",
    # "pickle_south_2025-12-18-10-50-42",
    # "run_00_pickle_2026-01-12-15-33-59",
    # "run_01_pickle_2026-01-12-16-05-32",
    # "run_02_pickle_2026-01-12-16-49-31",
    # "run_03_pickle_2026-01-12-17-03-52",
    # "run_04_pickle_2026-01-12-17-28-39",
    "run_00_onion-creek_2026-01-29-16-38-31",
]

IMAGE_SIZE = (720, 1280)  # H, W
K = np.array(
    [
        [640.51513671875, 0.0, 647.9937744140625],
        [0.0, 639.5934448242188, 359.2181091308594],
        [0.0, 0.0, 1.0],
    ]
)
D = np.array(
    [
        -0.054610688239336014,
        0.06513150036334991,
        -0.0003195908502675593,
        0.0008660743478685617,
        -0.0206373929977417,
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
        pose_df = pd.read_csv(utm_pose_path)
        synced_pose_df = sync_pose(pose_df, synced_df["ts"].values)

        before = len(synced_df)
        synced_df = synced_df.merge(synced_pose_df, on="ts", how="inner")
        synced_df.dropna(inplace=True)
        after = len(synced_df)
        logger.info(f"Merged pose data: {before} -> {after} frames after pose sync.")

        # save the synced pose data
        synced_pose = synced_df[["ts", "x", "y", "angle"]]
        synced_pose.to_csv(out_scene_dir / "utm_pose.csv", index=False, float_format="%.6f")
    else:
        logger.warning(f"Pose file {utm_pose_path} does not exist. Skipping pose synchronization.")

    np.savetxt(out_scene_dir / "timestamps.txt", synced_df["ts"].values, fmt="%.6f")

    # 2. load odometry data and synchronize with image timestamps
    for odom_type in ["odom_local", "odom_platform"]:
        if not (scene_dir / f"{odom_type}.csv").exists():
            logger.warning(f"Odometry file {odom_type}.csv does not exist. Skipping.")
            continue

        odom_df = pd.read_csv(scene_dir / f"{odom_type}.csv")
        synced_odom_df = sync_wheel_odom(odom_df, synced_df["ts"].values)
        synced_odom_df.to_csv(out_scene_dir / f"{odom_type}.csv", index=False, float_format="%.6f")

    # 3. copy synced rectified image and depth data
    copy_rect_image_depth(synced_df, K, D, IMAGE_SIZE, out_scene_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/robodata/ARL_SARA/ARL_Jackal/extracted")
    parser.add_argument("--outdir", type=str, default="/scratch/dlee/Datasets/ARL_Jackal")
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
