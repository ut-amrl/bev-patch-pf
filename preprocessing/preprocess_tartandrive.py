"""Preprocess the TartanDrive2.0 dataset by syncing the odometry and image data."""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from manifpy import SE2, SE3
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_scene(src_scene_dir: Path, dst_scene_dir: Path):
    # ----- load image data and create DataFrames -----
    color_ts = np.loadtxt(src_scene_dir / "timestamp_multisense_color.txt").reshape(-1)
    color_paths = sorted((src_scene_dir / "multisense" / "color").glob("*.png"))
    assert len(color_paths) == len(color_ts)
    left_ts = np.loadtxt(src_scene_dir / "timestamp_multisense_left.txt").reshape(-1)
    left_paths = sorted((src_scene_dir / "multisense" / "left").glob("*.png"))
    assert len(left_paths) == len(left_ts)
    right_ts = np.loadtxt(src_scene_dir / "timestamp_multisense_right.txt").reshape(-1)
    right_paths = sorted((src_scene_dir / "multisense" / "right").glob("*.png"))
    assert len(right_paths) == len(right_ts)

    df_color = pd.DataFrame({"ts_c": color_ts, "path_c": color_paths})
    df_left = pd.DataFrame({"ts_l": left_ts, "path_l": left_paths})
    df_right = pd.DataFrame({"ts_r": right_ts, "path_r": right_paths})
    redistribute_timestamps(df_color, ts_col="ts_c")
    redistribute_timestamps(df_left, ts_col="ts_l")
    redistribute_timestamps(df_right, ts_col="ts_r")

    # Sync cameras: merge with 0.2s tolerance
    image_data = pd.merge_asof(df_color, df_left, left_on="ts_c", right_on="ts_l", direction="nearest", tolerance=0.2)
    image_data = pd.merge_asof(
        image_data, df_right, left_on="ts_c", right_on="ts_r", direction="nearest", tolerance=0.2
    )
    image_data = image_data.dropna(subset=["ts_l", "ts_r"]).reset_index(drop=True)
    logger.info(f"{len(image_data)} synced image frames")

    # ----- load GT odometry data -----
    odom_file = src_scene_dir / "odom.csv"
    if not odom_file.exists():
        logger.error(f"No odometry data found for scene {src_scene_dir.name}")
        return

    # sync odometry data with image timestamps
    odom_data = pd.read_csv(odom_file)
    redistribute_timestamps(odom_data, ts_col="timestamp")

    synced_odom = sync_odom(odom_data, image_data["ts_c"].values)

    # convert to UTM coordinates (x: east, y: north, angle: counter-clockwise from east)
    synced_odom[["x", "y"]] = synced_odom[["y", "x"]].values
    synced_odom["x"] *= -1
    synced_odom["angle"] += np.pi / 2
    synced_odom["angle"] = np.arctan2(np.sin(synced_odom["angle"]), np.cos(synced_odom["angle"]))

    synced_df = pd.merge(image_data, synced_odom, left_on="ts_c", right_on="timestamp", how="inner")
    synced_df = synced_df.dropna(subset=["x", "y"]).reset_index(drop=True)
    logger.info(f"{len(synced_df)} synced image-odometry frames")

    # ---- Filter frames ----
    synced_good_df = get_good_segment(synced_df)

    # skip the scene if the trajectory is too short
    diff = synced_good_df[["x", "y"]].diff()
    dist = np.linalg.norm(diff.values, axis=1)
    traj_length = np.sum(dist[1:])
    if traj_length < 10 or len(synced_good_df) < 300:
        logger.warning(f"Skip {src_scene_dir.name} ({len(synced_good_df)} images, {traj_length:.2f} m)")
        if dst_scene_dir.exists():
            shutil.rmtree(dst_scene_dir)
        return

    # ----- Save the synced data -----
    dst_scene_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {len(synced_good_df)} frames")

    # 1. timestamps
    timestamps = synced_good_df["ts_c"].values
    np.savetxt(dst_scene_dir / "timestamps.txt", timestamps, fmt="%.6f")

    # 2. GT pose
    odom_out = synced_good_df[["timestamp", "x", "y", "angle"]]
    odom_out.to_csv(dst_scene_dir / "odom.csv", index=False, float_format="%.6f")

    # 3. images
    subdirs = {
        "path_c": dst_scene_dir / "color",
        "path_l": dst_scene_dir / "left",
        "path_r": dst_scene_dir / "right",
    }

    for path_key, image_out_dir in subdirs.items():
        image_out_dir.mkdir(parents=True, exist_ok=True)

        new_filenames = {f"{idx:06d}.png" for idx in range(len(synced_good_df))}
        existing_files = {f.name for f in image_out_dir.glob("*.png")}

        # delete stale files
        stale_files = existing_files - new_filenames
        for stale in stale_files:
            stale_path = image_out_dir / stale
            stale_path.unlink()

        # copy new files
        for idx, row in tqdm(synced_good_df.iterrows(), desc=f"Copying {image_out_dir.name}"):
            src = row[path_key]
            tgt = image_out_dir / f"{idx:06d}.png"
            shutil.copy2(src, tgt)

    # ----- (optional) sync and save LiDAR odometry -----
    lidar_odom_file = src_scene_dir / "lidar_odom.csv"
    if lidar_odom_file.exists():
        synced_lidar_odom = sync_lidar_odom(lidar_odom_file, timestamps)
        synced_lidar_odom.to_csv(dst_scene_dir / "lidar_odom.csv", index=False, float_format="%.6f")


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


def sync_odom(odom_data: pd.DataFrame, synced_ts: np.ndarray, ts_threshold: float = 0.8):
    """Sync odometry data with image timestamps (SE3 -> SE2)"""

    def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), -1.0 + 2.0 * (qw * qw + qx * qx))
        return yaw

    odom_ts = odom_data["timestamp"].values
    odom_SE2 = [
        SE2(*xy, quat_to_yaw(*q))
        for xy, q in zip(odom_data[["x", "y"]].values, odom_data[["qx", "qy", "qz", "qw"]].values)
    ]

    # interpolate odometry data to match image timestamps
    interp_odom = []
    for ts in synced_ts:
        idx = np.searchsorted(odom_ts, ts)

        # edge case (no bounds)
        if idx == 0 or idx == len(odom_ts):
            interp_odom.append(np.full(3, np.nan))
            continue

        t0, t1 = odom_ts[idx - 1], odom_ts[idx]

        # edge case (timestamp too far away for interpolation)
        if t1 - t0 > ts_threshold:
            interp_odom.append(np.full(3, np.nan))
            continue

        p0, p1 = odom_SE2[idx - 1], odom_SE2[idx]
        alpha = (ts - t0) / (t1 - t0)
        p_alpha = p0 + alpha * (p1 - p0)
        interp_odom.append(np.array([p_alpha.x(), p_alpha.y(), p_alpha.angle()]))
    interp_odom = np.array(interp_odom)

    # concat tstamps and odometry data
    interp_odom = np.concatenate((synced_ts[:, None], interp_odom), axis=1)
    synced_odom = pd.DataFrame(interp_odom, columns=["timestamp", "x", "y", "angle"])
    return synced_odom


def get_good_segment(
    df: pd.DataFrame,
    time_threshold: float = 0.8,
    velocity_threshold: float = 50.0,  # m/s
    stationary_threshold: float = 0.5,  # meter
):
    """Drop all rows with bad timestamps and odometry data."""
    time_diffs = df["ts_c"].diff()
    time_bad = time_diffs.gt(time_threshold)
    distances = np.hypot(df["x"].diff(), df["y"].diff())
    velocities = distances / time_diffs.replace(0, np.nan)  # Avoid division by zero
    velocity_bad = velocities.gt(velocity_threshold)

    bad = time_bad | velocity_bad
    seg_id = bad.cumsum()

    df = df[~bad].assign(seg_id=seg_id[~bad])
    seg_counts = df["seg_id"].value_counts()
    best_seg = seg_counts.idxmax()
    result = df[df["seg_id"] == best_seg].drop(columns=["seg_id"])

    if bad.sum() > 0:
        logger.warning(f"Dropping {time_bad.sum()} time-gap frames (>{time_threshold}s)")
        logger.warning(f"Dropping {velocity_bad.sum()} high-vel frames (>{velocity_threshold}m/s)")
        logger.warning(f"Dropping {bad.sum()} total bad frames")
        logger.info(f"Found {len(seg_counts)} segments: {seg_counts.to_dict()}. Keep {len(result)} frames")

    # remove stationary segments
    dists = np.hypot(result["x"].diff().fillna(0), result["y"].diff().fillna(0))
    cumdist = dists.cumsum()
    stationary = (cumdist.iloc[-1] - cumdist) < stationary_threshold
    if stationary.sum() > 100:
        result = result[~stationary]
        logger.warning(f"{stationary.sum()} stationary frames are removed")

    return result.reset_index(drop=True)


def sync_lidar_odom(lidar_odom_file: Path, synced_ts: np.ndarray):
    lidar_odom_data = pd.read_csv(lidar_odom_file)
    redistribute_timestamps(lidar_odom_data, ts_col="timestamp")

    lidar_odom_ts = lidar_odom_data["timestamp"].values
    lidar_odom_SE3 = [
        SE3(*xyz_qxyzw)
        for xyz_qxyzw in zip(
            lidar_odom_data[["x", "y", "z", "qx", "qy", "qz", "qw"]].values,
        )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="/robodata/dlee/Datasets/TartanDrive2")
    parser.add_argument("--dst", type=str, default="/scratch/dlee/Datasets/TartanDrive2")
    args = parser.parse_args()

    scene_dir_list = sorted(Path(args.src).iterdir())
    for i, src_scene_dir in enumerate(scene_dir_list):
        if not src_scene_dir.is_dir():
            continue

        print(f"\n[{i}/{len(scene_dir_list)}] Preprocessing scene {src_scene_dir.name}...")
        out_scene_dir = Path(args.dst) / src_scene_dir.name
        preprocess_scene(src_scene_dir, out_scene_dir)
