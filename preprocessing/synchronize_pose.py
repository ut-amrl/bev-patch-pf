"""Synchronize aligned pose CSVs to image timestamps for multi-scene datasets."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import redistribute_timestamps, sync_pose  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_POSE_COLUMNS = ("timestamp", "x", "y", "angle")
DEFAULT_TIMESTAMPS_PATTERN = "{scene}/2d_rect/timestamps.txt"
DEFAULT_POSE_PATTERN = "{scene}/fast_lio_aligned.csv"
DEFAULT_OUTPUT_NAME = "utm_pose.csv"


def _split_scene_pattern(pattern: str) -> tuple[str, str]:
    """Split a relative path template around the required {scene} placeholder."""
    if "{scene}" not in pattern:
        raise ValueError(f"Pattern must include '{{scene}}': {pattern}")
    return pattern.split("{scene}", maxsplit=1)


def resolve_scene_path(root: Path, pattern: str, scene: str) -> Path:
    return root / pattern.format(scene=scene)


def discover_scene_paths(root: Path, pattern: str) -> dict[str, Path]:
    prefix, suffix = _split_scene_pattern(pattern)
    scene_paths = {}

    # The substring that replaces {scene} becomes the scene key used to pair
    # timestamp files with pose files across the two patterns.
    for path in sorted(root.glob(pattern.replace("{scene}", "*"))):
        rel_path = path.relative_to(root).as_posix()
        if not rel_path.startswith(prefix):
            continue
        if suffix and not rel_path.endswith(suffix):
            continue

        stop = len(rel_path) - len(suffix) if suffix else len(rel_path)
        scene = rel_path[len(prefix) : stop]
        if not scene:
            continue
        scene_paths[scene] = path

    return scene_paths


def load_image_timestamps(ts_path: Path) -> np.ndarray:
    timestamps = np.atleast_1d(np.loadtxt(ts_path, dtype=np.float64))
    if timestamps.ndim != 1:
        raise ValueError(f"Expected a 1D timestamps file at {ts_path}")
    return timestamps


def load_aligned_pose(pose_path: Path) -> pd.DataFrame:
    pose_df = pd.read_csv(pose_path)
    missing_columns = [col for col in REQUIRED_POSE_COLUMNS if col not in pose_df.columns]
    if missing_columns:
        raise ValueError(f"Pose file {pose_path} is missing required columns: {missing_columns}")

    pose_df = pose_df.loc[:, list(REQUIRED_POSE_COLUMNS)].copy()
    pose_df.sort_values("timestamp", inplace=True)
    pose_df.reset_index(drop=True, inplace=True)
    if pose_df.empty:
        raise ValueError(f"Pose file {pose_path} is empty")

    duplicate_count = int(pose_df["timestamp"].duplicated().sum())
    if duplicate_count > 0:
        logger.warning(f"Normalizing {duplicate_count} duplicate pose timestamps in {pose_path}")
        redistribute_timestamps(pose_df, ts_col="timestamp")

    if not pose_df["timestamp"].is_monotonic_increasing:
        pose_df.sort_values("timestamp", inplace=True)
        pose_df.reset_index(drop=True, inplace=True)

    return pose_df


def pad_pose_boundaries_for_sync(pose_df: pd.DataFrame) -> pd.DataFrame:
    """Pad exact boundary timestamps so shared sync_pose keeps them."""
    first_row = pose_df.iloc[[0]].copy()
    first_row["timestamp"] = np.nextafter(first_row["timestamp"].iloc[0], -np.inf)

    last_row = pose_df.iloc[[-1]].copy()
    last_row["timestamp"] = np.nextafter(last_row["timestamp"].iloc[0], np.inf)

    padded_df = pd.concat([first_row, pose_df, last_row], ignore_index=True)
    padded_df.sort_values("timestamp", inplace=True)
    padded_df.reset_index(drop=True, inplace=True)
    return padded_df


###
# Main synchronization logic
###


def synchronize_scene(scene: str, ts_path: Path, pose_path: Path, output_name: str) -> None:
    logger.info(f"Synchronizing pose for {scene}")

    image_ts = load_image_timestamps(ts_path)
    pose_df = load_aligned_pose(pose_path)
    synced_pose_df = sync_pose(pad_pose_boundaries_for_sync(pose_df), image_ts)
    synced_pose_df.rename(columns={"ts": "timestamp"}, inplace=True)

    first_pose = pose_df.iloc[0]
    last_pose = pose_df.iloc[-1]
    before_mask = synced_pose_df["timestamp"] < first_pose["timestamp"]
    after_mask = synced_pose_df["timestamp"] > last_pose["timestamp"]

    synced_pose_df.loc[before_mask, ["x", "y", "angle"]] = first_pose[["x", "y", "angle"]].to_numpy()
    synced_pose_df.loc[after_mask, ["x", "y", "angle"]] = last_pose[["x", "y", "angle"]].to_numpy()

    remaining_nan = synced_pose_df[["x", "y", "angle"]].isna().any(axis=1)
    if remaining_nan.any():
        logger.warning(f"{scene}: dropping {int(remaining_nan.sum())} frames with unexpected invalid pose values")
        synced_pose_df = synced_pose_df.loc[~remaining_nan].copy()
    kept_frames = len(synced_pose_df)

    output_path = pose_path.parent / output_name
    synced_pose_df.to_csv(output_path, index=False, float_format="%.6f")

    logger.info(
        f"{scene}: image_ts={len(image_ts)} pose_rows={len(pose_df)} "
        f"clamped_before={int(before_mask.sum())} clamped_after={int(after_mask.sum())} "
        f"kept={kept_frames} output={output_path}"
    )


def main(root: Path, timestamps_pattern: str, pose_pattern: str, output_name: str) -> None:
    timestamp_scenes = discover_scene_paths(root, timestamps_pattern)
    aligned_pose_scenes = discover_scene_paths(root, pose_pattern)

    missing_pose_scenes = sorted(set(timestamp_scenes) - set(aligned_pose_scenes))
    for scene in missing_pose_scenes:
        logger.warning(f"Skipping {scene}: missing aligned pose {resolve_scene_path(root, pose_pattern, scene)}")

    missing_timestamp_scenes = sorted(set(aligned_pose_scenes) - set(timestamp_scenes))
    for scene in missing_timestamp_scenes:
        logger.warning(
            f"Skipping {scene}: missing timestamps under {resolve_scene_path(root, timestamps_pattern, scene)}"
        )

    scene_names = sorted(set(timestamp_scenes) & set(aligned_pose_scenes))
    if not scene_names:
        logger.warning(
            f"No scenes found with both timestamps and aligned pose files under {root} "
            f"(timestamps_pattern={timestamps_pattern}, pose_pattern={pose_pattern})"
        )
        return

    logger.info(f"Found {len(scene_names)} scenes with timestamps and aligned pose files")
    for scene in scene_names:
        try:
            synchronize_scene(scene, timestamp_scenes[scene], aligned_pose_scenes[scene], output_name)
        except Exception:
            logger.exception(f"Failed to synchronize pose for {scene}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize aligned pose CSVs to image timestamps across multiple scenes.\n\n"
            "A pattern is a relative path template under --root. It must include "
            "{scene}, which is replaced by each scene folder name."
        ),
        epilog=(
            "Example:\n"
            "  --root data/UT-SARA-GQ/processed\n"
            "  --timestamps-pattern '{scene}/2d_rect/timestamps.txt'\n"
            "  --pose-pattern '{scene}/fast_lio_aligned.csv'\n"
            "  scene=2024-08-13-11-28-57 resolves to:\n"
            "    data/UT-SARA-GQ/processed/2024-08-13-11-28-57/2d_rect/timestamps.txt\n"
            "    data/UT-SARA-GQ/processed/2024-08-13-11-28-57/fast_lio_aligned.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--root", type=str, default="data/UT-SARA-GQ/processed")
    parser.add_argument(
        "--timestamps-pattern",
        type=str,
        default=DEFAULT_TIMESTAMPS_PATTERN,
        help="Relative path template under --root for image timestamps. Must include {scene}.",
    )
    parser.add_argument(
        "--pose-pattern",
        type=str,
        default=DEFAULT_POSE_PATTERN,
        help="Relative path template under --root for aligned pose CSVs. Must include {scene}.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Filename to write next to each resolved pose CSV.",
    )
    args = parser.parse_args()

    main(Path(args.root), args.timestamps_pattern, args.pose_pattern, args.output_name)
