# Preprocessing

This folder converts extracted sensor logs into the dataset layout used by BEV-Patch-PF. If you are new to this pipeline, start with the UT-SARA-GQ flow below.

## Overview

Preprocessing has two steps:

1. Align odometry to a GeoTIFF map.
2. Synchronize the aligned pose to image timestamps.

The result is a per-frame pose CSV used for training and evaluation.

## Required inputs

Before running these scripts, you should already have:

- timestamped RGB-D images (e.g., [`rosbagkit`](https://github.com/ut-amrl/rosbagkit))
- timestamped odometry (e.g., [`FAST-LIO`](https://github.com/ut-amrl/FAST_LIO_ROS2) or [`Adaptive-LIO`](https://github.com/ut-amrl/Adaptive-LIO))
- a GeoTIFF in the same global frame as the odometry (e.g., [`QGIS`](https://qgis.org/))

Before alignment, the trajectory CSV should contain: `timestamp`, `x`, `y`, `qx`, `qy`, `qz`, and `qw`

## UT-SARA-GQ quickstart

**1. Align the trajectory to the GeoTIFF:**

```bash
python preprocessing/align_trajectory_geotiff.py \
    --geotiff /path/to/map.tif \
    --traj data/UT-SARA-GQ/processed/<scene>/fast_lio.csv
```

Optional flags:
- `--flat-mode off|on|auto`
- `--frame-transform /path/to/transform.yaml`

When the alignment looks correct, press `P` to save `fast_lio_aligned.csv` next to the original trajectory.

**2. Synchronize pose to image timestamps:**

```bash
python preprocessing/synchronize_pose.py
```

By default, this scans `data/UT-SARA-GQ/processed`, matches:

- `{scene}/2d_rect/timestamps.txt`
- `{scene}/fast_lio_aligned.csv`

and writes:

- `{scene}/utm_pose.csv`

If your layout is different, set `--root`, `--timestamps-pattern`, `--pose-pattern`, or `--output-name`.
The path patterns are relative to `--root` and must include `{scene}`.

**3. Verify the result.**

Each valid scene should now contain `utm_pose.csv` with: `timestamp`, `x`, `y`, and `angle`

## Notes

- The GeoTIFF and trajectory must already use the same global frame.
- `synchronize_pose.py` only processes scenes that have both a timestamps file and an aligned pose CSV.
- The pose CSV passed to `synchronize_pose.py` must contain `timestamp`, `x`, `y`, and `angle`.
- If a scene is skipped, check the expected paths first.
- If alignment looks wrong, inspect the original trajectory before re-running alignment.

## Other preprocessing scripts

```bash
python preprocessing/preprocess_tartandrive.py \
    --src /path/to/raw_dataset \
    --dst /path/to/output_dataset
```

For TartanDrive, this script synchronizes odometry to image timestamps and keeps the longest valid trajectory segment.