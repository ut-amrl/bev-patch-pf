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

## TartanDrive2

Extract the original **bagfiles** (not the KITTI release) using
[rosbagkit](https://github.com/ut-amrl/rosbagkit) with `timestamp_source: bag`.
Then synchronize the extracted stereo images and GT odometry:

```bash
python preprocessing/preprocess_tartandrive.py \
    --src /path/to/extracted_dataset --dst /path/to/processed_dataset
```

GT output uses x=east, y=north, and heading in radians counterclockwise from east.

**pycuVSLAM:** Install [PyCuVSLAM](https://github.com/nvidia-isaac/cuVSLAM)
and run in its environment:

```bash
python preprocessing/run_pycuvslam_tartandrive.py --root /path/to/processed_dataset
```

This writes `<scene>/pycuvslam.csv`. Published trajectories, matching timestamps,
calibration, and the GeoTIFF are available in the
[supplementary directory](https://web.corral.tacc.utexas.edu/texasrobotics/web_UT-SARA-GQ/supplementary/TartanDrive2/).

**FoundationStereo:** Set up [FoundationStereo](https://github.com/NVlabs/FoundationStereo)
and its `23-51-11` checkpoint. Copy `run_foundationstereo_tartandrive.py`
from this folder into FoundationStereo's `scripts/`, then run from that repository:

```bash
python scripts/run_foundationstereo_tartandrive.py \
    --root /path/to/processed_dataset \
    --ckpt_dir pretrained_models/23-51-11/model_best_bp2.pth \
    --scale 1.0 --hiera 1 --valid_iters 16 --remove_invisible 1
```

This writes `<scene>/depth/*.png` as uint16 depth in millimeters.