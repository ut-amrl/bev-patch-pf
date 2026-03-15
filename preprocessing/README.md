# Preprocessing

This folder contains the scripts that turn extracted sensor logs into the dataset layout used by BEV-Patch-PF. If you are new here, start with the UT-SARA-GQ flow below.

## What this folder does

The preprocessing contract is:

1. Start with timestamped images and timestamped odometry.
2. Align the odometry to a GeoTIFF map.
3. Interpolate the aligned pose onto the image timestamps.
4. Save synchronized per-frame pose files for downstream training and evaluation.

In short, this folder handles alignment plus synchronization.

## Before you start

You should already have timestamped images, timestamped odometry, and a GeoTIFF in the same global frame.
The images usually come from [`rosbagkit`](https://github.com/ut-amrl/rosbagkit),
and the odometry usually comes from something like ([`FAST-LIO`](https://github.com/ut-amrl/FAST_LIO_ROS2) or [`Adaptive-LIO`](https://github.com/ut-amrl/Adaptive-LIO)).

Before alignment, the trajectory CSV should contain `timestamp`, `x`, `y`, `qx`, `qy`, `qz`, and `qw`.

## UT-SARA-GQ Quickstart

1. Check that one scene has the files you need:

```text
data/UT-SARA-GQ/processed/<scene>/
├── 2d_rect/timestamps.txt
└── fast_lio.csv
```

2. Align that trajectory to the GeoTIFF:

```bash
python preprocessing/align_trajectory_geotiff.py \
    --geotiff /path/to/map.tif \
    --traj data/UT-SARA-GQ/processed/<scene>/fast_lio.csv
```

Optional flags: `--flat-mode off|on|auto` and `--frame-transform /path/to/transform.yaml`.

This step is a spatial alignment of the odometry track to the map. The dataset-specific preprocessing scripts below handle the temporal synchronization.

The viewer is interactive. When the alignment looks right, press `P` to save. That writes `data/UT-SARA-GQ/processed/<scene>/fast_lio_aligned.csv` next to the original file, with columns `timestamp`, `x`, `y`, and `angle`.

3. Run pose synchronization with the repo defaults:

```bash
python preprocessing/synchronize_pose.py
```

This reads `fast_lio_aligned.csv`, interpolates the aligned pose onto `2d_rect/timestamps.txt`, and writes `utm_pose.csv`. The same script can be reused for future custom datasets by changing the input path patterns and, if needed, `--output-name`.

That command uses:

- `--root data/UT-SARA-GQ`
- `--outdir data/UT-SARA-GQ/processed`
- `--timestamps-pattern processed/{scene}/2d_rect/timestamps.txt`
- `--pose-pattern processed/{scene}/fast_lio_aligned.csv`

If your layout differs, run:

```bash
python preprocessing/synchronize_pose.py \
    --root data/UT-SARA-GQ \
    --outdir data/UT-SARA-GQ/processed \
    --timestamps-pattern 'processed/{scene}/2d_rect/timestamps.txt' \
    --pose-pattern 'processed/{scene}/fast_lio_aligned.csv' \
    --output-name 'utm_pose.csv'
```

4. Verify success. Each valid scene should now contain `data/UT-SARA-GQ/processed/<scene>/utm_pose.csv`, and that file should contain `timestamp`, `x`, `y`, and `angle`.

## Sanity checks

- The GeoTIFF and trajectory must already use the same global coordinate frame.
- Synchronization depends on valid timestamps on both sides. Missing or irregular timestamps can cause dropped frames or skipped segments.
- `synchronize_pose.py` only processes scenes that have both `2d_rect/timestamps.txt` and `fast_lio_aligned.csv`.
- If alignment looks wrong, inspect `fast_lio.csv` before trying again.
- If preprocessing skips a scene, check the scene path and required filenames first.

## Other preprocessing scripts

```bash
python preprocessing/preprocess_tartandrive.py --src /path/to/raw_dataset --dst /path/to/output_dataset
```

For TartanDrive specifically, `preprocess_tartandrive.py` first synchronizes odometry to image timestamps, then keeps the longest valid trajectory segment. This helps when a run contains timestamp jumps, dropped-frame gaps, or other bad regions that should not be kept in the final sequence.
