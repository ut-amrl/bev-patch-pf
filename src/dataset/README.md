# Custom Dataset Integration

This directory contains dataset adapters for training and offline particle filtering.

To add a new dataset, usually create:

1. `src/dataset/custom.py` with `CustomDataset` and, for PF, `CustomSequence`
2. `config/dataset/custom.yaml`
3. `config/sequence/custom.yaml` for [`src/run_pf.py`](../run_pf.py)
4. preprocessing that writes the files your loader expects

No registry update is needed. Hydra only needs to import `_target_: dataset.custom.CustomDataset`.

## What the repo expects

### Training

[`src/train.py`](../train.py) and [`src/train_ddp.py`](../train_ddp.py) instantiate a config from [`config/dataset`](../../config/dataset) and call `GeoLocDataset.__getitem__()`.

The base class builds the full sample for you as long as your subclass provides:

* sorted image paths
* sorted depth paths
* `self.poses` with `self.poses[:, 1:4] == (x, y, angle)`
* `_load_depth()` returning `float32` depth in meters
* a valid `geo_tiff_path` in config

Training samples include:

* `ground_image`
* `ground_depth`
* `aerial_image`
* `pose_uvr`
* `gt_xyr`
* `info["K"]`
* `info["cam2enu"]`
* `info["resolution"]`

### Offline particle filtering

[`src/run_pf.py`](../run_pf.py) instantiates configs from [`config/sequence`](../../config/sequence).

A sequence class must add:

* `timestamp`
* `action`


## Reference patterns

Use the closest existing loader as a template:

* [`arl_jackal.py`](arl_jackal.py): simplest case, aligned SE2 poses and odometry
* [`gq.py`](gq.py): nested layout, separate timestamps, interpolated SE3 odometry
* [`tartandrive.py`](tartandrive.py): camera-frame VO converted with `CAM2ENU`

## Minimal data contract

There is no fixed folder layout. Your loader and config just need to agree.

A minimal training scene might look like:

```text
data/CustomDataset/<scene>/
├── image/
├── depth/
└── utm_pose.csv
```

Requirements:

* images and depth have the same frame count
* filenames sort in true temporal order
* `utm_pose.csv` has one row per frame
* `self.poses[:, 1:4]` is always `(x, y, angle)`
* `x, y` use the same projected frame as the GeoTIFF
* `angle` is in radians
* depth is in meters

For PF, also provide:

* timestamps
* odometry or VO that can be converted into per-frame relative actions `[dx, dy, dtheta]`

## `GeoLocDataset` contract

[`GeoLocDataset`](common.py) already handles:

* image loading and normalization
* depth resizing
* intrinsic rescaling
* GeoTIFF crop extraction
* sampled pose generation
* common sample packaging

Your subclass defines:

* `INTRINSIC`: `3x3` intrinsics at native image resolution
* `CAM2ENU`: `4x4` transform used by the model and odometry conversion
* `IMAGE_SIZE`: native `(H, W)`
* `_load_data(root: Path)`: fills `self.image_paths`, `self.depth_paths`, `self.poses`
* `_load_depth(depth_path: Path) -> np.ndarray`: returns depth in meters


## Minimal training skeleton

```python
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset.common import GeoLocDataset
from dataset.utils import load_csv_columns

GT_POSE_COLUMNS = ("timestamp", "x", "y", "angle")


class CustomDataset(GeoLocDataset):
    INTRINSIC = torch.tensor([
        [640.0, 0.0, 640.0],
        [0.0, 640.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    CAM2ENU = torch.tensor([
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    IMAGE_SIZE = (720, 1280)

    def _load_data(self, root: Path):
        scene_dir = root / self.scene
        self.image_paths = sorted((scene_dir / "image").glob("*.png"))
        self.depth_paths = sorted((scene_dir / "depth").glob("*.png"))
        self.poses = load_csv_columns(scene_dir / "utm_pose.csv", GT_POSE_COLUMNS)
        assert len(self.image_paths) == len(self.depth_paths) == len(self.poses)

    def _load_depth(self, depth_path: Path) -> np.ndarray:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth /= 1000.0  # remove this if already stored in meters
        depth[(depth > 65.5) | (depth < 0.1) | np.isnan(depth)] = 0
        return depth
```

## Minimal sequence skeleton

```python
from pathlib import Path

from dataset.utils import compute_se2_actions, load_csv_columns, load_timestamps, se2_poses_from_rows

ODOM_COLUMNS = ("x", "y", "angle")


class CustomSequence(CustomDataset):
    def __init__(self, root: str, scene: str, **kwargs):
        super().__init__(root, scene, **kwargs)

        scene_dir = Path(root) / scene
        self.timestamps = load_timestamps(scene_dir / "timestamps.txt")
        odom = load_csv_columns(scene_dir / "odom.csv", ODOM_COLUMNS)
        self.actions = compute_se2_actions(se2_poses_from_rows(odom))

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.update({"timestamp": self.timestamps[idx], "action": self.actions[idx]})
        return data
```

Use the helpers in [`utils.py`](utils.py) for actions:

* synchronized SE2 odometry: `compute_se2_actions(...)`
* SE3 odometry with different timestamps: interpolate first, then `compute_se3_actions(...)`
* camera-frame VO: use `camera_frame=True` and `cam2enu=self.CAM2ENU`

## Config templates

`config/dataset/custom.yaml`

```yaml
_target_: dataset.custom.CustomDataset
name: "CustomDataset"

root: data/CustomDataset
geo_tiff_path: ${.root}/maps/site_EPSG32614.tiff

ground_image_resize: [512, 512]
aerial_image_resize: [768, 768]

train_scenes: [scene_a]
val_scenes: [scene_b]
```

`config/sequence/custom.yaml`

```yaml
_target_: dataset.custom.CustomSequence
name: "CustomDataset"

root: data/CustomDataset
geo_tiff_path: ${.root}/maps/site_EPSG32614.tiff

ground_image_resize: [512, 512]
aerial_image_resize: [768, 768]

scenes: [scene_eval]
```

## Using the configs

`config/dataset/custom.yaml` is a dataset config group, so training still goes through a top-level train config.
To train on the custom dataset, create `config/train_custom.yaml` from [`config/train.yaml`](../../config/train.yaml), replace the dataset defaults with `dataset@datasets.0: custom`, set `dataset_name`, and run:

```bash
python src/train.py --config-name train_custom
```

`config/sequence/custom.yaml` is used by [`config/run_pf.yaml`](../../config/run_pf.yaml).
Run particle filtering with:

```bash
python src/run_pf.py sequence=custom ckpt_path=/path/to/model.pth
```
