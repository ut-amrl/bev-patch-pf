# Custom Dataset Integration

This directory contains dataset adapters for training and offline particle filtering.

To add a new dataset, usually create:

1. `src/dataset/custom.py` with `CustomDataset` and, for PF, `CustomSequence`
2. `config/dataset/custom.yaml` for [`src/train.py`](../train.py)
3. `config/sequence/custom.yaml` for [`src/run_pf.py`](../run_pf.py)
4. preprocessing that writes the files your loader expects


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
* `utm_pose.csv` has one row per frame with `(x, y, angle)`
* `x, y` use the same projected frame as the GeoTIFF
* `angle` is in radians
* depth is in meters

For PF, also provide:

* timestamps
* odometry or VO that can be converted into per-frame relative actions `[dx, dy, dtheta]`


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
