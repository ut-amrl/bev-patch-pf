# BEV-Patch-PF

Project Website: https://bev-patch-pf.github.io/

## Installation
```bash
conda create -y -n bev-patch-pf python=3.12
conda activate bev-patch-pf
conda install -y -c conda-forge manifpy
pip install -e .
```

## Training BEV-Patch-PF model
Train with single GPU
```bash
python src/train.py
```

Train with multi-GPU
```bash
accelerate launch --multi_gpu --num_processes=<num_of_GPUs> --mixed_precision=fp16 src/train_ddp.py
```

## Run Particle Filter
```bash
python src/run_pf.py sequence=<dataset> ckpt_path=<path/to/ckpt>
```


## Real-time experiment
1. Export ONNX model
```bash
python scripts/export_to_onnx.py --ckpt_path=<ckpt_path> --out_dir=<outdir>
```
2. Build TensorRT model and run ROS2 node
- https://github.com/ut-amrl/bev-patch-pf_ROS2


## Generate Trainig Dataset
### A) Generate Dataset from bagfiles
#### 1. Extract sensor data from bagfiles (RGB, depth, IMU, etc.)
- Use rosbagkit: https://github.com/ut-amrl/rosbagkit

#### 2. Generate a trajectory
- Run a SLAM/odometry pipeline to produce a pose trajectory as CSV.
- Example: FAST-LIO (ROS 2): https://github.com/ut-amrl/FAST_LIO_ROS2

#### 3. Export a GeoTIFF map (QGIS)
-  **Important**: ensure the GeoTIFF is in a correct UTM zone so distances are in meters.

#### 4. Align the SLAM trajectory into the GeoTIFF/map coordinate frame:
```bash
python preprocessing/align_trajectory_geotiff.py \
  --geotiff=<path/to/geotiff.tiff> \
  --traj=<path/to/trajectory.csv>
```

#### 5. Perform dataset-specific preprocessing (e.g., time synchronization, rectification, filtering).
- Example: `python preprocessing/preprocess_arl_jackal.py`

### B) Generate Dataset from TartanDrive2.0 dataset
1. Download bagfiles
  - https://github.com/ut-amrl/tartan_drive_2.0
2. Extract images and GT odom
  - https://github.com/ut-amrl/rosbagkit
3. Preprocess the extracted data
  - `python preprocessing/preprocess_tartandrive.py`