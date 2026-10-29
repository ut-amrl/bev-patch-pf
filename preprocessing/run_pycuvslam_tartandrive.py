# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

# Adapted from: https://github.com/nvidia-isaac/cuVSLAM/blob/main/examples/realsense/run_stereo.py
# Modified for synchronized TartanDrive2 stereo sequences.

import argparse
import logging
from pathlib import Path

import cuvslam
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# fmt: off
K0 = np.array(
    [
        477.6049499511719, 0.0, 499.5, 0.0,
        0.0, 477.6049499511719, 252.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    ]
).reshape(3, 4)
K1 = np.array(
    [
        477.6049499511719, 0.0, 499.5, -100.14994812011719,
        0.0, 477.6049499511719, 252.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    ]
).reshape(3, 4)
# fmt: on
Ks = np.array([K0, K1])  # (2, 3, 4)


def run(scene_dir: str):
    scene_dir = Path(scene_dir)
    timestamps = np.loadtxt(str(scene_dir / "timestamps.txt")).reshape(-1)
    left_images = sorted((scene_dir / "left").glob("*.png"))
    right_images = sorted((scene_dir / "right").glob("*.png"))
    if not len(timestamps) == len(left_images) == len(right_images):
        raise ValueError(f"Timestamp and stereo image counts do not match in {scene_dir}")
    if not len(timestamps):
        return

    # Initialize the cuvslam tracker
    size = Image.open(left_images[0]).size
    cameras = [cuvslam.Camera(), cuvslam.Camera()]
    for i in [0, 1]:
        cameras[i].size = size
        cameras[i].principal = [Ks[i][0][2], Ks[i][1][2]]
        cameras[i].focal = [Ks[i].diagonal()[0], Ks[i].diagonal()[1]]

    # stereo-camera baseline
    cameras[1].rig_from_camera.translation[0] = -Ks[1][0][3] / Ks[1][0][0]

    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_observations_export=True,
        enable_final_landmarks_export=False,
        horizontal_stereo_camera=False,
    )
    tracker = cuvslam.Tracker(cuvslam.Rig(cameras), cfg)

    # Track each frames in the dataset sequence
    poses = []  # (x, y, z, qx, qy, qz, qw)
    for idx, timestamp in enumerate(tqdm(timestamps)):
        # Load grayscale pixels as array for left and right absolute image paths
        images = [
            np.asarray(Image.open(left_images[idx]).convert("L")),
            np.asarray(Image.open(right_images[idx]).convert("L")),
        ]

        # Do tracking
        odom_pose_estimate, _ = tracker.track(int(timestamp * 1e9), images)
        odom_pose = odom_pose_estimate.world_from_rig.pose

        # Prepare visualization data
        poses.append(np.array([timestamp, *odom_pose.translation, *odom_pose.rotation]))

    header = "timestamp,x,y,z,qx,qy,qz,qw"
    output_file = scene_dir / "pycuvslam.csv"
    np.savetxt(
        output_file,
        np.array(poses),
        fmt=["%.6f"] + ["%.8f"] * 7,
        delimiter=",",
        header=header,
        comments="",
    )
    logger.info(f"Saved {len(poses)} poses to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track TartanDrive2 dataset")
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    for scene_dir in sorted(root.iterdir()):
        if scene_dir.is_dir():
            logger.info(f"Processing scene directory: {scene_dir}")
            run(scene_dir)
