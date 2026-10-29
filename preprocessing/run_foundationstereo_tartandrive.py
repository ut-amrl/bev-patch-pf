# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Adapted from: https://github.com/NVlabs/FoundationStereo/blob/master/scripts/run_demo.py
# Modified for synchronized TartanDrive2 stereo sequences.

import argparse
import logging
import os
import sys

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")
from pathlib import Path

import cv2
import numpy as np
from core.foundation_stereo import *
from core.utils.utils import InputPadder
from omegaconf import OmegaConf
from tqdm import tqdm
from Utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

K = np.array(
    [
        [477.60495, 0.0, 499.5],
        [0.0, 477.60495, 252.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
BASELINE = 0.2096920229


def main(args):
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    ckpt_dir = args.ckpt_dir
    logging.info(ckpt_dir)
    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt["model"])

    model.cuda()
    model.eval()

    scene_dir_list = sorted(Path(args.root).iterdir())
    for idx, scene_dir in enumerate(scene_dir_list):
        if not scene_dir.is_dir():
            continue

        logger.info(f"[{idx}/{len(scene_dir_list)}] Processing scene: {scene_dir.name}")
        left_image_paths = sorted((scene_dir / "left").glob("*.png"))
        right_image_paths = sorted((scene_dir / "right").glob("*.png"))
        if len(left_image_paths) != len(right_image_paths):
            logging.warning(f"Left and right image counts do not match in {scene_dir.name}")
            continue

        output_dir = scene_dir / "depth"
        output_dir.mkdir(parents=True, exist_ok=True)

        new_filenames = {f"{idx:06d}.png" for idx in range(len(left_image_paths))}
        existing_files = {f.name for f in output_dir.glob("*.png")}
        if new_filenames == existing_files:
            logging.info(f"All files are up to date in {output_dir}")
            continue

        for idx in tqdm(range(len(left_image_paths)), desc=f"{scene_dir.name}"):
            output_file = output_dir / f"{idx:06d}.png"
            if output_file.exists():
                continue

            img0 = imageio.imread(left_image_paths[idx])
            img1 = imageio.imread(right_image_paths[idx])

            scale = args.scale
            assert scale <= 1, "scale must be <=1"
            img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
            img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)

            # convert gray to rgb
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

            H, W = img0.shape[:2]

            img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            img0, img1 = padder.pad(img0, img1)

            with torch.amp.autocast("cuda", enabled=True):
                if not args.hiera:
                    disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
                else:
                    disp = model.run_hierachical(
                        img0,
                        img1,
                        iters=args.valid_iters,
                        test_mode=True,
                        small_ratio=0.5,
                    )
            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H, W)

            if args.remove_invisible:
                yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij")
                us_right = xx - disp
                invalid = us_right < 0
                disp[invalid] = np.inf

            depth = (K[0, 0] * scale) * BASELINE / disp

            # --- save depth image (.png, mm)
            depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(np.round(depth_mm * 1000), 0, 65535).astype(np.uint16)
            imageio.imwrite(str(output_dir / f"{idx:06d}.png"), depth_mm)


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root of synchronized TartanDrive2 scenes")
    parser.add_argument("--ckpt_dir", default="pretrained_models/23-51-11/model_best_bp2.pth")
    parser.add_argument(
        "--scale",
        default=1.0,
        type=float,
        help="downsize the image by scale, must be <=1",
    )
    parser.add_argument(
        "--hiera",
        default=1,
        type=int,
        help="hierarchical inference (only needed for high-resolution images (>1K))",
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=16,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument(
        "--remove_invisible",
        default=1,
        type=int,
        help="remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable",
    )
    args = parser.parse_args()

    main(args)
