"""Visualize the TartanDrive dataset train/val/test split on a map."""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AuxTransformBox, HPacker, TextArea
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredOffsetbox

from geotiff.handler import GeoTiffHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ORDERED_LABELS = [
    ("train", "Train"),
    ("val", "Validation"),
    ("test_intra", "Test (Seen Route)"),
    ("test_cross", "Test (Unseen Route)"),
]
LINE_PROPS = {
    "train": {"color": "#0072B0", "alpha": 0.7, "linewidth": 4, "offset": (-2, -2)},
    "val": {"color": "#009E70", "alpha": 0.7, "linewidth": 4, "offset": (2, 2)},
    "test_intra": {"color": "#E69F00", "alpha": 0.7, "linewidth": 4, "offset": (0, 0)},
    "test_cross": {"color": "#D55E00", "alpha": 0.7, "linewidth": 4, "offset": (0, 0)},
}


def viz_traj_tartandrive(args):
    geo_handler = GeoTiffHandler(args.tiff)

    # Load dataset config
    with open(args.config) as f:
        dataset_cfg = yaml.safe_load(f)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(geo_handler.image[:9300, :7100], alpha=0.5)

    # Define line properties for each split

    for split_option, label in ORDERED_LABELS:
        scene_names = dataset_cfg.get(f"{split_option}_scenes", [])
        for scene_name in scene_names:
            scene_dir = Path(args.root) / scene_name
            if not scene_dir.exists():
                logger.warning(f"Scene directory {scene_dir} does not exist")
                continue

            # odometry data
            odom_file = scene_dir / "odom.csv"
            if not odom_file.exists():
                logger.warning(f"No odometry data found for scene {scene_dir.name}")
                continue

            odom_data = np.genfromtxt(odom_file, delimiter=",", skip_header=1)
            odom_pixels = geo_handler.coords_to_pixel(odom_data[:, 1:3])

            # Apply slight offset for better visibility
            offset_x, offset_y = LINE_PROPS[split_option]["offset"]
            odom_pixels[:, 0] += offset_x
            odom_pixels[:, 1] += offset_y

            ax.plot(
                odom_pixels[:, 0],
                odom_pixels[:, 1],
                c=LINE_PROPS[split_option]["color"],
                linewidth=LINE_PROPS[split_option]["linewidth"],
                alpha=LINE_PROPS[split_option]["alpha"],
            )

    # Manually create legend handles
    handles = [
        Line2D([0], [0], color=LINE_PROPS[scene_type]["color"], linewidth=3, label=label)
        for scene_type, label in ORDERED_LABELS
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=25,
        markerscale=1.5,
        borderpad=0.8,
    )

    resolution = np.mean(geo_handler.resolution)
    draw_scale_bar(ax, resolution, 500, pad=3.0, font_size=20)

    ax.axis("off")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fig.savefig(args.out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_scale_bar(ax, m_per_pixel, length_m=100, color="black", pad=3.0, font_size=20):
    """Draw a horizontal metric scale bar in the lower-right corner of *ax*."""
    bar_px = length_m / m_per_pixel  # bar length in data pixels

    bar_box = AuxTransformBox(ax.transData)
    bar_box.add_artist(Line2D([0, bar_px], [0, 0], color=color, linewidth=5))
    bar_box.add_artist(Line2D([0, 0], [0, -50], color=color, linewidth=5))
    bar_box.add_artist(Line2D([bar_px, bar_px], [0, -50], color=color, linewidth=5))

    # text (still uses display coords internally, so no transform needed)
    label = f"{int(length_m)} m" if length_m < 1000 else f"{int(length_m / 1000)} km"
    txt = TextArea(label, textprops=dict(color=color, size=font_size, va="center", ha="right"))
    packed = HPacker(children=[txt, bar_box], align="center", pad=0, sep=10)

    anchored = AnchoredOffsetbox(
        loc="lower right", child=packed, pad=pad, borderpad=0.5, frameon=False
    )
    ax.add_artist(anchored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/TartanDrive2")
    parser.add_argument(
        "--tiff", type=str, default="data/TartanDrive2/gascola_EPSG32617_scale03_082725.tiff"
    )
    parser.add_argument("--config", type=str, default="config/dataset/tartandrive.yaml")
    parser.add_argument(
        "--out_path", type=str, default="paper_scripts/figure/tartandrive_split.png"
    )
    args = parser.parse_args()

    viz_traj_tartandrive(args)
