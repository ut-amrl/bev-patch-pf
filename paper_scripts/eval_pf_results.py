#!/usr/bin/env python3
"""
Simple ATE evaluation script for particle filter results
"""

import argparse
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from manifpy import SE2, SE3
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AuxTransformBox, HPacker, TextArea
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredOffsetbox

from geotiff.handler import GeoTiffHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trajectory(method_cfg: dict, scene: str, geo_handler: GeoTiffHandler):
    """Load trajectory from file with various formats

    Returns:
        timestamps: np.ndarray (N,)
        poses: list[SE2]
    """
    traj_file = method_cfg["traj_file"].replace("{scene}", scene)
    traj_path = Path(method_cfg["base_path"]) / traj_file
    traj_fmt = method_cfg["format"]
    traj_frame = method_cfg.get("frame", "enu")

    if not traj_path.exists():
        raise FileNotFoundError(f"File not found: {traj_path}")

    if traj_path.suffix == ".csv":
        data = np.genfromtxt(traj_path, delimiter=",", skip_header=1)
    elif traj_path.suffix == ".txt":
        data = np.loadtxt(traj_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {traj_path.suffix}")

    timestamps, poses = None, None
    if traj_fmt == "t_xyr":
        timestamps = data[:, 0]
        if traj_frame == "enu":
            poses = [SE2(x, y, theta) for x, y, theta in data[:, 1:4]]
        elif traj_frame == "ned":
            poses = [SE2(x, y, np.pi / 2 - theta) for x, y, theta in data[:, 1:4]]
        else:
            raise NotImplementedError(f"Unsupported coordinate frame: {traj_frame}")
        return timestamps, poses

    elif traj_fmt == "t_xyz_qxyzw":
        timestamps = data[:, 0]
        if traj_frame == "enu":
            poses_SE3 = [SE3(xyz_qxyzw) for xyz_qxyzw in data[:, 1:8]]
        elif traj_frame == "cam":
            CAM2ENU = SE3(np.array([0, 0, 0]), np.array([0.5, -0.5, 0.5, -0.5]))
            ENU2CAM = SE3(np.array([0, 0, 0]), np.array([0.5, -0.5, 0.5, 0.5]))
            poses_SE3 = [CAM2ENU * SE3(p) * ENU2CAM for p in data[:, 1:8]]
        elif traj_frame == "ned":
            NED2ENU = SE3(np.array([0, 0, 0]), np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]))
            ENU2NED = SE3(np.array([0, 0, 0]), np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]))
            poses_SE3 = [NED2ENU * SE3(p) * ENU2NED for p in data[:, 1:8]]
        else:
            raise NotImplementedError(f"Unsupported coordinate frame: {traj_frame}")

        # Convert SE3 -> SE2
        poses = []
        for p in poses_SE3:
            x, y, _ = p.translation()
            rot_mats = p.rotation()
            yaw = np.arctan2(rot_mats[1, 0], rot_mats[0, 0])
            poses.append(SE2(x, y, yaw))
        return timestamps, poses

    elif traj_fmt == "t_gps":
        timestamps = data[:, 0]
        latlons = data[:, [3, 4]]  # lat, lon
        utm_coords = geo_handler.latlon_to_coords(latlons)
        poses = [SE2(*xy, 0.0) for xy in utm_coords]
        return timestamps, poses

    else:
        raise NotImplementedError(f"Unsupported format: {traj_fmt}")


def compute_dead_reckoning(est_odom: list[SE2], ref_traj: list[SE2], est_t: np.ndarray, ref_t: np.ndarray):
    """Compute dead-reckoning trajectory by anchoring odometry to reference trajectory"""
    # get anchor pose from reference trajectory
    anchor_pose, anchor_idx = None, None
    for i in range(len(est_odom)):
        idx = np.searchsorted(ref_t, est_t[i])
        if idx == 0 or idx == len(ref_t):
            continue

        t0, t1 = ref_t[idx - 1], ref_t[idx]
        p0, p1 = ref_traj[idx - 1], ref_traj[idx]
        alpha = (est_t[i] - t0) / (t1 - t0)
        anchor_pose = p0 + alpha * (p1 - p0)
        anchor_idx = i
        break
    logger.info(f"  Found anchor pose at index {anchor_idx}")

    if anchor_pose is None or anchor_idx is None:
        raise ValueError("Could not find anchor pose")

    # Initialize trajectory list with proper size
    est_traj = [None] * len(est_odom)
    est_traj[anchor_idx] = anchor_pose

    # compute dead-reckoning backwards (anchor_idx to 0)
    current_pose = anchor_pose
    for i in range(anchor_idx - 1, -1, -1):
        p0, p1 = est_odom[i], est_odom[i + 1]  # i: t-1
        rel_motion = p1.between(p0)  # P^{t}_{t-1}
        current_pose = current_pose * rel_motion  # P^w_{t-1} = P^w_{t} * P^{t}_{t-1}
        est_traj[i] = current_pose

    # compute dead-reckoning forward (anchor_idx to the end)
    current_pose = anchor_pose
    for i in range(anchor_idx + 1, len(est_odom)):
        p0, p1 = est_odom[i - 1], est_odom[i]  # i: t+1
        rel_motion = p0.between(p1)  # P^{t}_{t+1}
        current_pose = current_pose * rel_motion  # P^w_{t+1} = P^w_{t} * P^{t}_{t+1}
        est_traj[i] = current_pose

    return est_t, est_traj


def compute_ate(ref_traj: list[SE2], est_traj: list[SE2], ref_t: np.ndarray, est_t: np.ndarray):
    """Compute ATE and orientation errors between reference and estimated trajectories"""

    # Align trajectories with interpolation
    aligned_ref, aligned_est = [], []
    for t, est_pos in zip(est_t, est_traj):
        idx = np.searchsorted(ref_t, t)
        if idx == 0 or idx == len(ref_t):
            continue

        t0, t1 = ref_t[idx - 1], ref_t[idx]
        p0, p1 = ref_traj[idx - 1], ref_traj[idx]
        alpha = (t - t0) / (t1 - t0)
        interp_ref_pos = p0 + alpha * (p1 - p0)
        aligned_ref.append([interp_ref_pos.x(), interp_ref_pos.y(), interp_ref_pos.angle()])
        aligned_est.append([est_pos.x(), est_pos.y(), est_pos.angle()])
    aligned_ref = np.array(aligned_ref)
    aligned_est = np.array(aligned_est)

    # Compute position errors
    pos_errors = np.linalg.norm(aligned_est[:, :2] - aligned_ref[:, :2], axis=1)

    # Compute orientation errors (in degrees for better interpretation)
    angle_diff = aligned_est[:, 2] - aligned_ref[:, 2]
    ori_errors = np.abs(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))
    ori_errors_deg = np.degrees(ori_errors)

    return pos_errors, ori_errors_deg


def evaluate_scene(scene: str, methods: list[dict], geo_handler: GeoTiffHandler):
    """Evaluate all methods for a scene (if reference available) and load trajectories"""
    results = {}
    trajectories = {}
    errors = {}

    # Load reference trajectory from methods with reference: true
    ref_method_cfg = None
    for method_cfg in methods:
        if method_cfg.get("reference", False):
            ref_method_cfg = method_cfg
            break

    ref_t, ref_traj = None, None
    if ref_method_cfg:
        try:
            ref_t, ref_traj = load_trajectory(ref_method_cfg, scene, geo_handler)
            trajectories[ref_method_cfg["name"]] = (ref_t, ref_traj)
            logger.info(f"  Loaded reference ({ref_method_cfg['name']}) with {len(ref_traj)} poses")
        except Exception as e:
            logger.warning(f"  Failed to load reference ({ref_method_cfg['name']}): {e}")
            ref_t, ref_traj = None, None
    else:
        logger.warning(f"  No reference method found for scene {scene}")

    # Load and evaluate other trajectories
    for method_cfg in methods:
        if method_cfg.get("reference", False):
            continue

        method_name = method_cfg["name"]

        try:
            est_t, est_traj = load_trajectory(method_cfg, scene, geo_handler)
            logger.info(f"  Loaded {method_name} with {len(est_traj)} poses")

            # dead-reckoning for odometry
            if method_cfg.get("odom", False):
                if ref_t is not None and ref_traj is not None:
                    est_t, est_traj = compute_dead_reckoning(est_traj, ref_traj, est_t, ref_t)
                else:
                    logger.warning(f"  Skipping dead-reckoning for {method_name}: no reference available")

            trajectories[method_name] = (est_t, est_traj)

            # Compute ATE and orientation errors only if GT available
            if ref_t is not None and ref_traj is not None:
                pos_errors, ori_errors = compute_ate(ref_traj, est_traj, ref_t, est_t)

                # Store errors for CDF plotting
                errors[method_name] = {"pos_errors": pos_errors, "ori_errors": ori_errors}

                # ATE and AOE metrics
                ate_rmse = np.sqrt(np.mean(pos_errors**2))
                aoe_rmse = np.sqrt(np.mean(ori_errors**2))
                results[method_name] = {
                    "ate_rmse": ate_rmse,
                    "ate_median": np.median(pos_errors),
                    "ate_mean": np.mean(pos_errors),
                    "ate_std": np.std(pos_errors),
                    "aoe_rmse": aoe_rmse,
                    "aoe_median": np.median(ori_errors),
                    "aoe_mean": np.mean(ori_errors),
                    "aoe_std": np.std(ori_errors),
                }

                logger.info(f"  {method_name}: ATE={ate_rmse:.3f}m, AOE={aoe_rmse:.2f}°")
            else:
                logger.info(f"  {method_name}: Loaded {len(est_t)} poses (no evaluation)")

        except Exception as e:
            logger.warning(f"  Failed to evaluate {method_name}: {e}")

    return trajectories, errors, results


def draw_scale_bar(
    ax,
    m_per_pixel,
    length_m=100,
    color="black",
    pad=3.0,
    font_size=20,
    max_frac=0.30,  # bar ≤ this fraction of visible x-range
    min_length_m=1.0,
    tick_px=40,  # vertical tick length in *pixels*
    linewidth=5,
):
    if m_per_pixel <= 0:
        raise ValueError("m_per_pixel must be > 0")

    # visible width in datma units (same coords you use for the bar)
    x0, x1 = ax.get_xlim()
    data_width = abs(x1 - x0)

    # desired bar length in data units
    bar_du = length_m / m_per_pixel
    while bar_du > max_frac * data_width and length_m > min_length_m:
        length_m *= 0.5
        bar_du *= 0.5

    def _data_dy_for_pixels(ax, n_px: float) -> float:
        # convert a vertical size in pixels to data units (handle inverted axes)
        y0 = ax.transData.transform((0, 0))[1]
        y1 = ax.transData.transform((0, 1))[1]
        ppd = abs(y1 - y0)  # pixels per 1 data-y
        return 0.0 if ppd == 0 else (n_px / ppd)

    # balanced tick: symmetric about baseline, length defined in pixels
    tick_du = _data_dy_for_pixels(ax, tick_px)
    y_lo, y_hi = -0.5 * tick_du, 0.5 * tick_du

    bar_box = AuxTransformBox(ax.transData)
    # horizontal bar
    bar_box.add_artist(Line2D([0, bar_du], [0, 0], color=color, linewidth=linewidth))
    # balanced vertical ticks at both ends
    bar_box.add_artist(Line2D([0, 0], [y_lo, y_hi], color=color, linewidth=linewidth))
    bar_box.add_artist(Line2D([bar_du, bar_du], [y_lo, y_hi], color=color, linewidth=linewidth))

    def _fmt_len_m(meters: float) -> str:
        return f"{int(round(meters / 1000))} km" if meters >= 1000 else f"{int(round(meters))} m"

    txt = TextArea(
        _fmt_len_m(length_m),
        textprops=dict(color=color, size=font_size, va="center", ha="right"),
    )
    packed = HPacker(children=[txt, bar_box], align="center", pad=0, sep=10)

    anchored = AnchoredOffsetbox(
        loc="lower right",
        child=packed,
        pad=pad,
        borderpad=0.5,
        frameon=False,
    )
    ax.add_artist(anchored)
    return anchored


def plot_trajectories(trajectories: dict, geo_handler: GeoTiffHandler, config: dict, save_path: Path):
    """Plot trajectories on map"""

    fig, ax = plt.subplots(figsize=(20, 20))

    # Setup colors
    colors = {}
    for method in config["methods"]:
        colors[method["name"]] = method.get("color", "blue")

    # Plot on map
    all_points = []
    for _, (_, traj) in trajectories.items():
        all_points.append([(p.x(), p.y()) for p in traj])
    all_points = np.vstack(all_points)

    pixels = geo_handler.coords_to_pixel(all_points)
    pad = 50  # Reduced padding
    x_min = max(0, int(pixels[:, 0].min() - pad))
    x_max = min(geo_handler.image.shape[1], int(pixels[:, 0].max() + pad))
    y_min = max(0, int(pixels[:, 1].min() - pad))
    y_max = min(geo_handler.image.shape[0], int(pixels[:, 1].max() + pad))

    ax.imshow(geo_handler.image[y_min:y_max, x_min:x_max], alpha=config["viz"]["alpha"])

    # Find reference method for styling
    reference_name = None
    for method in config["methods"]:
        if method.get("reference", False):
            reference_name = method["name"]
            break

    for name, (_, traj) in trajectories.items():
        coords = np.array([(p.x(), p.y()) for p in traj])
        pixels = geo_handler.coords_to_pixel(coords)
        style = "--" if name == reference_name else "-"
        linewidth = 8 if name == reference_name else (8 if name == "Ours" else 5)
        ax.plot(
            pixels[:, 0] - x_min,
            pixels[:, 1] - y_min,
            label=name,
            color=colors[name],
            linestyle=style,
            linewidth=linewidth,
        )

    # Add scale bar
    resolution = np.mean(geo_handler.resolution)
    draw_scale_bar(ax, resolution, 100, pad=3.0, font_size=25)

    # Legend
    leg = ax.legend(frameon=True, fontsize=25)
    for txt in leg.get_texts():
        if txt.get_text() == "Ours":
            txt.set_fontweight("bold")

    ax.axis("off")

    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_error_cdf(
    all_errors: dict[str, dict[str, list[np.ndarray]]],
    config: dict,
    save_path: Path,
    xlabel: str = "Absolute Position Error (m)",
    max_x: float = 10.0,
):
    """Plot CDF of errors with solid lines for seen, dotted for unseen"""

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {}
    for method in config["methods"]:
        colors[method["name"]] = method.get("color", "blue")

    n_plot = 1000

    for method_name, scene_errors in all_errors.items():
        color = colors.get(method_name, "gray")

        for scene_type in ["seen", "unseen"]:
            if scene_type not in scene_errors or not scene_errors[scene_type]:
                continue

            # Concatenate all errors for this scene type
            all_err = np.concatenate(scene_errors[scene_type])
            sorted_err = np.sort(all_err)
            cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)

            # Filter to max_x
            mask = sorted_err <= max_x
            sorted_err = sorted_err[mask]
            cdf = cdf[mask]

            # Subsample if too many points
            if len(sorted_err) > n_plot:
                idx = np.linspace(0, len(sorted_err) - 1, n_plot, dtype=int)
                sorted_err = sorted_err[idx]
                cdf = cdf[idx]

            # Style: solid for seen, dotted for unseen
            style = "-" if scene_type == "seen" else ":"
            # Only add label for seen (to avoid duplicate legend entries)
            label = method_name if scene_type == "seen" else None

            ax.plot(sorted_err, cdf, label=label, color=color, linestyle=style, linewidth=4)

            # Add horizontal line to extend to max_x if CDF reaches 1.0
            if len(cdf) > 0 and cdf[-1] >= 1.0 and sorted_err[-1] < max_x:
                ax.hlines(1.0, xmin=sorted_err[-1], xmax=max_x, colors=color, linestyles=style, linewidth=4)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("CDF", fontsize=20)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)

    # Add annotation for line styles
    ax.text(
        0.98,
        0.02,
        "solid = seen route\ndotted = unseen route",
        transform=ax.transAxes,
        fontsize=16,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(config: dict, output_dir: Path):
    dataset_cfg = config["dataset"]
    geo_handler = GeoTiffHandler(dataset_cfg["geotiff"])

    all_results = []
    scene_types = list(dataset_cfg["scenes"].keys())
    all_pos_errors = {m["name"]: {st: [] for st in scene_types} for m in config["methods"]}
    all_ori_errors = {m["name"]: {st: [] for st in scene_types} for m in config["methods"]}

    # Process scenes
    for scene_type, scenes in dataset_cfg["scenes"].items():
        for scene in scenes:
            logger.info(f"Scene ({scene_type}): {scene}")

            trajectories, scene_errs, results = evaluate_scene(scene, config["methods"], geo_handler)

            for method_name, method_errors in scene_errs.items():
                all_pos_errors[method_name][scene_type].append(method_errors["pos_errors"])
                all_ori_errors[method_name][scene_type].append(method_errors["ori_errors"])

            for method_name, metrics in results.items():
                all_results.append({"scene": scene, "scene_type": scene_type, "method": method_name, **metrics})

            # plot trajectory per scene
            plot_path = output_dir / f"{scene}_trajectory.png"
            plot_trajectories(trajectories, geo_handler, config, plot_path)

    if all_results:
        # Save results
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "results.csv", index=False)

        # Print summary by method and scene type
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY BY METHOD AND SCENE TYPE:")
        summary = df.groupby(["method", "scene_type"])[["ate_rmse", "ate_median", "aoe_rmse", "aoe_median"]].agg(
            ["mean", "std"]
        )
        print(summary.round(3))

        logger.info("\n" + "=" * 60)
        logger.info("OVERALL SUMMARY BY METHOD:")
        summary = df.groupby("method")[["ate_rmse", "ate_median", "aoe_rmse", "aoe_median"]].agg(["mean", "std"])
        print(summary.round(3))

    # Plot CDF
    plot_error_cdf(all_pos_errors, config, output_dir / "position_error_cdf.png")
    plot_error_cdf(
        all_ori_errors,
        config,
        output_dir / "orientation_error_cdf.png",
        xlabel="Absolute Orientation Error (degrees)",
        max_x=30,
    )
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="paper_scripts/config/experiment_tartandrive.yaml")
    parser.add_argument("--output", default="paper_scripts/pf_results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    main(config, output_dir)
