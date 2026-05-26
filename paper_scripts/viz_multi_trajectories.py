"""Visualize multiple trajectories on one GeoTIFF using explicit file paths."""

import argparse
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from manifpy import SE2, SE3
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AuxTransformBox, HPacker, TextArea
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredOffsetbox

from geotiff.handler import GeoTiffHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VerticalTupleHandler(HandlerBase):
    """Draw tuple handles as vertically stacked line swatches."""

    def __init__(self, pad_fraction: float = 0.8):
        super().__init__()
        self.pad_fraction = float(pad_fraction)

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        n = len(orig_handle)
        if n == 0:
            return []

        x0 = -xdescent
        x1 = -xdescent + width
        y_mid = (height - ydescent) / 2.0
        effective_h = max(1.0, float(height - ydescent))

        if n == 1:
            h = orig_handle[0]
            line = Line2D([x0, x1], [y_mid, y_mid])
            color = h.get_color()
            if h.get_linestyle() == "None":
                color = h.get_markerfacecolor()
            line.set_color(color)
            line.set_linewidth(max(2.5, float(h.get_linewidth())))
            line.set_solid_capstyle("round")
            line.set_transform(trans)
            return [line]

        base = effective_h / max(1.0, n + self.pad_fraction * (n - 1))
        stride = base * (1.0 + self.pad_fraction)
        offsets = np.linspace((n - 1) / 2.0, -(n - 1) / 2.0, n) * stride

        artists = []
        for offset, h in zip(offsets, orig_handle):
            y_center = y_mid + float(offset)
            line = Line2D([x0, x1], [y_center, y_center])
            color = h.get_color()
            if h.get_linestyle() == "None":
                color = h.get_markerfacecolor()
            line.set_color(color)
            line.set_linewidth(max(2.5, float(h.get_linewidth())))
            line.set_solid_capstyle("round")
            line.set_transform(trans)
            artists.append(line)
        return artists


def resolve_existing_path(raw_path: str, config_dir: Path, field_name: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate

    if candidate.is_absolute():
        raise FileNotFoundError(f"{field_name} not found: {candidate}")

    candidate_from_config = (config_dir / candidate).expanduser()
    if candidate_from_config.exists():
        return candidate_from_config

    raise FileNotFoundError(f"{field_name} not found: {raw_path}. Tried '{candidate}' and '{candidate_from_config}'.")


def resolve_output_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def normalize_legend_cfg(raw_legend_cfg: dict | None) -> dict:
    if raw_legend_cfg is None:
        raw_legend_cfg = {}
    if not isinstance(raw_legend_cfg, dict):
        raise ValueError("viz.legend must be a mapping when provided.")

    legend_cfg = dict(raw_legend_cfg)
    legend_mode = str(legend_cfg.get("mode", "default")).lower()
    if legend_mode not in {"default", "grouped"}:
        raise ValueError("viz.legend.mode must be one of ['default', 'grouped'].")
    legend_cfg["mode"] = legend_mode
    return legend_cfg


def _find_header_index(headers: list[str], candidates: list[str]) -> int | None:
    for candidate in candidates:
        if candidate in headers:
            return headers.index(candidate)
    return None


def _load_csv_header(traj_path: Path) -> list[str]:
    with open(traj_path) as f:
        line = f.readline().strip()
    if not line:
        return []
    return [col.strip().lower() for col in line.split(",")]


def load_numeric_table(traj_path: Path) -> tuple[np.ndarray, list[str]]:
    headers: list[str] = []
    if traj_path.suffix == ".csv":
        headers = _load_csv_header(traj_path)
        data = np.genfromtxt(traj_path, delimiter=",", skip_header=1, dtype=float)
    elif traj_path.suffix == ".txt":
        data = np.loadtxt(traj_path, dtype=float)
    else:
        raise NotImplementedError(f"Unsupported trajectory file extension: {traj_path.suffix}")

    if data.size == 0:
        raise ValueError(f"Trajectory file is empty: {traj_path}")

    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError(f"Expected 2D trajectory table, got shape {data.shape} for {traj_path}")

    return data, headers


def load_trajectory(traj_cfg: dict, geo_handler: GeoTiffHandler, config_dir: Path):
    raw_path = traj_cfg.get("path")
    if not raw_path:
        raise ValueError("Missing required key 'path' in trajectory entry")

    traj_path = resolve_existing_path(str(raw_path), config_dir, "Trajectory file")
    traj_fmt = traj_cfg.get("format", "t_xyr")
    traj_frame = traj_cfg.get("frame", "enu")
    data, headers = load_numeric_table(traj_path)
    timestamp_idx = _find_header_index(headers, ["timestamp", "time", "stamp", "t"])
    timestamps = data[:, timestamp_idx] if timestamp_idx is not None else data[:, 0]

    if traj_fmt == "t_xyr":
        if data.shape[1] < 4:
            raise ValueError(f"t_xyr expects >=4 columns [t,x,y,r], got {data.shape[1]} in {traj_path}")

        x_idx = _find_header_index(headers, ["x"])
        y_idx = _find_header_index(headers, ["y"])
        yaw_idx = _find_header_index(headers, ["angle", "yaw", "theta", "r", "heading"])
        if headers:
            if x_idx is None or y_idx is None:
                raise ValueError(f"t_xyr CSV must include x and y columns: {traj_path}")
            if yaw_idx is None:
                raise ValueError(
                    f"t_xyr CSV must include one of [angle,yaw,theta,r,heading]. Found columns {headers} in {traj_path}"
                )
            xy_yaw = data[:, [x_idx, y_idx, yaw_idx]]
        else:
            if data.shape[1] != 4:
                raise ValueError(f"Ambiguous t_xyr without headers; expected exactly 4 columns in {traj_path}")
            xy_yaw = data[:, 1:4]

        if traj_frame == "enu":
            poses = [SE2(x, y, theta) for x, y, theta in xy_yaw]
        elif traj_frame == "ned":
            poses = [SE2(x, y, np.pi / 2 - theta) for x, y, theta in xy_yaw]
        else:
            raise NotImplementedError(f"Unsupported coordinate frame for t_xyr: {traj_frame}")
        return timestamps, poses, traj_path

    if traj_fmt == "t_xyz_qxyzw":
        if data.shape[1] < 8:
            raise ValueError(
                f"t_xyz_qxyzw expects >=8 columns [t,x,y,z,qx,qy,qz,qw], got {data.shape[1]} in {traj_path}"
            )
        if headers:
            required = ["x", "y", "z", "qx", "qy", "qz", "qw"]
            missing = [name for name in required if name not in headers]
            if missing:
                raise ValueError(f"t_xyz_qxyzw CSV missing columns {missing} in {traj_path}")
            pose_cols = [headers.index(name) for name in required]
            pose_data = data[:, pose_cols]
        else:
            pose_data = data[:, 1:8]

        if traj_frame == "enu":
            poses_se3 = [SE3(xyz_qxyzw) for xyz_qxyzw in pose_data]
        elif traj_frame == "cam":
            cam2enu = SE3(np.array([0, 0, 0]), np.array([0.5, -0.5, 0.5, -0.5]))
            enu2cam = SE3(np.array([0, 0, 0]), np.array([0.5, -0.5, 0.5, 0.5]))
            poses_se3 = [cam2enu * SE3(pose) * enu2cam for pose in pose_data]
        elif traj_frame == "ned":
            ned2enu = SE3(np.array([0, 0, 0]), np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]))
            enu2ned = SE3(np.array([0, 0, 0]), np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]))
            poses_se3 = [ned2enu * SE3(pose) * enu2ned for pose in pose_data]
        else:
            raise NotImplementedError(f"Unsupported coordinate frame for t_xyz_qxyzw: {traj_frame}")

        poses = []
        for pose_se3 in poses_se3:
            x, y, _ = pose_se3.translation()
            rotation = pose_se3.rotation()
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            poses.append(SE2(x, y, yaw))
        return timestamps, poses, traj_path

    if traj_fmt == "t_gps":
        if data.shape[1] < 5:
            raise ValueError(f"t_gps expects >=5 columns with lat/lon at [3,4], got {data.shape[1]} in {traj_path}")
        if headers:
            lat_idx = _find_header_index(headers, ["lat", "latitude"])
            lon_idx = _find_header_index(headers, ["lon", "longitude", "long"])
            if lat_idx is None or lon_idx is None:
                raise ValueError(f"t_gps CSV must include lat/lon columns in {traj_path}")
            latlons = data[:, [lat_idx, lon_idx]]
        else:
            latlons = data[:, [3, 4]]
        utm_coords = geo_handler.latlon_to_coords(latlons)
        poses = [SE2(*xy, 0.0) for xy in utm_coords]
        return timestamps, poses, traj_path

    if traj_fmt == "utm_points":
        if headers:
            x_idx = _find_header_index(headers, ["utm_easting", "easting", "x"])
            y_idx = _find_header_index(headers, ["utm_northing", "northing", "y"])
            if x_idx is None or y_idx is None:
                raise ValueError(
                    f"utm_points CSV must include UTM columns. "
                    f"Expected one of [utm_easting/easting/x] and [utm_northing/northing/y] in {traj_path}"
                )
            xy = data[:, [x_idx, y_idx]]
        else:
            if data.shape[1] < 2:
                raise ValueError(f"utm_points without headers expects >=2 columns in {traj_path}")
            xy = data[:, :2]

        timestamps = np.arange(len(xy), dtype=np.float64)
        poses = [SE2(x, y, 0.0) for x, y in xy]
        return timestamps, poses, traj_path

    raise NotImplementedError(f"Unsupported trajectory format key: {traj_fmt}")


def uniquify_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        return name

    suffix = 2
    while f"{name} ({suffix})" in used_names:
        suffix += 1
    return f"{name} ({suffix})"


def load_trajectories(config: dict, geo_handler: GeoTiffHandler, config_dir: Path):
    traj_cfgs = config.get("trajectories")
    if not isinstance(traj_cfgs, list) or len(traj_cfgs) == 0:
        raise ValueError("Config must define a non-empty 'trajectories' list")

    trajectories = {}
    colors = {}
    styles = {}
    marker_sizes = {}
    failures = 0

    for idx, traj_cfg in enumerate(traj_cfgs):
        if not isinstance(traj_cfg, dict):
            logger.warning("Skipping trajectories[%d]: expected mapping, got %s", idx, type(traj_cfg).__name__)
            failures += 1
            continue

        try:
            est_t, est_traj, traj_path = load_trajectory(traj_cfg, geo_handler, config_dir)
            if len(est_traj) == 0:
                logger.warning("Skipping trajectories[%d]: no trajectory points in %s", idx, traj_path)
                failures += 1
                continue

            coords = np.array([(pose.x(), pose.y()) for pose in est_traj], dtype=float)
            pixels = geo_handler.coords_to_pixel(coords)
            width = geo_handler.image.shape[1]
            height = geo_handler.image.shape[0]
            inside = (
                np.isfinite(pixels[:, 0])
                & np.isfinite(pixels[:, 1])
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < height)
            )
            inside_indices = np.nonzero(inside)[0]
            if len(inside_indices) == 0:
                raise ValueError(
                    f"Trajectory appears outside GeoTIFF bounds for {traj_path}. "
                    "Check format/frame and coordinate reference."
                )
            if len(inside_indices) < len(est_traj):
                logger.warning(
                    "Clipping %s to in-bounds points: %d -> %d",
                    traj_path.name,
                    len(est_traj),
                    len(inside_indices),
                )
                est_t = est_t[inside_indices]
                est_traj = [est_traj[i] for i in inside_indices]

            display_name = traj_cfg.get("name", Path(traj_path).stem)
            unique_name = uniquify_name(display_name, set(trajectories))
            if unique_name != display_name:
                logger.warning("Duplicate trajectory name '%s'; using '%s'", display_name, unique_name)

            trajectories[unique_name] = (est_t, est_traj)
            colors[unique_name] = traj_cfg.get("color", "blue")
            styles[unique_name] = traj_cfg.get("style", "line")
            marker_sizes[unique_name] = float(traj_cfg.get("marker_size", 8.0))
            logger.info("Loaded %s from %s (%d poses)", unique_name, traj_path, len(est_traj))
        except Exception as exc:
            failures += 1
            logger.warning("Failed trajectories[%d]: %s", idx, exc)

    if not trajectories:
        raise ValueError("No trajectories loaded successfully from config")

    logger.info("Trajectory loading summary: %d loaded, %d failed", len(trajectories), failures)
    return trajectories, colors, styles, marker_sizes


def select_named_trajectories(
    trajectories: dict[str, tuple[np.ndarray, list[SE2]]],
    colors: dict[str, str],
    styles: dict[str, str],
    marker_sizes: dict[str, float],
    selected_names: list[str],
):
    selected_trajectories: dict[str, tuple[np.ndarray, list[SE2]]] = {}
    selected_colors: dict[str, str] = {}
    selected_styles: dict[str, str] = {}
    selected_marker_sizes: dict[str, float] = {}

    for name in selected_names:
        if name not in trajectories:
            available = ", ".join(sorted(trajectories))
            raise ValueError(f"Unknown trajectory name '{name}'. Available names: [{available}]")
        selected_trajectories[name] = trajectories[name]
        selected_colors[name] = colors[name]
        selected_styles[name] = styles[name]
        selected_marker_sizes[name] = marker_sizes[name]

    if not selected_trajectories:
        raise ValueError("No trajectories selected for figure")
    return selected_trajectories, selected_colors, selected_styles, selected_marker_sizes


def draw_scale_bar(
    ax,
    m_per_pixel,
    length_m=100,
    color="black",
    pad=3.0,
    font_size=20,
    max_frac=0.30,
    min_length_m=1.0,
    tick_px=40,
    linewidth=5,
):
    if m_per_pixel <= 0:
        raise ValueError("m_per_pixel must be > 0")

    x0, x1 = ax.get_xlim()
    data_width = abs(x1 - x0)

    bar_du = length_m / m_per_pixel
    while bar_du > max_frac * data_width and length_m > min_length_m:
        length_m *= 0.5
        bar_du *= 0.5

    def _data_dy_for_pixels(axis, n_px: float) -> float:
        y0 = axis.transData.transform((0, 0))[1]
        y1 = axis.transData.transform((0, 1))[1]
        ppd = abs(y1 - y0)
        return 0.0 if ppd == 0 else (n_px / ppd)

    tick_du = _data_dy_for_pixels(ax, tick_px)
    y_lo, y_hi = -0.5 * tick_du, 0.5 * tick_du

    bar_box = AuxTransformBox(ax.transData)
    bar_box.add_artist(Line2D([0, bar_du], [0, 0], color=color, linewidth=linewidth))
    bar_box.add_artist(Line2D([0, 0], [y_lo, y_hi], color=color, linewidth=linewidth))
    bar_box.add_artist(Line2D([bar_du, bar_du], [y_lo, y_hi], color=color, linewidth=linewidth))

    label = f"{int(round(length_m / 1000))} km" if length_m >= 1000 else f"{int(round(length_m))} m"
    text = TextArea(label, textprops=dict(color=color, size=font_size, va="center", ha="right"))
    packed = HPacker(children=[text, bar_box], align="center", pad=0, sep=10)

    anchored = AnchoredOffsetbox(
        loc="lower right",
        child=packed,
        pad=pad,
        borderpad=0.5,
        frameon=False,
    )
    ax.add_artist(anchored)


def compute_crop_bounds(
    trajectories: dict[str, tuple[np.ndarray, list[SE2]]],
    geo_handler: GeoTiffHandler,
    padding_px: int,
) -> tuple[int, int, int, int]:
    if not trajectories:
        raise ValueError("Cannot compute crop bounds from empty trajectory set.")

    all_coords = np.vstack([np.array([(pose.x(), pose.y()) for pose in traj]) for _, traj in trajectories.values()])
    all_pixels = geo_handler.coords_to_pixel(all_coords)

    x_min = max(0, int(np.floor(all_pixels[:, 0].min() - padding_px)))
    x_max = min(geo_handler.image.shape[1], int(np.ceil(all_pixels[:, 0].max() + padding_px)))
    y_min = max(0, int(np.floor(all_pixels[:, 1].min() - padding_px)))
    y_max = min(geo_handler.image.shape[0], int(np.ceil(all_pixels[:, 1].max() + padding_px)))
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            "Trajectory bounds do not overlap the GeoTIFF extent. Check trajectory alignment and coordinate frame."
        )
    return x_min, x_max, y_min, y_max


def plot_trajectories(
    trajectories: dict[str, tuple[np.ndarray, list[SE2]]],
    colors: dict[str, str],
    styles: dict[str, str],
    marker_sizes: dict[str, float],
    geo_handler: GeoTiffHandler,
    output_path: Path,
    map_alpha: float,
    padding_px: int,
    line_width: float,
    legend_cfg: dict | None = None,
    crop_bounds: tuple[int, int, int, int] | None = None,
):
    fig, ax = plt.subplots(figsize=(20, 20))

    if crop_bounds is None:
        x_min, x_max, y_min, y_max = compute_crop_bounds(trajectories, geo_handler, padding_px)
    else:
        x_min, x_max, y_min, y_max = crop_bounds

    ax.imshow(geo_handler.image[y_min:y_max, x_min:x_max], alpha=map_alpha)

    legend_proxies: dict[str, Line2D] = {}
    for name, (_, traj) in trajectories.items():
        coords = np.array([(pose.x(), pose.y()) for pose in traj])
        pixels = geo_handler.coords_to_pixel(coords)
        style = styles.get(name, "line")
        color = colors.get(name, "blue")
        if style == "points":
            marker_size = marker_sizes.get(name, 8.0)
            ax.scatter(
                pixels[:, 0] - x_min,
                pixels[:, 1] - y_min,
                label=name,
                color=color,
                s=marker_size,
                linewidths=0,
                alpha=0.9,
            )
            legend_proxies[name] = Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=max(2.0, float(np.sqrt(marker_size))),
                markerfacecolor=color,
                markeredgecolor=color,
                label=name,
            )
        else:
            ax.plot(
                pixels[:, 0] - x_min,
                pixels[:, 1] - y_min,
                label=name,
                color=color,
                linewidth=line_width,
            )
            legend_proxies[name] = Line2D([0], [0], color=color, linewidth=line_width, label=name)

    draw_scale_bar(ax, geo_handler.resolution, length_m=100, pad=3.0, font_size=24)
    legend_fontsize = 20
    show_legend = True
    if legend_cfg:
        show_legend = bool(legend_cfg.get("show", True))
        legend_fontsize = int(legend_cfg.get("fontsize", 20))
        if legend_fontsize <= 0:
            raise ValueError("viz.legend.fontsize must be > 0.")
    if show_legend and legend_cfg and legend_cfg.get("mode") == "grouped":
        groups = legend_cfg.get("groups")
        if not isinstance(groups, list) or len(groups) == 0:
            raise ValueError("viz.legend.groups must be a non-empty list when viz.legend.mode='grouped'.")

        handles: list[tuple[Line2D, ...]] = []
        labels: list[str] = []
        missing_members: set[str] = set()
        for idx, group in enumerate(groups):
            if not isinstance(group, dict):
                raise ValueError(f"viz.legend.groups[{idx}] must be a mapping.")
            label = group.get("label")
            members = group.get("members")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"viz.legend.groups[{idx}].label must be a non-empty string.")
            if not isinstance(members, list) or len(members) == 0:
                raise ValueError(f"viz.legend.groups[{idx}].members must be a non-empty list.")

            member_handles: list[Line2D] = []
            for member in members:
                member_name = str(member)
                handle = legend_proxies.get(member_name)
                if handle is None:
                    missing_members.add(member_name)
                else:
                    member_handles.append(handle)

            if member_handles:
                handles.append(tuple(member_handles))
                labels.append(label)

        if missing_members:
            available = ", ".join(sorted(legend_proxies))
            missing = ", ".join(sorted(missing_members))
            raise ValueError(
                f"viz.legend.groups references unknown trajectory names: [{missing}]. Available names: [{available}]"
            )
        if not handles:
            raise ValueError("Grouped legend resolved to zero handles.")

        line_gap_fraction = float(legend_cfg.get("line_gap_fraction", 0.8))
        if not (0.0 <= line_gap_fraction <= 2.0):
            raise ValueError("viz.legend.line_gap_fraction must be within [0, 2].")
        group_label_spacing = legend_cfg.get("group_label_spacing")
        if group_label_spacing is None:
            group_label_spacing = 0.8 + 0.5 * line_gap_fraction
        group_label_spacing = float(group_label_spacing)
        if not (0.0 <= group_label_spacing <= 5.0):
            raise ValueError("viz.legend.group_label_spacing must be within [0, 5].")
        group_handle_height = legend_cfg.get("group_handle_height")
        if group_handle_height is None:
            group_handle_height = 1.0 + 0.7 * line_gap_fraction
        group_handle_height = float(group_handle_height)
        if not (0.1 <= group_handle_height <= 6.0):
            raise ValueError("viz.legend.group_handle_height must be within [0.1, 6].")

        ax.legend(
            handles,
            labels,
            loc="upper left",
            frameon=True,
            fontsize=legend_fontsize,
            labelspacing=group_label_spacing,
            handleheight=group_handle_height,
            handler_map={tuple: VerticalTupleHandler(pad_fraction=line_gap_fraction)},
        )
    elif show_legend:
        ax.legend(loc="upper left", frameon=True, fontsize=legend_fontsize)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(args):
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    config_dir = config_path.parent
    geotiff_key = config.get("geotiff")
    if not geotiff_key:
        raise ValueError("Config is missing required top-level key: geotiff")

    geotiff_path = resolve_existing_path(str(geotiff_key), config_dir, "GeoTIFF")
    geo_handler = GeoTiffHandler(str(geotiff_path))

    viz_cfg = config.get("viz", {}) if isinstance(config.get("viz", {}), dict) else {}
    map_alpha = float(args.alpha if args.alpha is not None else viz_cfg.get("alpha", 0.6))
    padding_px = int(args.padding if args.padding is not None else viz_cfg.get("padding_px", 50))
    line_width = float(args.linewidth if args.linewidth is not None else viz_cfg.get("linewidth", 5))
    legend_cfg = normalize_legend_cfg(viz_cfg.get("legend", {}))

    logger.info("GeoTIFF: %s", geotiff_path)
    trajectories, colors, styles, marker_sizes = load_trajectories(config, geo_handler, config_dir=config_dir)

    figures_cfg = config.get("figures")
    if figures_cfg is None:
        output_cfg = config.get("output", {})
        if output_cfg is None:
            output_cfg = {}
        if not isinstance(output_cfg, dict):
            raise ValueError("Config key 'output' must be a mapping when provided.")

        if args.out:
            output_path = resolve_output_path(str(args.out))
        elif output_cfg.get("path"):
            output_path = resolve_output_path(str(output_cfg["path"]))
        else:
            raise ValueError("Output path is required: provide --out or output.path in config.")

        plot_trajectories(
            trajectories=trajectories,
            colors=colors,
            styles=styles,
            marker_sizes=marker_sizes,
            geo_handler=geo_handler,
            output_path=output_path,
            map_alpha=map_alpha,
            padding_px=padding_px,
            line_width=line_width,
            legend_cfg=legend_cfg,
        )
        logger.info("Saved visualization to %s", output_path)
        return

    if not isinstance(figures_cfg, list) or len(figures_cfg) == 0:
        raise ValueError("Config key 'figures' must be a non-empty list when provided.")
    if args.out and len(figures_cfg) != 1:
        raise ValueError("--out can only be used with multi-figure configs when exactly one figure is defined.")

    figure_plans = []
    shared_crop_names: list[str] = []
    available_names = set(trajectories)
    for idx, figure_cfg in enumerate(figures_cfg):
        if not isinstance(figure_cfg, dict):
            raise ValueError(f"figures[{idx}] must be a mapping.")

        selected_names = figure_cfg.get("trajectories")
        if not isinstance(selected_names, list) or len(selected_names) == 0:
            raise ValueError(f"figures[{idx}].trajectories must be a non-empty list of trajectory names.")

        deduped_names: list[str] = []
        seen = set()
        for raw_name in selected_names:
            name = str(raw_name)
            if name not in available_names:
                available = ", ".join(sorted(available_names))
                raise ValueError(f"figures[{idx}] references unknown trajectory '{name}'. Available: [{available}]")
            if name in seen:
                continue
            seen.add(name)
            deduped_names.append(name)
        if not deduped_names:
            raise ValueError(f"figures[{idx}] has no valid trajectory names after deduplication.")

        shared_crop_names.extend(deduped_names)
        figure_viz_cfg = figure_cfg.get("viz", {})
        if figure_viz_cfg is None:
            figure_viz_cfg = {}
        if not isinstance(figure_viz_cfg, dict):
            raise ValueError(f"figures[{idx}].viz must be a mapping when provided.")

        figure_map_alpha = float(figure_viz_cfg.get("alpha", map_alpha))
        figure_line_width = float(figure_viz_cfg.get("linewidth", line_width))

        figure_legend_cfg = dict(legend_cfg)
        if "legend" in figure_viz_cfg:
            legend_override = normalize_legend_cfg(figure_viz_cfg.get("legend"))
            figure_legend_cfg.update(legend_override)

        if args.out:
            figure_output_path = resolve_output_path(str(args.out))
        else:
            output_cfg = figure_cfg.get("output", {})
            if output_cfg is None:
                output_cfg = {}
            if not isinstance(output_cfg, dict):
                raise ValueError(f"figures[{idx}].output must be a mapping when provided.")
            if not output_cfg.get("path"):
                raise ValueError(f"figures[{idx}].output.path is required.")
            figure_output_path = resolve_output_path(str(output_cfg["path"]))

        figure_plans.append(
            {
                "index": idx,
                "trajectory_names": deduped_names,
                "map_alpha": figure_map_alpha,
                "line_width": figure_line_width,
                "legend_cfg": figure_legend_cfg,
                "output_path": figure_output_path,
            }
        )

    shared_crop_unique_names = list(dict.fromkeys(shared_crop_names))
    shared_trajectories, _, _, _ = select_named_trajectories(
        trajectories=trajectories,
        colors=colors,
        styles=styles,
        marker_sizes=marker_sizes,
        selected_names=shared_crop_unique_names,
    )
    shared_crop_bounds = compute_crop_bounds(shared_trajectories, geo_handler, padding_px)

    for plan in figure_plans:
        selected_trajectories, selected_colors, selected_styles, selected_marker_sizes = select_named_trajectories(
            trajectories=trajectories,
            colors=colors,
            styles=styles,
            marker_sizes=marker_sizes,
            selected_names=plan["trajectory_names"],
        )
        plot_trajectories(
            trajectories=selected_trajectories,
            colors=selected_colors,
            styles=selected_styles,
            marker_sizes=selected_marker_sizes,
            geo_handler=geo_handler,
            output_path=plan["output_path"],
            map_alpha=plan["map_alpha"],
            padding_px=padding_px,
            line_width=plan["line_width"],
            legend_cfg=plan["legend_cfg"],
            crop_bounds=shared_crop_bounds,
        )
        logger.info("Saved figures[%d] visualization to %s", plan["index"], plan["output_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize multiple trajectories on one GeoTIFF from explicit trajectory file paths."
    )
    parser.add_argument("--config", required=True, help="Path to direct-path YAML config.")
    parser.add_argument("--out", default=None, help="Optional output PNG path override.")
    parser.add_argument(
        "--alpha", type=float, default=None, help="GeoTIFF alpha override. Default: config viz.alpha or 0.6"
    )
    parser.add_argument(
        "--padding", type=int, default=None, help="Pixel padding override. Default: config viz.padding_px or 50"
    )
    parser.add_argument(
        "--linewidth", type=float, default=None, help="Line width override. Default: config viz.linewidth or 5"
    )
    main(parser.parse_args())
