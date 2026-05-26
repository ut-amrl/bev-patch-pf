#!/usr/bin/env python3
"""Compute Hausdorff distance between a reference trajectory and other runs."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import numpy as np
import yaml

from geotiff.handler import GeoTiffHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback path is tested by runtime behavior
    cKDTree = None


def resolve_existing_path(raw_path: str, config_dir: Path, field_name: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate

    if candidate.is_absolute():
        raise FileNotFoundError(f"{field_name} not found: {candidate}")

    candidate_from_config = (config_dir / candidate).expanduser()
    if candidate_from_config.exists():
        return candidate_from_config

    raise FileNotFoundError(
        f"{field_name} not found: {raw_path}. Tried '{candidate}' and '{candidate_from_config}'."
    )


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return sanitized or "experiment"


def _load_csv_header(traj_path: Path) -> list[str]:
    with open(traj_path, encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        return []
    return [col.strip().lower() for col in line.split(",")]


def _find_header_index(headers: list[str], candidates: list[str]) -> int | None:
    for candidate in candidates:
        if candidate in headers:
            return headers.index(candidate)
    return None


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
        return np.empty((0, 2), dtype=float), headers

    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError(f"Expected 2D trajectory table, got shape {data.shape} for {traj_path}")

    return data, headers


def _load_xy_t_xyr(data: np.ndarray, headers: list[str], traj_path: Path) -> np.ndarray:
    if headers:
        x_idx = _find_header_index(headers, ["x", "utm_easting", "easting"])
        y_idx = _find_header_index(headers, ["y", "utm_northing", "northing"])
        if x_idx is None or y_idx is None:
            raise ValueError(f"t_xyr CSV must include x and y columns in {traj_path}")
        return data[:, [x_idx, y_idx]]

    if data.shape[1] < 4:
        raise ValueError(f"t_xyr without headers expects >=4 columns [t,x,y,r] in {traj_path}")
    return data[:, 1:3]


def _load_xy_utm_points(data: np.ndarray, headers: list[str], traj_path: Path) -> np.ndarray:
    if headers:
        x_idx = _find_header_index(headers, ["utm_easting", "easting", "x"])
        y_idx = _find_header_index(headers, ["utm_northing", "northing", "y"])
        if x_idx is None or y_idx is None:
            raise ValueError(
                "utm_points CSV must include one of [utm_easting/easting/x] and "
                f"[utm_northing/northing/y] in {traj_path}"
            )
        return data[:, [x_idx, y_idx]]

    if data.shape[1] < 2:
        raise ValueError(f"utm_points without headers expects >=2 columns in {traj_path}")
    return data[:, :2]


def _load_xy_t_xyz_qxyzw(data: np.ndarray, headers: list[str], traj_path: Path) -> np.ndarray:
    if headers:
        x_idx = _find_header_index(headers, ["x"])
        y_idx = _find_header_index(headers, ["y"])
        if x_idx is None or y_idx is None:
            raise ValueError(f"t_xyz_qxyzw CSV missing x/y columns in {traj_path}")
        return data[:, [x_idx, y_idx]]

    if data.shape[1] < 8:
        raise ValueError(f"t_xyz_qxyzw without headers expects >=8 columns [t,x,y,z,qx,qy,qz,qw] in {traj_path}")
    return data[:, 1:3]


def _load_xy_t_gps(
    data: np.ndarray, headers: list[str], traj_path: Path, geo_handler: GeoTiffHandler | None
) -> np.ndarray:
    if geo_handler is None:
        raise ValueError(f"t_gps requires a valid geotiff in config; missing while loading {traj_path}")

    if headers:
        lat_idx = _find_header_index(headers, ["lat", "latitude"])
        lon_idx = _find_header_index(headers, ["lon", "longitude", "long"])
        if lat_idx is None or lon_idx is None:
            raise ValueError(f"t_gps CSV must include lat/lon columns in {traj_path}")
        latlons = data[:, [lat_idx, lon_idx]]
    else:
        if data.shape[1] < 5:
            raise ValueError(f"t_gps without headers expects >=5 columns with lat/lon at [3,4] in {traj_path}")
        latlons = data[:, [3, 4]]

    return geo_handler.latlon_to_coords(latlons)


def load_xy_trajectory(traj_cfg: dict, config_dir: Path, geo_handler: GeoTiffHandler | None) -> tuple[np.ndarray, Path]:
    raw_path = traj_cfg.get("path")
    if not raw_path:
        raise ValueError("Missing required key 'path' in trajectory entry")
    traj_path = resolve_existing_path(str(raw_path), config_dir, "Trajectory file")

    traj_fmt = traj_cfg.get("format", "t_xyr")
    data, headers = load_numeric_table(traj_path)
    if data.size == 0:
        return np.empty((0, 2), dtype=float), traj_path

    if traj_fmt == "t_xyr":
        xy = _load_xy_t_xyr(data, headers, traj_path)
    elif traj_fmt == "utm_points":
        xy = _load_xy_utm_points(data, headers, traj_path)
    elif traj_fmt == "t_xyz_qxyzw":
        xy = _load_xy_t_xyz_qxyzw(data, headers, traj_path)
    elif traj_fmt == "t_gps":
        xy = _load_xy_t_gps(data, headers, traj_path, geo_handler)
    else:
        raise NotImplementedError(f"Unsupported trajectory format key: {traj_fmt}")

    finite_mask = np.isfinite(xy).all(axis=1)
    if not np.all(finite_mask):
        dropped = int((~finite_mask).sum())
        logger.warning("Dropping %d non-finite rows from %s", dropped, traj_path)
        xy = xy[finite_mask]

    return xy.astype(float), traj_path


def remove_consecutive_duplicates(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if len(points) <= 1:
        return points

    keep = [0]
    for idx in range(1, len(points)):
        if np.linalg.norm(points[idx] - points[keep[-1]]) > eps:
            keep.append(idx)
    return points[np.array(keep, dtype=int)]


def resample_polyline(points: np.ndarray, step_m: float) -> np.ndarray:
    if step_m <= 0:
        raise ValueError("resample_step_m must be > 0")
    if len(points) == 0:
        return points

    points = remove_consecutive_duplicates(points)
    if len(points) <= 1:
        return points

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length <= 0:
        return points[:1]

    sample_d = np.arange(0.0, total_length, step_m, dtype=float)
    if sample_d.size == 0 or sample_d[-1] < total_length:
        sample_d = np.append(sample_d, total_length)

    x_interp = np.interp(sample_d, cumulative, points[:, 0])
    y_interp = np.interp(sample_d, cumulative, points[:, 1])
    return np.column_stack([x_interp, y_interp])


def _directed_hausdorff_bruteforce(source: np.ndarray, target: np.ndarray, chunk_size: int = 1024) -> float:
    max_min = 0.0
    for start in range(0, len(source), chunk_size):
        chunk = source[start : start + chunk_size]
        diffs = chunk[:, None, :] - target[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diffs, diffs, optimize=True)
        nearest = np.sqrt(np.min(d2, axis=1))
        local_max = float(np.max(nearest))
        if local_max > max_min:
            max_min = local_max
    return max_min


def directed_hausdorff(source: np.ndarray, target: np.ndarray) -> float:
    if len(source) == 0 or len(target) == 0:
        raise ValueError("Cannot compute Hausdorff with empty point sets")

    if cKDTree is not None:
        tree = cKDTree(target)
        dists, _ = tree.query(source, k=1)
        return float(np.max(dists))

    return _directed_hausdorff_bruteforce(source, target)


def overlap_bbox(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float] | None:
    ax0, ay0 = np.min(a, axis=0)
    ax1, ay1 = np.max(a, axis=0)
    bx0, by0 = np.min(b, axis=0)
    bx1, by1 = np.max(b, axis=0)

    x0 = max(ax0, bx0)
    x1 = min(ax1, bx1)
    y0 = max(ay0, by0)
    y1 = min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, x1, y0, y1


def clip_points_to_bbox(points: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1 = bbox
    mask = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    return points[mask]


def uniquify_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        return name

    suffix = 2
    while f"{name} ({suffix})" in used_names:
        suffix += 1
    return f"{name} ({suffix})"


def load_all_trajectories(config: dict, config_dir: Path) -> list[dict]:
    traj_cfgs = config.get("trajectories")
    if not isinstance(traj_cfgs, list) or not traj_cfgs:
        raise ValueError("Config must define non-empty 'trajectories'")

    geotiff_path_raw = config.get("geotiff")
    geo_handler = None
    if any(isinstance(cfg, dict) and cfg.get("format") == "t_gps" for cfg in traj_cfgs):
        if not geotiff_path_raw:
            raise ValueError("Config must define 'geotiff' when any trajectory format is t_gps")
        geotiff_path = resolve_existing_path(str(geotiff_path_raw), config_dir, "GeoTIFF")
        geo_handler = GeoTiffHandler(str(geotiff_path))

    loaded: list[dict] = []
    used_names: set[str] = set()
    for idx, traj_cfg in enumerate(traj_cfgs):
        if not isinstance(traj_cfg, dict):
            raise ValueError(f"trajectories[{idx}] must be a mapping")

        display_name = str(traj_cfg.get("name", f"trajectory_{idx + 1}"))
        unique_name = uniquify_name(display_name, used_names)
        used_names.add(unique_name)

        points, traj_path = load_xy_trajectory(traj_cfg, config_dir=config_dir, geo_handler=geo_handler)
        if len(points) == 0:
            logger.warning("Skipping %s: empty trajectory at %s", unique_name, traj_path)
            continue

        loaded.append(
            {
                "name": unique_name,
                "display_name": display_name,
                "path": traj_path,
                "points": points,
            }
        )
        logger.info("Loaded %-20s points=%d file=%s", unique_name, len(points), traj_path)

    if not loaded:
        raise ValueError("No trajectories loaded successfully")
    return loaded


def evaluate_hausdorff(
    loaded_trajectories: list[dict],
    reference_name: str,
    resample_step_m: float,
    clip_to_overlap: bool,
) -> list[dict]:
    reference_matches = [row for row in loaded_trajectories if row["display_name"] == reference_name]
    if len(reference_matches) == 0:
        available = ", ".join(row["display_name"] for row in loaded_trajectories)
        raise ValueError(f"Reference trajectory '{reference_name}' not found. Available: {available}")
    if len(reference_matches) > 1:
        raise ValueError(f"Reference trajectory name '{reference_name}' is ambiguous (appears multiple times)")

    reference = reference_matches[0]
    reference_resampled = resample_polyline(reference["points"], resample_step_m)
    if len(reference_resampled) < 2:
        raise ValueError(
            f"Reference trajectory '{reference['name']}' has <2 points after resampling; cannot evaluate Hausdorff"
        )

    rows: list[dict] = []
    for target in loaded_trajectories:
        if target is reference:
            continue

        target_resampled = resample_polyline(target["points"], resample_step_m)
        if len(target_resampled) < 2:
            logger.warning("Skipping %s: <2 points after resampling", target["name"])
            continue

        ref_eval = reference_resampled
        target_eval = target_resampled
        if clip_to_overlap:
            bbox = overlap_bbox(ref_eval, target_eval)
            if bbox is None:
                logger.warning("Skipping %s: no bounding-box overlap with reference", target["name"])
                continue
            ref_eval = clip_points_to_bbox(ref_eval, bbox)
            target_eval = clip_points_to_bbox(target_eval, bbox)
            if len(ref_eval) < 2 or len(target_eval) < 2:
                logger.warning("Skipping %s: insufficient overlap points after clipping", target["name"])
                continue

        d_ref_to_target = directed_hausdorff(ref_eval, target_eval)
        d_target_to_ref = directed_hausdorff(target_eval, ref_eval)
        hausdorff = max(d_ref_to_target, d_target_to_ref)

        rows.append(
            {
                "reference_name": reference["name"],
                "target_name": target["name"],
                "num_ref_points": int(len(ref_eval)),
                "num_target_points": int(len(target_eval)),
                "d_ref_to_target_m": float(d_ref_to_target),
                "d_target_to_ref_m": float(d_target_to_ref),
                "hausdorff_m": float(hausdorff),
            }
        )

        logger.info(
            "Hausdorff %-20s H=%.3f m (ref->target=%.3f, target->ref=%.3f)",
            target["name"],
            hausdorff,
            d_ref_to_target,
            d_target_to_ref,
        )

    if not rows:
        raise ValueError("No target trajectories evaluated. Check reference name and overlap/filter settings.")
    return rows


def write_results(rows: list[dict], out_dir: Path, experiment_name: str, config_path: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hausdorff_results.csv"
    summary_path = out_dir / "hausdorff_summary.txt"

    ordered_rows = sorted(rows, key=lambda r: r["hausdorff_m"])
    headers = [
        "experiment_name",
        "reference_name",
        "target_name",
        "num_ref_points",
        "num_target_points",
        "d_ref_to_target_m",
        "d_target_to_ref_m",
        "hausdorff_m",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "experiment_name": experiment_name,
                    **row,
                    "d_ref_to_target_m": f"{row['d_ref_to_target_m']:.6f}",
                    "d_target_to_ref_m": f"{row['d_target_to_ref_m']:.6f}",
                    "hausdorff_m": f"{row['hausdorff_m']:.6f}",
                }
            )

    hausdorff_vals = np.array([row["hausdorff_m"] for row in ordered_rows], dtype=float)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Reference: {args.reference_name}\n")
        f.write(f"Resample step (m): {args.resample_step_m}\n")
        f.write(f"Clip to overlap: {bool(args.clip_to_overlap)}\n")
        f.write(f"Evaluated trajectories: {len(ordered_rows)}\n\n")
        f.write("Per-target symmetric Hausdorff (meters):\n")
        for row in ordered_rows:
            f.write(
                f"  {row['target_name']}: {row['hausdorff_m']:.3f} "
                f"(ref->target={row['d_ref_to_target_m']:.3f}, target->ref={row['d_target_to_ref_m']:.3f})\n"
            )
        f.write("\n")
        f.write(f"Best (min): {float(np.min(hausdorff_vals)):.3f} m\n")
        f.write(f"Worst (max): {float(np.max(hausdorff_vals)):.3f} m\n")
        f.write(f"Mean: {float(np.mean(hausdorff_vals)):.3f} m\n")
        f.write(f"Median: {float(np.median(hausdorff_vals)):.3f} m\n")

    logger.info("Saved CSV: %s", csv_path)
    logger.info("Saved summary: %s", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Hausdorff metrics for trajectory YAML configs.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to trajectory YAML (e.g., paper_scripts/config/trajectories/sara_human_pickle_north.yaml)",
    )
    parser.add_argument(
        "--reference-name",
        default="Human Drawn",
        help="Trajectory name used as reference (matched against `name` in YAML).",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_scripts/output/hausdorff",
        help="Output root directory; final path is output_dir/<experiment_name>/",
    )
    parser.add_argument(
        "--resample-step-m",
        type=float,
        default=1.0,
        help="Arc-length resampling interval in meters before Hausdorff.",
    )
    parser.add_argument(
        "--clip-to-overlap",
        action="store_true",
        help="Clip both paths to the intersection of their axis-aligned bounding boxes before evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    experiment_name = config.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError(f"Config is missing required non-empty key `experiment_name`: {config_path}")
    experiment_name = experiment_name.strip()

    config_dir = config_path.parent
    loaded = load_all_trajectories(config, config_dir=config_dir)
    rows = evaluate_hausdorff(
        loaded_trajectories=loaded,
        reference_name=args.reference_name,
        resample_step_m=args.resample_step_m,
        clip_to_overlap=args.clip_to_overlap,
    )

    out_root = Path(args.output_dir).expanduser()
    out_dir = out_root / sanitize_filename(experiment_name)
    write_results(rows, out_dir=out_dir, experiment_name=experiment_name, config_path=config_path, args=args)


if __name__ == "__main__":
    main()
