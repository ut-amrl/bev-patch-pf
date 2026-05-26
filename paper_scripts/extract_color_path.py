#!/usr/bin/env python3
"""Extract ordered UTM sequence from a pure-color human-drawn path in a GeoTIFF."""

from __future__ import annotations

import argparse
import csv
import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from geotiff.handler import GeoTiffHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PURE_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}

N8_OFFSETS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]


def exact_color_mask(rgb: np.ndarray, color_name: str) -> np.ndarray:
    target_r, target_g, target_b = PURE_COLORS[color_name]
    return (rgb[..., 0] == target_r) & (rgb[..., 1] == target_g) & (rgb[..., 2] == target_b)


def keep_largest_component(mask: np.ndarray, min_pixels: int = 1) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask, dtype=bool)

    best_label = -1
    best_area = -1
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_pixels and area > best_area:
            best_area = area
            best_label = label

    if best_label < 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == best_label


def farthest_hull_pair(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return farthest hull points on mask contour as (row, col)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in mask")

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=True).reshape(-1, 2)  # (x, y)
    if len(hull) == 0:
        raise ValueError("Convex hull is empty")
    if len(hull) == 1:
        p = (int(hull[0, 1]), int(hull[0, 0]))
        return p, p

    diff = hull[:, None, :] - hull[None, :, :]
    dist_sq = np.sum(diff.astype(np.float64) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)
    p0 = (int(hull[i, 1]), int(hull[i, 0]))
    p1 = (int(hull[j, 1]), int(hull[j, 0]))
    return p0, p1


def nearest_true_pixel(mask: np.ndarray, pt: tuple[int, int]) -> tuple[int, int]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("Cannot snap point on empty mask")
    d = coords - np.array([[pt[0], pt[1]]], dtype=coords.dtype)
    idx = int(np.argmin((d[:, 0].astype(np.float64) ** 2) + (d[:, 1].astype(np.float64) ** 2)))
    return int(coords[idx, 0]), int(coords[idx, 1])


def ascend_to_center(dist_map: np.ndarray, mask: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    """Move a mask pixel to a local distance-transform maximum (stroke center)."""
    cur = start
    max_steps = 128
    eps = 1e-9
    for _ in range(max_steps):
        r, c = cur
        best = cur
        best_val = float(dist_map[r, c])
        for rr, cc in iter_neighbors(mask, r, c):
            v = float(dist_map[rr, cc])
            if (v > best_val + eps) or (abs(v - best_val) <= eps and (rr, cc) < best):
                best = (rr, cc)
                best_val = v
        if best == cur:
            break
        cur = best
    return cur


def iter_neighbors(mask: np.ndarray, r: int, c: int):
    h, w = mask.shape
    for dr, dc in N8_OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            yield rr, cc


def bfs_tree(mask: np.ndarray, start: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = np.full(mask.shape, -1, dtype=np.int32)
    prev_r = np.full(mask.shape, -1, dtype=np.int32)
    prev_c = np.full(mask.shape, -1, dtype=np.int32)
    dq: deque[tuple[int, int]] = deque([start])
    dist[start] = 0

    while dq:
        r, c = dq.popleft()
        for rr, cc in iter_neighbors(mask, r, c):
            if dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                prev_r[rr, cc] = r
                prev_c[rr, cc] = c
                dq.append((rr, cc))
    return dist, prev_r, prev_c


def reconstruct_path(
    start: tuple[int, int], end: tuple[int, int], prev_r: np.ndarray, prev_c: np.ndarray
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur = end
    while True:
        path.append(cur)
        if cur == start:
            break
        cr, cc = cur
        pr = int(prev_r[cr, cc])
        pc = int(prev_c[cr, cc])
        if pr < 0 or pc < 0:
            raise ValueError("Broken BFS tree during path reconstruction")
        cur = (pr, pc)
    path.reverse()
    return path


def ordered_path_from_start(mask: np.ndarray, start: tuple[int, int], min_branch_len_px: int) -> list[tuple[int, int]]:
    """Choose farthest reachable endpoint from start and return the BFS trunk path."""
    dist, prev_r, prev_c = bfs_tree(mask, start)
    reachable = np.argwhere(dist >= 0)
    if len(reachable) == 0:
        raise ValueError("No reachable pixels from selected start point")

    # Select farthest point from chosen start: shorter branches are dropped naturally.
    end = tuple(int(v) for v in reachable[np.argmax(dist[reachable[:, 0], reachable[:, 1]])])
    if int(dist[end]) < min_branch_len_px:
        raise ValueError(
            f"Extracted branch is shorter than --min-branch-len-px ({int(dist[end])} < {min_branch_len_px})"
        )
    if start == end:
        raise ValueError("Could not find distinct start/end points on path")

    return reconstruct_path(start, end, prev_r, prev_c)


def recenter_ordered_path(path_pixels: list[tuple[int, int]], mask: np.ndarray) -> list[tuple[int, int]]:
    """Project ordered trunk pixels to stroke center using distance-transform ascent."""
    if not path_pixels:
        return path_pixels

    dist = cv2.distanceTransform(mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
    centered = [ascend_to_center(dist, mask, p) for p in path_pixels]

    deduped: list[tuple[int, int]] = []
    for p in centered:
        if not deduped or p != deduped[-1]:
            deduped.append(p)

    return deduped


def resample_polyline(points_xy: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(points_xy) == 0:
        return points_xy
    if spacing_m < 0:
        raise ValueError("--spacing-m must be >= 0")
    if spacing_m == 0:
        return points_xy

    seg_len = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(s[-1])
    if total <= 1e-9:
        return points_xy[:1]

    samples = np.arange(0.0, total, spacing_m, dtype=np.float64)
    if len(samples) == 0 or samples[-1] < total:
        samples = np.concatenate((samples, [total]))

    x = np.interp(samples, s, points_xy[:, 0])
    y = np.interp(samples, s, points_xy[:, 1])
    return np.stack([x, y], axis=1)


def save_sequence_csv(path: Path, geo_handler: GeoTiffHandler, utm_xy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utm_easting", "utm_northing"])
        for xy in utm_xy:
            writer.writerow([f"{xy[0]:.3f}", f"{xy[1]:.3f}"])


def write_debug_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def write_debug_overlay(path: Path, rgb: np.ndarray, ordered_pixels: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = rgb.copy()
    for r, c in ordered_pixels:
        vis[r, c] = np.array([255, 0, 0], dtype=np.uint8)  # red path
    if ordered_pixels:
        sr, sc = ordered_pixels[0]
        er, ec = ordered_pixels[-1]
        cv2.circle(vis, (sc, sr), 5, (0, 255, 255), -1)  # start: yellow
        cv2.circle(vis, (ec, er), 5, (255, 0, 255), -1)  # end: magenta
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def default_output(input_path: Path, color: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{color}_sequence.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ordered UTM sequence from a pure-color human-drawn path in a GeoTIFF."
    )
    parser.add_argument("input", help="Path to the input GeoTIFF")
    parser.add_argument("--color", choices=["red", "green", "blue"], default="green", help="Target pure color")
    parser.add_argument("--output", default=None, help="Output sequence CSV path")
    parser.add_argument("--spacing-m", type=float, default=0.2, help="Resampling spacing in meters (0 disables)")
    parser.add_argument("--reverse", action="store_true", help="Reverse final sequence order")
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=50,
        help="Minimum pixel count for retained connected component",
    )
    parser.add_argument(
        "--min-branch-len-px",
        type=int,
        default=20,
        help="Minimum BFS trunk length in pixels",
    )
    parser.add_argument("--debug-mask", action="store_true", help="Write largest-component mask image")
    parser.add_argument("--debug-skeleton", action="store_true", help="Write extracted trunk image")
    parser.add_argument("--debug-ordered", action="store_true", help="Write ordered path overlay image")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    geo = GeoTiffHandler(str(input_path))
    if geo.image.ndim != 3 or geo.image.shape[2] < 3:
        raise ValueError(f"Expected RGB GeoTIFF with >=3 channels: {input_path}")

    rgb = geo.image[..., :3].astype(np.uint8)
    mask = exact_color_mask(rgb, args.color)
    n_color = int(np.count_nonzero(mask))
    if n_color == 0:
        raise ValueError(f"No pure {args.color} pixels found in {input_path}")
    logger.info("Found %d pure %s pixels", n_color, args.color)

    main_mask = keep_largest_component(mask, min_pixels=int(args.min_component_pixels))
    if np.count_nonzero(main_mask) == 0:
        raise ValueError("No valid connected component remained after filtering")

    p0, p1 = farthest_hull_pair(main_mask)
    start_hint = p0 if p0 <= p1 else p1
    start = nearest_true_pixel(main_mask, start_hint)
    dist_map = cv2.distanceTransform(main_mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
    start = ascend_to_center(dist_map, main_mask, start)

    ordered_pixels = ordered_path_from_start(main_mask, start, min_branch_len_px=int(args.min_branch_len_px))
    ordered_pixels = recenter_ordered_path(ordered_pixels, main_mask)
    if len(ordered_pixels) < 2:
        raise ValueError("Centered path has fewer than 2 unique points")
    uv = np.array([[c, r] for r, c in ordered_pixels], dtype=np.float64)
    utm_xy = geo.pixel_to_coords(uv)
    utm_seq = resample_polyline(utm_xy, float(args.spacing_m))

    if args.reverse:
        ordered_pixels = list(reversed(ordered_pixels))
        utm_seq = utm_seq[::-1]

    output_path = Path(args.output) if args.output else default_output(input_path, args.color)
    save_sequence_csv(output_path, geo, utm_seq)
    logger.info("Saved ordered sequence: %s (%d points)", output_path, len(utm_seq))

    if args.debug_mask:
        dbg = output_path.with_name(f"{output_path.stem}_mask.png")
        write_debug_mask(dbg, main_mask)
        logger.info("Saved debug mask: %s", dbg)
    if args.debug_skeleton:
        trunk = np.zeros_like(main_mask, dtype=bool)
        for r, c in ordered_pixels:
            trunk[r, c] = True
        dbg = output_path.with_name(f"{output_path.stem}_trunk.png")
        write_debug_mask(dbg, trunk)
        logger.info("Saved debug trunk: %s", dbg)
    if args.debug_ordered:
        dbg = output_path.with_name(f"{output_path.stem}_ordered.png")
        write_debug_overlay(dbg, rgb, ordered_pixels)
        logger.info("Saved debug ordered overlay: %s", dbg)


if __name__ == "__main__":
    main()
