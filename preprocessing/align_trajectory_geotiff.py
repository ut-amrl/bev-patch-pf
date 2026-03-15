import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from geotiff.handler import GeoTiffHandler

WINDOW_NAME = "Align Trajectory"

DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 900

MIN_ZOOM = 0.2
MAX_ZOOM = 25.0
ZOOM_STEP = 1.25
PAN_FRACTION = 0.12

MOVE_STEP_BASE_M = 0.2
ROT_STEP_BASE_RAD = np.deg2rad(0.2)

STEP_SCALE_MIN = 0.0625
STEP_SCALE_MAX = 16.0
STEP_SCALE_UP = 2.0
STEP_SCALE_DOWN = 0.5

ROTATE_DRAG_RAD_PER_PX = np.deg2rad(0.15)
PICK_RADIUS_PX = 14.0

HUD_LINE_HEIGHT = 22
HUD_TOP = 12
HUD_BOTTOM = 10
HUD_MAX_LINES = 8
STATUS_BAR_HEIGHT = 40

START_ARROW_LEN_PX = 56.0
START_LABEL_OFFSET_PX = 10

DEFAULT_FLAT_MODE = "auto"
FLAT_Z_RANGE_THRESH = 1.0
FLAT_Z_STD_THRESH = 0.35
FLAT_PLANE_RMS_THRESH = 0.25

LEFT_KEYS = {81, 2424832, 65361}
RIGHT_KEYS = {83, 2555904, 65363}
UP_KEYS = {82, 2490368, 65362}
DOWN_KEYS = {84, 2621440, 65364}


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _to_display_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]

    rgb = np.repeat(image, 3, axis=2) if image.shape[2] == 1 else image[..., :3]

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32, copy=False)
        sample = rgb[:: max(rgb.shape[0] // 2048, 1), :: max(rgb.shape[1] // 2048, 1)]
        lo = np.percentile(sample, 1, axis=(0, 1), keepdims=True)
        hi = np.percentile(sample, 99, axis=(0, 1), keepdims=True)
        scale = np.where(hi > lo, 255.0 / (hi - lo), 1.0)
        rgb = np.clip((rgb - lo) * scale, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb[..., ::-1])


def _yaw_from_xy(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(xy) == 1:
        return np.zeros((1,), dtype=np.float64)

    delta = np.diff(xy.astype(np.float64, copy=False), axis=0)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    return np.concatenate([yaw[:1], yaw]).astype(np.float64, copy=False)


def _path_length(points: np.ndarray | None) -> float:
    if points is None:
        return float("nan")
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points.astype(np.float64, copy=False), axis=0), axis=1).sum())


def _flatten_xyz_to_plane(
    xyz: np.ndarray,
    quats: np.ndarray | None = None,
    ref_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    rel = xyz.astype(np.float64, copy=False) - xyz[0].astype(np.float64, copy=False)
    _, _, vh = np.linalg.svd(rel, full_matrices=False)

    axis_u = vh[0] / max(np.linalg.norm(vh[0]), 1e-12)
    axis_v = vh[1] / max(np.linalg.norm(vh[1]), 1e-12)

    normal = np.cross(axis_u, axis_v)
    normal /= max(np.linalg.norm(normal), 1e-12)

    axis_v = np.cross(normal, axis_u)
    axis_v /= max(np.linalg.norm(axis_v), 1e-12)

    if ref_xy is not None and len(ref_xy) == len(rel) and len(rel) >= 2:
        ref_rel = ref_xy.astype(np.float64, copy=False) - ref_xy[0].astype(np.float64, copy=False)
        flat_xy = np.column_stack([rel @ axis_u, rel @ axis_v]).astype(np.float64, copy=False)
        fit, *_ = np.linalg.lstsq(flat_xy, ref_rel, rcond=None)

        if np.linalg.det(fit) < 0.0:
            axis_v = -axis_v

        flat_xy = np.column_stack([rel @ axis_u, rel @ axis_v]).astype(np.float64, copy=False)
        if float(np.sum(flat_xy * ref_rel)) < 0.0:
            axis_u = -axis_u
            axis_v = -axis_v

    normal = np.cross(axis_u, axis_v)
    normal /= max(np.linalg.norm(normal), 1e-12)

    flat_xy = np.column_stack([rel @ axis_u, rel @ axis_v]).astype(np.float64, copy=False)
    yaw = _yaw_from_xy(flat_xy)

    if quats is None:
        return flat_xy, yaw

    forward = R.from_quat(quats).as_matrix()[:, :, 0].astype(np.float64, copy=False)
    forward -= np.sum(forward * normal[None, :], axis=1, keepdims=True) * normal[None, :]
    norm = np.linalg.norm(forward, axis=1)
    valid = norm > 1e-9
    forward[valid] /= norm[valid, None]
    if np.any(valid):
        yaw[valid] = np.arctan2(forward[valid] @ axis_v, forward[valid] @ axis_u)

    return flat_xy, yaw


def _load_matrix_from_struct(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        mat = data.astype(np.float64, copy=False)
    elif isinstance(data, list):
        mat = np.asarray(data, dtype=np.float64)
    elif isinstance(data, dict):
        if {"rows", "cols", "data"}.issubset(data):
            rows = int(data["rows"])
            cols = int(data["cols"])
            mat = np.asarray(data["data"], dtype=np.float64).reshape(rows, cols)
        elif "extrinsic_matrix" in data:
            return _load_matrix_from_struct(data["extrinsic_matrix"])
        elif "transform" in data:
            return _load_matrix_from_struct(data["transform"])
        elif "matrix" in data:
            return _load_matrix_from_struct(data["matrix"])
        elif "T" in data:
            return _load_matrix_from_struct(data["T"])
        else:
            raise ValueError("Could not find a 4x4 matrix in transform file.")
    else:
        raise ValueError("Unsupported transform file format.")

    if mat.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got shape {mat.shape}.")
    return mat


def load_transform_matrix(path: str | None) -> np.ndarray | None:
    if path is None:
        return None

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npy":
        mat = np.load(p)
        return _load_matrix_from_struct(mat)

    text = p.read_text()

    if suffix in {".yaml", ".yml"}:
        return _load_matrix_from_struct(yaml.safe_load(text))
    if suffix == ".json":
        return _load_matrix_from_struct(json.loads(text))
    if suffix in {".csv", ".txt"}:
        try:
            mat = np.loadtxt(p, delimiter="," if suffix == ".csv" else None)
        except Exception:
            mat = np.loadtxt(p)
        return _load_matrix_from_struct(mat)

    # fallback: try yaml/json/text matrix
    try:
        return _load_matrix_from_struct(yaml.safe_load(text))
    except Exception:
        pass

    try:
        return _load_matrix_from_struct(json.loads(text))
    except Exception:
        pass

    try:
        return _load_matrix_from_struct(np.loadtxt(p))
    except Exception as exc:
        raise ValueError(f"Failed to parse transform matrix from {p}") from exc


def build_pose_components(df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    has_xyz = {"x", "y", "z"}.issubset(df.columns)
    has_xy = {"x", "y"}.issubset(df.columns)
    has_quat = {"qx", "qy", "qz", "qw"}.issubset(df.columns)
    has_angle = "angle" in df.columns

    xyz = None
    quats = None

    if has_xyz:
        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    elif has_xy and has_angle:
        xy = df[["x", "y"]].to_numpy(dtype=np.float64)
        xyz = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])
    elif has_xy and has_quat:
        xy = df[["x", "y"]].to_numpy(dtype=np.float64)
        xyz = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])

    if has_quat:
        quats = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    elif has_angle:
        yaw = df["angle"].to_numpy(dtype=np.float64)
        quats = R.from_euler("z", yaw).as_quat().astype(np.float64, copy=False)

    return xyz, quats


def apply_frame_transform(xyz: np.ndarray, quats: np.ndarray, T_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] != quats.shape[0]:
        raise ValueError("xyz and quats must have the same length.")

    n = len(xyz)
    mats = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n, axis=0)
    mats[:, :3, :3] = R.from_quat(quats).as_matrix()
    mats[:, :3, 3] = xyz

    T_inv = np.linalg.inv(T_frame)
    mats_out = T_frame[None, :, :] @ mats @ T_inv[None, :, :]

    xyz_out = mats_out[:, :3, 3].astype(np.float64, copy=False)
    quat_out = R.from_matrix(mats_out[:, :3, :3]).as_quat().astype(np.float64, copy=False)
    return xyz_out, quat_out


def load_trajectory(
    traj_path: str,
    flat_mode: str = DEFAULT_FLAT_MODE,
    frame_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(traj_path)

    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("Trajectory file must contain x and y columns.")
    if flat_mode not in {"off", "on", "auto"}:
        raise ValueError(f"Invalid flat_mode={flat_mode!r}; expected off, on, or auto.")

    timestamp = (
        df["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in df.columns
        else np.arange(len(df), dtype=np.float64)
    )

    ref_xy = df[["x", "y"]].to_numpy(dtype=np.float64)
    fallback_yaw = df["angle"].to_numpy(dtype=np.float64) if "angle" in df.columns else None

    xyz, quats = build_pose_components(df)

    if frame_transform is not None:
        if xyz is None or quats is None:
            raise ValueError(
                "frame_transform requires pose information. "
                "Provide x,y,z with quaternion, or x,y,z with angle, or x,y with angle."
            )
        xyz, quats = apply_frame_transform(xyz, quats, frame_transform)
        ref_xy = xyz[:, :2]

    z_std = z_range = plane_rms = float("nan")
    if xyz is not None:
        z = xyz[:, 2].astype(np.float64, copy=False)
        z_std = float(np.std(z))
        z_range = float(np.max(z) - np.min(z))

        if len(xyz) >= 3:
            rel = xyz.astype(np.float64, copy=False) - xyz[0].astype(np.float64, copy=False)
            _, _, vh = np.linalg.svd(rel, full_matrices=False)
            normal = vh[-1] / max(np.linalg.norm(vh[-1]), 1e-12)
            plane_rms = float(np.sqrt(np.mean((rel @ normal) ** 2)))
        else:
            plane_rms = 0.0

    if flat_mode == "off":
        flat_enabled = False
        flat_reason = "flat_mode=off"
    elif flat_mode == "on":
        flat_enabled = xyz is not None
        flat_reason = "flat_mode=on" if flat_enabled else "flat_mode=on but z missing"
    elif xyz is None:
        flat_enabled = False
        flat_reason = "flat_mode=auto but z missing"
    else:
        triggers = []
        if z_range > FLAT_Z_RANGE_THRESH:
            triggers.append(f"z_range={z_range:.3f}>{FLAT_Z_RANGE_THRESH:.3f}")
        if z_std > FLAT_Z_STD_THRESH:
            triggers.append(f"z_std={z_std:.3f}>{FLAT_Z_STD_THRESH:.3f}")
        if plane_rms > FLAT_PLANE_RMS_THRESH:
            triggers.append(f"plane_rms={plane_rms:.3f}>{FLAT_PLANE_RMS_THRESH:.3f}")
        flat_enabled = bool(triggers)
        flat_reason = ", ".join(triggers) if triggers else "all flatness metrics below thresholds"

    if flat_enabled and xyz is not None:
        xy, yaw = _flatten_xyz_to_plane(xyz, quats=quats, ref_xy=ref_xy)
    else:
        xy = ref_xy
        if quats is not None:
            rot = R.from_quat(quats).as_matrix()
            yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
        elif fallback_yaw is not None:
            yaw = fallback_yaw.astype(np.float64, copy=False)
        else:
            yaw = _yaw_from_xy(ref_xy)

    info = {
        "flat_enabled": flat_enabled,
        "flat_reason": flat_reason,
        "has_z": xyz is not None,
        "frame_transform_applied": frame_transform is not None,
        "len_xy": _path_length(ref_xy),
        "len_active": _path_length(xy),
        "z_std": z_std,
        "z_range": z_range,
        "plane_rms": plane_rms,
    }

    return timestamp, np.column_stack([xy[:, 0], xy[:, 1], yaw]), info


@dataclass
class ViewBox:
    u0: float = 0.0
    v0: float = 0.0
    scale: float = 1.0
    draw_x0: float = 0.0
    draw_y0: float = 0.0
    draw_w: float = 1.0
    draw_h: float = 1.0
    ready: bool = False


@dataclass
class DragState:
    mode: str | None = None
    start_xy: np.ndarray | None = None
    start_uv: np.ndarray | None = None
    start_center: np.ndarray | None = None
    start_offset_x: float | None = None
    start_offset_y: float | None = None
    start_offset_r: float | None = None
    anchor_base_xy: np.ndarray | None = None
    anchor_world_xy: np.ndarray | None = None


@dataclass
class Aligner:
    geo_handler: GeoTiffHandler
    traj_path: str
    timestamp: np.ndarray
    traj_xyr: np.ndarray
    traj_info: dict
    flat_mode: str

    map_bgr: np.ndarray = field(init=False)
    img_h: int = field(init=False)
    img_w: int = field(init=False)

    window_w: int = field(init=False)
    map_h: int = field(init=False)
    info_h: int = field(init=False)
    total_h: int = field(init=False)

    base_x: np.ndarray = field(init=False)
    base_y: np.ndarray = field(init=False)
    base_r: np.ndarray = field(init=False)
    num_points: int = field(init=False)

    full_x: np.ndarray = field(init=False)
    full_y: np.ndarray = field(init=False)
    full_r: np.ndarray = field(init=False)
    full_uv: np.ndarray = field(init=False)

    init_offset_x: float = field(init=False)
    init_offset_y: float = field(init=False)
    init_offset_r: float = 0.0

    offset_x: float = field(init=False)
    offset_y: float = field(init=False)
    offset_r: float = 0.0
    zoom: float = 1.0
    step_scale: float = 1.0
    view_center_uv: np.ndarray = field(init=False)

    pose_dirty: bool = True
    frame_dirty: bool = True
    cached_frame: np.ndarray | None = None

    status_text: str = "Press P to save, Q to quit"
    status_until: float = 0.0

    view: ViewBox = field(default_factory=ViewBox)
    drag: DragState = field(default_factory=DragState)

    def __post_init__(self) -> None:
        self.map_bgr = _to_display_bgr(self.geo_handler.image)
        self.img_h, self.img_w = self.map_bgr.shape[:2]

        self.window_w = int(max(320, min(DEFAULT_WINDOW_WIDTH, self.img_w)))
        self.info_h = HUD_TOP + HUD_BOTTOM + HUD_LINE_HEIGHT * HUD_MAX_LINES
        self.total_h = DEFAULT_WINDOW_HEIGHT
        self.map_h = max(240, self.total_h - self.info_h - STATUS_BAR_HEIGHT)

        self.base_x = self.traj_xyr[:, 0].astype(np.float64, copy=False)
        self.base_y = self.traj_xyr[:, 1].astype(np.float64, copy=False)
        self.base_r = self.traj_xyr[:, 2].astype(np.float64, copy=False)
        self.num_points = len(self.traj_xyr)

        self.full_x = np.empty_like(self.base_x)
        self.full_y = np.empty_like(self.base_y)
        self.full_r = np.empty_like(self.base_r)
        self.full_uv = np.empty((self.num_points, 2), dtype=np.float32)

        min_x, min_y, max_x, max_y = self.geo_handler.bounds
        self.init_offset_x = float((min_x + max_x) * 0.5)
        self.init_offset_y = float((min_y + max_y) * 0.5)

        self.offset_x = self.init_offset_x
        self.offset_y = self.init_offset_y
        self.view_center_uv = np.array([self.img_w * 0.5, self.img_h * 0.5], dtype=np.float64)

        self.refresh_pose()
        self.fit_view()

    @property
    def flat_state(self) -> str:
        return "enabled" if bool(self.traj_info["flat_enabled"]) else "disabled"

    @property
    def move_step_m(self) -> float:
        return MOVE_STEP_BASE_M * self.step_scale

    @property
    def rot_step_rad(self) -> float:
        return ROT_STEP_BASE_RAD * self.step_scale

    def now_s(self) -> float:
        return cv2.getTickCount() / cv2.getTickFrequency()

    def set_status(self, message: str, duration_s: float = 1.8) -> None:
        self.status_text = message
        self.status_until = self.now_s() + duration_s
        self.frame_dirty = True

    def refresh_pose(self) -> None:
        c, s = np.cos(self.offset_r), np.sin(self.offset_r)
        self.full_x[:] = c * self.base_x - s * self.base_y + self.offset_x
        self.full_y[:] = s * self.base_x + c * self.base_y + self.offset_y
        self.full_r[:] = self.base_r + self.offset_r
        self.full_uv[:] = self.geo_handler.coords_to_pixel(np.column_stack([self.full_x, self.full_y]))
        self.pose_dirty = False

    def fit_view(self) -> None:
        if self.pose_dirty:
            self.refresh_pose()

        if len(self.full_uv) == 0:
            self.view_center_uv[:] = (self.img_w * 0.5, self.img_h * 0.5)
            self.zoom = 1.0
        else:
            u_min, v_min = np.min(self.full_uv, axis=0)
            u_max, v_max = np.max(self.full_uv, axis=0)
            self.view_center_uv[:] = (
                np.clip((u_min + u_max) * 0.5, 0.0, self.img_w - 1.0),
                np.clip((v_min + v_max) * 0.5, 0.0, self.img_h - 1.0),
            )
            width = max(float(u_max - u_min) + 160.0, 1.0)
            height = max(float(v_max - v_min) + 160.0, 1.0)
            self.zoom = float(np.clip(min(self.window_w / width, self.map_h / height), MIN_ZOOM, MAX_ZOOM))

        self.frame_dirty = True

    def map_contains(self, x: int, y: int) -> bool:
        return self.view.ready and (
            self.view.draw_x0 <= x < self.view.draw_x0 + self.view.draw_w
            and self.view.draw_y0 <= y < self.view.draw_y0 + self.view.draw_h
        )

    def screen_to_uv(self, x: int, y: int) -> np.ndarray:
        x = float(np.clip(x - self.view.draw_x0, 0.0, max(self.view.draw_w - 1.0, 0.0)))
        y = float(np.clip(y - self.view.draw_y0, 0.0, max(self.view.draw_h - 1.0, 0.0)))
        scale = max(float(self.view.scale), 1e-12)
        return np.array([self.view.u0 + x / scale, self.view.v0 + y / scale], dtype=np.float64)

    def pick_traj_index(self, x: int, y: int, radius_px: float = PICK_RADIUS_PX) -> int:
        if self.pose_dirty:
            self.refresh_pose()

        if not self.map_contains(x, y):
            return -1

        scale = max(float(self.view.scale), 1e-12)
        sx = (self.full_uv[:, 0] - self.view.u0) * scale + self.view.draw_x0
        sy = (self.full_uv[:, 1] - self.view.v0) * scale + self.view.draw_y0

        mask = (sx >= x - radius_px) & (sx <= x + radius_px) & (sy >= y - radius_px) & (sy <= y + radius_px)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            return -1

        d2 = (sx[candidates] - float(x)) ** 2 + (sy[candidates] - float(y)) ** 2
        best = int(candidates[int(np.argmin(d2))])
        return best if float(np.min(d2)) <= radius_px * radius_px else -1

    def current_crop(self) -> tuple[int, int, int, int, float, int, int, int, int]:
        crop_w = min(self.img_w, max(64, int(round(self.window_w / self.zoom))))
        crop_h = min(self.img_h, max(64, int(round(self.map_h / self.zoom))))

        u0 = int(np.clip(round(self.view_center_uv[0] - crop_w * 0.5), 0, max(self.img_w - crop_w, 0)))
        v0 = int(np.clip(round(self.view_center_uv[1] - crop_h * 0.5), 0, max(self.img_h - crop_h, 0)))

        scale = min(self.window_w / crop_w, self.map_h / crop_h)
        draw_w = max(1, int(round(crop_w * scale)))
        draw_h = max(1, int(round(crop_h * scale)))
        draw_x0 = int(round((self.window_w - draw_w) * 0.5))
        draw_y0 = int(round((self.map_h - draw_h) * 0.5))

        return u0, v0, crop_w, crop_h, scale, draw_x0, draw_y0, draw_w, draw_h

    def draw_info_panel(self) -> np.ndarray:
        panel = np.full((self.info_h, self.window_w, 3), 26, dtype=np.uint8)
        cv2.line(panel, (0, 0), (self.window_w - 1, 0), (80, 80, 80), 1)

        lines = [
            f"Offset x={self.offset_x:.3f} m, y={self.offset_y:.3f} m, yaw={np.rad2deg(self.offset_r):.3f} deg",
            (
                f"Step scale={self.step_scale:.4g} | "
                f"move={self.move_step_m:.4f} m | "
                f"rot={np.rad2deg(self.rot_step_rad):.4f} deg"
            ),
            (
                f"Flat mode={self.flat_mode} ({self.flat_state}) | "
                f"frame_transform={'on' if self.traj_info['frame_transform_applied'] else 'off'}"
            ),
            f"len={self.traj_info['len_active']:.2f} m | ref={self.traj_info['len_xy']:.2f} m",
            "Move trajectory: WASD or arrows | Rotate: [ ] | Step: - / =",
            "Left drag map | Left drag near trajectory moves trajectory",
            "Right drag near trajectory rotates around clicked point",
            "Pan: IJKL | Zoom: wheel or Z/X | Fit: F | Reset: R | Save: P | Quit: Q/ESC",
        ]

        y = HUD_TOP + 18
        for line in lines[:HUD_MAX_LINES]:
            cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
            y += HUD_LINE_HEIGHT

        return panel

    def draw_status_bar(self) -> np.ndarray:
        bar = np.full((STATUS_BAR_HEIGHT, self.window_w, 3), 38, dtype=np.uint8)
        cv2.line(bar, (0, 0), (self.window_w - 1, 0), (70, 70, 70), 1)

        if self.now_s() < self.status_until:
            text = self.status_text
            color = (255, 255, 255)
        else:
            text = "Ready"
            color = (180, 180, 180)

        cv2.putText(
            bar,
            text,
            (18, STATUS_BAR_HEIGHT - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            1,
            cv2.LINE_AA,
        )
        return bar

    def draw_trajectory(self, canvas: np.ndarray, u0: int, v0: int, scale: float, draw_x0: int, draw_y0: int) -> None:
        def uv_to_screen(uv: np.ndarray) -> np.ndarray:
            out = np.empty_like(uv, dtype=np.float64)
            out[:, 0] = (uv[:, 0] - u0) * scale + draw_x0
            out[:, 1] = (uv[:, 1] - v0) * scale + draw_y0
            return out

        # Main polyline uses decimation for speed
        step = max(1, int(np.ceil(self.num_points / 12000)))
        draw_idx = np.arange(0, self.num_points, step, dtype=np.int64)
        if draw_idx[-1] != self.num_points - 1:
            draw_idx = np.append(draw_idx, self.num_points - 1)

        draw_uv = self.full_uv[draw_idx]
        pts = np.round(uv_to_screen(draw_uv)).astype(np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, (0, 0, 255), 1, cv2.LINE_AA)

        # Direction arrows use the full-resolution trajectory to avoid interpolation errors
        if self.num_points >= 2:
            arrow_spacing_px = 120.0
            arrow_len_px = 28.0

            full_pts = uv_to_screen(self.full_uv)
            seg = np.diff(full_pts, axis=0)
            seg_len = np.linalg.norm(seg, axis=1)

            valid = seg_len > 1e-6
            if np.any(valid):
                seg = seg[valid]
                seg_len = seg_len[valid]
                seg_start = full_pts[:-1][valid]
                seg_end = full_pts[1:][valid]

                cum = np.concatenate([[0.0], np.cumsum(seg_len)])
                total_len = float(cum[-1])

                if total_len > arrow_spacing_px:
                    for d in np.arange(arrow_spacing_px, total_len, arrow_spacing_px):
                        i = int(np.searchsorted(cum, d, side="right") - 1)
                        i = min(max(i, 0), len(seg_len) - 1)

                        local_d = d - cum[i]
                        t = local_d / seg_len[i]

                        mid = seg_start[i] + t * (seg_end[i] - seg_start[i])
                        direction = seg[i] / seg_len[i]

                        tail = mid - 0.5 * arrow_len_px * direction
                        head = mid + 0.5 * arrow_len_px * direction

                        tail_i = tuple(np.round(tail).astype(np.int32))
                        head_i = tuple(np.round(head).astype(np.int32))

                        cv2.arrowedLine(canvas, tail_i, head_i, (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.4)
                        cv2.arrowedLine(canvas, tail_i, head_i, (255, 200, 0), 1, cv2.LINE_AA, tipLength=0.4)

        start_u, start_v = self.full_uv[0]
        end_u, end_v = self.full_uv[-1]
        sx = int(round((start_u - u0) * scale + draw_x0))
        sy = int(round((start_v - v0) * scale + draw_y0))
        ex = int(round((end_u - u0) * scale + draw_x0))
        ey = int(round((end_v - v0) * scale + draw_y0))

        cv2.circle(canvas, (sx, sy), 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 4, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (ex, ey), 4, (0, 255, 255), -1, cv2.LINE_AA)

        dir_px = None
        if self.num_points >= 2:
            for i in range(1, self.num_points):
                dv = (self.full_uv[i] - self.full_uv[0]).astype(np.float64) * scale
                if float(np.hypot(dv[0], dv[1])) > 1e-8:
                    dir_px = dv
                    break

        if dir_px is None:
            return

        dir_norm = float(np.hypot(dir_px[0], dir_px[1]))
        tx = int(round(sx + dir_px[0] * START_ARROW_LEN_PX / dir_norm))
        ty = int(round(sy + dir_px[1] * START_ARROW_LEN_PX / dir_norm))

        cv2.arrowedLine(canvas, (sx, sy), (tx, ty), (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(canvas, (sx, sy), (tx, ty), (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.35)

        label_pos = (tx + START_LABEL_OFFSET_PX, ty - START_LABEL_OFFSET_PX)
        cv2.putText(canvas, "START", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, "START", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 1, cv2.LINE_AA)

    def draw_map(self) -> np.ndarray:
        u0, v0, crop_w, crop_h, scale, draw_x0, draw_y0, draw_w, draw_h = self.current_crop()

        crop = self.map_bgr[v0 : v0 + crop_h, u0 : u0 + crop_w]
        if draw_w != crop_w or draw_h != crop_h:
            crop = cv2.resize(crop, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.map_h, self.window_w, 3), 10, dtype=np.uint8)
        canvas[draw_y0 : draw_y0 + draw_h, draw_x0 : draw_x0 + draw_w] = crop

        self.view = ViewBox(
            u0=float(u0),
            v0=float(v0),
            scale=float(scale),
            draw_x0=float(draw_x0),
            draw_y0=float(draw_y0),
            draw_w=float(draw_w),
            draw_h=float(draw_h),
            ready=True,
        )

        self.draw_trajectory(canvas, u0, v0, scale, draw_x0, draw_y0)
        return canvas

    def draw(self) -> np.ndarray:
        frame = np.vstack((self.draw_map(), self.draw_info_panel(), self.draw_status_bar()))
        return np.ascontiguousarray(frame)

    def save(self) -> Path:
        if self.pose_dirty:
            self.refresh_pose()

        src = Path(self.traj_path)
        out_path = src.with_name(f"{src.stem}_aligned{src.suffix}" if src.suffix else f"{src.name}_aligned")

        pd.DataFrame(
            {
                "timestamp": np.char.mod("%.6f", self.timestamp),
                "x": np.char.mod("%.4f", self.full_x),
                "y": np.char.mod("%.4f", self.full_y),
                "angle": np.char.mod("%.4f", self.full_r),
            }
        ).to_csv(out_path, index=False)

        print(
            f"Aligned trajectory saved to {out_path} "
            f"(offset_x={self.offset_x:.4f}, offset_y={self.offset_y:.4f}, yaw_deg={np.rad2deg(self.offset_r):.6f})"
        )
        return out_path

    def start_drag(self, event: int, x: int, y: int) -> None:
        if not self.map_contains(x, y):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            traj_idx = self.pick_traj_index(x, y)
            if traj_idx >= 0:
                self.drag = DragState(
                    mode="traj",
                    start_uv=self.screen_to_uv(x, y),
                    start_offset_x=self.offset_x,
                    start_offset_y=self.offset_y,
                )
                self.set_status("Drag mode: trajectory", 0.8)
            else:
                self.drag = DragState(
                    mode="map",
                    start_xy=np.array([x, y]),
                    start_center=self.view_center_uv.copy(),
                )
                self.set_status("Drag mode: map", 0.8)

        elif event == cv2.EVENT_RBUTTONDOWN:
            traj_idx = self.pick_traj_index(x, y, radius_px=PICK_RADIUS_PX * 2.0)
            if traj_idx < 0:
                self.set_status("Hover trajectory first, then right-drag to rotate", 1.0)
                return

            self.drag = DragState(
                mode="rotate",
                start_xy=np.array([x, y]),
                start_offset_r=self.offset_r,
                anchor_base_xy=np.array([self.base_x[traj_idx], self.base_y[traj_idx]], dtype=np.float64),
                anchor_world_xy=np.array([self.full_x[traj_idx], self.full_y[traj_idx]], dtype=np.float64),
            )
            self.set_status("Drag mode: trajectory rotate", 0.8)

    def update_drag(self, x: int, y: int) -> None:
        if self.drag.mode == "traj":
            curr_uv = self.screen_to_uv(x, y)
            du, dv = curr_uv - self.drag.start_uv
            gt = self.geo_handler.geotransform
            self.offset_x = self.drag.start_offset_x + gt.a * du + gt.b * dv
            self.offset_y = self.drag.start_offset_y + gt.d * du + gt.e * dv
            self.pose_dirty = True
            self.frame_dirty = True

        elif self.drag.mode == "map":
            delta_px = np.array([x, y]) - self.drag.start_xy
            self.view_center_uv = self.drag.start_center - delta_px / max(float(self.view.scale), 1e-12)
            self.frame_dirty = True

        elif self.drag.mode == "rotate":
            self.offset_r = _wrap_angle(self.drag.start_offset_r + (x - self.drag.start_xy[0]) * ROTATE_DRAG_RAD_PER_PX)
            bx, by = self.drag.anchor_base_xy
            c, s = np.cos(self.offset_r), np.sin(self.offset_r)
            rotated_xy = np.array([bx * c - by * s, bx * s + by * c])
            self.offset_x, self.offset_y = self.drag.anchor_world_xy - rotated_xy
            self.pose_dirty = True
            self.frame_dirty = True

    def stop_drag(self) -> None:
        self.drag = DragState()

    def zoom_at(self, x: int, y: int, zoom_in: bool) -> None:
        if not self.map_contains(x, y):
            return

        self.zoom = float(np.clip(self.zoom * ZOOM_STEP if zoom_in else self.zoom / ZOOM_STEP, MIN_ZOOM, MAX_ZOOM))

        anchor_uv = self.screen_to_uv(x, y)
        _, _, crop_w, crop_h, scale, draw_x0, draw_y0, draw_w, draw_h = self.current_crop()

        x_in = float(np.clip(x - draw_x0, 0.0, draw_w - 1.0))
        y_in = float(np.clip(y - draw_y0, 0.0, draw_h - 1.0))
        u0 = anchor_uv[0] - x_in / scale
        v0 = anchor_uv[1] - y_in / scale
        self.view_center_uv = np.array([u0 + crop_w * 0.5, v0 + crop_h * 0.5], dtype=np.float64)
        self.frame_dirty = True

    def handle_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.start_drag(event, x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            self.update_drag(x, y)
            return

        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.stop_drag()
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = int(cv2.getMouseWheelDelta(flags))
            except Exception:
                delta = (flags >> 16) & 0xFFFF
                if delta & 0x8000:
                    delta -= 0x10000

            if delta != 0:
                self.zoom_at(x, y, zoom_in=(delta > 0))

    def pan_pixels(self, dx: float, dy: float) -> None:
        self.view_center_uv[0] += dx
        self.view_center_uv[1] += dy
        self.frame_dirty = True

    def pan_step(self) -> float:
        return max(int(round(min(self.window_w, self.map_h) * PAN_FRACTION / max(self.zoom, 1e-6))), 8)

    def handle_key(self, key: int) -> bool:
        if key in LEFT_KEYS or key in (ord("a"), ord("A")):
            self.offset_x -= self.move_step_m
            self.pose_dirty = True
        elif key in RIGHT_KEYS or key in (ord("d"), ord("D")):
            self.offset_x += self.move_step_m
            self.pose_dirty = True
        elif key in UP_KEYS or key in (ord("w"), ord("W")):
            self.offset_y += self.move_step_m
            self.pose_dirty = True
        elif key in DOWN_KEYS or key in (ord("s"), ord("S")):
            self.offset_y -= self.move_step_m
            self.pose_dirty = True
        elif key in (ord("["), ord("{")):
            self.offset_r = _wrap_angle(self.offset_r - self.rot_step_rad)
            self.pose_dirty = True
        elif key in (ord("]"), ord("}")):
            self.offset_r = _wrap_angle(self.offset_r + self.rot_step_rad)
            self.pose_dirty = True
        elif key in (ord("-"), ord("_")):
            self.step_scale = float(np.clip(self.step_scale * STEP_SCALE_DOWN, STEP_SCALE_MIN, STEP_SCALE_MAX))
            self.set_status(
                f"Step smaller: move={self.move_step_m:.4f} m, rot={np.rad2deg(self.rot_step_rad):.4f} deg",
                1.2,
            )
        elif key in (ord("="), ord("+")):
            self.step_scale = float(np.clip(self.step_scale * STEP_SCALE_UP, STEP_SCALE_MIN, STEP_SCALE_MAX))
            self.set_status(
                f"Step larger: move={self.move_step_m:.4f} m, rot={np.rad2deg(self.rot_step_rad):.4f} deg",
                1.2,
            )
        elif key in (ord("z"), ord("Z")):
            self.zoom = float(np.clip(self.zoom * ZOOM_STEP, MIN_ZOOM, MAX_ZOOM))
            self.frame_dirty = True
        elif key in (ord("x"), ord("X")):
            self.zoom = float(np.clip(self.zoom / ZOOM_STEP, MIN_ZOOM, MAX_ZOOM))
            self.frame_dirty = True
        elif key in (ord("i"), ord("I")):
            self.pan_pixels(0.0, -self.pan_step())
        elif key in (ord("k"), ord("K")):
            self.pan_pixels(0.0, self.pan_step())
        elif key in (ord("j"), ord("J")):
            self.pan_pixels(-self.pan_step(), 0.0)
        elif key in (ord("l"), ord("L")):
            self.pan_pixels(self.pan_step(), 0.0)
        elif key in (ord("f"), ord("F")):
            self.fit_view()
            self.set_status("View fit to trajectory")
        elif key in (ord("r"), ord("R")):
            self.offset_x = self.init_offset_x
            self.offset_y = self.init_offset_y
            self.offset_r = self.init_offset_r
            self.step_scale = 1.0
            self.pose_dirty = True
            self.fit_view()
            self.set_status("Alignment reset")
        elif key in (ord("p"), ord("P")):
            out_path = self.save()
            self.set_status(f"Saved: {out_path}", 2.2)
        elif key in (27, ord("q"), ord("Q")):
            return False

        return True

    def run(self) -> None:
        print(
            f"[info] flat_mode={self.flat_mode} {self.flat_state}: "
            f"{self.traj_info['flat_reason']} (has_z={self.traj_info['has_z']})"
        )
        print(
            "[diag] "
            f"len_xy={self.traj_info['len_xy']:.3f} "
            f"len_active={self.traj_info['len_active']:.3f} "
            f"z_std={self.traj_info['z_std']:.3f} "
            f"z_range={self.traj_info['z_range']:.3f} "
            f"plane_rms={self.traj_info['plane_rms']:.3f}"
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, lambda e, x, y, f, p: self.handle_mouse(e, x, y, f))

        while True:
            if self.pose_dirty:
                self.refresh_pose()
                self.frame_dirty = True

            if self.frame_dirty or self.cached_frame is None:
                self.cached_frame = self.draw()
                cv2.imshow(WINDOW_NAME, self.cached_frame)
                self.frame_dirty = False

            key = cv2.waitKeyEx(15)
            if key >= 0 and not self.handle_key(key):
                break

        cv2.destroyWindow(WINDOW_NAME)


def align_trajectory(
    geotiff_path: str,
    traj_path: str,
    flat_mode: str = DEFAULT_FLAT_MODE,
    frame_transform_path: str | None = None,
) -> None:
    geo_handler = GeoTiffHandler(geotiff_path)
    frame_transform = load_transform_matrix(frame_transform_path)
    timestamp, traj_xyr, traj_info = load_trajectory(
        traj_path,
        flat_mode=flat_mode,
        frame_transform=frame_transform,
    )

    if len(traj_xyr) == 0:
        raise ValueError("Trajectory CSV is empty.")

    Aligner(
        geo_handler=geo_handler,
        traj_path=traj_path,
        timestamp=timestamp,
        traj_xyr=traj_xyr,
        traj_info=traj_info,
        flat_mode=flat_mode,
    ).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align trajectory with a GeoTIFF map.")
    parser.add_argument("--geotiff", type=str, required=True, help="Path to GeoTIFF file")
    parser.add_argument("--traj", type=str, required=True, help="Path to trajectory CSV file")
    parser.add_argument(
        "--flat-mode",
        type=str,
        choices=["off", "on", "auto"],
        default=DEFAULT_FLAT_MODE,
        help="Use the 3D trajectory plane instead of raw x/y when z is available.",
    )
    parser.add_argument(
        "--frame-transform",
        type=str,
        default=None,
        help="Optional 4x4 transform T used as pose' = T @ pose @ inv(T). Supports yaml/json/txt/csv/npy.",
    )
    args = parser.parse_args()

    align_trajectory(args.geotiff, args.traj, flat_mode=args.flat_mode, frame_transform_path=args.frame_transform)
