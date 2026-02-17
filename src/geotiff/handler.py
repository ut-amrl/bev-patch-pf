from __future__ import annotations

import cv2
import numpy as np
import rasterio
from pyproj import CRS, Transformer


class GeoTiffHandler:
    def __init__(self, tiff_path: str) -> None:
        """Initialize with a GeoTIFF file"""
        with rasterio.open(tiff_path) as src:
            self.image = np.transpose(src.read(), (1, 2, 0))
            self.geotransform = src.transform
            tiff_crs = src.crs

        try:
            src_crs = CRS.from_epsg(4326)
            self.transformer = Transformer.from_crs(src_crs, tiff_crs, always_xy=True)
        except Exception as e:
            raise ValueError(f"Failed to create transformer from EPSG:4326 to {tiff_crs}: {e}")

    @property
    def resolution(self) -> float:
        """Resolution in meters per pixel (average of width and height)"""
        gt = self.geotransform
        res_x = np.hypot(gt.a, gt.b)
        res_y = np.hypot(gt.d, gt.e)
        return (res_x + res_y) / 2.0

    @property
    def origin_coords(self) -> np.ndarray:
        """Get the origin coordinates (top-left corner) in the GeoTIFF projection"""
        return np.array([self.geotransform.c, self.geotransform.f])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Returns the (min_x, min_y, max_x, max_y) in map coordinates (UTM)"""
        width, height = self.image.shape[1], self.image.shape[0]
        x_min, y_min = self.geotransform * (0, height)
        x_max, y_max = self.geotransform * (width, 0)
        return min(x_min, x_max), min(y_min, y_max), max(x_min, x_max), max(y_min, y_max)

    def latlon_to_coords(self, latlons: np.ndarray) -> np.ndarray:
        """Convert lat/lon (WGS 84) to coordinates in the GeoTIFF projection"""
        xs, ys = self.transformer.transform(latlons[..., 1], latlons[..., 0])
        return np.stack([xs, ys], axis=-1)

    def latlon_to_pixel(self, latlons: np.ndarray) -> np.ndarray:
        coords = self.latlon_to_coords(latlons)
        return self.coords_to_pixel(coords)

    def coords_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        u, v = (~self.geotransform) * (coords[..., 0], coords[..., 1])
        return np.stack([u, v], axis=-1)

    def coords_to_uvr(self, pose_xyr: np.ndarray) -> np.ndarray:
        uv = self.coords_to_pixel(pose_xyr[..., :2])
        if pose_xyr.shape[-1] > 2:
            return np.concatenate([uv, pose_xyr[..., 2:]], axis=-1)
        return uv


class AerialReframer:
    def __init__(
        self,
        geo_handler: GeoTiffHandler,
        u1: int,
        v1: int,
        out_size: tuple[int, int],
        scaled_size: tuple[int, int],
        rotation: int = 0,
    ) -> None:
        self.geo_handler = geo_handler
        self.u1 = int(u1)
        self.v1 = int(v1)
        self.out_size = tuple(map(int, out_size))
        self.scaled_size = tuple(map(int, scaled_size))
        self.rotation = rotation
        assert self.rotation in [0, 90, 180, 270], "Invalid rotation angle"

        H_out, W_out = out_size
        H_scaled, W_scaled = scaled_size
        self.scale = np.array([W_scaled / W_out, H_scaled / H_out], dtype=np.float32)

    @property
    def resolution(self) -> float:
        return self.geo_handler.resolution * np.mean(self.scale)

    @classmethod
    def from_query_coords(
        cls,
        geo_handler: GeoTiffHandler,
        query_coords: np.ndarray,
        out_size: tuple[int, int],
        scale: float = 1.0,
        rotation: int = 0,
    ) -> AerialReframer:
        H_out, W_out = out_size
        H_scaled = min(int(H_out * scale), geo_handler.image.shape[0])
        W_scaled = min(int(W_out * scale), geo_handler.image.shape[1])

        query_uv = geo_handler.coords_to_pixel(query_coords[None])[0]
        u1 = np.clip(query_uv[0] - W_scaled // 2, 0, geo_handler.image.shape[1] - W_scaled)
        v1 = np.clip(query_uv[1] - H_scaled // 2, 0, geo_handler.image.shape[0] - H_scaled)
        return cls(geo_handler, u1, v1, out_size, (H_scaled, W_scaled), rotation)

    def crop_image(self) -> np.ndarray:
        """Crop the image defined by x1, y1 and scale, then apply rotation."""
        H_out, W_out = self.out_size
        H_scaled, W_scaled = self.scaled_size
        cropped = self.geo_handler.image[self.v1 : self.v1 + H_scaled, self.u1 : self.u1 + W_scaled]
        resized = cv2.resize(cropped, (W_out, H_out), interpolation=cv2.INTER_LINEAR)

        # Apply rotation (0, 90, 180, 270 degrees CCW)
        if self.rotation == 90:
            resized = np.rot90(resized, k=1).copy()
        elif self.rotation == 180:
            resized = np.rot90(resized, k=2).copy()
        elif self.rotation == 270:
            resized = np.rot90(resized, k=3).copy()
        return resized

    def to_uvr(self, pose_xyr: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Convert world/projected coords (x, y, r) to (u, v, r) relative to crop.
        Returns UV in the resized output image coordinate system, accounting for rotation.
        """
        uv = self.geo_handler.coords_to_pixel(pose_xyr[..., :2]).astype(np.float32)
        uv = (uv - np.array([self.u1, self.v1], dtype=np.float32)) / self.scale
        H_out, W_out = self.out_size

        # apply rotation transformation to UV coordinates
        if self.rotation == 90:
            uv_rot = np.stack([uv[..., 1], (W_out - 1) - uv[..., 0]], axis=-1)
        elif self.rotation == 180:
            uv_rot = np.stack([(W_out - 1) - uv[..., 0], (H_out - 1) - uv[..., 1]], axis=-1)
        elif self.rotation == 270:
            uv_rot = np.stack([(H_out - 1) - uv[..., 1], uv[..., 0]], axis=-1)
        else:
            uv_rot = uv

        # clip to image bounds
        if clip:
            uv_rot = np.clip(uv_rot, 0, np.array([W_out - 1, H_out - 1], dtype=np.float32))

        # rotate the heading angle
        if pose_xyr.shape[-1] > 2:
            r_offset = np.deg2rad(self.rotation)
            r_rotated = pose_xyr[..., 2:3] + r_offset
            r_rotated = np.arctan2(np.sin(r_rotated), np.cos(r_rotated))
            return np.concatenate([uv_rot, r_rotated], axis=-1)

        return uv_rot

    def sample_poses(
        self,
        gt_xyr: np.ndarray,
        n_poses: int,
        dist_bounds: tuple[float, float],
        r_bounds: tuple[float, float],
        radial_power: float,
        margin_px: float = 50.0,
    ) -> np.ndarray:
        """Sample poses and guarantee projected (u,v) fall inside the crop."""
        H, W = self.out_size
        out = [gt_xyr[None]]
        need = n_poses - 1

        while need > 0:
            k = max(32, need * 2)
            candidates = sample_uniform_poses(gt_xyr, k, dist_bounds, r_bounds, radial_power)

            uvr = self.to_uvr(candidates, clip=False)
            u, v = uvr[:, 0], uvr[:, 1]
            mask = (u >= margin_px) & (u <= (W - margin_px)) & (v >= margin_px) & (v <= (H - margin_px))
            valid = candidates[mask]
            if valid.size == 0:
                continue

            take = valid[:need]
            out.append(take)
            need -= take.shape[0]

        return np.concatenate(out, axis=0)[:n_poses]


###
# Helper functions
###


def sample_uniform_poses(
    gt_xyr: np.ndarray,
    n_poses: int,
    dist_bounds: tuple[float, float],
    r_bounds: tuple[float, float],
    radial_power: float = 1.0,
) -> np.ndarray:
    """Sample uniform poses around the ground truth pose.
    Args:
        gt_xyr: ground truth pose (3,) - <x, y, theta>
        n_poses: number of poses to sample
        dist_bounds: radial distance bounds in meters (min, max)
        r_bounds: heading offset bounds in degrees (min, max)
        radial_power: power for radial distance sampling
    Returns:
        pose_xyr: sampled poses (N, 3) - <x, y, theta>
    """
    # Sample radial distance (area-uniform) and angle
    d_min, d_max = dist_bounds
    u = np.random.random(n_poses).astype(np.float32)
    p = float(radial_power)
    # CDF: (d^p - dmin^p) / (dmax^p - dmin^p) = u
    d = (u * (d_max**p - d_min**p) + d_min**p) ** (1.0 / p)

    # Compute xy offsets
    theta = np.random.random(n_poses) * 2 * np.pi
    offset_x = d * np.cos(theta)
    offset_y = d * np.sin(theta)

    # Sample heading offset
    r_min, r_max = r_bounds
    offset_r = np.random.random(n_poses) * (r_max - r_min) + r_min
    offset_r = np.deg2rad(offset_r)  # convert to radians

    offset_xyr = np.stack([offset_x, offset_y, offset_r], axis=-1)
    pose_xyr = gt_xyr[None, :] + offset_xyr
    return pose_xyr
