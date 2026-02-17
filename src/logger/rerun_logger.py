import os
import warnings
from collections import defaultdict

import matplotlib.cm as cm
import numpy as np
import rerun as rr
import torch
from manifpy import SE2
from sklearn.decomposition import PCA

from particle_filter import ParticleFilter
from utils.io import denormalize_image_tensor, tensor_to_image
from utils.visualization import depth_to_rgb, features_to_rgb

COLORS = {"gt": (0, 255, 0), "est": (255, 0, 0), "odom": (0, 0, 255), "particles": (255, 255, 0)}


class RerunLogger:
    """rerun logger for Particle Filter visualization."""

    def __init__(self, pf: ParticleFilter, output_dir: str):
        self.pf = pf
        self.geo_handler = pf.observation_model.geo_handler

        self.odom_pose = None
        self.frame_id = -1
        self.trajectories_uv = defaultdict(list)
        self.pca_models = {}
        self.trajectory_log_interval = 10

        # ReRun logger
        rr.init("pf_visualization", spawn=False)
        rr.save(os.path.join(output_dir, "pf_visualization.rrd"))
        self._log_image("map/geo_map", self.geo_handler.image, static=True)

    def log(self, frame: dict, results: dict) -> None:
        # update poses
        gt_pose = SE2(*frame["gt_xyr"][0])
        est_pose = SE2(*self.pf.pose)
        self.odom_pose = self.odom_pose * SE2(*frame["action"][0]) if self.odom_pose else gt_pose

        gt_xyr = np.array([gt_pose.x(), gt_pose.y(), gt_pose.angle()])
        est_xyr = np.array([est_pose.x(), est_pose.y(), est_pose.angle()])
        odom_xyr = np.array([self.odom_pose.x(), self.odom_pose.y(), self.odom_pose.angle()])

        # --- log data to rerun ---
        self.frame_id += 1
        rr.set_time("frame_id", sequence=self.frame_id)

        # ground image / depth
        ground_image = tensor_to_image(denormalize_image_tensor(frame["ground_image"][0], image_type="ground"))
        self._log_image("ground_image", ground_image)
        ground_depth = frame["ground_depth"][0].cpu().numpy()
        ground_depth_rgb = depth_to_rgb(ground_depth)
        self._log_image("ground_depth", ground_depth_rgb)

        # map frame
        gt_uvr = self.geo_handler.coords_to_uvr(gt_xyr)
        self._log_trajectory("map/gt", gt_uvr, COLORS["gt"], length=50.0, radii=15.0, draw_order=1)
        est_uvr = self.geo_handler.coords_to_uvr(est_xyr)
        self._log_trajectory("map/est", est_uvr, COLORS["est"], length=50.0, radii=15.0, draw_order=3)
        odom_uvr = self.geo_handler.coords_to_uvr(odom_xyr)
        self._log_trajectory("map/odom", odom_uvr, COLORS["odom"], length=50.0, radii=15.0, draw_order=2)
        particles_xyr = np.array([[p.x, p.y, p.theta] for p in self.pf.particles])
        particles_uvr = self.geo_handler.coords_to_uvr(particles_xyr)
        self._log_arrows("map/particles", particles_uvr, color=COLORS["particles"], length=30.0, radii=5.0)

        # local frame
        aerial_image = tensor_to_image(denormalize_image_tensor(results["aerial_image"][0], image_type="aerial"))
        self._log_image("local/aerial_image", aerial_image)
        self._log_arrows(
            "local/particles",
            results["particles_uvr"],
            color=COLORS["particles"],
            length=10.0,
            radii=np.exp(results["log_likelihood"]),
        )

        local_gt_uvr = results["reframer"].to_uvr(gt_xyr)
        local_est_uvr = results["reframer"].to_uvr(est_xyr)
        local_odom_uvr = results["reframer"].to_uvr(odom_xyr)
        self._log_arrows("local/gt", local_gt_uvr, color=COLORS["gt"], length=30.0, radii=5.0, draw_order=1)
        self._log_arrows("local/est", local_est_uvr, color=COLORS["est"], length=30.0, radii=5.0, draw_order=3)
        self._log_arrows("local/odom", local_odom_uvr, color=COLORS["odom"], length=30.0, radii=5.0, draw_order=2)

        self._log_scalar("var", results["var"], color=(128, 0, 0), width=2.0)
        self._log_scalar("var_ema", results["var_ema"], color=(128, 128, 0), width=2.0)
        self._log_scalar("alpha", results["alpha"], color=(0, 128, 0), width=2.0)

        # distinctiveness
        distinct_map = torch.sigmoid(results["distinct_logits"]).detach().squeeze(0).cpu().numpy()
        self._log_costmap(distinct_map, tag="bev/distinctiveness", vmin=0.0, vmax=1.0)

        # local aerial feature and BEV feature
        if False:
            self._log_feature(
                results["aer_feat"].detach().squeeze(0),
                results["bev_feat"].detach().squeeze(0),
                tags=["local/aer_feat", "bev/bev_feat"],
                upsample_scales=[4, 1],
                latent_space="bev",
            )

        # ground and BEV embeddings
        if False:
            self._log_feature(
                results["gnd_emb"].detach(),
                results["bev_emb"].detach(),
                tags=["gnd_emb", "bev/bev_emb"],
                latent_space="gnd",
            )

    ###
    # Helper functions (rerun logging)
    ###

    def _log_trajectory(
        self, tag: str, pose_uvr: np.ndarray, color: tuple[int, int, int], length: int, radii: int, draw_order: int = 0
    ):
        self._log_arrows(tag, pose_uvr, color=color, length=length, radii=radii, draw_order=draw_order)

        self.trajectories_uv[tag].append(pose_uvr[:2])
        if self.frame_id % self.trajectory_log_interval == 0:
            rr.log(f"{tag}_traj", rr.LineStrips2D(self.trajectories_uv[tag], radii=radii * 0.8, colors=color))

    def _log_arrows(
        self,
        tag: str,
        pose_uvr: np.ndarray | torch.Tensor,
        color: tuple[int, int, int],
        length: float,
        radii: np.ndarray | float,
        draw_order: int = 0,
    ):
        if isinstance(pose_uvr, torch.Tensor):
            pose_uvr = pose_uvr.detach().cpu().numpy().reshape(-1, 3)
        elif isinstance(pose_uvr, np.ndarray):
            pose_uvr = pose_uvr.astype(np.float32).reshape(-1, 3)
        else:
            raise TypeError(f"Unsupported pose type: {type(pose_uvr)}")

        origins, thetas = pose_uvr[:, :2], pose_uvr[:, 2]
        vectors = length * np.column_stack((np.cos(-thetas), np.sin(-thetas)))
        rr.log(tag, rr.Arrows2D(origins=origins, vectors=vectors, colors=color, radii=radii, draw_order=draw_order))

    def _log_image(self, tag: str, image: np.ndarray, jpeg_quality: int = 30, static: bool = False):
        image = image.astype(np.uint8) if image.dtype != np.uint8 else image
        rr.log(tag, rr.Image(image).compress(jpeg_quality=jpeg_quality), static=static)

    def _log_costmap(self, costmap: np.ndarray, tag: str, vmin: float, vmax: float):
        costmap = costmap.astype(np.float32) if costmap.dtype != np.float32 else costmap
        if costmap.ndim == 3 and costmap.shape[0] == 1:
            costmap = costmap[0]
        elif costmap.ndim != 2:
            raise ValueError(f"Expected costmap with shape (H, W) or (1, H, W), got {costmap.shape}")

        normalized = np.clip((costmap - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
        turbo_rgb = (cm.get_cmap("turbo")(normalized)[..., :3] * 255).astype(np.uint8)
        self._log_image(tag, turbo_rgb)

    def _log_scalar(self, tag: str, value: float, color: tuple[int, int, int], width: float = 1.0):
        rr.log(tag, rr.Scalars(value), rr.SeriesLines(colors=color, widths=width), static=True)

    def _log_feature(
        self,
        *features: np.ndarray,
        tags: list[str],
        upsample_scales: list[int] | None = None,
        latent_space: str = "default",
    ):
        if len(features) != len(tags):
            warnings.warn(f"{len(features)} != {len(tags)}")
            return

        if upsample_scales is None:
            upsample_scales = [1] * len(features)
        elif len(upsample_scales) != len(features):
            warnings.warn(f"{len(upsample_scales)} != {len(features)}")
            return

        if latent_space not in self.pca_models:
            self.pca_models[latent_space] = PCA(n_components=3)
        pca = self.pca_models[latent_space]

        feature_rgbs = features_to_rgb(*features, pca=pca)

        for tag, feature_rgb, scale in zip(tags, feature_rgbs, upsample_scales, strict=True):
            if scale > 1:
                feature_rgb = np.repeat(np.repeat(feature_rgb, scale, axis=0), scale, axis=1)
            self._log_image(tag, feature_rgb)
