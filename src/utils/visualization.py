from __future__ import annotations

import logging
import os

import matplotlib
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize

from utils.io import denormalize_image_tensor, tensor_to_image

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    }
)

COLORS = {"gt": "lime", "hard": "red", "easy": "blue", "default": "black"}


###
# Main visualization function
###


def visualize_feature_map(data: dict, output: dict, cfg: DictConfig, outdir: str | None = None, prefix: str = ""):
    """Visualize the feature map and similarity heatmap"""
    ground_image = tensor_to_image(denormalize_image_tensor(data["ground_image"], image_type="ground"))
    aerial_image = tensor_to_image(denormalize_image_tensor(data["aerial_image"], image_type="aerial"))

    bev_feat = output["bev_feat"].detach().cpu()
    bev_mask = output["bev_mask"].detach().cpu()
    aer_patches_feat = output["aer_patches_feat"].detach().cpu()
    aer_feat = output["aer_raw_feat"]

    pose_xyr = data["pose_xyr"].cpu()  # (B, N, 3)
    pose_uvr = data["pose_uvr"].cpu()  # (B, N, 3)

    score = output["score"].detach().cpu()  # (B, N)

    B, N_all, _, H, W = output["aer_patches_feat"].shape

    with torch.no_grad():
        similarity = torch.einsum("bchw,bnchw->bnhw", output["bev_feat"], output["aer_patches_feat"]).detach().cpu()
        raw_score = (similarity * bev_mask).sum(dim=(-1, -2)) / bev_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)[:, None]
        pos = similarity[:, 0]
        hardest_neg_idx = (score[:, 1:].argmax(dim=1) + 1)[:, None, None, None].expand(-1, 1, H, W)
        hardest_neg = torch.gather(similarity, dim=1, index=hardest_neg_idx).squeeze(1)
        margin = pos - hardest_neg
        distinct_target = torch.sigmoid(cfg.criterion.distinct_pos_coeff * pos) * torch.sigmoid(
            cfg.criterion.distinct_margin_coeff * margin
        )
        distinct_map = torch.sigmoid(output["distinct_logits"]).detach().cpu()  # (B, 1, H, W)
        uncertainty_map = output["uncertainty_map"].squeeze(1).detach().cpu()
        w_similarity = distinct_map * similarity  # (B, N, H, W)

    N = 3  # number of poses to visualize
    for b in range(B):
        # determine indices for visualization
        neg_scores = score[b, 1:]
        neg_idx_desc = torch.topk(neg_scores, k=neg_scores.numel(), largest=True, sorted=True).indices + 1
        hard_idx = neg_idx_desc[: N - 2].numpy()
        easy_idx = neg_idx_desc[-1:].numpy()
        viz_idx = np.concatenate([[0], hard_idx, easy_idx])
        labels = ["gt"] + ["hard"] * len(hard_idx) + ["easy"] * len(easy_idx)

        # false color visualization for BEV/aerial features
        pca = PCA(n_components=3)
        bev_feat_rgb, aer_feat_rgb, *aer_patches_rgb = features_to_rgb(
            bev_feat[b], aer_feat[b], *aer_patches_feat[b, viz_idx], pca=pca
        )

        # create subplots
        fig, axes = plt.subplots(N + 3, 3, figsize=(12 * 3, 12 * (N + 3)))

        for ax in axes.flatten():
            ax.axis("off")

        # show onboard and aerial images
        ax0, ax1, ax2 = axes[0]
        ax0.imshow(ground_image[b])
        ax0.set_title("Onboard Image")
        ax1.imshow(aerial_image[b])
        ax1.set_title("Aerial Image")
        plot_pose_markers(ax1, poses=pose_uvr[b], mode="dot", color_map={"default": "gray"})
        plot_pose_markers(ax1, poses=pose_uvr[b, viz_idx], labels=labels, mode="arrow")

        # show feature visualizations
        ax0, ax1, ax2 = axes[1]
        ax0.imshow(bev_feat_rgb)
        ax0.set_title("BEV Feature")
        ax1.imshow(aer_feat_rgb)
        ax1.set_title("Aerial Feature")

        # visualize distinctiveness & uncertainty
        ax0, ax1, ax2 = axes[2]
        plot_heatmap(distinct_map[b].squeeze(0), ax=ax0, vmin=0.0, vmax=1.0)
        ax0.set_title("Distinctiveness Map")
        plot_heatmap(distinct_target[b], ax=ax1, vmin=0.0, vmax=1.0)
        ax1.set_title("Distinctiveness Target")
        plot_heatmap(uncertainty_map[b], ax=ax2, vmin=-5.0, vmax=5.0)
        ax2.set_title(f"Uncertainty Map (log_var: {output['uncertainty'][b].item():.3f})")

        # visualize for each pose
        for i, idx in enumerate(viz_idx):
            ax0, ax1, ax2 = axes[i + 3]

            dx, dy, dr = pose_xyr[b, idx] - pose_xyr[b, 0]
            ax0.imshow(aer_patches_rgb[i])
            ax0.set_title(f"Idx: {idx} | dx={dx:.2f}, dy={dy:.2f}, dr={np.rad2deg(dr):.2f}°")

            mask = similarity[b][idx] != 0
            plot_heatmap(similarity[b][idx], ax=ax1, mask=mask, vmin=-1.0, vmax=1.0)
            ax1.set_title(f"Raw Similarity Map | Raw Score: {raw_score[b][idx]:.2f}")

            plot_heatmap(w_similarity[b][idx], ax=ax2, mask=mask, vmin=-1.0, vmax=1.0)
            ax2.set_title(f"Weighted Similarity Map | Score: {score[b][idx]:.2f}")

        # show/save figure
        rank = 1 + (score[b][1:] > score[b][0]).sum()
        temps = [0.01, 0.05, 0.1, 0.2]
        probs = [F.softmax(score[b] / temp, dim=0).numpy()[0] for temp in temps]
        formatted_probs = ", ".join(f"{p:.3f}" for p in probs)
        plt.suptitle(
            f"{data['name'][b]}\n"
            f"Resolution: {data['info']['resolution'][b].item():.2f} (m/px), "
            f"Rotation: {data['info']['rotation'][b].item()} (deg)\n"
            f"Rank: {rank}/{N_all}, Softmax@{temps}: {formatted_probs}"
        )
        plt.tight_layout(rect=(0, 0, 1, 0.97))

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f"{prefix}_{data['name'][b]}.png"), dpi=100)
        else:
            plt.show()
        plt.close(fig)
        fig.clf()
        plt.close("all")


###
# Plotting functions
###


def plot_heatmap(
    data: np.ndarray | torch.Tensor,
    ax: plt.Axes,
    mask: np.ndarray | torch.Tensor | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str = "turbo",
) -> None:
    """Generic heatmap plotter for a specific axis."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        data = np.ma.masked_where(~mask.astype(bool), data)

    valid = data[~np.isnan(data)]
    if valid.size == 0:
        data, vmin, vmax = np.zeros_like(data), 0, 1
    else:
        vmin = vmin if vmin is not None else np.min(valid)
        vmax = vmax if vmax is not None else np.max(valid)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")

    im = ax.imshow(data, cmap=cmap, norm=Normalize(vmin, vmax))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=20)


def plot_pose_markers(
    ax,
    poses: np.ndarray | torch.Tensor,
    labels: list[str] | None = None,
    mode: str = "arrow",
    scale: float = 1.0,
    color_map: dict | None = None,
) -> None:
    if isinstance(poses, torch.Tensor):
        poses = poses.cpu().numpy()

    poses = poses.reshape(-1, 3)
    xs, ys, rads = poses[:, 0] / scale, poses[:, 1] / scale, poses[:, 2]

    # Resolve colors
    current_cmap = {**COLORS, **(color_map or {})}
    colors = [current_cmap.get(l, current_cmap["default"]) for l in (labels or [])]
    if not colors:
        colors = [current_cmap["default"]] * len(poses)

    # Sort drawing order
    if labels:
        priority = {"gt": 2, "hard": 1, "easy": 0, "default": -1}
        order = np.argsort([priority.get(l, -1) for l in labels], kind="stable")
        xs, ys, rads = xs[order], ys[order], rads[order]
        colors = np.array(colors)[order]

    if mode == "arrow":
        dx, dy = 5 * np.cos(rads), -5 * np.sin(rads)
        ax.quiver(xs, ys, dx, dy, angles="xy", scale_units="xy", scale=0.1, color=colors)
    elif mode == "dot":
        ax.scatter(xs, ys, c=colors, s=100, edgecolor="white", linewidth=2.5)


###
# Helper functions
###


class PCA:
    def __init__(self, n_components: int = 3):
        self.n_components = int(n_components)
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> PCA:
        x = np.asarray(data, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array for PCA fit, got {x.shape}")
        if x.shape[0] == 0:
            raise ValueError("Cannot fit PCA on empty data.")

        self.mean_ = x.mean(axis=0, keepdims=True)
        x = x - self.mean_
        _, _, vt = np.linalg.svd(x, full_matrices=False)

        k = min(self.n_components, vt.shape[0])
        components = vt[:k].astype(np.float32, copy=False)
        if k < self.n_components:
            pad = np.zeros((self.n_components - k, x.shape[1]), dtype=np.float32)
            components = np.concatenate([components, pad], axis=0)
        self.components_ = components

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("PCA is not fitted. Call fit() before transform().")

        x = np.asarray(data, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array for PCA transform, got {x.shape}")
        if x.shape[1] != self.components_.shape[1]:
            raise ValueError(f"Input feature dimension mismatch: {x.shape[1]} != {self.components_.shape[1]}")

        mean = self.mean_ if self.mean_ is not None else np.zeros((1, x.shape[1]), dtype=np.float32)
        return (x - mean) @ self.components_.T


@torch.no_grad()
def features_to_rgb(
    *features: torch.Tensor | np.ndarray, pca: PCA | None = None, n_fit: int = 100_000, thr: float = 1e-3
) -> list[np.ndarray]:
    """Projects high-dim features to RGB using PCA."""
    if not features:
        return []

    pix_chunks, metadata = [], []
    for feat in features:
        feat_np = feat.detach().cpu().numpy() if isinstance(feat, torch.Tensor) else feat
        C, H, W = feat_np.shape

        valid_mask = np.sum(np.square(feat_np), axis=0) > thr**2
        idx = np.where(valid_mask.flatten())[0]
        metadata.append({"H": H, "W": W, "idx": idx})

        if idx.size > 0:
            vecs = feat_np.transpose(1, 2, 0).reshape(-1, C)[idx]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-6)
            vecs /= norms
            pix_chunks.append(vecs)
        else:
            pix_chunks.append(None)

    # edge case: all empty
    valid_pixels = [p for p in pix_chunks if p is not None]
    if not valid_pixels:
        return [np.zeros((m["H"], m["W"], 3), dtype=np.uint8) for m in metadata]

    # 2. fit PCA if needed
    if pca is None:
        pca = PCA(n_components=3)

    all_data = np.concatenate(valid_pixels, axis=0)
    needs_fit = (pca.components_ is None) or (pca.components_.shape[1] != all_data.shape[1])
    if needs_fit:
        subset = all_data[np.random.permutation(len(all_data))[: min(n_fit, len(all_data))]]
        pca.fit(subset)

    # Project once and reuse for both global normalization stats and per-image rendering.
    proj_chunks = [pca.transform(pixels) if pixels is not None else None for pixels in pix_chunks]
    all_proj = np.concatenate([proj for proj in proj_chunks if proj is not None], axis=0)
    p_min = np.percentile(all_proj, 1, axis=0)
    p_max = np.percentile(all_proj, 99, axis=0)

    output_images = []
    for proj, meta in zip(proj_chunks, metadata):
        img = np.zeros((meta["H"], meta["W"], 3), dtype=np.uint8)

        if proj is not None:
            proj = np.clip(proj, p_min, p_max)
            norm_proj = (proj - p_min) / (p_max - p_min + 1e-6)
            r, c = np.divmod(meta["idx"], meta["W"])
            img[r, c] = (norm_proj * 255).astype(np.uint8)

        output_images.append(img)

    return output_images


def depth_to_rgb(depth: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """Converts a depth map to an RGB image using a colormap."""
    valid = depth[~np.isnan(depth)]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    p_low = np.percentile(depth, 3)
    p_high = np.percentile(depth, 97)
    depth_clipped = np.clip(depth, p_low, p_high)
    depth_normalized = (depth_clipped - p_low) / (p_high - p_low)

    cmap_func = plt.get_cmap(cmap).copy()
    cmap_func.set_bad(color="black")
    rgba_img = cmap_func(depth_normalized)
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)

    return rgb_img
