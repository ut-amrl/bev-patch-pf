import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

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


def visualize_feature_map(data: dict, output: dict, outdir: str | None = None, prefix: str = ""):
    """Visualize the feature map and similarity heatmap"""
    ground_image = tensor_to_image(denormalize_image_tensor(data["ground_image"], image_type="ground"))
    aerial_image = tensor_to_image(denormalize_image_tensor(data["aerial_image"], image_type="aerial"))
    aerial_image2 = tensor_to_image(denormalize_image_tensor(data["aerial_image2"], image_type="aerial"))

    score = output["score"].detach().cpu()  # (B, N)
    similarity = output["similarity"].detach().cpu()
    raw_score = similarity.mean(dim=(-1, -2))  # (B, N)

    similarity_margin = output["similarity_margin"].detach().cpu()
    distinct_map = torch.sigmoid(output["distinct_logits"]).squeeze(1).detach().cpu()
    uncertainty_map = output["uncertainty_map"].squeeze(1).detach().cpu()
    temp = output.get("temp", 0.05)

    pose_xyr = data["pose_xyr"].cpu()  # (B, N, 3)
    pose_uvr = data["pose_uvr"].cpu()  # (B, N, 3)
    pose_uvr2 = data["pose_uvr2"].cpu()  # (B, N, 3)

    B, N_all, _ = pose_uvr.shape
    N = min(N_all, 4)

    for b in range(B):
        # determine indices for visualization
        neg_idx_desc = np.argsort(score[b, 1:].numpy())[::-1] + 1
        hard_idx = neg_idx_desc[: N - 2]
        easy_idx = neg_idx_desc[-1:]
        viz_idx = np.concatenate(([0], hard_idx, easy_idx))
        labels = ["gt"] + ["hard"] * len(hard_idx) + ["easy"] * len(easy_idx)

        # false color visualization for BEV/aerial features
        pca = PCA(n_components=3)
        bev_feat_rgb, aer_feat_rgb, *aer_patches_rgb = features_to_rgb(
            output["bev_feat"][b], output["aer_feat"][b], *output["aer_patches_feat"][b, viz_idx], pca=pca
        )
        aer_feat2_rgb, *aer_patches2_rgb = features_to_rgb(
            output["aer_feat2"][b], *output["aer_patches_feat2"][b, viz_idx], pca=pca
        )

        # create subplots
        fig, axes = plt.subplots(N + 3, 3, figsize=(12 * 3, 12 * (N + 4)))

        for ax in axes.flatten():
            ax.axis("off")

        # show onboard and aerial images
        axes[0, 0].imshow(ground_image[b])
        axes[0, 0].set_title("Onboard Image")
        axes[0, 1].imshow(aerial_image[b])
        axes[0, 1].set_title("Aerial Image")
        plot_pose_markers(axes[0, 1], poses=pose_uvr[b], mode="dot", color_map={"default": "gray"})
        plot_pose_markers(axes[0, 1], poses=pose_uvr[b, viz_idx], labels=labels, mode="arrow")

        axes[0, 2].imshow(aerial_image2[b])
        axes[0, 2].set_title("Aerial Image 2 (for consistency)")
        plot_pose_markers(axes[0, 2], poses=pose_uvr2[b], mode="dot", color_map={"default": "gray"})
        plot_pose_markers(axes[0, 2], poses=pose_uvr2[b, viz_idx], labels=labels, mode="arrow")

        # show feature visualizations
        axes[1, 0].imshow(bev_feat_rgb)
        axes[1, 0].set_title("BEV Feature")
        axes[1, 1].imshow(aer_feat_rgb)
        axes[1, 1].set_title("Aerial Feature")
        axes[1, 2].imshow(aer_feat2_rgb)
        axes[1, 2].set_title("Aerial Feature 2")

        # visualize distinctiveness
        plot_heatmap(distinct_map[b], ax=axes[2, 0], vmin=0.0, vmax=1.0)
        axes[2, 0].set_title("Distinctiveness Map")

        # visualize similarity margin (target of distinctiveness)
        plot_heatmap(similarity_margin[b], vmin=-1.0, vmax=1.0, ax=axes[2, 1])
        axes[2, 1].set_title("Similarity Margin Map")

        # visualize uncertainty
        plot_heatmap(uncertainty_map[b], vmin=-5.0, vmax=5.0, ax=axes[2, 2])
        axes[2, 2].set_title(f"Uncertainty Map (log_var: {output['uncertainty'][b].item():.3f})")

        # visualize for each pose
        for i, idx in enumerate(viz_idx):
            ax0, ax1, ax2 = axes[i + 3]

            dx, dy, dr = pose_xyr[b, idx] - pose_xyr[b, 0]
            ax0.imshow(aer_patches_rgb[i])
            ax0.set_title(f"Idx: {idx} | dx={dx:.2f}, dy={dy:.2f}, dr={np.rad2deg(dr):.2f}°")

            mask = similarity[b][idx] != 0
            plot_heatmap(similarity[b][idx], mask=mask, vmin=-1.0, vmax=1.0, ax=ax1)
            ax1.set_title(f"Similarity Map | Score: {score[b][idx]:.2f}, Raw: {raw_score[b][idx]:.2f}")

            ax2.imshow(aer_patches2_rgb[i])
            ax2.set_title("Aerial Patch Feature 2")

        # show/save figure
        rank = 1 + (score[b][1:] > score[b][0]).sum()
        temps = [0.01, 0.05, 0.1, round(temp, 3)]
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
        order = np.argsort([priority.get(l, -1) for l in labels])
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


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


@torch.no_grad()
def features_to_rgb(
    *features: torch.Tensor | np.ndarray, pca: PCA | None = None, n_fit: int = 100_000, thr: float = 1e-3
) -> list[np.ndarray]:
    """Projects high-dim features to RGB using PCA."""
    # 1. extract valid pixels from each feature map chunk by chunk
    pix_chunks, metadata = [], []
    for feat in features:
        feat_np = feat.detach().cpu().numpy() if isinstance(feat, torch.Tensor) else feat
        C, H, W = feat.shape

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

    if not hasattr(pca, "components_"):
        all_data = np.concatenate([p for p in pix_chunks if p is not None])
        subset = all_data[np.random.permutation(len(all_data))[:n_fit]]
        pca.fit(subset)

    # Calculate global min/max for normalization (1st/99th percentile)
    all_proj = np.concatenate([pca.transform(p) for p in valid_pixels])
    p_min = np.percentile(all_proj, 1, axis=0)
    p_max = np.percentile(all_proj, 99, axis=0)

    output_images = []
    for pixels, meta in zip(pix_chunks, metadata):
        img = np.zeros((meta["H"], meta["W"], 3), dtype=np.uint8)

        if pixels is not None:
            proj = pca.transform(pixels)
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
