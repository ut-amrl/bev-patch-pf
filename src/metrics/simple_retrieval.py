from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch


class SimpleRetrievalMetric:
    """Computes standard retrieval metrics: Recall@k and Mean Reciprocal Rank (MRR)."""

    def __init__(self, ks: tuple[int, ...] = (1, 5, 10), k_hard: int = 3) -> None:
        """Initializes the metric.

        Args:
            ks: Tuple of top-k values for which to calculate recall.
        """
        self.ks = ks
        self.k_hard = k_hard

        self.buf_margin = deque(maxlen=512)
        self.buf_uncertainty = deque(maxlen=512)

        self.reset()

    def reset(self):
        """Clears accumulated ranks."""
        self.all_ranks = []
        self.all_zsep = []
        self.all_zhard = []

        self.all_spos = []
        self.all_mu_neg = []
        self.all_std_neg = []
        self.all_max_neg = []
        self.all_gap_max = []  # s_pos - max_neg
        self.all_gap_hard = []  # s_pos - hard_neg

    @torch.no_grad()
    def update(self, score: torch.Tensor, uncertainty: torch.Tensor, **kwargs) -> dict[str, float]:
        """
        score: (B, N) - similarity scores where the first entry is the positive
        uncertainty: (B,) - frame-level uncertainty (log variance)
        """
        score = score.detach().cpu()
        uncertainty = uncertainty.detach().cpu()

        s_pos, s_neg = score[:, 0], score[:, 1:]
        self.all_spos.extend(s_pos.tolist())

        # rank
        rank = 1 + (s_neg >= s_pos.unsqueeze(1)).sum(dim=-1)
        self.all_ranks.extend(rank.tolist())

        # hard negatives
        k = min(self.k_hard, s_neg.shape[1])
        hard_neg = s_neg.topk(k, dim=1).values.mean(dim=1)  # (B,)
        max_neg = s_neg.max(dim=1).values
        self.all_max_neg.extend(max_neg.tolist())

        # margins / gaps
        gap_hard = s_pos - hard_neg
        gap_max = s_pos - max_neg
        self.all_gap_hard.extend(gap_hard.tolist())
        self.all_gap_max.extend(gap_max.tolist())
        # z-separation
        mu_neg = s_neg.mean(dim=1)
        std_neg = s_neg.std(dim=1, unbiased=False).clamp_min(1e-6)
        z_sep = (s_pos - mu_neg) / std_neg
        z_hard = (s_pos - hard_neg) / std_neg
        self.all_mu_neg.extend(mu_neg.tolist())
        self.all_std_neg.extend(std_neg.tolist())
        self.all_zsep.extend(z_sep.tolist())
        self.all_zhard.extend(z_hard.tolist())

        # correlation (margin vs uncertainty)
        for m, u in zip(gap_hard.tolist(), uncertainty.tolist()):
            self.buf_margin.append(m)
            self.buf_uncertainty.append(u)

        corr = float("nan")
        uncertainty_std = float("nan")
        if len(self.buf_margin) >= 64:
            x = torch.tensor(list(self.buf_margin))
            y = torch.tensor(list(self.buf_uncertainty))
            corr = pearson_corr(x, y)
            uncertainty_std = y.std(unbiased=False).item()

        return {
            "rank": rank.float().mean().item(),
            "recall@1": (rank <= 1).float().mean().item(),
            "gap_hard": gap_hard.mean().item(),
            "gap_max": gap_max.mean().item(),
            "z_sep": z_sep.mean().item(),
            "z_hard": z_hard.mean().item(),
            "gt_score": s_pos.mean().item(),
            "mu_neg": mu_neg.mean().item(),
            "std_neg": std_neg.mean().item(),
            "max_neg": max_neg.mean().item(),
            "correlation (gap_hard vs. uncertainty)": corr,
            "uncertainty_std": uncertainty_std,
        }

    def compute(self) -> dict[str, float]:
        """Computes and returns the final retrieval metrics.

        Returns:
            Dict with Recall@k for each k and the Mean Reciprocal Rank (MRR).
        """
        if not self.all_ranks:
            return {}

        all_ranks = torch.tensor(self.all_ranks)
        gap_hard = torch.tensor(self.all_gap_hard)
        gap_max = torch.tensor(self.all_gap_max)
        zsep = torch.tensor(self.all_zsep)
        zhards = torch.tensor(self.all_zhard)

        s_pos = torch.tensor(self.all_spos)
        mu_neg = torch.tensor(self.all_mu_neg)
        std_neg = torch.tensor(self.all_std_neg)
        max_neg = torch.tensor(self.all_max_neg)

        metrics = {f"recall@{k}": (all_ranks <= k).float().mean().item() for k in self.ks}
        metrics.update(
            {
                "mrr": (1.0 / all_ranks).mean().item(),
                "z_sep": zsep.mean().item(),
                "z_hard": zhards.mean().item(),
                "gt_score": s_pos.mean().item(),
                "mu_neg": mu_neg.mean().item(),
                "std_neg": std_neg.mean().item(),
                "max_neg": max_neg.mean().item(),
                "gap_hard": gap_hard.mean().item(),
                "gap_max": gap_max.mean().item(),
                "gap_max_p05": torch.quantile(gap_max, 0.05).item(),
                "gap_max_p50": torch.quantile(gap_max, 0.50).item(),
                "std_neg_p05": torch.quantile(std_neg, 0.05).item(),
                "fail_top1_rate": (all_ranks > 1).float().mean().item(),
            }
        )
        return metrics

    def plot_cdf(self, save_path: str, title: str = "Retrieval CDF"):
        """Generates and saves a CDF plot with subplots for ranks, z-separation, and z-hard.

        Args:
            save_path: Path to save the plot.
            title: Title for the plot.
        """
        if not self.all_ranks:
            return

        all_ranks = torch.cat(self.all_ranks, dim=0).numpy()
        all_zsep = torch.cat(self.all_zsep, dim=0).numpy()
        all_zhard = torch.cat(self.all_zhard, dim=0).numpy()

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Rank CDF
        sorted_ranks = np.sort(all_ranks)
        cdf_ranks = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        axes[0].plot(sorted_ranks, cdf_ranks, linewidth=2)
        axes[0].set_xlabel("Rank", fontsize=12)
        axes[0].set_ylabel("CDF", fontsize=12)
        axes[0].set_title("Rank CDF", fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, max(all_ranks) * 1.05])
        axes[0].set_ylim([0, 1])

        # Z-Separation CDF
        sorted_zsep = np.sort(all_zsep)
        cdf_zsep = np.arange(1, len(sorted_zsep) + 1) / len(sorted_zsep)
        axes[1].plot(sorted_zsep, cdf_zsep, linewidth=2)
        axes[1].set_xlabel("Z-Separation", fontsize=12)
        axes[1].set_ylabel("CDF", fontsize=12)
        axes[1].set_title("Z-Separation CDF", fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        # Z-Hard CDF
        sorted_zhard = np.sort(all_zhard)
        cdf_zhard = np.arange(1, len(sorted_zhard) + 1) / len(sorted_zhard)
        axes[2].plot(sorted_zhard, cdf_zhard, linewidth=2)
        axes[2].set_xlabel("Z-Hard", fontsize=12)
        axes[2].set_ylabel("CDF", fontsize=12)
        axes[2].set_title("Z-Hard CDF", fontsize=14)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1])

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


###
# Helper functions
###


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes Pearson correlation coefficient between two 1D tensors."""
    if x.numel() < 2:
        return 0.0

    x = x.float()
    y = y.float()

    vx = x - x.mean()
    vy = y - y.mean()

    std_x = x.std(unbiased=False)
    std_y = y.std(unbiased=False)

    # SAFETY CHECK: If std is 0 (constant output), correlation is undefined (0)
    if std_x < eps or std_y < eps:
        return 0.0

    return ((vx * vy).mean() / (std_x * std_y + eps)).item()
