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

        self.buf_uncertainty = deque(maxlen=512)
        self.buf_fail_top1 = deque(maxlen=512)

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

        self.all_uncertainty = []
        self.all_fail_top1 = []

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

        # rank / fail
        rank = 1 + (s_neg >= s_pos.unsqueeze(1)).sum(dim=-1)
        self.all_ranks.extend(rank.tolist())
        fail_vec = (rank > 1).float()
        fail_top1_rate = fail_vec.mean().item()

        self.buf_uncertainty.extend(uncertainty.tolist())
        self.buf_fail_top1.extend(fail_vec.tolist())

        auroc_u_fail = float("nan")
        if len(self.buf_uncertainty) >= 64:
            u = torch.tensor(list(self.buf_uncertainty))
            f = torch.tensor(list(self.buf_fail_top1))
            auroc_u_fail = auroc(u, f)

        # hard negatives
        k = min(self.k_hard, s_neg.shape[1])
        hard_neg = s_neg.topk(k, dim=1).values.mean(dim=1)  # (B,)
        max_neg = s_neg.max(dim=1).values
        self.all_max_neg.extend(max_neg.tolist())

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

        return {
            "rank": rank.float().mean().item(),
            "recall@1": (rank <= 1).float().mean().item(),
            "fail_top1_rate": fail_top1_rate,
            "auroc(uncertainty->fail_top1)": auroc_u_fail,
            "gap_hard": gap_hard.mean().item(),
            "gap_max": gap_max.mean().item(),
            "z_sep": z_sep.mean().item(),
            "z_hard": z_hard.mean().item(),
            "gt_score": s_pos.mean().item(),
            "mu_neg": mu_neg.mean().item(),
            "std_neg": std_neg.mean().item(),
            "max_neg": max_neg.mean().item(),
        }

    def compute(self) -> dict[str, float]:
        """Computes and returns the final retrieval metrics."""

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

        u_all = torch.tensor(self.all_uncertainty) if self.all_uncertainty else None
        f_all = torch.tensor(self.all_fail_top1) if self.all_fail_top1 else None
        auroc_epoch = auroc(u_all, f_all) if u_all is not None else float("nan")

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
                "fail_top1_rate": (all_ranks > 1).float().mean().item(),
                "auroc(uncertainty->fail_top1)": auroc_epoch,
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


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Computes Area Under the Receiver Operating Characteristic Curve (AUROC)."""
    scores = scores.flatten().float()
    labels = labels.flatten().float()

    P = labels.sum().item()
    N = labels.numel() - P
    if P < 1 or N < 1:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    s = scores[order]
    y = labels[order]

    # indices of the last element of each distinct score group
    change = (s[1:] != s[:-1]).nonzero(as_tuple=False).flatten()
    last = torch.cat([change, torch.tensor([s.numel() - 1], device=s.device)])

    tps = torch.cumsum(y, dim=0)[last]
    fps = torch.cumsum(1 - y, dim=0)[last]

    # prepend origin
    tps = torch.cat([torch.zeros(1, device=s.device), tps])
    fps = torch.cat([torch.zeros(1, device=s.device), fps])

    tpr = tps / P
    fpr = fps / N

    return torch.trapz(tpr, fpr).item()
