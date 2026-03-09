import os
import random

import hydra
import numpy as np
import torch
import torch.nn as nn


def as_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            value = value.float().mean()
        return float(value.item())
    if isinstance(value, np.generic):
        return float(value)
    return float(value)


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, torch and configure CuDNN for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path: str, model, criterion, optimizer, scheduler, epoch) -> None:
    ckpt = {
        "model": model.state_dict(),
        "criterion": criterion.state_dict() if criterion is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    criterion=None,
    optimizer=None,
    scheduler=None,
    strict_model: bool = True,
) -> tuple[int, bool]:
    """
    If `path` is a full training checkpoint (contains 'model' or 'epoch'), resume everything.
    Else, treat it as weights-only and just load into the model.
    Returns (start_epoch, resumed: bool)
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)

    def _strip_module(state_dict):
        return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    if isinstance(blob, dict) and (("model" in blob) or ("epoch" in blob) or ("optimizer" in blob)):
        model.load_state_dict(_strip_module(blob["model"]), strict=strict_model)
        if criterion is not None and blob.get("criterion") is not None:
            criterion.load_state_dict(_strip_module(blob["criterion"]))
        if optimizer is not None and blob.get("optimizer") is not None:
            optimizer.load_state_dict(blob["optimizer"])
        if scheduler is not None and blob.get("scheduler") is not None:
            scheduler.load_state_dict(blob["scheduler"])

        start_epoch = int(blob.get("epoch", -1)) + 1
        return start_epoch, True

    state = blob.get("state_dict") if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state, strict=strict_model)
    return 0, False


def load_model(
    model_cfg,
    ckpt_path: str | None = None,
    *,
    device: str | torch.device | None = None,
    strict_model: bool = True,
    eval_mode: bool = False,
):
    model = hydra.utils.instantiate(model_cfg)

    if ckpt_path is not None:
        load_checkpoint(ckpt_path, model, strict_model=strict_model)
        print(f"Loaded model from {ckpt_path}")

    if device is not None:
        model = model.to(device=device)

    if eval_mode:
        model = model.eval()

    return model
