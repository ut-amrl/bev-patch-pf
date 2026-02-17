import gc
import logging
import os
from functools import partial

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm as _tqdm

import wandb
from dataset.utils import SubsetWithAttributes, safe_collate
from metrics.simple_retrieval import SimpleRetrievalMetric
from utils.misc import load_checkpoint, save_checkpoint, seed_everything
from utils.visualization import visualize_feature_map

logger = logging.getLogger(__name__)
tqdm = partial(_tqdm, dynamic_ncols=True)


# ============ Training Pipeline ============


def train_step(batch, model, criterion, optimizer, cfg, scaler):
    # move batch to GPU
    ground_image = batch["ground_image"].to(cfg.device, non_blocking=True)
    ground_depth = batch["ground_depth"].to(cfg.device, non_blocking=True)
    aerial_image = batch["aerial_image"].to(cfg.device, non_blocking=True)
    pose_uvr = batch["pose_uvr"].to(cfg.device, non_blocking=True)
    info = {k: v.to(cfg.device, non_blocking=True) for k, v in batch["info"].items()}
    aerial_image2 = batch["aerial_image2"].to(cfg.device, non_blocking=True)
    pose_uvr2 = batch["pose_uvr2"].to(cfg.device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    # forward
    with torch.autocast(torch.device(cfg.device).type, enabled=cfg.amp):
        output = model(ground_image, ground_depth, aerial_image, info, pose_uvr, aerial_image2, pose_uvr2)
        losses = criterion(**output)
        output.update({"temp": losses.get("temp")})
        loss = losses["loss"]

    # backward
    if cfg.amp:
        scaler.scale(loss).backward()
        if cfg.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
    else:
        loss.backward()

    if cfg.clip_grad_norm > 0:
        params = list(model.parameters()) + list(criterion.parameters())
        nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)

    if cfg.amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return output, losses


def train_one_epoch(model, loader, criterion, optimizer, scheduler, cfg, epoch, scaler):
    model.train()

    metric = SimpleRetrievalMetric(ks=(1, 5, 10))

    for idx, batch in enumerate(tqdm(loader, desc=f"Train [{epoch}/{cfg.epochs}]")):
        step = epoch * len(loader) + idx

        # train step
        output, losses = train_step(batch, model, criterion, optimizer, cfg, scaler)
        scheduler.step(epoch + (idx + 1) / len(loader))  # fractional epoch

        # log metrics
        batch_results = metric.update(
            score=output["score"].detach(),
            uncertainty=output["uncertainty"].detach(),
            distinct_logits=output["distinct_logits"].detach(),
        )
        metric.reset()

        if cfg.wandb:
            wandb.log({f"train/{k}": v for k, v in losses.items()}, step=step)
            wandb.log({f"train (metrics)/{k}": v for k, v in batch_results.items()}, step=step)
            wandb.log({"train/lr": optimizer.param_groups[0]["lr"]}, step=step)

        # (debug) visualization
        if step % cfg.visualize_every_n_steps_train == 0:
            visualize_feature_map(batch, output, os.path.join(cfg.output_dir, "figures"), prefix=f"e{epoch}_s{step}")


# ============ Evaluation Pipeline ============


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, cfg: DictConfig, metrics=None, prefix: str = "eval"):
    model.eval()

    if metrics is None:
        metrics = [SimpleRetrievalMetric(ks=(1, 5, 10))]

    for i, batch in enumerate(tqdm(loader, desc="Evaluation")):
        # move batch data to GPU
        ground_image = batch["ground_image"].to(cfg.device, non_blocking=True)
        ground_depth = batch["ground_depth"].to(cfg.device, non_blocking=True)
        aerial_image = batch["aerial_image"].to(cfg.device, non_blocking=True)
        pose_uvr = batch["pose_uvr"].to(cfg.device, non_blocking=True)
        info = {k: v.to(cfg.device, non_blocking=True) for k, v in batch["info"].items()}
        aerial_image2 = batch["aerial_image2"].to(cfg.device, non_blocking=True)
        pose_uvr2 = batch["pose_uvr2"].to(cfg.device, non_blocking=True)

        # forward pass
        with torch.autocast(torch.device(cfg.device).type, enabled=cfg.amp):
            output = model(ground_image, ground_depth, aerial_image, info, pose_uvr, aerial_image2, pose_uvr2)

        # Update metrics
        for metric in metrics:
            metric.update(**output)

        # (debug) visualization
        if i % cfg.visualize_every_n_steps_eval == 0:
            visualize_feature_map(batch, output, os.path.join(cfg.output_dir, "figures"), prefix=prefix)


# ============ Loader / Dataset Helpers ============


def build_loader(cfg):
    train_scenes = []
    val_scenes = []
    for dataset_cfg in cfg.datasets.values():
        # Process training scenes
        for scene in dataset_cfg.train_scenes:
            ds = hydra.utils.instantiate(dataset_cfg, scene=scene, augment=True, n_poses=cfg.train_n_poses)
            train_scenes.append(ds)

        # Process validation scenes
        for scene in dataset_cfg.val_scenes:
            ds = hydra.utils.instantiate(dataset_cfg, scene=scene, augment=False, n_poses=cfg.val_n_poses)
            if cfg.max_frames_per_val_scene and len(ds) > cfg.max_frames_per_val_scene:
                idx = np.random.choice(len(ds), cfg.max_frames_per_val_scene, replace=False)
                ds = SubsetWithAttributes(ds, idx)
            val_scenes.append(ds)

    train_dataset = ConcatDataset(train_scenes)
    val_dataset = ConcatDataset(val_scenes)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=safe_collate, num_workers=0, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=safe_collate, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader


# ============ Main Training Function ============


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"), resolve=True)

    seed_everything(cfg.seed)

    # Load dataloaders
    train_loader, val_loader = build_loader(cfg)

    # Load model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.ckpt_path:
        load_checkpoint(cfg.ckpt_path, model)
        logger.info(f"Loaded model from {cfg.ckpt_path}")
    model = model.to(device=cfg.device)

    # Load criterion, optimizer, scheduler
    criterion = hydra.utils.instantiate(cfg.criterion).to(device=cfg.device)
    optim_ctor = hydra.utils.instantiate(cfg.optimizer, _partial_=True)
    optimizer = optim_ctor(
        params=[
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
            {"params": criterion.parameters(), "lr": 0.1 * cfg.optimizer.lr, "weight_decay": 0.0},
        ]
    )
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    scaler = torch.amp.GradScaler() if cfg.amp else None

    # configuration
    if cfg.wandb:
        dataset_names = "_".join([ds_cfg.name for ds_cfg in cfg.datasets.values()])
        wandb.init(
            project="BEV-Patch-PF",
            entity="ut-amrl-domlee",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.output_dir,
            name=f"{dataset_names}@{model.name}",
        )

    # --- Training loop ---
    for epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, scheduler, cfg, epoch, scaler)

        # validation step
        val_results = evaluate(model, val_loader, cfg=cfg, prefix=f"val_e{epoch}")

        # save model
        val_metric = val_results.get("z_sep", 0)
        ckpt_path = os.path.join(cfg.output_dir, f"{model.name}_e{epoch}_{val_metric:.2f}.pth")
        save_checkpoint(ckpt_path, model, criterion, optimizer, scheduler, epoch)

        gc.collect()
        torch.cuda.empty_cache()

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
