import gc
import json
import logging
import os
from functools import partial

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm as _tqdm

from dataset.utils import SubsetWithAttributes, safe_collate
from metrics.simple_retrieval import SimpleRetrievalMetric
from utils.misc import load_checkpoint, save_checkpoint
from utils.visualization import visualize_feature_map

logger = logging.getLogger(__name__)
tqdm = partial(_tqdm, dynamic_ncols=True)

# https://github.com/huggingface/accelerate/issues/2174
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"


# ============ Training Pipeline ============


def train_step(batch, model, criterion, optimizer, accelerator, cfg):
    optimizer.zero_grad(set_to_none=True)

    # forward
    with accelerator.autocast():
        output = model(**batch)
        losses = criterion(**output)
        output.update({"temp": losses.get("temp")})

    # backward
    accelerator.backward(losses["loss"])

    if cfg.clip_grad_norm > 0:
        params = list(model.parameters()) + list(criterion.parameters())
        accelerator.clip_grad_norm_(params, cfg.clip_grad_norm)

    optimizer.step()

    return output, losses


def train_one_epoch(model, loader, criterion, optimizer, scheduler, accelerator, cfg, epoch):
    model.train()

    metric = SimpleRetrievalMetric(ks=(1, 5, 10))

    is_main = accelerator.is_main_process
    for idx, batch in enumerate(tqdm(loader, desc=f"Train [{epoch}/{cfg.epochs}]", disable=not is_main)):
        step = epoch * len(loader) + idx

        # train step
        output, losses = train_step(batch, model, criterion, optimizer, accelerator, cfg)
        scheduler.step(epoch + (idx + 1) / len(loader))  # fractional epoch

        # log metrics
        batch_results = metric.update(
            score=accelerator.gather_for_metrics(output["score"].detach()),
            uncertainty=accelerator.gather_for_metrics(output["uncertainty"].detach()),
            distinct_logits=accelerator.gather_for_metrics(output["distinct_logits"].detach()),
        )
        metric.reset()

        accelerator.log({f"train_metrics/{k}": v for k, v in batch_results.items()}, step=step)
        accelerator.log({f"train/{k}": v for k, v in losses.items()}, step=step)
        accelerator.log({"train/lr": optimizer.param_groups[0]["lr"]}, step=step)

        # (debug) visualization
        if is_main and step % cfg.visualize_every_n_steps_train == 0:
            visualize_feature_map(batch, output, os.path.join(cfg.output_dir, "figures"), prefix=f"e{epoch}_s{step}")


# ============ Evaluation Pipeline ============


@torch.inference_mode()
def evaluate(model, loader, accelerator, cfg, metrics=None, prefix: str = "eval") -> None:
    model.eval()

    if metrics is None:
        metrics = [SimpleRetrievalMetric(ks=(1, 5, 10))]

    for i, batch in enumerate(tqdm(loader, disable=not accelerator.is_main_process)):
        with accelerator.autocast():
            output = model(**batch)

        # log metrics
        for metric in metrics:
            metric.update(
                score=accelerator.gather_for_metrics(output["score"].detach()),
                uncertainty=accelerator.gather_for_metrics(output["uncertainty"].detach()),
                distinct_logits=accelerator.gather_for_metrics(output["distinct_logits"].detach()),
            )

        # (debug) visualization
        if accelerator.is_main_process and i % cfg.visualize_every_n_steps_eval == 0:
            visualize_feature_map(batch, output, os.path.join(cfg.output_dir, "figures"), prefix=prefix)

    # log final metrics
    results = {}
    for metric in metrics:
        metric_result = metric.compute()
        results.update(metric_result)
        accelerator.log({f"eval_metrics/{k}": v for k, v in metric_result.items()})
        metric.reset()

    with open(os.path.join(cfg.output_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results


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
    accelerator = Accelerator(mixed_precision=("fp16" if cfg.amp else "no"), log_with="wandb")
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)
    set_seed(int(cfg.seed), device_specific=True)

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"), resolve=True)

    # dataloaders
    train_loader, val_loader = build_loader(cfg)

    # model / criterion / optimizer / scheduler
    model = hydra.utils.instantiate(cfg.model)
    criterion = hydra.utils.instantiate(cfg.criterion)
    optim_ctor = hydra.utils.instantiate(cfg.optimizer, _partial_=True)
    optimizer = optim_ctor(
        params=[
            {"params": [p for p in model.parameters() if p.requires_grad]},
            {"params": criterion.parameters(), "lr": 0.1 * cfg.optimizer.lr, "weight_decay": 0.0},
        ]
    )
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    # resume from checkpoint
    start_epoch, resumed = 0, False
    if cfg.ckpt_path:
        try:
            start_epoch, resumed = load_checkpoint(cfg.ckpt_path, model, criterion, optimizer, scheduler)
            msg = "Resumed full state" if resumed else "Loaded weights only"
            logger.info(f"{msg} (start_epoch={start_epoch}) from {cfg.ckpt_path}")
        except Exception as e:
            logger.exception(f"Failed to load ckpt_path={cfg.ckpt_path}: {e}")

    # prepare for accelerator
    model, criterion, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, criterion, optimizer, train_loader, val_loader, scheduler
    )

    # wandb logging
    accelerator.init_trackers(
        "BEV-Patch-PF",
        config=OmegaConf.to_container(cfg),
        init_kwargs={"wandb": {"entity": "ut-amrl-domlee", "name": f"{cfg.dataset_name}@{cfg.model.name}"}},
    )

    # --- Training loop ---
    for epoch in range(start_epoch, cfg.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, scheduler, accelerator, cfg, epoch)

        # validation step
        val_results = evaluate(model, val_loader, accelerator, cfg=cfg, prefix=f"val_e{epoch}")

        # save model
        if accelerator.is_main_process:
            val_metric = val_results.get("z_sep", 0)
            ckpt_path = os.path.join(cfg.output_dir, f"{cfg.model.name}_e{epoch}_{val_metric:.2f}.pth")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_criterion = accelerator.unwrap_model(criterion)
            save_checkpoint(ckpt_path, unwrapped_model, unwrapped_criterion, optimizer, scheduler, epoch)

        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()

    accelerator.end_training()


if __name__ == "__main__":
    main()
