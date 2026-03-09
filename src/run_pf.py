import os
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
from manifpy import SE2
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from logger.csv_logger import CSVLogger
from logger.rerun_logger import RerunLogger
from particle_filter import ParticleFilter
from utils.misc import load_model, seed_everything


def create_sequence_loader(cfg: DictConfig, pf: ParticleFilter, scene: str) -> DataLoader:
    sequence = hydra.utils.instantiate(cfg.sequence, scene=scene, augment=False, sample_poses=False)
    pf.observation_model.geo_handler = sequence.geo_handler
    sequence.geo_handler = None
    loader = DataLoader(sequence, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return loader


###
# Main function
###


def run_pf(
    pf: ParticleFilter,
    seq_loader: DataLoader,
    output_path: Path | None,
    cfg: DictConfig,
    *,
    rerun_logger_kwargs: dict | None = None,
):
    pf.reset()

    rerun_logger = RerunLogger(pf, output_path, **(rerun_logger_kwargs or {}))
    traj_logger_context = (
        CSVLogger(output_path / "pf_traj.csv", ["timestamp", "x", "y", "angle"])
        if output_path is not None
        else nullcontext(None)
    )
    with traj_logger_context as traj_logger:
        for i, frame in enumerate(seq_loader):
            gt_xyr = frame["gt_xyr"][0].numpy()

            if not pf.is_initialized:
                pf.initialize(gt_xyr, init_noise_sigma=np.array(cfg.run.init_noise_sigma))

            # particle filtering
            results = {}
            pf.predict(frame["action"][0].numpy())
            pf.update(frame["ground_image"], frame["ground_depth"], frame["info"], results)
            results["is_resampled"] = pf.resample()

            # log particles and pose
            if traj_logger is not None:
                traj_logger.log([frame["timestamp"][0], *pf.pose])

            rerun_logger.log(frame, results)

            var = results.get("var", -1e4)
            alpha = results.get("alpha", 0.0)
            pose_err = SE2(*gt_xyr).between(SE2(*pf.pose))
            t_err = np.linalg.norm(pose_err.translation())

            # resample
            resample_msg = "-> resampled" if results["is_resampled"] else ""

            print(
                f"\r[Frame {i + 1:>5}/{len(seq_loader)}] | alpha: {alpha:.2f} (var: {var:.4f}) | "
                f"error: {t_err:.2f} m {resample_msg:>13}",
                end="" if i < len(seq_loader) - 1 else "\n",
                flush=True,
            )


@hydra.main(version_base="1.3", config_path="../config", config_name="run_pf")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    if cfg.ckpt_path is None:
        raise ValueError("Checkpoint path (ckpt_path) must be provided to run the particle filter.")

    ckpt_path = cfg.ckpt_path
    model = load_model(cfg.model, ckpt_path, device="cuda", strict_model=False, eval_mode=True)
    pf = ParticleFilter(model, **cfg.run)

    # Run Particle Filter on each scene
    for scene in cfg.sequence.scenes:
        loader = create_sequence_loader(cfg, pf, scene)

        # Create output directory
        output_path = Path(cfg.output_dir) / cfg.model.name / Path(ckpt_path).stem / scene
        os.makedirs(output_path, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(output_path, "config.yaml"), resolve=True)

        # Run Particle Filter
        print(f"Running Particle Filter on scene: {scene}")
        run_pf(pf, loader, output_path, cfg)


if __name__ == "__main__":
    main()
