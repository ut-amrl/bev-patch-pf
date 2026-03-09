import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from particle_filter import ParticleFilter
from run_pf import create_sequence_loader, run_pf
from utils.misc import load_model, seed_everything

DEMO_SCENE_ASSETS = ("image", "depth", "timestamps.txt", "utm_pose.csv", "odom.csv")


def configure_demo_logging() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def download_demo_dataset_root(cfg: DictConfig) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        dataset_root = Path(
            snapshot_download(repo_id=cfg.hf_data_repo_id, repo_type="dataset", revision=cfg.hf_data_revision)
        )
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            "Demo dataset repo "
            f"`{cfg.hf_data_repo_id}` was not found or is not accessible at revision `{cfg.hf_data_revision}`. "
            "Publish it with "
            f"`python scripts/publish_hf.py --data-repo-id {cfg.hf_data_repo_id}` "
            "or update `hf_data_repo_id` if you are using a different dataset repo."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download demo dataset from `{cfg.hf_data_repo_id}` at revision `{cfg.hf_data_revision}`."
        ) from exc

    scene_dir = dataset_root / cfg.scene
    required_paths = [
        dataset_root,
        dataset_root / cfg.demo_geo_tiff_name,
        scene_dir,
        *(scene_dir / asset for asset in DEMO_SCENE_ASSETS),
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            f"Demo dataset assets are incomplete under {dataset_root} for scene `{cfg.scene}`.\n"
            f"Missing paths:\n{missing_text}"
        )

    print(f"Downloaded demo dataset to {dataset_root}")
    return dataset_root


def download_demo_checkpoint(cfg: DictConfig) -> str:
    from huggingface_hub import hf_hub_download

    try:
        ckpt_path = hf_hub_download(repo_id=cfg.hf_repo_id, filename=cfg.hf_filename, revision=cfg.hf_revision)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download `{cfg.hf_filename}` from `{cfg.hf_repo_id}` at revision `{cfg.hf_revision}`."
        ) from exc

    print(f"Downloaded demo checkpoint to {ckpt_path}")
    return ckpt_path


@hydra.main(version_base="1.3", config_path="../config", config_name="demo")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    configure_demo_logging()

    dataset_root = download_demo_dataset_root(cfg)
    cfg.sequence.root = str(dataset_root)
    cfg.sequence.geo_tiff_path = str(dataset_root / cfg.demo_geo_tiff_name)
    print(f"Downloaded demo dataset to {dataset_root}")

    ckpt_path = download_demo_checkpoint(cfg)
    print(f"Downloaded demo checkpoint to {ckpt_path}")

    model = load_model(cfg.model, ckpt_path, device="cuda", strict_model=False, eval_mode=True)
    pf = ParticleFilter(model, **cfg.run)
    loader = create_sequence_loader(cfg, pf, cfg.scene)

    print(f"Running demo on scene: {cfg.scene}")
    run_pf(pf, loader, None, cfg, rerun_logger_kwargs=OmegaConf.to_container(cfg.rerun, resolve=True))


if __name__ == "__main__":
    main()
