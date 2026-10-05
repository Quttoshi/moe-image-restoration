"""Shared training utilities: seeding, device, MLflow setup, loaders, image grids."""
from __future__ import annotations

import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints"
STUDY_DIR = ROOT / "optuna_studies"
OUT_DIR = ROOT / "outputs"


def load_config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_mlflow(experiment: str) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlruns" / "artifacts" / experiment).as_uri())
    mlflow.set_experiment(experiment)


def study_storage(name: str) -> str:
    STUDY_DIR.mkdir(exist_ok=True)
    return f"sqlite:///{(STUDY_DIR / f'{name}.db').as_posix()}"


def make_loader(ds, batch_size: int | None = None, shuffle: bool = False, batch_sampler=None, workers: int | None = None) -> DataLoader:
    # Each spawned worker re-imports torch+CUDA (~2 GB commit on Windows), so keep this small.
    workers = int(os.environ.get("NUM_WORKERS", 2)) if workers is None else workers
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    if batch_sampler is not None:
        return DataLoader(ds, batch_sampler=batch_sampler, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kw)


def error_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute error averaged over channels, rendered as a 3-channel 'hot' heatmap."""
    import matplotlib

    err = (pred - target).abs().mean(0).clamp(0, 1).cpu().numpy()
    rgb = matplotlib.colormaps["inferno"](np.clip(err * 3, 0, 1))[..., :3]
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def save_grid(rows: list[list[torch.Tensor]], path: Path, pad: int = 2) -> Path:
    """Save a grid where each row is a list of CHW tensors (same size)."""
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [t.detach().cpu().clamp(0, 1) for r in rows for t in r]
    save_image(torch.stack(flat), path, nrow=len(rows[0]), padding=pad, pad_value=1.0)
    return path


def close_loaders(*loaders) -> None:
    """Shut down persistent DataLoader workers now instead of waiting for garbage collection.

    Without this, sequential trainings inside one Optuna trial (e.g. three specialists)
    accumulate idle worker processes and exhaust Windows commit memory.
    """
    import gc

    for dl in loaders:
        it = getattr(dl, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
        if dl is not None:
            dl._iterator = None
    gc.collect()
