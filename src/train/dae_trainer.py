"""Generic train / validate loop for denoising autoencoders (Task 1 and Task 2 specialists)."""
from __future__ import annotations

import math
import time
from collections import defaultdict

import mlflow
import optuna
import torch

from src.data import pets
from src.data.corruptions import CLASSES
from src.losses.ssim_l1 import l1_ssim_loss, per_image_metrics
from src.models.dae import ConvDAE
from src.train.common import close_loaders, device, error_map, make_loader, save_grid


def val_score(psnr: float, ssim: float) -> float:
    """Validation objective combining reconstruction quality and structure.

    PSNR is mapped to roughly [0, 1] by dividing by 40 dB, then averaged with SSIM.
    """
    return 0.5 * ssim + 0.5 * psnr / 40.0


def build_loaders(kinds: list[str], batch_size: int, val_manifest=None):
    train_ds = pets.RuntimeCorruptionDataset(pets.load("train"), kinds=kinds, hflip=True)
    sampler = pets.BalancedBatchSampler(len(train_ds), batch_size, kinds=kinds)
    manifest = val_manifest or pets.MANIFEST_DIR / "val_manifest.json"
    # Specialists are validated only on their own corruption type.
    val_ds = pets.ManifestDataset(pets.load("val"), manifest, kinds=None if len(kinds) == 4 else kinds)
    return make_loader(train_ds, batch_sampler=sampler), make_loader(val_ds, batch_size=64, workers=0)


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    model.eval()
    sums, counts = defaultdict(float), defaultdict(int)
    for x, y, lab, _ in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        m = per_image_metrics(model(x).float(), y)
        for k, v in m.items():
            sums[k] += v.sum().item()
            for c in lab.unique().tolist():
                sel = (lab == c).to(dev)
                sums[f"{k}_{CLASSES[c]}"] += v[sel].sum().item()
        counts["all"] += len(lab)
        for c in lab.tolist():
            counts[CLASSES[c]] += 1
    out = {}
    for k, v in sums.items():
        parts = k.split("_", 1)
        out[k] = v / counts["all" if len(parts) == 1 else parts[1]]
    out["score"] = val_score(out["psnr"], out["ssim"])
    return out


@torch.no_grad()
def sample_grid(model, loader, dev, path, n: int = 8):
    model.eval()
    x, y, _, _ = next(iter(loader))
    x, y = x[:n].to(dev), y[:n].to(dev)
    p = model(x).float()
    rows = [[y[i], x[i], p[i], error_map(p[i], y[i]).to(dev)] for i in range(len(x))]
    return save_grid(rows, path)


def train_dae(
    hp: dict,
    kinds: list[str],
    epochs: int,
    trial: optuna.Trial | None = None,
    log_every: int = 5,
    out_prefix: str = "dae",
    patience: int | None = None,
) -> tuple[ConvDAE, dict]:
    """Train a ConvDAE with hyper-parameters ``hp``; logs to the active MLflow run.

    Returns the best model (by validation score) and its validation metrics.
    """
    dev = device()
    model = ConvDAE(hp["base_ch"], hp["latent_ch"], hp["dropout"], hp.get("skip_levels", 0), hp.get("latent_res", 8)).to(dev)
    train_dl, val_dl = build_loaders(kinds, hp["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay", 1e-5))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=epochs * len(train_dl), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, best_state, bad = -math.inf, None, 0
    try:
        best_state, best_metrics = _fit(model, opt, sched, scaler, train_dl, val_dl, dev, hp, epochs, trial, log_every, out_prefix, patience)
    finally:
        close_loaders(train_dl, val_dl)
    model.load_state_dict(best_state)
    return model, best_metrics


def _fit(model, opt, sched, scaler, train_dl, val_dl, dev, hp, epochs, trial, log_every, out_prefix, patience):
    best, best_state, best_metrics, bad = -math.inf, None, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), defaultdict(float), 0
        for x, y, _ in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                pred = model(x)
            loss, parts = l1_ssim_loss(pred.float(), y, hp["alpha"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot["loss"] += loss.item()
            tot["l1"] += parts["l1"]
            tot["ssim"] += parts["ssim"]
            n += 1
        val = evaluate(model, val_dl, dev)
        metrics = {f"train_{k}": v / n for k, v in tot.items()} | {f"val_{k}": v for k, v in val.items()}
        metrics["epoch_time_s"] = time.time() - t0
        metrics["lr"] = sched.get_last_lr()[0]
        mlflow.log_metrics(metrics, step=ep)
        print(f"[{out_prefix}] ep {ep:3d} loss {metrics['train_loss']:.4f} val psnr {val['psnr']:.2f} ssim {val['ssim']:.4f} score {val['score']:.4f} ({metrics['epoch_time_s']:.0f}s)")

        if val["score"] > best:
            best, bad = val["score"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = val
        else:
            bad += 1
        if trial is not None:
            trial.report(val["score"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if trial is None and (ep % log_every == 0 or ep == epochs):
            from src.train.common import OUT_DIR

            p = sample_grid(model, val_dl, dev, OUT_DIR / out_prefix / f"val_ep{ep:03d}.png")
            mlflow.log_artifact(str(p), artifact_path="val_samples")
        if patience and bad >= patience:
            print(f"[{out_prefix}] early stop at epoch {ep}")
            break
    return best_state, best_metrics
