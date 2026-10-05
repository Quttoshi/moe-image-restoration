"""Shared evaluation helpers for the restoration tasks (1-3) on the deterministic test manifest."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import pets
from src.losses.ssim_l1 import per_image_metrics
from src.train.common import device, error_map, make_loader, save_grid


def test_dataset(kinds=None) -> pets.ManifestDataset:
    return pets.ManifestDataset(pets.load("test"), pets.MANIFEST_DIR / "test_manifest.json", kinds=kinds)


@torch.no_grad()
def run(predict, ds: pets.ManifestDataset, batch_size: int = 64, extra=None) -> pd.DataFrame:
    """Run ``predict(x, label) -> (restored, info_dict_of_tensors)`` over the manifest.

    Returns one row per manifest entry with type, severity, input-vs-target and
    output-vs-target metrics plus any per-sample values ``predict`` returns.
    """
    import time

    dev = device()
    cols: dict[str, list] = {}
    # Items cost ~0.35 ms each, so worker processes (2 GB commit each on Windows) are not worth it.
    loader = make_loader(ds, batch_size=batch_size, workers=0)
    t0 = time.time()
    for b, (x, y, lab, idx) in enumerate(loader):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        pred, info = predict(x, lab.to(dev))
        batch = {f"in_{k}": v for k, v in per_image_metrics(x, y).items()}
        batch |= per_image_metrics(pred.float(), y)
        batch |= info
        # One device->host transfer per tensor per batch (not per sample).
        for k, v in batch.items():
            cols.setdefault(k, []).extend(v.detach().cpu().tolist())
        cols.setdefault("entry", []).extend(idx.tolist())
        if b % 100 == 0:
            print(f"  eval batch {b}/{len(loader)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(cols)
    meta = pd.DataFrame([{"image_index": e["image_index"], "type": e["params"]["type"], "severity": e["severity"]} for e in ds.entries])
    df = pd.concat([df[["entry"]], meta.iloc[df["entry"]].reset_index(drop=True), df.drop(columns="entry")], axis=1)
    df["psnr_gain"] = df["psnr"] - df["in_psnr"]
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per corruption type x severity, plus per-type and overall rows."""
    cols = ["in_psnr", "psnr", "psnr_gain", "in_ssim", "ssim", "l1"]
    by_sev = df.groupby(["type", "severity"])[cols].mean()
    by_type = df.groupby("type")[cols].mean().assign(severity="all").set_index("severity", append=True)
    # Clean inputs restored by an identity path have infinite PSNR (capped at 100 dB), so the
    # overall row averages the corrupted conditions only; clean is reported in its own row.
    overall = df[df["type"] != "clean"][cols].mean().to_frame().T.assign(type="overall", severity="corrupted").set_index(["type", "severity"])
    order = {"clean": 0, "salt_pepper": 1, "blur": 2, "occlusion": 3, "overall": 4}
    sev_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "all": 4, "corrupted": 5}
    t = pd.concat([by_sev, by_type, overall])
    t = t[~t.index.duplicated()]
    return t.sort_index(key=lambda s: s.map(order) if s.name == "type" else s.map(sev_order)).round(4)


@torch.no_grad()
def example_grid(predict, ds, entries: list[int], path: Path) -> Path:
    """Rows of [clean target | corrupted input | restored | |error| map]."""
    dev = device()
    rows = []
    for i in entries:
        x, y, lab, _ = ds[i]
        x, y = x[None].to(dev), y[None].to(dev)
        p, _ = predict(x, torch.tensor([lab], device=dev))
        p = p.float()[0]
        rows.append([y[0], x[0], p, error_map(p, y[0]).to(dev)])
    return save_grid(rows, path)


def pick_representative(df: pd.DataFrame, per_cell: int = 1, seed: int = 0) -> list[int]:
    """One random entry per (type, severity) cell -> 10 cells; topped up to >= 12."""
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in df.groupby(["type", "severity"]):
        picks += rng.choice(g["entry"].to_numpy(), size=per_cell, replace=False).tolist()
    extra = df[df["type"] != "clean"].sample(n=max(0, 12 - len(picks)), random_state=seed)["entry"].tolist()
    return picks + extra


def pick_failures(df: pd.DataFrame, n_per_type: int = 2) -> list[int]:
    """Worst restored outputs (lowest SSIM) for each corruption type."""
    out = []
    for t in ["salt_pepper", "blur", "occlusion"]:
        out += df[df["type"] == t].nsmallest(n_per_type, "ssim")["entry"].tolist()
    return out
