"""Export every inference model to ONNX and verify parity with PyTorch.

    uv run python -m src.export.export_all              # everything that has a checkpoint
    uv run python -m src.export.export_all --only t1 t4

Outputs models/*.onnx, models/parity_<task>.json and models/model_info.json
(metadata the backend reads: class order, tau, style names, ...).
"""
from __future__ import annotations

import argparse
import json

import torch

from src.data.corruptions import CLASSES
from src.export.onnx_utils import MODELS_DIR, export, verify, write_report
from src.models.cgan import UNetGenerator
from src.models.classifier import CorruptionClassifier
from src.models.dae import ConvDAE
from src.models.moe import EXPERT_ORDER
from src.train.common import CKPT_DIR


def _batch(n=4, seed=0):
    return torch.rand(n, 3, 128, 128, generator=torch.Generator().manual_seed(seed))


def _real_batch(n=4):
    """Real corrupted validation images (more representative than noise for parity)."""
    try:
        from src.data import pets

        ds = pets.ManifestDataset(pets.load("val"), pets.MANIFEST_DIR / "val_manifest.json")
        return torch.stack([ds[i][0] for i in range(n)])
    except Exception:
        return _batch(n)


def _dae(path):
    ck = torch.load(path, map_location="cpu")
    m = ConvDAE(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def export_t1(info):
    m, ck = _dae(CKPT_DIR / "t1_udae_final.pt")
    x = _real_batch()
    p = export(m, (x,), "t1_universal_dae", ["input"], ["restored"])
    info["t1_universal_dae"] = {"file": p.name, "config": ck["config"], "latent_dim": m.latent_dim}
    return [verify(m, p, (x,)), verify(m, p, (_batch(2, 1),))]


def export_t2(info):
    reports = []
    ck = torch.load(CKPT_DIR / "t2_classifier.pt", map_location="cpu")
    clf = CorruptionClassifier(tuple(ck["config"]["channels"]), ck["config"]["dropout"])
    clf.load_state_dict(ck["state_dict"])
    clf.eval()
    x = _real_batch()
    p = export(clf, (x,), "t2_classifier", ["input"], ["logits"])
    info["t2_classifier"] = {"file": p.name, "classes": CLASSES, "config": ck["config"]}
    reports.append(verify(clf, p, (x,)))
    for k in EXPERT_ORDER:
        m, ck = _dae(CKPT_DIR / f"t2_specialist_{k}.pt")
        p = export(m, (x,), f"t2_specialist_{k}", ["input"], ["restored"])
        info[f"t2_specialist_{k}"] = {"file": p.name, "config": ck["config"]}
        reports.append(verify(m, p, (x,)))
    return reports


def export_t3(info):
    from src.train.t3_moe import load_moe

    m = load_moe()
    x = _real_batch()
    p = export(m, (x,), "t3_soft_moe", ["input"], ["restored", "weights", "logits"])
    info["t3_soft_moe"] = {"file": p.name, "branches": ["identity"] + EXPERT_ORDER, "tau": float(m.tau)}
    return [verify(m, p, (x,))]


def export_t4(info):
    ck = torch.load(CKPT_DIR / "t4_generator.pt", map_location="cpu")
    g = UNetGenerator(**ck["config"])
    g.load_state_dict(ck["state_dict"])
    g.eval()
    x = _batch(3) * 2 - 1
    s = torch.tensor([0, 1, 2])
    p = export(g, (x, s), "t4_generator", ["photo", "style"], ["sketch"])
    info["t4_generator"] = {"file": p.name, "styles": ["Style 1", "Style 2", "Style 3"], "input_range": [-1, 1], "config": ck["config"]}
    return [verify(g, p, (x, s))]


TASKS = {"t1": export_t1, "t2": export_t2, "t3": export_t3, "t4": export_t4}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(TASKS))
    args = ap.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    info_path = MODELS_DIR / "model_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    for name in args.only or TASKS:
        try:
            write_report(TASKS[name](info), name)
        except FileNotFoundError as e:
            print(f"skip {name}: missing checkpoint ({e.filename})")
    info_path.write_text(json.dumps(info, indent=1))
