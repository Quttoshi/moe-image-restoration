"""Task 3 - soft mixture-of-experts: gate warm-up with frozen experts, then joint fine-tuning.

    uv run python -m src.train.t3_moe tune
    uv run python -m src.train.t3_moe train

Requires checkpoints/t2_classifier.pt and checkpoints/t2_specialist_{salt_pepper,blur,occlusion}.pt.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import mlflow
import numpy as np
import optuna
import torch
import torch.nn.functional as F

from src.data import pets
from src.losses.ssim_l1 import l1_ssim_loss, per_image_metrics
from src.models.classifier import CorruptionClassifier
from src.models.dae import ConvDAE
from src.models.moe import EXPERT_ORDER, SoftMoE, balance_loss
from src.train.common import CKPT_DIR, device, load_config, make_loader, seed_everything, setup_mlflow, study_storage
from src.train.dae_trainer import val_score
from src.tune.space import best_or_defaults, remaining_trials, suggest

CFG = load_config("t3")


def load_components() -> tuple[CorruptionClassifier, dict[str, ConvDAE]]:
    ck = torch.load(CKPT_DIR / "t2_classifier.pt", map_location="cpu")
    gate = CorruptionClassifier(tuple(ck["config"]["channels"]), ck["config"]["dropout"])
    gate.load_state_dict(ck["state_dict"])
    experts = {}
    for k in EXPERT_ORDER:
        ck = torch.load(CKPT_DIR / f"t2_specialist_{k}.pt", map_location="cpu")
        experts[k] = ConvDAE(**ck["config"])
        experts[k].load_state_dict(ck["state_dict"])
    return gate, experts


@torch.no_grad()
def evaluate(model: SoftMoE, loader, dev) -> dict:
    model.eval()
    s, n = defaultdict(float), 0
    wsum = torch.zeros(4)
    for x, y, lab, _ in loader:
        x, y = x.to(dev), y.to(dev)
        out, w, _ = model(x)
        m = per_image_metrics(out.float(), y)
        s["psnr"] += m["psnr"].sum().item()
        s["ssim"] += m["ssim"].sum().item()
        s["route_acc"] += (w.argmax(1).cpu() == lab).sum().item()
        wsum += w.sum(0).cpu()
        n += len(lab)
    r = {k: v / n for k, v in s.items()}
    r["score"] = val_score(r["psnr"], r["ssim"])
    for i, name in enumerate(["identity"] + EXPERT_ORDER):
        r[f"wbar_{name}"] = (wsum[i] / n).item()
    r["wbar_min"] = (wsum / n).min().item()
    return r


def run_epochs(model, opt, train_dl, val_dl, hp, epochs, phase, step0, dev, trial=None):
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, best_state, best_m, bad = -math.inf, None, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        model.set_experts_trainable(phase == "joint")
        t0, tot, n = time.time(), defaultdict(float), 0
        for x, y, lab in train_dl:
            x, y, lab = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                out, w, logits = model(x)
            rec, parts = l1_ssim_loss(out.float(), y, hp["lam_1"])        # lam_1*L1 + (1-lam_1)*(1-SSIM)
            ce = F.cross_entropy(logits.float() / model.tau, lab)          # = -log w_label
            bal = balance_loss(w.float())
            loss = rec + hp["lam_c"] * ce + hp["lam_b"] * bal
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            for k, v in (("loss", loss.item()), ("rec", rec.item()), ("ce", ce.item()), ("balance", bal.item()), ("l1", parts["l1"]), ("ssim", parts["ssim"])):
                tot[k] += v
            n += 1
        val = evaluate(model, val_dl, dev)
        step = step0 + ep
        mlflow.log_metrics({f"train_{k}": v / n for k, v in tot.items()} | {f"val_{k}": v for k, v in val.items()} | {"epoch_time_s": time.time() - t0, "phase_joint": float(phase == "joint")}, step=step)
        print(f"[moe:{phase}] ep {ep} loss {tot['loss'] / n:.4f} val psnr {val['psnr']:.2f} ssim {val['ssim']:.4f} route_acc {val['route_acc']:.3f} wbar_min {val['wbar_min']:.3f}")
        if trial is not None and phase == "joint":
            trial.report(val["score"], step)
            if val["wbar_min"] < CFG["fixed"]["collapse_threshold"]:
                trial.set_user_attr("pruned_reason", "routing_collapse")
                raise optuna.TrialPruned()
            if trial.should_prune():
                raise optuna.TrialPruned()
        if val["score"] > best:
            best, best_m, bad = val["score"], val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if phase == "joint" and trial is None and bad >= CFG["train"]["patience"]:
            break
    model.load_state_dict(best_state)
    return best_m, step0 + epochs


def train_moe(hp: dict, joint_epochs: int, trial=None):
    dev = device()
    gate, experts = load_components()
    model = SoftMoE(gate, experts, hp["tau"]).to(dev)
    train_ds = pets.RuntimeCorruptionDataset(pets.load("train"), hflip=True)
    train_dl = make_loader(train_ds, batch_sampler=pets.BalancedBatchSampler(len(train_ds), hp["batch_size"]))
    val_dl = make_loader(pets.ManifestDataset(pets.load("val"), pets.MANIFEST_DIR / "val_manifest.json"), batch_size=32, workers=0)

    base = evaluate(model, val_dl, dev)
    mlflow.log_metrics({f"init_{k}": v for k, v in base.items()})
    print("[moe] before warm-up:", {k: round(v, 4) for k, v in base.items()})

    # Stage 1: warm-up - experts frozen, only the gate learns.
    model.set_experts_trainable(False)
    opt = torch.optim.AdamW(model.gate.parameters(), lr=hp["lr_gate"], weight_decay=hp["weight_decay"])
    _, step = run_epochs(model, opt, train_dl, val_dl, hp, hp["warmup_epochs"], "warmup", 0, dev, trial)

    # Stage 2: joint fine-tuning of gate + experts with a smaller learning rate.
    model.set_experts_trainable(True)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr_joint"], weight_decay=hp["weight_decay"])
    m, _ = run_epochs(model, opt, train_dl, val_dl, hp, joint_epochs, "joint", step, dev, trial)
    return model, m


def tune(n_trials=None):
    setup_mlflow(CFG["experiment"])
    t = CFG["tune"]
    study = optuna.create_study(study_name=CFG["study_name"], storage=study_storage(CFG["study_name"]), direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(**t["pruner"]))

    def objective(trial):
        hp = suggest(trial, CFG["search_space"]) | CFG["fixed"]
        seed_everything(42)
        with mlflow.start_run(run_name=f"moe_trial_{trial.number}", tags={"stage": "optuna", "study": CFG["study_name"]}):
            mlflow.log_params(hp)
            _, m = train_moe(hp, t["joint_epochs"], trial=trial)
            mlflow.log_metric("objective", m["score"])
            return m["score"]

    study.optimize(objective, n_trials=remaining_trials(study, n_trials or t["n_trials"]), gc_after_trial=True)
    print("best", study.best_value, study.best_params)


def train(epochs=None):
    setup_mlflow(CFG["experiment"])
    hp, from_study = best_or_defaults(CFG["study_name"], study_storage(CFG["study_name"]), CFG["defaults"])
    hp |= CFG["fixed"]
    seed_everything(42)
    with mlflow.start_run(run_name="t3_moe_final", tags={"stage": "final", "hp_source": "optuna" if from_study else "defaults"}):
        mlflow.log_params(hp)
        model, m = train_moe(hp, epochs or CFG["train"]["joint_epochs"])
        path = CKPT_DIR / "t3_moe.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "gate_config": model.gate.config,
            "expert_configs": {k: e.config for k, e in zip(EXPERT_ORDER, model.experts)},
            "tau": float(model.tau),
            "hp": hp,
            "val": m,
        }, path)
        mlflow.log_artifact(str(path), artifact_path="checkpoints")
        print(m)


def load_moe(path=None) -> SoftMoE:
    ck = torch.load(path or CKPT_DIR / "t3_moe.pt", map_location="cpu")
    gate = CorruptionClassifier(tuple(ck["gate_config"]["channels"]), ck["gate_config"]["dropout"])
    experts = {k: ConvDAE(**c) for k, c in ck["expert_configs"].items()}
    m = SoftMoE(gate, experts, ck["tau"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "train"])
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    tune(a.n_trials) if a.cmd == "tune" else train(a.epochs)
