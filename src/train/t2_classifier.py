"""Task 2a - four-way corruption classifier trained on balanced runtime-corrupted batches.

    uv run python -m src.train.t2_classifier tune
    uv run python -m src.train.t2_classifier train
"""
from __future__ import annotations

import argparse
import math
import time

import mlflow
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.data import pets
from src.models.classifier import CorruptionClassifier
from src.train.common import CKPT_DIR, device, load_config, make_loader, seed_everything, setup_mlflow, study_storage
from src.tune.space import best_or_defaults, suggest

CFG = load_config("t2")
C = CFG["classifier"]


def parse_channels(s: str) -> tuple[int, ...]:
    return tuple(int(c) for c in s.split("-"))


@torch.no_grad()
def predict_loader(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for x, _, lab, _ in loader:
        ps.append(F.softmax(model(x.to(dev)).float(), 1).cpu())
        ys.append(lab)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train_classifier(hp: dict, epochs: int, trial=None, patience=None):
    dev = device()
    model = CorruptionClassifier(parse_channels(hp["channels"]), hp["dropout"]).to(dev)
    train_ds = pets.RuntimeCorruptionDataset(pets.load("train"), hflip=True)
    train_dl = make_loader(train_ds, batch_sampler=pets.BalancedBatchSampler(len(train_ds), hp["batch_size"]))
    val_dl = make_loader(pets.ManifestDataset(pets.load("val"), pets.MANIFEST_DIR / "val_manifest.json"), batch_size=128, workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=epochs * len(train_dl), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, bad, best_state, best_m = -math.inf, 0, None, None
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for x, _, lab in train_dl:
            x, lab = x.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), lab)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(lab)
            correct += (logits.argmax(1) == lab).sum().item()
            n += len(lab)
        y, p = predict_loader(model, val_dl, dev)
        val_loss = float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
        m = {"val_acc": float((p.argmax(1) == y).mean()), "val_macro_f1": float(f1_score(y, p.argmax(1), average="macro")), "val_loss": val_loss}
        mlflow.log_metrics({"train_loss": tot / n, "train_acc": correct / n, **m, "epoch_time_s": time.time() - t0}, step=ep)
        print(f"[clf] ep {ep:3d} train_loss {tot / n:.4f} val_acc {m['val_acc']:.4f} f1 {m['val_macro_f1']:.4f}")
        if m["val_macro_f1"] > best:
            best, bad, best_m = m["val_macro_f1"], 0, m
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if trial is not None:
            trial.report(m["val_macro_f1"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if patience and bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_m


def tune(n_trials=None):
    setup_mlflow(CFG["experiment"])
    t = C["tune"]
    study = optuna.create_study(study_name=C["study_name"], storage=study_storage(C["study_name"]), direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(**t["pruner"]))

    def objective(trial):
        hp = suggest(trial, C["search_space"])
        seed_everything(42)
        with mlflow.start_run(run_name=f"clf_trial_{trial.number}", tags={"stage": "optuna", "study": C["study_name"]}):
            mlflow.log_params(hp)
            _, m = train_classifier(hp, t["epochs"], trial=trial)
            mlflow.log_metric("objective", m["val_macro_f1"])
            return m["val_macro_f1"]

    study.optimize(objective, n_trials=n_trials or t["n_trials"], gc_after_trial=True)
    print("best", study.best_value, study.best_params)


def train(epochs=None):
    setup_mlflow(CFG["experiment"])
    hp, from_study = best_or_defaults(C["study_name"], study_storage(C["study_name"]), C["defaults"])
    seed_everything(42)
    with mlflow.start_run(run_name="t2_classifier_final", tags={"stage": "final", "hp_source": "optuna" if from_study else "defaults"}):
        mlflow.log_params(hp)
        model, m = train_classifier(hp, epochs or C["train"]["epochs"], patience=C["train"]["patience"])
        CKPT_DIR.mkdir(exist_ok=True)
        path = CKPT_DIR / "t2_classifier.pt"
        torch.save({"state_dict": model.state_dict(), "config": model.config, "hp": hp, "val": m}, path)
        mlflow.log_artifact(str(path), artifact_path="checkpoints")
        print(m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "train"])
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    tune(a.n_trials) if a.cmd == "tune" else train(a.epochs)
