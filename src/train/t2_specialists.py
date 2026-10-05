"""Task 2b - three specialist DAEs (salt-and-pepper, blur, occlusion).

A single shared Optuna search picks a common architecture/hyper-parameters
(objective = mean validation score of the three experts, each trained on its own
corruption only). The final experts are then trained independently.

    uv run python -m src.train.t2_specialists tune
    uv run python -m src.train.t2_specialists train            # all three
    uv run python -m src.train.t2_specialists train --kind blur
"""
from __future__ import annotations

import argparse

import mlflow
import numpy as np
import optuna
import torch

from src.train.common import CKPT_DIR, load_config, seed_everything, setup_mlflow, study_storage
from src.train.dae_trainer import train_dae
from src.tune.space import best_or_defaults, remaining_trials, suggest

CFG = load_config("t2")
S = CFG["specialists"]


def tune(n_trials=None):
    setup_mlflow(CFG["experiment"])
    t = S["tune"]
    study = optuna.create_study(study_name=S["study_name"], storage=study_storage(S["study_name"]), direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(**t["pruner"]))

    def objective(trial):
        hp = suggest(trial, S["search_space"]) | S["fixed"]
        scores = []
        with mlflow.start_run(run_name=f"spec_trial_{trial.number}", tags={"stage": "optuna", "study": S["study_name"]}):
            mlflow.log_params(hp)
            for kind in S["kinds"]:
                seed_everything(42)
                with mlflow.start_run(run_name=f"spec_trial_{trial.number}_{kind}", nested=True):
                    # Pruning is applied on the first expert only (later experts share the step axis).
                    _, m = train_dae(hp, [kind], t["epochs"], trial=trial if kind == S["kinds"][0] else None, out_prefix=f"spec_{kind}_t{trial.number}")
                scores.append(m["score"])
                mlflow.log_metric(f"score_{kind}", m["score"])
            obj = float(np.mean(scores))
            mlflow.log_metric("objective", obj)
            return obj

    if not study.trials:
        for params in S.get("enqueue", []):
            study.enqueue_trial(params)
    study.optimize(objective, n_trials=remaining_trials(study, n_trials or t["n_trials"]), gc_after_trial=True)
    print("best", study.best_value, study.best_params)


def train(kinds, epochs=None):
    setup_mlflow(CFG["experiment"])
    hp, from_study = best_or_defaults(S["study_name"], study_storage(S["study_name"]), S["defaults"])
    hp |= S["fixed"]
    CKPT_DIR.mkdir(exist_ok=True)
    for kind in kinds:
        seed_everything(42)
        with mlflow.start_run(run_name=f"t2_specialist_{kind}", tags={"stage": "final", "expert": kind, "hp_source": "optuna" if from_study else "defaults"}):
            mlflow.log_params(hp | {"kind": kind})
            model, m = train_dae(hp, [kind], epochs or S["train"]["epochs"], out_prefix=f"t2_specialist_{kind}", patience=S["train"]["patience"])
            path = CKPT_DIR / f"t2_specialist_{kind}.pt"
            torch.save({"state_dict": model.state_dict(), "config": model.config, "hp": hp, "kind": kind, "val": m}, path)
            mlflow.log_artifact(str(path), artifact_path="checkpoints")
            print(kind, m["psnr"], m["ssim"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "train"])
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--kind", choices=S["kinds"])
    a = ap.parse_args()
    tune(a.n_trials) if a.cmd == "tune" else train([a.kind] if a.kind else S["kinds"], a.epochs)
