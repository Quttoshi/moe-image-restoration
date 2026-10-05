"""Task 1 - Universal multi-corruption denoising autoencoder.

    uv run python -m src.train.t1_udae tune      # Optuna study (SQLite in optuna_studies/)
    uv run python -m src.train.t1_udae train     # full training with the best trial
    uv run python -m src.train.t1_udae train --skip-levels 1 --tag skip1   # ablation
"""
from __future__ import annotations

import argparse
import json

import mlflow
import optuna
import torch

from src.data.corruptions import CLASSES
from src.train.common import CKPT_DIR, load_config, seed_everything, setup_mlflow, study_storage
from src.train.dae_trainer import train_dae
from src.tune.space import best_or_defaults, suggest

CFG = load_config("t1")


def tune(n_trials: int | None) -> None:
    setup_mlflow(CFG["experiment"])
    storage = study_storage(CFG["study_name"])
    t = CFG["tune"]
    study = optuna.create_study(
        study_name=CFG["study_name"],
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(**t["pruner"]),
    )

    def objective(trial: optuna.Trial) -> float:
        hp = suggest(trial, CFG["search_space"]) | {"skip_levels": CFG["train"]["skip_levels"]}
        seed_everything(42)
        with mlflow.start_run(run_name=f"trial_{trial.number}", tags={"stage": "optuna", "study": CFG["study_name"]}):
            mlflow.log_params(hp)
            try:
                _, m = train_dae(hp, list(CLASSES), t["epochs"], trial=trial, out_prefix=f"t1_trial{trial.number}")
            except optuna.TrialPruned:
                mlflow.set_tag("state", "pruned")
                raise
            mlflow.log_metric("objective", m["score"])
            return m["score"]

    study.optimize(objective, n_trials=n_trials or t["n_trials"], gc_after_trial=True)
    print("best trial", study.best_trial.number, study.best_value, study.best_params)


def train(skip_levels: int | None, tag: str, epochs: int | None) -> None:
    setup_mlflow(CFG["experiment"])
    hp, from_study = best_or_defaults(CFG["study_name"], study_storage(CFG["study_name"]), CFG["defaults"])
    hp["skip_levels"] = CFG["train"]["skip_levels"] if skip_levels is None else skip_levels
    hp["weight_decay"] = CFG["train"]["weight_decay"]
    seed_everything(42)
    name = f"t1_udae_{tag}"
    with mlflow.start_run(run_name=name, tags={"stage": "final", "hp_source": "optuna" if from_study else "defaults"}):
        mlflow.log_params(hp)
        model, m = train_dae(hp, list(CLASSES), epochs or CFG["train"]["epochs"], out_prefix=name, patience=CFG["train"]["patience"])
        CKPT_DIR.mkdir(exist_ok=True)
        path = CKPT_DIR / f"{name}.pt"
        torch.save({"state_dict": model.state_dict(), "config": model.config, "hp": hp, "val": m}, path)
        mlflow.log_artifact(str(path), artifact_path="checkpoints")
        mlflow.log_dict({"hp": hp, "val": m}, "final_summary.json")
        print(json.dumps(m, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tune")
    p.add_argument("--n-trials", type=int)
    p = sub.add_parser("train")
    p.add_argument("--skip-levels", type=int)
    p.add_argument("--tag", default="final")
    p.add_argument("--epochs", type=int)
    a = ap.parse_args()
    tune(a.n_trials) if a.cmd == "tune" else train(a.skip_levels, a.tag, a.epochs)
