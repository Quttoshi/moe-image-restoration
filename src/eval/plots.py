"""Report figures: Optuna study summaries and training curves read back from MLflow.

    uv run python -m src.eval.plots

Writes to report/figures/ and outputs/optuna/: per-study optimisation history,
hyper-parameter importance, trials table (CSV + LaTeX) and best-trial JSON; and
train/validation curves for every final run.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd

from src.train.common import OUT_DIR, ROOT, setup_mlflow, study_storage

FIG = ROOT / "report" / "figures"
STUDIES = ["t1_udae", "t2_classifier", "t2_specialists", "t2_specialists_long", "t3_moe", "t4_cgan"]

# (experiment, run name, [(metric, label)], title, file stem)
CURVES = [
    ("t1_universal_dae", "t1_udae_final", [("train_loss", "train loss"), ("val_psnr", "val PSNR")], "Task 1 universal DAE", "t1_curves"),
    ("t2_hard_routing", "t2_classifier_final", [("train_loss", "train CE"), ("val_loss", "val CE"), ("train_acc", "train acc"), ("val_acc", "val acc")], "Task 2 classifier", "t2_clf_curves"),
    ("t2_hard_routing", "t2_specialist_salt_pepper", [("train_loss", "train loss"), ("val_psnr", "val PSNR")], "Task 2 S&P specialist", "t2_spec_sp_curves"),
    ("t2_hard_routing", "t2_specialist_blur", [("train_loss", "train loss"), ("val_psnr", "val PSNR")], "Task 2 blur specialist", "t2_spec_blur_curves"),
    ("t2_hard_routing", "t2_specialist_occlusion", [("train_loss", "train loss"), ("val_psnr", "val PSNR")], "Task 2 occlusion specialist", "t2_spec_occ_curves"),
    ("t3_soft_moe", "t3_moe_final", [("train_loss", "train loss"), ("val_psnr", "val PSNR"), ("val_route_acc", "gate accuracy"), ("val_wbar_min", "min mean weight")], "Task 3 soft MoE", "t3_curves"),
    ("t4_face2sketch", "t4_cgan_final", [("train_d_real", "D real"), ("train_d_fake", "D fake"), ("train_g_adv", "G adversarial"), ("train_g_l1", "G L1"), ("val_ssim", "val SSIM"), ("val_l1", "val L1")], "Task 4 cGAN", "t4_curves"),
]


def study_figures(name: str) -> dict | None:
    try:
        study = optuna.load_study(study_name=name, storage=study_storage(name))
    except KeyError:
        return None
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    out = OUT_DIR / "optuna"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}_trials.csv", index=False)
    done = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
    cols = ["number", "value"] + [c for c in df.columns if c.startswith("params_")]
    (out / f"{name}_top_trials.tex").write_text(done[cols].head(8).rename(columns=lambda c: c.replace("params_", "").replace("_", r"\_")).to_latex(index=False, float_format="%.4g"))

    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    vals = done.sort_values("number")
    axes[0].scatter(vals["number"], vals["value"], s=14, label="completed trial")
    axes[0].plot(vals["number"], vals["value"].cummax(), color="C3", label="best so far")
    pruned = df[df["state"] == "PRUNED"]
    axes[0].scatter(pruned["number"], [vals["value"].min()] * len(pruned), marker="x", color="grey", s=14, label="pruned")
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel("objective")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{name}: optimisation history")
    try:
        imp = optuna.importance.get_param_importances(study)
        axes[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="C0")
        axes[1].set_title("hyper-parameter importance (fANOVA)")
    except Exception as e:  # noqa: BLE001 - too few trials
        axes[1].text(0.5, 0.5, f"importance unavailable:\n{e}", ha="center", va="center", fontsize=8)
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / f"optuna_{name}.png", dpi=200)
    plt.close(fig)
    _ = plot_optimization_history, plot_param_importances
    summary = {
        "study": name,
        "n_trials": len(df),
        "n_complete": int((df["state"] == "COMPLETE").sum()),
        "n_pruned": int((df["state"] == "PRUNED").sum()),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    (out / f"{name}_best.json").write_text(json.dumps(summary, indent=1))
    return summary


def curves():
    c = mlflow.MlflowClient()
    for exp, run_name, metrics, title, stem in CURVES:
        runs = mlflow.search_runs(experiment_names=[exp], filter_string=f"attributes.run_name = '{run_name}'", output_format="list", order_by=["attributes.start_time DESC"])
        if not runs:
            continue
        rid = runs[0].info.run_id
        n = len(metrics)
        fig, axes = plt.subplots(1, min(n, 2) if n <= 2 else 2, figsize=(10, 3.4), squeeze=False)
        groups = [metrics[: n // 2 or 1], metrics[n // 2 or 1 :]] if n > 1 else [metrics]
        for ax, grp in zip(axes[0], groups):
            for key, label in grp:
                h = c.get_metric_history(rid, key)
                if h:
                    ax.plot([m.step for m in h], [m.value for m in h], label=label)
            ax.set_xlabel("epoch")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(FIG / f"{stem}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    setup_mlflow("t1_universal_dae")
    allsum = [s for s in (study_figures(n) for n in STUDIES) if s]
    (OUT_DIR / "optuna" / "all_studies.json").write_text(json.dumps(allsum, indent=1, default=str))
    curves()
    print(pd.DataFrame(allsum)[["study", "n_trials", "n_complete", "n_pruned", "best_trial", "best_value"]].to_string())
