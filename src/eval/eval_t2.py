"""Task 2 - classifier metrics and hard-routed restoration (oracle vs predicted) on the test manifest.

    uv run python -m src.eval.eval_t2

Outputs in outputs/t2_eval/: classifier_report.{csv,tex}, confusion_matrix.png, accuracy_by_severity.csv,
summary_{oracle,predicted}.csv, routing_comparison.csv, misrouted_failures.png, examples.png.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.data.corruptions import CLASSES
from src.eval import restoration as R
from src.eval.eval_t1 import load_dae
from src.models.classifier import CorruptionClassifier
from src.models.moe import EXPERT_ORDER
from src.train.common import CKPT_DIR, OUT_DIR, device, error_map, save_grid, setup_mlflow

LABELS = ["Clean", "Salt & pepper", "Blur", "Occlusion"]


def load_classifier():
    ck = torch.load(CKPT_DIR / "t2_classifier.pt", map_location="cpu")
    m = CorruptionClassifier(tuple(ck["config"]["channels"]), ck["config"]["dropout"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device()).eval()


def make_router(clf, experts, mode: str):
    """Returns predict(x, label) implementing hard routing with an identity bypass for 'clean'."""

    def predict(x, lab):
        probs = torch.softmax(clf(x).float(), 1)
        pred = probs.argmax(1)
        route = lab if mode == "oracle" else pred
        out = x.clone()  # identity bypass for clean
        for k, name in enumerate(EXPERT_ORDER, start=1):
            sel = route == k
            if sel.any():
                out[sel] = experts[name](x[sel]).float()
        return out, {"pred": pred, "route": route, "prob": probs}

    return predict


def plot_confusion(cm: np.ndarray, path, title: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    out = OUT_DIR / "t2_eval"
    out.mkdir(parents=True, exist_ok=True)
    clf = load_classifier()
    experts = {k: load_dae(CKPT_DIR / f"t2_specialist_{k}.pt") for k in EXPERT_ORDER}
    ds = R.test_dataset()

    dfs = {}
    for mode in ("oracle", "predicted"):
        df = R.run(make_router(clf, experts, mode), ds)
        df["true"] = df["type"].map({c: i for i, c in enumerate(CLASSES)})
        df.to_csv(out / f"per_sample_{mode}.csv", index=False)
        t = R.summary_table(df)
        t.to_csv(out / f"summary_{mode}.csv")
        (out / f"summary_{mode}.tex").write_text(t.to_latex(float_format="%.3f"))
        dfs[mode] = df

    # ---- classifier metrics (predicted mode carries the predictions) ----
    df = dfs["predicted"]
    y, p = df["true"].to_numpy(), df["pred"].to_numpy()
    rep = pd.DataFrame(classification_report(y, p, target_names=LABELS, output_dict=True, digits=4)).T
    rep.to_csv(out / "classifier_report.csv")
    (out / "classifier_report.tex").write_text(rep.to_latex(float_format="%.4f"))
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro")
    metrics = {"accuracy": accuracy_score(y, p), "macro_precision": pr, "macro_recall": rc, "macro_f1": f1}
    cm = confusion_matrix(y, p, normalize="true", labels=range(4))
    plot_confusion(cm, out / "confusion_matrix.png", "Corruption classifier (test, row-normalised)")
    np.savetxt(out / "confusion_matrix.csv", cm, delimiter=",", fmt="%.4f")
    df.assign(correct=y == p).groupby(["type", "severity"])["correct"].mean().to_csv(out / "accuracy_by_severity.csv")

    # ---- oracle vs predicted routing ----
    o, pdx = dfs["oracle"].set_index("entry"), dfs["predicted"].set_index("entry")
    cmp = pd.DataFrame({
        "type": o["type"], "severity": o["severity"],
        "psnr_oracle": o["psnr"], "psnr_predicted": pdx["psnr"],
        "ssim_oracle": o["ssim"], "ssim_predicted": pdx["ssim"],
        "misrouted": pdx["pred"] != pdx["true"],
    })
    cmp["psnr_loss"] = cmp["psnr_oracle"] - cmp["psnr_predicted"]
    comp = cmp.groupby(["type", "severity"]).agg(psnr_oracle=("psnr_oracle", "mean"), psnr_predicted=("psnr_predicted", "mean"),
                                                ssim_oracle=("ssim_oracle", "mean"), ssim_predicted=("ssim_predicted", "mean"),
                                                misrouted_rate=("misrouted", "mean")).round(4)
    comp.to_csv(out / "routing_comparison.csv")
    (out / "routing_comparison.tex").write_text(comp.to_latex(float_format="%.3f"))

    # Misrouted failures: largest restoration loss caused purely by the routing error.
    worst = cmp[cmp["misrouted"]].nlargest(8, "psnr_loss").index.tolist()
    dev = device()
    rows, notes = [], []
    with torch.no_grad():
        for i in worst:
            x, yy, lab, _ = ds[i]
            x, yy = x[None].to(dev), yy[None].to(dev)
            po, _ = make_router(clf, experts, "oracle")(x, torch.tensor([lab], device=dev))
            pp, info = make_router(clf, experts, "predicted")(x, torch.tensor([lab], device=dev))
            rows.append([yy[0], x[0], po[0], pp[0], error_map(pp[0], yy[0]).to(dev)])
            notes.append({"entry": i, "true": CLASSES[lab], "predicted": CLASSES[int(info["pred"][0])], "severity": ds.entries[i]["severity"],
                          "psnr_oracle": float(cmp.loc[i, "psnr_oracle"]), "psnr_predicted": float(cmp.loc[i, "psnr_predicted"])})
    if rows:
        save_grid(rows, out / "misrouted_failures.png")
        (out / "misrouted_failures.json").write_text(json.dumps(notes, indent=1))
    R.example_grid(make_router(clf, experts, "predicted"), ds, R.pick_representative(dfs["predicted"]), out / "examples.png")

    print(json.dumps(metrics, indent=1))
    print(comp.to_string())
    setup_mlflow("t2_hard_routing")
    with mlflow.start_run(run_name="t2_test_eval", tags={"stage": "test"}):
        mlflow.log_metrics({f"test_clf_{k}": float(v) for k, v in metrics.items()})
        for mode, d in dfs.items():
            mlflow.log_metrics({f"test_{mode}_psnr": d["psnr"].mean(), f"test_{mode}_ssim": d["ssim"].mean()})
        mlflow.log_artifacts(str(out), artifact_path="test_eval")


if __name__ == "__main__":
    main()
