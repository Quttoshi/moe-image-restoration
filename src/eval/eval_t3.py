"""Task 3 - soft MoE: restoration quality + gate behaviour analysis on the test manifest.

    uv run python -m src.eval.eval_t3

Outputs (outputs/t3_eval/): summary.csv, mean_weights.csv, routing_heatmap.png, weight_distribution.png,
expert_usage.json, dominant_examples.png, distributed_examples.png, mixed_corruptions.csv, method_comparison.csv
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

from src.data import pets
from src.data.corruptions import CLASSES, apply_corruption, level_params
from src.eval import restoration as R
from src.eval.eval_t1 import load_dae
from src.eval.eval_t2 import load_classifier, make_router
from src.losses.ssim_l1 import per_image_metrics
from src.models.moe import EXPERT_ORDER
from src.train.common import CKPT_DIR, OUT_DIR, device, error_map, save_grid, setup_mlflow
from src.train.t3_moe import load_moe

BRANCHES = ["identity"] + EXPERT_ORDER
BR_LABEL = ["Identity", "S&P expert", "Blur expert", "Occlusion expert"]
TYPE_LABEL = {"clean": "Clean", "salt_pepper": "Salt & pepper", "blur": "Blur", "occlusion": "Occlusion"}
MIXES = [("salt_pepper", "blur"), ("blur", "occlusion"), ("salt_pepper", "occlusion")]


def moe_predict(moe):
    def predict(x, lab):
        out, w, _ = moe(x)
        return out, {"w": w.float(), "top": w.argmax(1)}

    return predict


def heatmap(mw: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    labels = [f"{TYPE_LABEL[t]} · {s}" if s != "none" else TYPE_LABEL[t] for t, s in mw.index]
    sns.heatmap(mw.to_numpy(), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, xticklabels=BR_LABEL, yticklabels=labels, ax=ax, cbar_kws={"label": "mean routing weight"})
    ax.set_xlabel("Branch")
    ax.set_ylabel("True corruption · severity")
    ax.set_title("Soft-MoE gate: mean routing weights (test set)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def distribution(df: pd.DataFrame, path):
    long = pd.DataFrame([{"true": TYPE_LABEL[t], "branch": BR_LABEL[k], "weight": w[k]} for t, w in zip(df["type"], df["w"]) for k in range(4)])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sns.boxplot(data=long, x="true", y="weight", hue="branch", fliersize=1, ax=ax)
    ax.set_xlabel("True corruption")
    ax.set_ylabel("Routing weight")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def weighted_grid(moe, ds, entries, path):
    """Rows: clean | input | each weighted expert output | restored | error."""
    dev = device()
    rows, meta = [], []
    if not entries:
        return
    for i in entries:
        x, y, lab, _ = ds[i]
        x, y = x[None].to(dev), y[None].to(dev)
        out, w, _ = moe(x)
        branches = [x[0]] + [e(x)[0] for e in moe.experts]
        rows.append([y[0], x[0], *branches, out[0], error_map(out[0], y[0]).to(dev)])
        meta.append({"entry": i, "true": CLASSES[lab], "severity": ds.entries[i]["severity"], "weights": dict(zip(BRANCHES, w[0].tolist()))})
    save_grid(rows, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))


@torch.no_grad()
def mixed_test(moe, udae, clf, experts, n_images=300, seed=7) -> pd.DataFrame:
    """Images with two simultaneous corruptions (medium severity) - outside every model's training distribution."""
    dev = device()
    imgs = pets.load("test")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imgs), n_images, replace=False)
    hard = make_router(clf, experts, "predicted")
    rows = []
    for a, b in MIXES:
        xs, ys = [], []
        for i in idx:
            clean = imgs[int(i)].astype(np.float32) / 255.0
            x = apply_corruption(apply_corruption(clean, level_params(rng, a, "medium")), level_params(rng, b, "medium"))
            xs.append(pets.to_tensor(x))
            ys.append(pets.to_tensor(clean))
        for s in range(0, n_images, 50):
            x, y = torch.stack(xs[s : s + 50]).to(dev), torch.stack(ys[s : s + 50]).to(dev)
            soft, w, _ = moe(x)
            outs = {"input": x, "universal_dae": udae(x), "hard_routing": hard(x, None)[0], "soft_moe": soft}
            for name, o in outs.items():
                m = per_image_metrics(o.float(), y)
                rows += [{"mix": f"{a}+{b}", "method": name, "psnr": m["psnr"][j].item(), "ssim": m["ssim"][j].item()} for j in range(len(x))]
            rows += [{"mix": f"{a}+{b}", "method": "soft_moe_weights", **dict(zip(BRANCHES, w[j].tolist()))} for j in range(len(x))]
    return pd.DataFrame(rows)


def main():
    out = OUT_DIR / "t3_eval"
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    moe = load_moe().to(dev).eval()
    ds = R.test_dataset()
    df = R.run(moe_predict(moe), ds, batch_size=32)
    df["entropy"] = df["w"].apply(lambda w: float(-(np.array(w) * np.log(np.clip(w, 1e-12, 1))).sum() / np.log(4)))
    df.to_pickle(out / "per_sample.pkl")
    t = R.summary_table(df)
    t.to_csv(out / "summary.csv")
    (out / "summary.tex").write_text(t.to_latex(float_format="%.3f"))

    # ---- gate analysis ----
    W = np.stack(df["w"].to_numpy())
    wdf = pd.DataFrame(W, columns=BRANCHES).assign(type=df["type"], severity=df["severity"])
    order = [(t, s) for t in CLASSES for s in (["none"] if t == "clean" else ["low", "medium", "high"])]
    mw = wdf.groupby(["type", "severity"])[BRANCHES].mean().reindex(order)
    mw.round(4).to_csv(out / "mean_weights.csv")
    (out / "mean_weights.tex").write_text(mw.to_latex(float_format="%.3f"))
    heatmap(mw, out / "routing_heatmap.png")
    distribution(df, out / "weight_distribution.png")

    true_idx = df["type"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    top = W.argmax(1)
    usage = {
        "mean_weight_per_branch": dict(zip(BRANCHES, W.mean(0).round(4).tolist())),
        "top1_share_per_branch": dict(zip(BRANCHES, np.bincount(top, minlength=4).astype(float) / len(top))),
        "gate_top1_matches_true_label": float((top == true_idx).mean()),
        "inactive_branches(mean_w<0.02)": [b for b, v in zip(BRANCHES, W.mean(0)) if v < 0.02],
        # Weight a branch receives on inputs whose corruption it is NOT responsible for.
        "mean_weight_on_unrelated_inputs": {b: float(W[true_idx != k, k].mean()) for k, b in enumerate(BRANCHES)},
        "dominant_share(max_w>=0.9)": float((W.max(1) >= 0.9).mean()),
        "distributed_share(max_w<0.6)": float((W.max(1) < 0.6).mean()),
    }
    (out / "expert_usage.json").write_text(json.dumps(usage, indent=1, default=float))

    df["wmax"] = df["w"].apply(max)
    dom = df.sort_values("wmax", ascending=False).groupby("type").head(2)["entry"].tolist()
    dist = df[df["type"] != "clean"].nlargest(8, "entropy")["entry"].tolist()
    weighted_grid(moe, ds, dom[:8], out / "dominant_examples.png")
    weighted_grid(moe, ds, dist, out / "distributed_examples.png")
    R.example_grid(moe_predict(moe), ds, R.pick_representative(df), out / "examples.png")
    R.example_grid(moe_predict(moe), ds, R.pick_failures(df), out / "failures.png")

    # ---- comparison with Tasks 1 and 2 + mixed corruptions ----
    comp = {"soft_moe": t}
    for name, f in (("universal_dae", OUT_DIR / "t1_eval" / "summary.csv"), ("hard_predicted", OUT_DIR / "t2_eval" / "summary_predicted.csv"), ("hard_oracle", OUT_DIR / "t2_eval" / "summary_oracle.csv")):
        if f.exists():
            comp[name] = pd.read_csv(f, index_col=[0, 1])
    mc = pd.concat({k: v[["psnr", "ssim"]] for k, v in comp.items()}, axis=1).round(3)
    mc.to_csv(out / "method_comparison.csv")
    (out / "method_comparison.tex").write_text(mc.to_latex(float_format="%.3f"))

    udae = load_dae(CKPT_DIR / "t1_udae_final.pt")
    experts = {k: load_dae(CKPT_DIR / f"t2_specialist_{k}.pt") for k in EXPERT_ORDER}
    mixed = mixed_test(moe, udae, load_classifier(), experts)
    ms = mixed[mixed["method"] != "soft_moe_weights"].groupby(["mix", "method"])[["psnr", "ssim"]].mean().unstack("method").round(3)
    mwm = mixed[mixed["method"] == "soft_moe_weights"].groupby("mix")[BRANCHES].mean().round(3)
    ms.to_csv(out / "mixed_corruptions.csv")
    mwm.to_csv(out / "mixed_corruptions_weights.csv")
    (out / "mixed_corruptions.tex").write_text(ms.to_latex(float_format="%.3f"))

    print(t.to_string())
    print(mw.round(3).to_string())
    print(json.dumps(usage, indent=1, default=float))
    print(ms.to_string())
    print(mwm.to_string())
    setup_mlflow("t3_soft_moe")
    with mlflow.start_run(run_name="t3_test_eval", tags={"stage": "test"}):
        mlflow.log_metrics({"test_psnr": df["psnr"].mean(), "test_ssim": df["ssim"].mean(), "test_gate_acc": usage["gate_top1_matches_true_label"]})
        mlflow.log_artifacts(str(out), artifact_path="test_eval")


if __name__ == "__main__":
    main()
