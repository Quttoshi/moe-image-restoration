"""Task 1 - final evaluation of the universal DAE on the untouched test manifest.

    uv run python -m src.eval.eval_t1 [--ckpt checkpoints/t1_udae_final.pt]

Writes outputs/t1_eval/{per_sample.csv, summary.csv, summary.tex, examples.png, failures.png}
and logs them to MLflow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import torch

from src.eval import restoration as R
from src.models.dae import ConvDAE
from src.train.common import CKPT_DIR, OUT_DIR, device, setup_mlflow


def load_dae(path: Path) -> ConvDAE:
    ck = torch.load(path, map_location="cpu")
    m = ConvDAE(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device()).eval()


def main(ckpt: Path) -> None:
    model = load_dae(ckpt)
    predict = lambda x, lab: (model(x), {})  # noqa: E731 - the model never sees the label
    ds = R.test_dataset()
    df = R.run(predict, ds)
    out = OUT_DIR / "t1_eval"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_sample.csv", index=False)
    table = R.summary_table(df)
    table.to_csv(out / "summary.csv")
    (out / "summary.tex").write_text(table.to_latex(float_format="%.3f"))
    print(table.to_string())

    R.example_grid(predict, ds, R.pick_representative(df), out / "examples.png")
    R.example_grid(predict, ds, R.pick_failures(df), out / "failures.png")

    setup_mlflow("t1_universal_dae")
    with mlflow.start_run(run_name="t1_test_eval", tags={"stage": "test"}):
        mlflow.log_param("checkpoint", ckpt.name)
        overall = table.loc[("overall", "corrupted")]
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in overall.items()})
        mlflow.log_artifacts(str(out), artifact_path="test_eval")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT_DIR / "t1_udae_final.pt")
    main(ap.parse_args().ckpt)
