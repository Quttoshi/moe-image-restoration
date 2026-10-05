"""Task 4 - face-to-sketch cGAN evaluation on the official FS2K test split.

    uv run python -m src.eval.eval_t4

Metrics per style and overall: L1, PSNR, SSIM (paired, vs ground-truth sketch) and FID
(Inception-v3 pool features, real test sketches vs generated). Outputs in outputs/t4_eval/.
"""
from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from scipy import linalg

from src.data.fs2k import STYLE_NAMES, PairedSketchDataset
from src.models.cgan import UNetGenerator
from src.train.common import CKPT_DIR, OUT_DIR, device, make_loader, save_grid, setup_mlflow


def load_generator(path=None):
    ck = torch.load(path or CKPT_DIR / "t4_generator.pt", map_location="cpu")
    g = UNetGenerator(**ck["config"])
    g.load_state_dict(ck["state_dict"])
    return g.to(device()).eval()


class InceptionFeatures(torch.nn.Module):
    """2048-d pool features of torchvision's ImageNet Inception-v3 (the standard FID backbone)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = torch.nn.Identity()
        self.net = net.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @torch.no_grad()
    def forward(self, x01):
        x = F.interpolate(x01, size=299, mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


def fid(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


@torch.no_grad()
def main():
    out = OUT_DIR / "t4_eval"
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    G = load_generator()
    ds = PairedSketchDataset("test")
    try:
        incep = InceptionFeatures().to(dev)
    except Exception as e:  # noqa: BLE001 - weights download can fail offline; FID is then skipped
        print(f"FID disabled: could not load Inception-v3 weights ({e})")
        incep = None
    rows, f_real, f_fake = [], [], []
    k = 0
    for x, y, s in make_loader(ds, batch_size=32, workers=0):
        x, y, s = x.to(dev), y.to(dev), s.to(dev)
        p = ((G(x, s) + 1) / 2).clamp(0, 1)
        t = (y + 1) / 2
        mse = ((p - t) ** 2).flatten(1).mean(1)
        l1 = (p - t).abs().flatten(1).mean(1)
        ss = ssim(p, t, data_range=1.0, size_average=False)
        for j in range(len(x)):
            rows.append({"index": k + j, "style": int(s[j]), "l1": l1[j].item(), "psnr": (10 * torch.log10(1 / mse[j].clamp_min(1e-10))).item(), "ssim": ss[j].item()})
        k += len(x)
        if incep is not None:
            f_real.append(incep(t).cpu().numpy())
            f_fake.append(incep(p).cpu().numpy())
    df = pd.DataFrame(rows)
    styles = df["style"].to_numpy()
    fr, ff = (np.concatenate(f_real), np.concatenate(f_fake)) if f_real else (None, None)
    fid_or_nan = (lambda a, b: fid(a, b)) if fr is not None else (lambda a, b: float("nan"))
    summary = df.groupby("style")[["l1", "psnr", "ssim"]].mean()
    summary["n"] = df.groupby("style").size()
    # FID is unreliable for very small sets; report per style only where n >= 100.
    summary["fid"] = [fid_or_nan(fr[styles == st], ff[styles == st]) if fr is not None and (styles == st).sum() >= 100 else np.nan for st in summary.index]
    summary.index = [STYLE_NAMES[i] for i in summary.index]
    summary.loc["Overall"] = [df["l1"].mean(), df["psnr"].mean(), df["ssim"].mean(), len(df), fid_or_nan(fr, ff)]
    summary = summary.round(4)
    summary.to_csv(out / "summary.csv")
    (out / "summary.tex").write_text(summary.to_latex(float_format="%.3f"))
    df.to_csv(out / "per_sample.csv", index=False)

    def grid(indices, path, sweep=True):
        g_rows = []
        for i in indices:
            x, y, s = ds[i]
            xb = x[None].to(dev)
            own = G(xb, torch.tensor([s], device=dev))[0]
            r = [x, y, own.cpu()]
            if sweep:
                r += list(G(xb.repeat(3, 1, 1, 1), torch.arange(3, device=dev)).cpu())
            g_rows.append([(t + 1) / 2 for t in r])
        save_grid(g_rows, path)

    rng = np.random.default_rng(0)
    reps = [int(i) for st in range(3) for i in rng.choice(np.where(styles == st)[0], size=min(4, (styles == st).sum()), replace=False)]
    grid(reps, out / "examples_style_sweep.png")
    worst = df.nsmallest(8, "ssim")["index"].tolist()
    grid(worst, out / "failures.png", sweep=False)
    (out / "failures.json").write_text(json.dumps(df.nsmallest(8, "ssim").to_dict("records"), indent=1))

    print(summary.to_string())
    setup_mlflow("t4_face2sketch")
    with mlflow.start_run(run_name="t4_test_eval", tags={"stage": "test"}):
        mlflow.log_metrics({f"test_{c}": float(v) for c, v in summary.loc["Overall"].items() if c != "n"})
        mlflow.log_artifacts(str(out), artifact_path="test_eval")


if __name__ == "__main__":
    main()
