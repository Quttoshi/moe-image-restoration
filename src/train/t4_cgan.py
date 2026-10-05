"""Task 4 - style-conditioned pix2pix (U-Net G + PatchGAN D) on FS2K.

    uv run python -m src.train.t4_cgan tune
    uv run python -m src.train.t4_cgan train

L_D = 0.5 * [BCE(D(x,y,s), 1) + BCE(D(x,G(x,s),s), 0)]
L_G = BCE(D(x,G(x,s),s), 1) + lambda_L1 * |y - G(x,s)|_1
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import mlflow
import optuna
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

from src.data.fs2k import PairedSketchDataset
from src.models.cgan import PatchDiscriminator, UNetGenerator, init_weights
from src.train.common import CKPT_DIR, OUT_DIR, device, load_config, make_loader, save_grid, seed_everything, setup_mlflow, study_storage
from src.tune.space import best_or_defaults, remaining_trials, suggest

CFG = load_config("t4")


def m11_to_01(t: torch.Tensor) -> torch.Tensor:
    return (t + 1) / 2


def sketch_score(l1: float, ssim_v: float) -> float:
    """Validation objective: SSIM plus pixel fidelity (L1 measured on [0,1] images)."""
    return 0.5 * ssim_v + 0.5 * (1.0 - l1)


@torch.no_grad()
def evaluate(G, loader, dev) -> dict:
    G.eval()
    s, n = defaultdict(float), 0
    for x, y, st in loader:
        x, y, st = x.to(dev), y.to(dev), st.to(dev)
        p, t = m11_to_01(G(x, st)).clamp(0, 1), m11_to_01(y)
        mse = ((p - t) ** 2).flatten(1).mean(1)
        s["l1"] += (p - t).abs().flatten(1).mean(1).sum().item()
        s["psnr"] += (10 * torch.log10(1 / mse.clamp_min(1e-10))).sum().item()
        s["ssim"] += ssim(p, t, data_range=1.0, size_average=False).sum().item()
        n += len(x)
    r = {k: v / n for k, v in s.items()}
    r["score"] = sketch_score(r["l1"], r["ssim"])
    G.train()
    return r


@torch.no_grad()
def fixed_grid(G, fixed, dev, path):
    """Rows: photo | ground truth | G(x, own style) | G(x, style1) | G(x, style2) | G(x, style3)."""
    G.eval()
    x, y, st = (t.to(dev) for t in fixed)
    rows = []
    for i in range(len(x)):
        xi = x[i : i + 1].repeat(3, 1, 1, 1)
        sweep = G(xi, torch.arange(3, device=dev))
        own = G(x[i : i + 1], st[i : i + 1])[0]
        rows.append([m11_to_01(t) for t in [x[i], y[i], own, *sweep]])
    G.train()
    return save_grid(rows, path)


def train_gan(hp: dict, epochs: int, trial=None, sample_every: int | None = None, augment: bool = True, prefix: str = "t4"):
    dev = device()
    G = UNetGenerator(hp["base_ch"], hp["emb_dim"], hp["dropout"]).to(dev)
    D = PatchDiscriminator(hp["base_ch"], hp["emb_dim"]).to(dev)
    G.apply(init_weights)
    D.apply(init_weights)
    train_dl = make_loader(PairedSketchDataset("train", augment=augment), batch_size=hp["batch_size"], shuffle=True)
    val_ds = PairedSketchDataset("val")
    val_dl = make_loader(val_ds, batch_size=32, workers=0)
    # Same validation photos every time so generator progress can be compared across epochs.
    fixed_idx = list(range(0, len(val_ds), max(1, len(val_ds) // 8)))[:8]
    fixed = tuple(torch.stack([torch.as_tensor(val_ds[i][k]) for i in fixed_idx]) for k in range(3))

    betas = tuple(CFG["betas"])
    opt_g = torch.optim.Adam(G.parameters(), lr=hp["lr_g"], betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=hp["lr_d"], betas=betas)
    half = epochs // 2
    decay = lambda ep: 1.0 if ep < half else max(0.0, 1.0 - (ep - half) / max(1, epochs - half))  # noqa: E731
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, decay)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, decay)
    bce = F.binary_cross_entropy_with_logits
    best, best_state, best_m = -math.inf, None, None

    for ep in range(1, epochs + 1):
        t0, tot, n = time.time(), defaultdict(float), 0
        for x, y, st in train_dl:
            x, y, st = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), st.to(dev, non_blocking=True)
            fake = G(x, st)

            # --- discriminator ---
            d_real = D(x, y, st)
            d_fake = D(x, fake.detach(), st)
            loss_d_real = bce(d_real, torch.ones_like(d_real))
            loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # --- generator ---
            d_gen = D(x, fake, st)
            loss_g_adv = bce(d_gen, torch.ones_like(d_gen))
            loss_g_l1 = F.l1_loss(fake, y)
            loss_g = loss_g_adv + hp["lambda_l1"] * loss_g_l1
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            for k, v in (("d_real", loss_d_real), ("d_fake", loss_d_fake), ("d_total", loss_d), ("g_adv", loss_g_adv), ("g_l1", loss_g_l1), ("g_total", loss_g)):
                tot[k] += v.item()
            tot["d_real_acc"] += (d_real > 0).float().mean().item()
            tot["d_fake_acc"] += (d_fake < 0).float().mean().item()
            n += 1
        sch_g.step()
        sch_d.step()
        val = evaluate(G, val_dl, dev)
        mlflow.log_metrics({f"train_{k}": v / n for k, v in tot.items()} | {f"val_{k}": v for k, v in val.items()} | {"lr_g": sch_g.get_last_lr()[0], "epoch_time_s": time.time() - t0}, step=ep)
        print(f"[{prefix}] ep {ep:3d} D_real {tot['d_real'] / n:.3f} D_fake {tot['d_fake'] / n:.3f} G_adv {tot['g_adv'] / n:.3f} G_L1 {tot['g_l1'] / n:.4f} | val L1 {val['l1']:.4f} SSIM {val['ssim']:.4f}")
        if val["score"] > best:
            best, best_m = val["score"], val | {"epoch": ep}
            best_state = {k: v.detach().clone() for k, v in G.state_dict().items()}
        if sample_every and (ep % sample_every == 0 or ep == 1 or ep == epochs):
            p = fixed_grid(G, fixed, dev, OUT_DIR / prefix / f"fixed_val_ep{ep:03d}.png")
            mlflow.log_artifact(str(p), artifact_path="fixed_val_samples")
        if trial is not None:
            trial.report(val["score"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    final_state = {k: v.detach().clone() for k, v in G.state_dict().items()}
    return G, D, best_state, best_m, final_state, val


def tune(n_trials=None):
    setup_mlflow(CFG["experiment"])
    t = CFG["tune"]
    study = optuna.create_study(study_name=CFG["study_name"], storage=study_storage(CFG["study_name"]), direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(**t["pruner"]))

    def objective(trial):
        hp = suggest(trial, CFG["search_space"])
        seed_everything(42)
        with mlflow.start_run(run_name=f"gan_trial_{trial.number}", tags={"stage": "optuna", "study": CFG["study_name"]}):
            mlflow.log_params(hp)
            *_, best_m, _, _ = train_gan(hp, t["epochs"], trial=trial, augment=CFG["train"]["augment"], prefix=f"t4_trial{trial.number}")
            mlflow.log_metric("objective", best_m["score"])
            return best_m["score"]

    study.optimize(objective, n_trials=remaining_trials(study, n_trials or t["n_trials"]), gc_after_trial=True)
    print("best", study.best_value, study.best_params)


def train(epochs=None):
    setup_mlflow(CFG["experiment"])
    hp, from_study = best_or_defaults(CFG["study_name"], study_storage(CFG["study_name"]), CFG["defaults"])
    seed_everything(42)
    tc = CFG["train"]
    with mlflow.start_run(run_name="t4_cgan_final", tags={"stage": "final", "hp_source": "optuna" if from_study else "defaults"}):
        mlflow.log_params(hp)
        G, D, best_state, best_m, final_state, final_m = train_gan(hp, epochs or tc["epochs"], sample_every=tc["sample_every"], augment=tc["augment"], prefix="t4_final")
        CKPT_DIR.mkdir(exist_ok=True)
        # The end-of-schedule generator is kept as the deliverable (GAN val metrics
        # are noisy and favour blurrier outputs); the best-by-val is saved alongside.
        torch.save({"state_dict": final_state, "config": G.config, "hp": hp, "val": final_m}, CKPT_DIR / "t4_generator.pt")
        torch.save({"state_dict": best_state, "config": G.config, "hp": hp, "val": best_m}, CKPT_DIR / "t4_generator_bestval.pt")
        torch.save({"state_dict": D.state_dict(), "config": D.config}, CKPT_DIR / "t4_discriminator.pt")
        for f in ("t4_generator.pt", "t4_generator_bestval.pt"):
            mlflow.log_artifact(str(CKPT_DIR / f), artifact_path="checkpoints")
        print("final", final_m, "best", best_m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "train"])
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    tune(a.n_trials) if a.cmd == "tune" else train(a.epochs)
