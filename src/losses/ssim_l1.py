"""Reconstruction loss and image-quality metrics for Tasks 1-3."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def l1_ssim_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> tuple[torch.Tensor, dict]:
    """L = alpha * L1 + (1 - alpha) * (1 - SSIM)."""
    l1 = F.l1_loss(pred, target)
    s = ssim(pred, target, data_range=1.0, size_average=True)
    loss = alpha * l1 + (1 - alpha) * (1 - s)
    return loss, {"l1": l1.item(), "ssim": s.item()}


@torch.no_grad()
def per_image_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-image L1, MSE, PSNR (dB) and SSIM for a batch in [0, 1]."""
    pred = pred.clamp(0, 1)
    l1 = (pred - target).abs().flatten(1).mean(1)
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    s = ssim(pred, target, data_range=1.0, size_average=False)
    return {"l1": l1, "mse": mse, "psnr": psnr, "ssim": s}
