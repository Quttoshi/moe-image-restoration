"""Task 3 - differentiable soft mixture-of-experts restoration.

    w      = softmax(G(x~) / tau)              G = gate (initialised from the Task 2 classifier)
    x_hat  = w0 * x~ + w1 A_salt(x~) + w2 A_blur(x~) + w3 A_occ(x~)

Branch 0 is the identity (clean) path, so the order of ``w`` matches the
classifier's class order [clean, salt_pepper, blur, occlusion].
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.classifier import CorruptionClassifier
from src.models.dae import ConvDAE

EXPERT_ORDER = ["salt_pepper", "blur", "occlusion"]


class SoftMoE(nn.Module):
    def __init__(self, gate: CorruptionClassifier, experts: dict[str, ConvDAE], tau: float = 1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList([experts[k] for k in EXPERT_ORDER])
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        w = torch.softmax(logits / self.tau, dim=1)                       # (B, 4)
        outs = torch.stack([x] + [e(x) for e in self.experts], dim=1)      # (B, 4, 3, H, W)
        restored = (w[:, :, None, None, None] * outs).sum(1)
        return restored, w, logits

    def set_experts_trainable(self, flag: bool) -> None:
        for p in self.experts.parameters():
            p.requires_grad_(flag)
        # Frozen experts also keep BatchNorm running statistics fixed.
        self.experts.train(flag and self.training)


def balance_loss(w: torch.Tensor) -> torch.Tensor:
    """Sum_k (mean_batch(w_k) - 1/K)^2 - pushes average routing towards uniform.

    With the class-balanced batches used here, perfect routing has w_bar = 1/4 for
    every branch, so this term only penalises collapse, not correct routing.
    """
    k = w.shape[1]
    return ((w.mean(0) - 1.0 / k) ** 2).sum()
