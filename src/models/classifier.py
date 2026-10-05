"""Four-way corruption classifier (Task 2) - also the initial gate for the soft MoE (Task 3).

No pooling before the first two stages: salt-and-pepper and blur are
high-frequency phenomena and early downsampling would average them away.
The head concatenates global average and global max pooling - max pooling
helps detect sparse saturated pixels (S&P) and flat black regions (occlusion).
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.dae import conv_block


class CorruptionClassifier(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3, n_classes: int = 4):
        super().__init__()
        self.config = dict(channels=list(channels), dropout=dropout, n_classes=n_classes)
        layers, cin = [], 3
        for i, c in enumerate(channels):
            layers += [conv_block(cin, c), conv_block(c, c, stride=2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * cin, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(pooled)
