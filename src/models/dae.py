"""Convolutional denoising autoencoder used by Task 1 (universal) and Task 2 (specialists).

Encoder: ``n_down`` stride-2 stages (4 -> 8x8 latent, 3 -> 16x16 latent), channels c, 2c, 4c, 8c.
Bottleneck: 1x1 conv projection to ``latent_ch`` channels at ``latent_res`` x ``latent_res``,
i.e. a latent of ``latent_ch * latent_res**2`` values versus 3*128*128 = 49,152
input values. Over the search range this is a 3x-96x compression, so the network
cannot simply copy its input; a spatial (rather than fully-connected) latent keeps
the object layout while still forcing the corruption to be discarded.
Decoder: bilinear upsampling + conv (avoids transposed-conv checkerboard
artefacts), sigmoid output in [0, 1].

``skip_levels`` optionally adds *limited* additive skips from the deepest
encoder stages only (the first ``skip_levels`` decoder resolutions). Full-
resolution skips are never used because they would let corrupted pixels bypass
the bottleneck.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(conv_block(cin, cout, 2), conv_block(cout, cout))

    def forward(self, x):
        return self.body(x)


class Up(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.body = nn.Sequential(conv_block(cin, cout), nn.Dropout2d(dropout), conv_block(cout, cout))

    def forward(self, x):
        return self.body(self.up(x))


class ConvDAE(nn.Module):
    def __init__(self, base_ch: int = 32, latent_ch: int = 16, dropout: float = 0.1, skip_levels: int = 0, latent_res: int = 8):
        super().__init__()
        if latent_res not in (8, 16):
            raise ValueError("latent_res must be 8 or 16")
        n = int(math.log2(128 // latent_res))
        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8][:n]
        self.config = dict(base_ch=base_ch, latent_ch=latent_ch, dropout=dropout, skip_levels=skip_levels, latent_res=latent_res)
        self.n, self.skip_levels = n, skip_levels
        self.stem = conv_block(3, c[0])
        self.downs = nn.ModuleList([Down(c[max(i - 1, 0)], c[i]) for i in range(n)])  # res 64, 32, 16(, 8)
        self.to_latent = nn.Sequential(nn.Conv2d(c[-1], latent_ch, 1), nn.BatchNorm2d(latent_ch), nn.Tanh())
        self.from_latent = nn.Sequential(conv_block(latent_ch, c[-1]), nn.Dropout2d(dropout))
        self.ups = nn.ModuleList([Up(c[n - 1 - i], c[max(n - 2 - i, 0)], dropout if i < n - 2 else 0.0) for i in range(n)])
        self.head = nn.Conv2d(c[0], 3, 3, 1, 1)

    @property
    def latent_dim(self) -> int:
        return self.config["latent_ch"] * self.config["latent_res"] ** 2

    def encode(self, x: torch.Tensor):
        feats = []
        h = self.stem(x)
        for d in self.downs:
            h = d(h)
            feats.append(h)
        return self.to_latent(h), feats

    def decode(self, z: torch.Tensor, feats=None) -> torch.Tensor:
        h = self.from_latent(z)
        for i, up in enumerate(self.ups):
            h = up(h)
            if feats is not None and i < self.skip_levels and self.n - 2 - i >= 0:
                h = h + feats[self.n - 2 - i]
        return torch.sigmoid(self.head(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, feats = self.encode(x)
        return self.decode(z, feats if self.skip_levels else None)
