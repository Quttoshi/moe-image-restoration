"""Task 4 - style-conditioned pix2pix: U-Net generator G(x, s) and PatchGAN discriminator D(x, y, s).

Style conditioning uses a learned ``nn.Embedding(3, d)`` in both networks:
  * Generator - the embedding is (a) concatenated to the 1x1 bottleneck and
    (b) turned into per-channel FiLM scale/shift for every decoder block, so the
    style can affect both global layout and fine stroke texture.
  * Discriminator - the embedding is projected and tiled to a spatial map that
    is concatenated with the (photo, sketch) pair, so the discriminator judges
    whether a sketch is real *for that style*.
"""
from __future__ import annotations

import torch
from torch import nn


class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class UpFiLM(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(cout, affine=False)
        self.film = nn.Linear(emb_dim, 2 * cout)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, e):
        h = self.norm(self.conv(x))
        gamma, beta = self.film(e).chunk(2, dim=1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.act(self.drop(h))


class UNetGenerator(nn.Module):
    """7-level U-Net for 128x128 (128 -> 1x1 bottleneck), full skip connections as in pix2pix."""

    def __init__(self, base_ch: int = 64, emb_dim: int = 16, dropout: float = 0.5, n_styles: int = 3):
        super().__init__()
        self.config = dict(base_ch=base_ch, emb_dim=emb_dim, dropout=dropout, n_styles=n_styles)
        c = base_ch
        chs = [c, 2 * c, 4 * c, 8 * c, 8 * c, 8 * c, 8 * c]           # 64,32,16,8,4,2,1
        self.embed = nn.Embedding(n_styles, emb_dim)
        self.downs = nn.ModuleList([Down(3, chs[0], norm=False)] + [Down(chs[i - 1], chs[i], norm=i < 6) for i in range(1, 7)])
        ups, cin = [], chs[6] + emb_dim
        for i, cout in enumerate([8 * c, 8 * c, 8 * c, 4 * c, 2 * c, c]):  # 2,4,8,16,32,64
            ups.append(UpFiLM(cin, cout, emb_dim, dropout if i < 3 else 0.0))
            cin = cout + chs[5 - i]
        self.ups = nn.ModuleList(ups)
        self.out = nn.Sequential(nn.ConvTranspose2d(cin, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        e = self.embed(style)
        skips, h = [], x
        for d in self.downs:
            h = d(h)
            skips.append(h)
        h = torch.cat([h, e[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])], 1)
        for i, up in enumerate(self.ups):
            h = up(h, e)
            h = torch.cat([h, skips[5 - i]], 1)
        return self.out(h)


class PatchDiscriminator(nn.Module):
    """70x70-receptive-field PatchGAN on (photo, sketch, style map); outputs a grid of logits."""

    def __init__(self, base_ch: int = 64, emb_dim: int = 16, n_styles: int = 3, style_ch: int = 4):
        super().__init__()
        self.config = dict(base_ch=base_ch, emb_dim=emb_dim, n_styles=n_styles, style_ch=style_ch)
        self.embed = nn.Embedding(n_styles, emb_dim)
        self.proj = nn.Linear(emb_dim, style_ch)
        c = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(6 + style_ch, c, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, 2 * c, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * c), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1, bias=False), nn.BatchNorm2d(4 * c), nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * c, 8 * c, 4, 1, 1, bias=False), nn.BatchNorm2d(8 * c), nn.LeakyReLU(0.2, True),
            nn.Conv2d(8 * c, 1, 4, 1, 1),
        )

    def forward(self, x, y, style):
        s = self.proj(self.embed(style))[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, y, s], 1))


def init_weights(m: nn.Module) -> None:
    """pix2pix initialisation: N(0, 0.02) for conv, N(1, 0.02) for BatchNorm scale."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
