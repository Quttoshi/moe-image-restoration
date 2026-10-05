"""Corruption operators shared by training, manifest generation and the API.

Kept NumPy-only (no torch) so the FastAPI backend can import it without
pulling in PyTorch. Images are float32 arrays in HWC layout with values in [0, 1].

Every corruption is split into two steps:
  * ``sample_params(rng, kind)`` draws a JSON-serialisable parameter dict;
  * ``apply_corruption(img, params)`` applies it deterministically.
This split is what makes the validation/test manifests reproducible: the
manifest stores the parameter dict and replaying it gives identical pixels.
"""
from __future__ import annotations

import math

import numpy as np

CLASSES = ["clean", "salt_pepper", "blur", "occlusion"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Training-time ranges (assignment specification).
SP_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_COUNT_RANGE = (1, 3)
OCC_AREA_RANGE = (0.10, 0.35)

# Fixed test-time severity levels (assignment specification).
TEST_LEVELS = {
    "salt_pepper": {"low": {"p": 0.03}, "medium": {"p": 0.08}, "high": {"p": 0.15}},
    "blur": {
        "low": {"kernel": 3, "sigma": 0.7},
        "medium": {"kernel": 5, "sigma": 1.5},
        "high": {"kernel": 7, "sigma": 2.5},
    },
    "occlusion": {
        "low": {"n_rects": 1, "area": 0.10},
        "medium": {"n_rects": 2, "area": 0.20},
        "high": {"n_rects": 3, "area": 0.35},
    },
}
SEVERITIES = ["low", "medium", "high"]


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def salt_pepper(img: np.ndarray, p: float, seed: int) -> np.ndarray:
    """Replace a fraction ``p`` of pixels (all channels) with black or white (50/50)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    hit = rng.random((h, w)) < p
    white = rng.random((h, w)) < 0.5
    out = img.copy()
    out[hit & white] = 1.0
    out[hit & ~white] = 0.0
    return out


def gaussian_kernel1d(kernel: int, sigma: float) -> np.ndarray:
    x = np.arange(kernel, dtype=np.float64) - (kernel - 1) / 2.0
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(img: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (same convention as torchvision)."""
    k = gaussian_kernel1d(kernel, sigma)
    r = kernel // 2
    h, w = img.shape[:2]
    padded = np.pad(img, ((r, r), (r, r), (0, 0)), mode="reflect")
    tmp = sum(k[j] * padded[:, j : j + w] for j in range(kernel))
    out = sum(k[j] * tmp[j : j + h] for j in range(kernel))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def occlude(img: np.ndarray, rects: list[list[int]]) -> np.ndarray:
    """Fill each ``[x, y, w, h]`` rectangle with black."""
    out = img.copy()
    for x, y, w, h in rects:
        out[y : y + h, x : x + w] = 0.0
    return out


def _overlaps(a: list[int], b: list[int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def sample_rects(
    rng: np.random.Generator, n: int, area_frac: float, h: int = 128, w: int = 128
) -> list[list[int]]:
    """Place ``n`` non-overlapping rectangles whose union covers ~``area_frac``.

    Non-overlap makes the union area equal the sum of the rectangle areas, so the
    coverage requirement is met exactly up to integer rounding.
    """
    total = area_frac * h * w
    for _ in range(200):
        # Split the target area randomly among the rectangles (each >= 20% share).
        shares = rng.dirichlet(np.full(n, 2.0)) if n > 1 else np.array([1.0])
        shares = np.maximum(shares, 0.2)
        shares /= shares.sum()
        rects: list[list[int]] = []
        ok = True
        for s in shares:
            a = s * total
            aspect = rng.uniform(0.5, 2.0)
            rw = int(round(min(w, max(4, math.sqrt(a * aspect)))))
            rh = int(round(min(h, max(4, a / rw))))
            for _ in range(100):
                x = int(rng.integers(0, w - rw + 1))
                y = int(rng.integers(0, h - rh + 1))
                cand = [x, y, rw, rh]
                if not any(_overlaps(cand, r) for r in rects):
                    rects.append(cand)
                    break
            else:
                ok = False
                break
        if ok:
            return rects
    raise RuntimeError("could not place non-overlapping occlusion rectangles")


def occluded_fraction(rects: list[list[int]], h: int = 128, w: int = 128) -> float:
    mask = np.zeros((h, w), dtype=bool)
    for x, y, rw, rh in rects:
        mask[y : y + rh, x : x + rw] = True
    return float(mask.mean())


# --------------------------------------------------------------------------- #
# Parameter sampling / application
# --------------------------------------------------------------------------- #
def sample_params(rng: np.random.Generator, kind: str, h: int = 128, w: int = 128) -> dict:
    """Draw training-distribution parameters for one corruption ``kind``."""
    if kind == "clean":
        return {"type": "clean"}
    if kind == "salt_pepper":
        return {
            "type": kind,
            "p": float(rng.uniform(*SP_RANGE)),
            "seed": int(rng.integers(0, 2**31 - 1)),
        }
    if kind == "blur":
        return {
            "type": kind,
            "kernel": int(rng.choice(BLUR_KERNELS)),
            "sigma": float(rng.uniform(*BLUR_SIGMA_RANGE)),
        }
    if kind == "occlusion":
        n = int(rng.integers(OCC_COUNT_RANGE[0], OCC_COUNT_RANGE[1] + 1))
        area = float(rng.uniform(*OCC_AREA_RANGE))
        rects = sample_rects(rng, n, area, h, w)
        return {"type": kind, "n_rects": n, "area": area, "rects": rects}
    raise ValueError(f"unknown corruption kind: {kind}")


def level_params(rng: np.random.Generator, kind: str, severity: str, h: int = 128, w: int = 128) -> dict:
    """Parameters for one of the fixed test severity levels."""
    base = dict(TEST_LEVELS[kind][severity])
    params = {"type": kind, **base}
    if kind == "salt_pepper":
        params["seed"] = int(rng.integers(0, 2**31 - 1))
    elif kind == "occlusion":
        params["rects"] = sample_rects(rng, base["n_rects"], base["area"], h, w)
    return params


def apply_corruption(img: np.ndarray, params: dict) -> np.ndarray:
    kind = params["type"]
    if kind == "clean":
        return img.copy()
    if kind == "salt_pepper":
        return salt_pepper(img, params["p"], params["seed"])
    if kind == "blur":
        return gaussian_blur(img, params["kernel"], params["sigma"])
    if kind == "occlusion":
        return occlude(img, params["rects"])
    raise ValueError(f"unknown corruption kind: {kind}")


def severity_bucket(params: dict) -> str:
    """Map training-distribution parameters to low/medium/high (equal-width thirds)."""
    kind = params["type"]
    if kind == "clean":
        return "none"
    if kind == "salt_pepper":
        v, lo, hi = params["p"], *SP_RANGE
    elif kind == "blur":
        v, lo, hi = params["sigma"], *BLUR_SIGMA_RANGE
    else:
        v, lo, hi = params["area"], *OCC_AREA_RANGE
    t = (v - lo) / (hi - lo)
    return "low" if t < 1 / 3 else "medium" if t < 2 / 3 else "high"
