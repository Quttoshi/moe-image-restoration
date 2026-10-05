"""21i2525-A1 inference API.

Endpoints (all under /api):
  GET  /health              model load status, ONNX Runtime providers
  GET  /info                model metadata + ONNX/PyTorch parity reports
  GET  /samples             bundled sample images; GET /samples/{group}/{name}
  POST /corrupt             apply a corruption (type + severity or custom params)
  POST /restore/universal   Task 1 - universal DAE
  POST /restore/hard        Task 2 - classifier + hard-routed specialist
  POST /restore/soft        Task 3 - soft mixture-of-experts
  POST /sketch              Task 4 - style-conditioned face-to-sketch
"""
from __future__ import annotations

import math
from contextlib import asynccontextmanager
import os
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(os.environ.get("SHARED_SRC", ROOT))))
from src.data.corruptions import CLASSES, TEST_LEVELS, apply_corruption, level_params, occluded_fraction, sample_params, sample_rects  # noqa: E402

from app import imaging as im  # noqa: E402
from app.registry import Registry  # noqa: E402

SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", Path(__file__).resolve().parents[1] / "samples"))
EXPERT_LABELS = {"clean": "Identity bypass", "salt_pepper": "Salt-and-pepper expert", "blur": "Blur expert", "occlusion": "Occlusion expert"}
STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]

REG: Registry | None = None
STARTED = time.time()


@asynccontextmanager
async def lifespan(_app):
    global REG
    REG = Registry()
    yield


app = FastAPI(title="21i2525-A1 Generative Restoration & Sketch API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])
def reg() -> Registry:
    assert REG is not None
    return REG


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
CorruptionT = Literal["none", "salt_pepper", "blur", "occlusion"]
SeverityT = Literal["low", "medium", "high", "random", "custom"]


async def load_source(file: UploadFile | None, sample: str | None, crop_square: bool = False) -> tuple[np.ndarray, str]:
    if file is not None and file.filename:
        return im.to_array(await im.read_upload(file), crop_square), f"upload:{file.filename}"
    if sample:
        group, _, name = sample.partition("/")
        path = (SAMPLES_DIR / group / name).resolve()
        if not path.is_file() or SAMPLES_DIR.resolve() not in path.parents:
            raise HTTPException(404, f"unknown sample {sample!r}")
        return im.to_array(Image.open(path).convert("RGB"), crop_square), f"sample:{sample}"
    raise HTTPException(400, "provide an image file or a sample name")


def build_params(corruption: str, severity: str, seed: int | None, p: float | None, kernel: int | None, sigma: float | None, n_rects: int | None, area: float | None) -> dict | None:
    if corruption == "none":
        return None
    rng = np.random.default_rng(seed)
    if severity in ("low", "medium", "high"):
        return level_params(rng, corruption, severity)
    if severity == "random":
        return sample_params(rng, corruption)
    # custom - validated against the training ranges (slightly relaxed for exploration)
    if corruption == "salt_pepper":
        if p is None or not 0 < p <= 0.5:
            raise HTTPException(422, "custom salt-and-pepper needs 0 < p <= 0.5")
        return {"type": corruption, "p": p, "seed": int(rng.integers(0, 2**31 - 1))}
    if corruption == "blur":
        if kernel not in (3, 5, 7, 9) or sigma is None or not 0.1 <= sigma <= 5:
            raise HTTPException(422, "custom blur needs kernel in {3,5,7,9} and 0.1 <= sigma <= 5")
        return {"type": corruption, "kernel": kernel, "sigma": sigma}
    if n_rects not in (1, 2, 3) or area is None or not 0.02 <= area <= 0.5:
        raise HTTPException(422, "custom occlusion needs n_rects in {1,2,3} and 0.02 <= area <= 0.5")
    return {"type": corruption, "n_rects": n_rects, "area": area, "rects": sample_rects(rng, n_rects, area)}


def describe(params: dict | None) -> dict:
    if params is None:
        return {"type": "none", "label": "No corruption applied (input used as-is)"}
    t = params["type"]
    if t == "salt_pepper":
        label = f"Salt-and-pepper, p = {params['p']:.3f}"
    elif t == "blur":
        label = f"Gaussian blur, kernel {params['kernel']}, sigma = {params['sigma']:.2f}"
    else:
        label = f"Occlusion, {len(params['rects'])} rectangle(s), {occluded_fraction(params['rects']) * 100:.1f}% covered"
    return {**params, "label": label}


async def prepare(file, sample, corruption, severity, seed, p, kernel, sigma, n_rects, area):
    t0 = time.perf_counter()
    src, origin = await load_source(file, sample)
    params = build_params(corruption, severity, seed, p, kernel, sigma, n_rects, area)
    x = apply_corruption(src, params) if params else src
    # The clean reference is only known when the corruption was applied here.
    clean = src if params else None
    return src, x, clean, params, origin, (time.perf_counter() - t0) * 1000


def quality(out: np.ndarray, x: np.ndarray, clean: np.ndarray | None) -> dict:
    if clean is None:
        return {}
    return {"psnr_input": round(im.psnr(x, clean), 2), "psnr_output": round(im.psnr(out, clean), 2), "error_map": im.to_data_url(im.error_map(out, clean))}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


CorruptionForm = Form("none")
SeverityForm = Form("medium")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    r = reg()
    s = r.status()
    return {"status": "ok" if not s["missing"] else "degraded", "uptime_s": round(time.time() - STARTED, 1), **s}


@app.get("/api/info")
def info():
    r = reg()
    return {"models": r.info, "parity": r.parity, "classes": CLASSES, "test_levels": TEST_LEVELS, "styles": STYLE_NAMES}


@app.get("/api/samples")
def samples():
    out = {}
    for g in ("pets", "faces"):
        d = SAMPLES_DIR / g
        out[g] = sorted(f"{g}/{p.name}" for p in d.glob("*.png")) if d.exists() else []
    return out


@app.get("/api/samples/{group}/{name}")
def sample_file(group: str, name: str):
    path = (SAMPLES_DIR / group / name).resolve()
    if not path.is_file() or SAMPLES_DIR.resolve() not in path.parents:
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/corrupt")
async def corrupt(
    file: UploadFile | None = File(None), sample: str | None = Form(None),
    corruption: CorruptionT = CorruptionForm, severity: SeverityT = SeverityForm, seed: int | None = Form(None),
    p: float | None = Form(None), kernel: int | None = Form(None), sigma: float | None = Form(None),
    n_rects: int | None = Form(None), area: float | None = Form(None),
):
    src, x, _, params, origin, t_pre = await prepare(file, sample, corruption, severity, seed, p, kernel, sigma, n_rects, area)
    return {"source": origin, "clean": im.to_data_url(src), "input": im.to_data_url(x), "corruption": describe(params), "timing_ms": {"preprocess": round(t_pre, 2)}}


@app.post("/api/restore/universal")
async def restore_universal(
    file: UploadFile | None = File(None), sample: str | None = Form(None),
    corruption: CorruptionT = CorruptionForm, severity: SeverityT = SeverityForm, seed: int | None = Form(None),
    p: float | None = Form(None), kernel: int | None = Form(None), sigma: float | None = Form(None),
    n_rects: int | None = Form(None), area: float | None = Form(None),
):
    r = reg()
    r.require("universal")
    src, x, clean, params, origin, t_pre = await prepare(file, sample, corruption, severity, seed, p, kernel, sigma, n_rects, area)
    (out,), t_inf = r.run("universal", im.to_nchw(x))
    out = im.from_nchw(out)
    return {
        "task": "Universal Restoration", "model": "t1_universal_dae.onnx", "source": origin,
        "clean": im.to_data_url(clean) if clean is not None else None,
        "input": im.to_data_url(x), "output": im.to_data_url(out), "corruption": describe(params),
        **quality(out, x, clean),
        "timing_ms": {"preprocess": round(t_pre, 2), "inference": round(t_inf, 2)},
    }


@app.post("/api/restore/hard")
async def restore_hard(
    file: UploadFile | None = File(None), sample: str | None = Form(None),
    corruption: CorruptionT = CorruptionForm, severity: SeverityT = SeverityForm, seed: int | None = Form(None),
    p: float | None = Form(None), kernel: int | None = Form(None), sigma: float | None = Form(None),
    n_rects: int | None = Form(None), area: float | None = Form(None),
    routing: Literal["predicted", "oracle"] = Form("predicted"),
):
    r = reg()
    r.require("classifier", "salt_pepper", "blur", "occlusion")
    src, x, clean, params, origin, t_pre = await prepare(file, sample, corruption, severity, seed, p, kernel, sigma, n_rects, area)
    xin = im.to_nchw(x)
    (logits,), t_clf = r.run("classifier", xin)
    probs = softmax(logits[0])
    predicted = CLASSES[int(probs.argmax())]
    if routing == "oracle":
        if params is None:
            raise HTTPException(422, "oracle routing needs a corruption applied by the server (known label)")
        route = params["type"]
    else:
        route = predicted
    if route == "clean":
        out, t_exp = x, 0.0  # identity bypass - no expert is executed
    else:
        (o,), t_exp = r.run(route, xin)
        out = im.from_nchw(o)
    return {
        "task": "Hard-Routed Restoration", "source": origin, "routing_mode": routing,
        "clean": im.to_data_url(clean) if clean is not None else None,
        "input": im.to_data_url(x), "output": im.to_data_url(out), "corruption": describe(params),
        "probabilities": {c: round(float(v), 5) for c, v in zip(CLASSES, probs)},
        "predicted": predicted, "confidence": round(float(probs.max()), 5),
        "selected_expert": route, "selected_expert_label": EXPERT_LABELS[route],
        "true_label": params["type"] if params else None,
        "correct": (predicted == params["type"]) if params else None,
        **quality(out, x, clean),
        "timing_ms": {"preprocess": round(t_pre, 2), "classifier": round(t_clf, 2), "expert": round(t_exp, 2), "inference": round(t_clf + t_exp, 2)},
    }


@app.post("/api/restore/soft")
async def restore_soft(
    file: UploadFile | None = File(None), sample: str | None = Form(None),
    corruption: CorruptionT = CorruptionForm, severity: SeverityT = SeverityForm, seed: int | None = Form(None),
    p: float | None = Form(None), kernel: int | None = Form(None), sigma: float | None = Form(None),
    n_rects: int | None = Form(None), area: float | None = Form(None),
):
    r = reg()
    r.require("moe")
    src, x, clean, params, origin, t_pre = await prepare(file, sample, corruption, severity, seed, p, kernel, sigma, n_rects, area)
    (out, w, logits), t_inf = r.run("moe", im.to_nchw(x))
    out, w = im.from_nchw(out), w[0]
    branches = ["identity", "salt_pepper", "blur", "occlusion"]
    entropy = float(-(w * np.log(np.clip(w, 1e-12, 1))).sum() / math.log(len(w)))
    order = np.argsort(-w)
    return {
        "task": "Soft Mixture-of-Experts Restoration", "model": "t3_soft_moe.onnx", "source": origin,
        "clean": im.to_data_url(clean) if clean is not None else None,
        "input": im.to_data_url(x), "output": im.to_data_url(out), "corruption": describe(params),
        "weights": {b: round(float(v), 5) for b, v in zip(branches, w)},
        "dominant": branches[int(order[0])],
        "ranking": [branches[int(i)] for i in order],
        "normalized_entropy": round(entropy, 4),
        "routing_style": "dominant" if w.max() >= 0.7 else "distributed",
        "true_label": params["type"] if params else None,
        **quality(out, x, clean),
        "timing_ms": {"preprocess": round(t_pre, 2), "inference": round(t_inf, 2)},
    }


@app.post("/api/sketch")
async def sketch(file: UploadFile | None = File(None), sample: str | None = Form(None), style: int = Form(1)):
    r = reg()
    r.require("sketch")
    if style not in (1, 2, 3):
        raise HTTPException(422, "style must be 1, 2 or 3")
    t0 = time.perf_counter()
    photo, origin = await load_source(file, sample, crop_square=True)
    t_pre = (time.perf_counter() - t0) * 1000
    x = im.to_nchw(photo) * 2 - 1
    (y,), t_inf = r.run("sketch", x, np.array([style - 1], dtype=np.int64))
    out = im.from_nchw((y + 1) / 2)
    return {
        "task": "Face-to-Sketch Generator", "model": "t4_generator.onnx", "source": origin,
        "style": style, "style_name": STYLE_NAMES[style - 1],
        "photo": im.to_data_url(photo), "sketch": im.to_data_url(out),
        "timing_ms": {"preprocess": round(t_pre, 2), "inference": round(t_inf, 2)},
    }
