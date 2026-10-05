"""Upload validation, preprocessing and PNG/base64 encoding."""
from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
SIZE = 128


async def read_upload(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"unsupported file type {file.content_type!r}; use PNG, JPEG, WebP or BMP")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "file is not a valid image")
    if min(img.size) < 16:
        raise HTTPException(400, "image is too small (min 16 px)")
    return ImageOps.exif_transpose(img).convert("RGB")


def to_array(img: Image.Image, crop_square: bool = False) -> np.ndarray:
    """PIL -> float32 HWC [0,1] at 128x128. ``crop_square`` centre-crops first (faces)."""
    if crop_square:
        img = ImageOps.fit(img, (min(img.size),) * 2, Image.BICUBIC)
    img = img.resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_nchw(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


def from_nchw(t: np.ndarray) -> np.ndarray:
    return np.clip(t[0].transpose(1, 2, 0), 0.0, 1.0)


def to_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """|pred - target| averaged over channels, mapped to a simple black-red-yellow ramp."""
    e = np.clip(np.abs(pred - target).mean(2) * 3, 0, 1)
    return np.stack([np.clip(e * 2, 0, 1), np.clip(e * 2 - 1, 0, 1), np.zeros_like(e)], axis=2)


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(((pred - target) ** 2).mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))
