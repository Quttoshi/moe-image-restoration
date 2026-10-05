"""FS2K paired photo -> sketch data for Task 4.

Expected raw layout (official release, unzipped into data/raw/FS2K):
    FS2K/photo/photo{1,2,3}/imageXXXX.{jpg,png}
    FS2K/sketch/sketch{1,2,3}/sketchXXXX.{jpg,png}
    FS2K/anno_train.json, FS2K/anno_test.json   (official split; 'style' in {0,1,2})

The sketch path is derived from the photo path exactly as the official
tools/split_train_test.py does, which guarantees correct pairing.
A stratified (by style) 15% validation split is carved from the official
training portion with seed 42; the official test set is never touched in training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.data.pets import PROC_DIR, RAW_DIR

FS2K_DIR = RAW_DIR / "FS2K"
IMG_SIZE = 128
SPLIT_SEED = 42
N_STYLES = 3
STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]


def _with_ext(base: Path) -> Path:
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    raise FileNotFoundError(base)


def pair_paths(image_name: str) -> tuple[Path, Path]:
    photo = _with_ext(FS2K_DIR / "photo" / image_name)
    sketch_name = image_name.replace("image", "sketch").replace("photo", "sketch")
    sketch = _with_ext(FS2K_DIR / "sketch" / sketch_name)
    return photo, sketch


def _load_rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))


def _read_split(split: str) -> list[dict]:
    return json.loads((FS2K_DIR / f"anno_{split}.json").read_text())


def prepare(force: bool = False) -> None:
    out = PROC_DIR / f"fs2k_{IMG_SIZE}.npz"
    if out.exists() and not force:
        return
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split in ("train", "test"):
        anno = _read_split(split)
        photos, sketches, styles, names = [], [], [], []
        for a in anno:
            ph, sk = pair_paths(a["image_name"])
            photos.append(_load_rgb(ph))
            sketches.append(_load_rgb(sk))
            styles.append(int(a["style"]))
            names.append(a["image_name"])
        arrays[f"{split}_photo"] = np.stack(photos)
        arrays[f"{split}_sketch"] = np.stack(sketches)
        arrays[f"{split}_style"] = np.array(styles)
        arrays[f"{split}_name"] = np.array(names)
    # Normalise style ids to 0..2 in case a release uses 1..3.
    uniq = sorted(set(arrays["train_style"].tolist()) | set(arrays["test_style"].tolist()))
    assert len(uniq) == N_STYLES, f"expected 3 styles, found {uniq}"
    remap = {s: i for i, s in enumerate(uniq)}
    for split in ("train", "test"):
        arrays[f"{split}_style"] = np.array([remap[s] for s in arrays[f"{split}_style"]])

    idx = np.arange(len(arrays["train_style"]))
    tr, va = train_test_split(idx, test_size=0.15, random_state=SPLIT_SEED, stratify=arrays["train_style"])
    arrays["train_idx"], arrays["val_idx"] = np.sort(tr), np.sort(va)
    np.savez_compressed(out, **arrays)
    info = {
        "seed": SPLIT_SEED,
        "style_remap": {str(k): v for k, v in remap.items()},
        "counts": {
            s: np.bincount(arrays["train_style"][i] if s != "test" else arrays["test_style"], minlength=3).tolist()
            for s, i in (("train", arrays["train_idx"]), ("val", arrays["val_idx"]), ("test", None))
        },
        "val_names": arrays["train_name"][arrays["val_idx"]].tolist(),
    }
    (PROC_DIR / "fs2k_split.json").write_text(json.dumps(info, indent=1))


def load(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (photos uint8 NHWC, sketches uint8 NHWC, styles int) for train/val/test."""
    f = PROC_DIR / f"fs2k_{IMG_SIZE}.npz"
    if not f.exists():
        prepare()
    d = np.load(f)
    if split == "test":
        return d["test_photo"], d["test_sketch"], d["test_style"]
    idx = d[f"{split}_idx"]
    return d["train_photo"][idx], d["train_sketch"][idx], d["train_style"][idx]


def to_m11(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> float CHW in [-1, 1] (generator uses tanh)."""
    return torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0


class PairedSketchDataset(Dataset):
    """Returns (photo, sketch, style). With ``augment`` the *same* random jitter
    (upscale to 142 + identical random crop) and the *same* horizontal flip are
    applied to both images, preserving pixel correspondence."""

    def __init__(self, split: str, augment: bool = False):
        self.photos, self.sketches, self.styles = load(split)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.styles)

    def __getitem__(self, i: int):
        x, y = to_m11(self.photos[i]), to_m11(self.sketches[i])
        if self.augment:
            pair = torch.cat([x, y], 0)[None]
            pair = F.interpolate(pair, size=142, mode="bilinear", align_corners=False)[0]
            top, left = torch.randint(0, 142 - IMG_SIZE + 1, (2,)).tolist()
            pair = pair[:, top : top + IMG_SIZE, left : left + IMG_SIZE]
            if torch.rand(()) < 0.5:
                pair = pair.flip(-1)
            x, y = pair[:3], pair[3:]
        return x, y, int(self.styles[i])
