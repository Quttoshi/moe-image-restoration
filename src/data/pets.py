"""Oxford-IIIT Pet data pipeline for Tasks 1-3.

* Official ``trainval`` -> 80/20 train/val split with seed 42; ``test`` is untouched.
* Images are converted to RGB and resized to 128x128 once, then cached as a
  uint8 ``.npy`` array (clean images only - no corrupted copies are stored).
* Corruptions are applied at load time (training) or replayed from a
  deterministic manifest (validation / test).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.data.corruptions import CLASS_TO_IDX, CLASSES, apply_corruption, sample_params

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MANIFEST_DIR = ROOT / "data" / "manifests"
IMG_SIZE = 128
SPLIT_SEED = 42


def _load_split(split: str) -> list[Path]:
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(root=str(RAW_DIR), split=split, download=True)
    return list(ds._images)


def _to_array(paths: list[Path]) -> np.ndarray:
    out = np.empty((len(paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        out[i] = np.asarray(img)
    return out


def split_indices(n: int, seed: int = SPLIT_SEED, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    n_train = int(round(train_frac * n))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def prepare(force: bool = False) -> dict[str, np.ndarray]:
    """Download (if needed), split and cache the 128x128 clean images."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    files = {s: PROC_DIR / f"pets_{s}_{IMG_SIZE}.npy" for s in ("train", "val", "test")}
    if force or not all(f.exists() for f in files.values()):
        trainval = _load_split("trainval")
        test = _load_split("test")
        tr_idx, va_idx = split_indices(len(trainval))
        arr = _to_array(trainval)
        np.save(files["train"], arr[tr_idx])
        np.save(files["val"], arr[va_idx])
        np.save(files["test"], _to_array(test))
        split_info = {
            "seed": SPLIT_SEED,
            "n_trainval": len(trainval),
            "n_test": len(test),
            "train_files": [trainval[i].name for i in tr_idx],
            "val_files": [trainval[i].name for i in va_idx],
            "test_files": [p.name for p in test],
        }
        (PROC_DIR / "pets_split.json").write_text(json.dumps(split_info, indent=1))
    return {s: np.load(f) for s, f in files.items()}


class LazyArray:
    """Memory-mapped .npy that pickles as just its path.

    DataLoader workers on Windows are spawned and receive a pickled copy of the
    dataset; this keeps that copy tiny and lets all workers share the OS page cache.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._arr = None
        self.shape = np.load(self.path, mmap_mode="r").shape

    def _get(self) -> np.ndarray:
        if self._arr is None:
            self._arr = np.load(self.path, mmap_mode="r")
        return self._arr

    def __getitem__(self, i):
        return np.asarray(self._get()[i])

    def __len__(self) -> int:
        return self.shape[0]

    def __getstate__(self):
        return {"path": self.path, "shape": self.shape}

    def __setstate__(self, s):
        self.path, self.shape, self._arr = s["path"], s["shape"], None


def load(split: str) -> LazyArray:
    f = PROC_DIR / f"pets_{split}_{IMG_SIZE}.npy"
    if not f.exists():
        prepare()
    return LazyArray(f)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class RuntimeCorruptionDataset(Dataset):
    """Applies a freshly sampled corruption every time an image is loaded.

    ``kinds`` restricts the set of input conditions (e.g. ``["blur"]`` for the
    blur specialist). Indexing with a plain int samples the condition uniformly;
    indexing with ``(idx, class_idx)`` forces the condition, which is what the
    :class:`BalancedBatchSampler` uses.

    Returns ``(corrupted, clean, label)`` with images as float CHW in [0, 1].
    """

    def __init__(self, images: np.ndarray, kinds: list[str] | None = None, hflip: bool = False):
        self.images = images
        self.kinds = kinds or list(CLASSES)
        self.hflip = hflip

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, item):
        # torch's RNG is seeded per DataLoader worker, so derive the NumPy RNG from it.
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        if isinstance(item, tuple):
            idx, label = item
            kind = CLASSES[label]
        else:
            idx = item
            kind = self.kinds[int(rng.integers(len(self.kinds)))]
        clean = self.images[idx].astype(np.float32) / 255.0
        if self.hflip and rng.random() < 0.5:
            clean = clean[:, ::-1].copy()
        corrupted = apply_corruption(clean, sample_params(rng, kind))
        return to_tensor(corrupted), to_tensor(clean), CLASS_TO_IDX[kind]


class BalancedBatchSampler(Sampler):
    """Yields batches with exactly ``batch_size / len(kinds)`` samples per condition.

    Each yielded element is ``(image_index, class_index)``; the dataset applies
    that condition. Images are drawn without replacement within an epoch.
    """

    def __init__(self, n_images: int, batch_size: int, kinds: list[str] | None = None, drop_last: bool = True):
        self.kinds = kinds or list(CLASSES)
        if batch_size % len(self.kinds):
            raise ValueError("batch_size must be divisible by the number of conditions")
        self.n, self.bs, self.drop_last = n_images, batch_size, drop_last

    def __len__(self) -> int:
        return self.n // self.bs if self.drop_last else -(-self.n // self.bs)

    def __iter__(self):
        perm = torch.randperm(self.n).tolist()
        labels = [CLASS_TO_IDX[k] for k in self.kinds]
        per = self.bs // len(self.kinds)
        for b in range(len(self)):
            chunk = perm[b * self.bs : (b + 1) * self.bs]
            lab = (labels * per)[: len(chunk)]
            lab = [lab[i] for i in torch.randperm(len(lab)).tolist()]
            yield list(zip(chunk, lab))


class ManifestDataset(Dataset):
    """Replays a deterministic corruption manifest (validation or test).

    Returns ``(corrupted, clean, label, entry_index)``; metadata such as the
    severity can be looked up with ``self.entries[entry_index]``.
    """

    def __init__(self, images: np.ndarray, manifest: str | Path | list[dict], kinds: list[str] | None = None):
        self.images = images
        if not isinstance(manifest, list):
            manifest = json.loads(Path(manifest).read_text())["entries"]
        self.entries = [e for e in manifest if kinds is None or e["params"]["type"] in kinds]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        clean = self.images[e["image_index"]].astype(np.float32) / 255.0
        corrupted = apply_corruption(clean, e["params"])
        return to_tensor(corrupted), to_tensor(clean), CLASS_TO_IDX[e["params"]["type"]], i
