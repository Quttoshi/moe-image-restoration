"""Copy a few held-out test images into backend/samples for the application's sample gallery.

    uv run python scripts/make_samples.py

Oxford-IIIT Pet images are CC BY-SA 4.0; FS2K photos are for research use (see README).
Images are stored at their original resolution (the API resizes to 128x128).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import fs2k, pets  # noqa: E402

OUT = ROOT / "backend" / "samples"


def main(n_pets: int = 12, n_faces_per_style: int = 3) -> None:
    rng = np.random.default_rng(2525)
    split = json.loads((pets.PROC_DIR / "pets_split.json").read_text())
    (OUT / "pets").mkdir(parents=True, exist_ok=True)
    (OUT / "faces").mkdir(parents=True, exist_ok=True)
    for name in rng.choice(split["test_files"], n_pets, replace=False):
        img = Image.open(pets.RAW_DIR / "oxford-iiit-pet" / "images" / name).convert("RGB")
        img.thumbnail((320, 320))
        img.save(OUT / "pets" / f"{Path(name).stem}.png")
    anno = json.loads((fs2k.FS2K_DIR / "anno_test.json").read_text())
    for style in range(3):
        cand = [a["image_name"] for a in anno if int(a["style"]) == style]
        for name in rng.choice(cand, min(n_faces_per_style, len(cand)), replace=False):
            photo, _ = fs2k.pair_paths(name)
            img = Image.open(photo).convert("RGB")
            img.thumbnail((320, 320))
            img.save(OUT / "faces" / f"style{style + 1}_{name.replace('/', '_')}.png")
    print(sorted(p.name for p in OUT.rglob("*.png")))


if __name__ == "__main__":
    main()
