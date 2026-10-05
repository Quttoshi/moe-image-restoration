"""Download and prepare all datasets, then build the corruption manifests.

    uv run python scripts/download_data.py            # both datasets
    uv run python scripts/download_data.py --only pets

* Oxford-IIIT Pet  - via torchvision (official trainval/test lists).
* FS2K             - official Google Drive release (DengPingFan/FS2K). If the
  automatic download fails (Drive quota), download FS2K.zip manually from
  https://drive.google.com/file/d/1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp and place it in data/raw/.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import fs2k, manifests, pets  # noqa: E402

FS2K_GDRIVE_ID = "1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp"


def get_pets() -> None:
    arrays = pets.prepare()
    print({k: v.shape for k, v in arrays.items()})
    sys.argv = [sys.argv[0]]
    manifests.main()


def get_fs2k() -> None:
    zip_path = pets.RAW_DIR / "FS2K.zip"
    if not fs2k.FS2K_DIR.exists():
        if not zip_path.exists():
            import gdown

            pets.RAW_DIR.mkdir(parents=True, exist_ok=True)
            gdown.download(id=FS2K_GDRIVE_ID, output=str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(pets.RAW_DIR)
    fs2k.prepare()
    print((pets.PROC_DIR / "fs2k_split.json").read_text()[:400])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["pets", "fs2k"])
    a = ap.parse_args()
    if a.only in (None, "pets"):
        get_pets()
    if a.only in (None, "fs2k"):
        get_fs2k()
