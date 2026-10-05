"""Generate the deterministic validation and test corruption manifests (run once).

Validation: one condition per validation image, conditions assigned round-robin
over a seeded shuffle (balanced), parameters drawn from the training ranges.

Test: every test image appears 10 times - clean plus the three fixed severity
levels of each corruption.

Each entry stores the image index, its source file name, the per-entry seed and
the full parameter dict (S&P pixel seed, blur kernel/sigma, rectangle coords),
so the corruption can be reproduced exactly.

    uv run python -m src.data.manifests
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.data import pets
from src.data.corruptions import CLASSES, SEVERITIES, level_params, occluded_fraction, sample_params, severity_bucket

VAL_SEED = 2024
TEST_SEED = 4202


def build_val(files: list[str], seed: int = VAL_SEED) -> list[dict]:
    order = np.random.default_rng(seed).permutation(len(files))
    entries = []
    for rank, idx in enumerate(order):
        entry_seed = seed * 100_000 + int(idx)
        rng = np.random.default_rng(entry_seed)
        params = sample_params(rng, CLASSES[rank % len(CLASSES)])
        e = {
            "image_index": int(idx),
            "file": files[idx],
            "seed": entry_seed,
            "severity": severity_bucket(params),
            "params": params,
        }
        if params["type"] == "occlusion":
            e["covered_fraction"] = occluded_fraction(params["rects"])
        entries.append(e)
    return sorted(entries, key=lambda e: e["image_index"])


def build_test(files: list[str], seed: int = TEST_SEED) -> list[dict]:
    entries = []
    for idx, name in enumerate(files):
        entries.append({"image_index": idx, "file": name, "seed": None, "severity": "none", "params": {"type": "clean"}})
        for k, kind in enumerate(CLASSES[1:]):
            for s, sev in enumerate(SEVERITIES):
                entry_seed = seed * 100_000 + idx * 10 + k * 3 + s
                params = level_params(np.random.default_rng(entry_seed), kind, sev)
                e = {"image_index": idx, "file": name, "seed": entry_seed, "severity": sev, "params": params}
                if kind == "occlusion":
                    e["covered_fraction"] = occluded_fraction(params["rects"])
                entries.append(e)
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing manifests")
    args = ap.parse_args()

    pets.prepare()
    split = json.loads((pets.PROC_DIR / "pets_split.json").read_text())
    pets.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder, files, seed in (
        ("val", build_val, split["val_files"], VAL_SEED),
        ("test", build_test, split["test_files"], TEST_SEED),
    ):
        path = pets.MANIFEST_DIR / f"{name}_manifest.json"
        if path.exists() and not args.force:
            print(f"{path} exists - skipping (use --force to regenerate)")
            continue
        entries = builder(files, seed)
        path.write_text(json.dumps({"split": name, "base_seed": seed, "image_size": pets.IMG_SIZE, "entries": entries}))
        print(f"wrote {len(entries)} entries -> {path}")


if __name__ == "__main__":
    main()
