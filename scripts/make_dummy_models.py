"""Export *untrained* models with the production names/signatures for API smoke tests.

    uv run python scripts/make_dummy_models.py [out_dir]     # default: outputs/dummy_models
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.export.onnx_utils as ou  # noqa: E402
from src.models.cgan import UNetGenerator  # noqa: E402
from src.models.classifier import CorruptionClassifier  # noqa: E402
from src.models.dae import ConvDAE  # noqa: E402
from src.models.moe import EXPERT_ORDER, SoftMoE  # noqa: E402


def main(out: Path) -> None:
    ou.MODELS_DIR = out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128)
    small = dict(base_ch=16, latent_ch=8, dropout=0.0)
    ou.export(ConvDAE(**small), (x,), "t1_universal_dae", ["input"], ["restored"])
    clf = CorruptionClassifier((16, 32, 64, 64))
    ou.export(clf, (x,), "t2_classifier", ["input"], ["logits"])
    experts = {k: ConvDAE(**small) for k in EXPERT_ORDER}
    for k, m in experts.items():
        ou.export(m, (x,), f"t2_specialist_{k}", ["input"], ["restored"])
    ou.export(SoftMoE(clf, experts, 1.0), (x,), "t3_soft_moe", ["input"], ["restored", "weights", "logits"])
    ou.export(UNetGenerator(16, 8, 0.0), (x * 2 - 1, torch.tensor([0])), "t4_generator", ["photo", "style"], ["sketch"])
    print("dummy models ->", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "outputs" / "dummy_models")
