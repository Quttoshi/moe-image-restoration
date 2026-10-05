import numpy as np
import pytest
import torch

from src.data.corruptions import (
    CLASSES,
    TEST_LEVELS,
    apply_corruption,
    gaussian_blur,
    level_params,
    occluded_fraction,
    sample_params,
)
from src.data.manifests import build_test, build_val
from src.data.pets import BalancedBatchSampler, RuntimeCorruptionDataset

H = W = 128


@pytest.fixture
def img():
    return np.random.default_rng(0).random((H, W, 3), dtype=np.float32) * 0.8 + 0.1


def test_training_param_ranges():
    rng = np.random.default_rng(1)
    for _ in range(300):
        sp = sample_params(rng, "salt_pepper")
        assert 0.02 <= sp["p"] <= 0.15
        bl = sample_params(rng, "blur")
        assert bl["kernel"] in (3, 5, 7) and 0.5 <= bl["sigma"] <= 2.5
        oc = sample_params(rng, "occlusion")
        assert 1 <= len(oc["rects"]) <= 3
        assert 0.09 <= occluded_fraction(oc["rects"]) <= 0.36


def test_salt_pepper_fraction_and_values(img):
    params = {"type": "salt_pepper", "p": 0.15, "seed": 3}
    out = apply_corruption(img, params)
    changed = np.any(out != img, axis=2)
    assert abs(changed.mean() - 0.15) < 0.01
    vals = out[changed]
    assert np.all((vals == 0.0) | (vals == 1.0))
    white = np.all(vals == 1.0, axis=1).mean()
    assert 0.45 < white < 0.55


def test_blur_matches_torchvision(img):
    from torchvision.transforms.functional import gaussian_blur as tv_blur

    for k, s in ((3, 0.7), (5, 1.5), (7, 2.5)):
        ours = gaussian_blur(img, k, s)
        ref = tv_blur(torch.from_numpy(img.transpose(2, 0, 1)), [k, k], [s, s]).numpy().transpose(1, 2, 0)
        assert np.abs(ours - ref).max() < 1e-5


def test_test_levels_occlusion_coverage():
    rng = np.random.default_rng(5)
    for sev, cfg in TEST_LEVELS["occlusion"].items():
        for _ in range(50):
            p = level_params(rng, "occlusion", sev)
            assert len(p["rects"]) == cfg["n_rects"]
            assert abs(occluded_fraction(p["rects"]) - cfg["area"]) < 0.01


def test_occlusion_is_black(img):
    p = level_params(np.random.default_rng(2), "occlusion", "high")
    out = apply_corruption(img, p)
    x, y, w, h = p["rects"][0]
    assert np.all(out[y : y + h, x : x + w] == 0.0)


def test_manifests_deterministic():
    files = [f"img_{i}.jpg" for i in range(40)]
    assert build_val(files) == build_val(files)
    assert build_test(files[:5]) == build_test(files[:5])
    val = build_val(files)
    counts = {c: sum(e["params"]["type"] == c for e in val) for c in CLASSES}
    assert set(counts.values()) == {10}
    assert len(build_test(files[:5])) == 5 * 10


def test_balanced_sampler_and_dataset():
    images = (np.random.default_rng(0).random((64, H, W, 3)) * 255).astype(np.uint8)
    ds = RuntimeCorruptionDataset(images)
    sampler = BalancedBatchSampler(len(images), 16)
    for batch in sampler:
        labels = [lab for _, lab in batch]
        assert all(labels.count(k) == 4 for k in range(4))
        x, y, lab = ds[batch[0]]
        assert x.shape == (3, H, W) and y.shape == (3, H, W) and lab == batch[0][1]
