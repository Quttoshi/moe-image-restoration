"""End-to-end API tests against untrained ONNX models with the production signatures."""
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    models = tmp_path_factory.mktemp("models")
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_dummy_models import main as make

    make(models)
    samples = tmp_path_factory.mktemp("samples")
    for g in ("pets", "faces"):
        (samples / g).mkdir()
        Image.fromarray((np.random.default_rng(0).random((160, 200, 3)) * 255).astype(np.uint8)).save(samples / g / "a.png")
    os.environ["MODELS_DIR"] = str(models)
    os.environ["SAMPLES_DIR"] = str(samples)
    sys.path.insert(0, str(ROOT / "backend"))
    for m in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[m]
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def png_bytes(w=150, h=120):
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(1).random((h, w, 3)) * 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and len(r["loaded"]) == 7


def test_samples(client):
    s = client.get("/api/samples").json()
    assert s["pets"] == ["pets/a.png"]
    assert client.get("/api/samples/pets/a.png").status_code == 200
    assert client.get("/api/samples/pets/..%2F..%2Fsecret").status_code == 404


@pytest.mark.parametrize("corruption,severity", [("none", "medium"), ("salt_pepper", "low"), ("blur", "high"), ("occlusion", "random")])
def test_universal(client, corruption, severity):
    r = client.post("/api/restore/universal", data={"sample": "pets/a.png", "corruption": corruption, "severity": severity, "seed": 3})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["output"].startswith("data:image/png;base64,") and "inference" in j["timing_ms"]
    assert (j["clean"] is None) == (corruption == "none")


def test_upload_and_custom(client):
    files = {"file": ("x.png", png_bytes(), "image/png")}
    r = client.post("/api/restore/universal", files=files, data={"corruption": "blur", "severity": "custom", "kernel": 5, "sigma": 1.2})
    assert r.status_code == 200, r.text
    assert r.json()["corruption"]["sigma"] == 1.2


def test_validation(client):
    assert client.post("/api/restore/universal", data={}).status_code == 400
    bad = {"file": ("x.txt", b"hello", "text/plain")}
    assert client.post("/api/restore/universal", files=bad).status_code == 415
    fake = {"file": ("x.png", b"not an image", "image/png")}
    assert client.post("/api/restore/universal", files=fake).status_code == 400
    r = client.post("/api/restore/universal", data={"sample": "pets/a.png", "corruption": "salt_pepper", "severity": "custom", "p": 0.9})
    assert r.status_code == 422


def test_hard(client):
    r = client.post("/api/restore/hard", data={"sample": "pets/a.png", "corruption": "occlusion", "severity": "high"}).json()
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
    assert r["selected_expert"] == r["predicted"]
    r = client.post("/api/restore/hard", data={"sample": "pets/a.png", "corruption": "blur", "severity": "low", "routing": "oracle"}).json()
    assert r["selected_expert"] == "blur"
    assert client.post("/api/restore/hard", data={"sample": "pets/a.png", "routing": "oracle"}).status_code == 422


def test_soft(client):
    r = client.post("/api/restore/soft", data={"sample": "pets/a.png", "corruption": "salt_pepper", "severity": "medium"}).json()
    assert abs(sum(r["weights"].values()) - 1) < 1e-3 and r["dominant"] in r["weights"]


def test_sketch(client):
    for style in (1, 2, 3):
        r = client.post("/api/sketch", files={"file": ("f.png", png_bytes(200, 120), "image/png")}, data={"style": style})
        assert r.status_code == 200 and r.json()["style_name"] == f"Style {style}"
    assert client.post("/api/sketch", data={"sample": "faces/a.png", "style": 4}).status_code == 422
