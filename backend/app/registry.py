"""Loads the ONNX models once at startup and exposes timed inference helpers."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parents[2] / "models"))

MODEL_FILES = {
    "universal": "t1_universal_dae.onnx",
    "classifier": "t2_classifier.onnx",
    "salt_pepper": "t2_specialist_salt_pepper.onnx",
    "blur": "t2_specialist_blur.onnx",
    "occlusion": "t2_specialist_occlusion.onnx",
    "moe": "t3_soft_moe.onnx",
    "sketch": "t4_generator.onnx",
}


class Registry:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.dir = models_dir
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.errors: dict[str, str] = {}
        self.info: dict = {}
        self.parity: dict = {}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", 4))
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        for key, fname in MODEL_FILES.items():
            path = self.dir / fname
            if not path.exists():
                self.errors[key] = f"missing {fname}"
                continue
            try:
                self.sessions[key] = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
            except Exception as e:  # noqa: BLE001 - surface any load failure through /health
                self.errors[key] = f"{type(e).__name__}: {e}"
        if (self.dir / "model_info.json").exists():
            self.info = json.loads((self.dir / "model_info.json").read_text())
        for p in self.dir.glob("parity_*.json"):
            self.parity[p.stem.removeprefix("parity_")] = json.loads(p.read_text())
        self._warmup()

    def _warmup(self) -> None:
        x = np.zeros((1, 3, 128, 128), np.float32)
        for key, s in self.sessions.items():
            feed = {s.get_inputs()[0].name: x}
            if key == "sketch":
                feed[s.get_inputs()[1].name] = np.zeros((1,), np.int64)
            s.run(None, feed)

    def require(self, *keys: str) -> None:
        from fastapi import HTTPException

        missing = [k for k in keys if k not in self.sessions]
        if missing:
            raise HTTPException(503, f"model(s) not loaded: {', '.join(missing)} - see /api/health")

    def run(self, key: str, *inputs: np.ndarray) -> tuple[list[np.ndarray], float]:
        s = self.sessions[key]
        feed = {i.name: x for i, x in zip(s.get_inputs(), inputs)}
        t0 = time.perf_counter()
        out = s.run(None, feed)
        return out, (time.perf_counter() - t0) * 1000.0

    def status(self) -> dict:
        return {
            "loaded": {k: {"file": MODEL_FILES[k], "providers": s.get_providers(), "inputs": [i.name for i in s.get_inputs()], "outputs": [o.name for o in s.get_outputs()]} for k, s in self.sessions.items()},
            "missing": self.errors,
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
        }
