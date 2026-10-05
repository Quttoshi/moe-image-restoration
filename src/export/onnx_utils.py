"""ONNX export + PyTorch/ONNX Runtime parity verification shared by all tasks."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.train.common import ROOT

MODELS_DIR = ROOT / "models"


def export(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], name: str, input_names: list[str], output_names: list[str]) -> Path:
    """Export with a dynamic batch dimension on every input and output."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.onnx"
    model = model.eval().cpu()
    inputs = tuple(t.cpu() for t in inputs)
    axes = {n: {0: "batch"} for n in input_names + output_names}
    torch.onnx.export(model, inputs, str(path), input_names=input_names, output_names=output_names, dynamic_axes=axes, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def verify(model: torch.nn.Module, path: Path, inputs: tuple[torch.Tensor, ...], atol: float = 1e-4) -> dict:
    """Compare PyTorch and ONNX Runtime outputs on the same inputs; returns a report dict."""
    model = model.eval().cpu()
    with torch.no_grad():
        ref = model(*[t.cpu() for t in inputs])
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {i.name: t.cpu().numpy() for i, t in zip(sess.get_inputs(), inputs)}
    t0 = time.perf_counter()
    got = sess.run(None, feed)
    ms = (time.perf_counter() - t0) * 1000
    report = {"model": path.name, "batch": int(inputs[0].shape[0]), "ort_ms": round(ms, 2), "outputs": {}}
    ok = True
    for o, r, g in zip(sess.get_outputs(), ref, got):
        diff = np.abs(r.numpy() - g)
        report["outputs"][o.name] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
        ok &= bool(diff.max() < atol)
    report["passed"] = ok
    report["atol"] = atol
    return report


def write_report(reports: list[dict], name: str) -> Path:
    p = MODELS_DIR / f"parity_{name}.json"
    p.write_text(json.dumps(reports, indent=1))
    for r in reports:
        diffs = ", ".join(f"{k}: max {v['max_abs_diff']:.2e}" for k, v in r["outputs"].items())
        print(f"{'PASS' if r['passed'] else 'FAIL'} {r['model']} ({diffs})")
    return p
