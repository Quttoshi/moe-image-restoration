#!/usr/bin/env bash
# Deadline run: retrain the blur/occlusion specialists with the 15-epoch-study configuration,
# re-evaluate Task 2, then train/evaluate/export Tasks 3 and 4 with the reduced budgets in configs/.
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
LOG=outputs/logs; mkdir -p "$LOG"
step() { local name=$1; shift; echo "[start] $name $(date +%H:%M:%S)"; "$@" >"$LOG/$name.log" 2>&1 || { echo "[FAIL] $name"; tail -20 "$LOG/$name.log"; exit 1; }; echo "[done] $name $(date +%H:%M:%S)"; }

rm -f checkpoints/t2_specialist_blur.pt checkpoints/t2_specialist_occlusion.pt
step t2_spec_train_blur      uv run python -m src.train.t2_specialists train --kind blur
step t2_spec_train_occlusion uv run python -m src.train.t2_specialists train --kind occlusion
rm -rf outputs/t2_eval
step t2_eval   uv run python -m src.eval.eval_t2
step t2_export uv run python -m src.export.export_all --only t2
bash scripts/run_pipeline.sh t3 t4
step plots uv run python -m src.eval.plots
echo "[all done] $(date +%H:%M:%S)"
