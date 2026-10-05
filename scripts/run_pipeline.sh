#!/usr/bin/env bash
# Full research pipeline: Optuna studies -> final training -> test evaluation -> ONNX export.
# Each stage is skipped if its output already exists, so the script can be re-run after an interruption.
#   bash scripts/run_pipeline.sh            # all stages
#   bash scripts/run_pipeline.sh t2 t3      # selected tasks
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
LOG=outputs/logs; mkdir -p "$LOG"
TASKS=${*:-t1 t2 t3 t4}

stage() {  # stage <name> <done-marker-file> <command...>
  local name=$1 marker=$2; shift 2
  if [[ -n "$marker" && -e "$marker" ]]; then echo "[skip] $name ($marker exists)"; return; fi
  echo "[start] $name $(date +%H:%M:%S)"
  "$@" >"$LOG/$name.log" 2>&1 || { echo "[FAIL] $name - see $LOG/$name.log"; tail -20 "$LOG/$name.log"; exit 1; }
  echo "[done] $name $(date +%H:%M:%S)"
}

for t in $TASKS; do case $t in
  t1)
    stage t1_tune   outputs/logs/.t1_tune_done  bash -c "uv run python -m src.train.t1_udae tune && touch outputs/logs/.t1_tune_done"
    stage t1_train  checkpoints/t1_udae_final.pt uv run python -m src.train.t1_udae train
    stage t1_eval   outputs/t1_eval/summary.csv uv run python -m src.eval.eval_t1
    stage t1_export models/t1_universal_dae.onnx uv run python -m src.export.export_all --only t1 ;;
  t2)
    stage t2_clf_tune   outputs/logs/.t2_clf_tune_done  bash -c "uv run python -m src.train.t2_classifier tune && touch outputs/logs/.t2_clf_tune_done"
    stage t2_clf_train  checkpoints/t2_classifier.pt   uv run python -m src.train.t2_classifier train
    stage t2_spec_tune  outputs/logs/.t2_spec_tune_done bash -c "uv run python -m src.train.t2_specialists tune && touch outputs/logs/.t2_spec_tune_done"
    stage t2_spec_train checkpoints/t2_specialist_occlusion.pt uv run python -m src.train.t2_specialists train
    stage t2_eval   outputs/t2_eval/summary_predicted.csv uv run python -m src.eval.eval_t2
    stage t2_export models/t2_classifier.onnx uv run python -m src.export.export_all --only t2 ;;
  t3)
    stage t3_tune   outputs/logs/.t3_tune_done bash -c "uv run python -m src.train.t3_moe tune && touch outputs/logs/.t3_tune_done"
    stage t3_train  checkpoints/t3_moe.pt uv run python -m src.train.t3_moe train
    stage t3_eval   outputs/t3_eval/summary.csv uv run python -m src.eval.eval_t3
    stage t3_export models/t3_soft_moe.onnx uv run python -m src.export.export_all --only t3 ;;
  t4)
    stage t4_tune   outputs/logs/.t4_tune_done bash -c "uv run python -m src.train.t4_cgan tune && touch outputs/logs/.t4_tune_done"
    stage t4_train  checkpoints/t4_generator.pt uv run python -m src.train.t4_cgan train
    stage t4_eval   outputs/t4_eval/summary.csv uv run python -m src.eval.eval_t4
    stage t4_export models/t4_generator.onnx uv run python -m src.export.export_all --only t4 ;;
esac; done
echo "[pipeline complete] $(date +%H:%M:%S)"
