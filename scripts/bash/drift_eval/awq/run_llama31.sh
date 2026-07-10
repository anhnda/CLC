#!/usr/bin/env bash
# Inter-layer mean-shift accumulation eval (E1 accumulated + E2 isolated).
# Builds FP16 / AWQ-base / AWQ+CLC and measures how the layer-output shift
# behaves across depth on a fixed evaluation stream.
#
# Mirrors scripts/bash/clc/awq/run_llama31.sh for model/calibration settings.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
SEED="${SEED:-42}"

# Drift-measurement stream (kept separate & small from the calibration set).
DRIFT_EVAL_DATASET="${DRIFT_EVAL_DATASET:-wikitext2}"
DRIFT_N_SAMPLES="${DRIFT_N_SAMPLES:-32}"
DRIFT_MAX_LENGTH="${DRIFT_MAX_LENGTH:-512}"

# CLC hyperparameters (defaults matching the main-paper config).
BITS="${BITS:-3}"
KNEE_TOLERANCE="${KNEE_TOLERANCE:-0.0}"
MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-0.05}"

ORIGIN_METHOD="awq"
RUN_NAME="${RUN_NAME:-${ORIGIN_METHOD}_drift_b${BITS}}"

echo "==> drift_eval :: ${MODEL_PATH} :: origin=${ORIGIN_METHOD} :: bits=${BITS}"
"$PYTHON_BIN" main.py drift_eval \
  --model-path "$MODEL_PATH" \
  --models-root "$MODELS_ROOT" \
  --origin-method "$ORIGIN_METHOD" \
  --results-eval-dir "$RESULTS_EVAL_DIR" \
  --calibration-cache-dir "$CALIBRATION_CACHE_DIR" \
  --calib-dataset "$CALIB_DATASET" \
  --n-calib "$N_CALIB" \
  --calib-seqlen "$CALIB_SEQLEN" \
  --seed "$SEED" \
  --bits "$BITS" \
  --knee-tolerance "$KNEE_TOLERANCE" \
  --max-flip-percent "$MAX_FLIP_PERCENT" \
  --drift-eval-dataset "$DRIFT_EVAL_DATASET" \
  --drift-n-samples "$DRIFT_N_SAMPLES" \
  --drift-max-length "$DRIFT_MAX_LENGTH" \
  --run-name "$RUN_NAME" \
  --no-lm-eval \
  --no-c4 \
  --no-wandb

echo "==> drift curves written under ${RESULTS_EVAL_DIR}/${RUN_NAME}_drift.json"
