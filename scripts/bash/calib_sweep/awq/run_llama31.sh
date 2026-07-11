#!/usr/bin/env bash
# Calibration-size sweep (perplexity only) on LLaMA-3.1-8B, AWQ 4-bit.
#
# FP16 perplexity is measured once (calibration-independent). For each C4
# calibration size in CALIB_SWEEP, AWQ (base) and AWQ+CLC are quantized, scored
# on WikiText-2 + C4 perplexity, then their model dirs are deleted before the
# next size so disk usage stays flat.
#
# Defaults here: knee-tolerance = -10 (full support, knee mask effectively off)
#                max-flip-percent = 1 (flip cap = 100% per channel).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"

CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-500}"

BITS="${BITS:-4}"
KNEE_TOLERANCE="${KNEE_TOLERANCE:--10}"
MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-1}"

# Calibration sizes to sweep (space-separated).
CALIB_SWEEP="${CALIB_SWEEP:-64 128 256 512}"

RUN_NAME="${RUN_NAME:-llama31_8b_awq_clc_calibsweep_b${BITS}}"

echo "==> calib_sweep :: ${MODEL_PATH} :: AWQ ${BITS}-bit :: sizes=[${CALIB_SWEEP}]"
echo "    knee_tolerance=${KNEE_TOLERANCE} (full support)  max_flip_percent=${MAX_FLIP_PERCENT}"
"$PYTHON_BIN" main.py calib_sweep \
  --model-path "$MODEL_PATH" \
  --models-root "$MODELS_ROOT" \
  --origin-method "awq" \
  --results-models-dir "$RESULTS_MODELS_DIR" \
  --results-eval-dir "$RESULTS_EVAL_DIR" \
  --calibration-cache-dir "$CALIBRATION_CACHE_DIR" \
  --eval-cache-dir "$EVAL_CACHE_DIR" \
  --calib-seqlen "$CALIB_SEQLEN" \
  --seed "$SEED" \
  --stride "$STRIDE" \
  --max-length "$MAX_LENGTH" \
  --c4-samples "$C4_SAMPLES" \
  --bits "$BITS" \
  --knee-tolerance "$KNEE_TOLERANCE" \
  --max-flip-percent "$MAX_FLIP_PERCENT" \
  --calib-sweep ${CALIB_SWEEP} \
  --run-name "$RUN_NAME" \
  --no-lm-eval \
  --no-wandb

echo "==> sweep results written to ${RESULTS_EVAL_DIR}/${RUN_NAME}.json"
