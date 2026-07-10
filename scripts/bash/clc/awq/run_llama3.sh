#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-500}"
LM_EVAL_TASK_PRESET="${LM_EVAL_TASK_PRESET:-extended}"
INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"
INCLUDE_C4="${INCLUDE_C4:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-egbc}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

FLOAT_ARGS=(
  --model-path "$MODEL_PATH"
  --models-root "$MODELS_ROOT"
  --results-eval-dir "$RESULTS_EVAL_DIR"
  --eval-cache-dir "$EVAL_CACHE_DIR"
  --seed "$SEED"
  --stride "$STRIDE"
  --max-length "$MAX_LENGTH"
  --c4-samples "$C4_SAMPLES"
  --lm-eval-task-preset "$LM_EVAL_TASK_PRESET"
)

QUANT_BASE_ARGS=(
  --model-path "$MODEL_PATH"
  --models-root "$MODELS_ROOT"
  --results-models-dir "$RESULTS_MODELS_DIR"
  --results-eval-dir "$RESULTS_EVAL_DIR"
  --calibration-cache-dir "$CALIBRATION_CACHE_DIR"
  --eval-cache-dir "$EVAL_CACHE_DIR"
  --calib-dataset "$CALIB_DATASET"
  --n-calib "$N_CALIB"
  --calib-seqlen "$CALIB_SEQLEN"
  --seed "$SEED"
  --stride "$STRIDE"
  --max-length "$MAX_LENGTH"
  --c4-samples "$C4_SAMPLES"
  --lm-eval-task-preset "$LM_EVAL_TASK_PRESET"
)

if [ "$INCLUDE_LM_EVAL" != "1" ]; then
  FLOAT_ARGS+=(--no-lm-eval)
  QUANT_BASE_ARGS+=(--no-lm-eval)
fi

if [ "$INCLUDE_C4" != "1" ]; then
  FLOAT_ARGS+=(--no-c4)
  QUANT_BASE_ARGS+=(--no-c4)
fi

if [ "$USE_WANDB" = "1" ]; then
  FLOAT_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  QUANT_BASE_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [ -n "$WANDB_ENTITY" ]; then
    FLOAT_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    QUANT_BASE_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

ORIGIN_METHOD="awq"
POST_CORRECTION="clc"
FLOAT_RUN_NAME="${FLOAT_RUN_NAME:-${ORIGIN_METHOD}_float}"
RAW_RUN_NAME="${RAW_RUN_NAME:-${ORIGIN_METHOD}_raw}"
BITS_VALUES=(4)
KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)
MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)

echo "==> float_model :: ${MODEL_PATH}"
"$PYTHON_BIN" main.py float_model   "${FLOAT_ARGS[@]}"   --run-name "$FLOAT_RUN_NAME"

echo "==> raw_quantize :: ${MODEL_PATH} :: origin=${ORIGIN_METHOD}"
RAW_ARGS=(
  "${QUANT_BASE_ARGS[@]}"
  --origin-method "$ORIGIN_METHOD"
  --post-correction none
  --run-name "$RAW_RUN_NAME"
  --bits "4"
)

"$PYTHON_BIN" main.py quantize "${RAW_ARGS[@]}"

for bits in "${BITS_VALUES[@]}"; do
  for knee in "${KNEE_VALUES[@]}"; do
    for max_flip in "${MAX_FLIP_VALUES[@]}"; do
      run_name="${ORIGIN_METHOD}_clc_b${bits}_k${knee}_f${max_flip}"
      echo "==> clc :: ${MODEL_PATH} :: origin=${ORIGIN_METHOD} :: bits=${bits} :: knee=${knee} :: max_flip=${max_flip}"
      QUANT_ARGS=(
        "${QUANT_BASE_ARGS[@]}"
        --origin-method "$ORIGIN_METHOD"
        --post-correction "$POST_CORRECTION"
        --bits "$bits"
        --knee-tolerance "$knee"
        --max-flip-percent "$max_flip"
        --run-name "$run_name"
      )

      "$PYTHON_BIN" main.py quantize "${QUANT_ARGS[@]}"
    done
  done
done
