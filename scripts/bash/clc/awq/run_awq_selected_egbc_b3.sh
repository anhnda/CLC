#!/usr/bin/env bash
set -euo pipefail

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
RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-1}"
RUN_RAW_QUANTIZE="${RUN_RAW_QUANTIZE:-1}"

FLOAT_ARGS_BASE=(
  --models-root "$MODELS_ROOT"
  --results-eval-dir "$RESULTS_EVAL_DIR"
  --eval-cache-dir "$EVAL_CACHE_DIR"
  --seed "$SEED"
  --stride "$STRIDE"
  --max-length "$MAX_LENGTH"
  --c4-samples "$C4_SAMPLES"
  --lm-eval-task-preset "$LM_EVAL_TASK_PRESET"
)

QUANT_BASE_ARGS_BASE=(
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
  FLOAT_ARGS_BASE+=(--no-lm-eval)
  QUANT_BASE_ARGS_BASE+=(--no-lm-eval)
fi

if [ "$INCLUDE_C4" != "1" ]; then
  FLOAT_ARGS_BASE+=(--no-c4)
  QUANT_BASE_ARGS_BASE+=(--no-c4)
fi

if [ "$USE_WANDB" = "1" ]; then
  FLOAT_ARGS_BASE+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  QUANT_BASE_ARGS_BASE+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [ -n "$WANDB_ENTITY" ]; then
    FLOAT_ARGS_BASE+=(--wandb-entity "$WANDB_ENTITY")
    QUANT_BASE_ARGS_BASE+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

ORIGIN_METHOD="awq"
POST_CORRECTION="clc"
BITS="3"

MODEL_PATHS=(
  "mistralai/Mistral-7B-v0.3"
  "meta-llama/Meta-Llama-3.1-8B"
  "Qwen/Qwen2.5-7B"
  "meta-llama/Meta-Llama-3-8B"
)

get_params_for_model() {
  local model_path="$1"
  case "$model_path" in
    "mistralai/Mistral-7B-v0.3") echo "0.02 0.02" ;;
    "meta-llama/Meta-Llama-3.1-8B") echo "0.02 0.01" ;;
    "Qwen/Qwen2.5-7B") echo "0.03 0.02" ;;
    "meta-llama/Meta-Llama-3-8B") echo "0.01 0.05" ;;
    *) echo "Unknown model: $model_path" >&2; return 1 ;;
  esac
}

for model_path in "${MODEL_PATHS[@]}"; do
  read -r knee max_flip < <(get_params_for_model "$model_path")

  MODEL_SLUG="${model_path##*/}"
  FLOAT_RUN_NAME="${FLOAT_RUN_NAME:-${ORIGIN_METHOD}_float_${MODEL_SLUG}}"
  RAW_RUN_NAME="${RAW_RUN_NAME:-${ORIGIN_METHOD}_raw_${MODEL_SLUG}}"
  RUN_NAME="${ORIGIN_METHOD}_clc_${MODEL_SLUG}_b${BITS}_k${knee}_f${max_flip}"

  FLOAT_ARGS=("${FLOAT_ARGS_BASE[@]}" --model-path "$model_path")
  QUANT_BASE_ARGS=("${QUANT_BASE_ARGS_BASE[@]}" --model-path "$model_path")

  if [ "$RUN_FLOAT_MODEL" = "1" ]; then
    echo "==> float_model :: ${model_path}"
    "$PYTHON_BIN" main.py float_model "${FLOAT_ARGS[@]}" --run-name "$FLOAT_RUN_NAME"
  else
    echo "==> skipping float_model :: ${model_path}"
  fi

  if [ "$RUN_RAW_QUANTIZE" = "1" ]; then
    echo "==> raw_quantize :: ${model_path} :: origin=${ORIGIN_METHOD} :: bits=${BITS}"
    RAW_ARGS=(
      "${QUANT_BASE_ARGS[@]}"
      --origin-method "$ORIGIN_METHOD"
      --post-correction none
      --run-name "$RAW_RUN_NAME"
      --bits "$BITS"
    )
    "$PYTHON_BIN" main.py quantize "${RAW_ARGS[@]}"
  else
    echo "==> skipping raw_quantize :: ${model_path}"
  fi

  echo "==> clc :: ${model_path} :: origin=${ORIGIN_METHOD} :: bits=${BITS} :: knee=${knee} :: max_flip=${max_flip}"
  QUANT_ARGS=(
    "${QUANT_BASE_ARGS[@]}"
    --origin-method "$ORIGIN_METHOD"
    --post-correction "$POST_CORRECTION"
    --bits "$BITS"
    --knee-tolerance "$knee"
    --max-flip-percent "$max_flip"
    --run-name "$RUN_NAME"
  )
  "$PYTHON_BIN" main.py quantize "${QUANT_ARGS[@]}"
done

