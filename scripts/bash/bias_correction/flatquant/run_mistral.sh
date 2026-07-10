#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-v0.3}"
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

FLATQUANT_EPOCHS="${FLATQUANT_EPOCHS:-15}"
FLATQUANT_CALI_BSZ="${FLATQUANT_CALI_BSZ:-4}"
FLATQUANT_LR="${FLATQUANT_LR:-5e-3}"
FLATQUANT_DIAG_INIT="${FLATQUANT_DIAG_INIT:-sq_style}"
FLATQUANT_DIAG_ALPHA="${FLATQUANT_DIAG_ALPHA:-0.3}"
FLATQUANT_CALI_TRANS="${FLATQUANT_CALI_TRANS:-1}"
FLATQUANT_ADD_DIAG="${FLATQUANT_ADD_DIAG:-1}"
FLATQUANT_LWC="${FLATQUANT_LWC:-1}"
FLATQUANT_LAC="${FLATQUANT_LAC:-1}"

add_flatquant_args() {
  local -n args_ref=$1
  args_ref+=(
    --flatquant-epochs "$FLATQUANT_EPOCHS"
    --flatquant-cali-bsz "$FLATQUANT_CALI_BSZ"
    --flatquant-lr "$FLATQUANT_LR"
    --flatquant-diag-init "$FLATQUANT_DIAG_INIT"
    --flatquant-diag-alpha "$FLATQUANT_DIAG_ALPHA"
  )
  if [ "$FLATQUANT_CALI_TRANS" = "1" ]; then
    args_ref+=(--flatquant-cali-trans)
  else
    args_ref+=(--no-flatquant-cali-trans)
  fi
  if [ "$FLATQUANT_ADD_DIAG" = "1" ]; then
    args_ref+=(--flatquant-add-diag)
  else
    args_ref+=(--no-flatquant-add-diag)
  fi
  if [ "$FLATQUANT_LWC" = "1" ]; then
    args_ref+=(--flatquant-lwc)
  else
    args_ref+=(--no-flatquant-lwc)
  fi
  if [ "$FLATQUANT_LAC" = "1" ]; then
    args_ref+=(--flatquant-lac)
  else
    args_ref+=(--no-flatquant-lac)
  fi
}

ORIGIN_METHOD="flatquant"
POST_CORRECTION="bias_correction"
MODEL_SLUG="${MODEL_PATH##*/}"
FLOAT_RUN_NAME="${FLOAT_RUN_NAME:-${ORIGIN_METHOD}_float_${MODEL_SLUG}}"
RAW_RUN_NAME="${RAW_RUN_NAME:-${ORIGIN_METHOD}_raw_${MODEL_SLUG}}"
RAW_MODEL_DIR="${RAW_MODEL_DIR:-${RESULTS_MODELS_DIR}/${ORIGIN_METHOD}_raw/${RAW_RUN_NAME}}"
CORR_RUN_NAME="${CORR_RUN_NAME:-${ORIGIN_METHOD}_bias_correction_${MODEL_SLUG}}"
BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"
BITS="${BITS:-4}"

if [ "$RUN_FLOAT_MODEL" = "1" ]; then
  echo "==> float_model :: ${MODEL_PATH}"
  "$PYTHON_BIN" main.py float_model   "${FLOAT_ARGS[@]}"   --run-name "$FLOAT_RUN_NAME"
else
  echo "==> skipping float_model :: ${MODEL_PATH}"
fi

if [ "$RUN_RAW_QUANTIZE" = "1" ]; then
  echo "==> raw_quantize :: ${MODEL_PATH} :: origin=${ORIGIN_METHOD}"
  RAW_ARGS=(
    "${QUANT_BASE_ARGS[@]}"
    --origin-method "$ORIGIN_METHOD"
    --post-correction none
    --run-name "$RAW_RUN_NAME"
    --bits "$BITS"
  )
  add_flatquant_args RAW_ARGS
  "$PYTHON_BIN" main.py quantize "${RAW_ARGS[@]}"
else
  if [ ! -f "$RAW_MODEL_DIR/flat_parameters.pth" ]; then
    echo "Missing raw FlatQuant parameters at ${RAW_MODEL_DIR}/flat_parameters.pth" >&2
    exit 1
  fi
  echo "==> skipping raw_quantize :: ${MODEL_PATH} :: using existing raw model at ${RAW_MODEL_DIR}"
fi

echo "==> bias_correction :: ${MODEL_PATH} :: origin=${ORIGIN_METHOD}"
CORR_ARGS=(
  "${QUANT_BASE_ARGS[@]}"
  --origin-method "$ORIGIN_METHOD"
  --post-correction "$POST_CORRECTION"
  --bias-correction-samples "$BIAS_CORRECTION_SAMPLES"
  --run-name "$CORR_RUN_NAME"
  --flatquant-raw-path "$RAW_MODEL_DIR"
  --bits "$BITS"
)
add_flatquant_args CORR_ARGS
"$PYTHON_BIN" main.py quantize "${CORR_ARGS[@]}"
