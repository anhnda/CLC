#!/usr/bin/env bash
# Compare FP16 / AWQ / AWQ+CLC on Qwen3-8B:
#   * perplexity (WikiText-2 + C4, same sliding-window eval as the main tables)
#   * GSM8K accuracy (lm-eval task gsm8k_cot, 8-shot), matching:
#       lm-eval --model hf --tasks gsm8k_cot --num_fewshot 8 --batch_size auto ...
#
# All three models are scored in one run so the numbers are directly comparable.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B-Base}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-./results/eval/lm_eval}"

CALIB_DATASET="${CALIB_DATASET:-c4}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-500}"

BITS="${BITS:-4}"
KNEE_TOLERANCE="${KNEE_TOLERANCE:--10}"
MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-1}"

GSM8K_TASKS="${GSM8K_TASKS:-gsm8k_cot}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-8}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"

RUN_NAME="${RUN_NAME:-qwen3_8b_awq_clc_gsm8k_b${BITS}}"

# Keep the quantized model dirs on disk after eval? (0 = delete, 1 = keep)
KEEP_QUANTIZED="${KEEP_QUANTIZED:-0}"

ARGS=(
  --model-path "$MODEL_PATH"
  --models-root "$MODELS_ROOT"
  --origin-method "awq"
  --results-models-dir "$RESULTS_MODELS_DIR"
  --results-eval-dir "$RESULTS_EVAL_DIR"
  --calibration-cache-dir "$CALIBRATION_CACHE_DIR"
  --eval-cache-dir "$EVAL_CACHE_DIR"
  --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
  --calib-dataset "$CALIB_DATASET"
  --n-calib "$N_CALIB"
  --calib-seqlen "$CALIB_SEQLEN"
  --seed "$SEED"
  --stride "$STRIDE"
  --max-length "$MAX_LENGTH"
  --c4-samples "$C4_SAMPLES"
  --bits "$BITS"
  --knee-tolerance "$KNEE_TOLERANCE"
  --max-flip-percent "$MAX_FLIP_PERCENT"
  --gsm8k-tasks "$GSM8K_TASKS"
  --gsm8k-num-fewshot "$GSM8K_NUM_FEWSHOT"
  --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
  --run-name "$RUN_NAME"
  --no-wandb
)

if [ "$KEEP_QUANTIZED" = "1" ]; then
  ARGS+=(--keep-quantized)
fi

echo "==> gsm8k_compare :: ${MODEL_PATH} :: AWQ ${BITS}-bit :: FP16 vs AWQ vs AWQ+CLC"
"$PYTHON_BIN" main.py gsm8k_compare "${ARGS[@]}"

echo "==> results (PPL + GSM8K) written under ${RESULTS_EVAL_DIR}/${RUN_NAME}.json"
echo "==> per-model lm-eval payloads under ${LM_EVAL_OUTPUT_DIR}/"
