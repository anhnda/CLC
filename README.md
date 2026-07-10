# CLC

CLC is a research framework for **weight-only post-training quantization (PTQ)** of large language models. It pairs standard quantizers (AWQ, FlatQuant) with a lightweight correction stage, **CLC**, that refines the quantized weights so that the layer output stays close to the original float output.

Everything runs through a single entrypoint, `main.py`. The scripts under `scripts/bash/` are thin wrappers around it for running the common recipes on a given model.

## What CLC does

A uniform quantizer rounds each weight to the nearest grid point. This minimizes the per-weight rounding error, but it ignores how that error propagates to the layer output once it is multiplied by the activations. As a result, the rounding noise can accumulate into a structured bias at the output.

CLC is a post-correction step applied right after quantization. For each output channel it looks at the rounding decisions and selectively **flips** a small number of them (rounding up instead of down, or vice versa) so that the expected output error — measured against the activation statistics — is reduced. It is fast, calibration-light, and applied on top of an existing quantized model rather than retraining it.

Three ideas keep the correction stable:

- **Expectation under activations.** Weight adjustments are scored by their estimated effect on the output, using per-channel activation means rather than raw weight error.
- **James–Stein shrinkage** (`--use-james-stein`, on by default) shrinks the per-channel mean estimates toward their grand mean, reducing variance when the calibration set is small.
- **Knee-based outlier masking** (`--knee-tolerance`) detects the transition point in the sorted activation magnitudes and leaves the most extreme channels untouched, so a few large activations do not dominate the correction.
- **A per-output flip budget** (`--max-flip-percent`) caps how many weights may be flipped per output channel, keeping the change small and well-conditioned.

CLC works as a post-correction stage on either an AWQ or a FlatQuant base, and sits alongside a simpler `bias_correction` stage.

## Repository layout

- `main.py` — the CLI for quantization and evaluation
- `src/quantization/` — quantization pipeline, AWQ and FlatQuant adapters, bias correction
- `src/post_correction/` — the `clc` correction and other post-correction stages
- `rtn_utils.py` — round-to-nearest helper used by the FlatQuant flow
- `src/evaluation/` — perplexity evaluation and `lm-evaluation-harness` integration
- `flatquant/` — the FlatQuant library, vendored and reused
- `scripts/bash/` — `.sh` wrappers grouped by correction stage and model family
- `datasets/` — local datasets that some FlatQuant loaders read
- `data/cache/` — runtime calibration/evaluation cache
- `results/models/` — quantized model artifacts
- `results/eval/` — evaluation results (JSON)

## Installation

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

`torch` is intentionally left out of `requirements.txt` because the right build depends on your CUDA version. Install it separately afterwards, matching your setup — for example:

```bash
# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Pick the index URL for your CUDA version (see https://pytorch.org for the current matrix).

To use a private Hugging Face model or log to Weights & Biases, create a `.env` file at the repo root:

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

`main.py` loads `.env` automatically when it exists.

## Concepts

There are two main flows:

1. `float_model` — evaluate the original float model.
2. `quantize` — quantize, then immediately evaluate the result.

`quantize` is configured by two choices:

- `--origin-method awq|flatquant` — the base quantizer
- `--post-correction none|clc|bias_correction` — the correction applied on top

The other modes are shortcuts:

- `raw_quantize` = `--post-correction none`
- `flip_quantize` = `--post-correction clc`
- `compare_all` = evaluate `float`, `raw`, and corrected models together

### Model path resolution

`--model-path` is resolved in this order:

1. used directly if it is an existing local path
2. otherwise tried as `<models_root>/<model_path>` (`--models-root` defaults to `/models`)
3. otherwise treated as a Hugging Face model id

```bash
--model-path /models/Mistral-7B-v0.3
--model-path Mistral-7B-v0.3 --models-root /models
--model-path mistralai/Mistral-7B-v0.3
```

## Usage

Inspect the CLI any time:

```bash
python main.py -h
python main.py quantize -h
```

### Evaluate a float model

```bash
python main.py float_model \
  --model-path mistralai/Mistral-7B-v0.3
```

### AWQ (raw, no correction)

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method awq \
  --post-correction none \
  --bits 4 \
  --run-name awq_raw_mistral
```

### AWQ + CLC

A standard run on Mistral with `max_flip_percent = 0.05` and `knee_tolerance = 0`:

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method awq \
  --post-correction clc \
  --bits 4 \
  --knee-tolerance 0 \
  --max-flip-percent 0.05 \
  --run-name awq_clc_mistral
```

`--knee-tolerance` and `--max-flip-percent` are the two main knobs: the first controls how aggressively outlier channels are masked, the second caps the per-output flip budget.

### AWQ + bias correction

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method awq \
  --post-correction bias_correction \
  --bits 4 \
  --bias-correction-samples 4096 \
  --run-name awq_bias_correction_mistral
```

### FlatQuant (raw)

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method flatquant \
  --post-correction none \
  --bits 4 \
  --flatquant-epochs 15 \
  --flatquant-cali-bsz 4 \
  --flatquant-lr 5e-3 \
  --run-name flatquant_raw_mistral
```

### FlatQuant + CLC

FlatQuant correction recipes reuse a previously produced raw artifact via `--flatquant-raw-path`:

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method flatquant \
  --post-correction clc \
  --bits 4 \
  --knee-tolerance 0 \
  --max-flip-percent 0.05 \
  --flatquant-raw-path ./results/models/flatquant_raw/flatquant_raw_mistral \
  --run-name flatquant_clc_mistral
```

### FlatQuant + bias correction

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method flatquant \
  --post-correction bias_correction \
  --bits 4 \
  --bias-correction-samples 4096 \
  --flatquant-raw-path ./results/models/flatquant_raw/flatquant_raw_mistral \
  --run-name flatquant_bias_correction_mistral
```

### Compare several models

```bash
python main.py compare_all \
  --model-path mistralai/Mistral-7B-v0.3 \
  --raw-path ./results/models/awq_raw/awq_raw_mistral \
  --flip-path ./results/models/awq_clc/awq_clc_mistral
```

## Running with the bash wrappers

The wrappers are grouped by correction stage and base quantizer:

- `scripts/bash/clc/awq/`
- `scripts/bash/clc/flatquant/`
- `scripts/bash/bias_correction/awq/`
- `scripts/bash/bias_correction/flatquant/`

Each group has one script per model family: `run_mistral.sh`, `run_llama3.sh`, `run_llama31.sh`, `run_qwen25.sh`.

Basic use:

```bash
bash scripts/bash/clc/awq/run_mistral.sh
```

Override the model:

```bash
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
MODELS_ROOT=/models \
bash scripts/bash/clc/awq/run_llama3.sh
```

The `clc/awq` scripts run the float model, run AWQ raw, then sweep the `knee_tolerance` × `max_flip_percent` grid for CLC. The `clc/flatquant` scripts produce a FlatQuant raw artifact first, then run CLC on top; set `RUN_RAW_QUANTIZE=0` with `RAW_MODEL_DIR=...` to reuse an existing raw artifact:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
RUN_RAW_QUANTIZE=0 \
RAW_MODEL_DIR=./results/models/flatquant_raw/flatquant_raw_Mistral-7B-v0.3 \
bash scripts/bash/clc/flatquant/run_mistral.sh
```

The `bias_correction/*` scripts follow the same pattern with a single bias-correction run instead of a grid sweep.

### Common environment variables

All wrappers honor:

`MODEL_PATH`, `MODELS_ROOT`, `PYTHON_BIN`, `RESULTS_MODELS_DIR`, `RESULTS_EVAL_DIR`, `CALIBRATION_CACHE_DIR`, `EVAL_CACHE_DIR`, `CALIB_DATASET`, `N_CALIB`, `CALIB_SEQLEN`, `SEED`, `STRIDE`, `MAX_LENGTH`, `C4_SAMPLES`, `LM_EVAL_TASK_PRESET`, `INCLUDE_LM_EVAL`, `INCLUDE_C4`, `USE_WANDB`, `WANDB_PROJECT`, `WANDB_ENTITY`.

FlatQuant wrappers additionally honor: `RUN_FLOAT_MODEL`, `RUN_RAW_QUANTIZE`, `RAW_MODEL_DIR`, `FLATQUANT_EPOCHS`, `FLATQUANT_CALI_BSZ`, `FLATQUANT_LR`, `FLATQUANT_DIAG_INIT`, `FLATQUANT_DIAG_ALPHA`, `FLATQUANT_CALI_TRANS`, `FLATQUANT_ADD_DIAG`, `FLATQUANT_LWC`, `FLATQUANT_LAC`.

## Evaluation

Each run writes a JSON report into `results/eval/`. Two evaluation paths are available:

- the default sliding-window path (`src/evaluation/sliding_window.py`) downloads WikiText-2 and C4 from Hugging Face and caches them under `data/cache/eval`
- the FlatQuant path (`src/evaluation/flatquant_runner.py`) uses local dataset scripts under `datasets/`

By default WikiText-2, C4, and `lm_eval` are all enabled, with the `extended` `lm_eval` preset. Useful options:

- `--no-c4`
- `--no-lm-eval`
- `--lm-eval-task-preset core|extended`
- `--lm-eval-tasks arc_easy hellaswag ...`
- `--use-wandb`

## Datasets

Two distinct things:

1. `data/cache/...` — runtime cache the repo creates while running calibration/evaluation.
2. `datasets/...` — local datasets that some FlatQuant loaders read directly.

For the AWQ flows and sliding-window evaluation, calibration (`c4`) and evaluation data (WikiText-2, C4) are downloaded automatically from Hugging Face — no manual setup is needed.

The FlatQuant loaders (`flatquant/data_utils.py`, `src/evaluation/flatquant_data_utils.py`) instead read from local paths:

- `datasets/wikitext`
- `datasets/allenai/c4`
- `datasets/ptb_text_only`
- `datasets/pile-val-backup`

At minimum prepare `datasets/wikitext`, which is the most common FlatQuant setup pitfall. The repo ships `datasets/wikitext/wikitext.py`, which reads a local zip:

```bash
mkdir -p datasets/wikitext
cd datasets/wikitext
wget -O wikitext-2-raw-v1.zip \
  "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip?download=true"
```

Resulting layout:

```text
datasets/
  wikitext/
    wikitext.py
    wikitext-2-raw-v1.zip
```

The other local datasets are only needed if you call their corresponding FlatQuant loaders.

## Testing

The test suite uses `unittest`, run via `pytest`:

```bash
# Everything
python -m pytest tests/ -v

# A single file
python -m pytest tests/test_quantization_pipeline.py -v

# A single test
python -m pytest tests/test_main.py::ParserModeTests::test_parser_accepts_single_model_modes -v
```

Key files: `tests/test_main.py` (CLI and the quantize flow), `tests/test_quantization_pipeline.py` (pipeline factory and configs), `tests/test_bash_scripts.py` (wrapper layout and contents), and `tests/test_flatquant_*.py` / `tests/test_lm_eval_runner.py` (FlatQuant and lm-eval integration).

## Output locations

- model artifacts: `results/models/<variant>/<run_name>/`
- evaluation report: `results/eval/<run_name>.json`
- run metadata: `results/models/<variant>/<run_name>/metadata.json`

For `flatquant` with `--post-correction none`, the raw output is kept so a later correction stage can reuse it. Other temporary quantized outputs may be removed after evaluation when the pipeline no longer needs them.
