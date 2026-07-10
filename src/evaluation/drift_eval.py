"""Inter-layer mean-shift accumulation evaluation (E1 + E2).

This module measures how the layer-output shift induced by weight quantization
behaves *across depth*, to test whether CLC only reduces the isolated per-layer
first-moment shift (the quantity it optimizes) or also suppresses the shift that
*accumulates* through the network on the real quantized inference stream.

Two quantities are measured, per decoder block l, against the FP16 reference:

  E1 (accumulated / real-stream): each model is run end-to-end on the same input.
      The block-l output therefore already carries the quantization error of all
      earlier blocks. We report, at block l:
        - mean_shift(l)  = mean_over(tokens,channels)( Y_q^(l) - Y_fp^(l) )
        - rel_l2(l)      = ||Y_q^(l) - Y_fp^(l)||_2 / ||Y_fp^(l)||_2
      This is the quantity the reviewer asks about: does the mean shift compound
      with depth, and does CLC flatten that curve?

  E2 (isolated / teacher-forced): the FP16 hidden state entering block l is fed
      into the *quantized* block l, so the input carries no upstream quantization
      error. The resulting shift is the per-layer effect in isolation -- exactly
      the object CLC's theory operates on. Comparing E1 vs E2 exposes the part of
      the drift that is due to accumulation (E1 - E2) and shows that CLC still
      helps on the accumulated stream even though it is only trained on E2.

The evaluation reuses the AWQ quantization pipeline: it builds three models in
turn (FP16 reference, base quantizer, base+CLC), captures per-block outputs with
forward hooks, and writes a JSON with the per-depth curves plus a compact
summary. No model weights are needed on disk; everything is computed in memory.
"""

from __future__ import annotations

import copy
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch


# --------------------------------------------------------------------------- #
# Helpers to locate the decoder blocks in a HF causal-LM (LLaMA/Mistral/Qwen). #
# --------------------------------------------------------------------------- #
def get_decoder_blocks(model) -> torch.nn.ModuleList:
    """Return the list of decoder blocks for standard HF causal LMs."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(
            "Could not locate decoder blocks at model.model.layers; "
            "drift_eval currently supports LLaMA/Mistral/Qwen-style models."
        )
    return layers


def _block_output_tensor(output) -> torch.Tensor:
    """Decoder blocks return either a Tensor or a tuple whose first entry is it."""
    if isinstance(output, tuple):
        return output[0]
    return output


# --------------------------------------------------------------------------- #
# Capture buffers                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class _BlockAccumulator:
    """Streaming accumulator for one block, to avoid storing full activations.

    We accumulate the running sum of (Y_q - Y_fp), the sum of squared error, and
    the sum of squared FP16 magnitude, over all tokens seen. Everything is kept
    in float64 on CPU for numerical stability across many calibration samples.
    """

    sum_signed: float = 0.0          # sum over tokens,channels of (q - fp)
    count_elems: int = 0             # number of scalar entries summed
    sum_sq_err: float = 0.0          # sum of (q - fp)^2
    sum_sq_ref: float = 0.0          # sum of fp^2

    def update(self, q: torch.Tensor, fp: torch.Tensor):
        diff = (q.to(torch.float64) - fp.to(torch.float64)).reshape(-1)
        ref = fp.to(torch.float64).reshape(-1)
        self.sum_signed += diff.sum().item()
        self.count_elems += diff.numel()
        self.sum_sq_err += (diff * diff).sum().item()
        self.sum_sq_ref += (ref * ref).sum().item()

    def mean_shift(self) -> float:
        return self.sum_signed / max(self.count_elems, 1)

    def rel_l2(self) -> float:
        denom = self.sum_sq_ref ** 0.5
        if denom < 1e-12:
            return 0.0
        return (self.sum_sq_err ** 0.5) / denom


# --------------------------------------------------------------------------- #
# Reference (FP16) capture: store per-block hidden states for every sample.    #
# --------------------------------------------------------------------------- #
class ReferenceCapture:
    """Runs the FP16 model and stores, per sample, the input and output hidden
    state of every decoder block. These are reused as:
      * the ground truth for E1/E2, and
      * the teacher-forced inputs for E2.
    """

    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.blocks = get_decoder_blocks(model)
        self.num_blocks = len(self.blocks)
        # Per sample: list over blocks of input/output hidden states (CPU).
        self.block_inputs: List[List[torch.Tensor]] = []
        self.block_outputs: List[List[torch.Tensor]] = []
        self._cur_in: Dict[int, torch.Tensor] = {}
        self._cur_out: Dict[int, torch.Tensor] = {}

    def _make_hook(self, idx: int) -> Callable:
        def hook(_module, inputs, output):
            inp = inputs[0] if isinstance(inputs, tuple) else inputs
            self._cur_in[idx] = inp.detach().to("cpu")
            self._cur_out[idx] = _block_output_tensor(output).detach().to("cpu")
        return hook

    @torch.no_grad()
    def run(self, input_batches: List[dict]):
        handles = [blk.register_forward_hook(self._make_hook(i)) for i, blk in enumerate(self.blocks)]
        try:
            for inputs in input_batches:
                self._cur_in, self._cur_out = {}, {}
                self.model(**{k: v.to(self.device) for k, v in inputs.items()}, use_cache=False)
                self.block_inputs.append([self._cur_in[i] for i in range(self.num_blocks)])
                self.block_outputs.append([self._cur_out[i] for i in range(self.num_blocks)])
        finally:
            for h in handles:
                h.remove()
        return self


# --------------------------------------------------------------------------- #
# E1: accumulated / real-stream shift for a quantized model.                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_accumulated(model, device: str, input_batches: List[dict],
                        reference: ReferenceCapture) -> List[_BlockAccumulator]:
    blocks = get_decoder_blocks(model)
    accs = [_BlockAccumulator() for _ in range(len(blocks))]
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            captured[idx] = _block_output_tensor(output).detach().to("cpu")
        return hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]
    try:
        for s, inputs in enumerate(input_batches):
            captured.clear()
            model(**{k: v.to(device) for k, v in inputs.items()}, use_cache=False)
            for i in range(len(blocks)):
                accs[i].update(captured[i], reference.block_outputs[s][i])
    finally:
        for h in handles:
            h.remove()
    return accs


# --------------------------------------------------------------------------- #
# E2: isolated / teacher-forced shift for a quantized model.                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_isolated(model, device: str, reference: ReferenceCapture,
                     attention_kwargs_per_sample: List[dict]) -> List[_BlockAccumulator]:
    """Feed the FP16 input hidden state of each block into the *quantized* block
    and compare its output to the FP16 output of that same block. The block only
    sees clean upstream activations, so this isolates the per-layer shift.
    """
    blocks = get_decoder_blocks(model)
    accs = [_BlockAccumulator() for _ in range(len(blocks))]

    for s in range(len(reference.block_inputs)):
        extra = attention_kwargs_per_sample[s] if s < len(attention_kwargs_per_sample) else {}
        for i, blk in enumerate(blocks):
            fp_in = reference.block_inputs[s][i].to(device)
            out = blk(fp_in, **extra)
            q_out = _block_output_tensor(out).detach().to("cpu")
            accs[i].update(q_out, reference.block_outputs[s][i])
            del fp_in, out, q_out
        torch.cuda.empty_cache()
    return accs


# --------------------------------------------------------------------------- #
# Building the per-sample attention kwargs needed to call a block directly.    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_block_call_kwargs(model, device: str, input_batches: List[dict]) -> List[dict]:
    """Capture the auxiliary kwargs (position_ids / position_embeddings /
    attention_mask) that each decoder block receives, so E2 can call the block
    standalone with the same geometry the model used. Captured on the FP16 model.
    """
    blocks = get_decoder_blocks(model)
    per_sample: List[dict] = []
    captured: Dict[str, object] = {}

    def hook(_module, args, kwargs):
        # Only the first block is enough: position info is shared across blocks.
        keep = {}
        for key in ("attention_mask", "position_ids", "position_embeddings"):
            if key in kwargs and kwargs[key] is not None:
                val = kwargs[key]
                if isinstance(val, tuple):
                    keep[key] = tuple(v.detach().to(device) for v in val)
                else:
                    keep[key] = val.detach().to(device)
        captured.clear()
        captured.update(keep)

    handle = blocks[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for inputs in input_batches:
            captured.clear()
            model(**{k: v.to(device) for k, v in inputs.items()}, use_cache=False)
            per_sample.append(dict(captured))
    finally:
        handle.remove()
    return per_sample


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def _summarize(accs: List[_BlockAccumulator]) -> dict:
    mean_shift = [a.mean_shift() for a in accs]
    rel_l2 = [a.rel_l2() for a in accs]
    abs_mean = [abs(m) for m in mean_shift]
    return {
        "per_layer_mean_shift": mean_shift,
        "per_layer_abs_mean_shift": abs_mean,
        "per_layer_rel_l2": rel_l2,
        "final_layer_abs_mean_shift": abs_mean[-1] if abs_mean else 0.0,
        "final_layer_rel_l2": rel_l2[-1] if rel_l2 else 0.0,
        "mean_abs_mean_shift": (sum(abs_mean) / len(abs_mean)) if abs_mean else 0.0,
        "mean_rel_l2": (sum(rel_l2) / len(rel_l2)) if rel_l2 else 0.0,
    }


@torch.no_grad()
def run_drift_eval(
    build_model: Callable[[], "torch.nn.Module"],
    quantize_inplace: Callable[["torch.nn.Module", str], None],
    tokenizer,
    eval_texts: List[str],
    device: str,
    max_length: int = 512,
    output_path: Optional[Path] = None,
) -> dict:
    """Full E1 + E2 sweep for FP16 / base / CLC.

    Args:
      build_model: returns a fresh FP16 model on `device`.
      quantize_inplace: given (model, mode) with mode in {"base", "clc"},
                        quantizes the model in place (mutates its weights).
      eval_texts: list of raw strings used as the fixed evaluation stream.
    """
    # Tokenize a fixed evaluation stream (shared across all three models).
    input_batches: List[dict] = []
    for text in eval_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_batches.append({k: v for k, v in enc.items()})

    # 1) FP16 reference: capture per-block inputs/outputs and block-call kwargs.
    print("[drift_eval] building FP16 reference model ...")
    fp_model = build_model()
    fp_model.eval()
    reference = ReferenceCapture(fp_model, device).run(input_batches)
    block_kwargs = build_block_call_kwargs(fp_model, device, input_batches)
    num_blocks = reference.num_blocks
    print(f"[drift_eval] captured reference over {len(input_batches)} samples, {num_blocks} blocks")

    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    results: Dict[str, dict] = {"num_blocks": num_blocks, "num_samples": len(input_batches)}

    for mode in ("base", "clc"):
        print(f"[drift_eval] quantizing model :: mode={mode}")
        q_model = build_model()
        q_model.eval()
        quantize_inplace(q_model, mode)
        q_model.eval()

        print(f"[drift_eval]   measuring E1 (accumulated) :: mode={mode}")
        acc_e1 = measure_accumulated(q_model, device, input_batches, reference)
        print(f"[drift_eval]   measuring E2 (isolated)    :: mode={mode}")
        acc_e2 = measure_isolated(q_model, device, reference, block_kwargs)

        results[mode] = {
            "E1_accumulated": _summarize(acc_e1),
            "E2_isolated": _summarize(acc_e2),
        }

        del q_model
        gc.collect()
        torch.cuda.empty_cache()

    # Convenience deltas: how much accumulation exceeds the isolated shift, and
    # how much CLC reduces each, at the final layer.
    def _final_abs(mode, key):
        return results[mode][key]["final_layer_abs_mean_shift"]

    results["summary"] = {
        "base_E1_final_abs_mean_shift": _final_abs("base", "E1_accumulated"),
        "clc_E1_final_abs_mean_shift": _final_abs("clc", "E1_accumulated"),
        "base_E2_final_abs_mean_shift": _final_abs("base", "E2_isolated"),
        "clc_E2_final_abs_mean_shift": _final_abs("clc", "E2_isolated"),
        "E1_accumulation_gap_base": (
            _final_abs("base", "E1_accumulated") - _final_abs("base", "E2_isolated")
        ),
        "clc_reduces_E1_final_by": (
            _final_abs("base", "E1_accumulated") - _final_abs("clc", "E1_accumulated")
        ),
        "clc_reduces_E2_final_by": (
            _final_abs("base", "E2_isolated") - _final_abs("clc", "E2_isolated")
        ),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"[drift_eval] wrote curves to {output_path}")

    return results
