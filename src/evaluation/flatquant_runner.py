from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation import flatquant_data_utils, flatquant_eval_utils


FLATQUANT_ROOT = Path("/workspace/FlatQuant")


class _WorkingDirectory:
    def __init__(self, path: Path):
        self.path = path
        self.previous = None

    def __enter__(self):
        self.previous = Path.cwd()
        os.chdir(self.path)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.previous is not None:
            os.chdir(self.previous)


def _load_model_and_tokenizer(model_ref, hf_token: str | None, eval_device: str):
    if isinstance(model_ref, dict) and {"model", "tokenizer"}.issubset(model_ref):
        model = model_ref["model"]
        tokenizer = model_ref["tokenizer"]
        if not hasattr(model, "seqlen"):
            model.seqlen = 2048
        if hasattr(model, "to"):
            model = model.to(eval_device)
        model.eval()
        return model, tokenizer, False

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        use_fast=False,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
        token=hf_token,
    )
    if hasattr(model, "to"):
        model = model.to(eval_device)
    model.seqlen = 2048
    model.eval()
    return model, tokenizer, True


def run_flatquant_perplexity_evaluation(args, model_paths: dict[str, str], hf_token: str | None = None) -> dict:
    results = {}
    dataset_names = {
        "wikitext2": "WikiText-2",
        "c4": "C4",
    }
    eval_device = "cuda" if torch.cuda.is_available() else "cpu"

    with _WorkingDirectory(FLATQUANT_ROOT):
        for eval_dataset, display_name in dataset_names.items():
            results[display_name] = {}
            for model_name, model_ref in model_paths.items():
                model, tokenizer, should_cleanup = _load_model_and_tokenizer(model_ref, hf_token, eval_device)
                try:
                    testloader = flatquant_data_utils.get_loaders(
                        args,
                        eval_dataset,
                        tokenizer,
                        seqlen=model.seqlen,
                        eval_mode=True,
                    )
                    perplexity = flatquant_eval_utils.ppl_eval(model, testloader)
                    total_tokens = int(testloader.input_ids.numel())
                    results[display_name][model_name] = {
                        "perplexity": perplexity,
                        "total_tokens": total_tokens,
                    }
                finally:
                    if should_cleanup:
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    return results
