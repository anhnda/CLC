from __future__ import annotations

import gc
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from src.quantization.state import IntegerQuantizedTensorState

if TYPE_CHECKING:
    from src.post_correction.bias_correction import BiasCorrectionCorrection
    from src.post_correction.clc import CLCCorrection


@dataclass
class FlatQuantConfig:
    w_bits: int = 4
    a_bits: int = 4
    q_bits: int = 16
    k_bits: int = 16
    v_bits: int = 16
    epochs: int = 15
    cali_bsz: int = 4
    flat_lr: float = 5e-3
    cali_trans: bool = True
    add_diag: bool = True
    lwc: bool = True
    lac: bool = True
    diag_init: str = "sq_style"
    diag_alpha: float = 0.3
    w_asym: bool = False
    deactive_amp: bool = False
    direct_inv: bool = False
    separate_vtrans: bool = False
    warmup: bool = False
    rtn_mse: bool = False
    w_groupsize: int = -1
    debug_diagnostics: bool = False
    debug_sample_limit: int = 256


class _FlatQuantLogger:
    def info(self, message):
        print(message)


class _ActivationCollector:
    def __init__(self, sample_limit: int, track_mean: bool, storage_dtype: torch.dtype = torch.float16):
        self.sample_limit = max(0, int(sample_limit))
        self.track_mean = track_mean
        self.storage_dtype = storage_dtype
        self._samples = []
        self._stored_rows = 0
        self._feature_dim = None
        self._sum = None
        self._count = 0

    def add(self, batch: torch.Tensor):
        if batch.numel() == 0:
            return

        rows = batch.detach().reshape(-1, batch.shape[-1]).cpu()
        if self._feature_dim is None:
            self._feature_dim = int(rows.shape[-1])

        if self.track_mean:
            rows_fp32 = rows.to(dtype=torch.float32)
            batch_sum = rows_fp32.sum(dim=0)
            self._sum = batch_sum if self._sum is None else self._sum + batch_sum
            self._count += int(rows_fp32.shape[0])

        if self._stored_rows >= self.sample_limit:
            return

        remaining = self.sample_limit - self._stored_rows
        if remaining <= 0:
            return

        kept = rows[:remaining].to(dtype=self.storage_dtype)
        self._samples.append(kept)
        self._stored_rows += int(kept.shape[0])

    def get_sample_rows(self):
        if not self._samples:
            if self._feature_dim is None:
                return None
            return torch.empty((0, self._feature_dim), dtype=torch.float32)
        return torch.cat(self._samples, dim=0).to(dtype=torch.float32)

    def get_mean(self):
        if not self.track_mean or self._sum is None or self._count == 0:
            return None
        return self._sum / self._count

class FlatQuantRTNQuantizer:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        config: Optional[FlatQuantConfig] = None,
        post_correction: Optional["CLCCorrection | BiasCorrectionCorrection"] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or FlatQuantConfig()
        self.post_correction = post_correction
        self.layer_stats: Dict[str, dict] = {}
        self._rtn_quantizers = {}
        self._imports = None
        self._post_flatquant_float_weights = {}
        self._artifact_dir: str | None = None
        self.debug_report = {
            "enabled": bool(self.config.debug_diagnostics),
            "sample_limit": int(self.config.debug_sample_limit),
            "evaluation_target": self.describe_evaluation_target(),
        }

        if not hasattr(self.model, "seqlen"):
            self.model.seqlen = 2048

        print("\n[FlatQuant RTN Quantizer Initialized]")
        print(f"  Config: {asdict(self.config)}")
        print(f"  Post correction: {type(self.post_correction).__name__ if self.post_correction else 'none'}")

    @staticmethod
    def empty_flip_stats() -> dict:
        return {"total": 0}

    def describe_evaluation_target(self) -> dict:
        model_name = getattr(self.model, "name_or_path", None) or getattr(self.model.config, "_name_or_path", None)
        return {
            "kind": "in_memory_model",
            "model_type": getattr(self.model.config, "model_type", None),
            "model_name_or_path": model_name,
            "debug_diagnostics": bool(self.config.debug_diagnostics),
            "debug_sample_limit": int(self.config.debug_sample_limit),
        }

    @staticmethod
    def _build_weight_summary(weight: torch.Tensor) -> dict:
        weight_fp32 = weight.detach().to(dtype=torch.float32)
        return {
            "mean": float(weight_fp32.mean().item()),
            "std": float(weight_fp32.std(unbiased=False).item()),
            "min": float(weight_fp32.min().item()),
            "max": float(weight_fp32.max().item()),
            "l2_norm": float(weight_fp32.norm().item()),
        }

    def _build_stage_snapshot(self, weight: torch.Tensor, output_mse: float, **extra) -> dict:
        snapshot = self._build_weight_summary(weight)
        snapshot["output_mse"] = float(output_mse)
        snapshot.update(extra)
        return snapshot

    @staticmethod
    def _flatquant_root() -> Path:
        return Path(__file__).resolve().parents[2] / "flatquant"

    def _ensure_flatquant_importable(self):
        flatquant_root = str(self._flatquant_root())
        if flatquant_root not in sys.path:
            sys.path.insert(0, flatquant_root)

    def _load_flatquant_imports(self):
        if self._imports is not None:
            return self._imports

        self._ensure_flatquant_importable()
        from flatquant.flat_utils import load_flat_parameters, reparameterize_model
        from flatquant.train_utils import cali_flat_quant

        self._imports = {
            "load_flat_parameters": load_flat_parameters,
            "reparameterize_model": reparameterize_model,
            "cali_flat_quant": cali_flat_quant,
        }
        return self._imports

    def set_artifact_dir(self, artifact_dir: str | Path | None):
        self._artifact_dir = None if artifact_dir is None else str(artifact_dir)

    def _build_flatquant_args(self, nsamples: int) -> SimpleNamespace:
        model_name = getattr(self.model, "name_or_path", None) or getattr(self.model.config, "_name_or_path", "model")
        artifact_dir = self._artifact_dir or "./results/models/clc-flatquant"
        return SimpleNamespace(
            model=model_name,
            seed=0,
            hf_token=None,
            a_bits=self.config.a_bits,
            a_groupsize=-1,
            a_asym=False,
            w_bits=self.config.w_bits,
            w_groupsize=self.config.w_groupsize,
            w_asym=self.config.w_asym,
            rtn_mse=self.config.rtn_mse,
            epochs=self.config.epochs,
            nsamples=nsamples,
            cali_bsz=self.config.cali_bsz,
            flat_lr=self.config.flat_lr,
            cali_trans=self.config.cali_trans,
            add_diag=self.config.add_diag,
            lwc=self.config.lwc,
            lac=self.config.lac,
            resume=False,
            save_matrix=False,
            reload_matrix=False,
            matrix_path=None,
            diag_init=self.config.diag_init,
            diag_alpha=self.config.diag_alpha,
            warmup=self.config.warmup,
            deactive_amp=self.config.deactive_amp,
            direct_inv=self.config.direct_inv,
            separate_vtrans=self.config.separate_vtrans,
            q_bits=self.config.q_bits,
            q_asym=False,
            q_groupsize=-1,
            k_bits=self.config.k_bits,
            k_asym=False,
            k_groupsize=-1,
            v_bits=self.config.v_bits,
            v_asym=False,
            v_groupsize=-1,
            output_dir=artifact_dir,
            exp_name=Path(artifact_dir).name,
            lm_eval=False,
            tasks=[],
            lm_eval_batch_size=1,
            distribute_model=False,
            quantized_save=False,
            quantize=True,
            model_name=model_name.split("/")[-1],
            exp_dir=artifact_dir,
        )

    def _select_apply_fn(self):
        self._load_flatquant_imports()
        model_name = (getattr(self.model, "name_or_path", None) or getattr(self.model.config, "_name_or_path", "")).lower()
        model_type = getattr(self.model.config, "model_type", "")

        if model_type == "qwen2":
            from flatquant.model_tools.qwen_utils import apply_flatquant_to_qwen
            return apply_flatquant_to_qwen
        if model_type == "mistral":
            from src.quantization.flatquant_mistral import apply_flatquant_to_mistral
            return apply_flatquant_to_mistral
        if model_type == "llama" and "3.1" in model_name:
            from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31
            return apply_flatquant_to_llama_31
        if model_type == "llama":
            from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
            return apply_flatquant_to_llama
        raise ValueError(f"FlatQuant backend does not support model type: {model_type}")

    @staticmethod
    def _infer_wrapped_weight_device(module, projection_names):
        for projection_name in projection_names:
            projection = getattr(module, projection_name, None)
            linear = getattr(projection, "linear", None)
            weight = getattr(linear, "weight", None)
            device = getattr(weight, "device", None)
            if device is not None:
                return device
        return None

    @classmethod
    def _align_reused_flatquant_module_devices(cls, model):
        layers = getattr(getattr(model, "model", None), "layers", [])
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            attn_device = None if attn is None else cls._infer_wrapped_weight_device(attn, ("q_proj", "k_proj", "v_proj", "o_proj"))
            if attn is not None and attn_device is not None:
                layer.self_attn = attn.to(attn_device)
                input_layernorm = getattr(layer, "input_layernorm", None)
                if input_layernorm is not None and hasattr(input_layernorm, "to"):
                    layer.input_layernorm = input_layernorm.to(attn_device)

            mlp = getattr(layer, "mlp", None)
            mlp_device = None if mlp is None else cls._infer_wrapped_weight_device(mlp, ("up_proj", "gate_proj", "down_proj"))
            if mlp is not None and mlp_device is not None:
                layer.mlp = mlp.to(mlp_device)
                post_attention_layernorm = getattr(layer, "post_attention_layernorm", None)
                if post_attention_layernorm is not None and hasattr(post_attention_layernorm, "to"):
                    layer.post_attention_layernorm = post_attention_layernorm.to(mlp_device)

    def _build_calibration_loader(self, calibration_data: Iterable[str | torch.Tensor]):
        loader = []
        seqlen = int(getattr(self.model, "seqlen", 2048))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        for sample in calibration_data:
            if isinstance(sample, torch.Tensor):
                input_ids = sample.detach().clone().to(dtype=torch.long, device="cpu")
                if input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)
            else:
                encoded = self.tokenizer(
                    sample,
                    return_tensors="pt",
                    truncation=True,
                    max_length=seqlen,
                    add_special_tokens=False,
                )
                input_ids = encoded["input_ids"]
            if input_ids.shape[1] < seqlen:
                pad = torch.full((1, seqlen - input_ids.shape[1]), pad_id, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, pad], dim=1)
            elif input_ids.shape[1] > seqlen:
                input_ids = input_ids[:, :seqlen]
            target = input_ids.clone()
            loader.append((input_ids, target))
        return loader

    def _get_layer_inputs(self, calibration_loader):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        layers = self.model.model.layers
        model_device = self.device
        model_dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((len(calibration_loader), self.model.seqlen, self.model.config.hidden_size), dtype=model_dtype, device=model_device)
        cache = {"i": 0, "attention_mask": None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs.get("attention_mask")
                cache["position_ids"] = kwargs.get("position_ids")
                raise ValueError

        self.model.model.embed_tokens = self.model.model.embed_tokens.to(model_device)
        if hasattr(self.model.model, "norm"):
            self.model.model.norm = self.model.model.norm.to(model_device)
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(model_device)
        layers[0] = layers[0].to(model_device)
        layers[0] = Catcher(layers[0])

        with torch.no_grad():
            for batch in calibration_loader:
                try:
                    self.model(batch[0].to(model_device), use_cache=False)
                except ValueError:
                    pass

        layers[0] = layers[0].module.cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        if hasattr(self.model.model, "norm"):
            self.model.model.norm = self.model.model.norm.cpu()
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.config.use_cache = use_cache
        return inps, cache["attention_mask"], cache["position_ids"]

    @staticmethod
    def _find_linear_modules(module, name=""):
        if type(module) is nn.Linear:
            return {name: module}
        result = {}
        for child_name, child in module.named_children():
            child_path = f"{name}.{child_name}" if name else child_name
            result.update(FlatQuantRTNQuantizer._find_linear_modules(child, child_path))
        return result

    @staticmethod
    def _extract_hidden_states(output):
        if isinstance(output, tuple):
            return output[0]
        return output

    @staticmethod
    def _flatten_activations(activation_batches) -> Optional[torch.Tensor]:
        if isinstance(activation_batches, _ActivationCollector):
            samples = activation_batches.get_sample_rows()
            if samples is None or samples.numel() == 0:
                return None
            return samples

        batches = [batch.reshape(-1, batch.shape[-1]) for batch in activation_batches if batch.numel() > 0]
        if not batches:
            return None
        return torch.cat(batches, dim=0)

    def _collect_subset_activations(self, layer, subset, inps, attention_mask, position_ids):
        sample_limit = 1024
        if self.post_correction is not None and hasattr(self.post_correction, "config"):
            sample_limit = max(sample_limit, int(getattr(self.post_correction.config, "max_samples", sample_limit)))
        track_mean = self.post_correction is not None and type(self.post_correction).__name__ != "BiasCorrectionCorrection"
        activation_data = {
            name: _ActivationCollector(sample_limit=sample_limit, track_mean=track_mean)
            for name in subset
        }

        def make_hook(name):
            def hook(_module, inputs, _output):
                inp = inputs[0] if isinstance(inputs, tuple) else inputs
                activation_data[name].add(inp)
            return hook

        handles = [module.register_forward_hook(make_hook(name)) for name, module in subset.items()]
        outs = torch.zeros_like(inps)
        with torch.no_grad():
            for idx in range(inps.shape[0]):
                layer_output = layer(
                    inps[idx].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                outs[idx] = self._extract_hidden_states(layer_output)
        for handle in handles:
            handle.remove()
        return activation_data, outs

    def _compute_output_mse(
        self,
        float_weight: torch.Tensor,
        quant_weight: torch.Tensor,
        activation_batches: Iterable[torch.Tensor],
        bias_delta: Optional[torch.Tensor] = None,
    ) -> float:
        activation_rows = self._flatten_activations(activation_batches)
        if activation_rows is None:
            return 0.0

        sample_limit = 1024
        if self.config.debug_diagnostics:
            sample_limit = max(1, int(self.config.debug_sample_limit))
        max_samples = min(sample_limit, activation_rows.shape[0])
        if activation_rows.shape[0] > max_samples:
            activation_rows = activation_rows[:max_samples]
        x_samples = activation_rows.to(device=float_weight.device, dtype=float_weight.dtype)
        y_orig = torch.matmul(x_samples, float_weight.t())
        y_quant = torch.matmul(x_samples, quant_weight.t())
        if bias_delta is not None:
            y_quant = y_quant + bias_delta.to(device=y_quant.device, dtype=y_quant.dtype)
        return float((y_orig - y_quant).pow(2).mean().item())

    def _build_quant_state_from_rtn(self, float_weight: torch.Tensor, quantizer=None) -> IntegerQuantizedTensorState:
        if quantizer is None:
            self._ensure_flatquant_importable()
            from flatquant.quant_utils import WeightQuantizer

            quantizer = WeightQuantizer()
            quantizer.configure(self.config.w_bits, perchannel=True, sym=not self.config.w_asym, mse=self.config.rtn_mse)
            quantizer.find_params(float_weight)
        scale = quantizer.scale.to(device=float_weight.device, dtype=float_weight.dtype)
        zero = quantizer.zero.to(device=float_weight.device, dtype=float_weight.dtype)
        max_int = int(quantizer.maxq.item()) if hasattr(quantizer.maxq, "item") else int(quantizer.maxq)

        if bool(getattr(quantizer, "sym", False)):
            min_int = -(max_int + 1)
            pre_round = float_weight / scale
            integer_weights = torch.round(pre_round).clamp(min_int, max_int)
            zero = torch.zeros_like(scale)
        else:
            min_int = 0
            pre_round = float_weight / scale + zero
            integer_weights = torch.round(pre_round).clamp(min_int, max_int)

        return IntegerQuantizedTensorState(
            float_weights=float_weight,
            pre_round=pre_round,
            integer_weights=integer_weights,
            scale=scale,
            zero_point=zero,
            max_int=max_int,
            min_int=min_int,
            in_features=float_weight.shape[1],
            padded_in_features=float_weight.shape[1],
            original_dtype=float_weight.dtype,
        )

    def _build_post_mean(self, activation_batches: Iterable[torch.Tensor], dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        if self.post_correction is None:
            return None

        if isinstance(activation_batches, _ActivationCollector):
            raw_mean = activation_batches.get_mean()
            if raw_mean is None:
                return None
        else:
            activation_rows = self._flatten_activations(activation_batches)
            if activation_rows is None:
                return None
            raw_mean = activation_rows.mean(dim=0)

        prepared = self.post_correction.prepare_activation_means(raw_mean)
        return prepared.to(device=device, dtype=dtype)

    @property
    def bias_correction(self):
        return self.post_correction

    def _quantize_module(self, name: str, module: nn.Linear, activation_batches: Iterable[torch.Tensor], float_weight_override: torch.Tensor | None = None, quantizer_override=None):
        float_weight = float_weight_override if float_weight_override is not None else module.weight.data.clone()
        quant_state = self._build_quant_state_from_rtn(float_weight, quantizer_override)
        quant_weight = quant_state.dequantize_truncated()
        quant_error = self._compute_output_mse(float_weight, quant_weight, activation_batches)

        stage_diagnostics = None
        if self.config.debug_diagnostics:
            stage_diagnostics = {
                "post_flatquant": self._build_stage_snapshot(float_weight, 0.0),
                "post_rtn": self._build_stage_snapshot(quant_weight, quant_error),
            }

        if self.post_correction is None:
            module.weight.data = quant_weight.to(module.weight.data.dtype)
            stats = {
                "error": quant_error,
                "outlier_percent": 0.0,
                "flip_stats": self.empty_flip_stats(),
            }
            if stage_diagnostics is not None:
                stats["stage_diagnostics"] = stage_diagnostics
            self.layer_stats[name] = stats
            return

        correction_name = type(self.post_correction).__name__
        if correction_name == "BiasCorrectionCorrection":
            bias_delta = self.bias_correction.compute_bias_delta(
                module,
                quant_weight,
                activation_batches,
                device=self.device,
            )
            module.weight.data = quant_weight.to(module.weight.data.dtype)
            self.bias_correction.apply_bias_delta(module, bias_delta, self.device, module.weight.data.dtype)
            corrected_error = self._compute_output_mse(float_weight, quant_weight, activation_batches, bias_delta=bias_delta)
            stats = {
                "error": corrected_error,
                "pre_correction_error": quant_error,
                "post_correction_error": corrected_error,
                "outlier_percent": 0.0,
                "flip_stats": {"total": 0, "bias_corrected": True},
                "bias_delta_norm": float(bias_delta.norm().item()),
            }
            if stage_diagnostics is not None:
                stage_diagnostics["post_correction"] = self._build_stage_snapshot(
                    quant_weight,
                    corrected_error,
                    bias_delta_norm=float(bias_delta.norm().item()),
                )
                stats["stage_diagnostics"] = stage_diagnostics
            self.layer_stats[name] = stats
            return

        post_mean = self._build_post_mean(activation_batches, float_weight.dtype, float_weight.device)
        if post_mean is None:
            post_mean = torch.zeros(float_weight.shape[1], device=float_weight.device, dtype=float_weight.dtype)
        corrected_weight, outlier_percent, flip_stats = self.post_correction.apply(quant_state, post_mean)
        corrected_error = self._compute_output_mse(float_weight, corrected_weight, activation_batches)
        module.weight.data = corrected_weight.to(module.weight.data.dtype)
        stats = {
            "error": corrected_error,
            "pre_correction_error": quant_error,
            "post_correction_error": corrected_error,
            "outlier_percent": outlier_percent,
            "flip_stats": flip_stats,
        }
        if stage_diagnostics is not None:
            stage_diagnostics["post_correction"] = self._build_stage_snapshot(
                corrected_weight,
                corrected_error,
                outlier_percent=float(outlier_percent),
                flip_total=int(flip_stats.get("total", 0)) if isinstance(flip_stats, dict) else 0,
            )
            stats["stage_diagnostics"] = stage_diagnostics
        self.layer_stats[name] = stats

    def _run_flatquant_raw(self, n_samples: int, reuse_flat_parameters_path: str | None = None):
        self._ensure_flatquant_importable()
        import flatquant.data_utils as fq_data_utils
        import flatquant.flat_utils as fq_flat_utils
        import flatquant.train_utils as fq_train_utils
        import flatquant.utils as fq_utils
        import rtn_utils as fq_rtn_utils

        flatquant_args = self._build_flatquant_args(n_samples)
        full_args = getattr(self, "_full_args", None)
        if full_args is not None:
            flatquant_args.cali_dataset = getattr(full_args, "calib_dataset", getattr(flatquant_args, "cali_dataset", "wikitext2"))
            flatquant_args.nsamples = getattr(full_args, "n_calib", getattr(flatquant_args, "nsamples", n_samples))
            flatquant_args.seed = getattr(full_args, "seed", flatquant_args.seed)

        trainloader = fq_data_utils.get_loaders(
            flatquant_args,
            flatquant_args.cali_dataset,
            self.tokenizer,
            nsamples=flatquant_args.nsamples,
            seqlen=self.model.seqlen,
            eval_mode=False,
        )

        apply_fn = self._select_apply_fn()
        self.model = apply_fn(flatquant_args, self.model)
        Path(flatquant_args.exp_dir).mkdir(parents=True, exist_ok=True)

        if reuse_flat_parameters_path:
            fq_flat_utils.load_flat_parameters(flatquant_args, self.model, path=reuse_flat_parameters_path)
            self._align_reused_flatquant_module_devices(self.model)
        elif flatquant_args.resume:
            fq_flat_utils.load_flat_parameters(flatquant_args, self.model)
        elif flatquant_args.reload_matrix:
            fq_flat_utils.load_flat_matrices(flatquant_args, self.model, path=flatquant_args.matrix_path)
        elif (flatquant_args.cali_trans or flatquant_args.add_diag or flatquant_args.lwc or flatquant_args.lac):
            fq_train_utils.cali_flat_quant(flatquant_args, self.model, trainloader, fq_utils.DEV, logger=_FlatQuantLogger())

        if flatquant_args.save_matrix and not flatquant_args.reload_matrix:
            fq_flat_utils.save_flat_matrices(flatquant_args, self.model)

        fq_flat_utils.reparameterize_model(self.model)
        self._post_flatquant_float_weights = self._snapshot_float_weights()

        quantizers = None
        if flatquant_args.w_bits < 16:
            quantizers = fq_rtn_utils.rtn_fwrd(self.model, fq_utils.DEV, flatquant_args)

        self._rtn_quantizers = quantizers or {}

        if flatquant_args.quantized_save:
            fq_flat_utils.save_quantized_weights_with_safetensors(flatquant_args, self.model, quantizers)

        if flatquant_args.distribute_model:
            fq_utils.distribute_model(self.model)
        else:
            self.model.to(fq_utils.DEV)

        self.model.eval()
        return trainloader, flatquant_args

    def _snapshot_float_weights(self):
        float_weights = {}
        model_body = getattr(self.model, "model", None)
        layers = getattr(model_body, "layers", None)
        if layers is None:
            return float_weights
        for layer_idx, layer in enumerate(layers):
            full = self._find_linear_modules(layer)
            for name, module in full.items():
                full_name = f"model.layers.{layer_idx}.{name}"
                float_weights[full_name] = module.weight.detach().clone()
        return float_weights

    def quantize_model_sequential(self, calibration_data, n_samples: int = 128, reuse_flat_parameters_path: str | None = None):
        trainloader, _flatquant_args = self._run_flatquant_raw(n_samples, reuse_flat_parameters_path=reuse_flat_parameters_path)
        float_weights = self._post_flatquant_float_weights or self._snapshot_float_weights()
        if self.post_correction is None:
            return

        calibration_subset = [batch[0] for batch in trainloader][:n_samples]
        calibration_loader = self._build_calibration_loader(calibration_subset)

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        inps, attention_mask, position_ids = self._get_layer_inputs(calibration_loader)
        outs = torch.zeros_like(inps)
        layers = self.model.model.layers

        print("\n" + "=" * 80)
        print("FlatQuant RTN Sequential Quantization")
        print("=" * 80)
        print(f"  Weight bits: {self.config.w_bits}")
        print(f"  Activation bits: {self.config.a_bits}")
        print("  KV cache quantization: disabled")

        for layer_idx in tqdm(range(len(layers)), desc="Quantizing FlatQuant layers"):
            layer = layers[layer_idx].to(self.device)
            full = self._find_linear_modules(layer)
            subset = {name: module for name, module in full.items() if name.endswith(".linear")}
            activation_data, _ = self._collect_subset_activations(layer, subset, inps, attention_mask, position_ids)

            for name, module in subset.items():
                full_name = f"model.layers.{layer_idx}.{name}"
                float_weight = float_weights.get(full_name) if float_weights is not None else None
                if float_weight is not None:
                    float_weight = float_weight.to(module.weight.device)
                quantizer = self._rtn_quantizers.get(full_name) if self._rtn_quantizers is not None else None
                self._quantize_module(full_name, module, activation_data.get(name, []), float_weight_override=float_weight, quantizer_override=quantizer)

            with torch.no_grad():
                for sample_idx in range(inps.shape[0]):
                    layer_output = layer(
                        inps[sample_idx].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )
                    outs[sample_idx] = self._extract_hidden_states(layer_output)

            layers[layer_idx] = layer.cpu()
            del layer
            inps, outs = outs, inps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.model.config.use_cache = use_cache
        self.model.eval()

    def build_evaluation_target(self) -> dict:
        target = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "evaluation_target": self.describe_evaluation_target(),
        }
        if self.config.debug_diagnostics:
            target["layer_stats"] = self.layer_stats
        return target



