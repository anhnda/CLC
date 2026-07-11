from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Dict, Optional, TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.quantization.state import IntegerQuantizedTensorState

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

if TYPE_CHECKING:
    from src.post_correction.clc import CLCCorrection


@dataclass
class AWQConfig:
    bits: int = 4
    n_grid: int = 20
    group_size: int = 128
    max_tokens_per_sample: int = 2048
    layer_batch_size: int = 16
    lmhead_chunks: int = 4
    skip_lmhead: bool = True


class AWQQuantizerXL:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        config: Optional[AWQConfig] = None,
        post_correction: Optional["CLCCorrection"] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or AWQConfig()
        self.post_correction = post_correction
        self.activation_data: Dict[str, list] = {}
        self.layer_stats: Dict[str, dict] = {}

        print("\n[AWQ Quantizer XL Initialized]")
        print(f"  Config: {asdict(self.config)}")
        print(f"  Post correction: {type(self.post_correction).__name__ if self.post_correction else 'none'}")

    def get_hook(self, name: str):
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []

            inp = input[0] if isinstance(input, tuple) else input
            if inp.dim() == 3 and inp.shape[1] > self.config.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len)[: self.config.max_tokens_per_sample]
                indices = indices.sort()[0]
                inp = inp[:, indices, :]

            self.activation_data[name].append(inp.detach().cpu().float())

        return hook

    @torch.no_grad()
    def get_activation_stats(self, name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if name not in self.activation_data or not self.activation_data[name]:
            return None, None

        x_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in x_list)
        in_features = x_list[0].shape[-1]

        l2_sum = torch.zeros(in_features, dtype=torch.float32)
        mean_sum = torch.zeros(in_features, dtype=torch.float32)

        for x in x_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            l2_sum += x_flat.pow(2).sum(dim=0)
            mean_sum += x_flat.sum(dim=0)

        salience = l2_sum / total_samples
        raw_mean = mean_sum / total_samples
        return salience, raw_mean

    @staticmethod
    def empty_flip_stats() -> dict:
        return {"total": 0}

    @torch.no_grad()
    def quantize_weight_groupwise_raw(self, w: torch.Tensor) -> IntegerQuantizedTensorState:
        out_features, in_features = w.shape
        device = w.device

        n_groups = (in_features + self.config.group_size - 1) // self.config.group_size
        padded_in_features = n_groups * self.config.group_size

        if padded_in_features > in_features:
            w_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=w.dtype)
            w_padded[:, :in_features] = w
        else:
            w_padded = w

        w_g = w_padded.reshape(out_features, n_groups, self.config.group_size)
        w_min = w_g.min(dim=2, keepdim=True)[0]
        w_max = w_g.max(dim=2, keepdim=True)[0]
        max_int = 2 ** self.config.bits - 1

        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        scale_flat = scale.repeat(1, 1, self.config.group_size).reshape(out_features, padded_in_features)
        zp_flat = zp.repeat(1, 1, self.config.group_size).reshape(out_features, padded_in_features)

        w_div = w_padded / scale_flat
        pre_round = w_div + zp_flat
        w_int = torch.round(pre_round).clamp(0, max_int)

        return IntegerQuantizedTensorState(
            float_weights=w_padded,
            pre_round=pre_round,
            integer_weights=w_int,
            scale=scale_flat,
            zero_point=zp_flat,
            max_int=max_int,
            min_int=0,
            in_features=in_features,
            padded_in_features=padded_in_features,
            original_dtype=w.dtype,
        )

    @torch.no_grad()
    def search_best_scale(self, name: str, module: nn.Linear):
        if name not in self.activation_data or not self.activation_data[name]:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience, _ = self.get_activation_stats(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)

        x_list = self.activation_data[name]
        x_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in x_list], dim=0)
        max_samples = min(2048, x_cpu.shape[0])
        if x_cpu.shape[0] > max_samples:
            indices = torch.randperm(x_cpu.shape[0])[:max_samples]
            x_search = x_cpu[indices].to(self.device)
        else:
            x_search = x_cpu.to(self.device)

        del x_cpu

        if x_search.dtype != module.weight.dtype:
            x_search = x_search.to(module.weight.dtype)

        w = module.weight.data
        y_orig = torch.matmul(x_search, w.t())

        best_error = float("inf")
        best_alpha = 0.0
        best_scales = torch.ones(w.shape[1], device=self.device)

        activation_salience = activation_salience.clamp(min=1e-5)
        for grid_idx in range(self.config.n_grid + 1):
            alpha = grid_idx / self.config.n_grid
            scales = activation_salience.pow(alpha)
            w_scaled = w * scales.unsqueeze(0)
            quant_state = self.quantize_weight_groupwise_raw(w_scaled)
            w_recon = quant_state.dequantize_truncated() / scales.unsqueeze(0)
            y_quant = torch.matmul(x_search, w_recon.t())
            error = (y_orig - y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

            del w_scaled, quant_state, w_recon, y_quant, scales

        del x_search, y_orig
        torch.cuda.empty_cache()
        return best_scales, best_alpha, best_error

    def build_post_correction_means(self, name: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.post_correction is None:
            return None

        _, raw_mean = self.get_activation_stats(name)
        if raw_mean is None:
            return None

        prepared = self.post_correction.prepare_activation_means(raw_mean)
        return prepared.to(self.device).to(dtype)

    @torch.no_grad()
    def quantize_layer(self, name: str, module: nn.Linear):
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        w = module.weight.data
        original_dtype = w.dtype
        w_scaled = w * best_scales.unsqueeze(0)
        quant_state = self.quantize_weight_groupwise_raw(w_scaled)

        outlier_pct = 0.0
        flip_stats = self.empty_flip_stats()
        if self.post_correction is not None:
            post_mean = self.build_post_correction_means(name, w.dtype)
            if post_mean is None:
                post_mean = torch.zeros(w.shape[1], device=w.device, dtype=w.dtype)
            scaled_act_mean = post_mean / best_scales
            w_quant, outlier_pct, flip_stats = self.post_correction.apply(quant_state, scaled_act_mean)
        else:
            w_quant = quant_state.dequantize_truncated()

        w_final = (w_quant / best_scales.unsqueeze(0)).to(original_dtype)
        module.weight.data = w_final

        self.layer_stats[name] = {
            "alpha": best_alpha,
            "error": best_error,
            "outlier_percent": outlier_pct,
            "flip_stats": flip_stats,
        }

        del best_scales, w_scaled, quant_state, w_quant, w_final
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def quantize_lmhead_half_by_half(self, name: str, module: nn.Linear, debug: bool = False, num_chunks: int = 4):
        w = module.weight.data
        original_dtype = w.dtype
        out_features, in_features = w.shape

        _, raw_mean = self.get_activation_stats(name)
        if raw_mean is None:
            post_mean = torch.zeros(in_features, device=self.device, dtype=w.dtype)
        elif self.post_correction is None:
            post_mean = raw_mean.to(self.device).to(w.dtype)
        else:
            post_mean = self.post_correction.prepare_activation_means(raw_mean).to(self.device).to(w.dtype)

        chunk_size = out_features // num_chunks
        chunk_boundaries = [
            (i * chunk_size, out_features if i == num_chunks - 1 else (i + 1) * chunk_size)
            for i in range(num_chunks)
        ]

        w_final_chunks = []
        chunk_stats = []
        activation_salience, _ = self.get_activation_stats(name)
        activation_salience = activation_salience.to(self.device).to(w.dtype).clamp(min=1e-5)
        x_list = self.activation_data[name]
        x_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in x_list], dim=0)

        for start_idx, end_idx in chunk_boundaries:
            w_chunk = w[start_idx:end_idx, :]
            max_samples = min(1024, x_cpu.shape[0])
            if x_cpu.shape[0] > max_samples:
                indices = torch.randperm(x_cpu.shape[0])[:max_samples]
                x_search = x_cpu[indices].to(self.device)
            else:
                x_search = x_cpu.to(self.device)
            if x_search.dtype != w.dtype:
                x_search = x_search.to(w.dtype)

            y_orig = torch.matmul(x_search, w_chunk.t())
            best_error = float("inf")
            best_alpha = 0.0
            best_scales = torch.ones(w_chunk.shape[1], device=self.device)

            for grid_idx in range(self.config.n_grid + 1):
                alpha = grid_idx / self.config.n_grid
                scales = activation_salience.pow(alpha)
                w_scaled = w_chunk * scales.unsqueeze(0)
                quant_state = self.quantize_weight_groupwise_raw(w_scaled)
                w_recon = quant_state.dequantize_truncated() / scales.unsqueeze(0)
                y_quant = torch.matmul(x_search, w_recon.t())
                error = (y_orig - y_quant).pow(2).mean().item()

                if error < best_error:
                    best_error = error
                    best_alpha = alpha
                    best_scales = scales.clone()

                del w_scaled, quant_state, w_recon, y_quant, scales

            w_scaled = w_chunk * best_scales.unsqueeze(0)
            quant_state = self.quantize_weight_groupwise_raw(w_scaled)
            if self.post_correction is not None:
                scaled_act_mean = post_mean / best_scales
                w_quant, outlier_pct, flip_stats = self.post_correction.apply(
                    quant_state,
                    scaled_act_mean,
                    debug=(debug and start_idx == 0),
                )
            else:
                w_quant = quant_state.dequantize_truncated()
                outlier_pct = 0.0
                flip_stats = self.empty_flip_stats()
            w_final_chunk = (w_quant / best_scales.unsqueeze(0)).to(original_dtype)

            w_final_chunks.append(w_final_chunk)
            chunk_stats.append(
                {
                    "alpha": best_alpha,
                    "error": best_error,
                    "outlier_percent": outlier_pct,
                    "flip_stats": flip_stats,
                }
            )

            del x_search, y_orig, w_chunk, w_scaled, quant_state, w_quant, w_final_chunk
            torch.cuda.empty_cache()

        module.weight.data = torch.cat(w_final_chunks, dim=0)
        self.layer_stats[name] = {
            "alpha": float(np.mean([s["alpha"] for s in chunk_stats])),
            "error": float(np.mean([s["error"] for s in chunk_stats])),
            "outlier_percent": float(np.mean([s["outlier_percent"] for s in chunk_stats])),
            "flip_stats": {"total": int(sum(s["flip_stats"]["total"] for s in chunk_stats))},
        }

        del w_final_chunks, chunk_stats, post_mean, x_cpu
        torch.cuda.empty_cache()

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples: int = 500):
        print(f"  Calibrating {len(layer_names_batch)} layers...")
        self.model.eval()
        handles = []

        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append(handle)

        successful = 0
        with torch.no_grad():
            for text in tqdm(calibration_data[:n_samples], desc="  Calibration", leave=False):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1

                    if (successful + 1) % 32 == 0:
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        for handle in handles:
            handle.remove()

        torch.cuda.empty_cache()
        gc.collect()

    def quantize_model_sequential(self, calibration_data, n_samples: int = 500):
        print("\n" + "=" * 80)
        print("Batched Sequential Quantization")
        print("=" * 80)
        print(f"  Strategy: Process {self.config.layer_batch_size} layers per batch")

        if HAS_PSUTIL:
            print(f"  Initial System RAM: {psutil.virtual_memory().percent:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]
        num_layers = len(layer_names)
        num_batches = (num_layers + self.config.layer_batch_size - 1) // self.config.layer_batch_size

        quantized_count = 0
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.config.layer_batch_size
            batch_end = min(batch_start + self.config.layer_batch_size, num_layers)
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Layers {batch_start}-{batch_end - 1}")
            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            print(f"  Quantizing {len(batch_layers)} layers...")
            for name, module in tqdm(batch_layers, desc="  Quantization", leave=False):
                try:
                    is_lmhead = "lm_head" in name.lower() or name.endswith("lm_head")
                    if is_lmhead and self.config.skip_lmhead:
                        print(f"  Skipping lm_head (skip_lmhead=True): {name}")
                        continue
                    if is_lmhead:
                        self.quantize_lmhead_half_by_half(
                            name,
                            module,
                            debug=(quantized_count < 2),
                            num_chunks=self.config.lmhead_chunks,
                        )
                    else:
                        self.quantize_layer(name, module)
                    quantized_count += 1
                except Exception as exc:
                    print(f"\nWarning: error quantizing {name}: {exc}")
                    continue

            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

        print("\n" + "=" * 80)
        print("Quantization Complete")
        print(f"  Total layers quantized: {quantized_count}/{num_layers}")
        print("=" * 80)

