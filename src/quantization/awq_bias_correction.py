from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn as nn

from src.post_correction.bias_correction import BiasCorrectionCorrection
from src.quantization.awq import AWQQuantizerXL


class AWQBiasCorrectionQuantizerXL(AWQQuantizerXL):
    def __init__(self, *args, post_correction: BiasCorrectionCorrection, **kwargs):
        super().__init__(*args, post_correction=post_correction, **kwargs)

    @property
    def bias_correction(self) -> BiasCorrectionCorrection:
        return self.post_correction

    def empty_flip_stats(self) -> dict:
        return {"total": 0, "bias_corrected": True}

    @torch.no_grad()
    def quantize_layer(self, name: str, module: nn.Linear):
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        w = module.weight.data
        original_dtype = w.dtype
        w_scaled = w * best_scales.unsqueeze(0)
        quant_state = self.quantize_weight_groupwise_raw(w_scaled)
        w_quant = quant_state.dequantize_truncated()
        w_final = (w_quant / best_scales.unsqueeze(0)).to(original_dtype)

        bias_delta = self.bias_correction.compute_bias_delta(
            module,
            w_final,
            self.activation_data.get(name, []),
            device=self.device,
        )

        module.weight.data = w_final
        self.bias_correction.apply_bias_delta(module, bias_delta, self.device, original_dtype)

        self.layer_stats[name] = {
            "alpha": best_alpha,
            "error": best_error,
            "outlier_percent": 0.0,
            "flip_stats": self.empty_flip_stats(),
            "bias_delta_norm": float(bias_delta.norm().item()),
        }

        del best_scales, w_scaled, quant_state, w_quant, w_final, bias_delta
        if name in self.activation_data:
            del self.activation_data[name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def quantize_lmhead_half_by_half(self, name: str, module: nn.Linear, debug: bool = False, num_chunks: int = 4):
        del debug
        w = module.weight.data
        original_dtype = w.dtype
        out_features, _ = w.shape

        chunk_size = out_features // num_chunks
        chunk_boundaries = [
            (i * chunk_size, out_features if i == num_chunks - 1 else (i + 1) * chunk_size)
            for i in range(num_chunks)
        ]

        x_list = self.activation_data.get(name, [])
        x_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in x_list], dim=0)
        activation_salience, _ = self.get_activation_stats(name)
        activation_salience = activation_salience.to(self.device).to(w.dtype).clamp(min=1e-5)

        w_final_chunks = []
        bias_deltas = []
        chunk_stats = []

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
            w_quant = quant_state.dequantize_truncated()
            w_final_chunk = (w_quant / best_scales.unsqueeze(0)).to(original_dtype)

            temp_module = nn.Linear(w_chunk.shape[1], w_chunk.shape[0], bias=False, device=w.device, dtype=original_dtype)
            temp_module.weight.data = w_chunk
            bias_delta = self.bias_correction.compute_bias_delta(
                temp_module,
                w_final_chunk,
                self.activation_data.get(name, []),
                device=self.device,
            )

            w_final_chunks.append(w_final_chunk)
            bias_deltas.append(bias_delta)
            chunk_stats.append(
                {
                    "alpha": best_alpha,
                    "error": best_error,
                    "outlier_percent": 0.0,
                    "flip_stats": self.empty_flip_stats(),
                }
            )

            del x_search, y_orig, w_chunk, w_scaled, quant_state, w_quant, w_final_chunk, temp_module, bias_delta
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        module.weight.data = torch.cat(w_final_chunks, dim=0)
        full_bias_delta = torch.cat(bias_deltas, dim=0)
        self.bias_correction.apply_bias_delta(module, full_bias_delta, self.device, original_dtype)

        self.layer_stats[name] = {
            "alpha": float(np.mean([s["alpha"] for s in chunk_stats])),
            "error": float(np.mean([s["error"] for s in chunk_stats])),
            "outlier_percent": 0.0,
            "flip_stats": self.empty_flip_stats(),
            "bias_delta_norm": float(full_bias_delta.norm().item()),
        }

        del w_final_chunks, bias_deltas, full_bias_delta, chunk_stats, x_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
