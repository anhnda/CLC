from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn


@dataclass
class BiasCorrectionConfig:
    max_samples: int = 4096


class BiasCorrectionCorrection:
    def __init__(self, config: Optional[BiasCorrectionConfig] = None):
        self.config = config or BiasCorrectionConfig()

    def prepare_activation_means(self, raw_means: torch.Tensor) -> torch.Tensor:
        return raw_means

    def _flatten_activations(self, activation_batches: Iterable[torch.Tensor]) -> torch.Tensor | None:
        if hasattr(activation_batches, "get_sample_rows"):
            samples = activation_batches.get_sample_rows()
            if samples is None or samples.numel() == 0:
                return None
            return samples

        batches = [batch.reshape(-1, batch.shape[-1]) for batch in activation_batches if batch.numel() > 0]
        if not batches:
            return None
        return torch.cat(batches, dim=0)

    @torch.no_grad()
    def compute_bias_delta(
        self,
        module: nn.Linear,
        quantized_weights: torch.Tensor,
        activation_batches: Iterable[torch.Tensor],
        device: str,
    ) -> torch.Tensor:
        activation_rows = self._flatten_activations(activation_batches)
        if activation_rows is None:
            return torch.zeros(module.weight.shape[0], device=device, dtype=module.weight.dtype)

        max_samples = min(self.config.max_samples, activation_rows.shape[0])
        if activation_rows.shape[0] > max_samples:
            sample_indices = torch.randperm(activation_rows.shape[0])[:max_samples]
            activation_rows = activation_rows[sample_indices]

        x_samples = activation_rows.to(device=device, dtype=module.weight.dtype)
        y_orig = torch.matmul(x_samples, module.weight.data.t())
        y_quant = torch.matmul(x_samples, quantized_weights.t())
        bias_delta = (y_orig - y_quant).mean(dim=0)

        del activation_rows, x_samples, y_orig, y_quant
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return bias_delta

    @torch.no_grad()
    def apply_bias_delta(self, module: nn.Linear, bias_delta: torch.Tensor, device: str, dtype: torch.dtype):
        adjustment = bias_delta.to(device=device, dtype=dtype)
        if module.bias is None:
            module.bias = nn.Parameter(adjustment)
            return

        module.bias.data = module.bias.data + adjustment.to(module.bias.data.dtype)
