"""Compatibility shim for older imports."""

from src.quantization.awq import AWQConfig as QuantizationConfig
from src.quantization.awq import AWQQuantizerXL as CLCAWQQuantizerXL
from src.quantization.state import IntegerQuantizedTensorState

__all__ = ["QuantizationConfig", "CLCAWQQuantizerXL", "IntegerQuantizedTensorState"]
