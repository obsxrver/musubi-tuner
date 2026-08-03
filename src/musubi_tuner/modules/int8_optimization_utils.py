"""INT8 ConvRot linear support for frozen base models used during LoRA training.

Comfy checkpoints store a ConvRot linear as a row-wise INT8 weight plus one
FP32 scale per output channel.  The base weight stays frozen; the custom
autograd function therefore only needs to propagate a gradient to the input.
The forward uses INT8 kernels. Backward dequantizes small row chunks to the
compute dtype so training does not add another INT8 approximation or ever
materialize a full BF16 copy of a large base weight.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn

logger = logging.getLogger(__name__)

_COMFY_KITCHEN = None
_COMFY_KITCHEN_CHECKED = False
_COMFY_KITCHEN_FAILED = False
_MAX_INT32_ACCUMULATOR_BYTES = 256 * 1024 * 1024
_MAX_DEQUANT_WEIGHT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Int8ConvRotConfig:
    group_size: int
    scale_shape: tuple[int, ...]


def _validate_group_size(group_size: int) -> None:
    # Comfy's regular ConvRot matrix is a Kronecker product of H4 matrices.
    reduced = group_size
    while reduced >= 4 and reduced % 4 == 0:
        reduced //= 4
    if group_size < 4 or reduced != 1:
        raise ValueError(f"INT8 ConvRot group size must be a power of 4, got {group_size}")


def _convrot_hadamard(value: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply Comfy's normalized regular Hadamard matrix along the last axis."""

    _validate_group_size(group_size)
    if value.shape[-1] % group_size:
        raise ValueError(f"ConvRot group size {group_size} does not divide feature size {value.shape[-1]}")

    original_shape = value.shape
    transformed = value.reshape(-1, value.shape[-1] // group_size, group_size)
    stride = 1
    while stride < group_size:
        rows = transformed.reshape(*transformed.shape[:-1], group_size // (4 * stride), 4, stride)
        a, b, c, d = rows.unbind(dim=-2)
        transformed = torch.stack((a + b + c - d, a + b - c + d, a - b + c + d, -a + b + c + d), dim=-2).reshape(*transformed.shape)
        stride *= 4
    return (transformed / math.sqrt(group_size)).reshape(original_shape)


def _quantize_int8_rowwise(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (value.detach().abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp_min_(1e-30)
    quantized = (value.float() / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return quantized, scale


def _int8_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Run an INT8 matmul, padding CUDA shapes rejected by some cuBLASLt paths."""

    if lhs.dtype != torch.int8 or rhs.dtype != torch.int8 or lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError("_int8_mm expects two 2-D INT8 tensors")
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(f"INT8 matmul shape mismatch: {tuple(lhs.shape)} and {tuple(rhs.shape)}")

    original_m, original_n = lhs.shape[0], rhs.shape[1]
    if lhs.is_cuda:
        padded_m = max(32, ((original_m + 31) // 32) * 32)
        padded_k = ((lhs.shape[1] + 7) // 8) * 8
        padded_n = ((original_n + 7) // 8) * 8
        if padded_m != lhs.shape[0] or padded_k != lhs.shape[1]:
            padded_lhs = torch.zeros((padded_m, padded_k), dtype=torch.int8, device=lhs.device)
            padded_lhs[: lhs.shape[0], : lhs.shape[1]] = lhs
            lhs = padded_lhs
        if padded_k != rhs.shape[0] or padded_n != rhs.shape[1]:
            padded_rhs = torch.zeros((padded_k, padded_n), dtype=torch.int8, device=rhs.device)
            padded_rhs[: rhs.shape[0], : rhs.shape[1]] = rhs
            rhs = padded_rhs

    int8_mm = getattr(torch, "int8_mm", None) or torch._int_mm
    result = int8_mm(lhs.contiguous(), rhs.contiguous())
    return result[:original_m, :original_n]


def _matmul_chunk_rows(output_features: int) -> int:
    return max(1, _MAX_INT32_ACCUMULATOR_BYTES // (output_features * torch.int32.itemsize))


def _dequant_chunk_rows(input_features: int, dtype: torch.dtype) -> int:
    return max(1, _MAX_DEQUANT_WEIGHT_BYTES // (input_features * dtype.itemsize))


def _get_comfy_kitchen():
    global _COMFY_KITCHEN, _COMFY_KITCHEN_CHECKED
    if not _COMFY_KITCHEN_CHECKED:
        _COMFY_KITCHEN_CHECKED = True
        try:
            import comfy_kitchen

            _COMFY_KITCHEN = comfy_kitchen
            logger.info("Using comfy-kitchen for MiniMax H3 INT8 ConvRot forward kernels")
        except Exception as exc:  # noqa: BLE001 - optional binary backends can fail during import
            logger.info("comfy-kitchen is unavailable; using PyTorch INT8 ConvRot kernels: %s", exc)
    return _COMFY_KITCHEN


def _int8_convrot_forward(
    value: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    global _COMFY_KITCHEN_FAILED
    kitchen = _get_comfy_kitchen() if value.is_cuda and not _COMFY_KITCHEN_FAILED else None
    if kitchen is not None:
        try:
            return kitchen.int8_linear(
                value.contiguous(),
                weight.contiguous(),
                weight_scale,
                bias,
                value.dtype,
                convrot=True,
                convrot_groupsize=group_size,
            )
        except Exception as exc:  # noqa: BLE001 - fall back for unsupported devices and shapes
            _COMFY_KITCHEN_FAILED = True
            logger.warning("comfy-kitchen INT8 ConvRot failed; falling back to PyTorch kernels: %s", exc)

    original_shape = value.shape
    rotated = _convrot_hadamard(value.reshape(-1, original_shape[-1]), group_size)
    quantized, input_scale = _quantize_int8_rowwise(rotated)
    transposed_weight = weight.t().contiguous()
    output = torch.empty((quantized.shape[0], weight.shape[0]), dtype=value.dtype, device=value.device)
    chunk_rows = _matmul_chunk_rows(weight.shape[0])
    channel_scale = weight_scale.float().reshape(1, -1)
    for start in range(0, quantized.shape[0], chunk_rows):
        stop = min(start + chunk_rows, quantized.shape[0])
        accumulated = _int8_mm(quantized[start:stop], transposed_weight)
        output[start:stop] = (accumulated.float() * input_scale[start:stop] * channel_scale).to(value.dtype)
    if bias is not None:
        output = output + bias.to(device=output.device, dtype=output.dtype)
    return output.reshape(*original_shape[:-1], weight.shape[0])


class _Int8ConvRotLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, weight_scale, bias, group_size):
        ctx.group_size = int(group_size)
        ctx.save_for_backward(weight, weight_scale)
        return _int8_convrot_forward(value, weight, weight_scale, bias, ctx.group_size)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        weight, weight_scale = ctx.saved_tensors
        original_shape = grad_output.shape
        compute_dtype = grad_output.dtype
        grad_rows = grad_output.reshape(-1, original_shape[-1]).to(compute_dtype)

        # W_rot = diag(weight_scale) @ weight_int8. Dequantize bounded row
        # chunks for an accurate input gradient without expanding the entire
        # frozen weight at once.
        rotated_grad_input = torch.empty((grad_rows.shape[0], weight.shape[1]), dtype=compute_dtype, device=grad_output.device)
        rotated_grad_input.zero_()
        chunk_rows = _dequant_chunk_rows(weight.shape[1], compute_dtype)
        for start in range(0, weight.shape[0], chunk_rows):
            stop = min(start + chunk_rows, weight.shape[0])
            dequantized_weight = weight[start:stop].to(device=grad_output.device, dtype=compute_dtype)
            scale_chunk = weight_scale if weight_scale.numel() == 1 else weight_scale[start:stop]
            dequantized_weight.mul_(scale_chunk.to(device=grad_output.device, dtype=compute_dtype))
            rotated_grad_input.addmm_(grad_rows[:, start:stop], dequantized_weight)
        grad_input = _convrot_hadamard(rotated_grad_input, ctx.group_size)
        grad_input = grad_input.to(grad_output.dtype).reshape(*original_shape[:-1], weight.shape[1])
        return grad_input, None, None, None, None


def int8_convrot_linear_forward(self: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    if self.weight.dtype != torch.int8:
        raise RuntimeError(f"INT8 ConvRot layer has unexpected weight dtype {self.weight.dtype}")
    return _Int8ConvRotLinearFunction.apply(value, self.weight, self.weight_scale, self.bias, self.int8_convrot_group_size)


def apply_int8_convrot_monkey_patch(model: nn.Module, layer_configs: Mapping[str, Int8ConvRotConfig]) -> nn.Module:
    """Patch selected ``nn.Linear`` instances while preserving their class name for LoRA discovery."""

    remaining = set(layer_configs)
    for name, module in model.named_modules():
        if name not in layer_configs:
            continue
        if module.__class__.__name__ != "Linear" or not isinstance(module, nn.Linear):
            raise TypeError(f"INT8 ConvRot checkpoint layer {name} does not map to nn.Linear")

        config = layer_configs[name]
        _validate_group_size(config.group_size)
        if module.in_features % config.group_size:
            raise ValueError(f"ConvRot group size {config.group_size} does not divide {name}'s {module.in_features} input features")
        expected_scale = (module.out_features, 1)
        if config.scale_shape not in {(), (1,), expected_scale}:
            raise ValueError(f"INT8 scale for {name} has shape {config.scale_shape}; expected scalar or {expected_scale}")

        # assign=True preserves the destination Parameter's requires_grad flag.
        # INT8 tensors cannot require gradients, and the base is frozen anyway.
        module.weight.requires_grad_(False)
        module.register_buffer(
            "weight_scale",
            torch.empty(config.scale_shape, dtype=torch.float32, device=module.weight.device),
        )
        module.int8_convrot_group_size = config.group_size
        module.forward = MethodType(int8_convrot_linear_forward, module)
        remaining.remove(name)

    if remaining:
        preview = ", ".join(sorted(remaining)[:8])
        raise ValueError(f"INT8 ConvRot checkpoint refers to {len(remaining)} unknown Linear layers: {preview}")
    logger.info("Enabled INT8 ConvRot training for %d Linear layers", len(layer_configs))
    return model
