import torch
from typing import Tuple


def quantize_kv_to_int8(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t_fp32 = tensor.float()
    t_min = t_fp32.amin(dim=-1, keepdim=True)
    t_max = t_fp32.amax(dim=-1, keepdim=True)
    scale = (t_max - t_min) / 255.0
    scale = torch.where(scale < 1e-9, torch.ones_like(scale) * 1e-9, scale)
    zero_point = t_min
    quantized = ((t_fp32 - zero_point) / scale).round().clamp(0, 255).to(torch.uint8)
    return quantized, scale.to(tensor.dtype), zero_point.to(tensor.dtype)


def dequantize_kv_from_int8(
    quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    return quantized.float() * scale + zero_point
