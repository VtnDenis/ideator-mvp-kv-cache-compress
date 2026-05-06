import torch
from typing import Tuple, Optional
from .quantize import quantize_kv_to_int8, dequantize_kv_from_int8


def _svd_truncate(
    X: torch.Tensor, keep_ratio: float = 0.85
) -> Tuple[torch.Tensor, int, float]:
    device = X.device
    dtype = X.dtype
    B, H, S, D = X.shape
    X_reshaped = X.reshape(-1, D).float()

    U, S_vals, Vt = torch.linalg.svd(X_reshaped, full_matrices=False)

    total_energy = (S_vals**2).sum()
    cumulative = torch.cumsum(S_vals**2, dim=0)
    k = int(torch.searchsorted(cumulative / total_energy, keep_ratio).item()) + 1
    k = max(1, min(k, len(S_vals)))

    S_trunc = S_vals[:k]
    U_trunc = U[:, :k]
    Vt_trunc = Vt[:k, :]

    X_denoised = (U_trunc * S_trunc.unsqueeze(0)) @ Vt_trunc
    retained_energy = (S_vals[:k] ** 2).sum() / total_energy

    return X_denoised.to(dtype).reshape(B, H, S, D), k, retained_energy.item()


def eOptShrinkQ(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    keep_ratio: float = 0.85,
    quantize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if keep_ratio >= 1.0 and not quantize:
        return (
            key_cache,
            value_cache,
            None,
            None,
            {"truncated_ranks": 0, "retained_energy": 1.0},
        )

    k_denoised, k_rank, k_energy = _svd_truncate(key_cache, keep_ratio)
    v_denoised, v_rank, v_energy = _svd_truncate(value_cache, keep_ratio)

    meta = {
        "key_truncated_rank": k_rank,
        "value_truncated_rank": v_rank,
        "key_retained_energy": round(k_energy, 4),
        "value_retained_energy": round(v_energy, 4),
        "keep_ratio": keep_ratio,
    }

    if quantize:
        k_quant, k_scale, k_zp = quantize_kv_to_int8(k_denoised)
        v_quant, v_scale, v_zp = quantize_kv_to_int8(v_denoised)
        meta["quantized"] = True
        return k_quant, v_quant, (k_scale, k_zp), (v_scale, v_zp), meta
    else:
        meta["quantized"] = False
        return k_denoised, v_denoised, None, None, meta


def eOptShrinkQ_reconstruct(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_params: Optional[Tuple[torch.Tensor, torch.Tensor]],
    v_params: Optional[Tuple[torch.Tensor, torch.Tensor]],
    quantized: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if quantized and k_params is not None:
        k_scale, k_zp = k_params
        v_scale, v_zp = v_params
        key_cache = dequantize_kv_from_int8(key_cache.float(), k_scale, k_zp)
        value_cache = dequantize_kv_from_int8(value_cache.float(), v_scale, v_zp)
    return key_cache, value_cache
