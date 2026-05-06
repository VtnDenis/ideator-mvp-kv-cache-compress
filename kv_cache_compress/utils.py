import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time


def create_circular_mask(
    seq_len: int, past_len: int, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros((seq_len, past_len + seq_len), device=device)
    for i in range(seq_len):
        mask[i, : past_len + i + 1] = 1.0
    return mask


def estimate_memory_savings(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    compression_ratio: float = 4.0,
    dtype_bytes: int = 2,
) -> Dict[str, Any]:
    cache_size = num_layers * 2 * num_heads * seq_len * head_dim * dtype_bytes
    compressed_size = cache_size / compression_ratio
    saved = cache_size - compressed_size

    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "sequence_length": seq_len,
        "original_cache_MB": round(cache_size / (1024 * 1024), 2),
        "compressed_cache_MB": round(compressed_size / (1024 * 1024), 2),
        "memory_saved_MB": round(saved / (1024 * 1024), 2),
        "compression_ratio": round(compression_ratio, 1),
        "reduction_percent": round((1 - 1 / compression_ratio) * 100, 1),
    }


class CacheStats:
    def __init__(self):
        self.total_compression_time_ms = 0.0
        self.total_decompression_time_ms = 0.0
        self.compression_calls = 0
        self.memory_saved_total_MB = 0.0
        self.layer_stats = []

    def update(self, layer_idx: int, orig_size: int, comp_size: int, elapsed_ms: float):
        self.total_compression_time_ms += elapsed_ms
        self.compression_calls += 1
        saved = orig_size - comp_size
        self.memory_saved_total_MB += saved / (1024 * 1024)
        self.layer_stats.append(
            {
                "layer": layer_idx,
                "original_bytes": orig_size,
                "compressed_bytes": comp_size,
                "ratio": round(orig_size / max(comp_size, 1), 2),
                "time_ms": round(elapsed_ms, 3),
            }
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "compression_calls": self.compression_calls,
            "total_compression_time_ms": round(self.total_compression_time_ms, 2),
            "avg_compression_time_ms": round(
                self.total_compression_time_ms / max(self.compression_calls, 1), 2
            ),
            "total_memory_saved_MB": round(self.memory_saved_total_MB, 2),
            "layer_stats": self.layer_stats,
        }
