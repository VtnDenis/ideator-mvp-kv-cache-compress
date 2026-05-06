import torch
import time
from typing import Optional, Dict, Any, List
from transformers import PreTrainedModel
from .shrinkq import eOptShrinkQ, eOptShrinkQ_reconstruct
from .utils import CacheStats


class KVCacheCompressionCallback:
    def __init__(
        self,
        keep_ratio: float = 0.85,
        quantize: bool = True,
        compress_every_n_tokens: int = 128,
        min_cache_length: int = 64,
        layers_to_compress: Optional[List[int]] = None,
        verbose: bool = False,
    ):
        self.keep_ratio = keep_ratio
        self.quantize = quantize
        self.compress_every_n_tokens = compress_every_n_tokens
        self.min_cache_length = min_cache_length
        self.layers_to_compress = layers_to_compress
        self.verbose = verbose
        self.stats = CacheStats()
        self._compressed_state = {}
        self._token_count = 0
        self._last_compress_at = 0

    def should_compress(self, cache_length: int) -> bool:
        if cache_length < self.min_cache_length:
            return False
        tokens_since_last = cache_length - self._last_compress_at
        return tokens_since_last >= self.compress_every_n_tokens

    def compress_kv_cache(self, past_key_values: tuple) -> tuple:
        if past_key_values is None:
            return past_key_values

        seq_len = past_key_values[0][0].shape[2]
        if not self.should_compress(seq_len):
            return past_key_values

        new_cache = []
        layers = self.layers_to_compress or range(len(past_key_values))

        for layer_idx, layer in enumerate(past_key_values):
            k_cache, v_cache = layer

            if layer_idx not in layers:
                new_cache.append((k_cache, v_cache))
                continue

            orig_bytes = (
                k_cache.numel() * k_cache.element_size()
                + v_cache.numel() * v_cache.element_size()
            )

            t0 = time.perf_counter()
            k_comp, v_comp, k_params, v_params, meta = eOptShrinkQ(
                k_cache,
                v_cache,
                keep_ratio=self.keep_ratio,
                quantize=self.quantize,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            comp_bytes = (
                k_comp.numel() * k_comp.element_size()
                + v_comp.numel() * v_comp.element_size()
            )

            self.stats.update(layer_idx, orig_bytes, comp_bytes, elapsed_ms)
            self._compressed_state[layer_idx] = {
                "k_params": k_params,
                "v_params": v_params,
                "quantized": self.quantize,
            }

            new_cache.append((k_comp, v_comp))

            if self.verbose:
                print(
                    f"[KV-Compress] Layer {layer_idx}: "
                    f"rank={meta.get('key_truncated_rank', '?')}, "
                    f"energy={meta.get('key_retained_energy', '?'):.4f}, "
                    f"time={elapsed_ms:.2f}ms"
                )

        self._last_compress_at = seq_len
        return tuple(new_cache)

    def reconstruct_kv_cache(self, past_key_values: tuple) -> tuple:
        if past_key_values is None or not self._compressed_state:
            return past_key_values

        new_cache = []
        for layer_idx, layer in enumerate(past_key_values):
            k_cache, v_cache = layer
            state = self._compressed_state.get(layer_idx)

            if state is None:
                new_cache.append((k_cache, v_cache))
            else:
                k_rec, v_rec = eOptShrinkQ_reconstruct(
                    k_cache,
                    v_cache,
                    state["k_params"],
                    state["v_params"],
                    state["quantized"],
                )
                new_cache.append((k_rec, v_rec))

        return tuple(new_cache)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.summary()

    def on_generate(self, model: PreTrainedModel, past_key_values: tuple) -> tuple:
        return self.compress_kv_cache(past_key_values)
