import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_cache_compress import KVCacheCompressionCallback, estimate_memory_savings


def main():
    parser = argparse.ArgumentParser(description="KV-Cache-Compress Demo")
    parser.add_argument("--model", default="gpt2", help="HF model name")
    parser.add_argument("--prompt", default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument(
        "--keep-ratio", type=float, default=0.85, help="SVD energy retention"
    )
    parser.add_argument(
        "--compress-every", type=int, default=64, help="Compress every N tokens"
    )
    parser.add_argument(
        "--no-quantize", action="store_true", help="Disable quantization"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    callback = KVCacheCompressionCallback(
        keep_ratio=args.keep_ratio,
        quantize=not args.no_quantize,
        compress_every_n_tokens=args.compress_every,
        verbose=args.verbose,
    )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    print(f"Generating {args.max_tokens} tokens...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.7,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    stats = callback.get_stats()

    print("\n" + "=" * 60)
    print("GENERATED TEXT:")
    print(generated)
    print("\n" + "=" * 60)
    print("COMPRESSION STATS:")
    print(f"  Compression calls: {stats['compression_calls']}")
    print(f"  Total time: {stats['total_compression_time_ms']:.2f}ms")
    print(f"  Avg time/call: {stats['avg_compression_time_ms']:.2f}ms")
    print(f"  Memory saved: {stats['total_memory_saved_MB']:.2f} MB")
    print("=" * 60)

    savings = estimate_memory_savings(
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_attention_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        seq_len=args.max_tokens,
    )
    print("\nESTIMATED SAVINGS:")
    print(f"  Original: {savings['original_cache_MB']} MB")
    print(f"  Compressed: {savings['compressed_cache_MB']} MB")
    print(f"  Reduction: {savings['reduction_percent']}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
