import json
import os
from urllib.parse import parse_qs
import mimetypes

STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)

MODEL_CONFIGS = {
    "gpt2": {"layers": 12, "heads": 12, "head_dim": 64, "dtype": "fp16"},
    "gpt2-medium": {"layers": 24, "heads": 16, "head_dim": 64, "dtype": "fp16"},
    "gpt2-large": {"layers": 36, "heads": 20, "head_dim": 64, "dtype": "fp16"},
    "gpt2-xl": {"layers": 48, "heads": 25, "head_dim": 64, "dtype": "fp16"},
    "llama-7b": {"layers": 32, "heads": 32, "head_dim": 128, "dtype": "fp16"},
    "llama-13b": {"layers": 40, "heads": 40, "head_dim": 128, "dtype": "fp16"},
    "llama-70b": {"layers": 80, "heads": 64, "head_dim": 128, "dtype": "fp16"},
}


def estimate_memory(
    num_layers, num_heads, head_dim, seq_len, compression_ratio, dtype_bytes
):
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


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")
    query = environ.get("QUERY_STRING", "")

    if method == "OPTIONS":
        start_response(
            "200 OK",
            [
                ("Access-Control-Allow-Origin", "*"),
                ("Access-Control-Allow-Methods", "GET, OPTIONS"),
                ("Access-Control-Allow-Headers", "*"),
            ],
        )
        return [b""]

    if path == "/api/configs":
        body = json.dumps({"models": MODEL_CONFIGS}).encode()
        start_response(
            "200 OK",
            [
                ("Content-Type", "application/json"),
                ("Access-Control-Allow-Origin", "*"),
            ],
        )
        return [body]

    if path == "/api/estimate":
        params = parse_qs(query)
        result = estimate_memory(
            num_layers=int(params.get("layers", [32])[0]),
            num_heads=int(params.get("heads", [32])[0]),
            head_dim=int(params.get("head_dim", [128])[0]),
            seq_len=int(params.get("seq_len", [2048])[0]),
            compression_ratio=float(params.get("ratio", [4.0])[0]),
            dtype_bytes=int(params.get("dtype", [2])[0]),
        )
        start_response(
            "200 OK",
            [
                ("Content-Type", "application/json"),
                ("Access-Control-Allow-Origin", "*"),
            ],
        )
        return [json.dumps(result).encode()]

    filepath = path.lstrip("/") or "index.html"
    full_path = os.path.normpath(os.path.join(STATIC_DIR, filepath))

    if not full_path.startswith(os.path.normpath(STATIC_DIR)):
        start_response("403 Forbidden", [("Content-Type", "text/plain")])
        return [b"Forbidden"]

    if os.path.exists(full_path) and os.path.isfile(full_path):
        mime, _ = mimetypes.guess_type(full_path)
        with open(full_path, "rb") as f:
            content = f.read()
        headers = [("Content-Type", mime or "application/octet-stream")]
        if mime and mime.startswith("text/"):
            headers.append(("Cache-Control", "public, max-age=3600"))
        start_response("200 OK", headers)
        return [content]

    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]


app = application
