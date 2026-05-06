from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv_cache_compress import estimate_memory_savings


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/estimate":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            params = parse_qs(parsed.query)
            config = {
                "num_layers": int(params.get("layers", [32])[0]),
                "num_heads": int(params.get("heads", [32])[0]),
                "head_dim": int(params.get("head_dim", [128])[0]),
                "seq_len": int(params.get("seq_len", [2048])[0]),
                "compression_ratio": float(params.get("ratio", [4.0])[0]),
                "dtype_bytes": int(params.get("dtype", [2])[0]),
            }

            result = estimate_memory_savings(**config)
            self.wfile.write(json.dumps(result).encode())
            return

        if parsed.path == "/api/configs":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            configs = {
                "models": {
                    "gpt2": {
                        "layers": 12,
                        "heads": 12,
                        "head_dim": 64,
                        "dtype": "fp16",
                    },
                    "gpt2-medium": {
                        "layers": 24,
                        "heads": 16,
                        "head_dim": 64,
                        "dtype": "fp16",
                    },
                    "gpt2-large": {
                        "layers": 36,
                        "heads": 20,
                        "head_dim": 64,
                        "dtype": "fp16",
                    },
                    "gpt2-xl": {
                        "layers": 48,
                        "heads": 25,
                        "head_dim": 64,
                        "dtype": "fp16",
                    },
                    "llama-7b": {
                        "layers": 32,
                        "heads": 32,
                        "head_dim": 128,
                        "dtype": "fp16",
                    },
                    "llama-13b": {
                        "layers": 40,
                        "heads": 40,
                        "head_dim": 128,
                        "dtype": "fp16",
                    },
                    "llama-70b": {
                        "layers": 80,
                        "heads": 64,
                        "head_dim": 128,
                        "dtype": "fp16",
                    },
                }
            }
            self.wfile.write(json.dumps(configs).encode())
            return

        if parsed.path == "/" or parsed.path == "":
            self.path = "/static/index.html"

        return SimpleHTTPRequestHandler.do_GET(self)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"KV-Cache-Compress server running on port {port}")
    server.serve_forever()
