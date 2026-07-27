"""Minimal stand-in for a vLLM OpenAI-compatible server.

Purpose: prove the harness speaks the real wire protocol before renting a GPU.
This validates request shape and response parsing — NOT numerics. It answers
"will our client ask for logprobs and understand what comes back", which is the
failure that would silently waste the first paid hour.

Response shape mirrors vLLM's /v1/chat/completions with logprobs=true.
"""

from __future__ import annotations

import json
import math
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

RECEIVED: list[dict] = []


def _logprobs(n_tokens: int, top_k: int, seed: int) -> list[dict]:
    """Build an OpenAI-shaped `logprobs.content` array."""
    state = seed
    content = []
    for pos in range(n_tokens):
        state = (1103515245 * state + 12345) % (2**31)
        base = -0.05 - (state / (2**31)) * 0.4
        tops = []
        for rank in range(top_k):
            state = (1103515245 * state + 12345) % (2**31)
            lp = base - rank * 1.7 - (state / (2**31)) * 0.2
            tops.append({"token": f"t{rank}", "logprob": lp, "bytes": [110]})
        content.append(
            {
                "token": f"t0",
                "logprob": tops[0]["logprob"],
                "bytes": [110],
                "top_logprobs": tops,
            }
        )
    return content


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence access log
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append(body)

        n = min(int(body.get("max_tokens") or 8), 8)
        want = bool(body.get("logprobs"))
        top_k = int(body.get("top_logprobs") or 0) or 5

        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "length",
        }
        if want:
            choice["logprobs"] = {"content": _logprobs(n, top_k, SEED)}

        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "mock"),
            "choices": [choice],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": n,
                "total_tokens": 5 + n,
            },
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        payload = {
            "object": "list",
            "data": [{"id": "Qwen3-8B", "object": "model", "owned_by": "mock"}],
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
