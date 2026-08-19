"""Day 38: run nanoserve as an HTTP server over the real Llama-3.2-1B.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python serve.py
    cd ~/nanoserve && .venv/bin/python serve.py --max-batch-size 16 --max-model-len 4096
    cd ~/nanoserve && .venv/bin/python serve.py --device cpu --kv-cache-bytes 2147483648

Then, from anywhere:

    curl localhost:8000/health
    curl localhost:8000/v1/completions -H 'content-type: application/json' \
      -d '{"model":"nanoserve","prompt":"The test of a","max_tokens":32}'

Everything interesting is in `nanoserve.launch`; this file is the flags and
`uvicorn.run`. The one thing worth knowing before you type a number: `--max-batch-size`
and `--max-model-len` are not just admission limits, they are *memory* limits, because
the profile run sizes the KV pool against the largest forward those two allow. Doubling
either shrinks the pool it leaves behind, and the boot line prints what you got.

A CPU launch has no VRAM to divide, so it needs `--kv-cache-bytes` (or
`--num-blocks`) spelled out. On a GPU both are optional and the launcher measures.
"""

import argparse
import sys

try:
    import uvicorn
except ImportError:
    sys.exit("uvicorn not installed; pip install -e '.[server]' (or use .venv/bin/python)")

from nanoserve.launch import build_app


def main() -> None:
    p = argparse.ArgumentParser(description="serve nanoserve over HTTP")
    p.add_argument("--weights", default="weights", help="path to the weights dir")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--dtype", default="auto", help="auto | bfloat16 | float16 | float32")
    p.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    p.add_argument("--max-batch-size", type=int, default=8, help="concurrent slots")
    p.add_argument("--max-model-len", type=int, default=2048, help="prompt + generation")
    p.add_argument(
        "--utilization",
        type=float,
        default=0.90,
        help="share of the whole card this engine may occupy",
    )
    p.add_argument(
        "--kv-cache-bytes",
        type=int,
        default=None,
        help="size the pool from this many bytes instead of probing the device",
    )
    p.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="skip sizing entirely and take exactly this many blocks",
    )
    p.add_argument(
        "--no-profile",
        action="store_true",
        help="estimate the activation reserve on paper instead of measuring it",
    )
    p.add_argument("--model-name", default="nanoserve", help="the name /v1/completions serves")
    args = p.parse_args()

    print(f"loading {args.weights} ...", file=sys.stderr, flush=True)
    app = build_app(
        args.weights,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        max_batch_size=args.max_batch_size,
        max_model_len=args.max_model_len,
        utilization=args.utilization,
        kv_cache_bytes=args.kv_cache_bytes,
        num_blocks=args.num_blocks,
        profile=not args.no_profile,
    )
    # The one line that says what the launcher decided on your behalf. Every number
    # in it was either a flag you passed or a division against what the card had left.
    print(app.state.plan.describe(), file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
