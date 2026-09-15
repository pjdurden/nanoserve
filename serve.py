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

Day 56 adds `--cuda-graphs`, which is Weeks 12 and 13 switched on: the decode shape
is rounded to a closed set, its inputs are held at addresses that do not move, and
every shape in the set is recorded once at startup off a batch of pure padding. It
prints a second boot line, because the capture list is a second sizing decision and
it is made against a second probe: what the card has left *after* the KV pool. The
list is `row buckets x width buckets`, so it can outgrow what one process will hold,
and when it does the width is what gives. `--warm-rows` and `--warm-width` are how
you trim it on purpose instead of letting the graph limit do it for you.

Day 58 adds the persistent batch, and `--cuda-graphs` turns it on because without it
the recording covers about a quarter of what it was made for. A graph reads the
window `slots[:rows]`, so it can only replay a step whose rows are `(0, 1, ... n-1)`,
and that stops being true the moment one request in a batch finishes before its
neighbour. Compaction moves the survivor down into the hole instead of leaving it
where it was, which costs one row of addressing per completion and nothing of the
K/V. `--no-persistent-batch` separates the two again, for measuring one without the
other.
"""

import argparse
import sys

try:
    import uvicorn
except ImportError:
    sys.exit("uvicorn not installed; pip install -e '.[server]' (or use .venv/bin/python)")

from nanoserve.launch import boot_lines, build_app


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
    p.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="bucket the decode shape, hold its inputs still, and record a graph per shape",
    )
    p.add_argument(
        "--persistent-batch",
        action="store_true",
        default=None,
        help="keep the running rows compacted (default: on with --cuda-graphs)",
    )
    p.add_argument(
        "--no-persistent-batch",
        dest="persistent_batch",
        action="store_false",
        help="leave a survivor in whatever row it was in, the Day-57 behaviour",
    )
    p.add_argument(
        "--no-warm",
        action="store_true",
        help="record the graphs lazily, mid-run, instead of at startup",
    )
    p.add_argument(
        "--warm-rows",
        type=int,
        default=None,
        help="only record shapes up to this many rows (default: the slot count)",
    )
    p.add_argument(
        "--warm-width",
        type=int,
        default=None,
        help="only record shapes up to this context (default: the served length)",
    )
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
        # Day 56. One flag, three switches, and the bundling is allowed here and
        # nowhere below: `Engine.build` refuses two out of three, because a capture
        # over an open shape set or an input allocated per step is not a slower
        # engine, it is a wrong one. An operator asking for CUDA graphs means all
        # three, and should not have to know that.
        bucket_decode=args.cuda_graphs,
        persist_inputs=args.cuda_graphs,
        capture_decode=args.cuda_graphs,
        # Day 58, and the tri-state is the honest shape of the question. Unset, the
        # persistent batch follows the capture, because a capture without it replays
        # a quarter of the loop and that is not what an operator asking for CUDA
        # graphs means. Set either way, it is held on its own: compaction is a
        # scheduler property, it is correct with the graphs off, and separating the
        # two is how a benchmark attributes a number to one of them.
        compact_rows=args.cuda_graphs if args.persistent_batch is None
        else args.persistent_batch,
        warm=not args.no_warm,
        warm_rows=args.warm_rows,
        warm_width=args.warm_width,
    )
    # The lines that say what the launcher decided on your behalf. Every number in
    # them was either a flag you passed or a division against what the card had left,
    # and with graphs on there are two such divisions made at two different moments.
    for line in boot_lines(app.state.plan, app.state.capture, app.state.warmup):
        print(line, file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
