"""Day 54: how many graphs a decode run asks for, and what one pool holds.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python capturebench.py
    cd ~/nanoserve && .venv/bin/python capturebench.py --steps 8,32,64,128 \
        --csv docs/daily/data/day-54-capturebench.csv

No weights: a two-layer toy model over a real batched cache, so the whole decode
path runs (plan, write, paged read, logits) at a size that fits on a laptop.

**The timing column is not the point and this file will not pretend otherwise.**
On CPU there is no `torch.cuda.graph`, so the recorder is `eager_recorder`: a
stand-in that reproduces a capture's *semantics* (fixed input addresses, one fixed
output buffer, a replay that takes no arguments) and none of its speed. It re-runs
the forward and copies the result into the output buffer, so the captured arm is a
forward plus a copy and is slower by exactly that copy. A real replay's saving is
host-side launch overhead, which this box cannot produce because it has nothing to
launch. `launch_saving_s` is where that arithmetic lives.

The column that decides the day is `captures`. A bucketed run records one graph per
shape in a closed set and then stops; an unbucketed one finds a new shape every step,
because the read rectangle grows by one column per token, so it records a graph per
step forever. That is the whole of Day 52 restated in the currency that pays for it.

The second table is the memory question Day 53 ended on and it answers in a direction
I did not expect. Every capture is handed the first one's pool, so the arena is sized
by the largest shape rather than by all of them. The saving is about 5x over a
36-shape list and not 36x, because a bucket set is geometric and the biggest shape is
most of the bill. The absolute number is the one that matters: this engine's read
materialises a `[rows, heads, 1, ctx]` score rectangle, and at serving size that one
intermediate is hundreds of megabytes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.batch import pad_prompts
from nanoserve.benchmark import measure_call
from nanoserve.buckets import DecodeBuckets
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.captured import (
    CapturedDecode,
    check_all_shapes_captured,
    check_pool_shared,
    check_replays_dominate,
    eager_recorder,
    private_pool_bytes,
    shared_pool_bytes,
    workspace_bytes,
)
from nanoserve.config import ModelConfig
from nanoserve.inputs import input_bytes
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.plan import plan_decode
from nanoserve.slots import table_bytes

#: The shape a real deployment would be capturing over, for the second table. The
#: same two numbers Day 53's benchmark used, plus Llama-3.2-1B's head count, so the
#: three days' memory tables are readable side by side.
SERVING_ROWS = 256
SERVING_LEN = 8192
SERVING_HEADS = 32


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _model(cfg: ModelConfig) -> LlamaModel:
    torch.manual_seed(0)
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg))


def _cache(cfg, rows, prompt, steps, block_size, *, bucketed):
    """A cache with `rows` rows prefilled, ready to decode."""
    per_row = -(-(prompt + steps) // block_size) + 2
    allocator = BlockAllocator(num_blocks=per_row * rows + 1, block_size=block_size)
    cache = BatchedPagedKVCache(
        cfg,
        allocator,
        rows,
        max_model_len=prompt + steps + block_size,
        bucket_decode=bucketed,
        persist_inputs=True,
    )
    batch = pad_prompts([list(range(1, prompt + 1))] * rows, pad_id=0)
    torch.manual_seed(1)
    for layer in range(cfg.num_hidden_layers):
        shape = (rows, cfg.num_key_value_heads, batch.max_length, cfg.head_dim)
        cache.write(layer, torch.randn(*shape), torch.randn(*shape), batch.attention_mask)
    return cache


def run(cfg, model, rows, prompt, steps, block_size, *, bucketed, capture):
    """Decode `steps` tokens, optionally through a capture. Returns the wrapper."""
    cache = _cache(cfg, rows, prompt, steps, block_size, bucketed=bucketed)
    forward = CapturedDecode(
        model.forward,
        mode="capture" if capture else "off",
        recorder=eager_recorder,
        warmup=0,
        limit=steps + 1,
        precheck=bucketed,
    )
    index = tuple(range(rows))
    for _ in range(steps):
        plan = plan_decode(cache, rows=index)
        view = cache.view(index, plan=plan)
        ids = cache.decode_inputs.set_input_ids([1] * rows, window=plan.graph_rows)
        forward(ids, plan.positions, cache=view)
    return forward


def sweep(steps_list, rows, prompt, block_size, repeats):
    cfg = _config()
    model = _model(cfg)
    out = []
    for steps in steps_list:
        open_set = run(
            cfg, model, rows, prompt, steps, block_size, bucketed=False, capture=True
        )
        closed = run(
            cfg, model, rows, prompt, steps, block_size, bucketed=True, capture=True
        )
        check_all_shapes_captured(closed)
        check_pool_shared(closed)
        check_replays_dominate(closed)

        def best(**kwargs) -> float:
            return min(
                measure_call(
                    lambda: run(cfg, model, rows, prompt, steps, block_size, **kwargs)
                )
                for _ in range(repeats)
            )

        eager_s = best(bucketed=True, capture=False)
        replay_s = best(bucketed=True, capture=True)
        row = {
            "steps": steps,
            "rows": rows,
            "prompt": prompt,
            "captures_open": open_set.captures,
            "captures_closed": closed.captures,
            "replays_closed": closed.replays,
            "reuse_closed": round(closed.reuse, 4),
            "output_bytes_closed": closed.output_bytes,
            "eager_ms": round(eager_s * 1e3, 3),
            "replayed_ms": round(replay_s * 1e3, 3),
        }
        out.append(row)
        print(
            f"  steps={steps:<5} captures  open {row['captures_open']:>4}  ->  "
            f"closed {row['captures_closed']:<3}  replays {row['replays_closed']:>4}  "
            f"reuse {closed.reuse:5.0%}  run {eager_s * 1e3:8.3f}ms -> "
            f"{replay_s * 1e3:8.3f}ms",
            file=sys.stderr,
            flush=True,
        )
    return out


def pool_table(multiples) -> None:
    """One shared pool against a private one per graph, at serving size."""
    print(
        f"\ncapture pool at {SERVING_ROWS} rows x {SERVING_LEN} tokens, "
        f"{SERVING_HEADS} heads, fp32 scores:",
        file=sys.stderr,
    )
    for multiple in multiples:
        buckets = DecodeBuckets(SERVING_ROWS, SERVING_LEN, width_multiple=multiple)
        shapes = buckets.shapes
        shared = shared_pool_bytes(shapes, SERVING_HEADS)
        private = private_pool_bytes(shapes, SERVING_HEADS)
        print(
            f"  width multiple {multiple:>5}: {buckets.count:>4} shapes, shared "
            f"{shared / 1e6:>9.1f} MB, private {private / 1e6:>10.1f} MB, "
            f"{private / shared:>5.1f}x",
            file=sys.stderr,
        )
    biggest = DecodeBuckets(SERVING_ROWS, SERVING_LEN).shapes[-1]
    print(
        "\nwhat a decode step holds at the top of the list, side by side:\n"
        f"  the score rectangle  {workspace_bytes(biggest, SERVING_HEADS):>12,} bytes\n"
        f"  the Day-51 slot table{table_bytes(SERVING_ROWS, SERVING_LEN):>13,} bytes\n"
        f"  the Day-53 inputs    {input_bytes(SERVING_ROWS):>12,} bytes",
        file=sys.stderr,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "steps",
        "rows",
        "prompt",
        "captures_open",
        "captures_closed",
        "replays_closed",
        "reuse_closed",
        "output_bytes_closed",
        "eager_ms",
        "replayed_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve decode capture benchmark")
    p.add_argument("--steps", type=_int_list, default=[8, 32, 64, 128])
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--prompt", type=int, default=16)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--repeats", type=int, default=3, help="whole-run repeats, best-of")
    p.add_argument("--csv", default="docs/daily/data/day-54-capturebench.csv")
    args = p.parse_args()

    print(
        f"decode capture, host only ({args.rows} rows, {args.prompt}-token prompt, "
        "eager_recorder stand-in):",
        file=sys.stderr,
    )
    rows = sweep(args.steps, args.rows, args.prompt, args.block_size, args.repeats)
    pool_table([128, 512, 2048, 8192])
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
