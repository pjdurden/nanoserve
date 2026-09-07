"""Day 52: count the shapes a decode run presents, open against bucketed.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python bucketbench.py
    cd ~/nanoserve && .venv/bin/python bucketbench.py --rows 3 --max-batch 8 \
        --prompt 16 --steps 8,64,128,512 --csv docs/daily/data/day-52-bucketbench.csv

No weights and no model, and no compiler either. What is measured is the *shape
history* a run presents to a guard, which is a property of the addressing and not
of the arithmetic: both arms grow the same block tables through the same
`plan_decode` and the two rectangles agree on every real cell before anything is
counted. Timing a `torch.compile` sweep would measure this box's compiler; counting
shapes measures the design.

  open (Day 51):  `[rows, max_ctx]` with `max_ctx` growing by one a step. One new
    shape per step, so `static` mode asks for one build per step and dynamo
    abandons the frame after eight.
  bucketed (Day 52): the batch padded up to a row bucket and the rectangle read at
    a width multiple. The shape count stops depending on the run length and starts
    depending only on how many buckets the run crossed.

The columns that matter are `shapes_open` against `shapes_bucketed` (what the day
bought) and `waste` (what it cost). `plan_ms` is here to answer a narrower
question, which is whether the padding made the *host* side slower: it should not,
because a padded read is the same basic slice one row taller and the extra write
slots are a list append.

The second table is the one that decides whether any of this is a capture list.
`count` is a product of the two axes, so at a real `max_model_len` a 128-token
width multiple closes the set into 576 shapes, which is worse than leaving it open.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from nanoserve.benchmark import measure_call
from nanoserve.buckets import DecodeBuckets, check_capture_budget, waste
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.compiled import RECOMPILE_LIMIT, DecodeShape
from nanoserve.config import ModelConfig
from nanoserve.plan import plan_decode

#: The widths a real deployment would be closing over, for the second table. A
#: serving `max_model_len` and a serving `max_batch_size`, not the toy ones the
#: sweep uses.
SERVING_ROWS = 256
SERVING_LEN = 8192


def _config() -> ModelConfig:
    """A model small enough that the cache is pure addressing and no K/V is written."""
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _cache(rows: int, max_batch: int, prompt: int, steps: int, block_size: int):
    """A cache of `max_batch` rows with the first `rows` of them prefilled.

    The two numbers are deliberately different, because that is where the row axis
    comes from: a scheduler admits whatever fits and the cache keeps every slot, so
    a batch of 3 in a cache of 8 is the ordinary case and the one a row bucket has
    something to pad into. The prefill goes straight into the block tables rather
    than through `write`, because no K/V is needed: every arm here reads addressing
    and nothing else.
    """
    per_row = -(-(prompt + steps) // block_size) + 1
    allocator = BlockAllocator(num_blocks=per_row * rows + 1, block_size=block_size)
    cache = BatchedPagedKVCache(
        _config(), allocator, max_batch, max_model_len=prompt + steps
    )
    for table in cache.tables[:rows]:
        table.append(prompt)
    return cache


def run(rows: int, max_batch: int, prompt: int, steps: int, block_size: int, buckets):
    """Plan `steps` decode steps and hand back the shapes they presented."""
    cache = _cache(rows, max_batch, prompt, steps, block_size)
    index = tuple(range(rows))
    shapes = []
    plan = None
    for _ in range(steps):
        plan = plan_decode(cache, rows=index, buckets=buckets)
        shapes.append(DecodeShape(rows=plan.graph_rows, context_width=plan.graph_width))
    return tuple(shapes), plan


def sweep(steps_list, rows, max_batch, prompt, block_size, width_multiple, repeats):
    out = []
    for steps in steps_list:
        buckets = DecodeBuckets(max_batch, prompt + steps, width_multiple=width_multiple)
        open_shapes, open_plan = run(rows, max_batch, prompt, steps, block_size, None)
        bucket_shapes, bucket_plan = run(rows, max_batch, prompt, steps, block_size, buckets)
        real = open_plan.slot_mapping[: open_plan.batch_size, : open_plan.max_ctx]
        padded = bucket_plan.slot_mapping[: bucket_plan.batch_size, : bucket_plan.max_ctx]
        assert real.tolist() == padded.tolist(), f"rectangles differ at {steps} steps"
        assert (
            open_plan.context_lens.tolist()
            == bucket_plan.context_lens.tolist()[: bucket_plan.batch_size]
        ), f"lengths differ at {steps} steps"

        def best(fn) -> float:
            return min(measure_call(fn) for _ in range(repeats))

        open_s = best(lambda: run(rows, max_batch, prompt, steps, block_size, None))
        bucket_s = best(lambda: run(rows, max_batch, prompt, steps, block_size, buckets))
        row = {
            "steps": steps,
            "rows": rows,
            "max_batch": max_batch,
            "prompt": prompt,
            "row_bucket": buckets.row_bucket(rows),
            "width_multiple": width_multiple,
            "shapes_open": len(set(open_shapes)),
            "shapes_bucketed": len(set(bucket_shapes)),
            "falls_back_open": int(len(set(open_shapes)) > RECOMPILE_LIMIT),
            "falls_back_bucketed": int(len(set(bucket_shapes)) > RECOMPILE_LIMIT),
            "waste": round(waste(open_shapes, buckets), 4),
            "cells_open": sum(s.cells for s in open_shapes),
            "cells_bucketed": sum(s.cells for s in bucket_shapes),
            "open_ms": round(open_s * 1e3, 3),
            "bucketed_ms": round(bucket_s * 1e3, 3),
        }
        out.append(row)
        print(
            f"  steps={steps:<5} shapes {row['shapes_open']:>4} -> "
            f"{row['shapes_bucketed']:<3}  waste {row['waste']:6.1%}  "
            f"cells {row['cells_open']:>10,} -> {row['cells_bucketed']:>10,}  "
            f"plan {open_s * 1e3:8.3f}ms -> {bucket_s * 1e3:8.3f}ms",
            file=sys.stderr,
            flush=True,
        )
    return out


def budget_table(multiples) -> None:
    """What the closed set costs at a serving shape, per width multiple."""
    print(
        f"\nclosed-set size at {SERVING_ROWS} rows x {SERVING_LEN} tokens:",
        file=sys.stderr,
    )
    for multiple in multiples:
        buckets = DecodeBuckets(SERVING_ROWS, SERVING_LEN, width_multiple=multiple)
        try:
            check_capture_budget(buckets, limit=64)
            verdict = "a capture list"
        except AssertionError:
            verdict = "a compile bill"
        print(
            f"  width multiple {multiple:>5}: {len(buckets.rows)} rows x "
            f"{len(buckets.widths):>3} widths = {buckets.count:>4} shapes   {verdict}",
            file=sys.stderr,
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "steps",
        "rows",
        "max_batch",
        "prompt",
        "row_bucket",
        "width_multiple",
        "shapes_open",
        "shapes_bucketed",
        "falls_back_open",
        "falls_back_bucketed",
        "waste",
        "cells_open",
        "cells_bucketed",
        "open_ms",
        "bucketed_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve decode-shape bucketing benchmark")
    p.add_argument("--steps", type=_int_list, default=[8, 64, 128, 512])
    p.add_argument("--rows", type=int, default=3, help="rows the batch actually holds")
    p.add_argument("--max-batch", type=int, default=8, help="rows the cache has")
    p.add_argument("--prompt", type=int, default=16)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--width-multiple", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3, help="whole-run repeats, best-of")
    p.add_argument("--csv", default="docs/daily/data/day-52-bucketbench.csv")
    args = p.parse_args()

    print(
        f"decode shapes, host only ({args.rows} of {args.max_batch} rows, "
        f"{args.prompt}-token prompt, width multiple {args.width_multiple}):",
        file=sys.stderr,
    )
    rows = sweep(
        args.steps,
        args.rows,
        args.max_batch,
        args.prompt,
        args.block_size,
        args.width_multiple,
        args.repeats,
    )
    budget_table([128, 512, 2048, 8192])
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
