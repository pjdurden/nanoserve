"""Day 53: time a decode step's input build, fresh tensors against fixed buffers.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python inputbench.py
    cd ~/nanoserve && .venv/bin/python inputbench.py --rows 3 --max-batch 8 \
        --prompt 16 --steps 8,64,128,512 --csv docs/daily/data/day-53-inputbench.csv

No weights and no model. What is measured is the host side of `plan_decode`: the
three addressing tensors a step hands the forward, built one way or the other. Both
arms grow the same block tables through the same call and every real cell of every
tensor is compared before anything is timed, so the only difference is where the
numbers land.

  fresh (Day 52):  `positions`, `write_slots` and `context_lens` are built from
    Python lists into new tensors, three allocations a step at three new addresses.
  buffers (Day 53): the same three are written into `[max_batch]` int64 buffers
    allocated once, and the forward is handed `buffer[:graph_rows]`.

The column that decides the day is `allocations`, not `ms`. On CPU the two arms
should be close, because a `[3]` int64 tensor is cheap to allocate and the buffered
write still has to get a Python list into storage; what changes is that one arm ends
the run with the same four addresses it started with and the other has handed out
three new ones per step. `moves` is the other one: a buffer that was reallocated
mid-run is a captured graph reading storage that is no longer its input.

The second table is the memory question the day set out to answer and answers in
the other direction. One buffer set covers a whole capture list, because a buffer is
indexed by batch position and every shape's window is a prefix of it. It would not
have mattered much either way: the input side of a decode step is four `[max_batch]`
vectors, which is kilobytes against the slot table's megabytes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from nanoserve.benchmark import measure_call
from nanoserve.buckets import DecodeBuckets
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.config import ModelConfig
from nanoserve.inputs import (
    DECODE_INPUTS,
    check_addresses_stable,
    check_one_set_covers,
    fresh_allocations,
    input_bytes,
    per_shape_bytes,
    sharing_ratio,
)
from nanoserve.plan import plan_decode
from nanoserve.slots import table_bytes

#: The shape a real deployment would be capturing over, for the second table. A
#: serving `max_model_len` and a serving `max_batch_size`, not the sweep's toys.
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


def _cache(rows: int, max_batch: int, prompt: int, steps: int, block_size: int, persist: bool):
    """A cache of `max_batch` rows with the first `rows` of them prefilled.

    The prefill goes straight into the block tables rather than through `write`,
    because no K/V is needed: every arm here builds addressing and nothing else.
    """
    per_row = -(-(prompt + steps) // block_size) + 1
    allocator = BlockAllocator(num_blocks=per_row * rows + 1, block_size=block_size)
    cache = BatchedPagedKVCache(
        _config(),
        allocator,
        max_batch,
        max_model_len=prompt + steps,
        persist_inputs=persist,
    )
    for table in cache.tables[:rows]:
        table.append(prompt)
    return cache


def run(rows, max_batch, prompt, steps, block_size, persist, buckets=None):
    """Plan `steps` decode steps and hand back the cache and the last plan."""
    cache = _cache(rows, max_batch, prompt, steps, block_size, persist)
    index = tuple(range(rows))
    plan = None
    for _ in range(steps):
        plan = plan_decode(cache, rows=index, buckets=buckets)
    return cache, plan


def sweep(steps_list, rows, max_batch, prompt, block_size, repeats):
    out = []
    for steps in steps_list:
        fresh_cache, fresh_plan = run(rows, max_batch, prompt, steps, block_size, False)
        held_cache, held_plan = run(rows, max_batch, prompt, steps, block_size, True)
        assert fresh_plan.positions.tolist() == held_plan.positions.tolist(), "positions"
        assert fresh_plan.write_slots.tolist() == held_plan.write_slots.tolist(), "slots"
        assert fresh_plan.context_lens.tolist() == held_plan.context_lens.tolist(), "lengths"
        assert held_cache.decode_inputs.owns(held_plan.positions), "positions not a window"
        check_addresses_stable(held_cache.decode_inputs, held_cache.decode_inputs.addresses)

        def best(persist: bool) -> float:
            return min(
                measure_call(lambda: run(rows, max_batch, prompt, steps, block_size, persist))
                for _ in range(repeats)
            )

        fresh_s, held_s = best(False), best(True)
        inputs = held_cache.decode_inputs
        row = {
            "steps": steps,
            "rows": rows,
            "max_batch": max_batch,
            "prompt": prompt,
            "allocations_fresh": fresh_allocations(steps, DECODE_INPUTS[1:]),
            "allocations_buffered": 0,
            "writes": inputs.writes,
            "written_cells": inputs.written_cells,
            "moves": inputs.moves,
            "buffer_bytes": inputs.bytes,
            "fresh_ms": round(fresh_s * 1e3, 3),
            "buffered_ms": round(held_s * 1e3, 3),
        }
        out.append(row)
        print(
            f"  steps={steps:<5} allocations {row['allocations_fresh']:>6} -> "
            f"{row['allocations_buffered']:<2}  writes {row['writes']:>6}  "
            f"cells {row['written_cells']:>7,}  moves {row['moves']}  "
            f"plan {fresh_s * 1e3:8.3f}ms -> {held_s * 1e3:8.3f}ms",
            file=sys.stderr,
            flush=True,
        )
    return out


def memory_table(multiples) -> None:
    """What one buffer set costs against what a set per captured shape would."""
    print(
        f"\ninput buffers at {SERVING_ROWS} rows, against the day's other buffers:",
        file=sys.stderr,
    )
    shared = input_bytes(SERVING_ROWS)
    table = table_bytes(SERVING_ROWS, SERVING_LEN)
    print(
        f"  one set of {len(DECODE_INPUTS)} inputs   {shared:>12,} bytes\n"
        f"  the Day-51 slot table  {table:>12,} bytes  "
        f"({table / shared:.0f}x the inputs)",
        file=sys.stderr,
    )
    for multiple in multiples:
        buckets = DecodeBuckets(SERVING_ROWS, SERVING_LEN, width_multiple=multiple)
        shapes = buckets.shapes
        check_one_set_covers(shapes, _inputs(SERVING_ROWS))
        print(
            f"  width multiple {multiple:>5}: {buckets.count:>4} shapes, "
            f"{per_shape_bytes(shapes):>10,} bytes a set each, "
            f"{sharing_ratio(shapes, SERVING_ROWS):>6.1f}x what one set costs",
            file=sys.stderr,
        )


def _inputs(rows: int):
    from nanoserve.inputs import DecodeInputs

    return DecodeInputs(rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "steps",
        "rows",
        "max_batch",
        "prompt",
        "allocations_fresh",
        "allocations_buffered",
        "writes",
        "written_cells",
        "moves",
        "buffer_bytes",
        "fresh_ms",
        "buffered_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve decode-input buffer benchmark")
    p.add_argument("--steps", type=_int_list, default=[8, 64, 128, 512])
    p.add_argument("--rows", type=int, default=3, help="rows the batch actually holds")
    p.add_argument("--max-batch", type=int, default=8, help="rows the cache has")
    p.add_argument("--prompt", type=int, default=16)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--repeats", type=int, default=3, help="whole-run repeats, best-of")
    p.add_argument("--csv", default="docs/daily/data/day-53-inputbench.csv")
    args = p.parse_args()

    print(
        f"decode inputs, host only ({args.rows} of {args.max_batch} rows, "
        f"{args.prompt}-token prompt):",
        file=sys.stderr,
    )
    rows = sweep(
        args.steps, args.rows, args.max_batch, args.prompt, args.block_size, args.repeats
    )
    memory_table([128, 512, 2048, 8192])
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
