"""Day 51: time the read rectangle rebuilt against the read rectangle appended to.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python slotbench.py
    cd ~/nanoserve && .venv/bin/python slotbench.py --rows 4 --prompt 16 \
        --steps 8,64,128,512 --csv docs/daily/data/day-51-slotbench.csv

No weights and no model. The addressing does not know what a token is: both arms
walk the same `BlockTable`s over the same pool and produce the same
`[rows, max_ctx]` int64 rectangle, so this times the host-side bookkeeping of a
decode step and nothing else. That is the point. Day 50's rectangle was correct and
cost `rows * max_ctx` Python-level `slot()` calls plus one host-to-device build,
every step, with `max_ctx` growing by one each time.

  rebuild (Day 50): `rebuild_mapping(tables, rows)`, the whole rectangle from
    scratch. Quadratic in the length of the generation.
  window (Day 51): one `SlotTable.append` of `rows` cells into a persistent
    `[max_batch_size, max_model_len]` buffer, then `read`, which narrows it to
    `slots[:rows, :width]` and hands back the buffer's own storage.

Both arms grow the same block tables the same way, and the two rectangles are
asserted equal at every measured length before anything is timed, because a
speedup over a different answer is not a speedup.

The honest CPU result is that the win is entirely in the growth rate rather than in
a constant: at eight steps the two are close, and by 512 the rebuild is doing
hundreds of times the work to say the same thing. `cells_written` is the column
that does not depend on this box.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from nanoserve.benchmark import measure_call
from nanoserve.cache import BlockAllocator, BlockTable
from nanoserve.slots import SlotTable, incremental_cells, rebuild_mapping, table_bytes
from nanoserve.plan import rebuild_cells


def _tables(rows: int, prompt: int, steps: int, block_size: int) -> list[BlockTable]:
    """One prefilled block table per row, over a pool big enough for the whole run."""
    per_row = -(-(prompt + steps) // block_size) + 1
    allocator = BlockAllocator(num_blocks=per_row * rows + 1, block_size=block_size)
    tables = []
    for _ in range(rows):
        table = BlockTable(allocator)
        table.append(prompt)
        tables.append(table)
    return tables


def rebuild_run(rows: int, prompt: int, steps: int, block_size: int):
    """Day 50: grow every row by one and build the whole rectangle, every step."""
    tables = _tables(rows, prompt, steps, block_size)
    index = tuple(range(rows))
    mapping = lens = None
    for _ in range(steps):
        for table in tables:
            table.append(1)
        mapping, lens = rebuild_mapping(tables, index)
    return mapping, lens, None


def window_run(rows: int, prompt: int, steps: int, block_size: int, scatter: bool = False):
    """Day 51: catch the mirror up once, then one cell per row per step.

    `scatter` is the gotcha arm. The rectangle is only a view when the forward's
    rows are a prefix of the table's, so this puts the same rows at even slots of a
    table twice as tall and reads `(0, 2, 4, ...)`: identical answer, an
    `index_select` instead of a narrow, and a fresh allocation every step. That is
    what a scheduler holding non-adjacent slots costs, and it is the reason
    `check_mapping_is_window` names the row set rather than the cache.
    """
    tables = _tables(rows, prompt, steps, block_size)
    index = tuple(range(0, 2 * rows, 2)) if scatter else tuple(range(rows))
    slot_table = SlotTable(2 * rows if scatter else rows, prompt + steps)
    for table, row in zip(tables, index):
        slot_table.resync(table, row)
    mapping = lens = None
    for _ in range(steps):
        slots = []
        for table in tables:
            start = table.num_tokens
            table.append(1)
            slots.append(table.slot(start))
        slot_table.append(index, slots)
        mapping, lens = slot_table.read(index, tables[0].num_tokens)
    return mapping, lens, slot_table


def sweep(steps_list, rows, prompt, block_size, repeats):
    out = []
    for steps in steps_list:
        want_map, want_lens, _ = rebuild_run(rows, prompt, steps, block_size)
        got_map, got_lens, table = window_run(rows, prompt, steps, block_size)
        scattered_map, _, scattered = window_run(rows, prompt, steps, block_size, True)
        assert got_map.tolist() == want_map.tolist(), f"rectangles differ at {steps} steps"
        assert got_lens.tolist() == want_lens.tolist(), f"lengths differ at {steps} steps"
        assert scattered_map.tolist() == want_map.tolist(), f"gather differs at {steps}"
        assert table.window_share == 1.0 and scattered.window_share == 0.0

        def best(fn) -> float:
            return min(measure_call(fn) for _ in range(repeats))

        rebuild_s = best(lambda: rebuild_run(rows, prompt, steps, block_size))
        window_s = best(lambda: window_run(rows, prompt, steps, block_size))
        gather_s = best(lambda: window_run(rows, prompt, steps, block_size, True))
        row = {
            "steps": steps,
            "rows": rows,
            "prompt": prompt,
            "rebuild_ms": round(rebuild_s * 1e3, 3),
            "window_ms": round(window_s * 1e3, 3),
            "gather_ms": round(gather_s * 1e3, 3),
            "speedup": round(rebuild_s / window_s, 2) if window_s else 0.0,
            "rebuild_cells": rebuild_cells(steps, rows, prompt),
            "window_cells": incremental_cells(steps, rows, prompt),
            "cell_ratio": round(
                rebuild_cells(steps, rows, prompt) / incremental_cells(steps, rows, prompt), 1
            ),
            "table_bytes": table_bytes(rows, prompt + steps),
        }
        out.append(row)
        print(
            f"  steps={steps:<5} rebuild {rebuild_s * 1e3:9.3f}ms  "
            f"window {window_s * 1e3:9.3f}ms  gather {gather_s * 1e3:9.3f}ms  "
            f"-> {row['speedup']:6.2f}x   "
            f"cells {row['rebuild_cells']:>9,} vs {row['window_cells']:>7,} "
            f"({row['cell_ratio']}x)",
            file=sys.stderr,
            flush=True,
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "steps",
        "rows",
        "prompt",
        "rebuild_ms",
        "window_ms",
        "gather_ms",
        "speedup",
        "rebuild_cells",
        "window_cells",
        "cell_ratio",
        "table_bytes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve slot-table microbenchmark")
    p.add_argument("--steps", type=_int_list, default=[8, 64, 128, 512])
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--prompt", type=int, default=16)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--repeats", type=int, default=3, help="whole-run repeats, best-of")
    p.add_argument("--csv", default="docs/daily/data/day-51-slotbench.csv")
    args = p.parse_args()

    print(
        f"decode addressing, host only ({args.rows} rows, {args.prompt}-token prompt):",
        file=sys.stderr,
    )
    rows = sweep(args.steps, args.rows, args.prompt, args.block_size, args.repeats)
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
