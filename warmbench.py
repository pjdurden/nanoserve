"""Day 55: where a run's recordings happen, and what moving them costs.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python warmbench.py
    cd ~/nanoserve && .venv/bin/python warmbench.py --steps 8,32,64,128 \
        --csv docs/daily/data/day-55-warmbench.csv

No weights: the same two-layer toy model over a real batched cache that Day 54's
benchmark used, so the two tables read side by side.

**The column that decides the day is `cold`**, and it is a count of recordings that
happened *inside* the decode loop, in front of a client waiting for a token. A lazy
engine's cold count is not one. It is one per width bucket the run crosses, so a
long generation keeps paying: `lazy_captures` is that arithmetic and the `cold`
column is it measured. A warmed engine's cold count is zero at every length, which
is the entire claim.

**The timing columns are not the point, for Day 54's reason.** On CPU there is no
`torch.cuda.graph`, so the recorder is `eager_recorder`, a stand-in with a capture's
semantics and none of its speed. What the `warm_ms` column *is* honest about is the
shape of the trade: warming pays for every shape in the closed set, up front, and a
lazy run pays only for the ones it reaches. The second table prices both sides at a
plausible per-capture cost so the two numbers can be compared without pretending
this box measured either.

The third table is the budget Day 54 could not compute. A shared pool is sized by
its largest member, so a byte budget is a statement about exactly one number: how
wide the widest capture in the list may be. `width_ceiling` is that inversion, and
at serving size it is a small number.
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
from nanoserve.captured import CapturedDecode, eager_recorder, workspace_bytes
from nanoserve.config import ModelConfig
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.plan import plan_decode
from nanoserve.warmup import (
    check_all_warm,
    check_no_cold_captures,
    check_warm_graphs_unbound,
    lazy_captures,
    stall_seconds,
    warm_decode,
    warm_shapes,
    warmup_seconds,
    width_ceiling,
)

#: The shape a real deployment would warm over, for the second and third tables.
#: The same three numbers Day 54's pool table used, so the two days line up.
SERVING_ROWS = 256
SERVING_LEN = 8192
SERVING_HEADS = 32

#: A plausible cost of recording one graph on a real card, in seconds. Not measured
#: here and not pretended to be: it is a knob so the trade can be read in the units
#: it is actually paid in. vLLM's capture pass over a few dozen shapes is seconds,
#: not milliseconds, which is where this number comes from.
PER_CAPTURE_S = 0.05


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


def _cache(cfg, rows, prompt, steps, block_size, width_multiple):
    """A cache with `rows` rows prefilled, ready to decode."""
    per_row = -(-(prompt + steps) // block_size) + 2
    allocator = BlockAllocator(num_blocks=per_row * rows + 1, block_size=block_size)
    cache = BatchedPagedKVCache(
        cfg,
        allocator,
        rows,
        max_model_len=prompt + steps + block_size,
        bucket_decode=True,
        persist_inputs=True,
    )
    cache.decode_buckets = DecodeBuckets(
        rows, cache.max_model_len, width_multiple=width_multiple
    )
    batch = pad_prompts([list(range(1, prompt + 1))] * rows, pad_id=0)
    torch.manual_seed(1)
    for layer in range(cfg.num_hidden_layers):
        shape = (rows, cfg.num_key_value_heads, batch.max_length, cfg.head_dim)
        cache.write(layer, torch.randn(*shape), torch.randn(*shape), batch.attention_mask)
    return cache


def run(cfg, model, rows, prompt, steps, block_size, width_multiple, *, warm):
    """Decode `steps` tokens, with the capture list warmed first or not.

    Returns `(captured, warmed, warm_s)`: the wrapper, how many graphs the warm-up
    recorded, and what it cost. Everything the wrapper records *after* that is a
    recording that happened in the decode loop.
    """
    cache = _cache(cfg, rows, prompt, steps, block_size, width_multiple)
    captured = CapturedDecode(
        model.forward, mode="capture", recorder=eager_recorder, warmup=0, limit=4096
    )
    warmed, warm_s = 0, 0.0
    if warm:
        report = warm_decode(captured, cache, warm_shapes(cache.decode_buckets))
        check_all_warm(captured, report.shapes)
        check_warm_graphs_unbound(captured)
        warmed, warm_s = report.captures, report.seconds
    index = tuple(range(rows))
    for _ in range(steps):
        plan = plan_decode(cache, rows=index)
        view = cache.view(index, plan=plan)
        ids = cache.decode_inputs.set_input_ids([1] * rows, window=plan.graph_rows)
        captured(ids, plan.positions, cache=view)
    return captured, warmed, warm_s


def sweep(steps_list, rows, prompt, block_size, width_multiple, repeats):
    cfg = _config()
    model = _model(cfg)
    out = []
    for steps in steps_list:
        lazy, _, _ = run(cfg, model, rows, prompt, steps, block_size, width_multiple, warm=False)
        hot, warmed, warm_s = run(
            cfg, model, rows, prompt, steps, block_size, width_multiple, warm=True
        )
        check_no_cold_captures(hot, warmed=warmed)

        def best(**kwargs) -> float:
            return min(
                measure_call(
                    lambda: run(
                        cfg, model, rows, prompt, steps, block_size, width_multiple, **kwargs
                    )
                )
                for _ in range(repeats)
            )

        lazy_s = best(warm=False)
        hot_s = best(warm=True)
        predicted = lazy_captures(
            steps=steps, start_width=prompt, width_multiple=width_multiple
        )
        row = {
            "steps": steps,
            "rows": rows,
            "prompt": prompt,
            "width_multiple": width_multiple,
            "cold_lazy": lazy.captures,
            "cold_lazy_predicted": predicted,
            "cold_warm": hot.captures - warmed,
            "warmed": warmed,
            "warm_ms": round(warm_s * 1e3, 3),
            "lazy_ms": round(lazy_s * 1e3, 3),
            "warmed_run_ms": round(hot_s * 1e3, 3),
        }
        out.append(row)
        print(
            f"  steps={steps:<5} cold: lazy {row['cold_lazy']:>3} (predicted "
            f"{predicted:>3})  ->  warmed {row['cold_warm']:>3}   warm list "
            f"{warmed:>3} graphs in {warm_s * 1e3:7.1f}ms   run "
            f"{lazy_s * 1e3:8.3f}ms -> {hot_s * 1e3:8.3f}ms",
            file=sys.stderr,
            flush=True,
        )
    return out


def stall_table(lengths, width_multiple) -> None:
    """The same recordings, priced where each design pays for them.

    The uncomfortable column is the last one but two. Warming the *whole* closed set
    is not free and it is not small: the set is a product, and paying for all of it
    up front costs more wall clock than any single run's stalls ever add up to. The
    warm list is therefore a serving decision on both axes, which is what
    `warm_shapes(max_rows=..., max_width=...)` is for, and the trimmed columns are
    that decision priced.
    """
    buckets = DecodeBuckets(SERVING_ROWS, SERVING_LEN, width_multiple=width_multiple)
    full = warmup_seconds(len(warm_shapes(buckets)), PER_CAPTURE_S)
    by_rows = warmup_seconds(len(warm_shapes(buckets, max_rows=8)), PER_CAPTURE_S)
    both = warmup_seconds(
        len(warm_shapes(buckets, max_rows=8, max_width=2048)), PER_CAPTURE_S
    )
    print(
        f"\nwhere the recordings are paid for, at {PER_CAPTURE_S * 1e3:.0f} ms a graph "
        f"({buckets.count} shapes in the full set):",
        file=sys.stderr,
    )
    print(
        "  generated   lazy: graphs   in a client's latency   warmed at startup: "
        "full   rows<=8   rows<=8, ctx<=2048",
        file=sys.stderr,
    )
    for steps in lengths:
        stall = stall_seconds(
            steps=steps, start_width=0, width_multiple=width_multiple, per_capture_s=PER_CAPTURE_S
        )
        n = lazy_captures(steps=steps, start_width=0, width_multiple=width_multiple)
        print(
            f"  {steps:>9}   {n:>12}   {stall:>17.2f}s   {full:>21.2f}s "
            f"{by_rows:>8.2f}s {both:>18.2f}s",
            file=sys.stderr,
        )


def budget_table(budgets) -> None:
    """A byte budget, read as the only thing a shared pool's budget is about."""
    print(
        f"\nwhat a capture pool budget buys at {SERVING_ROWS} rows, {SERVING_HEADS} "
        "heads, fp32 scores:",
        file=sys.stderr,
    )
    for budget in budgets:
        ceiling = width_ceiling(
            rows=SERVING_ROWS, num_heads=SERVING_HEADS, budget_bytes=budget
        )
        buckets = DecodeBuckets(SERVING_ROWS, SERVING_LEN, width_multiple=128)
        fits = [s for s in buckets.shapes if s.context_width <= ceiling]
        print(
            f"  {budget / 1e6:>8.0f} MB free -> widest capture {ceiling:>6} tokens, "
            f"{len(fits):>4} of {buckets.count} shapes in the list",
            file=sys.stderr,
        )
    top = DecodeBuckets(SERVING_ROWS, SERVING_LEN, width_multiple=128).shapes[-1]
    print(
        f"  the whole list needs {workspace_bytes(top, SERVING_HEADS) / 1e6:.0f} MB, "
        f"sized by {top.rows} x {top.context_width} alone",
        file=sys.stderr,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "steps",
        "rows",
        "prompt",
        "width_multiple",
        "cold_lazy",
        "cold_lazy_predicted",
        "cold_warm",
        "warmed",
        "warm_ms",
        "lazy_ms",
        "warmed_run_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve capture warm-up benchmark")
    p.add_argument("--steps", type=_int_list, default=[8, 32, 64, 128])
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--prompt", type=int, default=16)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--width-multiple", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3, help="whole-run repeats, best-of")
    p.add_argument("--csv", default="docs/daily/data/day-55-warmbench.csv")
    args = p.parse_args()

    print(
        f"capture warm-up, host only ({args.rows} rows, {args.prompt}-token prompt, "
        f"width multiple {args.width_multiple}, eager_recorder stand-in):",
        file=sys.stderr,
    )
    rows = sweep(
        args.steps, args.rows, args.prompt, args.block_size, args.width_multiple, args.repeats
    )
    stall_table([128, 512, 2048, 8192], 128)
    budget_table([64e6, 268e6, 1e9, 4e9])
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
