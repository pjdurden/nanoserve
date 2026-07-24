"""Day 29: measure static batching. Throughput against batch size, and the blocking.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python batchbench.py
    cd ~/nanoserve && .venv/bin/python batchbench.py --sizes 1,2,4,8,16 \
        --prompt-len 64 --new-tokens 16 --csv docs/daily/data/day-29-batchbench.csv

Two measurements, kept apart on purpose, because they are the two halves of the
answer to "was static batching enough?".

  **The sweep.** One prompt replicated to each batch size, every row the same
  length, every row running to the cap. Nothing is ragged, so goodput equals issued
  throughput and the curve is a clean read of what one more row costs in step time.
  This is the good news: decode is memory-bound, so the weights come out of memory
  once per step no matter how many rows ride along, and the efficiency stays high
  until something saturates. The `knee` names the last size that paid its way.

  **The blocking run.** One long row and several short ones in a single batch, with
  the short rows stopping early. A static batch is fixed at the start, so it runs
  until its slowest row is done and returns every row at that moment. Two costs come
  out: the wasted work (`waste_fraction`, tokens the forward computed for rows that
  had already finished) and the wasted time (`max_hol_inflation`, how many times
  longer the first row to finish waited than its own work took). That pair is the
  argument for Week 8, and it is why vLLM and SGLang schedule at the iteration level
  instead of the batch level.

No weights are needed: batching cost is a function of shapes and step counts, not of
which tokens are present, so this builds a model on random weights from the default
Llama-3.2-1B config by default. Point `--weights ./weights` at the real checkpoint to
run the same sweep on real logits; the numbers move, the shape of the story does not.

The rows stop on a step budget rather than on EOS, because output lengths belong to
the model and the prompt while the raggedness is the thing being varied. Every
forward is real, and a finished row keeps its query, its slot and its block, which is
exactly the bill being counted. The timing math lives in `nanoserve.batchbench` and is
unit-tested with a fake clock; this file only wires a real model into it and
prints/saves the numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.batchbench import sweep_batch_sizes, time_model_batch
from nanoserve.config import ModelConfig
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes, load_weights
from nanoserve.model import LlamaModel


def build_model(weights_dir: str | None) -> LlamaModel:
    """The real checkpoint if one is pointed at, otherwise random weights.

    Random weights are honest here in a way they would not be for a correctness
    test: a decode step reads the same bytes and does the same matmuls whatever the
    numbers in them are, and the rows stop on a step budget rather than on what the
    model chose to emit. What is being timed is the shape of the work.
    """
    if weights_dir:
        return LlamaModel(ModelConfig.from_json(weights_dir), load_weights(weights_dir))
    config = ModelConfig()
    torch.manual_seed(0)
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(config).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(config, Weights(tensors, config))


def run_sweep(model, sizes, prompt_len, new_tokens, block_size):
    """Time the uniform batch at each size and print the scaling as it goes."""
    prompt = list(range(1, prompt_len + 1))
    print(
        f"uniform batch sweep (prompt {prompt_len} tokens, {new_tokens} new, CPU, fp32):",
        file=sys.stderr,
    )
    scaling, timings = sweep_batch_sizes(
        model, prompt, sizes, max_new_tokens=new_tokens, block_size=block_size
    )
    rows = []
    for timing in timings:
        size = timing.batch_size
        rows.append(
            {
                "batch_size": size,
                "step_ms": round(timing.median_step_s * 1e3, 3),
                "goodput_tps": round(timing.goodput_tps, 2),
                "speedup": round(scaling.speedup(size), 3),
                "efficiency": round(scaling.efficiency(size), 3),
            }
        )
        print(
            f"  batch {size:<4} step {timing.median_step_s * 1e3:8.2f}ms  "
            f"{timing.goodput_tps:8.1f} tok/s  "
            f"{scaling.speedup(size):5.2f}x  eff {scaling.efficiency(size):.2f}",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"\n  best {scaling.best_tps:.1f} tok/s at batch {scaling.best_size}; "
        f"knee at batch {scaling.knee(0.8)} (efficiency >= 0.80)",
        file=sys.stderr,
        flush=True,
    )
    return rows


def run_blocking(model, batch_size, prompt_len, long_tokens, short_tokens, block_size):
    """One long row, the rest short: the head-of-line number in a single batch.

    The short rows finish at `short_tokens` and then sit in the batch until the long
    row is done, still forwarded, still cached, still charged, and still unanswered.
    """
    prompt = list(range(1, prompt_len + 1))
    prompts = [list(prompt) for _ in range(batch_size)]
    # Row 0 is the straggler; every other row stops early. Budgets are decode steps,
    # so a row generating N tokens spends N-1 of them (the prefill emits the first).
    stop_steps = [long_tokens - 1] + [short_tokens - 1] * (batch_size - 1)
    timing = time_model_batch(
        model,
        prompts,
        max_new_tokens=long_tokens,
        stop_steps=stop_steps,
        block_size=block_size,
    )
    print(
        f"\nhead-of-line run ({batch_size - 1} rows of {short_tokens} tokens behind "
        f"1 row of {long_tokens}):",
        file=sys.stderr,
    )
    print(
        f"  issued {timing.issued_tokens} tokens, collected {timing.useful_tokens}: "
        f"{timing.waste_fraction:.1%} of the decode was computed for finished rows",
        file=sys.stderr,
    )
    print(
        f"  goodput {timing.goodput_tps:.1f} tok/s vs an issued {timing.issued_tps:.1f} tok/s "
        f"(the number a uniform benchmark would have quoted)",
        file=sys.stderr,
    )
    print(
        f"  a short row finished at {timing.row_finish_s(1):.2f}s and was returned at "
        f"{timing.total_s:.2f}s: {timing.max_hol_inflation:.1f}x its own latency, "
        f"{timing.max_hol_delay_s:.2f}s of it dead",
        file=sys.stderr,
        flush=True,
    )
    return {
        "batch_size": batch_size,
        "long_tokens": long_tokens,
        "short_tokens": short_tokens,
        "issued_tokens": timing.issued_tokens,
        "useful_tokens": timing.useful_tokens,
        "waste_fraction": round(timing.waste_fraction, 4),
        "goodput_tps": round(timing.goodput_tps, 2),
        "issued_tps": round(timing.issued_tps, 2),
        "max_hol_delay_s": round(timing.max_hol_delay_s, 4),
        "max_hol_inflation": round(timing.max_hol_inflation, 3),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve static-batching benchmark")
    p.add_argument("--sizes", type=_int_list, default=[1, 2, 4, 8])
    p.add_argument("--prompt-len", type=int, default=32)
    p.add_argument("--new-tokens", type=int, default=8, help="tokens per row in the sweep")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--long-tokens", type=int, default=32, help="the straggler's output length")
    p.add_argument("--short-tokens", type=int, default=4, help="every other row's length")
    p.add_argument("--weights", default=None, help="checkpoint dir; random weights if omitted")
    p.add_argument("--csv", default="docs/daily/data/day-29-batchbench.csv")
    args = p.parse_args()

    model = build_model(args.weights)
    rows = run_sweep(model, args.sizes, args.prompt_len, args.new_tokens, args.block_size)
    blocking = run_blocking(
        model,
        max(args.sizes),
        args.prompt_len,
        args.long_tokens,
        args.short_tokens,
        args.block_size,
    )

    out = Path(args.csv)
    write_csv(out, rows, ["batch_size", "step_ms", "goodput_tps", "speedup", "efficiency"])
    hol = out.with_name(out.stem + "-hol" + out.suffix)
    write_csv(hol, [blocking], list(blocking))
    print(f"\nwrote {len(rows)} rows to {out} and the blocking run to {hol}", file=sys.stderr)


if __name__ == "__main__":
    main()
