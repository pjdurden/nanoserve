"""Day 13: measure the KV cache as a curve, not a single number.

Run from the repo root with the venv python so torch + the weights are found:

    cd ~/nanoserve && .venv/bin/python bench.py
    cd ~/nanoserve && .venv/bin/python bench.py --gen-lens 8,16,32,64 \
        --naive-max-gen 32 --csv docs/daily/data/day-13-bench.csv

Day 11 reported one point: 40 tokens, 87s naive versus 15s cached, a 5.74x. This
sweeps lengths so the *shape* shows: cached decode is O(n), so its tokens/sec
stays roughly flat as the sequence grows; naive recompute is O(n^2), so its
tokens/sec falls off as length climbs, and the speedup widens. Both paths run pure
greedy so they do identical work and only the cache differs.

Two sweeps:
  1. generation-length sweep at a fixed prompt: naive vs cached vs speedup. Naive
     is capped (``--naive-max-gen``) because its O(n^2) cost makes the long points
     genuinely slow on CPU; cached runs the full range.
  2. prompt-length (prefill) sweep at a fixed short generation, cached only: shows
     TTFT climbing with prompt length while steady-state decode stays flat, which
     is the Week-4 motivation (this one contiguous buffer must hold the whole
     prompt for every sequence).

The timing math lives in ``nanoserve.benchmark`` and is unit-tested with a fake
clock; this file only wires the real model into it and prints/saves the results.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.benchmark import RunTiming, measure_call, measure_stream, speedup
from nanoserve.config import ModelConfig
from nanoserve.loader import load_weights
from nanoserve.model import LlamaModel


def build_prompt(length: int, token_id: int) -> torch.Tensor:
    """A synthetic [1, length] prompt of one repeated id.

    Timing depends on sequence length, not on which tokens are present, so a
    repeated valid id is the cheapest way to dial prompt length exactly.
    """
    return torch.full((1, length), token_id, dtype=torch.long)


def run_cached(model: LlamaModel, ids: torch.Tensor, gen: int) -> RunTiming:
    """Cached greedy decode, timed per token (TTFT + inter-token latencies)."""
    stream = model.generate_stream(ids, max_new_tokens=gen, temperature=0.0, eos_id=None)
    return measure_stream(stream)


def run_naive(model: LlamaModel, ids: torch.Tensor, gen: int) -> float:
    """Naive (no-cache) greedy decode, timed end to end (no per-token view)."""
    return measure_call(lambda: model.greedy_generate(ids, max_new_tokens=gen, eos_id=None))


def gen_length_sweep(model, prompt_len, gen_lens, naive_max_gen, token_id):
    ids = build_prompt(prompt_len, token_id)
    rows = []
    for gen in gen_lens:
        cached = run_cached(model, ids, gen)
        naive_s = run_naive(model, ids, gen) if gen <= naive_max_gen else None
        row = {
            "sweep": "gen_length",
            "prompt_len": prompt_len,
            "gen_len": gen,
            "cached_total_s": round(cached.total_s, 4),
            "cached_ttft_s": round(cached.ttft_s, 4),
            "cached_median_itl_s": round(cached.median_itl_s, 5),
            "cached_tok_per_s": round(cached.tokens_per_s, 3),
            "naive_total_s": round(naive_s, 4) if naive_s is not None else "",
            "naive_tok_per_s": round(gen / naive_s, 3) if naive_s else "",
            "speedup": round(speedup(naive_s=naive_s, cached_s=cached.total_s), 2)
            if naive_s is not None
            else "",
        }
        rows.append(row)
        tag = f"{row['speedup']}x" if naive_s is not None else "naive skipped"
        print(
            f"  gen={gen:<4} cached {cached.total_s:6.2f}s "
            f"({cached.tokens_per_s:6.2f} tok/s)  "
            f"naive {naive_s:7.2f}s  -> {tag}"
            if naive_s is not None
            else f"  gen={gen:<4} cached {cached.total_s:6.2f}s "
            f"({cached.tokens_per_s:6.2f} tok/s)  naive skipped (>{naive_max_gen})",
            file=sys.stderr,
            flush=True,
        )
    return rows


def prompt_length_sweep(model, prompt_lens, gen, token_id):
    rows = []
    for plen in prompt_lens:
        ids = build_prompt(plen, token_id)
        cached = run_cached(model, ids, gen)
        rows.append(
            {
                "sweep": "prompt_length",
                "prompt_len": plen,
                "gen_len": gen,
                "cached_total_s": round(cached.total_s, 4),
                "cached_ttft_s": round(cached.ttft_s, 4),
                "cached_median_itl_s": round(cached.median_itl_s, 5),
                "cached_tok_per_s": round(cached.tokens_per_s, 3),
                "naive_total_s": "",
                "naive_tok_per_s": "",
                "speedup": "",
            }
        )
        print(
            f"  prompt={plen:<5} ttft {cached.ttft_s:6.2f}s  "
            f"median ITL {cached.median_itl_s * 1000:6.1f}ms  "
            f"decode {cached.decode_tps:6.2f} tok/s",
            file=sys.stderr,
            flush=True,
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sweep",
        "prompt_len",
        "gen_len",
        "cached_total_s",
        "cached_ttft_s",
        "cached_median_itl_s",
        "cached_tok_per_s",
        "naive_total_s",
        "naive_tok_per_s",
        "speedup",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve KV-cache benchmark sweep")
    p.add_argument("--weights", default="weights", help="path to the weights dir")
    p.add_argument("--prompt-len", type=int, default=16, help="prompt len for the gen sweep")
    p.add_argument("--gen-lens", type=_int_list, default=[8, 16, 32, 64])
    p.add_argument("--naive-max-gen", type=int, default=32, help="skip naive above this gen len")
    p.add_argument("--prompt-lens", type=_int_list, default=[16, 64, 128, 256])
    p.add_argument("--prefill-gen", type=int, default=16, help="gen len for the prompt sweep")
    p.add_argument("--token-id", type=int, default=1, help="id used to fill synthetic prompts")
    p.add_argument("--csv", default="docs/daily/data/day-13-bench.csv")
    args = p.parse_args()

    print("loading nanoserve (Llama-3.2-1B)...", file=sys.stderr, flush=True)
    model = LlamaModel(ModelConfig.from_json(args.weights), load_weights(args.weights))

    print(f"\ngeneration-length sweep (prompt_len={args.prompt_len}):", file=sys.stderr)
    rows = gen_length_sweep(
        model, args.prompt_len, args.gen_lens, args.naive_max_gen, args.token_id
    )

    print(f"\nprompt-length sweep (gen_len={args.prefill_gen}):", file=sys.stderr)
    rows += prompt_length_sweep(model, args.prompt_lens, args.prefill_gen, args.token_id)

    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
