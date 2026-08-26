"""Day 44: this engine against HuggingFace `generate`, on one box, same tokens.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python hfbench.py --weights ./weights --device cpu \
        --requests 8 --max-tokens 16 --batch-sizes 1,2,4,8

    cd ~/nanoserve && .venv/bin/python hfbench.py --weights ./weights --device cpu \
        --requests 8 --max-tokens 16 --batch-sizes 1,2,4,8 \
        --csv docs/daily/data/day-44-hfbench.csv

The baseline is `transformers`' own `model.generate`, greedy, KV cache on, one
request at a time, because that is the only way a `generate` call can serve a
request that arrived while another one was running. The engine gets the same
prompts, the same budget and the same greedy sampler, all submitted at once.

Then the run refuses to report anything unless the two produced *the same tokens*,
which is the only version of this comparison worth publishing. `nanoserve.baseline`
holds the gate and the arithmetic and is unit-tested against hand-placed
timestamps; this file only builds prompts, drives the two systems and prints.

The table's point is the two columns after the speedup. A throughput ratio on its
own is a number to put in a README; `batch x per-served rate` is the same number
saying which half of the engine earned it, and on any box the second column is
below 1.0. Sweeping the slot count shows the trade directly: the batch climbs
towards the slot count while what one row produces falls, and the product stops
growing at the point where this hardware ran out of whatever batching was
exploiting.

Both factorisations are printed because they are not the same claim. `batch` is
slot-seconds and is a statement about the hardware; `occ_sys` is the mean number of
requests in the system and is inflated by the queue, which an offline run has a lot
of, because every prompt is handed over before the first step. At one slot they
read 1.00 and 4.83 for the same run.

`--sanity` is on by default and stops the run if the baseline ever overlapped two
requests or the engine's mean batch never got above 1.5 rows. Either one means the
numbers describe a different experiment than the one the flags asked for.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from nanoserve.baseline import (
    RunUnsound,
    UnfairComparison,
    check_concurrent,
    check_serial,
    compare,
    hf_generate_one,
    run_concurrent,
    run_serial,
)
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.launch import place_weights
from nanoserve.loader import load_weights
from nanoserve.model import LlamaModel

DEFAULT_PROMPT = "The capital of France is"


def build_prompts(tokenizer, text: str, n: int) -> list[list[int]]:
    """n identical prompts.

    Identical for the same reason Day 42's plans are: the sweep moves the batch
    size and nothing else should move with it. Ragged prompts are the more
    realistic workload and they put a second variable into a two-column table.
    """
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    return [list(ids) for _ in range(n)]


def load_pair(weights: Path, device: str, dtype: str):
    """The same weights, twice: once as `transformers`, once as this engine.

    Loaded separately and both resident, which is the whole memory cost of this
    script and the reason it is a CPU experiment at 1B. Loading them in turn and
    timing each would be cheaper and would put a page-cache difference inside the
    measurement.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(weights)
    hf = AutoModelForCausalLM.from_pretrained(weights, torch_dtype=torch_dtype)
    hf = hf.to(device).eval()

    config = ModelConfig.from_json(weights)
    loaded = place_weights(
        load_weights(weights, config, dtype=None), torch.device(device), torch_dtype
    )
    return tokenizer, hf, LlamaModel(config, loaded), config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("./weights"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--no-sanity", action="store_true")
    args = parser.parse_args(argv)

    sizes = [int(s) for s in args.batch_sizes.split(",") if s]
    tokenizer, hf, model, _ = load_pair(args.weights, args.device, args.dtype)
    prompts = build_prompts(tokenizer, args.prompt, args.requests)
    eos_id = tokenizer.eos_token_id

    print(
        f"{args.requests} requests x {args.max_tokens} tokens, "
        f"{len(prompts[0])}-token prompt, {args.device}/{args.dtype}"
    )

    started = time.perf_counter()
    baseline = run_serial(
        prompts,
        hf_generate_one(hf, max_new_tokens=args.max_tokens, eos_id=eos_id, device=args.device),
        name="hf-generate",
    )
    print(
        f"  hf-generate  {baseline.throughput_tps:6.2f} tok/s   "
        f"batch {baseline.batch_occupancy:4.2f}   "
        f"prefill {baseline.mean_prefill_s:5.2f}s   "
        f"decode {baseline.mean_decode_tps:5.2f} tok/s/req   "
        f"[{time.perf_counter() - started:.1f}s]"
    )
    if not args.no_sanity:
        check_serial(baseline)

    header = (
        f"\n{'slots':>5} {'tok/s':>8} {'speedup':>8} {'batch':>7} {'per_srv':>8} "
        f"{'occ_sys':>8} {'per_req':>8} {'queue':>6} {'ttft':>7} {'prefill':>8} "
        f"{'latency':>8} {'decode':>7}"
    )
    print(header)
    rows = []
    for size in sizes:
        engine = Engine.build(
            model,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            max_batch_size=size,
        )
        run = run_concurrent(engine, prompts, max_new_tokens=args.max_tokens, eos_id=eos_id)
        try:
            result = compare(baseline, run)
            result.check()
        except UnfairComparison as exc:
            print(f"\nthe comparison at {size} slots is not one: {exc}", file=sys.stderr)
            return 2
        if not args.no_sanity and size > 1:
            # 1.5 rather than the slot count: a run ramps up and drains, so the
            # mean batch is always a little under the slots it was given, and the
            # question here is only whether this run batched at all.
            try:
                check_concurrent(run, min_occupancy=1.5)
            except RunUnsound as exc:
                print(f"\n{exc}", file=sys.stderr)
                return 2
        print(
            f"{size:>5} {run.throughput_tps:>8.2f} {result.throughput_speedup:>7.2f}x "
            f"{result.batch_gain:>6.2f}x {result.per_served_ratio:>7.2f}x "
            f"{result.occupancy_gain:>7.2f}x {result.per_request_ratio:>7.2f}x "
            f"{run.queue_share:>5.0%} {run.mean_ttft_s:>6.2f}s {run.mean_prefill_s:>7.2f}s "
            f"{run.mean_latency_s:>7.2f}s {run.mean_decode_tps:>6.2f}"
        )
        rows.append(
            {
                "slots": size,
                "requests": args.requests,
                "max_tokens": args.max_tokens,
                "hf_tps": round(baseline.throughput_tps, 3),
                "engine_tps": round(run.throughput_tps, 3),
                "speedup": round(result.throughput_speedup, 4),
                "batch_gain": round(result.batch_gain, 4),
                "per_served_ratio": round(result.per_served_ratio, 4),
                "occupancy_gain": round(result.occupancy_gain, 4),
                "per_request_ratio": round(result.per_request_ratio, 4),
                "engine_batch": round(run.batch_occupancy, 4),
                "engine_occupancy": round(run.occupancy, 4),
                "engine_queue_share": round(run.queue_share, 4),
                "hf_prefill_s": round(baseline.mean_prefill_s, 4),
                "engine_prefill_s": round(run.mean_prefill_s, 4),
                "prefill_ratio": round(result.prefill_ratio, 4),
                "engine_ttft_s": round(run.mean_ttft_s, 4),
                "hf_latency_s": round(baseline.mean_latency_s, 4),
                "engine_latency_s": round(run.mean_latency_s, 4),
                "hf_decode_tps": round(baseline.mean_decode_tps, 4),
                "engine_decode_tps": round(run.mean_decode_tps, 4),
            }
        )
        for line in result.summary_lines():
            print(f"      {line}")

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
