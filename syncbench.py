"""Day 47: the sample phase, before and after the tokens stopped coming home.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python syncbench.py

    cd ~/nanoserve && .venv/bin/python syncbench.py --slots 1,2,4,8 \
        --csv docs/daily/data/day-47-syncbench.csv

Day 46 profiled a decode step and found `sample` holding 86% to 89% of the engine's
whole Python loop at every batch size. Not because sampling is arithmetic: because
`sample_batch` returned `list[int]`, so every row's token was its own
`int(tensor)`. Day 47 changed the return type to a `[rows]` device tensor and moved
the single remaining readback into `collect`.

This script measures the two things that changed, separately, because they are not
the same thing and a benchmark that adds them together is lying:

  **The count.** How many times a step brings a tensor home. `nanoserve.output`
  derives it (`syncs_per_step`) and `Engine.output` counts it live, and the two
  have to agree. That number is what matters on a GPU, where a readback is a
  synchronisation and the host stops running ahead.

  **The Python.** How long the sample phase takes with the per-row loop and with
  the batched one, on the same real `[slots, 128256]` logits tensor. That number is
  what matters here, on a box with no card, and it is the honest thing this
  hardware can measure.

**A CPU cannot price the first one, and this script says so instead of guessing.**
`.tolist()` here is a memcpy out of the same RAM the interpreter runs in: no bus,
no queue to drain, no host that was ever running ahead. `--transfer-us` supplies a
per-synchronisation latency from a machine that has one, and only then is the
seconds column printed. Without it the sync column is a count and stays a count.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.launch import place_weights
from nanoserve.loader import load_weights
from nanoserve.model import LlamaModel
from nanoserve.output import (
    OutputUnsound,
    check_measurable,
    check_single_transfer,
    render,
    saving_per_step,
    sample_ceiling,
    strategy_speedup,
    syncs_per_step,
)
from nanoserve.profiler import StepRecorder, device_timer_for
from nanoserve.sampling import GREEDY, BatchedSampler, SamplingParams, top_k_filter, top_p_filter
from nanoserve.scheduler import Request

DEFAULT_PROMPT = "The capital of France is"


def load_engine_model(weights: Path, device: str, dtype: str):
    """This engine's own model on the real weights, and its tokenizer."""
    from transformers import AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(weights)
    config = ModelConfig.from_json(weights)
    loaded = place_weights(
        load_weights(weights, config, dtype=None), torch.device(device), torch_dtype
    )
    return tokenizer, LlamaModel(config, loaded)


def rowwise_sample(sampler: BatchedSampler, logits: torch.Tensor, rows) -> list[int]:
    """Day 40's `sample_batch`, kept here and only here, as the baseline.

    Not left in `sampling.py` as a dead second path: two implementations of the
    same draw is exactly how the fast one quietly stops matching the slow one. It
    lives in the benchmark that needs a before, and it is a copy of the body Day 47
    replaced, so a difference in the numbers below is a difference in this loop and
    nothing else.
    """
    tokens = [0] * len(rows)
    greedy = [i for i, (_, p) in enumerate(rows) if p.is_greedy]
    if greedy:
        for i, token in zip(greedy, logits[greedy].argmax(dim=-1).tolist()):
            tokens[i] = int(token)
    groups: dict[tuple[int, float], list[int]] = {}
    for i, (_, params) in enumerate(rows):
        if not params.is_greedy:
            groups.setdefault(params.filter_key, []).append(i)
    for (top_k, top_p), members in groups.items():
        block = logits[members]
        temperatures = torch.tensor(
            [rows[i][1].temperature for i in members], dtype=block.dtype, device=block.device
        ).unsqueeze(1)
        block = top_p_filter(top_k_filter(block / temperatures, top_k), top_p)
        probs = block.softmax(dim=-1)
        for offset, i in enumerate(members):
            tokens[i] = int(
                torch.multinomial(
                    probs[offset], num_samples=1, generator=sampler._generator_for(*rows[i])
                )
            )
    return tokens


def median_ms(fn, repeats: int) -> float:
    """Median of `repeats` calls, in milliseconds. The median for the usual reason.

    One descheduled iteration is a large positive outlier and there is no matching
    negative one, so a mean over a hundred calls is partly a measurement of the
    operating system.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2] * 1e3


def measure_phase(model, prompt, *, slots: int, sampling, repeats: int) -> dict:
    """Both sample paths over one real `[slots, vocab]` logits tensor.

    The logits are the model's own, not `randn`. Top-p sorts the vocabulary and
    top-k partitions it, and both are sensitive to how the mass is actually spread:
    a real distribution has a short head and a long flat tail, and random logits do
    not, so timing the filters on noise times a different problem.
    """
    engine = Engine.build(model, num_blocks=512, block_size=16, max_batch_size=slots)
    requests = [
        Request(
            request_id=f"r{i}",
            prompt_token_ids=list(prompt),
            max_new_tokens=4,
            sampling=sampling,
        )
        for i in range(slots)
    ]
    for request in requests:
        engine.add_request(request)
    engine.step()  # one prefill, so the rows exist and hold real K/V
    rows = [(r.request_id, r.sampling) for r in requests]
    ids = torch.tensor([[r.output_token_ids[-1]] for r in requests], dtype=torch.long)
    positions = torch.tensor(
        [[engine.cache.tables[r.slot].num_tokens] for r in requests], dtype=torch.long
    )
    logits = engine.model.forward(
        ids, positions, cache=engine.cache.view([r.slot for r in requests])
    )[:, -1]

    sampler = BatchedSampler(seed=0)
    before = median_ms(lambda: rowwise_sample(sampler, logits, rows), repeats)
    after = median_ms(lambda: sampler.sample_batch_device(logits, rows), repeats)
    listed = median_ms(lambda: sampler.sample_batch(logits, rows), repeats)
    return {"vocab": logits.shape[-1], "rowwise_ms": before, "device_ms": after, "list_ms": listed}


def measure_loop(model, prompts, *, slots: int, max_tokens: int, device: str, warmup: int) -> dict:
    """One instrumented burst, for the phase split and the live transfer count."""
    engine = Engine.build(model, num_blocks=512, block_size=16, max_batch_size=slots)
    engine.recorder = StepRecorder(device_timer=device_timer_for(device))
    for i, prompt in enumerate(prompts):
        engine.add_request(
            Request(request_id=f"r{i}", prompt_token_ids=list(prompt), max_new_tokens=max_tokens)
        )
    engine.run_to_completion()
    check_single_transfer(engine.output)
    profile = engine.recorder.profile(f"{slots} slots", warmup=warmup).select("decode")
    totals = profile.phase_totals()
    return {
        "profile": profile,
        "step_ms": profile.mean_step_s * 1e3,
        "sample_ms": totals["sample"].host_s / profile.steps * 1e3,
        "collect_ms": totals["collect"].host_s / profile.steps * 1e3,
        "transfers_per_step": engine.output.transfers_per_step,
        "ceiling": sample_ceiling(profile) if profile.total_device_s > 0.0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("./weights"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--slots", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument(
        "--transfer-us",
        type=float,
        default=None,
        help="per-synchronisation latency, in microseconds, measured on a machine that "
        "has a device. Without it the sync column stays a count",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    sizes = [int(s) for s in args.slots.split(",") if s]
    tokenizer, model = load_engine_model(args.weights, args.device, args.dtype)
    prompt = list(tokenizer(args.prompt, return_tensors=None)["input_ids"])
    prompts = [prompt for _ in range(args.requests)]
    sampled = SamplingParams(temperature=0.8, top_p=0.9)

    try:
        check_measurable(args.device)
        priced = True
    except OutputUnsound as exc:
        priced = False
        print(f"note: {exc}\n")

    print(
        f"{args.requests} requests x {args.max_tokens} tokens, {len(prompt)}-token prompt, "
        f"{args.device}/{args.dtype}, {args.repeats} repeats, median"
    )
    print(
        "\nthe sample phase, one real logits tensor, per-row readback vs one device tensor:"
    )
    print(
        f"  {'slots':>5}{'params':>9}{'rowwise':>11}{'device':>11}{'+tolist':>11}"
        f"{'faster':>9}{'syncs before':>14}{'after':>7}"
    )

    rows = []
    for slots in sizes:
        loop = measure_loop(
            model,
            prompts,
            slots=slots,
            max_tokens=args.max_tokens,
            device=args.device,
            warmup=args.warmup,
        )
        for label, params, num_sampled in (("greedy", GREEDY, 0), ("top-p", sampled, slots)):
            phase = measure_phase(
                model, prompt, slots=slots, sampling=params, repeats=args.repeats
            )
            before = syncs_per_step(slots, num_sampled=num_sampled, strategy="per_row")
            after = syncs_per_step(slots, num_sampled=num_sampled, strategy="one_transfer")
            print(
                f"  {slots:>5}{label:>9}{phase['rowwise_ms']:>10.3f}ms"
                f"{phase['device_ms']:>10.3f}ms{phase['list_ms']:>10.3f}ms"
                f"{phase['rowwise_ms'] / phase['device_ms']:>8.2f}x"
                f"{before:>14.0f}{after:>7.0f}"
            )
            rows.append(
                {
                    "slots": slots,
                    "params": label,
                    "vocab": phase["vocab"],
                    "rowwise_ms": round(phase["rowwise_ms"], 4),
                    "device_ms": round(phase["device_ms"], 4),
                    "list_ms": round(phase["list_ms"], 4),
                    "phase_speedup": round(phase["rowwise_ms"] / phase["device_ms"], 4),
                    "syncs_before": before,
                    "syncs_after": after,
                    "step_ms": round(loop["step_ms"], 4),
                    "sample_ms": round(loop["sample_ms"], 4),
                    "collect_ms": round(loop["collect_ms"], 4),
                    "transfers_per_step": round(loop["transfers_per_step"], 4),
                    "sample_ceiling": (
                        None if loop["ceiling"] is None else round(loop["ceiling"], 4)
                    ),
                }
            )

    print("\nthe decode loop, as the engine actually runs it (greedy):")
    print(f"  {'slots':>5}{'step':>11}{'sample':>11}{'collect':>11}{'transfers/step':>16}")
    for slots in sizes:
        row = next(r for r in rows if r["slots"] == slots and r["params"] == "greedy")
        print(
            f"  {slots:>5}{row['step_ms']:>10.3f}ms{row['sample_ms']:>10.3f}ms"
            f"{row['collect_ms']:>10.3f}ms{row['transfers_per_step']:>16.2f}"
        )

    if priced and args.transfer_us is not None:
        latency = args.transfer_us / 1e6
        top = sizes[-1]
        step_s = rows[-1]["step_ms"] / 1e3
        print()
        print(
            render(
                top,
                num_sampled=top,
                latency_s=latency,
                window=8,
                step_s=step_s,
                title=f"{top} sampled rows at {args.transfer_us:.1f}us per synchronisation",
            )
        )
        saved = saving_per_step(top, num_sampled=top, latency_s=latency)
        print(
            f"  batching the readback saves {saved * 1e6:.1f}us of a "
            f"{step_s * 1e3:.3f}ms step: {strategy_speedup(step_s, saved):.4f}x"
        )
    elif args.transfer_us is not None:
        print(
            "\nrefusing to price the synchronisations: pass --device cuda, or take the "
            "latency on a machine that has a bus to cross"
        )

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
