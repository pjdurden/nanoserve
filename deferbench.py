"""Day 48: the deferral window, in journeys saved and rows thrown away.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python deferbench.py

    cd ~/nanoserve && .venv/bin/python deferbench.py --windows 0,1,2,4,8 \
        --csv docs/daily/data/day-48-deferbench.csv

Day 47 left the decode loop with exactly one synchronisation in it, in `collect`,
because `Request.append_token` applies the stop rules and a stop rule needs a
Python int. Day 48 holds the tokens on the device instead and lets several steps
go home together. This script measures what that is worth and what it costs, on
the same engine, in one process, at several window sizes.

Three columns and they are three different things:

  **transfers/step.** Counted live off `Engine.output`, not derived. A window of k
  should read back once every k steps in the steady state and more often than that
  in a run where rows come and go, because a held tensor whose rows are not this
  step's rows cannot be this step's input and the engine pays for the ints early.

  **build_inputs.** The saving a CPU can actually see. With a window the decode
  input is `held.unsqueeze(1)`, a view of the tensor the sampler left, instead of
  `torch.tensor([[r.output_token_ids[-1]] for r in requests])` built out of Python
  lists every step.

  **waste.** Rows the forwards computed that no request kept. Continuous batching
  drove this to zero on Day 29 and deferral puts some of it back: a request whose
  stop token is still on the device is still in the batch. It is the price of the
  window and it is linear in the window, while the saving saturates.

**A CPU cannot price the synchronisations and this script says so instead of
guessing.** `.tolist()` here is a memcpy out of the same RAM the interpreter runs
in. `--transfer-us` supplies a per-journey latency from a machine that has a bus,
and only then is the arithmetic table printed.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

from nanoserve.config import ModelConfig
from nanoserve.deferred import best_window, check_tokens_conserved, render
from nanoserve.engine import Engine
from nanoserve.launch import place_weights
from nanoserve.loader import load_weights
from nanoserve.model import LlamaModel
from nanoserve.profiler import StepRecorder, device_timer_for
from nanoserve.scheduler import Request

DEFAULT_PROMPT = "The capital of France is"


def load_engine_model(weights: Path, device: str, dtype: str):
    """This engine's own model on the real weights, exactly as profbench loads it."""
    import torch

    from transformers import AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(weights)
    config = ModelConfig.from_json(weights)
    loaded = place_weights(
        load_weights(weights, config, dtype=None), torch.device(device), torch_dtype
    )
    return tokenizer, LlamaModel(config, loaded)


def run_window(model, prompts, *, window: int, slots: int, max_tokens: int, device: str,
               warmup: int, num_blocks: int, block_size: int) -> dict:
    """One burst through an engine deferring `window` steps. 0 is Day 47's loop.

    Every request is handed over before the first step, so the queue is full from
    the start and the batch is as close to `slots` as the pool allows: a steady
    loop rather than a ramp, which is the shape the transfer rate is a statement
    about.
    """
    engine = Engine.build(
        model,
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch_size=slots,
        defer_window=window,
    )
    engine.recorder = StepRecorder(device_timer=device_timer_for(device))
    for i, prompt in enumerate(prompts):
        engine.add_request(
            Request(
                request_id=f"r{i}", prompt_token_ids=list(prompt), max_new_tokens=max_tokens
            )
        )
    finished = engine.run_to_completion()
    profile = engine.recorder.profile(f"window {window}", warmup=warmup).select("decode")
    totals = profile.phase_totals()

    def phase_ms(name: str) -> float:
        total = totals.get(name)
        return total.host_s / total.steps * 1e3 if total and total.steps else 0.0

    row = {
        "window": window,
        "slots": slots,
        "steps": engine.iterations,
        "transfers_per_step": round(engine.output.transfers_per_step, 4),
        "build_inputs_ms": round(phase_ms("build_inputs"), 4),
        "collect_ms": round(phase_ms("collect"), 4),
        "settle_ms": round(phase_ms("settle"), 4),
        "step_ms": round(statistics.median(s.host_s for s in profile.samples) * 1e3, 3),
        "issued": engine.issued_tokens,
        "collected": engine.collected_tokens,
        "overshoot": getattr(engine.output, "overshoot_tokens", 0),
        "abandoned": getattr(engine.output, "abandoned_tokens", 0),
        "waste_pct": round(engine.waste_fraction * 100, 3),
        "tokens": sum(r.num_output_tokens for r in finished),
    }
    if window:
        # Every deferred token is applied, overshot, abandoned or still held, and a
        # bench that reports a rate off a processor that lost a batch is reporting
        # a rate about nothing.
        check_tokens_conserved(engine.output)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("./weights"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--windows", default="0,1,2,4,8")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--transfer-us",
        type=float,
        default=None,
        help="seconds per readback, from a machine with a bus to cross",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.weights.exists():
        print(f"no weights at {args.weights}", file=sys.stderr)
        return 2
    windows = [int(w) for w in args.windows.split(",")]

    tokenizer, model = load_engine_model(args.weights, args.device, args.dtype)
    prompt_ids = tokenizer(args.prompt)["input_ids"]
    prompts = [list(prompt_ids) for _ in range(args.requests)]

    rows = []
    baseline_tokens = None
    print(
        f"{'window':>7}{'transfers/step':>16}{'build_inputs':>14}{'collect':>10}"
        f"{'settle':>9}{'step':>10}{'overshoot':>11}{'waste':>8}"
    )
    for window in windows:
        row = run_window(
            model,
            prompts,
            window=window,
            slots=args.slots,
            max_tokens=args.max_tokens,
            device=args.device,
            warmup=args.warmup,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
        )
        # The property the whole day rests on: deferral changes when a token is
        # looked at, not what was drawn. A run whose token count moved is a bug,
        # not a benchmark.
        if baseline_tokens is None:
            baseline_tokens = row["tokens"]
        elif row["tokens"] != baseline_tokens:
            print(
                f"\nrefusing the sweep: window {window} kept {row['tokens']} tokens "
                f"and window {windows[0]} kept {baseline_tokens}",
                file=sys.stderr,
            )
            return 1
        rows.append(row)
        print(
            f"{row['window']:>7}{row['transfers_per_step']:>16.3f}"
            f"{row['build_inputs_ms']:>12.4f}ms{row['collect_ms']:>8.4f}ms"
            f"{row['settle_ms']:>7.4f}ms{row['step_ms']:>8.2f}ms"
            f"{row['overshoot']:>11}{row['waste_pct']:>7.2f}%"
        )

    if args.transfer_us is not None:
        latency = args.transfer_us * 1e-6
        row_s = rows[0]["step_ms"] * 1e-3 / max(args.slots, 1)
        print()
        print(
            render(
                latency_s=latency,
                row_s=row_s,
                output_tokens=args.max_tokens,
                title=(
                    f"a {args.transfer_us:.1f}us journey against a "
                    f"{row_s * 1e3:.2f}ms row, over {args.max_tokens} tokens"
                ),
            )
        )
        print(
            "\nbest window here: "
            f"{best_window(latency_s=latency, row_s=row_s, output_tokens=args.max_tokens)}"
        )
    else:
        print(
            "\nrefusing to price the journeys: on this box a `.tolist()` is a memcpy "
            "out of the same RAM the interpreter runs in, so there is nothing to "
            "price. Pass --transfer-us from a machine with a bus to cross."
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
