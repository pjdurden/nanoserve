"""Day 29: measure static batching. What another row buys, and what it costs.

Day 27 batched the prefill and Day 28 batched the decode, so `greedy_generate_batch`
now runs N sequences end to end over one shared block pool. That closes Week 7's
code and opens its only remaining question, which is the one the daily log has been
promising since the module docstring of `batch.py`: how much does batching actually
buy, and where does it stop paying? Week 8's continuous batching is justified
entirely by the second half of that answer, so the answer has to be measured rather
than asserted.

Two numbers, and they pull in opposite directions.

**Throughput.** The matmuls in a 1B decode step are memory-bound *on a card*: the
weights are read out of HBM every step whether one row or eight rows are riding
along, so the extra rows ride in arithmetic that was idle anyway and batch size is
close to free throughput until something saturates. `BatchScaling` below reports how
much of that ideal survived: `speedup(b)` against the single-row baseline, and
`efficiency(b)`, the share of perfect linear scaling still there. The `knee` is the
largest size still paying its way.

That "nearly free" is a claim about one hardware ratio, bandwidth against FLOPs, and
not a property of transformers, and this repo's own numbers are the caveat. Measured
on CPU with fp32 GEMMs, where the arithmetic dominates rather than the weight read,
the median decode step went 279ms, 474ms, 940ms, 2092ms across batches of 1, 2, 4
and 8: efficiency 1.00, 0.59, 0.30, 0.13, peaking at 1.19x on batch 4. All that
batch size buys there is the per-step overhead. Which is the argument for measuring
rather than asserting, and also why the blocking half below travels better: it is
counted in steps and rows, so it holds on any box.

**Head-of-line blocking.** A static batch is fixed at the start, so it runs until
its *slowest* row finishes and hands back every row at that moment. A row that hit
EOS at step 8 still gets a query, a slot and a block for the next 192 steps, and its
caller still waits. Those are two distinct costs and this module keeps them apart:

  - the wasted *work*, `waste_fraction`, issued tokens the forward computed that
    nobody collected. This is throughput the card spent on nothing.
  - the wasted *time*, `hol_delay_s`, how long a finished row sits in the batch
    before its answer is returned. This is latency, and it is the one a user feels.

Hence the vocabulary the whole module runs on. An **issued** token is one the
forward computed. A **useful** token is one a row collected. `issued_tps` is the
flattering number a naive batching benchmark quotes; `goodput_tps` is the honest
one. On a batch of equal-length rows they are the same number, which is exactly why
a benchmark that sweeps only uniform prompts makes static batching look finished.

The timing core is stdlib-only, model-free, and the clock is injected, the same
split Day 13 drew and Day 20 repeated: the runner at the bottom builds a real model
and a real `BatchedPagedKVCache` and hands this core two opaque callables, while the
pure tests hand it a fake clock and scripted done vectors and pin the arithmetic to
the decimal. A benchmark whose own math is unverified is just a confident guess.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

Clock = Callable[[], float]

# A prefill or a decode step, as the timing core sees it: run the forward, and
# report which rows are finished afterwards. Everything else about the engine
# (the cache, the sampler, the token ids) is the closure's business, not this
# module's, which is what keeps the arithmetic testable without torch.
DoneFn = Callable[[], Sequence[bool]]


@dataclass
class BatchTiming:
    """One static-batch run: what the clock saw, and what each row got out of it.

    batch_size:  rows in the batch, fixed for the whole run (that is what "static"
                 means; Week 8 is the week this stops being true).
    prefill_s:   seconds for the padded prefill forward, which also emits every
                 row's first token.
    step_s:      seconds per decode step, one entry per step, in step order. The
                 batch runs `len(step_s)` steps no matter how early a row finished.
    finished_at: per row, how many decode steps that row's own generation needed.
                 0 means the row was done at the prefill token; a row that never
                 finished is recorded at the step count, since that is all the run
                 observed.

    The invariant that makes the head-of-line arithmetic meaningful: no row can
    finish after the batch does, because the batch stops when the last row is done.
    A `finished_at` past `n_steps` is a bookkeeping bug in the caller, so it is
    rejected here rather than quietly producing a negative delay.
    """

    batch_size: int
    prefill_s: float
    step_s: list[float] = field(default_factory=list)
    finished_at: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.finished_at) != self.batch_size:
            raise ValueError(
                "a batch timing needs one finish per row; got "
                f"{len(self.finished_at)} for batch_size={self.batch_size}"
            )
        if any(f < 0 for f in self.finished_at):
            raise ValueError(f"a row cannot finish before it starts; got {self.finished_at}")
        if any(f > self.n_steps for f in self.finished_at):
            raise ValueError(
                "a row cannot finish after the batch does (the batch stops when the "
                f"last row is done); got finished_at={self.finished_at} over "
                f"{self.n_steps} steps"
            )

    # --- what the clock saw ---------------------------------------------------

    @property
    def n_steps(self) -> int:
        """Decode steps the batch ran, i.e. what its slowest row needed."""
        return len(self.step_s)

    @property
    def decode_s(self) -> float:
        """Seconds spent decoding, prefill excluded (the steady-state cost)."""
        return sum(self.step_s)

    @property
    def total_s(self) -> float:
        """Whole wall time: the prefill plus every decode step."""
        return self.prefill_s + self.decode_s

    @property
    def mean_step_s(self) -> float:
        """Average decode step; the number that should barely move with batch size."""
        return statistics.fmean(self.step_s) if self.step_s else 0.0

    @property
    def median_step_s(self) -> float:
        """Middle decode step; robust to the occasional slow one."""
        return statistics.median(self.step_s) if self.step_s else 0.0

    @property
    def min_step_s(self) -> float:
        """Fastest decode step: best-of, the same least-contaminated estimate the
        read benchmark quotes, since noise on a shared box only pushes a sample
        slower."""
        return min(self.step_s) if self.step_s else 0.0

    # --- issued versus useful -------------------------------------------------

    @property
    def issued_tokens(self) -> int:
        """Decode tokens the forward computed: every row, every step, no exceptions.

        A finished row is still a column in the batch's query tensor, still reads
        its blocks and still writes a new slot, because the batch was sized once
        and cannot shrink. This is the work the card did.
        """
        return self.batch_size * self.n_steps

    @property
    def useful_tokens(self) -> int:
        """Decode tokens anyone collected: the sum of the rows' own step counts."""
        return sum(self.finished_at)

    @property
    def wasted_tokens(self) -> int:
        """Issued minus useful: tokens computed for rows that were already done."""
        return self.issued_tokens - self.useful_tokens

    @property
    def waste_fraction(self) -> float:
        """Share of the decode forward spent on rows that had already finished.

        Zero on a batch of equal-length rows, which is why a uniform-prompt
        benchmark reports static batching as solved. It climbs with the spread of
        the output lengths, and it is the throughput half of what iteration-level
        scheduling recovers. The latency half is `hol_delay_s`.
        """
        return self.wasted_tokens / self.issued_tokens if self.issued_tokens else 0.0

    @property
    def goodput_tps(self) -> float:
        """Collected tokens per second of decoding: the honest throughput number.

        Zero when no decode step ran (degenerate, but better than a
        ZeroDivisionError mid-sweep), matching how the rest of the benchmarks guard
        an empty run.
        """
        return self.useful_tokens / self.decode_s if self.decode_s > 0.0 else 0.0

    @property
    def issued_tps(self) -> float:
        """Computed tokens per second: what the hardware moved, not what it served.

        Always >= `goodput_tps`, equal only when the rows are the same length. It
        is a real number about the machine and a misleading one about the service,
        so both are reported and neither is called "throughput" on its own.
        """
        return self.issued_tokens / self.decode_s if self.decode_s > 0.0 else 0.0

    # --- head-of-line blocking ------------------------------------------------

    @property
    def straggler(self) -> int:
        """Index of the row that set the batch's length (ties go to the first).

        The row nobody is waiting on, and the one everybody is waiting for.
        """
        return max(range(self.batch_size), key=lambda i: self.finished_at[i])

    def row_finish_s(self, row: int) -> float:
        """When row `row`'s own last token was computed, seconds from the start.

        The prefill plus that row's own decode steps. This is the latency the row
        *would* have had on a machine that could hand it back the moment it was
        done, which is precisely what continuous batching makes possible.
        """
        return self.prefill_s + sum(self.step_s[: self.finished_at[row]])

    def hol_delay_s(self, row: int) -> float:
        """Seconds row `row` spent finished but still inside the batch.

        Static batching returns the whole batch at once, so a row's observed
        latency is `total_s` regardless of when its own work ended. The difference
        is dead time charged to a caller who is owed nothing more. Zero for the
        straggler, by construction.
        """
        return self.total_s - self.row_finish_s(row)

    def hol_inflation(self, row: int) -> float:
        """How many times longer row `row` waited than its own work took.

        `total_s / row_finish_s(row)`: 1.0 for the straggler, 25.0 for a row that
        finished in a twenty-fifth of the batch's life. The multiplier is the
        readable form of the delay, because a 1.5s wait means something different
        on a 2s request than on a 200s one.
        """
        own = self.row_finish_s(row)
        return self.total_s / own if own > 0.0 else 1.0

    @property
    def max_hol_delay_s(self) -> float:
        """The worst wait in the batch: the first row to finish is the one punished."""
        return max(self.hol_delay_s(i) for i in range(self.batch_size))

    @property
    def mean_hol_delay_s(self) -> float:
        """Average dead time per row; the batch-level summary of the same cost."""
        return statistics.fmean(self.hol_delay_s(i) for i in range(self.batch_size))

    @property
    def max_hol_inflation(self) -> float:
        """The worst latency multiplier in the batch, the headline blocking number."""
        return max(self.hol_inflation(i) for i in range(self.batch_size))


def time_batched_decode(
    prefill_fn: DoneFn,
    step_fn: DoneFn,
    *,
    batch_size: int,
    max_steps: int,
    clock: Clock = time.perf_counter,
) -> BatchTiming:
    """Time one static-batch run: the prefill, then decode steps until all rows stop.

    prefill_fn: runs the padded prefill forward (which emits every row's first
                token) and returns the per-row done flags afterwards.
    step_fn:    runs one full-batch decode step and returns the per-row done flags.
                It is called for the *whole* batch every step, finished rows
                included, because that is what static batching does and measuring
                anything else would measure a scheduler that does not exist yet.
    max_steps:  cap on decode steps, the `max_new_tokens - 1` of the generate loop.
                The loop stops early once every row reports done.

    Each closure is timed with two clock reads, so a run of S steps reads the clock
    2 + 2*S times and the fake clock in the tests can be scripted exactly. The done
    vectors are only *read*: a row's `finished_at` is the step at which it first
    came back True, so a closure that flaps a flag back to False cannot un-finish
    a row. Rows still running at the cap are recorded at the step count, which is
    all the run observed about them.

    Raises `ValueError` for a negative cap, or for a done vector that is not one
    flag per row (which would otherwise drop a row's finish silently).
    """
    if max_steps < 0:
        raise ValueError(f"max_steps cannot be negative; got {max_steps}")

    def _check(done: Sequence[bool]) -> Sequence[bool]:
        if len(done) != batch_size:
            raise ValueError(
                f"a batch step must report one done flag per row; got {len(done)} "
                f"for batch_size={batch_size}"
            )
        return done

    t0 = clock()
    done = _check(prefill_fn())
    prefill_s = clock() - t0

    # None until the row reports done; filled in at the first step it does.
    finished: list[int | None] = [0 if d else None for d in done]
    step_s: list[float] = []
    steps = 0
    while steps < max_steps and not all(f is not None for f in finished):
        t0 = clock()
        done = _check(step_fn())
        step_s.append(clock() - t0)
        steps += 1
        for i, d in enumerate(done):
            if d and finished[i] is None:
                finished[i] = steps

    return BatchTiming(
        batch_size=batch_size,
        prefill_s=prefill_s,
        step_s=step_s,
        # A row still running at the cap gets the step count: the run never saw it
        # finish, and charging it the batch's full length is the honest reading.
        finished_at=[steps if f is None else f for f in finished],
    )


@dataclass
class BatchScaling:
    """Goodput across batch sizes, and how much of ideal linear scaling survived.

    sizes: the batch sizes swept, in the order given, including 1.
    tps:   goodput at each of those sizes, aligned by index.

    The point of the sweep is that decode is memory-bound, so the second row in a
    batch is nearly free and the eighth is nearly free too, right up until
    something (compute, cache bandwidth, the pool) saturates. `efficiency` is where
    that shows: it stays near 1.0 while rows are free and falls once they are not.
    `knee` names the last size that was still worth adding.
    """

    sizes: list[int] = field(default_factory=list)
    tps: list[float] = field(default_factory=list)

    def _at(self, size: int) -> float:
        """Goodput at `size`; `KeyError` if that size was not swept.

        Deliberately not an interpolation. The sweep measured what it measured, and
        a plausible-looking number for a size nobody ran is exactly the kind of
        thing a benchmark should refuse to invent.
        """
        try:
            return self.tps[self.sizes.index(size)]
        except ValueError:
            raise KeyError(f"batch size {size} was not swept; have {self.sizes}") from None

    @property
    def baseline_tps(self) -> float:
        """Goodput at batch size 1: one sequence at a time, the Day-26 engine."""
        return self._at(1)

    def speedup(self, size: int) -> float:
        """Goodput at `size` over the single-row baseline. Ideal would be `size`."""
        return self._at(size) / self.baseline_tps

    def efficiency(self, size: int) -> float:
        """`speedup(size) / size`: the share of perfect linear scaling that survived.

        1.0 means the row was free. 0.5 means half the batch is paying for itself
        and half is not, which on a memory-bound decode usually means the sweep has
        found the point where the step time started growing with the batch.
        """
        return self.speedup(size) / size

    @property
    def best_size(self) -> int:
        """The batch size with the highest goodput (ties go to the smaller size).

        Not necessarily the largest size swept: past saturation another row adds
        step time without adding tokens, and the curve turns over.
        """
        return max(self.sizes, key=lambda s: (self._at(s), -s))

    @property
    def best_tps(self) -> float:
        """Goodput at `best_size`, the peak the sweep actually reached."""
        return self._at(self.best_size)

    def knee(self, threshold: float = 0.8) -> int:
        """Largest swept size whose efficiency is still at or above `threshold`.

        The operating point a serving stack would pick if it only cared about
        throughput: past the knee each new row costs more step time than it earns.
        Always answerable, because batch size 1 sits at efficiency 1.0 by
        definition and is guaranteed to be in the sweep.
        """
        return max(s for s in self.sizes if self.efficiency(s) >= threshold)


def fit_batch_scaling(points: list[tuple[int, float]]) -> BatchScaling:
    """Build a `BatchScaling` from (batch_size, goodput) points, validating them.

    Raises `ValueError` on anything that would make the ratios meaningless: fewer
    than two points (nothing to compare), a missing batch size 1 (no baseline to
    divide by, and inventing one from the smallest size swept would silently
    redefine every speedup in the table), a repeated size (two answers to one
    question), or a non-positive size or rate. Catching these here keeps a
    nonsense efficiency out of a printed table, where it would read as a result.
    """
    if len(points) < 2:
        raise ValueError(f"batch scaling needs at least two points; got {len(points)}")
    sizes = [size for size, _ in points]
    rates = [rate for _, rate in points]
    if any(size <= 0 for size in sizes) or any(rate <= 0.0 for rate in rates):
        raise ValueError(
            f"every batch size and goodput must be positive; got sizes={sizes}, tps={rates}"
        )
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"a repeated batch size has two answers to one question; got {sizes}")
    if 1 not in sizes:
        raise ValueError(
            f"batch scaling is measured against batch size 1, which is missing; got {sizes}"
        )
    return BatchScaling(sizes=sizes, tps=rates)


# --- the runner: the closures wired to the engine's own batched decode --------
#
# Everything above is the timing core: stdlib-only, model-free, the clock injected,
# the prefill and the step handed in as opaque callables. The pure tests feed it
# `lambda: [False]`. That measures the harness, not the engine, which is the same
# gap Day 26 closed for the read benchmark.
#
# Below, the closures are the real thing: a `BatchedPagedKVCache` over a real
# `BlockAllocator`, the Day-27 padded prefill with its mask and positions, and the
# Day-28 decode step, one query per row at each row's own absolute position. It is
# `LlamaModel.greedy_generate_batch`'s loop turned inside out so the timing core can
# drive it one step at a time, and a test pins that the tokens it collects are the
# ones that method emits.
#
# One deliberate substitution: a row reports done on a *step budget* rather than on
# EOS. Output lengths are a property of the model and the prompt, not of the batching
# machinery, and the tiny test model emits no EOS at all. Making the raggedness a
# knob is what lets head-of-line blocking be measured at a chosen spread (the "seven
# 8-token rows behind one 200-token row" case) instead of whatever a prompt happened
# to produce. The forwards are entirely real; only the stopping rule is prescribed,
# and a finished row is still forwarded, still cached and still charged, exactly as
# static batching charges it.
#
# torch and the model are imported inside the functions, not at module top, so the
# timing core above stays importable without them and keeps its stdlib-only shape.


@dataclass
class DecodeRun:
    """The two timed closures for one batched run, plus what they are writing into.

    prefill_fn: the padded prefill forward; emits every row's first token.
    step_fn:    one full-batch decode step; emits one token per row.
    rows:       `prompt + generated` per row, appended to as the closures run, so a
                caller (or a test) can hold the benchmark to the engine's output.
    cache:      the live `BatchedPagedKVCache`, so what the run actually cost in
                slots and blocks is inspectable after the fact.

    Not frozen: this is the mutable state of a run in progress, which is exactly
    what the timing core is stepping through.
    """

    prefill_fn: DoneFn
    step_fn: DoneFn
    rows: list[list[int]]
    cache: object


def build_batched_decode(
    model,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    stop_steps: list[int] | None = None,
    block_size: int = 16,
    pad_id: int = 0,
    num_blocks: int | None = None,
) -> DecodeRun:
    """Stand up a real batched decode and return it as a `DecodeRun` of closures.

    model:          a `LlamaModel`, real weights or tiny random ones; the benchmark
                    only needs its forward.
    prompts:        one list of token ids per row, ragged.
    max_new_tokens: cap on generated tokens per row, so the decode runs at most
                    `max_new_tokens - 1` steps (the prefill emits the first token).
    stop_steps:     per row, how many decode steps that row runs before it reports
                    done. `None` means every row runs to the cap (a uniform batch,
                    which is the shape the throughput sweep wants). 0 means the row
                    is finished at the prefill token.
    block_size:     tokens per physical block.
    pad_id:         filler for the prefill rectangle; never attended to, never
                    written to the cache.
    num_blocks:     pool size; defaults to exactly what the run needs, per row and
                    not on the total, since a row's partial last block is its own.

    A finished row keeps being forwarded: its query is still a column of the batch,
    its K/V still lands in a fresh slot, and it still takes a block when it crosses
    one. It simply stops collecting. That is what makes the measured
    `waste_fraction` the real bill rather than an estimate of one.

    Raises `ValueError` for a cap below 1, a `stop_steps` that is not one entry per
    prompt, or a budget past the cap (a row that could never report done would be
    recorded as capped and quietly turn a ragged sweep into a uniform one).
    """
    import torch

    from .batch import last_token_logits, pad_prompts
    from .cache import BatchedPagedKVCache, BlockAllocator

    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1; got {max_new_tokens}")
    max_steps = max_new_tokens - 1
    if stop_steps is None:
        stop_steps = [max_steps] * len(prompts)
    if len(stop_steps) != len(prompts):
        raise ValueError(
            "a batched run needs one step budget per prompt; got "
            f"{len(stop_steps)} for {len(prompts)} prompts"
        )
    if any(s < 0 for s in stop_steps):
        raise ValueError(f"a step budget cannot be negative; got {stop_steps}")
    if any(s > max_steps for s in stop_steps):
        raise ValueError(
            f"a step budget past max_new_tokens-1={max_steps} is never reached; got {stop_steps}"
        )

    batch = pad_prompts(prompts, pad_id=pad_id, side="left")
    if num_blocks is None:
        num_blocks = sum(
            (len(p) + max_new_tokens + block_size - 1) // block_size for p in prompts
        )
    cache = BatchedPagedKVCache(model.config, BlockAllocator(num_blocks, block_size), len(prompts))

    rows = [list(prompt) for prompt in prompts]
    state = {"next": None, "step": 0}

    def prefill_fn():
        # Day 27's padded rectangle, with the cache listening: each row's real K/V
        # goes into that row's own blocks, and the pads are never stored.
        with torch.no_grad():
            logits = model.forward(
                batch.input_ids,
                batch.position_ids,
                cache=cache,
                attention_mask=batch.attention_mask,
            )
        nxt = last_token_logits(logits, batch).argmax(dim=-1)
        state["next"] = nxt
        for i, token in enumerate(nxt.tolist()):
            rows[i].append(token)
        return [s == 0 for s in stop_steps]

    def step_fn():
        # Day 28's decode step: one query per row, each at its own cached length,
        # no key mask (the cache is ragged, so there is no pad key to silence).
        positions = torch.tensor(
            [[n] for n in cache.seq_lens], dtype=torch.long, device=batch.input_ids.device
        )
        with torch.no_grad():
            logits = model.forward(state["next"][:, None], position_ids=positions, cache=cache)
        nxt = logits[:, -1].argmax(dim=-1)
        state["next"] = nxt
        state["step"] += 1
        for i, token in enumerate(nxt.tolist()):
            if state["step"] <= stop_steps[i]:
                rows[i].append(token)  # a finished row is forwarded, not collected
        return [state["step"] >= s for s in stop_steps]

    return DecodeRun(prefill_fn=prefill_fn, step_fn=step_fn, rows=rows, cache=cache)


def time_model_batch(
    model,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    clock: Clock = time.perf_counter,
    **build_kwargs,
) -> BatchTiming:
    """Time one real batched run end to end and return its `BatchTiming`.

    Builds the engine's decode (`build_batched_decode`) and hands the two closures
    to `time_batched_decode` under the given clock. With a ragged `stop_steps` this
    is the head-of-line measurement: the waste fraction is the throughput the batch
    threw away, and `max_hol_inflation` is how much longer the first row to finish
    waited than its own work took.
    """
    run = build_batched_decode(model, prompts, max_new_tokens=max_new_tokens, **build_kwargs)
    return time_batched_decode(
        run.prefill_fn,
        run.step_fn,
        batch_size=len(prompts),
        max_steps=max_new_tokens - 1,
        clock=clock,
    )


def sweep_batch_sizes(
    model,
    prompt: list[int],
    sizes: list[int],
    *,
    max_new_tokens: int,
    clock: Clock = time.perf_counter,
    **build_kwargs,
) -> tuple[BatchScaling, list[BatchTiming]]:
    """Replicate one prompt across each batch size, time each run, fit the scaling.

    The rows are deliberately identical here: same prompt, same output length, so
    nothing is ragged, every issued token is collected, and the curve is a clean
    read of what adding a row to the batch costs in step time. Raggedness is the
    *other* measurement (`time_model_batch` with a `stop_steps`), and mixing the two
    into one number is how a batching benchmark ends up unable to say which effect
    it is showing.

    Returns `(scaling, timings)`: the fit over goodput at each size, plus the raw
    per-size timings, since the step-time spread is what explains the fit's shape.
    """
    timings: list[BatchTiming] = []
    for size in sizes:
        timings.append(
            time_model_batch(
                model,
                [list(prompt) for _ in range(size)],
                max_new_tokens=max_new_tokens,
                clock=clock,
                **build_kwargs,
            )
        )
    scaling = fit_batch_scaling([(t.batch_size, t.goodput_tps) for t in timings])
    return scaling, timings
