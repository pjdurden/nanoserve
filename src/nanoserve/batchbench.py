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

**Day 32 adds the other side of the same measurement.** Day 31 built the scheduled
loop that is supposed to remove both bills, and a claim like that is worth exactly
what the harness behind it is worth, so the continuous run is measured by the same
module, in the same vocabulary, over the same workload. `time_continuous_run` is the
mirror of `time_batched_decode`: one timed callable per *iteration* instead of per
step, because under continuous batching there is no separable prefill phase to hold
out of the denominator, only iterations in which some rows happened to be new. It
records the batch size of every iteration, which is the thing a static run cannot
have (that number is a constant there and a variable here), and the iteration each
request finished in, which is what turns "the batch finished" into "this request
finished" and is the entire latency argument of the week.

`compare_batching` puts the two side by side and refuses to do it unless both runs
collected the same tokens for the same requests. That check is the point of the
function. Two batching strategies timed on workloads that quietly differ produce a
speedup number that is not about batching at all, and it is very easy to do by
accident, since the static path stops rows on a scripted step budget and the
continuous path stops them on the request's own `max_new_tokens`.
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

    @property
    def end_to_end_goodput_tps(self) -> float:
        """Collected tokens per second of the *whole* run, prefill included.

        Added Day 32, for one reason: the scheduled loop has no separable prefill to
        hold out, so comparing its rate against `goodput_tps` would charge static
        batching for its decode only and continuous batching for everything it did.
        Two changes from the number above and they pull in opposite directions: the
        prefill seconds enter the denominator, and every row's prefill token enters
        the numerator, because the prefill really did emit one token per row and
        every row really did keep it.
        """
        useful = self.useful_tokens + self.batch_size
        return useful / self.total_s if self.total_s > 0.0 else 0.0

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


# --- Day 32: the same measurement, taken of the scheduled loop ----------------


@dataclass(frozen=True)
class IterationOutcome:
    """One engine iteration, as the timer is allowed to see it.

    batch_size: rows in this iteration's forward. The number that is a constant
                under static batching and a variable here, which is the whole
                mechanism the comparison is attributing its win to.
    collected:  tokens a request actually kept this iteration. Under the Day-31
                engine this equals `batch_size`, because a row is in the forward
                only while it still wants a token, and that equality is the claim
                rather than an assumption: it is counted, not assumed, so a future
                scheduler that speculates or drops a token still measures honestly.
    finished:   request ids that reached a terminal state *in* this iteration, at
                their own last token. Not the ids the scheduler released, which it
                does one iteration later: the answer is ready when it is sampled,
                and charging the reap to the caller would flatter neither side.

    Day 34 adds the three that make preemption visible:

    preempted:        request ids evicted in this iteration. Per iteration and not
                      as a running total, because who was evicted and when is what
                      turns a pool-level count into a per-request charge.
    forward_tokens:   positions the forward computed. 0 means "not reported", which
                      is charged as one per row; see `work_tokens`.
    recomputed_tokens: positions in that forward whose K/V had already been computed
                      once, i.e. the context of a resumed request. A subset of
                      `work_tokens`, never an extra charge alongside it.

    Frozen: it describes one iteration that already happened.
    """

    batch_size: int
    collected: int
    finished: tuple[str, ...] = ()
    preempted: tuple[str, ...] = ()
    forward_tokens: int = 0
    recomputed_tokens: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 0 or self.collected < 0:
            raise ValueError(
                f"an iteration cannot have negative rows or tokens; got "
                f"batch_size={self.batch_size}, collected={self.collected}"
            )
        if self.collected > self.batch_size:
            raise ValueError(
                "a row collects at most one token per iteration; got "
                f"{self.collected} tokens from {self.batch_size} rows"
            )
        if self.forward_tokens < 0 or self.recomputed_tokens < 0:
            raise ValueError(
                "an iteration cannot have negative forward or recomputed tokens; got "
                f"forward_tokens={self.forward_tokens}, "
                f"recomputed_tokens={self.recomputed_tokens}"
            )
        if self.forward_tokens and self.forward_tokens < self.batch_size:
            raise ValueError(
                "a forward computes at least one position per row; got "
                f"{self.forward_tokens} positions for {self.batch_size} rows"
            )
        if self.recomputed_tokens > self.work_tokens:
            raise ValueError(
                "an iteration cannot recompute more positions than it forwarded; got "
                f"{self.recomputed_tokens} of {self.work_tokens}"
            )

    @property
    def work_tokens(self) -> int:
        """Positions this iteration's forward computed, however it was reported.

        An unreported `forward_tokens` is charged one position per row, which is
        exactly what a decode-only iteration costs and is the shape every caller fed
        this class before Day 34 gave it a prefill length to carry. So the default is
        a correct measurement of the common case rather than a placeholder.
        """
        return self.forward_tokens if self.forward_tokens else self.batch_size


@dataclass
class ContinuousTiming:
    """One scheduled run: what the clock saw, and when each request got its answer.

    num_requests:   requests submitted to the run. Not the batch size: the whole
                    point is that these two numbers come apart.
    max_batch_size: slots the engine had, i.e. rows in the cache. The denominator
                    of `occupancy`.
    step_s:         seconds per iteration, in order.
    batch_sizes:    rows forwarded in each iteration, aligned with `step_s`.
    collected:      tokens kept in each iteration, aligned with `step_s`.
    finished_at:    request id to the index of the iteration it finished in.
    forward_tokens: positions each iteration's forward computed, aligned with
                    `step_s`. Empty means one per row for every iteration, which is
                    what a run of pure decodes costs and what a Day-32 timing meant.
    recomputed:     of those positions, the ones bought a second time. Empty means
                    zero throughout: nothing was preempted, so nothing was resumed.
    preempted_at:   request id to the iterations it was evicted in, one entry per
                    eviction. A request preempted three times appears once with a
                    list of three, because the cost is per eviction and the identity
                    is per request.

    The shape difference from `BatchTiming` is the result, not an accident of the
    API. There, `finished_at` is a list indexed by row, because a row *is* the
    request and there are exactly `batch_size` of them for the whole run. Here it
    is a dict keyed by request id, because a row is a slot that several requests
    take turns holding, and a request that has not been admitted yet still exists.

    A request missing from `finished_at` never finished within the run, and asking
    for its latency raises rather than returning the makespan, which would quietly
    read as a completed request that was merely slow.
    """

    num_requests: int
    max_batch_size: int
    step_s: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    collected: list[int] = field(default_factory=list)
    finished_at: dict[str, int] = field(default_factory=dict)
    forward_tokens: list[int] = field(default_factory=list)
    recomputed: list[int] = field(default_factory=list)
    preempted_at: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fill the Day-34 series before validating them, so every timing ever built
        # has one entry per iteration and no reader downstream has to branch on
        # whether preemption was measured. The defaults are measurements and not
        # placeholders: a forward with no reported length ran one position per row.
        if not self.forward_tokens:
            self.forward_tokens = list(self.batch_sizes)
        if not self.recomputed:
            self.recomputed = [0] * self.n_iterations
        if self.num_requests < 1:
            raise ValueError(f"a run needs at least one request; got {self.num_requests}")
        if self.max_batch_size < 1:
            raise ValueError(f"an engine needs at least one slot; got {self.max_batch_size}")
        if len(self.batch_sizes) != self.n_iterations or len(self.collected) != self.n_iterations:
            raise ValueError(
                "a timing needs one entry per iteration in each series; got "
                f"{self.n_iterations} times, {len(self.batch_sizes)} batch sizes, "
                f"{len(self.collected)} token counts"
            )
        if any(b > self.max_batch_size for b in self.batch_sizes):
            raise ValueError(
                f"a forward cannot have more rows than the cache has slots "
                f"({self.max_batch_size}); got {self.batch_sizes}"
            )
        if any(c > b for c, b in zip(self.collected, self.batch_sizes)):
            raise ValueError(
                "a row collects at most one token per iteration; got "
                f"collected={self.collected} over batch_sizes={self.batch_sizes}"
            )
        if len(self.finished_at) > self.num_requests:
            raise ValueError(
                f"more requests finished than were submitted: {len(self.finished_at)} "
                f"finishes for {self.num_requests} requests"
            )
        if any(not 0 <= i < self.n_iterations for i in self.finished_at.values()):
            raise ValueError(
                f"a request finished outside the run's {self.n_iterations} "
                f"iterations; got {self.finished_at}"
            )
        if (
            len(self.forward_tokens) != self.n_iterations
            or len(self.recomputed) != self.n_iterations
        ):
            raise ValueError(
                "a timing needs one entry per iteration in each series; got "
                f"{self.n_iterations} times, {len(self.forward_tokens)} forward "
                f"counts, {len(self.recomputed)} recompute counts"
            )
        if any(f < b for f, b in zip(self.forward_tokens, self.batch_sizes)):
            raise ValueError(
                "a forward computes at least one position per row; got "
                f"forward_tokens={self.forward_tokens} over batch_sizes={self.batch_sizes}"
            )
        if any(r > f for r, f in zip(self.recomputed, self.forward_tokens)):
            raise ValueError(
                "an iteration cannot recompute more positions than it forwarded; got "
                f"recomputed={self.recomputed} over forward_tokens={self.forward_tokens}"
            )
        if any(
            not 0 <= i < self.n_iterations
            for iterations in self.preempted_at.values()
            for i in iterations
        ):
            raise ValueError(
                f"a request was preempted outside the run's {self.n_iterations} "
                f"iterations; got {self.preempted_at}"
            )

    # --- what the clock saw ---------------------------------------------------

    @property
    def n_iterations(self) -> int:
        """Iterations the engine ran, the drain included."""
        return len(self.step_s)

    @property
    def forward_iterations(self) -> int:
        """Iterations that actually forwarded a row.

        The engine's last step reaps the final request and schedules an empty batch.
        It costs real time, so it belongs in the makespan; it forwards nothing, so
        it must not dilute `occupancy`.
        """
        return sum(1 for b in self.batch_sizes if b > 0)

    @property
    def total_s(self) -> float:
        """Makespan: every iteration, prefill and drain included."""
        return sum(self.step_s)

    def elapsed_s(self, iteration: int) -> float:
        """Seconds from the start of the run to the end of `iteration`."""
        return sum(self.step_s[: iteration + 1])

    # --- issued versus collected ----------------------------------------------

    @property
    def issued_tokens(self) -> int:
        """Token-slots the forwards computed: the sum of the batch sizes."""
        return sum(self.batch_sizes)

    @property
    def collected_tokens(self) -> int:
        """Tokens a request kept. Equal to the issued count on the Day-31 loop."""
        return sum(self.collected)

    @property
    def wasted_tokens(self) -> int:
        return self.issued_tokens - self.collected_tokens

    @property
    def waste_fraction(self) -> float:
        """Share of the forwards nobody kept. Day 29's number, for this loop.

        Zero here by construction rather than by luck: the batch is rebuilt every
        iteration out of requests that still want a token. Measuring it anyway is
        what makes it evidence instead of a restatement of the design.
        """
        issued = self.issued_tokens
        return self.wasted_tokens / issued if issued else 0.0

    @property
    def goodput_tps(self) -> float:
        """Collected tokens per second of the whole run.

        Over the makespan and not over a decode phase, because there is no phase to
        exclude: an iteration with newly admitted rows runs a prefill and the rest
        run a decode, and both are the loop. `BatchTiming.end_to_end_goodput_tps`
        is the static side's matching number.
        """
        return self.collected_tokens / self.total_s if self.total_s > 0.0 else 0.0

    @property
    def issued_tps(self) -> float:
        """Computed tokens per second: what the box moved, not what it served."""
        return self.issued_tokens / self.total_s if self.total_s > 0.0 else 0.0

    # --- how full the batch stayed --------------------------------------------

    @property
    def mean_batch_size(self) -> float:
        """Average rows per forwarding iteration."""
        forwards = self.forward_iterations
        return self.issued_tokens / forwards if forwards else 0.0

    @property
    def occupancy(self) -> float:
        """Share of the available slots a forward actually filled.

        The number that says whether the queue kept the engine fed. It falls for two
        opposite reasons and telling them apart is the operational skill: a drained
        queue (nothing left to admit, which is fine) or a dry pool (requests waiting
        on blocks the running set has reserved and is not using, which is the
        Week-9 debt showing up as idle rows).
        """
        slots = self.max_batch_size * self.forward_iterations
        return self.issued_tokens / slots if slots else 0.0

    # --- what preemption cost -------------------------------------------------

    @property
    def total_forward_tokens(self) -> int:
        """Positions the forwards computed across the run: the work denominator.

        Not `issued_tokens`, which counts rows. The two differ by exactly the thing
        Day 34 exists to expose: a decode row is one position and a prefill row is a
        whole context, so a resumed request rejoining after an eviction arrives as
        *one row* carrying dozens of positions. Divide the recompute by rows and a
        run that bought a fifth of its K/V twice reports a few percent.
        """
        return sum(self.forward_tokens)

    @property
    def recomputed_tokens(self) -> int:
        """Positions whose K/V was computed, thrown away, and computed again."""
        return sum(self.recomputed)

    @property
    def recompute_fraction(self) -> float:
        """Share of the forward work spent rebuilding K/V that already existed.

        The recompute surcharge, and the number to put next to a claim that
        incremental allocation raised occupancy. Occupancy went up because the pool
        stopped holding blocks for text nobody wrote; this is what was paid for it,
        and a run can trade a very good-looking occupancy for a quarter of its
        forward spent on work it had already done.

        It is also the honest half of the recompute-versus-swap argument. Swapping a
        victim's blocks to host memory and back pays PCIe bandwidth instead of FLOPs,
        so the number it would replace this one with is a transfer, not a forward,
        and comparing them needs both measured on the same workload rather than one
        measured and the other asserted.
        """
        forwarded = self.total_forward_tokens
        return self.recomputed_tokens / forwarded if forwarded else 0.0

    @property
    def num_preemptions(self) -> int:
        """Evictions in the run, counting a twice-evicted request twice."""
        return sum(len(iterations) for iterations in self.preempted_at.values())

    def preemptions(self, request_id: str) -> int:
        """How many times this request was evicted. 0 for one that never was.

        Unlike `latency_s`, an unknown id is not an error here. "Was this request
        preempted?" has an answer for every request the caller submitted, including
        the ones that sailed through, and that answer is zero.
        """
        return len(self.preempted_at.get(request_id, ()))

    @property
    def preempted_requests(self) -> tuple[str, ...]:
        """Ids that were evicted at least once, in order of first eviction."""
        order = sorted(self.preempted_at.items(), key=lambda kv: (min(kv[1]), kv[0]))
        return tuple(request_id for request_id, _ in order)

    @property
    def preemptions_per_request(self) -> float:
        """Evictions divided by requests submitted: the load-shaped summary.

        A total says how hard the pool was pressed and a per-request rate says what
        an arriving request should expect, which is the one a caller is entitled to
        ask about. Both hide the same thing, so `preempted_requests` is next door:
        one request evicted four times and four evicted once are the same rate and
        very different services.
        """
        return self.num_preemptions / self.num_requests

    # --- latency, per request -------------------------------------------------

    def latency_s(self, request_id: str) -> float:
        """Seconds from the start of the run to this request's own last token.

        The number static batching cannot produce. There, a row's observed latency
        is the batch's `total_s` whenever it finished, because the batch is returned
        as one object; here every request has its own, and the spread between them
        is the head-of-line delay, paid back.
        """
        try:
            iteration = self.finished_at[request_id]
        except KeyError:
            raise KeyError(
                f"request {request_id!r} never finished in this run, so it has no "
                "latency; the run's makespan is not an answer to that question"
            ) from None
        return self.elapsed_s(iteration)

    @property
    def latencies_s(self) -> list[float]:
        """Every finished request's latency, in finish order."""
        order = sorted(self.finished_at.items(), key=lambda kv: kv[1])
        return [self.elapsed_s(i) for _, i in order]

    @property
    def min_latency_s(self) -> float:
        """The first answer out of the engine: the row static batching punished most."""
        latencies = self.latencies_s
        return min(latencies) if latencies else 0.0

    @property
    def mean_latency_s(self) -> float:
        latencies = self.latencies_s
        return statistics.fmean(latencies) if latencies else 0.0

    @property
    def max_latency_s(self) -> float:
        """The straggler, which waits about as long either way. That is the point."""
        latencies = self.latencies_s
        return max(latencies) if latencies else 0.0

    # --- what preemption cost the caller --------------------------------------

    @property
    def preempted_latencies_s(self) -> list[float]:
        """Latency of every finished request that was evicted at least once."""
        return [
            self.latency_s(request_id)
            for request_id in sorted(self.finished_at)
            if self.preemptions(request_id)
        ]

    @property
    def straight_through_latencies_s(self) -> list[float]:
        """Latency of every finished request that was never evicted: the control."""
        return [
            self.latency_s(request_id)
            for request_id in sorted(self.finished_at)
            if not self.preemptions(request_id)
        ]

    @property
    def mean_preempted_latency_s(self) -> float:
        latencies = self.preempted_latencies_s
        return statistics.fmean(latencies) if latencies else 0.0

    @property
    def mean_straight_through_latency_s(self) -> float:
        latencies = self.straight_through_latencies_s
        return statistics.fmean(latencies) if latencies else 0.0

    @property
    def preemption_latency_penalty(self) -> float:
        """Mean victim latency over mean straight-through latency: what it felt like.

        The recompute is paid twice and in two currencies. `recompute_fraction` is
        the engine's half, in forward work, and it is spread across everybody. This
        is the caller's half, and it lands on one request: its K/V was thrown away,
        it went back to the queue, and it waited for blocks it used to own. 1.4 means
        a victim waited 40% longer than a request the scheduler never touched.

        Two guards, and they say different things on purpose. Nothing preempted is
        1.0: no request was punished, so the multiplier really is one. Nothing left
        unpreempted is 0.0: there is no control group, and a run in which everybody
        was a victim cannot say what not being one would have cost. Comparing the
        victims against themselves would produce 1.0 there and read as "preemption
        was free", which is the one wrong answer available.

        It is a comparison across requests and not across runs, so it carries their
        differences: a victim is the youngest running request, and the youngest tends
        to be the one that was admitted last and would have finished later anyway.
        The clean version of this number is the same request measured in two pools,
        which is what `sweep_pool_sizes` is for.
        """
        victims = self.mean_preempted_latency_s
        if not victims:
            return 1.0
        control = self.mean_straight_through_latency_s
        return victims / control if control > 0.0 else 0.0


# One timed engine iteration: run it, and say what it did.
IterationFn = Callable[[], IterationOutcome]
# Whether the engine has any reason to run another iteration.
UnfinishedFn = Callable[[], bool]


def time_continuous_run(
    step_fn: IterationFn,
    unfinished_fn: UnfinishedFn,
    *,
    num_requests: int,
    max_batch_size: int,
    clock: Clock = time.perf_counter,
    max_iterations: int = 100_000,
) -> ContinuousTiming:
    """Time a scheduled run: one timed iteration until the queues are empty.

    step_fn:        runs one `Engine.step()` and reports what that iteration did.
    unfinished_fn:  whether anything is still waiting or running. Asked before each
                    iteration and outside the timed window, so the clock only ever
                    sees the forward.
    num_requests:   how many requests the run was given, for the timing's own
                    validation.
    max_batch_size: slots the engine has.
    max_iterations: hang guard, not a policy, the same one `run_to_completion`
                    carries: every iteration either emits a token or releases a
                    request, so a loop that does not drain is a bug and a raise is
                    a better way to hear about it than a wedged benchmark.

    Two clock reads per iteration and none anywhere else, so a run of N iterations
    reads it exactly 2N times and a scripted fake clock lines up with the tests.

    Raises `ValueError` if a request reports finishing twice. That is not a
    harmless duplicate: the second one silently overwrites the first, moving the
    request's latency later, which makes the engine look slower than it is in
    exactly the direction that would be believed. It raises on an eviction reported
    for an already finished request for the same reason: a finished request owns
    nothing to take back, so that is an id mix-up, and it would otherwise land in
    the per-request counts as a victim that never was.
    """
    step_s: list[float] = []
    batch_sizes: list[int] = []
    collected: list[int] = []
    finished_at: dict[str, int] = {}
    forward_tokens: list[int] = []
    recomputed: list[int] = []
    preempted_at: dict[str, list[int]] = {}

    while unfinished_fn():
        if len(step_s) >= max_iterations:
            raise RuntimeError(
                f"the engine did not drain in {max_iterations} iterations; "
                f"{len(finished_at)} of {num_requests} requests finished"
            )
        t0 = clock()
        outcome = step_fn()
        step_s.append(clock() - t0)

        index = len(step_s) - 1
        batch_sizes.append(outcome.batch_size)
        collected.append(outcome.collected)
        forward_tokens.append(outcome.work_tokens)
        recomputed.append(outcome.recomputed_tokens)
        for request_id in outcome.preempted:
            if request_id in finished_at:
                raise ValueError(
                    f"request {request_id!r} was preempted after it finished, at "
                    f"iterations {finished_at[request_id]} and {index}"
                )
            preempted_at.setdefault(request_id, []).append(index)
        for request_id in outcome.finished:
            if request_id in finished_at:
                raise ValueError(
                    f"request {request_id!r} finished twice, at iterations "
                    f"{finished_at[request_id]} and {index}"
                )
            finished_at[request_id] = index

    return ContinuousTiming(
        num_requests=num_requests,
        max_batch_size=max_batch_size,
        step_s=step_s,
        batch_sizes=batch_sizes,
        collected=collected,
        finished_at=finished_at,
        forward_tokens=forward_tokens,
        recomputed=recomputed,
        preempted_at=preempted_at,
    )


@dataclass(frozen=True)
class BatchingComparison:
    """The two runs side by side: what the scheduled loop stopped paying.

    static:     the Day-29 `BatchTiming` for a fixed batch of the same requests.
    continuous: the Day-32 `ContinuousTiming` for the same requests, scheduled.

    Built through `compare_batching`, which refuses to pair two runs that did not
    produce the same tokens for the same requests. Every ratio below is a ratio of
    two measurements of *the same work*, or it is nothing at all.

    The counts (`work_ratio`, `waste_removed`) travel: they are numbers of rows and
    tokens and they hold on any box. The times (`makespan_speedup`, the latency
    ratios) do not: a shrinking batch also makes each forward cheaper, and how much
    that is worth depends entirely on where this hardware sits between bandwidth
    and FLOPs, which is the same caveat the throughput sweep carries.
    """

    static: BatchTiming
    continuous: ContinuousTiming

    # --- work ------------------------------------------------------------------

    @property
    def static_issued_tokens(self) -> int:
        """Token-slots the static run computed, its prefill row included."""
        return self.static.issued_tokens + self.static.batch_size

    @property
    def continuous_issued_tokens(self) -> int:
        """Token-slots the scheduled run computed."""
        return self.continuous.issued_tokens

    @property
    def work_ratio(self) -> float:
        """How many times more token-slots static batching computed for the same answers.

        The honest headline of the week, because it is a count. 4.3x means the card
        did 4.3 times the forward work to produce the identical text.
        """
        issued = self.continuous_issued_tokens
        return self.static_issued_tokens / issued if issued else 0.0

    @property
    def waste_removed(self) -> float:
        """Waste fraction removed, in points: static's minus the scheduled loop's."""
        return self.static.waste_fraction - self.continuous.waste_fraction

    # --- time ------------------------------------------------------------------

    @property
    def makespan_speedup(self) -> float:
        """Static wall time over scheduled wall time, for the whole workload.

        The least flattering of these ratios and the one to quote first. The
        straggler sets the makespan on both sides, so this can sit near 1.0 (or
        under it, by the drain iteration) even while every other row's latency
        collapses. Continuous batching is not primarily a makespan optimisation.
        """
        return self.static.total_s / self.continuous.total_s

    @property
    def mean_latency_speedup(self) -> float:
        """Mean per-request latency, static over scheduled.

        Static batching hands the whole batch back at once, so *every* row's
        latency is the run's `total_s`, which is what makes the numerator a single
        number rather than an average.
        """
        mean = self.continuous.mean_latency_s
        return self.static.total_s / mean if mean > 0.0 else 0.0

    @property
    def first_answer_speedup(self) -> float:
        """How much sooner the first request got its answer. The user-visible one.

        The shortest request is the one static batching punishes hardest, and it is
        also the one a serving stack has the most of.
        """
        first = self.continuous.min_latency_s
        return self.static.total_s / first if first > 0.0 else 0.0

    @property
    def goodput_speedup(self) -> float:
        """Scheduled goodput over static goodput, both end to end, prefill included."""
        static_tps = self.static.end_to_end_goodput_tps
        return self.continuous.goodput_tps / static_tps if static_tps > 0.0 else 0.0


def compare_batching(static: BatchTiming, continuous: ContinuousTiming) -> BatchingComparison:
    """Pair a static and a scheduled run of the same workload, validating the pairing.

    Three checks, and they are the function. Same number of requests, or the two
    runs are not describing the same submission. Same collected tokens, or one of
    them generated less text than the other and the speedup is measuring that
    instead. Non-zero measured time on both sides, or the ratios are divisions by
    zero dressed up as results.

    The token check is the subtle one, because the two counters are shaped
    differently on purpose. `BatchTiming.useful_tokens` counts decode tokens only,
    since its prefill is a separate timed phase that emitted one token per row;
    `ContinuousTiming.collected_tokens` counts every token any request kept. So the
    same workload satisfies `useful_tokens + batch_size == collected_tokens`, and
    off-by-`batch_size` is exactly the mistake this catches.
    """
    if static.batch_size != continuous.num_requests:
        raise ValueError(
            "two runs of the same workload need the same number of requests; got "
            f"{static.batch_size} static rows and {continuous.num_requests} requests"
        )
    static_collected = static.useful_tokens + static.batch_size
    if static_collected != continuous.collected_tokens:
        raise ValueError(
            "two runs are comparable only if they collected the same tokens; got "
            f"{static_collected} static (decode plus one prefill token per row) "
            f"against {continuous.collected_tokens} scheduled"
        )
    if static.total_s <= 0.0 or continuous.total_s <= 0.0:
        raise ValueError(
            "a run with no measured time cannot be compared; got "
            f"{static.total_s}s static and {continuous.total_s}s scheduled"
        )
    return BatchingComparison(static=static, continuous=continuous)


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


# --- the runner: the closures wired to the Day-31 engine ----------------------
#
# The static runner above had to bolt a stopping rule onto the batch, because a row
# there cannot leave and something has to decide when it is done. Here the rule is
# already the request's: `max_new_tokens` is what the scheduler admitted against and
# what `append_token` stops on, so raggedness is expressed by giving each request its
# own budget rather than by scripting a done vector. That is the same shape as the
# static `stop_steps` (budget b means b-1 decode steps after the prefill token) and
# it is worth noticing that only one side needed the scaffolding.


@dataclass
class ContinuousRun:
    """The timed closure for one scheduled run, plus what it is running over.

    step_fn:       one `Engine.step()`, reported as an `IterationOutcome`.
    unfinished_fn: whether the engine has anything left to do.
    engine:        the live `Engine`, so the pool, the queues and the cache can be
                   inspected after the run.
    requests:      the submitted `Request` objects, in submission order, each
                   accumulating its own tokens as the run proceeds, so a test can
                   hold the benchmark to what the engine generates.
    """

    step_fn: IterationFn
    unfinished_fn: UnfinishedFn
    engine: object
    requests: list


def build_continuous_run(
    model,
    prompts: list[list[int]],
    *,
    max_new_tokens: int | list[int],
    block_size: int = 16,
    max_batch_size: int | None = None,
    num_blocks: int | None = None,
    pad_id: int = 0,
    eos_id: int | None = None,
) -> ContinuousRun:
    """Stand up a real scheduled run and return it as a `ContinuousRun` of closures.

    model:          a `LlamaModel`, real weights or tiny random ones.
    prompts:        one list of token ids per request, ragged.
    max_new_tokens: one budget for every request, or one per request. The ragged
                    list is the head-of-line measurement; the scalar is the uniform
                    case a throughput sweep wants.
    max_batch_size: slots, i.e. rows in the cache. Defaults to one per prompt, which
                    is the fair setting against a static batch of the same prompts:
                    same width of forward available, different rule for filling it.
    num_blocks:     pool size. Defaults to every request's *worst case* reservation,
                    so nothing ever queues on blocks. That default is bigger than
                    the static runner's for the same workload, and the difference is
                    the Day-30 debt showing up in a benchmark default: admission
                    reserves the whole budget up front. Pass a smaller pool to
                    measure a queue.

    The timed closure reports the ids that reached a terminal state *during* that
    iteration, which is one iteration earlier than the scheduler releases them.
    Deferred release is right for the pool (nothing is freed while a forward may
    still be reading it) and wrong for latency: the answer exists the moment its
    last token is sampled, and a benchmark that waited for the reap would charge
    every request one extra iteration and quietly understate the win.

    Raises `ValueError` for a budget below 1 or a budget list that is not one entry
    per prompt.
    """
    import torch

    from .cache import BatchedPagedKVCache, BlockAllocator
    from .engine import Engine
    from .scheduler import Request, Scheduler

    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * len(prompts)
    else:
        budgets = list(max_new_tokens)
    if len(budgets) != len(prompts):
        raise ValueError(
            "a scheduled run needs one token budget per prompt; got "
            f"{len(budgets)} for {len(prompts)} prompts"
        )
    if any(b < 1 for b in budgets):
        raise ValueError(f"every token budget must be at least 1; got {budgets}")

    if max_batch_size is None:
        max_batch_size = len(prompts)
    if num_blocks is None:
        num_blocks = sum(
            (len(prompt) + budget + block_size - 1) // block_size
            for prompt, budget in zip(prompts, budgets)
        )

    allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    engine = Engine(
        model,
        Scheduler(allocator, max_batch_size=max_batch_size),
        BatchedPagedKVCache(model.config, allocator, batch_size=max_batch_size),
        pad_id=pad_id,
    )
    requests = [
        engine.add_request(
            Request(
                request_id=f"r{i}",
                prompt_token_ids=list(prompt),
                max_new_tokens=budget,
                eos_token_id=eos_id,
            )
        )
        for i, (prompt, budget) in enumerate(zip(prompts, budgets))
    ]
    # Insertion-ordered, so the finishes a step reports are in submission order.
    pending = {request.request_id: request for request in requests}

    def step_fn() -> IterationOutcome:
        # `no_grad` for the same reason the static closures use it: timing a graph
        # nobody will backward through would charge one side for bookkeeping the
        # other does not do.
        before = engine.collected_tokens
        # Deltas rather than totals: the engine counts for the whole run and the
        # timing wants this iteration, and taking the difference is what lets the
        # recompute be attributed to the iteration that paid it.
        before_forward = engine.forward_tokens
        before_recompute = engine.recomputed_tokens
        with torch.no_grad():
            out = engine.step()
        finished = tuple(rid for rid, r in pending.items() if r.is_finished)
        for request_id in finished:
            del pending[request_id]
        return IterationOutcome(
            batch_size=out.batch_size,
            collected=engine.collected_tokens - before,
            finished=finished,
            # The scheduler's own victim list, taken from the output rather than
            # inferred from the states afterwards: a request can be evicted and
            # re-admitted in later iterations, so "who is waiting now" is not the
            # same question as "who was evicted this iteration".
            preempted=tuple(r.request_id for r in out.preempted),
            forward_tokens=engine.forward_tokens - before_forward,
            recomputed_tokens=engine.recomputed_tokens - before_recompute,
        )

    return ContinuousRun(
        step_fn=step_fn,
        unfinished_fn=engine.has_unfinished,
        engine=engine,
        requests=requests,
    )


def time_model_continuous(
    model,
    prompts: list[list[int]],
    *,
    max_new_tokens: int | list[int],
    clock: Clock = time.perf_counter,
    **build_kwargs,
) -> ContinuousTiming:
    """Time one real scheduled run end to end and return its `ContinuousTiming`."""
    run = build_continuous_run(model, prompts, max_new_tokens=max_new_tokens, **build_kwargs)
    return time_continuous_run(
        run.step_fn,
        run.unfinished_fn,
        num_requests=len(prompts),
        max_batch_size=run.engine.scheduler.max_batch_size,
        clock=clock,
    )


def sweep_pool_sizes(
    model,
    prompts: list[list[int]],
    *,
    max_new_tokens: int | list[int],
    pool_sizes: list[int],
    clock: Clock = time.perf_counter,
    **build_kwargs,
) -> list[ContinuousTiming]:
    """Run one workload through several pool sizes and return a timing for each.

    The preemption dial. Everything else is held fixed (same prompts, same budgets,
    same slots, same clock) and only the number of blocks moves, so the differences
    between the timings are the price of memory pressure and nothing else. A roomy
    pool preempts nobody and reports `recompute_fraction` 0.0; shrink it and the
    victims, the surcharge and the victims' extra waiting appear together, which is
    the shape of the Week-9 trade.

    Sweeping downward is the readable order but not a requirement; the sizes are
    used as given and returned aligned with the timings.

    The sweep checks the one thing that makes the numbers worth quoting: **every pool
    must have produced the same text**. A recompute that resumes instead of restarts
    is the whole correctness claim of Day 33, and it fails silently, so a harness that
    reported a surcharge for a run that generated something else would be quoting the
    price of a bug. Raises `ValueError` if two pools disagree.

    Also raises for an empty or repeated pool size, and passes through the
    `KVCacheExhausted` from `add_request` for a pool too small to hold the largest
    request's worst case, which is a refusal at the door rather than a slow run.
    """
    if not pool_sizes:
        raise ValueError("a pool sweep needs at least one pool size")
    if len(set(pool_sizes)) != len(pool_sizes):
        raise ValueError(f"a repeated pool size has two answers to one question; got {pool_sizes}")
    if any(size < 1 for size in pool_sizes):
        raise ValueError(f"every pool size must be at least one block; got {pool_sizes}")

    timings: list[ContinuousTiming] = []
    texts: list[list[int]] | None = None
    for size in pool_sizes:
        run = build_continuous_run(
            model, prompts, max_new_tokens=max_new_tokens, num_blocks=size, **build_kwargs
        )
        timings.append(
            time_continuous_run(
                run.step_fn,
                run.unfinished_fn,
                num_requests=len(prompts),
                max_batch_size=run.engine.scheduler.max_batch_size,
                clock=clock,
            )
        )
        generated = [list(request.token_ids) for request in run.requests]
        if texts is None:
            texts = generated
        elif generated != texts:
            raise ValueError(
                f"a pool of {size} blocks returned different text from a pool of "
                f"{pool_sizes[0]}: preemption is only allowed to cost time, so this "
                "is a recompute that restarted a request instead of resuming it"
            )
    return timings


def compare_model_batching(
    model,
    prompts: list[list[int]],
    *,
    max_new_tokens: int | list[int],
    clock: Clock = time.perf_counter,
    **build_kwargs,
) -> BatchingComparison:
    """Run one workload both ways, back to back, and pair the timings.

    The same prompts and the same per-request output lengths, once as a static batch
    that runs until its slowest row is done and once through the Day-31 scheduler.
    Translating between the two stopping rules is the whole of the wiring: a budget
    of `b` tokens is `b - 1` decode steps for the static runner, whose prefill emits
    the first one, and the static batch is sized by the longest budget because that
    is what a fixed rectangle costs.

    Both runs share the injected clock, which matters for a fake one: a scripted or
    counting clock keeps counting across the pair rather than restarting, so the
    ratios stay exact.
    """
    budgets = (
        [max_new_tokens] * len(prompts) if isinstance(max_new_tokens, int) else list(max_new_tokens)
    )
    if len(budgets) != len(prompts):
        raise ValueError(
            "a comparison needs one token budget per prompt; got "
            f"{len(budgets)} for {len(prompts)} prompts"
        )
    static = time_model_batch(
        model,
        prompts,
        max_new_tokens=max(budgets),
        stop_steps=[b - 1 for b in budgets],
        clock=clock,
        **build_kwargs,
    )
    continuous = time_model_continuous(
        model, prompts, max_new_tokens=budgets, clock=clock, **build_kwargs
    )
    return compare_batching(static, continuous)


# --- the watermark, measured ----------------------------------------------------
#
# Day 30 gave admission a watermark: a share of the pool it refuses to spend, so a
# newcomer cannot take the last block and be evicted for it one step later. Day 36
# is the first day that asks whether it works, and the first thing the question
# turns up is that at this project's scale it has never been switched on, since
# `int(0.01 * num_blocks)` is 0 for any pool under a hundred blocks.
#
# The measurement below has no model in it, on purpose. What a watermark moves is
# preemptions and iterations, and both are integers the scheduler computes without
# a tensor in sight, so timing it on a CPU model would add noise to a quantity that
# has none. This is the same split as the Day-29 timing core: the arithmetic is
# tested where it can be pinned exactly, and the model-based runners live one layer
# up. The token the rows are fed is a constant, because what is being measured is
# who ran when, not what they said.

# The token every row is handed. Any in-vocab id does: nothing here reads it, and a
# scheduler that behaved differently for different token values would be the bug.
_SWEEP_TOKEN = 7


@dataclass(frozen=True)
class WatermarkPoint:
    """One workload through one reserve size: what the brake changed, and what it cost.

    watermark_blocks:    blocks admission held back for this run.
    num_blocks:          the pool it held them out of.
    num_requests:        requests submitted.
    completed:           requests that finished. Anything less is a starved queue,
                         and the sweep raises rather than reporting it, since a
                         reserve that hangs is not a reserve with better numbers.
    iterations:          scheduler iterations to drain. The watermark's price: a
                         request it holds back is a slot standing idle.
    admissions:          admissions performed, resumptions included. `admissions -
                         num_requests` is how many times somebody had to be let in
                         a second time.
    preemptions:         evictions over the run.
    thrashed_admissions: evictions of a request that had not yet filled the block
                         its admission bought, i.e. one that generated fewer than
                         `block_size` tokens between being let in and being thrown
                         out. This is the number the watermark is actually about,
                         and it is not the same as `preemptions`: evicting a request
                         that has been running for thirty iterations is the pool
                         being too small, and no admission policy can help it.
                         Evicting one that has not filled its first block is a
                         prefill the engine bought less than a block of progress
                         with, and that is the thrash a reserve converts into a wait.

                         The window is a block wide rather than one iteration wide
                         because that is what the measurement showed. A newcomer
                         does not take the pool's last block and die a step later;
                         it is admitted with a block that has room left in it, spends
                         that tail one token per iteration, and runs the pool dry
                         when it reaches the boundary. The delay between the
                         admission and the eviction it caused is the tail, not a
                         step.
    forward_tokens:      positions the forwards would compute, counted the Day-34
                         way: every prefilled position, plus one per decode row.
    recomputed_tokens:   of those, the ones a resumed request had already paid for.
    """

    watermark_blocks: int
    num_blocks: int
    num_requests: int
    completed: int
    iterations: int
    admissions: int
    preemptions: int
    thrashed_admissions: int
    forward_tokens: int
    recomputed_tokens: int

    @property
    def resumptions(self) -> int:
        """Admissions that were not a request's first. Equal to `preemptions` on a drained run."""
        return self.admissions - self.num_requests

    @property
    def recompute_fraction(self) -> float:
        """Share of the forward positions bought a second time. Day 34's number, per reserve."""
        return self.recomputed_tokens / self.forward_tokens if self.forward_tokens else 0.0

    @property
    def thrash_fraction(self) -> float:
        """Share of admissions evicted before they had filled the block they bought."""
        return self.thrashed_admissions / self.admissions if self.admissions else 0.0


def sweep_watermarks(
    prompts: list[list[int]],
    *,
    max_new_tokens: int | list[int],
    num_blocks: int,
    block_size: int = 16,
    max_batch_size: int = 8,
    watermarks: list[int],
    max_iterations: int = 10_000,
) -> list[WatermarkPoint]:
    """Run one workload through several admission reserves and count what each changed.

    Everything is held fixed except `watermark_blocks`: same prompts, same budgets,
    same pool, same slots, and a constant token, so the differences between the
    points are the admission policy and nothing else.

    The reserves are given in blocks rather than as shares, which is the unit the
    thing is spent in. A share cannot state "one block of ten" without a truncation
    the caller did not write, and truncation is precisely how the 1% default came to
    be zero everywhere without anybody noticing.

    Raises for an empty sweep, a repeated reserve (one question with two answers), a
    reserve the pool cannot hold, or a run that does not drain. That last one is not
    a timeout dressed up: before Day 36 a large enough reserve really could wedge the
    scheduler forever, and a sweep whose failure mode is a hung process would have
    reported that as a very slow run.
    """
    from .cache import BlockAllocator
    from .scheduler import Request, Scheduler

    if not watermarks:
        raise ValueError("a watermark sweep needs at least one reserve size")
    if len(set(watermarks)) != len(watermarks):
        raise ValueError(
            f"a repeated reserve has two answers to one question; got {watermarks}"
        )
    if any(not 0 <= w < num_blocks for w in watermarks):
        raise ValueError(
            f"every watermark_blocks must be in [0, {num_blocks}); got {watermarks}"
        )

    budgets = (
        [max_new_tokens] * len(prompts) if isinstance(max_new_tokens, int) else list(max_new_tokens)
    )
    if len(budgets) != len(prompts):
        raise ValueError(
            "a watermark sweep needs one token budget per prompt; got "
            f"{len(budgets)} for {len(prompts)} prompts"
        )

    points: list[WatermarkPoint] = []
    for reserve in watermarks:
        allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
        scheduler = Scheduler(
            allocator, max_batch_size=max_batch_size, watermark_blocks=reserve
        )
        requests = [
            Request(
                request_id=f"r{i}",
                prompt_token_ids=list(prompt),
                max_new_tokens=budget,
            )
            for i, (prompt, budget) in enumerate(zip(prompts, budgets))
        ]
        for request in requests:
            scheduler.add_request(request)

        # Length at admission, so an eviction can be charged against the progress
        # that admission actually bought rather than against wall-clock iterations.
        admitted_len: dict[str, int] = {}
        iterations = admissions = preemptions = thrashed = 0
        forward_tokens = recomputed_tokens = 0
        while scheduler.has_unfinished():
            if iterations >= max_iterations:
                raise RuntimeError(
                    f"a reserve of {reserve} blocks did not drain in {max_iterations} "
                    f"iterations: {scheduler.num_running} running, "
                    f"{scheduler.num_waiting} waiting, {allocator.num_free} blocks free"
                )
            out = scheduler.schedule()
            for request in out.preempted:
                preemptions += 1
                # A request that was never admitted cannot be preempted, so the
                # default is unreachable; it is here so a bookkeeping bug reads as
                # zero thrash rather than as a KeyError out of a measurement.
                grown = request.num_tokens - admitted_len.get(request.request_id, 0)
                if grown < block_size:
                    thrashed += 1
            for request in out.admitted:
                admissions += 1
                admitted_len[request.request_id] = request.num_tokens
                # A prefill runs the request's whole current context, which for a
                # resumed one is the prompt plus everything it had already emitted.
                forward_tokens += request.num_tokens
                if request.num_preemptions:
                    recomputed_tokens += request.num_tokens
            forward_tokens += len(out.decode)
            for request in out.scheduled:
                request.append_token(_SWEEP_TOKEN)
            iterations += 1

        completed = sum(1 for request in requests if request.is_finished)
        short = [r.request_id for r in requests if r.num_output_tokens != r.max_new_tokens]
        if short:
            raise ValueError(
                f"a reserve of {reserve} blocks left {short[0]!r} with "
                f"{next(r for r in requests if r.request_id == short[0]).num_output_tokens} "
                "tokens instead of its whole budget: preemption is only allowed to cost "
                "time, so this is a recompute that restarted a request instead of "
                "resuming it"
            )
        points.append(
            WatermarkPoint(
                watermark_blocks=reserve,
                num_blocks=num_blocks,
                num_requests=len(requests),
                completed=completed,
                iterations=iterations,
                admissions=admissions,
                preemptions=preemptions,
                thrashed_admissions=thrashed,
                forward_tokens=forward_tokens,
                recomputed_tokens=recomputed_tokens,
            )
        )
    return points
