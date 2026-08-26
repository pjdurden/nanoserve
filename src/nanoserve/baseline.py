"""The outside baseline: HuggingFace `generate`, and what a speedup is made of. Day 44.

Week 12 has measured this engine against itself for three days. Day 42 timed it
through a socket, Day 43 took the server-side wait apart into five parts that sum
exactly, and both of those answer "where did my time go" without ever answering
"compared to what". This module is the "compared to what". The other system is
`transformers`' own `model.generate`, which is the code almost everybody actually
runs, and it is a fair opponent for exactly one reason: it does the same arithmetic
this engine does, in the same framework, on the same weights, on the same box, one
request at a time.

**One request at a time is not a handicap I imposed; it is the shape of the tool.**
`generate` takes a batch, but a batch there is a static rectangle fixed before the
call: every row runs until the longest one is done, no row can join late, and a
serving process that wants to answer a request arriving now has nothing to add it
to. Day 29 measured what that costs and Week 8 replaced it. So the honest baseline
is `generate` used the way a naive server would have to use it, and the honest
comparison is not "mine is faster" but the sentence this module exists to produce:

    throughput = occupancy x per-request rate

Both factors are measured, the product is the throughput *exactly*, and the whole
point is that they move in opposite directions.

  * **occupancy** is the mean number of requests *in the system* over the run: the
    area under the in-flight curve divided by the wall clock, which is
    `sum(per-request durations) / wall_clock`. It is an identity rather than a
    Little's-law estimate, because it is computed as the area and not inferred from
    a steady state that a 30-second benchmark does not have. For a one-at-a-time
    baseline it is 1.0 minus whatever the harness itself wasted between calls.
  * **per-request rate** is `total output tokens / total request-seconds`: how fast
    an average request in the system was actually being served. This is the factor
    a throughput headline hides, and under batching it is almost always *below* the
    baseline's, because a request sharing a forward pass with seven others waits
    for all eight rows at every step.

So a "4x faster than HuggingFace" is really "8x the requests in the system, each
going at half the speed", and those two numbers ask for completely different
follow-up work: the first is scheduler and memory work (Weeks 8-10, and it is
done), the second is kernel and compile work (Week 13, and it is not). A single
ratio cannot tell you which one you are short of. That factorisation is the day.

**Occupancy is not batch size, and confusing the two is this module's own first
bug.** An offline run submits every prompt before the first step, so a request is
"in the system" from the moment it is built, including the entire time it sits on
the waiting queue doing nothing. Measured here at one slot, eight requests: the
occupancy is 4.83 and the batch size is 1. The identity is still exact, because a
queued request's seconds land in the denominator of the per-request rate and
cancel; it is the *reading* that is wrong, and read wrong it credits the scheduler
with a queue.

So there are two factorisations of the same throughput, and every run reports both:

    throughput = occupancy       x tokens per request-second   (Little, on the system)
    throughput = batch occupancy x tokens per served-second    (the same, on service)

`batch_occupancy` counts only the seconds a request held a slot, so it really is
the mean batch size, and `queue_share` is the fraction of request-seconds that were
a wait rather than service. The first pair answers "what did the caller get"; the
second answers "what did the hardware do", and only the second one may be quoted as
a batching result. `check_concurrent` therefore gates on the batch occupancy, since
gating on the system occupancy would wave through a single-slot run that merely had
a long queue.

**What `compare` refuses to do is the rest of the day.** A ratio between two runs
is a statement about speed only if the two runs did the same work, and there are
four cheap ways to accidentally not do the same work, all of which make the new
system look good: fewer requests, shorter prompts, fewer generated tokens, or
different generated tokens. The last one is the dangerous one, because the run
completes, the numbers look plausible, and the engine was quietly computing
something else. Greedy on both sides makes the token sequences an exact equality
rather than a similarity, so the gate is a comparison and not a judgement call, and
`first_divergence` says where they split when they do.

**Two rules about which latencies may be put side by side.** The baseline has no
queue: nothing is ever waiting for a slot, so its TTFT *is* its prefill, and
`run_serial` records it as both. This engine under load has a queue that Day 43
measured at up to 85.8% of TTFT. Pitting the engine's loaded TTFT against the
baseline's TTFT is therefore mostly a measurement of the queue, which is a real
number about the server and a useless number about the model code. The pair that
compares like with like is prefill against prefill, `prefill_ratio`, and it is the
one to quote when the question is "is your forward pass slower than theirs".

vLLM's and SGLang's published numbers are this comparison, run properly: same
model, same hardware, same output lengths, HF `generate` as the one-at-a-time
reference. The multiples they report come overwhelmingly from the first factor.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

Clock = Callable[[], float]

#: Slack on the identities below. Everything here is a handful of additions of
#: `perf_counter` values, so the only error is float rounding; a microsecond is
#: orders of magnitude above that and far below anything a real bug produces.
_TOLERANCE_S = 1e-6

#: Relative slack when checking that the two factors multiply back to the speedup.
#: A ratio of ratios, so the tolerance is relative rather than in seconds.
_TOLERANCE_REL = 1e-9


class RunUnsound(AssertionError):
    """One run does not support the numbers taken from it.

    Not a bug in the system under test: a statement about the *experiment*. A
    baseline that turns out to have overlapped two requests is not a baseline, and
    an engine run whose requests happened to go one at a time is a measurement of
    a different system that still prints a throughput. Both produce real numbers
    about the wrong thing, which is the failure mode a benchmark has to be loud
    about, so this is an `AssertionError` like Day 42's `MeasurementUnsound` and
    for the same reason.
    """


class UnfairComparison(AssertionError):
    """Two runs were paired that did not do the same work.

    Separate from `RunUnsound` because the two are fixed differently. An unsound
    run is re-run; an unfair pairing means the ratio itself was never meaningful,
    however carefully each side was measured. Every check that raises this one is
    a check that would otherwise have produced a flattering number.
    """


# --- one request, on either side of the comparison ----------------------------


@dataclass(frozen=True)
class RunRecord:
    """One request's life, in whichever clock the system that served it uses.

    request_id:       the handle, matched positionally across the two runs.
    prompt_tokens:    how long the prompt was. Part of the fairness gate: the same
                      prompt has to be the same number of tokens on both sides, or
                      one system prefilled less than the other.
    output_token_ids: what it generated, prompt excluded. The correctness gate.
                      Kept as ids rather than text because text hides a
                      detokenizer disagreement inside a comparison that is
                      supposed to be about arithmetic.
    started_s:        when the caller handed the request over. For the engine that
                      is `RequestTimeline.created_at`, before the loop has heard of
                      it, so the queue wait is inside this span where it belongs.
    finished_s:       the last token.
    first_token_s:    the first token, or None if the harness could not see one.
    prefill_s:        the forward that produced the first token, when the system
                      can report it. For a queueless baseline it equals the TTFT
                      and `run_serial` fills it in; for this engine it is Day 43's
                      stamp, which excludes the queue and any thrown-away prefill.
    served_s:         of the duration, how much was spent holding a slot rather
                      than waiting for one. This is what separates the mean batch
                      size from the mean number of requests in the system, and it
                      is the difference between a batching result and a queueing
                      result. None means "this system has no queue, so all of it",
                      which is the honest default for anything driven serially.

    Frozen, unlike `RequestTimeline`, because a record is written once when a
    request is already over. The mutable object is the one that gets stamped
    while the request is alive; this is the transcript.
    """

    request_id: str
    prompt_tokens: int
    output_token_ids: tuple[int, ...]
    started_s: float
    finished_s: float
    first_token_s: float | None = None
    prefill_s: float | None = None
    served_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_token_ids", tuple(self.output_token_ids))
        if self.finished_s < self.started_s - _TOLERANCE_S:
            raise ValueError(
                f"request {self.request_id!r} finished at {self.finished_s} and that is "
                f"before it started at {self.started_s}"
            )
        first = self.first_token_s
        if first is not None and not (
            self.started_s - _TOLERANCE_S <= first <= self.finished_s + _TOLERANCE_S
        ):
            raise ValueError(
                f"request {self.request_id!r} put its first token at {first}, which is "
                f"outside the {self.started_s}..{self.finished_s} it was alive for"
            )
        served = self.served_s
        if served is not None and not (
            -_TOLERANCE_S <= served <= self.finished_s - self.started_s + _TOLERANCE_S
        ):
            raise ValueError(
                f"request {self.request_id!r} claims {served}s of service inside a life "
                f"of {self.finished_s - self.started_s}s: a request cannot be served for "
                "longer than it existed"
            )

    @property
    def output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def duration_s(self) -> float:
        """End to end, as the caller experienced it. The denominator of occupancy."""
        return self.finished_s - self.started_s

    @property
    def service_s(self) -> float:
        """Seconds holding a slot. The whole duration when the system has no queue."""
        return self.duration_s if self.served_s is None else self.served_s

    @property
    def queued_s(self) -> float:
        """Seconds waiting for a slot. Zero for anything driven one request at a time."""
        return self.duration_s - self.service_s

    @property
    def ttft_s(self) -> float | None:
        """Handover to first token. None when nothing stamped it, never a zero."""
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.started_s

    @property
    def decode_s(self) -> float | None:
        """First token to last. The other half of the duration, exactly."""
        if self.first_token_s is None:
            return None
        return self.finished_s - self.first_token_s

    @property
    def decode_tps(self) -> float:
        """Tokens *after the first*, over the decode span.

        The subtraction is the whole point. The first token was bought by TTFT and
        crediting it to decode as well counts it twice, which on a short run is a
        large flattering error: four tokens over three seconds of decode is three
        tokens per second, not four. Zero when there is no decode span rather than
        an exception, because a one-token request has a real answer here and it is
        that there was no inter-token time to measure.
        """
        span = self.decode_s
        if span is None or span <= 0.0 or self.output_tokens < 2:
            return 0.0
        return (self.output_tokens - 1) / span


# --- one whole run of one system ----------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class SystemRun:
    """What one system did with one workload, and the two factors of its throughput.

    name:         what to print. "hf-generate", "nanoserve", and it ends up in
                  every error message the gate raises, which is most of its job.
    records:      one per request, in submission order.
    wall_clock_s: the window the throughput is denominated in. `from_records`
                  makes it the span the requests actually occupied, first handover
                  to last token, which is the only window that contains the run and
                  nothing else. It can be passed explicitly for a run measured
                  inside a longer window, and `check` will still insist the records
                  fit in it.

    The two factors below are not a model of the run, they are a rearrangement of
    it. `throughput_tps` is tokens over the window; `occupancy` is request-seconds
    over the window; `per_request_tps` is tokens over request-seconds; and the
    request-seconds cancel. That is why `check` can assert the product rather than
    approximate it, and it is why the split is worth trusting when the two systems
    disagree by a factor of five.
    """

    name: str
    records: tuple[RunRecord, ...] = ()
    wall_clock_s: float = 0.0

    @classmethod
    def from_records(cls, name: str, records: Iterable[RunRecord]) -> SystemRun:
        """Build a run whose window is exactly the span its requests occupied.

        Not `sum of durations` and not "when I started my timer": the first would
        count a batched run's overlapping seconds twice and the second would put
        the harness's own model-loading inside the denominator. First handover to
        last token is the interval during which this system was the thing being
        measured.
        """
        ordered = tuple(records)
        if not ordered:
            return cls(name=name, records=(), wall_clock_s=0.0)
        span = max(r.finished_s for r in ordered) - min(r.started_s for r in ordered)
        return cls(name=name, records=ordered, wall_clock_s=span)

    # --- counts ---------------------------------------------------------------

    @property
    def n_requests(self) -> int:
        return len(self.records)

    @property
    def n_answered(self) -> int:
        """Requests that produced a first token, and the denominator of `mean_ttft_s`."""
        return sum(1 for r in self.records if r.first_token_s is not None)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    # --- the two factors ------------------------------------------------------

    @property
    def resident_s(self) -> float:
        """Request-seconds: the area under the in-flight curve, in seconds x requests."""
        return sum(r.duration_s for r in self.records)

    @property
    def occupancy(self) -> float:
        """Mean requests *in the system* over the window. 1.0 for a gapless serial run.

        Computed as area over window, so it needs no steady state and no arrival
        process: it is what the integral of the in-flight count says, on this run,
        including the ramp at the start and the drain at the end. A serial run
        comes out slightly *under* 1.0 by whatever the harness spent between calls,
        and that gap is real time the box was idle, so it belongs in the number.

        Not the batch size. An offline run hands over every prompt at once, so
        seven requests can be "in the system" while one is being served, and this
        number will happily read 4.83 on a single slot. `batch_occupancy` is the
        one to quote about batching.
        """
        if self.wall_clock_s <= 0.0:
            return 0.0
        return self.resident_s / self.wall_clock_s

    @property
    def per_request_tps(self) -> float:
        """Tokens per request-second: how fast an average request in the system went.

        The factor a throughput headline never quotes, and the one batching pays
        with. Not the mean of the per-request rates: that would weight a
        two-token request the same as a two-hundred-token one, and it would not
        multiply back to the throughput.
        """
        resident = self.resident_s
        if resident <= 0.0:
            return 0.0
        return self.output_tokens / resident

    @property
    def throughput_tps(self) -> float:
        """Output tokens over the window. Equals `occupancy * per_request_tps`."""
        if self.wall_clock_s <= 0.0:
            return 0.0
        return self.output_tokens / self.wall_clock_s

    # --- the same algebra, restricted to service ------------------------------

    @property
    def service_s(self) -> float:
        """Slot-seconds: the area under the *batch size* curve, not the queue's."""
        return sum(r.service_s for r in self.records)

    @property
    def batch_occupancy(self) -> float:
        """Mean batch size over the window: slot-seconds over wall clock.

        The number a batching claim is denominated in. It cannot exceed the slot
        count, which is what makes it a check on the run: an engine given eight
        slots and reading 1.2 here spent the run mostly empty, whatever its
        `occupancy` says.
        """
        if self.wall_clock_s <= 0.0:
            return 0.0
        return self.service_s / self.wall_clock_s

    @property
    def per_served_tps(self) -> float:
        """Tokens per slot-second: what one row of the batch actually produced.

        The factor that collapses under batching on hardware whose forward pass is
        compute-bound, and stays flat on hardware where it is bandwidth-bound. It
        is the cleanest one-number summary of what a wider batch costs a row, and
        pairing it with `batch_occupancy` gives the second exact factorisation of
        the same throughput.
        """
        service = self.service_s
        if service <= 0.0:
            return 0.0
        return self.output_tokens / service

    @property
    def queue_share(self) -> float:
        """Fraction of request-seconds that were a wait rather than service.

        0.0 for anything serial, and it climbs as the slots run out: the same
        thing Day 43 measured inside TTFT, here measured over a whole life and
        over the whole run. It is also exactly the gap between the two
        occupancies, which is why it belongs next to them.
        """
        resident = self.resident_s
        if resident <= 0.0:
            return 0.0
        return 1.0 - self.service_s / resident

    # --- what one caller felt -------------------------------------------------

    @property
    def mean_latency_s(self) -> float:
        return _mean([r.duration_s for r in self.records])

    @property
    def mean_ttft_s(self) -> float:
        """Over answered requests only. A request that never spoke has no TTFT, and
        letting it in as a zero would pull the mean towards a latency nobody had."""
        return _mean([r.ttft_s for r in self.records if r.ttft_s is not None])

    @property
    def mean_prefill_s(self) -> float:
        """Over the records that carry one. 0.0 when the system could not report it."""
        return _mean([r.prefill_s for r in self.records if r.prefill_s is not None])

    @property
    def mean_decode_tps(self) -> float:
        """Per-request inter-token rate, averaged over requests that had a decode span."""
        rates = [r.decode_tps for r in self.records if r.decode_tps > 0.0]
        return _mean(rates)

    # --- the invariant --------------------------------------------------------

    def check(self) -> None:
        """Assert the window contains the run and the factors multiply back.

        Two claims and they fail differently. A record outside the window means
        the denominator is wrong and every rate above is wrong with it, usually
        because a timer was started after the first request went out. A broken
        product means the arithmetic drifted, which cannot happen by construction
        and is therefore worth an assertion rather than a comment; Day 35 made the
        argument and Day 43's `RequestTimeline.check` is the same idea one level
        down.
        """
        if not self.records:
            return
        span = max(r.finished_s for r in self.records) - min(r.started_s for r in self.records)
        if span > self.wall_clock_s + _TOLERANCE_S:
            raise RunUnsound(
                f"{self.name}: the requests span {span:.9f}s but the run was measured "
                f"over {self.wall_clock_s:.9f}s, so at least one request is outside the "
                "measured window and every rate is denominated wrong"
            )
        pairs = (
            ("occupancy", self.occupancy, "per-request rate", self.per_request_tps),
            ("batch occupancy", self.batch_occupancy, "per-served rate", self.per_served_tps),
        )
        for left, left_value, right, right_value in pairs:
            product = left_value * right_value
            if abs(product - self.throughput_tps) > _TOLERANCE_REL * max(
                1.0, self.throughput_tps
            ):
                raise RunUnsound(
                    f"{self.name}: {left} {left_value} x {right} {right_value} is "
                    f"{product} tok/s, but the throughput is {self.throughput_tps} tok/s"
                )


# --- was this run the run it claims to be -------------------------------------


def check_serial(run: SystemRun) -> None:
    """Refuse a "one at a time" baseline in which two requests were ever both alive.

    The baseline's entire meaning is that it is the unbatched number, so an
    overlap does not make it a slightly better baseline, it makes it a small
    static batch and the comparison stops being the one anybody wanted. Cheap to
    check and easy to get wrong: a harness that hands `generate` a batch, or one
    that overlaps the tokenizer of request N+1 with request N, is a plausible
    accident.
    """
    ordered = sorted(run.records, key=lambda r: r.started_s)
    for before, after in zip(ordered, ordered[1:]):
        if after.started_s < before.finished_s - _TOLERANCE_S:
            raise RunUnsound(
                f"{run.name}: requests {before.request_id!r} and {after.request_id!r} "
                f"overlap ({after.started_s:.9f} starts before {before.finished_s:.9f} "
                "ends), so this run is not one request at a time"
            )


def check_concurrent(run: SystemRun, *, min_occupancy: float = 2.0) -> None:
    """Refuse an engine run that did not actually hold several requests at once.

    The mirror of Day 42's `check_offered_load`, one layer down and for the same
    reason. Every number this module produces about batching is a number about
    rows sharing a forward pass, and a run whose prompts drained one after another
    shares nothing while still printing a throughput. The default of 2.0 is the
    smallest batch at which the word "batching" is true.

    The gate is on `batch_occupancy` and not on `occupancy`, and the difference is
    the reason this function is worth writing down. An offline run submits every
    prompt before the first step, so the system occupancy is inflated by the queue
    and reads well above 2.0 even on a single slot: gating on it would pass every
    run ever measured, including the one with no batching in it at all.
    """
    if run.batch_occupancy < min_occupancy:
        raise RunUnsound(
            f"{run.name}: the mean batch was {run.batch_occupancy:.2f} rows against a "
            f"required {min_occupancy:.2f} (system occupancy {run.occupancy:.2f}, "
            f"{run.queue_share:.0%} of it queue), so this run went effectively one at "
            "a time and measures something other than batching"
        )


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """The first index at which two token sequences differ, or None if they agree.

    A length difference counts as a divergence at the shorter length, because
    that is the position where one system had a token and the other did not. The
    index is worth returning rather than a bool: "diverged at 0" is a loader or a
    prompt bug, "diverged at 137" is usually two dtypes drifting apart under a
    long greedy decode, and those are not the same morning.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


# --- the two runs together ----------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """The engine against the baseline, with the speedup taken apart into its factors.

    baseline: the one-at-a-time run. `hf-generate` in the real thing.
    engine:   this engine, several requests in flight.

    Built through `compare`, which refuses any pairing whose two sides did not do
    the same work. Every ratio below is engine-over-baseline where larger is better
    for the engine, except the latency ratios, which are baseline-over-engine so
    that "larger is better" stays true for all of them.
    """

    baseline: SystemRun
    engine: SystemRun

    # --- the headline and its two factors -------------------------------------

    @property
    def throughput_speedup(self) -> float:
        """Output tokens per second, engine over baseline. The number people quote."""
        base = self.baseline.throughput_tps
        return self.engine.throughput_tps / base if base > 0.0 else 0.0

    @property
    def occupancy_gain(self) -> float:
        """How many times more requests were in flight. The scheduler's contribution."""
        base = self.baseline.occupancy
        return self.engine.occupancy / base if base > 0.0 else 0.0

    @property
    def per_request_ratio(self) -> float:
        """Per-request token rate, engine over baseline. Below 1.0 is the normal case.

        This is what a served request pays for riding in a batch: it shares every
        forward pass with its batchmates, so it advances one token per iteration
        while the iteration got longer. A number well under 1.0 next to a large
        `occupancy_gain` is a healthy batching win. A number under 1.0 next to an
        `occupancy_gain` near 1.0 is a slower engine with extra steps.
        """
        base = self.baseline.per_request_tps
        return self.engine.per_request_tps / base if base > 0.0 else 0.0

    @property
    def batch_gain(self) -> float:
        """How many times wider the batch was. The honest version of `occupancy_gain`.

        This one really is a batching number: it compares slot-seconds, so a queue
        cannot inflate it and an offline submission cannot flatter it. Against a
        serial baseline it is just the engine's mean batch size.
        """
        base = self.baseline.batch_occupancy
        return self.engine.batch_occupancy / base if base > 0.0 else 0.0

    @property
    def per_served_ratio(self) -> float:
        """Tokens per slot-second, engine over baseline. What a row paid to be batched.

        The most interesting number on the table, because it is the one that
        depends on the hardware rather than on the scheduler. On a card, where the
        decode forward is bandwidth-bound, the extra rows ride in arithmetic that
        was idle and this stays near 1.0, so the batch gain flows almost undiluted
        into the speedup. On this CPU box, where the GEMMs are compute-bound, it
        collapses, and the entire batching win is eaten from the inside.

        At a batch of one it is a clean head-to-head of the two implementations'
        forward loops, with no batching in it at all.
        """
        base = self.baseline.per_served_tps
        return self.engine.per_served_tps / base if base > 0.0 else 0.0

    # --- what one caller felt -------------------------------------------------

    @property
    def mean_latency_ratio(self) -> float:
        """Baseline over engine, end to end. Under 1.0 means callers waited longer."""
        mine = self.engine.mean_latency_s
        return self.baseline.mean_latency_s / mine if mine > 0.0 else 0.0

    @property
    def ttft_ratio(self) -> float:
        """Baseline TTFT over engine TTFT, and read it with Day 43 in hand.

        The baseline has no queue, so its TTFT is a prefill; the engine's under
        load is mostly a wait for a slot. This ratio is therefore a statement
        about the *offered load*, not about the forward pass, and it gets worse
        the more requests you hand the engine at once, which is not a regression.
        `prefill_ratio` is the one that compares like with like.
        """
        mine = self.engine.mean_ttft_s
        return self.baseline.mean_ttft_s / mine if mine > 0.0 else 0.0

    @property
    def prefill_ratio(self) -> float:
        """Baseline prefill over engine prefill: the honest per-forward comparison.

        Both sides encode the same prompt with the same weights, so this asks the
        one question the queue cannot contaminate: is this engine's prompt forward
        faster than `transformers`' own? 0.0 when either side did not report a
        prefill, rather than a ratio against a missing number.
        """
        mine = self.engine.mean_prefill_s
        base = self.baseline.mean_prefill_s
        if mine <= 0.0 or base <= 0.0:
            return 0.0
        return base / mine

    @property
    def decode_rate_ratio(self) -> float:
        """Per-request inter-token rate, engine over baseline. The other side of the trade."""
        base = self.baseline.mean_decode_tps
        return self.engine.mean_decode_tps / base if base > 0.0 else 0.0

    # --- the invariant --------------------------------------------------------

    def check(self) -> None:
        """Assert the speedup really is the product of the two factors it was split into.

        The factorisation is algebra, not a fit, so a failure here is arithmetic
        drift rather than a surprising workload. It is asserted anyway because the
        whole argument of the day rests on the product being an identity, and an
        identity nobody evaluates is a sentence in a docstring.
        """
        self.baseline.check()
        self.engine.check()
        pairs = (
            ("occupancy gain", self.occupancy_gain, "per-request", self.per_request_ratio),
            ("batch gain", self.batch_gain, "per-served", self.per_served_ratio),
        )
        for left, left_value, right, right_value in pairs:
            product = left_value * right_value
            if abs(product - self.throughput_speedup) > _TOLERANCE_REL * max(
                1.0, self.throughput_speedup
            ):
                raise UnfairComparison(
                    f"the speedup is {self.throughput_speedup}, but {left} {left_value} "
                    f"x {right} ratio {right_value} is {product}"
                )

    def summary_lines(self) -> list[str]:
        """The comparison as the lines a benchmark script should print.

        Deliberately puts the factorisation directly under the headline, because
        the headline on its own is the number that gets quoted and it is the number
        that means the least.
        """
        base, mine = self.baseline, self.engine
        return [
            f"{base.name:>12}   {base.n_requests:3d} req   batch {base.batch_occupancy:5.2f}   "
            f"in-system {base.occupancy:5.2f}   {base.throughput_tps:7.2f} tok/s",
            f"{mine.name:>12}   {mine.n_requests:3d} req   batch {mine.batch_occupancy:5.2f}   "
            f"in-system {mine.occupancy:5.2f}   {mine.throughput_tps:7.2f} tok/s"
            f"   ({mine.queue_share:.0%} queue)",
            f"  throughput  {self.throughput_speedup:6.2f}x",
            f"    = batch     {self.batch_gain:6.2f}x"
            f"  x  per-served rate  {self.per_served_ratio:6.2f}x",
            f"    = occupancy {self.occupancy_gain:6.2f}x"
            f"  x  per-request rate {self.per_request_ratio:6.2f}x",
            f"  mean latency  {self.mean_latency_ratio:6.2f}x   "
            f"prefill {self.prefill_ratio:6.2f}x   decode rate {self.decode_rate_ratio:6.2f}x",
        ]


def compare(baseline: SystemRun, engine: SystemRun) -> Comparison:
    """Pair a baseline run with an engine run, refusing everything that is not a pair.

    Four gates, in the order that names the real problem first.

      1. **Same number of requests**, or the two runs are not describing one
         submission and nothing below is comparable.
      2. **Same prompt lengths, positionally.** A shorter prompt is less prefill,
         and prompt lists get reordered by accident all the time.
      3. **Same number of output tokens, per request.** Generating less is the
         easiest way in the world to look faster, and it happens for honest
         reasons: a differently configured stop token, a budget applied to the
         prompt-plus-output on one side and the output on the other.
      4. **The same output tokens.** Greedy on both sides makes this an exact
         equality. It is the gate that matters, because a system computing
         something else finishes and prints a beautiful number, and every other
         check here passes while it does.

    Then a division-by-zero guard, which is the only one that is about arithmetic
    rather than about honesty.
    """
    if baseline.n_requests != engine.n_requests:
        raise UnfairComparison(
            f"two runs of one workload need the same number of requests; got "
            f"{baseline.n_requests} from {baseline.name!r} and {engine.n_requests} "
            f"from {engine.name!r}"
        )
    for i, (a, b) in enumerate(zip(baseline.records, engine.records)):
        if a.prompt_tokens != b.prompt_tokens:
            raise UnfairComparison(
                f"request {i} was given a prompt of {a.prompt_tokens} tokens by "
                f"{baseline.name!r} and {b.prompt_tokens} by {engine.name!r}: the two "
                "runs did not prefill the same thing"
            )
        if a.output_tokens != b.output_tokens:
            raise UnfairComparison(
                f"request {i} produced {a.output_tokens} tokens under {baseline.name!r} "
                f"and {b.output_tokens} under {engine.name!r}: generating less is not "
                "generating faster"
            )
        split = first_divergence(a.output_token_ids, b.output_token_ids)
        if split is not None:
            raise UnfairComparison(
                f"request {i} diverged at index {split}: {baseline.name!r} generated "
                f"{a.output_token_ids[split]} and {engine.name!r} generated "
                f"{b.output_token_ids[split]}. A speedup against a system that said "
                "something else is not a speedup"
            )
    for run in (baseline, engine):
        if run.wall_clock_s <= 0.0:
            raise UnfairComparison(
                f"{run.name!r} has no measured time, so every ratio against it is a "
                "division by zero wearing a percentage sign"
            )
    return Comparison(baseline=baseline, engine=engine)


# --- the runners: the closures wired to a real model --------------------------
#
# Everything above is stdlib-only and model-free: records on a ruler, ratios, and
# the gate. The tests drive it with hand-placed timestamps, which is the only way
# to know the arithmetic is right, because a benchmark whose own math is unverified
# is a confident guess. It is the same split Day 13 drew, Day 20 repeated and Day 29
# repeated again.
#
# Below is the wiring. `run_serial` drives any one-at-a-time generator, and the one
# that matters is `hf_generate_one`; `run_concurrent` drives this engine and reads
# the stamps Day 43 already put on every request rather than timing anything itself.
# torch and transformers are imported inside the functions so the core above stays
# importable without either.


def run_serial(
    prompts: Sequence[Sequence[int]],
    generate_one: Callable[[list[int], Callable[[], None]], Sequence[int]],
    *,
    name: str = "baseline",
    clock: Clock = time.perf_counter,
) -> SystemRun:
    """Time a generator that can only do one request at a time.

    `generate_one(prompt, stamp)` returns the output token ids, prompt excluded,
    and calls `stamp()` at the moment the first of them exists. The stamp is a
    callback rather than a return value because the first token happens in the
    middle of the call, and a harness that timed it at the end would report the
    entire generation as time to first token, which is the single most common way
    a baseline's TTFT is quoted wrong.

    Every record gets `prefill_s = ttft_s` and leaves `served_s` at None, which
    means "all of it". Neither is a convenience. A system with no queue spends the
    whole wait for token one inside the prompt forward, so TTFT and prefill really
    are the same span; and it never has a request waiting for a slot, so every
    second of every request's life was service, which is what makes the baseline's
    batch occupancy exactly 1.0 and the denominator of `per_served_ratio` a fair
    one.
    """
    records: list[RunRecord] = []
    for i, prompt in enumerate(prompts):
        stamped: list[float] = []
        started = clock()
        output = generate_one(list(prompt), lambda: stamped.append(clock()))
        finished = clock()
        first = stamped[0] if stamped else None
        records.append(
            RunRecord(
                request_id=f"req-{i + 1}",
                prompt_tokens=len(prompt),
                output_token_ids=tuple(output),
                started_s=started,
                finished_s=finished,
                first_token_s=first,
                prefill_s=None if first is None else first - started,
            )
        )
    return SystemRun.from_records(name, records)


def run_concurrent(
    engine,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
    sampling=None,
    name: str = "nanoserve",
) -> SystemRun:
    """Submit every prompt at once, drain the engine, and read the timelines back.

    Nothing here is timed by this function, and that is the design. Day 43 hung
    the clock off the request state machine, so the engine already knows when each
    request was created, admitted, first spoke and finished, including under
    preemption. A benchmark that wrapped `run_to_completion` in its own timer
    could only recover the wall clock, and would have to guess at everything
    inside it.

    All the prompts go in before the first step, which makes this the *offline*
    shape: maximum offered load, no arrival process, the case a throughput number
    should be quoted from. Day 42's open-loop generator is the other shape and it
    is the one to quote latency from.
    """
    from .sampling import SamplingParams
    from .scheduler import Request

    requests = [
        engine.add_request(
            Request(
                request_id=f"req-{i + 1}",
                prompt_token_ids=list(prompt),
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                sampling=sampling or SamplingParams(),
            )
        )
        for i, prompt in enumerate(prompts)
    ]
    engine.run_to_completion()

    records = []
    for request in requests:
        timeline = request.timeline
        timeline.check()
        records.append(
            RunRecord(
                request_id=request.request_id,
                prompt_tokens=request.num_prompt_tokens,
                output_token_ids=tuple(request.output_token_ids),
                started_s=timeline.created_at,
                finished_s=timeline.finished_at,
                first_token_s=timeline.first_token_at,
                prefill_s=timeline.prefill_s,
                # Day 43's residency accumulator, and the reason the batch size is
                # recoverable at all: it is the seconds this request held a slot,
                # summed across admissions, so preemption is already accounted for.
                served_s=timeline.running_s,
            )
        )
    return SystemRun.from_records(name, records)


def hf_generate_one(model, *, max_new_tokens: int, eos_id: int | None = None, device=None):
    """A `generate_one` over `transformers`' `model.generate`. The real baseline.

    Greedy (`do_sample=False`, one beam) so the tokens are an exact equality
    against this engine's greedy path rather than two samples from one
    distribution, and `use_cache=True` because the baseline should be the fast way
    to run HF and not a straw man: without it every step re-encodes the whole
    prefix and the comparison becomes Day 11's cached-versus-naive lesson wearing
    somebody else's name.

    The first-token stamp is a `LogitsProcessor`, which is the trick worth
    keeping. `generate` is one opaque call, so there is no obvious place to time
    the prefill from the outside, and calling it twice (once with
    `max_new_tokens=1`) would time a different computation and double the work.
    A processor is invoked with the scores of each step *as they are produced*, so
    its first call is the instant the prompt forward finished and the first token's
    distribution existed. It stamps and returns the scores untouched, so the
    generation it is measuring is the generation that would have happened.
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class _FirstTokenStamp(LogitsProcessor):
        def __init__(self, stamp):
            self.stamp = stamp
            self.stamped = False

        def __call__(self, input_ids, scores):
            if not self.stamped:
                self.stamped = True
                self.stamp()
            return scores

    def generate_one(prompt: list[int], stamp: Callable[[], None]) -> list[int]:
        ids = torch.tensor([prompt], dtype=torch.long)
        if device is not None:
            ids = ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                logits_processor=LogitsProcessorList([_FirstTokenStamp(stamp)]),
                eos_token_id=eos_id,
                pad_token_id=eos_id if eos_id is not None else 0,
            )
        return out[0, len(prompt) :].tolist()

    return generate_one
