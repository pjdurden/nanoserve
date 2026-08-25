"""Where a request's latency went, measured inside the server. Week 12, Day 43.

Day 42 measured this engine from outside the socket for the first time and found
that at 0.8 requests per second the time to first token was 22.26s against 1.64s
for one client alone. The conclusion, that 93% of the wait was queueing, came from
subtracting those two numbers, and a subtraction is not a measurement. It assumes
the unloaded request's TTFT is the loaded one's prefill, which is a guess about a
forward pass whose batch had a different shape; it silently attributes preemption,
recompute and the bridge's own scheduling to whichever half it happens to land in;
and it cannot see anything that happens before the scheduler has heard of the
request, because from the client's side that time is indistinguishable from model
time.

This module is the inside view. Every request carries a `RequestTimeline`, the
timeline is stamped at the four moments that already exist in this engine as state
changes, and TTFT comes out as five named parts that sum to it exactly:

    created ---inbox---> queued ---queue_wait---> admitted ---prefill---> 1st token
                            ^                        |
                            \\----- requeue ----------/   (+ lost_prefill)

  * **inbox_s**: built by an HTTP handler, not yet seen by the loop. The bridge's
    own latency, and it is not small: arrivals are drained at the top of an
    iteration, so a request that lands mid-step waits for the whole forward before
    the scheduler knows it exists. It belongs to the server, it is invisible from
    outside, and no amount of model optimisation touches it.
  * **queue_wait_s**: on the waiting queue, wanting a slot or blocks. The wait
    Day 42 could only estimate.
  * **prefill_s**: the admission that actually produced the first token. In this
    engine that forward *is* the first token, because `_prefill` samples from the
    last position of the prompt: there is no "first decode" component in TTFT at
    all, which is a thing worth knowing and not what Day 42 assumed.
  * **requeue_s** and **lost_prefill_s**: zero unless the request was preempted
    before it ever spoke. Recompute means its earlier prefill was thrown away
    (`lost_prefill_s`) and it went back to the queue for another wait
    (`requeue_s`). These two are the latency half of Day 33's bill, the half
    `Engine.recompute_fraction` deliberately does not carry because it is denominated
    in tokens and this one is denominated in somebody's patience.

Two rules keep this honest.

**The parts sum to the whole, exactly, in every path.** Not approximately, and not
"for the common case". A request is, at every instant of its life, in exactly one
of three places (inbox, queue, forward), so the sum is an identity rather than a
model, and `check()` asserts it. A decomposition that only adds up when nothing
went wrong is a decomposition that hides exactly the runs worth looking at.

**Means decompose and percentiles do not.** `LatencyReport` prints percentiles for
the whole TTFT, because that is what a caller experienced, and *means* for the five
parts, because the p99 of a sum is not the sum of the p99s: the request with the
worst queue wait is usually not the one with the worst prefill, so a "p99 split"
adds up to a number no request ever had and typically overshoots the real p99 by a
lot. Reporting a split as percentiles is the standard way to make a latency budget
look like it was measured when it was invented.

vLLM and SGLang both export this split (`time_in_queue`, prefill and decode spans)
in their metrics, and it is what makes the difference between "the server is slow"
and "the server has too few slots", which are fixed by completely different work.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

#: The five parts of TTFT, in the order they are paid. Also the order the report
#: prints them, so a reader sees the timeline left to right.
TTFT_PARTS = (
    "inbox_s",
    "queue_wait_s",
    "requeue_s",
    "lost_prefill_s",
    "prefill_s",
)

#: Slack allowed when checking that the parts add up. The arithmetic is a handful
#: of float additions of `perf_counter` values, so the error is rounding and
#: nothing else; a microsecond is several orders of magnitude above it and several
#: below anything a scheduler bug would produce.
_TOLERANCE_S = 1e-6


def percentile(values: Sequence[float], q: float) -> float:
    """The nearest-rank percentile: `ceil(q/100 * n)`-th smallest, 1-indexed.

    Day 42 wrote this in `servebench.py` and Day 43 moved it here, where both
    sides of the socket can read it. The rule is the same one and it matters for
    the same reason: every number this returns is a latency some request really
    had. An interpolated p99 of 1.47s can be a value no caller experienced, which
    is fine for a distribution and misleading in a report whose job is to describe
    what happened to people.

    p0 is the minimum and p100 the maximum. An empty sequence is 0.0 rather than
    an exception, because a window in which nothing was answered should print a
    report rather than raise inside the printer, and `n_answered` is right there
    saying why.
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"a percentile is between 0 and 100; got {q}")
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(q / 100.0 * len(ordered))
    return ordered[max(rank - 1, 0)]


class TimelineBroken(RuntimeError):
    """A timeline was stamped in an order its own life does not allow.

    Loud, like `IllegalTransition`, and for the same reason. A latency report is
    read by somebody deciding what to optimise, so a timeline that quietly
    accumulated a negative queue wait or missed an admission does not produce an
    error, it produces a plausible number pointing at the wrong component. The
    checks here are cheap and they run on the same edges the state machine
    already validates.
    """


@dataclass
class RequestTimeline:
    """One request's whole life in the server's clock, stamped as it happens.

    clock:            the time source, injectable so the tests can move it by hand
                      and so a simulated run can drive it. `perf_counter` by
                      default: monotonic and unaffected by the wall clock being
                      corrected under a long run, which `time.time` is not. The
                      cost is that these numbers are not comparable across
                      processes, which is correct, because every span here is a
                      difference of two stamps taken by this one server.
    created_at:       when the `Request` object was built, which for a served
                      request is inside the HTTP handler, before the loop has been
                      told anything.

    Everything else is None until the moment it happens, so "did this request ever
    get a slot?" is a question the object answers rather than one the reader
    infers from a zero.

    Mutable, and stamped from two threads at different times: the handler thread
    creates it, the loop thread queues it, the worker thread inside `Engine.step`
    stamps the first token. That is safe without a lock only because those three
    never overlap for one request. The bridge hands a request to the engine at an
    iteration boundary and never touches it again until it is finished, which is
    the same invariant `serving.py` already relies on for the request itself.
    """

    clock: Callable[[], float] = time.perf_counter
    created_at: float = field(init=False)

    # --- the stamps -----------------------------------------------------------
    queued_at: float | None = None
    #: The most recent admission, cleared on preemption. `None` means "not in a
    #: forward right now", which is what makes the residency accumulators safe.
    admitted_at: float | None = None
    first_admitted_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None

    # --- the accumulators -----------------------------------------------------
    #: Total seconds on the waiting queue and in the running set. They exist
    #: separately from the stamps because preemption makes both of them a sum of
    #: intervals rather than one span.
    queued_s: float = 0.0
    running_s: float = 0.0
    #: The same two, restricted to the part of the life before the first token.
    #: This is what makes the TTFT split exact under preemption.
    ttft_queued_s: float = 0.0
    ttft_running_s: float = 0.0
    #: The final, successful prefill: admitted_at -> first_token_at. Every earlier
    #: one was thrown away and is in `lost_prefill_s`.
    prefill_s: float | None = None
    #: How many times this request has been prefilled. 1 on the happy path; one
    #: more for every preemption, whether or not it had spoken yet.
    num_prefills: int = 0

    _phase: str = field(default="new", init=False, repr=False)
    _last_change: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.created_at = self.clock()
        self._last_change = self.created_at

    # --- events ---------------------------------------------------------------

    def on_queued(self) -> None:
        """The loop has taken this request off the inbox and onto the queue."""
        if self.queued_at is not None:
            raise TimelineBroken("this request was already queued")
        now = self.clock()
        self.queued_at = now
        self._phase = "queued"
        self._last_change = now

    def on_admitted(self) -> None:
        """A slot and blocks were handed over: WAITING -> RUNNING.

        A request that was never queued is queued here, at this instant, rather
        than refused. It is the honest reading (it waited zero seconds for a slot)
        and it keeps this object usable by the scheduler tests, which drive
        transitions by hand over requests no queue ever held.
        """
        if self._phase == "running":
            raise TimelineBroken("this request is already running")
        if self.queued_at is None:
            self.on_queued()
        now = self._advance()
        self.admitted_at = now
        if self.first_admitted_at is None:
            self.first_admitted_at = now
        self.num_prefills += 1
        self._phase = "running"

    def on_preempted(self) -> None:
        """The blocks and the slot were taken back: RUNNING -> WAITING.

        The running time up to here stays on the books. It was really spent, the
        request really occupied a row for it, and if it happened before the first
        token then the work it bought no longer exists, which is exactly what
        `lost_prefill_s` reports.
        """
        if self._phase != "running":
            raise TimelineBroken("this request is not running, so it cannot be preempted")
        self._advance()
        self.admitted_at = None
        self._phase = "queued"

    def on_first_token(self) -> None:
        """The prefill sampled this request's first output token.

        Not a phase change: the request was running before it and is running
        after. What it does is close the TTFT accumulators, which is why it is an
        event at all. Stamping it at the *token* rather than at the end of the
        forward is deliberate; the token is what the caller is waiting for.
        """
        if self.first_token_at is not None:
            raise TimelineBroken("this request already emitted a first token")
        if self._phase != "running":
            raise TimelineBroken("a first token can only be emitted while running")
        now = self._advance()
        self.first_token_at = now
        self.prefill_s = now - self.admitted_at

    def on_finished(self) -> None:
        """Terminal, from either phase: a finished request and an aborted queue entry."""
        if self.finished_at is not None:
            raise TimelineBroken("this request already finished")
        self.finished_at = self._advance()
        self._phase = "done"

    def _advance(self) -> float:
        """Bank the interval since the last event under the phase that spent it.

        The one place time is accounted, so there is exactly one rule for it: the
        seconds since the previous stamp belong to whatever the request was doing
        during them, and they belong to the TTFT half as well until the first token
        has been emitted. Everything else in this class is a subtraction of two
        stamps; this is the part that survives preemption.
        """
        now = self.clock()
        elapsed = now - self._last_change
        if self._phase == "queued":
            self.queued_s += elapsed
            if self.first_token_at is None:
                self.ttft_queued_s += elapsed
        elif self._phase == "running":
            self.running_s += elapsed
            if self.first_token_at is None:
                self.ttft_running_s += elapsed
        self._last_change = now
        return now

    # --- the split ------------------------------------------------------------

    @property
    def answered(self) -> bool:
        """Whether this request ever produced a token. No token, no TTFT."""
        return self.first_token_at is not None

    @property
    def inbox_s(self) -> float | None:
        """Built by a handler, not yet on the queue. The bridge's own latency."""
        if self.queued_at is None:
            return None
        return self.queued_at - self.created_at

    @property
    def queue_wait_s(self) -> float | None:
        """Queued to first admission: the wait for a slot, before any work at all."""
        if self.first_admitted_at is None or self.queued_at is None:
            return None
        return self.first_admitted_at - self.queued_at

    @property
    def requeue_s(self) -> float:
        """Extra queue time caused by being preempted before the first token.

        Zero on every run where the pool was big enough, which is why it is a
        float rather than an optional: a request that was never evicted did not
        wait a second time, and 0.0 is the true answer rather than a missing one.
        """
        wait = self.queue_wait_s
        if wait is None:
            return 0.0
        return max(0.0, self.ttft_queued_s - wait)

    @property
    def lost_prefill_s(self) -> float:
        """Forward time spent on prefills that were evicted before they answered."""
        if self.prefill_s is None:
            return 0.0
        return max(0.0, self.ttft_running_s - self.prefill_s)

    @property
    def ttft_s(self) -> float | None:
        """Creation to first token: the whole server-side wait, inbox included."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.created_at

    @property
    def ttft_parts(self) -> dict[str, float]:
        """The five parts, which sum to `ttft_s` exactly. `check()` enforces that."""
        if not self.answered:
            return {}
        return {
            "inbox_s": self.inbox_s,
            "queue_wait_s": self.queue_wait_s,
            "requeue_s": self.requeue_s,
            "lost_prefill_s": self.lost_prefill_s,
            "prefill_s": self.prefill_s,
        }

    @property
    def decode_s(self) -> float | None:
        """First token to finish: every token after the first, plus any eviction."""
        if self.first_token_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.first_token_at

    @property
    def total_s(self) -> float | None:
        """Creation to finish. Not the client's number: the socket is outside this."""
        if self.finished_at is None:
            return None
        return self.finished_at - self.created_at

    # --- the invariant --------------------------------------------------------

    def check(self) -> None:
        """Assert that this timeline describes a life that could have happened.

        Three claims, and they are the reason to trust a number from this module:
        the stamps are in order, the residency accounts for every second between
        creation and finish, and the five parts of TTFT sum to TTFT. Day 35 put
        the scheduler's invariants in a function that runs every iteration for the
        same reason: an invariant nobody evaluates is a comment.
        """
        stamps = [
            ("created_at", self.created_at),
            ("queued_at", self.queued_at),
            ("first_admitted_at", self.first_admitted_at),
            ("first_token_at", self.first_token_at),
            ("finished_at", self.finished_at),
        ]
        seen = [(name, at) for name, at in stamps if at is not None]
        for (before, at_before), (after, at_after) in zip(seen, seen[1:]):
            if at_after < at_before - _TOLERANCE_S:
                raise TimelineBroken(
                    f"{after} ({at_after}) is before {before} ({at_before}): a request "
                    "cannot reach a later moment of its life first"
                )
        if self.total_s is not None:
            residency = (self.inbox_s or 0.0) + self.queued_s + self.running_s
            if abs(residency - self.total_s) > _TOLERANCE_S:
                raise TimelineBroken(
                    f"residency {residency:.9f}s does not account for the "
                    f"{self.total_s:.9f}s this request was alive: a request is in the "
                    "inbox, on the queue or in a forward, always, and there is no "
                    "fourth place for the difference to be"
                )
        if self.answered:
            parts = sum(self.ttft_parts.values())
            if abs(parts - self.ttft_s) > _TOLERANCE_S:
                raise TimelineBroken(
                    f"the TTFT parts sum to {parts:.9f}s but TTFT is {self.ttft_s:.9f}s"
                )


# --- many of them ---------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class LatencyReport:
    """What a window of finished requests says about where the time went.

    n:           timelines in the window, answered or not.
    n_answered:  the ones that produced a token, and the denominator of every
                 mean and every percentile below. A request that was aborted in
                 the queue has no TTFT, and letting it in as a zero would drag
                 every percentile towards a latency nobody had. It stays in `n`,
                 where somebody will ask about the difference.
    n_preempted: answered requests that were evicted at least once.
    reprefills:  prefills paid for beyond the first, summed over the window. The
                 token-side of the same bill is `Engine.recompute_fraction`.

    The five `mean_*` components sum to `mean_ttft_s` exactly. The `*_p50_s` and
    `*_p99_s` numbers do not sum to anything and are not meant to: they are the
    marginal distribution of one component, and the request holding the p99 queue
    wait is usually not the one holding the p99 prefill.
    """

    n: int = 0
    n_answered: int = 0
    n_preempted: int = 0
    reprefills: int = 0

    mean_inbox_s: float = 0.0
    mean_queue_wait_s: float = 0.0
    mean_requeue_s: float = 0.0
    mean_lost_prefill_s: float = 0.0
    mean_prefill_s: float = 0.0
    mean_decode_s: float = 0.0
    mean_total_s: float = 0.0

    ttft_p50_s: float = 0.0
    ttft_p90_s: float = 0.0
    ttft_p99_s: float = 0.0
    queue_p50_s: float = 0.0
    queue_p99_s: float = 0.0
    prefill_p50_s: float = 0.0
    prefill_p99_s: float = 0.0

    @property
    def mean_ttft_s(self) -> float:
        """The sum of the five components, by construction rather than by average.

        Computed rather than stored so the identity cannot drift: if a component
        is ever added or renamed, this number moves with it and the report cannot
        print a total that its own parts disagree with.
        """
        return (
            self.mean_inbox_s
            + self.mean_queue_wait_s
            + self.mean_requeue_s
            + self.mean_lost_prefill_s
            + self.mean_prefill_s
        )

    def _share(self, seconds: float) -> float:
        total = self.mean_ttft_s
        return seconds / total if total else 0.0

    @property
    def inbox_share(self) -> float:
        """Share of the average TTFT spent waiting for the loop to look at the inbox."""
        return self._share(self.mean_inbox_s)

    @property
    def queue_share(self) -> float:
        """Share of the average TTFT spent waiting for a slot. Day 42 guessed 93%."""
        return self._share(self.mean_queue_wait_s)

    @property
    def prefill_share(self) -> float:
        """Share of the average TTFT that was the forward pass the caller paid for."""
        return self._share(self.mean_prefill_s)

    @property
    def waste_share(self) -> float:
        """Share of the average TTFT that preemption added and nobody received."""
        return self._share(self.mean_requeue_s + self.mean_lost_prefill_s)

    def summary(self) -> str:
        """The report as text, with the sample count next to the percentiles.

        `n=` is printed because a p99 over ten samples is the maximum wearing a
        name it has not earned, which Day 42 made a rule and this report inherits.
        """
        n = self.n_answered
        return "\n".join(
            [
                f"requests n={self.n} answered={n} preempted={self.n_preempted} "
                f"reprefills={self.reprefills}",
                f"ttft    mean {self.mean_ttft_s * 1e3:8.1f}ms  "
                f"p50 {self.ttft_p50_s * 1e3:8.1f}ms  "
                f"p90 {self.ttft_p90_s * 1e3:8.1f}ms  "
                f"p99 {self.ttft_p99_s * 1e3:8.1f}ms  (n={n})",
                f"  inbox   {self.mean_inbox_s * 1e3:8.1f}ms  {self.inbox_share:6.1%}",
                f"  queue   {self.mean_queue_wait_s * 1e3:8.1f}ms  {self.queue_share:6.1%}"
                f"   p50 {self.queue_p50_s * 1e3:8.1f}ms  p99 {self.queue_p99_s * 1e3:8.1f}ms",
                f"  requeue {self.mean_requeue_s * 1e3:8.1f}ms",
                f"  lost    {self.mean_lost_prefill_s * 1e3:8.1f}ms",
                f"  prefill {self.mean_prefill_s * 1e3:8.1f}ms  {self.prefill_share:6.1%}"
                f"   p50 {self.prefill_p50_s * 1e3:8.1f}ms  p99 {self.prefill_p99_s * 1e3:8.1f}ms",
                f"decode  mean {self.mean_decode_s * 1e3:8.1f}ms   "
                f"total mean {self.mean_total_s * 1e3:8.1f}ms",
            ]
        )


def summarize(timelines: Iterable[RequestTimeline]) -> LatencyReport:
    """Aggregate a window of timelines into one report.

    Answered requests only, for everything except `n`: see `LatencyReport`. An
    empty window returns a report of zeros rather than raising, because a health
    endpoint asked about a server that has served nothing should answer, and `n=0`
    is the answer.
    """
    timelines = list(timelines)
    answered = [tl for tl in timelines if tl.answered]
    ttfts = [tl.ttft_s for tl in answered]
    queues = [tl.queue_wait_s for tl in answered]
    prefills = [tl.prefill_s for tl in answered]
    decodes = [tl.decode_s for tl in answered if tl.decode_s is not None]
    totals = [tl.total_s for tl in answered if tl.total_s is not None]
    return LatencyReport(
        n=len(timelines),
        n_answered=len(answered),
        n_preempted=sum(1 for tl in answered if tl.num_prefills > 1),
        reprefills=sum(max(0, tl.num_prefills - 1) for tl in answered),
        mean_inbox_s=_mean([tl.inbox_s for tl in answered]),
        mean_queue_wait_s=_mean(queues),
        mean_requeue_s=_mean([tl.requeue_s for tl in answered]),
        mean_lost_prefill_s=_mean([tl.lost_prefill_s for tl in answered]),
        mean_prefill_s=_mean(prefills),
        mean_decode_s=_mean(decodes),
        mean_total_s=_mean(totals),
        ttft_p50_s=percentile(ttfts, 50),
        ttft_p90_s=percentile(ttfts, 90),
        ttft_p99_s=percentile(ttfts, 99),
        queue_p50_s=percentile(queues, 50),
        queue_p99_s=percentile(queues, 99),
        prefill_p50_s=percentile(prefills, 50),
        prefill_p99_s=percentile(prefills, 99),
    )
