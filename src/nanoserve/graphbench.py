"""The Week 13 acceptance test: two servers, one recorded, one not. Day 57.

Week 13 made the decode step something a CUDA graph can hold, one property a day.
Day 49 compiled it, Day 50 moved the addressing out of it, Day 51 made the read a
window on a persistent table, Day 52 closed the shape set, Day 53 fixed the input
buffers, Day 54 recorded it, Day 55 warmed the list before the door opened and
Day 56 wired the whole thing to `--cuda-graphs`. Every one of those days asserted
its property against an engine object it was holding in the same process.

This is the week's claim asked the other way round: two servers, the same weights,
the same requests, one launched with the graphs and one without, and what falls out
is a comparison rather than an assertion.

  1. **A replay answers what an eager forward would.** Byte-identical text per
     client, across two processes, under a crowd. Everything else in the week is an
     optimisation; this is the thing that must not have changed, and the honest place
     to ask it is over a socket, from a caller that knows nothing about buckets.

     Under a crowd, and that word is the day. The first version of this file compared
     answers from a pass where each client ran alone, and it passed against an engine
     that was returning another request's continuation: a solo request sits in cache
     row zero, and a recorded read is the window `slots[:rows]`, so row zero is the
     one batch a misaligned replay gets exactly right. The bug needs two requests of
     different lengths, which is the first thing any real server sees and the last
     thing a component test does. See `rows_are_a_prefix`.
  2. **The process can say what its capture did.** Day 56 published the boot
     *decision*: how many shapes this server means to hold, how many it warmed. That
     is a statement about a moment before the first request. A server whose trimmed
     list is quietly recording at the top of the width axis, or worse, falling
     through to eager, published exactly the same payload as one replaying every
     step. So `/health` carries the counters now, and Day 54's two gates take a
     reading as readily as they take the object.
  3. **A claim about a run is a difference of two readings.** A counter is
     cumulative and a health check is an instant. "This server recorded nothing while
     it was serving" is a subtraction, and the control that proves the subtraction
     works is a server launched `--no-warm`: same engine, same list, identical
     tokens, and the recordings land in front of clients where a p99 will find them.
  4. **The tail is the number, not the mean.** A capture's claim is about host-side
     launch overhead per step, which lives in the middle of the ITL distribution. A
     warm-up's claim is about recordings that would otherwise happen in front of
     whoever asked for that context first, which is only ever the tail. Reporting one
     average over both hides each behind the other.

Two things this file refuses to do. It will not compare two arms that did different
amounts of work: a graphed arm that answered fewer tokens has a beautiful ITL and a
different denominator, so that is a `MeasurementUnsound` and not a result. And it
will not claim a speedup on a box with no CUDA, where the recorder is
`eager_recorder` and a "replay" is a forward plus a copy: the arithmetic of the
comparison is testable anywhere, the speedup is not, and `graphbench.py` is where
the two arms meet a card.

Everything here is a test and benchmark tool. It imports `httpx`, it drives servers
through `nanoserve.acceptance` and measures them through `nanoserve.servebench`, and
nothing in the engine imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from .acceptance import AcceptanceFailure, ClientPlan, run_crowd, wait_until_idle
from .captured import (
    CaptureStats,
    CaptureUnsound,
    check_all_shapes_captured,
    check_no_scattered_rows,
    check_replays_dominate,
)
from .servebench import LoadReport, MeasurementUnsound, run_open_loop

#: What a p99 needs before it is a percentile rather than a worst-of. Day 42's note
#: on nearest-rank ranks applies with more force here, because the whole comparison
#: is a ratio of two tails and a ratio of two worst-ofs is noise over noise.
DEFAULT_MIN_SAMPLES = 20

#: How much slower the recorded arm's tail is allowed to be before it is called a
#: regression. Not zero: a p99 is the noisiest number in the report, and a gate with
#: no slack fails on a box that was busy, which is how a real regression ends up
#: being explained away as flakiness the third time it fires.
DEFAULT_TOLERANCE = 0.10


# --- reading a capture off a server --------------------------------------------------


def capture_from_health(payload: dict) -> CaptureStats:
    """Lift the live counters out of a `/health` reading.

    Three cases, and keeping them apart is the function's whole job. No
    `cuda_graphs` at all is a server launched without the capture: a legitimate
    configuration, and the eager arm of every comparison in this file. A section with
    a `runtime` in it is the answer. A section *without* one is the Day-56 payload,
    which names a decision and says nothing about whether the decision is still true,
    and it is refused rather than read as zeros: zeros would make a server whose
    counters are missing indistinguishable from one that has served nothing, and the
    checks below would pass it.
    """
    graphs = payload.get("cuda_graphs")
    if graphs is None:
        return CaptureStats()
    runtime = graphs.get("runtime", graphs)
    if "mode" not in runtime:
        raise AcceptanceFailure(
            "this payload names a capture and does not say what it has done: a boot "
            "decision with no counters next to it describes a moment before the first "
            "request, and a server falling through to eager publishes it unchanged"
        )
    return CaptureStats.from_dict(runtime)


def boot_from_health(payload: dict) -> dict:
    """The boot half of the section, without the counters nested inside it."""
    graphs = payload.get("cuda_graphs") or {}
    return {k: v for k, v in graphs.items() if k != "runtime"}


# --- the requests both arms run ------------------------------------------------------


def paired_plans(
    n: int,
    *,
    prompts: Sequence[str],
    max_tokens: Sequence[int] = (8, 16),
    seed: int = 0,
    start: int = 0,
) -> list[ClientPlan]:
    """The load, built so the same list can be run against two different servers.

    Every plan streams, because the arms are compared on per-token cadence and a
    unary answer arrives in one piece: its TTFT is its end-to-end time and its
    inter-token latency does not exist.

    Sampling rotates on 3 so both arms carry greedy and seeded requests. That matters
    more here than it looks: the capture is recorded around the forward and the
    sampler runs outside it (see `check_output_not_held`), so "a seed means the same
    thing on a graphed server" is a claim about where the recorded region *ends*, and
    the only way to find out that it ends in the wrong place is to draw.

    Nobody hangs up. A departed client has no answer, and an answer that does not
    exist cannot be compared with the other arm's, so a crowd with disconnects in it
    would quietly shrink claim 1 to whoever stayed. Week 11 is where disconnects are
    the subject.

    Deterministic, and not for tidiness: the two arms have to be sent the same bytes
    or the comparison is between two workloads.
    """
    if n < 0:
        raise ValueError(f"a load cannot have a negative number of requests; got {n}")
    if not prompts:
        raise ValueError("a load needs at least one prompt")
    plans: list[ClientPlan] = []
    for i in range(n):
        g = start + i
        sampling: dict[str, Any] = {}
        if i % 3 == 1:
            sampling = {"temperature": 0.8, "top_p": 0.9, "seed": seed + g}
        elif i % 3 == 2:
            sampling = {"temperature": 1.0, "top_k": 4, "seed": seed + g}
        plans.append(
            ClientPlan(
                client_id=f"a{g}",
                prompt=prompts[i % len(prompts)],
                max_tokens=max_tokens[i % len(max_tokens)],
                stream=True,
                **sampling,
            )
        )
    return plans


# --- one arm ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmReport:
    """One server's whole side of the comparison.

    name:    "graphs" or "eager", and it is in the failure messages because at 3am
             "the arms disagree" is not actionable and "the recorded arm dropped a
             token" is.
    texts:   what each client got, from a pass where they all ran at once. A crowd
             and not a solo pass, which is the correction this day made to itself: a
             solo request is one row in cache row zero, and a capture that replays a
             misaligned window is exactly right for that batch and wrong for every
             other. The bug this file exists to catch cannot happen to a caller
             running alone.
    load:    the open-loop run, which is where every latency number comes from. A
             separate pass because it records frame arrival times and not text: a
             benchmark that reassembled every answer would be measuring its own
             string concatenation at p99.
    peak_running: how many rows the scheduler really had in one batch, sampled from
             `/health` while the crowd was in flight. Day 41's instrument, and it is
             admissibility rather than a statistic: every claim below passes against
             a crowd of one.
    before/after: two readings of the capture's counters, taken around everything
             this arm did. Not statistics: the difference between them is the only
             way to say what happened *during* a run, because the counters are
             cumulative over the process.
    boot:    what the launcher decided, for the report. Fixed for the life of the
             process, which is exactly why it is kept apart from the counters.
    """

    name: str
    texts: dict[str, str]
    load: LoadReport
    before: CaptureStats
    after: CaptureStats
    boot: dict
    peak_running: int = 0

    @property
    def served(self) -> CaptureStats:
        """What the capture did between the two readings: this arm's window."""
        return self.after.since(self.before)

    @property
    def graphed(self) -> bool:
        return self.after.mode == "capture"

    def render(self) -> str:
        load = self.load
        capture = self.served.render() if self.graphed else "no capture"
        return (
            f"{self.name:>7}  ITL p50 {load.itl_p50 * 1e3:.1f} ms  p99 "
            f"{load.itl_p99 * 1e3:.1f} ms  (n={load.n_itl_samples})  TTFT p50 "
            f"{load.ttft_p50 * 1e3:.1f} ms  {load.output_tokens} tokens  |  {capture}"
        )


async def read_health(base_url: str, *, timeout: float = 30.0) -> dict:
    """One `/health` reading, over the socket a client would use."""
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        return (await client.get("/health")).json()


async def run_arm(
    base_url: str,
    plans: Sequence[ClientPlan],
    arrivals: Sequence[float],
    *,
    name: str,
    model: str = "nanoserve",
    target_rate: float | None = None,
    timeout: float = 120.0,
) -> ArmReport:
    """Run the whole comparison against one server and report what it did.

    Two passes, because the two claims need different runs and neither run can
    answer the other's question. The crowd pass is where the answers come from, and
    it has to be a crowd: a server answering one caller has one row in cache row
    zero, which is the one batch shape a misaligned replay gets right. The load pass
    is open loop, which is the only way the queue exists at all, and it is where the
    cadence is measured.

    The first reading is taken before the crowd pass and not between the two, so the
    window covers every request this arm ever sent. That is the difference between
    this and a warm-up check: a lazily recorded server would do all of its recording
    in the first pass and show a spotless load window, which is precisely the server
    this is supposed to catch.

    The last reading waits for the loop to go idle first. A stream that ended is not
    a request that has left: an abort is queued and applied at the top of the next
    turn, so a reading taken the instant `run_open_loop` returns can be one decode
    step short and the window comes back missing calls nobody can account for.
    """
    before_health = await read_health(base_url, timeout=timeout)
    crowd = await run_crowd(base_url, plans, model=model, timeout=timeout)
    load = await run_open_loop(
        base_url, plans, arrivals, model=model, target_rate=target_rate, timeout=timeout
    )
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        await wait_until_idle(client)
        after_health = (await client.get("/health")).json()
    return ArmReport(
        name=name,
        texts=crowd.texts,
        load=load,
        before=capture_from_health(before_health),
        after=capture_from_health(after_health),
        boot=boot_from_health(before_health),
        peak_running=crowd.peak_running,
    )


# --- the two arms, side by side --------------------------------------------------------


@dataclass(frozen=True)
class ArmDelta:
    """The comparison itself: two arms and the ratios between them.

    Ratios rather than differences as the headline, and the direction is the one
    nobody has to look up: above one means the recorded arm was faster. The saved
    milliseconds are published next to them because a ratio hides the size of the
    thing it is a ratio of, and 1.4x on a 2 ms step is a different engineering
    decision from 1.4x on a 40 ms one.

    p50 and p99 are never collapsed into one number here. They are answers to two
    different questions this week asked: the middle is what a replay saves per step,
    and the tail is what a warm-up saves in front of one unlucky client.
    """

    graphs: ArmReport
    eager: ArmReport

    @staticmethod
    def _ratio(slow: float, fast: float) -> float:
        """`slow / fast`, and 0.0 when there is nothing to divide.

        Zero rather than an exception because a delta is built before it is checked:
        a run with no frame gaps in it has to be reportable long enough for
        `check_arms_comparable` to say why it is not a result.
        """
        return slow / fast if fast > 0.0 else 0.0

    @property
    def itl_p50_speedup(self) -> float:
        return self._ratio(self.eager.load.itl_p50, self.graphs.load.itl_p50)

    @property
    def itl_p99_speedup(self) -> float:
        return self._ratio(self.eager.load.itl_p99, self.graphs.load.itl_p99)

    @property
    def ttft_p50_speedup(self) -> float:
        return self._ratio(self.eager.load.ttft_p50, self.graphs.load.ttft_p50)

    @property
    def itl_p50_saved_ms(self) -> float:
        return (self.eager.load.itl_p50 - self.graphs.load.itl_p50) * 1e3

    @property
    def itl_p99_saved_ms(self) -> float:
        return (self.eager.load.itl_p99 - self.graphs.load.itl_p99) * 1e3

    @property
    def throughput_ratio(self) -> float:
        return self._ratio(self.graphs.load.output_tps, self.eager.load.output_tps)

    def row(self) -> dict:
        """One line of the table, in the units a reader can compare."""
        return {
            "graphs_itl_p50_ms": round(self.graphs.load.itl_p50 * 1e3, 3),
            "eager_itl_p50_ms": round(self.eager.load.itl_p50 * 1e3, 3),
            "itl_p50_speedup": round(self.itl_p50_speedup, 3),
            "graphs_itl_p99_ms": round(self.graphs.load.itl_p99 * 1e3, 3),
            "eager_itl_p99_ms": round(self.eager.load.itl_p99 * 1e3, 3),
            "itl_p99_speedup": round(self.itl_p99_speedup, 3),
            "graphs_ttft_p50_ms": round(self.graphs.load.ttft_p50 * 1e3, 3),
            "eager_ttft_p50_ms": round(self.eager.load.ttft_p50 * 1e3, 3),
            "graphs_tok_s": round(self.graphs.load.output_tps, 2),
            "eager_tok_s": round(self.eager.load.output_tps, 2),
            "itl_samples": self.graphs.load.n_itl_samples,
            "graphs_held": self.graphs.boot.get("graphs_held", 0),
            "recorded_while_serving": self.graphs.served.captures,
        }

    def render(self) -> str:
        return "\n".join(
            [
                self.graphs.render(),
                self.eager.render(),
                f"  ITL p50  {self.itl_p50_speedup:.2f}x  "
                f"({self.itl_p50_saved_ms:+.2f} ms a token)",
                f"  ITL p99  {self.itl_p99_speedup:.2f}x  "
                f"({self.itl_p99_saved_ms:+.2f} ms a token)",
                f"  tok/s    {self.throughput_ratio:.2f}x",
            ]
        )


# --- claim 1: the same answers ------------------------------------------------------


def check_same_answers(graphs: ArmReport, eager: ArmReport) -> None:
    """The week's correctness claim, across two processes.

    A missing client is a failure rather than a skip, for the same reason Day 41's
    `check_answers` says so: a comparison keyed on one side's clients is a comparison
    that passes by having nothing to check, and it would rot into that silently the
    first time one arm dropped a request.

    Two empty arms are the extreme of that and are refused by name, because "no
    client got a different answer" is true of a run where nobody got an answer.
    """
    if not graphs.texts and not eager.texts:
        raise AcceptanceFailure(
            "neither arm produced any answers, so the comparison has no answers to "
            "compare: a pass here would mean the two servers agreed about nothing"
        )
    everybody = sorted(set(graphs.texts) | set(eager.texts))
    missing: list[str] = []
    mismatched: list[str] = []
    for client_id in everybody:
        left = graphs.texts.get(client_id)
        right = eager.texts.get(client_id)
        if left is None or right is None:
            absent = graphs.name if left is None else eager.name
            missing.append(f"{client_id} (no answer from the {absent} arm)")
            continue
        if left != right:
            mismatched.append(f"{client_id}: {graphs.name} {left!r}, {eager.name} {right!r}")
    if missing:
        raise AcceptanceFailure(
            f"{len(missing)} client(s) were answered by one arm and not the other, so "
            f"there is nothing to compare for them: {', '.join(missing)}"
        )
    if mismatched:
        raise AcceptanceFailure(
            f"{len(mismatched)} client(s) got different text from the recorded server "
            "than from the eager one, which means a replay is not computing what the "
            "forward computes:\n  " + "\n  ".join(mismatched)
        )


# --- claims 2 and 3: what the capture did while it was serving -----------------------


def check_arm_replayed(arm: ArmReport, *, min_reuse: float = 0.5) -> None:
    """Day 54's two gates, asked of a server that has actually served traffic.

    Both of them have existed since Day 54 and until today no running process had
    ever been asked either, because the counters they read were inside the process
    and `/health` published a boot decision instead. Asked over a window they are
    the difference between "this server is configured for graphs" and "this server
    is using them".

    An arm with the capture switched off fails this rather than passing it. That is
    not pedantry: the eager arm of the comparison would otherwise satisfy every
    claim in this file about a capture it does not have, and a check that a control
    passes is a check that has stopped distinguishing anything.
    """
    if not arm.graphed:
        raise AcceptanceFailure(
            f"the {arm.name} arm has no capture: its counters say mode "
            f"{arm.after.mode!r}, so there is no replay to make a claim about and a "
            "pass here would be this check agreeing with a server that was launched "
            "without the flag"
        )
    window = arm.served
    if window.calls < 1:
        raise AcceptanceFailure(
            f"the {arm.name} arm ran no decode steps between its two readings: a "
            "window with no calls in it satisfies every gate below by having nothing "
            "to fail, so the run proved nothing about the capture"
        )
    try:
        check_all_shapes_captured(window)
        check_replays_dominate(window, min_reuse=min_reuse)
    except CaptureUnsound as exc:
        raise AcceptanceFailure(
            f"the {arm.name} arm served traffic its capture did not cover: {exc}"
        ) from exc


def check_arm_replayed_every_step(arm: ArmReport) -> None:
    """Refuse an arm whose capture sat out part of the loop. Day 57's finding.

    Kept apart from `check_arm_replayed` because the two say different things and
    only one of them is currently true of this engine. That one is about the capture
    being *used*: the graphs were recorded once, they are being replayed, and nothing
    is recording in front of a client. This one is about the capture being used
    *everywhere*, and it fails on any run where a request finished before its
    neighbour, because the scheduler leaves the survivor in whatever row it was in
    and a recorded read is a window from row zero.

    A separate function because a failure here is a number to act on and not a bug to
    fix in a hurry: the steps it names ran the same forward an engine with no capture
    runs, so the tokens are right and what was lost is the launch overhead the week
    was spent removing. The share is the argument for a persistent batch, which is
    what vLLM and SGLang keep for exactly this reason.
    """
    if not arm.graphed:
        raise AcceptanceFailure(
            f"the {arm.name} arm has no capture, so there is no coverage to measure"
        )
    try:
        check_no_scattered_rows(arm.served)
    except CaptureUnsound as exc:
        raise AcceptanceFailure(
            f"the {arm.name} arm's capture did not cover its whole loop: {exc}"
        ) from exc


def check_nothing_recorded_while_serving(arm: ArmReport) -> None:
    """Refuse a server that paid for a recording in front of a client. Day 55's claim.

    The counter a production process would watch, and it is only meaningful as a
    window: `captures` over the life of the process is the warm-up's own work, and a
    warm server's is exactly the length of its list. What must be zero is the number
    of recordings that happened after the door opened, because a recording is
    hundreds of milliseconds of stall paid by whoever asked for that context first,
    and it is invisible in everything except a tail.
    """
    if not arm.graphed:
        raise AcceptanceFailure(
            f"the {arm.name} arm has no capture, so it records nothing by not having "
            "anything to record: this claim is about a warm capture list and passing "
            "it here would say nothing about one"
        )
    recorded = arm.served.captures
    if recorded:
        raise AcceptanceFailure(
            f"the {arm.name} arm recorded {recorded} graph(s) while it was serving: a "
            "capture taken in front of a client is a stall inside somebody's stream, "
            "and the only place it shows up is the tail of the inter-token latency"
        )


# --- claim 4: the comparison is between two of the same thing -------------------------


def check_arm_was_crowded(arm: ArmReport, *, min_running: int = 2) -> None:
    """Refuse an arm whose clients never shared a batch.

    Day 41's admissibility check, and today is the day it earned a second home. The
    answers this file compares are a crowd's, and the reason they are a crowd's is
    that a solo request occupies cache row zero and therefore cannot catch a replay
    reading the wrong row. An arm that happened to serialise its clients would pass
    every claim in this file, on exactly the batch shape that cannot fail.

    `peak_running` is *sampled*, which means zero and one mean different things. One
    is a run that really did serve its clients one at a time. Zero is a run that
    finished between two polls of a 5 ms watcher, which says nothing about the batch
    and everything about the workload being too short to observe. Both fail this, and
    they want opposite fixes: more slots for the first, longer generations for the
    second.
    """
    if arm.peak_running < min_running:
        raise MeasurementUnsound(
            f"the {arm.name} arm never had more than {arm.peak_running} request(s) "
            f"running at once (wanted {min_running}): its answers were produced one "
            "row at a time, in cache row zero, which is the only batch a replay over "
            "a misaligned window gets right"
        )


def check_arms_comparable(
    graphs: ArmReport, eager: ArmReport, *, min_samples: int = DEFAULT_MIN_SAMPLES
) -> None:
    """Refuse a comparison whose two sides did different amounts of work.

    A `MeasurementUnsound` and not an `AcceptanceFailure`, because the two are
    different mornings: one says the capture made the server worse, the other says
    this pair of runs cannot be used to say anything about the capture at all.

    The token count is the check that earns its place. Latency per token is a ratio,
    and an arm that answered half the tokens has a numerator from one experiment and
    a denominator from another: it reports a beautiful ITL, it passes a threshold,
    and what it measured is the requests that did not fail.
    """
    if set(graphs.texts) != set(eager.texts):
        only_graphs = sorted(set(graphs.texts) - set(eager.texts))
        only_eager = sorted(set(eager.texts) - set(graphs.texts))
        raise MeasurementUnsound(
            f"the two arms served different clients ({graphs.name} only: "
            f"{only_graphs}, {eager.name} only: {only_eager}), so the latencies below "
            "are summaries of two different workloads"
        )
    for arm in (graphs, eager):
        if arm.load.n_failed:
            worst = arm.load.failures[0]
            raise MeasurementUnsound(
                f"{arm.load.n_failed} request(s) failed in the {arm.name} arm "
                f"({worst.client_id}: {worst.status} {worst.error!r}), and a run that "
                "dropped requests has a latency distribution over whatever was left"
            )
    if graphs.load.output_tokens != eager.load.output_tokens:
        raise MeasurementUnsound(
            f"the arms delivered different token counts ({graphs.name} "
            f"{graphs.load.output_tokens}, {eager.name} {eager.load.output_tokens}): "
            "inter-token latency is a ratio, and these two have different "
            "denominators"
        )
    for arm in (graphs, eager):
        if arm.load.n_itl_samples < min_samples:
            raise MeasurementUnsound(
                f"the {arm.name} arm has {arm.load.n_itl_samples} frame gap(s) and "
                f"this comparison wants {min_samples}: a p99 of a handful of samples "
                "is the worst of a handful of samples wearing a name it did not earn, "
                "and the tail is the number this day is about"
            )


def check_tail_not_worse(
    delta: ArmDelta, *, tolerance: float = DEFAULT_TOLERANCE
) -> None:
    """Refuse a capture that moved the p99 backwards.

    The gate is on the tail rather than on the mean because the tail is where both
    of this week's claims live, and because a mean can absorb a stall: one 200 ms
    recording spread over a thousand steps is 0.2 ms of mean and it is the whole of
    the p99. A regression that only a percentile can see is exactly the kind this
    week is able to introduce.

    The tolerance is slack, not a fudge. A p99 over a few hundred samples on a box
    that is also running the benchmark moves by several percent between identical
    runs, and a gate with no slack fires on noise until somebody stops believing it.
    """
    if tolerance < 0.0:
        raise ValueError(f"a tolerance is not negative; got {tolerance}")
    graphed = delta.graphs.load.itl_p99
    plain = delta.eager.load.itl_p99
    if graphed <= 0.0 or plain <= 0.0:
        raise MeasurementUnsound(
            "one of these arms has no inter-token latency to compare: a run with no "
            "frame gaps in it cannot support a claim about a tail"
        )
    if graphed > plain * (1.0 + tolerance):
        raise AcceptanceFailure(
            f"the recorded arm's ITL p99 is {graphed * 1e3:.2f} ms against the eager "
            f"arm's {plain * 1e3:.2f} ms, which is {graphed / plain:.2f}x and past the "
            f"{tolerance:.0%} this comparison allows for noise: the capture is meant to "
            "remove per-step launch overhead, so a slower tail is a recording that is "
            "still happening or a replay that is doing more work than the forward"
        )
