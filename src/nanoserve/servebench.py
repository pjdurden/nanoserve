"""Day 42: measuring the server, and the queue a closed loop cannot see.

Phase 4 finished with a system that serves. Phase 5 has to say how fast, and the
honest version of that sentence is harder than it sounds, because "how fast" is
three different numbers that move in opposite directions:

  - **TTFT**, time to first token. The wait before anything appears. Under load
    this is mostly *queueing*, not prefill, and that is the finding, not a caveat.
  - **ITL**, inter-token latency. The cadence once text is flowing. This is what a
    reader experiences and it is the number batching degrades.
  - **throughput**, tokens per second out of the whole server. This is what
    batching improves, and it is bought with the two above.

An engine tuned for the third at the cost of the first two is a perfectly good
batch processor and a bad chat server, so a benchmark that reports one number is
not measuring an inference engine. vLLM's and SGLang's benchmark suites report all
three against an *offered rate*, and this module is the small version of that.

**The load generator decides what you are allowed to find out.** There are two
ways to drive a server and they answer different questions.

A **closed loop** keeps N clients busy: each one sends its next request when its
previous one comes back. It is the easy thing to write and it has a property that
disqualifies it from measuring latency, which is that the load it offers *falls*
when the server slows down. A server that takes twice as long gets half as many
requests per second from the same clients, so it is never asked to hold a queue,
so its reported TTFT is its service time with the queueing removed. This is
Gil Tene's coordinated omission, and the reason it matters here is that the worst
latency in a serving system is almost entirely the wait for a slot.

An **open loop** sends on a schedule that does not care what the server is doing:
requests arrive at `t = 0, 1/rate, 2/rate, ...` (or on a Poisson process, which is
the same rate with the burstiness a real arrival process has). Offer more than the
server can serve and the queue grows without bound and the latency curve goes
vertical, which is exactly the knee a capacity plan needs. The cost is that the
*generator* can now be the bottleneck, and a generator that falls behind its own
schedule has quietly become a closed loop again. So every record carries
`send_lag_s`, the gap between when a request was due and when it actually left, and
`check_schedule_kept` refuses a report whose numbers were produced by a harness
that could not keep up.

Three measurement decisions worth naming.

**The throughput denominator is the run window, not the summed request time.**
Forty tokens delivered by four concurrent requests in one second is forty tokens
per second, and dividing by the 3.6 seconds those requests collectively spent
in flight reports 11, which is one request's rate wearing the run's name. The
summed residency is still worth having, as `mean_in_flight`: by Little's law it is
the average number of requests in the system, and it is this module's
admissibility instrument, the same job Day 41's `peak_running` did. A benchmark
that ran one request at a time measured latency and called it load.

**Percentiles are nearest-rank, and `n` is reported next to them.** A p99 is a
latency that actually happened, not an interpolation between two that did, and a
p99 of twenty samples is the worst of twenty samples wearing a name it did not
earn. The ITL pool is across requests rather than per request, which weights long
generations heavily on purpose: a 100-token answer contributes 99 gaps and a
4-token answer contributes 3, and the tail of the ITL distribution is a statement
about the tokens, not about the requests.

**A frame is not a token.** The stream is the only endpoint that can be timed per
token, because a unary response arrives all at once and can only report end to
end. But Day 39's incremental detokenizer holds a token back when it is half a
UTF-8 character, and the server does not emit an empty frame, so the frames a
client counts can be fewer than the tokens the engine produced. `median_itl_s` is
the frame cadence (what the screen did) and `mean_token_itl_s` is the decode span
over the real token count (what the engine did), and quoting either as the other
is off by `tokens_per_frame`. The final frame is a receipt, not a token: it carries
`finish_reason` and the usage bill, so it ends the run rather than contributing a
gap.

Everything here is a benchmark tool. It imports `httpx`, reuses Day 41's
`ClientPlan` and `live_server`, and nothing in the engine imports it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx

from .acceptance import ClientPlan

# Day 43 moved `percentile` down into `latency.py`, which needs the same
# nearest-rank rule for the server-side split. It had to move rather than be
# shared from here, because the split is stamped by the scheduler and the
# scheduler cannot import a benchmark module that imports an HTTP client. It is
# re-exported by this import: `servebench.percentile` still resolves.
from .latency import percentile

Clock = Callable[[], float]


class MeasurementUnsound(AssertionError):
    """The run happened, and it does not support the numbers it produced.

    Not an error from the server and not a bug in the engine: a statement about
    the *experiment*. A run that never had two requests in flight, or one whose
    generator fell behind its own schedule, produces latency numbers that are real
    measurements of the wrong system. An `AssertionError` because that is what it
    is, and a distinct class because "the benchmark is invalid" and "the benchmark
    found a regression" must not be the same failure.
    """


# --- the percentile definition ------------------------------------------------------


# `percentile` is imported at the top of this file rather than defined here: see
# the note there. Every report below still reads it as `percentile`.


# --- one request, as the client saw it ----------------------------------------------


@dataclass(frozen=True)
class RequestRecord:
    """One request's whole timeline, in the clock of the machine that sent it.

    scheduled_at: when the arrival schedule said this request was due. In an open
                  loop it is decided before the run starts; in a closed loop there
                  is no schedule and it equals `sent_at`, which is precisely the
                  criticism of a closed loop expressed as a field.
    sent_at:      when the request actually went out.
    frame_times:  arrival time of each SSE frame that carried text, in order. The
                  terminal frame is not here: it is a receipt, and counting it
                  would add a gap that no token caused.
    done_at:      when the stream ended, i.e. when the terminal frame landed.
    output_tokens/prompt_tokens: the server's own bill, from the usage on the
                  terminal frame. Not `len(frame_times)`, which is smaller whenever
                  the detokenizer held a partial character back.

    Every latency below is a subtraction between two of these, which is the reason
    the class stores timestamps and not durations: a derived number can be checked
    against the timeline it came from, and a stored duration cannot.
    """

    client_id: str
    scheduled_at: float
    sent_at: float
    frame_times: tuple[float, ...] = ()
    done_at: float | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    status: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.sent_at < self.scheduled_at:
            raise ValueError(
                f"{self.client_id} was sent at {self.sent_at} before it was scheduled "
                f"at {self.scheduled_at}"
            )
        if self.frame_times:
            if self.frame_times[0] < self.sent_at:
                raise ValueError(
                    f"{self.client_id} saw a frame at {self.frame_times[0]}, before the "
                    f"request was sent at {self.sent_at}"
                )
            if any(b < a for a, b in zip(self.frame_times, self.frame_times[1:])):
                raise ValueError(f"{self.client_id} has frame times out of order")
            if self.done_at is not None and self.done_at < self.frame_times[-1]:
                raise ValueError(
                    f"{self.client_id} finished at {self.done_at}, before its last frame "
                    f"at {self.frame_times[-1]}"
                )

    # --- did this request count -----------------------------------------------

    @property
    def ok(self) -> bool:
        """A 200, no error, and at least one token delivered.

        The last clause is the one that does work. A "successful" request that
        delivered nothing has no TTFT and no cadence, and letting it into the
        percentiles as a zero pulls every one of them towards a latency no caller
        ever had. It belongs in `n_failed`, where somebody will ask about it.
        """
        return self.status == 200 and self.error is None and bool(self.frame_times)

    @property
    def n_frames(self) -> int:
        return len(self.frame_times)

    # --- the waits --------------------------------------------------------------

    @property
    def send_lag_s(self) -> float:
        """How late the generator was in sending this. Zero is the healthy value."""
        return self.sent_at - self.scheduled_at

    @property
    def ttft_s(self) -> float:
        """Time to first token, measured from the send: what the server owes.

        Queueing on the server is *inside* this number, which is the intended
        reading: from out here, waiting for a slot and waiting for a prefill are
        the same wait, and separating them is the server's business (Day 43's).
        """
        return self.frame_times[0] - self.sent_at if self.frame_times else 0.0

    @property
    def arrival_ttft_s(self) -> float:
        """Time to first token measured from when the request was *due*.

        The user's number. It differs from `ttft_s` only when the harness was late,
        and when it does differ, the difference is the part of the latency the
        benchmark caused rather than measured.
        """
        return self.frame_times[0] - self.scheduled_at if self.frame_times else 0.0

    @property
    def e2e_s(self) -> float:
        """The whole request, send to last byte."""
        return self.done_at - self.sent_at if self.done_at is not None else 0.0

    @property
    def arrival_e2e_s(self) -> float:
        """The whole request from when it was due, harness lateness included."""
        return self.done_at - self.scheduled_at if self.done_at is not None else 0.0

    # --- the cadence ------------------------------------------------------------

    @property
    def itls(self) -> list[float]:
        """Gaps between consecutive text frames: what the reader's screen did."""
        return [b - a for a, b in zip(self.frame_times, self.frame_times[1:])]

    @property
    def median_itl_s(self) -> float:
        """Middle frame gap; robust to the one step that hit a preemption."""
        gaps = self.itls
        return statistics.median(gaps) if gaps else 0.0

    @property
    def decode_span_s(self) -> float:
        """First text frame to last text frame: the window the tokens came out in."""
        if len(self.frame_times) < 2:
            return 0.0
        return self.frame_times[-1] - self.frame_times[0]

    @property
    def mean_token_itl_s(self) -> float:
        """The decode span over the gaps between *tokens*, not between frames.

        Equal to the mean frame gap when every token got its own frame, and smaller
        when the detokenizer held some back. Reporting the frame gap as a per-token
        latency overstates it by exactly `tokens_per_frame`.
        """
        gaps = self.output_tokens - 1
        span = self.decode_span_s
        return span / gaps if gaps > 0 and span > 0.0 else 0.0

    @property
    def tokens_per_frame(self) -> float:
        """How many tokens rode in the average frame. 1.0 when nothing was buffered."""
        return self.output_tokens / self.n_frames if self.n_frames else 0.0


# --- the run ------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadReport:
    """Every record from one run, plus the window they happened in.

    The window is the two numbers that make throughput mean anything: `started_at`
    is when the schedule began (not when the first request was sent, which would
    hide a generator that started late) and `finished_at` is when the last one
    ended. Every rate below divides by that span, so an idle stretch at the head of
    a paced schedule counts against the run, which is correct: a benchmark at 4
    requests per second is claiming a rate over a period, not a rate while busy.
    """

    records: tuple[RequestRecord, ...] = ()
    started_at: float = 0.0
    finished_at: float = 0.0
    target_rate: float | None = None
    concurrency: int | None = None

    def __post_init__(self) -> None:
        if self.finished_at < self.started_at:
            raise ValueError(
                f"a run cannot finish before it started; got {self.started_at} to "
                f"{self.finished_at}"
            )

    # --- who finished -----------------------------------------------------------

    @property
    def completed(self) -> list[RequestRecord]:
        return [r for r in self.records if r.ok]

    @property
    def failures(self) -> list[RequestRecord]:
        return [r for r in self.records if not r.ok]

    @property
    def n_ok(self) -> int:
        return len(self.completed)

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    # --- the rates --------------------------------------------------------------

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at

    @property
    def output_tokens(self) -> int:
        """Tokens delivered to callers who stayed. The numerator of goodput."""
        return sum(r.output_tokens for r in self.completed)

    @property
    def output_tps(self) -> float:
        """Output tokens per second of the run window: the server's throughput."""
        return self.output_tokens / self.duration_s if self.duration_s > 0.0 else 0.0

    @property
    def request_tps(self) -> float:
        """Completed requests per second of the run window."""
        return self.n_ok / self.duration_s if self.duration_s > 0.0 else 0.0

    @property
    def offered_rate(self) -> float | None:
        """Requests per second the schedule asked for; None for a closed loop."""
        return self.target_rate

    @property
    def achieved_rate(self) -> float:
        """Requests per second that actually completed. Same as `request_tps`."""
        return self.request_tps

    @property
    def kept_up(self) -> bool:
        """Did the server serve roughly what was offered?

        False means the queue was growing, which makes every latency in this report
        a measurement of an overloaded system. That is a legitimate thing to
        measure and an illegitimate thing to quote as "our p99", so it is a field
        rather than a footnote. Always True when no rate was offered, because a
        closed loop cannot fall behind a schedule it does not have.
        """
        if self.target_rate is None:
            return True
        return self.achieved_rate >= 0.95 * self.target_rate

    @property
    def mean_in_flight(self) -> float:
        """Average number of requests inside the server, by Little's law.

        Summed residency over the window. This is the admissibility number: a run
        whose `mean_in_flight` is 1.0 measured one request at a time no matter how
        many it sent, and every latency in it is a service time with no queueing in
        it at all. Day 41's `peak_running` is the same instrument on the server's
        side of the socket.
        """
        if self.duration_s <= 0.0:
            return 0.0
        return sum(r.e2e_s for r in self.completed) / self.duration_s

    @property
    def max_send_lag_s(self) -> float:
        """The latest any request left relative to when it was due."""
        return max((r.send_lag_s for r in self.records), default=0.0)

    # --- the distributions ------------------------------------------------------

    @property
    def ttfts(self) -> list[float]:
        return [r.ttft_s for r in self.completed]

    @property
    def e2es(self) -> list[float]:
        return [r.e2e_s for r in self.completed]

    @property
    def itls(self) -> list[float]:
        """Every frame gap from every completed request, pooled.

        Pooled and not averaged per request, which weights long generations
        heavily: an 11-frame answer contributes 10 samples and a 2-frame answer
        contributes 1. That is the right weighting for a question about token
        cadence and the wrong one for a question about requests, so `n_itl_samples`
        is published next to it.
        """
        return [gap for r in self.completed for gap in r.itls]

    @property
    def n_itl_samples(self) -> int:
        return len(self.itls)

    @property
    def ttft_p50(self) -> float:
        return percentile(self.ttfts, 50)

    @property
    def ttft_p90(self) -> float:
        return percentile(self.ttfts, 90)

    @property
    def ttft_p99(self) -> float:
        return percentile(self.ttfts, 99)

    @property
    def itl_p50(self) -> float:
        return percentile(self.itls, 50)

    @property
    def itl_p99(self) -> float:
        return percentile(self.itls, 99)

    @property
    def e2e_p50(self) -> float:
        return percentile(self.e2es, 50)

    @property
    def e2e_p99(self) -> float:
        return percentile(self.e2es, 99)

    def summary(self) -> str:
        """The report as a block of text, with `n` next to every percentile.

        Deliberately prints the offered rate, the achieved rate and
        `mean_in_flight` above the latencies, because those three decide whether
        the latencies below them mean anything.

        The three kinds of load are named apart on the first line. A burst is an
        open loop with no nominal rate, and printing it as "closed loop" because
        both leave `target_rate` empty would label the run that measures queueing
        as the run that cannot.
        """
        if self.concurrency is not None:
            offered = f"closed loop, {self.concurrency} clients"
        elif self.target_rate is None:
            offered = "burst (every request at once)"
        else:
            offered = f"{self.target_rate:.2f} req/s"
        lines = [
            f"  requests     {self.n_ok} ok, {self.n_failed} failed, over "
            f"{self.duration_s:.2f}s",
            f"  offered      {offered}",
            f"  achieved     {self.achieved_rate:.2f} req/s"
            + ("" if self.kept_up else "  (BEHIND: the queue was growing)"),
            f"  in flight    {self.mean_in_flight:.2f} requests on average",
            f"  send lag     {self.max_send_lag_s * 1e3:.1f} ms worst",
            f"  throughput   {self.output_tps:.1f} output tok/s ({self.output_tokens} tokens)",
            f"  TTFT         p50 {self.ttft_p50 * 1e3:.1f} ms  p90 "
            f"{self.ttft_p90 * 1e3:.1f} ms  p99 {self.ttft_p99 * 1e3:.1f} ms  "
            f"(n={self.n_ok})",
            f"  ITL          p50 {self.itl_p50 * 1e3:.1f} ms  p99 "
            f"{self.itl_p99 * 1e3:.1f} ms  (n={self.n_itl_samples} frame gaps)",
            f"  end to end   p50 {self.e2e_p50:.2f} s  p99 {self.e2e_p99:.2f} s",
        ]
        return "\n".join(lines)


# --- the checks a benchmark owes itself ----------------------------------------------


def check_offered_load(report: LoadReport, *, min_in_flight: float = 2.0) -> None:
    """Refuse a report that never put two requests in the server at once.

    Every latency number in this module is defined for a run of one, and all of
    them are wrong in the same direction: with no queue there is nothing for TTFT
    to include but prefill, and no batch for ITL to be degraded by. A run like that
    has measured the model, not the server, and it will pass any threshold a
    regression test sets. This is Day 41's admissibility check in the vocabulary of
    a benchmark rather than of a crowd.
    """
    if report.mean_in_flight < min_in_flight:
        raise MeasurementUnsound(
            f"this run held {report.mean_in_flight:.2f} requests on average, so it "
            f"measured a server serving one request at a time (wanted at least "
            f"{min_in_flight:.2f}). Its TTFT contains no queueing and its ITL contains "
            "no batching."
        )


def check_schedule_kept(report: LoadReport, *, max_lag_s: float = 0.05) -> None:
    """Refuse a report whose generator fell behind its own arrival schedule.

    A late send is coordinated omission arriving through the back door: the
    requests that should have piled up while the server was slow were not sent
    while the server was slow, so the queue they would have formed does not appear
    in anybody's TTFT. The failure is in the harness, and the symptom is a latency
    graph that looks better the more overloaded the server gets.

    Vacuous on a closed-loop report, where `scheduled_at` is `sent_at` by
    construction and the lag is exactly zero. That is not this function passing, it
    is the closed loop having no schedule to fall behind, which is the whole
    argument for driving an open one.
    """
    if report.max_send_lag_s > max_lag_s:
        raise MeasurementUnsound(
            f"the load generator was {report.max_send_lag_s * 1e3:.0f} ms behind its own "
            f"schedule (allowed {max_lag_s * 1e3:.0f} ms), so it stopped offering load "
            "exactly when the server got slow and the queueing it should have measured "
            "was never created"
        )


# --- arrival schedules ----------------------------------------------------------------


def _check_n(n: int) -> None:
    if n < 0:
        raise ValueError(f"a schedule cannot have a negative number of arrivals; got {n}")


def _check_rate(rate: float) -> None:
    if rate <= 0.0:
        raise ValueError(f"an arrival rate must be positive; got {rate}")


def burst_arrivals(n: int) -> list[float]:
    """Everybody at once. The cheapest way to guarantee a queue exists.

    Not a realistic arrival process and not meant to be: it is the shape that
    forces the scheduler to admit, queue and preempt in one run, so it is what the
    tests use to make sure the harness can see those things at all.
    """
    _check_n(n)
    return [0.0] * n


def fixed_arrivals(n: int, *, rate: float) -> list[float]:
    """Perfectly paced arrivals, `1/rate` apart, starting at zero.

    The gentlest possible load at a given rate, and that is its weakness: a server
    whose service time is below the interval never sees two requests at once, so a
    fixed schedule can be *under* capacity and still report a queue-free latency
    right up until the moment it saturates. Real traffic is bursty, which is what
    `poisson_arrivals` is for.
    """
    _check_n(n)
    _check_rate(rate)
    return [i / rate for i in range(n)]


def poisson_arrivals(n: int, *, rate: float, seed: int) -> list[float]:
    """Arrivals from a Poisson process: exponential gaps with mean `1/rate`.

    The standard model for independent callers, and the reason to prefer it over a
    fixed interval is that it queues at the same average rate. Exponential gaps are
    memoryless, so short gaps are common: requests land on top of each other even
    when the mean rate is well under capacity, and the resulting latency tail is
    the one production has.

    Seeded, not random. A benchmark that fails at one rate and passes at the same
    rate ten minutes later has told you nothing, and the variety wanted here is
    burstiness, not entropy.
    """
    _check_n(n)
    _check_rate(rate)
    rng = random.Random(seed)
    out: list[float] = []
    t = 0.0
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate)
    return out


# --- timing one request ---------------------------------------------------------------


async def time_stream(
    client: httpx.AsyncClient,
    plan: ClientPlan,
    *,
    scheduled_at: float | None = None,
    model: str = "nanoserve",
    clock: Clock = time.perf_counter,
) -> RequestRecord:
    """Send one streaming request and record when every frame arrived.

    Streaming only, and the exception is on purpose: a unary response arrives in
    one piece, so it can report an end-to-end time and nothing else. A benchmark
    that reported "latency" from the unary endpoint would be publishing a number
    with no TTFT and no cadence in it at all, and it would look excellent.

    `scheduled_at=None` means "this request was due when it was sent", which is the
    closed loop's answer and makes its send lag exactly 0.0 rather than the two
    microseconds between two clock reads. The difference is not precision, it is
    meaning: a closed loop has no schedule, and a lag of 0.0 is the correct way to
    say so.

    Never raises for anything the server or the socket did. A load run is a
    `gather` over dozens of these, and one client raising there cancels the rest,
    which turns "b7 got a 400" into a benchmark that produced no numbers.

    The terminal frame ends the run rather than counting as a token: it carries
    `finish_reason` and the usage bill, and the bill is where `output_tokens` comes
    from, because the frames a client counted are not the tokens the engine made.
    """
    if not plan.stream:
        raise ValueError(
            f"{plan.client_id} is not a streaming plan, and a unary response cannot be "
            "timed per token: it arrives in one piece, so its TTFT is its end-to-end "
            "time and its inter-token latency does not exist"
        )
    frames: list[float] = []
    prompt_tokens = 0
    output_tokens = 0
    done_at: float | None = None
    error: str | None = None
    status: int | None = None
    sent_at = clock()
    due = sent_at if scheduled_at is None else scheduled_at
    try:
        async with client.stream("POST", "/v1/completions", json=plan.body(model)) as response:
            status = response.status_code
            if status != 200:
                await response.aread()
                return RequestRecord(
                    client_id=plan.client_id,
                    scheduled_at=due,
                    sent_at=sent_at,
                    done_at=clock(),
                    status=status,
                    error=response.text,
                )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                now = clock()
                payload = json.loads(data)
                if "error" in payload:
                    error = payload["error"]["message"]
                    continue
                usage = payload.get("usage")
                if usage is not None:
                    prompt_tokens = usage["prompt_tokens"]
                    output_tokens = usage["completion_tokens"]
                if payload["choices"][0]["finish_reason"] is not None:
                    done_at = now
                    continue
                frames.append(now)
    except Exception as exc:  # noqa: BLE001 - a transport failure is this record's result
        return RequestRecord(
            client_id=plan.client_id,
            scheduled_at=due,
            sent_at=sent_at,
            frame_times=tuple(frames),
            done_at=clock(),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            status=status,
            error=f"{type(exc).__name__}: {exc}",
        )
    if done_at is None:
        # No terminal frame means the body stopped early. Recorded as an error
        # rather than as a shorter answer, so a truncation shows up as a failure
        # instead of quietly improving the latency numbers.
        error = error or "the stream ended without a final frame"
        done_at = clock()
    return RequestRecord(
        client_id=plan.client_id,
        scheduled_at=due,
        sent_at=sent_at,
        frame_times=tuple(frames),
        done_at=done_at,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        status=status,
        error=error,
    )


# --- driving a load -------------------------------------------------------------------


async def run_open_loop(
    base_url: str,
    plans: Sequence[ClientPlan],
    arrivals: Sequence[float],
    *,
    model: str = "nanoserve",
    target_rate: float | None = None,
    timeout: float = 120.0,
    clock: Clock = time.perf_counter,
) -> LoadReport:
    """Send each plan at its scheduled offset, whatever the server is doing.

    The loop does not wait for anything: a request due at t=0.4 goes out at t=0.4
    even if every earlier one is still in flight. That is the definition of open
    loop and it is the only way the queue gets to grow, which is the only way the
    latency knee gets measured.

    `target_rate` is recorded rather than derived, because the schedule is the
    input and a rate is a summary of it: a Poisson schedule at rate 10 has no
    constant interval to read back, and a burst has no rate at all. Pass the number
    that describes the schedule you built, or nothing.

    Every task is created up front and the sleeping happens inside them, so the
    generator's own lateness lands in `send_lag_s` where it can be checked, rather
    than being hidden inside a loop that sleeps between sends and therefore drifts
    by however long each send took.
    """
    if len(arrivals) != len(plans):
        raise ValueError(
            f"the schedule has {len(arrivals)} arrivals for {len(plans)} plans; every "
            "request needs a time it was due or its send lag is undefined"
        )
    for plan in plans:
        if not plan.stream:
            raise ValueError(
                f"{plan.client_id} is not a streaming plan; an open-loop latency run "
                "needs per-token arrival times"
            )

    started = clock()
    limits = httpx.Limits(max_connections=max(len(plans), 10) + 10)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits) as client:

        async def _fire(plan: ClientPlan, offset: float) -> RequestRecord:
            due = started + offset
            delay = due - clock()
            if delay > 0.0:
                await asyncio.sleep(delay)
            return await time_stream(
                client, plan, scheduled_at=due, model=model, clock=clock
            )

        tasks = [
            asyncio.create_task(_fire(plan, offset))
            for plan, offset in zip(plans, arrivals)
        ]
        try:
            records = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks, return_exceptions=True)
    return LoadReport(
        records=tuple(records),
        started_at=started,
        finished_at=clock(),
        target_rate=target_rate,
    )


async def run_closed_loop(
    base_url: str,
    plans: Sequence[ClientPlan],
    *,
    concurrency: int,
    model: str = "nanoserve",
    timeout: float = 120.0,
    clock: Clock = time.perf_counter,
) -> LoadReport:
    """Keep `concurrency` clients busy until every plan has been run once.

    Here for contrast, and because it is what most benchmarks are. It is the right
    driver for exactly one question, which is "what does this server do with
    exactly N concurrent users", and the wrong one for "what is the p99 at 20
    requests per second", because it cannot offer 20 requests per second to a
    server that can only do 12: it offers 12 and reports a healthy latency.

    `scheduled_at` is `sent_at` for every record, so `max_send_lag_s` is zero and
    `check_schedule_kept` passes trivially. That is the finding, not an oversight.
    """
    if concurrency < 1:
        raise ValueError(f"a closed loop needs at least one client; got {concurrency}")
    for plan in plans:
        if not plan.stream:
            raise ValueError(
                f"{plan.client_id} is not a streaming plan; a latency run needs "
                "per-token arrival times"
            )

    queue: asyncio.Queue[ClientPlan] = asyncio.Queue()
    for plan in plans:
        queue.put_nowait(plan)
    records: list[RequestRecord] = []
    started = clock()
    limits = httpx.Limits(max_connections=concurrency + 10)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits) as client:

        async def _worker() -> None:
            while True:
                try:
                    plan = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                records.append(await time_stream(client, plan, model=model, clock=clock))

        await asyncio.gather(*(_worker() for _ in range(concurrency)))
    return LoadReport(
        records=tuple(records),
        started_at=started,
        finished_at=clock(),
        concurrency=concurrency,
    )
