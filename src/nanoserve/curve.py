"""The latency-versus-throughput curve, and the knee as arithmetic. Day 45.

Week 12 measured four things and drew none of them. Day 42 swept the offered rate
through a socket and got a table, Day 43 split the server-side wait into five
parts, Day 44 put this engine next to `transformers` and swept the slot count, and
every one of those days ended with rows. A row per setting is the right thing to
store and the wrong thing to reason with, because the question a serving engine
actually gets asked is not "what is your throughput" but "how much throughput can
I have before the latency stops being acceptable", and that question is about the
*shape* the settings trace out, not about any one of them.

This module is that shape. One `OperatingPoint` per setting, two axes that are both
measurements, and four questions answered by comparison rather than by eye.

**The curve is parametric, and that is the whole reason it is interesting.** A
plot of y against x is a function: one y per x, and it goes left to right forever.
This is not that. Throughput is on x, latency is on y, *both are outputs*, and the
knob that produced them (offered requests per second, or the slot count) appears on
neither axis. So the curve is free to bend backwards, and a real one always does:
past capacity, turning the knob up buys latency and gives throughput back. Day 42's
sweep achieved 0.159 req/s at both 0.4 and 0.8 offered, with p50 TTFT climbing from
17.5s to 22.3s; Day 44's slot sweep peaked at 4 slots and lost throughput at 8.
Neither of those is a function of the offered load, and drawing them against the
offered load hides the only thing worth seeing.

**Dominance is the ruler.** Point `b` dominates point `a` when it delivers at least
as much throughput at no more latency, and strictly beats it on one of the two.
A dominated setting is not a tradeoff, it is a mistake: something else on the same
sweep was better on both axes at once, so there is no taste, budget or SLO under
which you would run it. The points with no dominator are the *frontier*, and they
are the entire menu. Everything else is the part of the sweep that found the wall.

**The knee is `argmax throughput / latency`.** That ratio is Kleinrock's power, and
maximising it is the standard parameter-free answer to "where does this curve
bend": it is the operating point where you are getting the most throughput per
second of latency you are paying for. Parameter-free matters, because the
alternative is a threshold ("the knee is where latency exceeds 2x the unloaded
value"), and a threshold is a number somebody picked that then decides the answer.
Power has a closed form to check against: for an M/M/1 queue with service rate mu,
throughput is lambda and latency is `1/(mu - lambda)`, so power is
`lambda(mu - lambda)` and is maximised at exactly half the service rate, a
utilisation of 0.5, whatever mu is. The test suite points the knee finder at that
curve, where the answer is known before the code runs.

Its units are tokens per second-squared, which is not a quantity anybody has an
intuition for, and that is fine: power is only ever compared within one curve, and
the comparison is what the knee is.

**An M/M/1 queue has no dominated points, and a real engine does.** M/M/1 latency
climbs to infinity but throughput never falls, so nothing is ever worse on both
axes and the frontier is the whole curve. Every backward bend in a measured curve
is therefore something the queueing model does not have: rows in a batch competing
for the same arithmetic (Day 44's second factor), a preempted sequence recomputing
its prefill (Day 34's), a KV pool that ran out. The dominated region is not noise
around the model, it is the part of the system the model leaves out.

**So the gate is the mirror image of the previous days'.** Day 42's
`check_offered_load` refuses a run that did not actually load the server and Day
44's `check_concurrent` refuses one that did not actually batch. Here,
`check_swept_to_saturation` refuses a *sweep* with no dominated point, because a
curve that never bent back is a sweep that stopped early, and its rightmost point
is not a capacity, it is just the largest number that was tried. Reporting it as a
capacity is the most common way a benchmark overstates a system, and it happens
without anybody typing a wrong number.

**Two rules about the axes, both of which are ways to lie with a true dataset.**
The x axis is *achieved* throughput and never the offered load: an overloaded
server keeps accepting a rate it is not serving, so plotting against what you asked
for stretches the curve out to the right and turns a wall into a slope. And both
axes start at zero, which `cell_for` enforces rather than accepts, because a
truncated axis makes a 5% difference occupy half the plot. Both of those produce a
picture that is drawn correctly from data that is measured correctly.

vLLM's and SGLang's capacity numbers are read off this curve: a latency SLO on the
y axis, and the throughput where the curve crosses it. `capacity_under_slo` is that
reading, and it is almost always to the left of the peak, which is why a peak
throughput and a served throughput are different claims.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Relative slack for calling two measured quantities equal. Throughputs here are
#: ratios of measured counts to measured seconds, so two settings that produced
#: literally the same number differ only in float rounding, while any real
#: difference between two runs is many orders of magnitude above this.
_TOLERANCE_REL = 1e-9

#: Absolute floor for the same comparison, so a pair of near-zero latencies does
#: not become "different" through relative tolerance alone.
_TOLERANCE_ABS = 1e-12

#: Markers `render` hands out, in order. `#` is missing on purpose: it is reserved
#: for the knee, and a curve whose marker collided with it would draw a knee at
#: every point.
_MARKERS = ("o", "x", "+", "*", "=", "%")

#: Which latency a `LoadReport` (Day 42) can be read on, and how.
_LOAD_LATENCIES = {
    "ttft_p50": lambda r: r.ttft_p50,
    "ttft_p90": lambda r: r.ttft_p90,
    "ttft_p99": lambda r: r.ttft_p99,
    "e2e_p50": lambda r: r.e2e_p50,
    "e2e_p99": lambda r: r.e2e_p99,
}

#: Which latency a `SystemRun` (Day 44) can be read on. An offline run has no
#: percentiles worth the name at eight requests, so these are the means, and the
#: name is different from the load report's on purpose: they are not comparable
#: and `check_same_axes` is what stops them being drawn as if they were.
_RUN_LATENCIES = {
    "mean_latency": lambda r: r.mean_latency_s,
    "mean_ttft": lambda r: r.mean_ttft_s,
}


class CurveUnsound(AssertionError):
    """A set of points that cannot be read as a tradeoff curve.

    The same family as Day 42's `MeasurementUnsound` and Day 44's `RunUnsound`,
    and raised for the same kind of reason: not "the system is slow" but "the
    picture you are about to draw does not mean what it looks like". Every case
    here renders a perfectly good plot if it is allowed through, which is exactly
    why it is an assertion and not a warning.
    """


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_TOLERANCE_REL, abs_tol=_TOLERANCE_ABS)


def _at_least(a: float, b: float) -> bool:
    return a > b or _close(a, b)


def _better(a: float, b: float) -> bool:
    return a > b and not _close(a, b)


# --- one setting, and what it produced ------------------------------------------------


@dataclass(frozen=True)
class OperatingPoint:
    """One setting of one knob, and the two numbers it produced.

    label:          what to print next to it. "0.4 rps", "4 slots".
    knob:           the name of the thing that was turned. Not an axis: it is the
                    hidden parameter of a parametric curve, and the reason the
                    curve may bend backwards.
    knob_value:     where it was turned to. Orders the curve, and nothing else.
    throughput_tps: output tokens per second the run *achieved*. Never the offered
                    load, which is a number that was typed rather than measured.
    latency_s:      the latency this curve is trading against, in seconds. Which
                    one it is lives on the curve rather than on the point, because
                    a point is only meaningful next to the other points measured
                    the same way.

    Both fields are outputs of the same run, so a point is a joint measurement and
    not two independent ones; that is what makes a pair of them comparable at all.
    """

    label: str
    knob: str
    knob_value: float
    throughput_tps: float
    latency_s: float

    def __post_init__(self) -> None:
        if self.throughput_tps < 0.0:
            raise ValueError(
                f"{self.label}: throughput cannot be negative; got {self.throughput_tps}"
            )
        if self.latency_s <= 0.0:
            raise ValueError(
                f"{self.label}: a latency of {self.latency_s} is not a measurement "
                "(a request that took no time did not happen)"
            )

    @property
    def power(self) -> float:
        """Throughput over latency: how much work per second of waiting.

        Kleinrock's power. Only comparable inside one curve, and the point that
        maximises it is the knee. It is worth understanding *why* the ratio finds
        the bend: on the flat part of the curve latency is roughly constant, so
        power rises with throughput; past the bend throughput is roughly constant,
        so power falls as latency climbs. The maximum is where those two regimes
        meet, which is what "the knee" has always informally meant.
        """
        return self.throughput_tps / self.latency_s


def dominates(a: OperatingPoint, b: OperatingPoint) -> bool:
    """True when `a` is at least as good on both axes and strictly better on one.

    The whole ordering this module has. Note that it is a *partial* order: two
    points where one is faster and the other snappier are simply incomparable, and
    that incomparability is the tradeoff the curve is named after. A total order
    would need a weight saying how many seconds of latency a token per second is
    worth, which is the caller's business and not a benchmark's.
    """
    no_worse = _at_least(a.throughput_tps, b.throughput_tps) and _at_least(
        b.latency_s, a.latency_s
    )
    strictly_better = _better(a.throughput_tps, b.throughput_tps) or _better(
        b.latency_s, a.latency_s
    )
    return no_worse and strictly_better


# --- what one step of the knob cost ---------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """The step between two adjacent settings, priced.

    The curve as a whole says where to sit; a segment says what the next notch is
    worth, which is the question you actually have while sweeping. Day 44 asked it
    in one dimension ("each doubling of the batch multiplies throughput by 1.35,
    then 1.19, then 0.90") and this is the same question with the latency the
    doubling cost attached to it.
    """

    lo: OperatingPoint
    hi: OperatingPoint

    @property
    def d_throughput(self) -> float:
        return self.hi.throughput_tps - self.lo.throughput_tps

    @property
    def d_latency(self) -> float:
        return self.hi.latency_s - self.lo.latency_s

    @property
    def backward(self) -> bool:
        """True when turning the knob up did not buy any throughput at all."""
        return not _better(self.hi.throughput_tps, self.lo.throughput_tps)

    @property
    def free(self) -> bool:
        """True when the next setting bought throughput *and* gave latency back.

        Exactly the case where `hi` dominates `lo`, said about the step rather than
        about the pair. It happens more often than a queueing intuition expects:
        every step of Day 44's slot sweep below the peak is one, because the burst
        arrives all at once and a request that is not resident is waiting, so a slot
        that lets it in removes queue time from both ends at once. A price in
        seconds per token per second is the wrong thing to print here, since the
        seconds went the other way.
        """
        return _better(self.hi.throughput_tps, self.lo.throughput_tps) and _at_least(
            self.lo.latency_s, self.hi.latency_s
        )

    @property
    def latency_per_tps(self) -> float:
        """Seconds of latency paid per extra token per second, or infinity.

        Infinity rather than a large number or a negative one, because a backward
        step did not have a price: no amount of latency buys throughput there, and
        the arithmetic of a ratio whose denominator went the wrong way produces a
        finite, plausible, meaningless figure. `backward` is the flag to read; this
        is the number to sort by, and infinity sorts where it belongs.
        """
        if self.backward:
            return math.inf
        return self.d_latency / self.d_throughput


# --- the curve ------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeoffCurve:
    """One knob swept, as a sequence of points on a shared pair of axes.

    name:         what to call it in a legend. "rate sweep", "slot sweep".
    points:       in knob order, which `from_points` guarantees.
    latency_name: which latency is on y. Carried by the curve rather than the
                  point because it is a property of the *experiment*, and because
                  `check_same_axes` needs somewhere to look before overlaying two
                  of these on one picture.
    marker:       the character `render` draws it with.

    Everything below is a comparison over `points`. There is no fitting, no
    smoothing and no interpolation anywhere in this module, deliberately: a knee
    read off a fitted curve is a knee of the fit, and with four to six settings in
    a sweep the fit has about as many parameters as it has data.
    """

    name: str
    points: tuple[OperatingPoint, ...]
    latency_name: str
    marker: str = "o"
    knob: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "knob", self.points[0].knob if self.points else "")

    @classmethod
    def from_points(
        cls,
        name: str,
        points: Iterable[OperatingPoint],
        *,
        latency_name: str,
        marker: str = "o",
    ) -> TradeoffCurve:
        """Sort by the knob and refuse the three sets of points that are not a sweep."""
        ordered = tuple(sorted(points, key=lambda p: p.knob_value))
        if len(ordered) < 2:
            raise CurveUnsound(
                f"{name}: a tradeoff curve needs at least two settings to trade "
                f"between; got {len(ordered)} (one point is a measurement, not a curve)"
            )
        knobs = {p.knob for p in ordered}
        if len(knobs) > 1:
            raise CurveUnsound(
                f"{name}: every point has to have turned the same knob; got "
                f"{sorted(knobs)}, which is two experiments drawn as one curve"
            )
        for before, after in zip(ordered, ordered[1:]):
            if _close(before.knob_value, after.knob_value):
                raise CurveUnsound(
                    f"{name}: {before.knob} = {before.knob_value:g} appears twice "
                    f"({before.label!r} and {after.label!r}), so this is one setting "
                    "measured repeatedly rather than a sweep"
                )
        for point in ordered:
            if not _better(point.throughput_tps, 0.0):
                raise CurveUnsound(
                    f"{name}: {point.label!r} has no throughput, so it is a failed run "
                    "rather than an operating point; drop it or explain it, but it "
                    "cannot sit on a tradeoff curve"
                )
        return cls(name=name, points=ordered, latency_name=latency_name, marker=marker)

    # --- the extremes ------------------------------------------------------------

    @property
    def max_throughput(self) -> float:
        return max(p.throughput_tps for p in self.points)

    @property
    def max_latency(self) -> float:
        return max(p.latency_s for p in self.points)

    @property
    def peak(self) -> OperatingPoint:
        """The setting that produced the most throughput: this box's capacity.

        Ties go to the lower knob setting, which is the conservative reading: if two
        settings produced the same throughput, the one doing less work for it is the
        one that found the capacity.
        """
        best = self.points[0]
        for point in self.points[1:]:
            if _better(point.throughput_tps, best.throughput_tps):
                best = point
        return best

    @property
    def knee(self) -> OperatingPoint:
        """The setting that maximises throughput per second of latency.

        Ties again go to the lower setting. Provably on the frontier: if some `b`
        dominated the knee then `b` would have at least as much throughput over no
        more latency with one of the two strict, and its power would therefore be
        strictly greater, contradicting the maximum. The suite asserts it anyway,
        because a proof about the definition is not a proof about the code.
        """
        best = self.points[0]
        for point in self.points[1:]:
            if _better(point.power, best.power):
                best = point
        return best

    # --- the menu ----------------------------------------------------------------

    @property
    def frontier(self) -> tuple[OperatingPoint, ...]:
        """The settings nothing else beats on both axes: every choice worth making."""
        return tuple(
            p for p in self.points if not any(dominates(other, p) for other in self.points)
        )

    @property
    def dominated(self) -> tuple[OperatingPoint, ...]:
        """The settings that were worse than another one on both axes at once.

        Not a tradeoff and not a preference: a strictly wasted run. Every point is
        in exactly one of `frontier` and this, since "has a dominator" is the
        negation of the frontier's test.
        """
        return tuple(p for p in self.points if any(dominates(other, p) for other in self.points))

    @property
    def found_the_wall(self) -> bool:
        """True when at least one setting was dominated: the sweep went far enough.

        The property `check_swept_to_saturation` gates on. A false here does not
        mean the system has no wall, it means this sweep did not reach it, and the
        two are indistinguishable from the numbers, which is the point.
        """
        return bool(self.dominated)

    @property
    def knee_bracketed(self) -> bool:
        """True when the knee has a measured setting on either side of it.

        The low-end twin of `found_the_wall`, and just as easy to miss. Power rises
        to the knee and falls after it, so a knee sitting on the first or last point
        of a sweep is not a maximum that was found, it is the edge the sweep stopped
        at: the real one is somewhere outside, and all the curve can honestly say is
        "at least this far". Both of Week 12's sweeps fail this, in opposite
        directions, which is a fact about how they were parameterised and not about
        the engine.
        """
        return self.knee is not self.points[0] and self.knee is not self.points[-1]

    @property
    def segments(self) -> tuple[Segment, ...]:
        return tuple(Segment(lo, hi) for lo, hi in zip(self.points, self.points[1:]))

    def summary(self) -> str:
        """The curve as a block of text, in the shape Day 42's `LoadReport` prints."""
        knee, peak = self.knee, self.peak
        lines = [
            f"  {self.name}  ({len(self.points)} settings of {self.knob}, "
            f"y = {self.latency_name})",
            f"    knee      {knee.label} at {knee.throughput_tps:.2f} tok/s, "
            f"{knee.latency_s:.2f}s (power {knee.power:.3f})",
            f"    peak      {peak.label} at {peak.throughput_tps:.2f} tok/s, "
            f"{peak.latency_s:.2f}s",
        ]
        if not self.knee_bracketed:
            edge, beyond = ("first", "below") if knee is self.points[0] else ("last", "above")
            lines.append(
                f"    edge      the knee is the {edge} setting swept, so the real one is "
                f"{beyond} it and was never measured"
            )
        if self.found_the_wall:
            worse = ", ".join(p.label for p in self.dominated)
            lines.append(
                f"    wall      {len(self.dominated)} of {len(self.points)} settings "
                f"dominated ({worse})"
            )
        else:
            lines.append(
                "    wall      no wall: nothing was dominated, so this sweep stopped "
                "early and the peak above is not a capacity"
            )
        for seg in self.segments:
            if seg.backward:
                price = "backward"
            elif seg.free:
                price = "free (latency fell too)"
            else:
                price = f"{seg.latency_per_tps:.2f} s per tok/s"
            lines.append(
                f"    {seg.lo.label:>9} -> {seg.hi.label:<9} "
                f"{seg.d_throughput:+.2f} tok/s for {seg.d_latency:+.2f}s   {price}"
            )
        return "\n".join(lines)


# --- the questions asked of a curve ----------------------------------------------------


def capacity_under_slo(curve: TradeoffCurve, slo_s: float) -> OperatingPoint | None:
    """The most throughput available without breaking a latency budget, or None.

    The number capacity planning actually wants, and it is not the peak. The peak
    is what the box can do with every caller waiting as long as it takes; this is
    what the box can do while keeping a promise. On any real curve the two are
    different settings, and quoting the first while advertising the second is the
    standard way a serving benchmark ends up unreproducible in production.

    Ties on throughput go to the lower latency, which also makes the answer always
    a frontier point: if a dominated point had the most throughput under the
    budget, its dominator has at least as much at no more latency, so it meets the
    budget too and is returned instead.

    None when nothing meets the budget. Not an exception: "this box cannot serve
    that SLO at any setting" is a legitimate and important result of a sweep, and
    the caller should print it rather than crash on it.
    """
    eligible = [p for p in curve.points if _at_least(slo_s, p.latency_s)]
    if not eligible:
        return None
    best = eligible[0]
    for point in eligible[1:]:
        if _better(point.throughput_tps, best.throughput_tps):
            best = point
        elif _close(point.throughput_tps, best.throughput_tps) and _better(
            best.latency_s, point.latency_s
        ):
            best = point
    return best


def check_swept_to_saturation(curve: TradeoffCurve) -> None:
    """Refuse a sweep in which no setting was ever worse than another on both axes.

    Day 42 refuses a load test that did not load, Day 44 refuses a batch test that
    did not batch, and this refuses a capacity sweep that did not find a capacity.
    The failure is quiet in exactly the same way: the run completes, the table is
    full, the last row has the biggest throughput in it, and somebody writes it
    down as the maximum. All that row actually says is that the sweep ran out of
    settings before the machine ran out of headroom.

    The fix is never in the code under test: widen the sweep until it bends.
    """
    if not curve.found_the_wall:
        raise CurveUnsound(
            f"{curve.name}: the curve never bent back, so no setting was dominated and "
            f"this sweep never found the wall; the peak ({curve.peak.label}, "
            f"{curve.peak.throughput_tps:.2f} tok/s) is the largest {curve.knob} that was "
            "tried, not a capacity"
        )


def check_same_axes(curves: Sequence[TradeoffCurve]) -> None:
    """Refuse to put two curves measuring different latencies on one pair of axes.

    The subtlest error this module can make, because it produces a picture that is
    beautiful and internally consistent. A TTFT of 3s and an end-to-end of 3s are
    the same height on the plot and mean entirely different things, and the reader
    has no way to tell, since the y axis can only carry one label. Two knobs on one
    pair of axes is fine and is the point of the day; two *quantities* is not.
    """
    names = {c.latency_name for c in curves}
    if len(names) > 1:
        raise CurveUnsound(
            "these curves do not share a y axis: "
            + ", ".join(f"{c.name} measures {c.latency_name}" for c in curves)
            + "; two latencies drawn at the same height are not comparable"
        )


# --- from the previous days' reports ---------------------------------------------------


def point_from_load_report(
    report: Any, *, latency: str = "ttft_p50", label: str | None = None
) -> OperatingPoint:
    """One point from Day 42's open-loop `LoadReport`: offered rate as the knob.

    The throughput is `output_tps`, which is *achieved*: tokens that came back over
    the window. The knob is the offered rate, which is the number that was asked
    for. Keeping those apart is the whole reason this adapter exists, because the
    tempting thing is to plot latency against the offered rate and call it a curve,
    and past capacity the offered rate keeps rising while the achieved one does
    not. Day 42 offered 0.4 and 0.8 req/s and got 0.159 both times.

    A closed-loop report has no offered rate at all: its load is a client count and
    its arrival process is "whenever the last one finished", so it cannot be a
    point on this axis and is refused rather than defaulted to zero.
    """
    if latency not in _LOAD_LATENCIES:
        raise CurveUnsound(
            f"unknown latency {latency!r}; a load report can be read on "
            + ", ".join(sorted(_LOAD_LATENCIES))
        )
    rate = report.offered_rate
    if rate is None:
        raise CurveUnsound(
            "this report has no offered rate to use as a knob (a closed-loop or burst "
            "run varies its client count, not its arrival rate, so it belongs on a "
            "different sweep)"
        )
    return OperatingPoint(
        label=label or f"{rate:g} rps",
        knob="offered_rps",
        knob_value=rate,
        throughput_tps=report.output_tps,
        latency_s=_LOAD_LATENCIES[latency](report),
    )


def point_from_run(
    report: Any,
    *,
    knob_value: float,
    knob: str = "slots",
    latency: str = "mean_latency",
    label: str | None = None,
) -> OperatingPoint:
    """One point from Day 44's offline `SystemRun`: the slot count as the knob.

    The other half of "one pair of axes". An offline sweep hands over every prompt
    at once and varies how many may be resident, so its knob is a slot count and
    its latencies are means over a handful of requests rather than percentiles over
    a stream. Both sweeps still produce a throughput in output tokens per second
    and a latency in seconds, which is exactly enough to draw them together, and
    `latency_name` on the curve is what stops that from being an accident.

    Worth saying plainly, because the plot makes it obvious and a table never
    would: the two sweeps trace the axes in *opposite* directions when TTFT is on
    y. More offered load fills the queue and TTFT climbs; more slots drain the same
    burst faster and TTFT falls. Same axes, same engine, opposite slope, and
    neither is wrong.
    """
    if latency not in _RUN_LATENCIES:
        raise CurveUnsound(
            f"unknown latency {latency!r}; a system run can be read on "
            + ", ".join(sorted(_RUN_LATENCIES))
        )
    return OperatingPoint(
        label=label or f"{knob_value:g} {knob}",
        knob=knob,
        knob_value=knob_value,
        throughput_tps=report.throughput_tps,
        latency_s=_RUN_LATENCIES[latency](report),
    )


# --- the picture ------------------------------------------------------------------------


def cell_for(
    point: OperatingPoint, *, width: int, height: int, max_tps: float, max_latency: float
) -> tuple[int, int]:
    """Which (row, col) of a `height` x `width` grid a point lands in. Row 0 is the top.

    Both axes run from zero to the maximum, and the zero is not negotiable. A plot
    whose x axis starts at 2.4 because the smallest measurement was 2.44 turns the
    4% between two settings into half the width of the picture, and every reader
    who does not check the tick labels reads it as a large effect. Since this
    module exists to make a table legible, a legible wrong reading is the worst
    thing it could produce, so the origin is built into the arithmetic rather than
    left as a flag somebody remembers to set.
    """
    if width < 2 or height < 2:
        raise ValueError(f"a plot needs at least 2x2 cells; got {width}x{height}")
    if max_tps <= 0.0 or max_latency <= 0.0:
        raise ValueError(f"the axes need a positive extent; got {max_tps} by {max_latency}")
    col = round(point.throughput_tps / max_tps * (width - 1))
    row = (height - 1) - round(point.latency_s / max_latency * (height - 1))
    return (max(0, min(height - 1, row)), max(0, min(width - 1, col)))


def _assign_markers(curves: Sequence[TradeoffCurve]) -> list[str]:
    """Each curve's own marker where it is free, the next unused one where it is not."""
    used: set[str] = set()
    markers: list[str] = []
    for curve in curves:
        marker = curve.marker
        if marker in used or marker == "#":
            marker = next((m for m in _MARKERS if m not in used), "?")
        used.add(marker)
        markers.append(marker)
    return markers


def render(
    curves: Sequence[TradeoffCurve],
    *,
    width: int = 64,
    height: int = 18,
    title: str | None = None,
    slo_s: float | None = None,
) -> str:
    """The curves as ASCII, on one pair of axes, with each knee marked `#`.

    A terminal plot rather than a PNG on purpose: it goes in the same place the
    numbers go, it needs no plotting dependency, and every cell it draws is
    `cell_for` arithmetic the suite can assert on directly. The picture is the
    argument the day is making, so it should be as testable as the numbers.

    The SLO line, when given, is the reading the plot exists for: everything below
    it meets the budget, and the rightmost marker under the line is
    `capacity_under_slo`.
    """
    if not curves:
        raise CurveUnsound("no curves to draw")
    check_same_axes(curves)

    max_tps = max(p.throughput_tps for c in curves for p in c.points)
    max_latency = max(p.latency_s for c in curves for p in c.points)
    if slo_s is not None:
        max_latency = max(max_latency, slo_s)

    grid = [[" "] * width for _ in range(height)]
    if slo_s is not None:
        row = (height - 1) - round(slo_s / max_latency * (height - 1))
        row = max(0, min(height - 1, row))
        for col in range(0, width, 2):
            grid[row][col] = "-"

    markers = _assign_markers(curves)
    for curve, marker in zip(curves, markers):
        knee = curve.knee
        for point in curve.points:
            row, col = cell_for(
                point, width=width, height=height, max_tps=max_tps, max_latency=max_latency
            )
            grid[row][col] = "#" if point is knee else marker

    pad = " " * 9
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append(f"{pad} {curves[0].latency_name} (s)")
    for i, row_cells in enumerate(grid):
        if i in (0, (height - 1) // 2, height - 1):
            value = max_latency * (height - 1 - i) / (height - 1)
            gutter = f"{value:8.2f} |"
        else:
            gutter = f"{pad}|"
        lines.append(gutter + "".join(row_cells))
    lines.append(f"{pad}+" + "-" * width)
    left = f"{0.0:.2f}"
    right = f"{max_tps:.2f}"
    lines.append(f"{pad} {left}{' ' * max(1, width - len(left) - len(right))}{right}")
    lines.append(f"{pad} {'output tok/s'.center(width)}")

    legend = "  ".join(
        f"{marker} {curve.name} (knee {curve.knee.label})"
        for curve, marker in zip(curves, markers)
    )
    lines.append(f"{pad} {legend}   # knee")
    if slo_s is not None:
        lines.append(f"{pad} - slo {slo_s:.2f}s {curves[0].latency_name}")
    return "\n".join(line.rstrip() for line in lines)
