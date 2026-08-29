"""The decode step under a stopwatch, and where the seconds actually go. Day 46.

Week 12 measured this engine from the outside and drew the result. Day 42 swept the
offered rate through a socket, Day 43 split the server-side wait into five parts,
Day 44 put nanoserve next to HuggingFace `generate` and named the factor it was
losing on, and Day 45 drew the curve those settings trace out. Every one of those
days answered "how fast is it" and none of them answered "why", because none of them
ever looked inside a step. Week 13 is the inside, and this module is the instrument.

The unit is one iteration of `Engine.step`: schedule, sync the rows, build the input
tensors, forward, sample, collect. Each of those is a *phase*, and a phase has two
times that are not the same time.

**Host time is not device time, and the difference is the whole day.** A phase's
host time is how long Python was inside it. Its device time is how long the GPU
spent on the kernels that phase launched. On a decode step at batch 4 those numbers
are nothing like each other: the arithmetic is one token per row through a 1B model,
which is a handful of small matmuls, while the host has to run a scheduler, build
two tensors, dispatch a hundred-odd aten ops and read a token back per row. The
difference `host - device` is the *overhead*: the part of the step where the GPU had
nothing to do because Python had not told it anything yet. It is a bubble, and on a
small model at a small batch it is most of the step.

**That is the number `torch.compile` and CUDA graphs attack, and neither of them
touches the arithmetic.** A compiled decode step fuses the pointwise work and, more
importantly, replays a recorded launch sequence instead of walking the Python
dispatcher again, so `overhead` shrinks and `device` does not. Which means the
ceiling on the whole optimisation is known *before* you attempt it, from a profile
you already have: it is `host / device`, and `speedup_if(profile,
overhead_factor=inf)` is that division. If the answer is 1.1x, the week is over
before it starts and the real problem is elsewhere.

**The overlap model decides whether the overhead costs anything at all.** CUDA
launches are asynchronous. If nothing in the step reads a device tensor back to the
host, Python can run ahead while the GPU works through the queue, and the step takes
`max(host_overhead, device)`: overhead that fits underneath the kernels is free.
If something *does* read back, the host blocks until the queue drains, the two
cannot overlap, and the step takes `overhead + device`. Both models are here, and
choosing between them is not taste: `recommended_model` reads it off the phases,
because a phase that syncs is a fact about the code, not a modelling assumption.

nanoserve syncs. `Engine._sample` returns a `list[int]` and `_collect` calls
`int(token)`, so every single decode step drags the sampled tokens across the
PCIe bus and waits. That is why the serial model is the right one here, and it is
why per-step Python overhead lands on the critical path in full rather than hiding
under the forward. vLLM and SGLang both spend real engineering on exactly this
boundary (CUDA graph capture of the decode step, output handling moved off the
critical path), and the reason is visible in a profile of forty steps.

**Ranking hotspots by host time finds the wrong phase.** `forward` owns half the
host time on this box, and almost all of it is device time that no amount of Python
work will remove. `sample` owns a fifth of the host time and nearly none of the
device time, so it is a bigger *hotspot* despite being a smaller line in the table.
`hotspots` ranks by overhead for that reason: the question is not "where did the
seconds go" but "which seconds could go away".

**Three ways to get a believable profile that means nothing.** A profile that still
contains its first step is measuring lazy CUDA context creation, allocator growth
and autotuning, all of which happen once and none of which is the steady state;
`check_warmed_up` catches it. A profile whose phases do not add up to the steps has
a hole in it, and the hotspot list is then a list of the places somebody happened
to put a timer rather than a list of the expensive ones; `check_accounted` catches
that. And a profile whose host timings were taken around asynchronous launches
records the *launch* cost and pushes the real compute onto whichever later phase
happened to synchronise, which is how `forward` comes out at 0.1ms and `sample`
comes out at 30ms; `check_attributable` catches that one by arithmetic, since a
phase whose device time exceeds its host time is a phase the host did not wait for.

The last function here is the bridge back to Day 45. `project_point` takes a measured
`OperatingPoint` and a speedup and returns where that point would move, so this
week's before picture can be redrawn without pretending the new numbers were
measured. They were not. They are a ceiling, and labelled as one.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Protocol

from nanoserve.curve import OperatingPoint

#: Relative slack for comparing two measured durations. Everything in this module
#: is a subtraction over floats that came from a clock, so exact equality is never
#: the right test and any real difference is orders of magnitude above this.
_TOLERANCE_REL = 1e-9

#: Absolute floor for the same comparison, in seconds. Well under the resolution of
#: any clock a step is timed with.
_TOLERANCE_ABS = 1e-12

#: The two ways a step's host overhead and device time can combine into a wall.
#: `serial` is what you get when the step reads a device tensor back to the host;
#: `overlapped` is what you get when it does not and the launch queue runs ahead.
MODELS = ("serial", "overlapped")

#: How much larger than the median a first step may be before `check_warmed_up`
#: decides the warmup is still in the profile. Chosen loosely: a genuine warmup step
#: on this box is 10x to 100x the steady state, not 2.1x, so the gate is nowhere near
#: the noise it would have to distinguish itself from.
_WARMUP_FACTOR = 2.0

#: Minimum share of the measured wall that the phases have to claim before the
#: hotspot table is worth reading. Five percent of unattributed step is already
#: enough to hide a phase bigger than the third hotspot.
_MIN_ACCOUNTED = 0.95


class ProfileUnsound(AssertionError):
    """A set of step timings that cannot be read as a profile.

    The same family as Day 42's `MeasurementUnsound`, Day 44's `RunUnsound` and Day
    45's `CurveUnsound`, raised for the same kind of reason: not "the engine is
    slow" but "the table you are about to rank hotspots from does not say what it
    looks like it says". Every case here prints a perfectly plausible profile if it
    is allowed through, which is exactly why it is an assertion and not a warning.
    """


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_TOLERANCE_REL, abs_tol=_TOLERANCE_ABS)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# --- one phase of one step ------------------------------------------------------------


@dataclass(frozen=True)
class Phase:
    """One named region of a step, with the host time and the device time it owns.

    name:      what it is. `forward`, `sample`, `schedule`. These are the labels the
               hotspot table ranks, so they are the granularity at which the day can
               have an answer: a phase you did not name cannot be a hotspot.
    host_s:    seconds the host spent inside this region, wall clock.
    device_s:  seconds the device spent on the kernels this region launched. Zero
               for a region that launched none, which is most of them.
    syncs:     whether this region reads a device tensor back to the host. Not a
               timing: a fact about the code, and the one that decides which overlap
               model is allowed to be used on the step it belongs to.

    The invariant that makes the whole module work is that `host_s` *contains*
    `device_s`: the host waited for the kernels it launched before it stopped its
    timer. When that holds, `overhead_s` is the Python and driver cost of the phase
    and nothing else. When it does not, the phase is `hidden` and the timings are
    launch times, which is a different measurement wearing the same units.
    """

    name: str
    host_s: float
    device_s: float = 0.0
    syncs: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a phase needs a name; the hotspot table ranks by it")
        if self.host_s < 0.0:
            raise ValueError(f"{self.name}: host time cannot be negative; got {self.host_s}")
        if self.device_s < 0.0:
            raise ValueError(f"{self.name}: device time cannot be negative; got {self.device_s}")

    @property
    def overhead_s(self) -> float:
        """Host time this phase did not spend waiting for the device.

        Python, the dispatcher, the allocator, tensor construction, the scheduler's
        own bookkeeping. This is the quantity an optimisation like `torch.compile`
        or a captured CUDA graph removes, and the quantity a faster GPU does not.
        """
        return self.host_s - self.device_s

    @property
    def hidden(self) -> bool:
        """True when more device work was launched here than the host waited for.

        Which means this phase's host time is a launch time. Legitimate on an async
        stream and fatal to attribution, because the compute has not vanished: it
        will be paid by whichever later phase synchronises, and show up there.
        """
        return self.device_s > self.host_s and not _close(self.device_s, self.host_s)


@dataclass(frozen=True)
class PhaseTotal:
    """One phase summed across every step of a profile. Same fields, plural."""

    name: str
    host_s: float
    device_s: float
    steps: int

    @property
    def overhead_s(self) -> float:
        return self.host_s - self.device_s


@dataclass(frozen=True)
class Hotspot:
    """One phase's claim on the overhead, ranked.

    share:            this phase's overhead over the profile's total overhead.
    cumulative_share: this phase's share plus every share above it, so the top of
                      the list answers "how many phases do I have to fix to reach
                      80% of the removable time" without further arithmetic.
    per_step_s:       the overhead this phase costs on one step, which is the number
                      that is comparable across profiles of different lengths.
    """

    name: str
    host_s: float
    device_s: float
    overhead_s: float
    share: float
    cumulative_share: float
    per_step_s: float


# --- one step -------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSample:
    """One iteration of the engine loop, broken into phases.

    index:      which step this was, from the start of the recording. Kept so that
                dropping the warmup is visible in the data rather than implied by a
                shorter list.
    kind:       `decode` or `prefill`. They are wildly different shapes (one token
                per row against a padded rectangle of context) and averaging them
                together produces a step time that describes neither, which is what
                `select` exists to prevent.
    batch_size: rows in this step. One output token each, so it is also the token
                count, and it is the divisor that turns fixed per-step overhead into
                a per-token cost.
    phases:     in execution order. The order is content: the sequence *is* the step
                timeline, and `render` draws it as one.
    wall_s:     the whole step measured end to end, when it was measured separately.
                `None` means nobody measured it and the phases are taken to be the
                step, which is the honest default rather than a zero.
    """

    index: int
    kind: str
    batch_size: int
    phases: tuple[Phase, ...]
    wall_s: float | None = None

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError(f"step {self.index} has no phases; there is nothing to attribute")
        if self.batch_size <= 0:
            raise ValueError(f"step {self.index}: batch of {self.batch_size} is not a step")
        seen: set[str] = set()
        for phase in self.phases:
            if phase.name in seen:
                raise ValueError(
                    f"step {self.index}: phase {phase.name!r} appears twice; "
                    "sum it before recording it or the totals double-count"
                )
            seen.add(phase.name)
        if self.wall_s is not None and self.wall_s < self.host_s and not _close(
            self.wall_s, self.host_s
        ):
            raise ValueError(
                f"step {self.index}: phases sum to {self.host_s:.6f}s, which is more than the "
                f"step's own {self.wall_s:.6f}s; the phases overlap or the clocks disagree"
            )

    @property
    def host_s(self) -> float:
        return sum(p.host_s for p in self.phases)

    @property
    def device_s(self) -> float:
        return sum(p.device_s for p in self.phases)

    @property
    def overhead_s(self) -> float:
        """The bubble: host time on this step that the device did not spend working."""
        return self.host_s - self.device_s

    @property
    def wall(self) -> float:
        """The step's wall clock, falling back to the phases when it was not measured."""
        return self.host_s if self.wall_s is None else self.wall_s

    @property
    def unaccounted_s(self) -> float:
        """Wall the phases did not claim. Zero unless the step was timed separately."""
        return self.wall - self.host_s

    @property
    def device_utilisation(self) -> float:
        """Share of the host's time in this step that the device was busy for."""
        return 0.0 if self.host_s <= 0.0 else self.device_s / self.host_s

    @property
    def bound(self) -> str:
        """`overhead` when driving the GPU cost more than using it, else `compute`.

        Deliberately a majority vote and not a ratio with a threshold somebody
        picked: at exactly half and half neither name is more true, and the useful
        content is which of the two an optimisation should aim at.
        """
        return "overhead" if self.overhead_s > self.device_s else "compute"

    @property
    def sync_points(self) -> int:
        """How many phases in this step pull a device tensor back to the host."""
        return sum(1 for p in self.phases if p.syncs)


# --- many steps -----------------------------------------------------------------------


@dataclass(frozen=True)
class StepProfile:
    """A run of steps, with the warmup already off the front.

    name:     what to print, and what every gate's error message names.
    samples:  the retained steps, in order.
    dropped:  how many leading steps were discarded as warmup. Recorded rather than
              forgotten, because "40 steps" and "40 steps after dropping 5" are
              different claims and only one of them is checkable.
    """

    name: str
    samples: tuple[StepSample, ...] = ()
    dropped: int = 0

    @classmethod
    def from_samples(
        cls, name: str, samples: Iterable[StepSample], *, warmup: int = 0
    ) -> StepProfile:
        """Build a profile from recorded steps, discarding the first `warmup` of them.

        The first step of a torch program on CUDA is not a step. It creates the
        context, grows the caching allocator, picks algorithms for every convolution
        and matmul shape it has not seen, and on a compiled model it runs the
        compiler. Ten milliseconds of steady state hiding behind one 900ms first
        step is a mean that describes nothing, and the median hides it rather than
        fixing it, since the warmup step is real time that a benchmark of a long run
        should amortise and a profile of a step should not contain at all.
        """
        ordered = tuple(samples)
        if not ordered:
            raise ProfileUnsound(f"{name}: no steps to profile")
        if warmup < 0:
            raise ValueError(f"{name}: cannot drop {warmup} warmup steps")
        kept = ordered[warmup:]
        if not kept:
            raise ProfileUnsound(
                f"{name}: dropping {warmup} warmup steps of {len(ordered)} leaves nothing left"
            )
        return cls(name=name, samples=kept, dropped=warmup)

    # --- shape ------------------------------------------------------------------

    @property
    def steps(self) -> int:
        return len(self.samples)

    @property
    def tokens(self) -> int:
        """One output token per row per step, which is what a step is defined to emit."""
        return sum(s.batch_size for s in self.samples)

    def select(self, kind: str) -> StepProfile:
        """The sub-profile of one step kind.

        Prefill and decode are averaged together nowhere in this module, because a
        step that runs a 200-token context through a padded rectangle and a step that
        runs one token per row share a name and nothing else.
        """
        kept = tuple(s for s in self.samples if s.kind == kind)
        if not kept:
            raise ProfileUnsound(f"{self.name}: no {kind} steps in this profile")
        return StepProfile(name=f"{self.name}/{kind}", samples=kept, dropped=self.dropped)

    # --- totals -----------------------------------------------------------------

    @property
    def total_host_s(self) -> float:
        return sum(s.host_s for s in self.samples)

    @property
    def total_device_s(self) -> float:
        return sum(s.device_s for s in self.samples)

    @property
    def total_overhead_s(self) -> float:
        return self.total_host_s - self.total_device_s

    @property
    def total_wall_s(self) -> float:
        return sum(s.wall for s in self.samples)

    @property
    def mean_step_s(self) -> float:
        return self.total_wall_s / self.steps

    @property
    def steps_per_second(self) -> float:
        mean = self.mean_step_s
        return 0.0 if mean <= 0.0 else 1.0 / mean

    @property
    def tokens_per_second(self) -> float:
        """Output tokens per second this loop sustained, from the inside.

        Comparable to Day 44's `throughput_tps` for an offline run with no queue,
        and deliberately not the same measurement: that one is taken at the request
        boundary and contains everything this one leaves out.
        """
        return 0.0 if self.total_wall_s <= 0.0 else self.tokens / self.total_wall_s

    @property
    def idle_fraction(self) -> float:
        """Share of host time the device spent doing nothing.

        Also, exactly, the fraction Amdahl's law wants: it is the part of the step
        that a launch-overhead optimisation is allowed to attack, so
        `amdahl(idle_fraction, inf)` and `speedup_if(profile, overhead_factor=inf)`
        are the same number by two routes.
        """
        return 0.0 if self.total_host_s <= 0.0 else self.total_overhead_s / self.total_host_s

    @property
    def device_utilisation(self) -> float:
        return 1.0 - self.idle_fraction

    @property
    def bound(self) -> str:
        return "overhead" if self.total_overhead_s > self.total_device_s else "compute"

    @property
    def sync_points(self) -> int:
        """Sync points on a typical step. Zero is what makes overlap possible."""
        return max(s.sync_points for s in self.samples)

    @property
    def mean_batch_size(self) -> float:
        return self.tokens / self.steps

    @property
    def overhead_per_token_s(self) -> float:
        """Per-step overhead spread over the rows that shared it.

        The reason batching helps a small model at all. The scheduler runs once per
        step whether one row or thirty-two are resident, so its cost per token falls
        like 1/B while the arithmetic per token stays flat. It is the same fact Day
        44's slot sweep measured from the outside (each doubling bought less than a
        doubling) approached from the side where the constant is visible.
        """
        return 0.0 if self.tokens == 0 else self.total_overhead_s / self.tokens

    # --- attribution ------------------------------------------------------------

    def phase_totals(self) -> dict[str, PhaseTotal]:
        """Each phase summed over the steps, in the order the steps ran them."""
        host: dict[str, float] = {}
        device: dict[str, float] = {}
        counts: dict[str, int] = {}
        for sample in self.samples:
            for phase in sample.phases:
                host[phase.name] = host.get(phase.name, 0.0) + phase.host_s
                device[phase.name] = device.get(phase.name, 0.0) + phase.device_s
                counts[phase.name] = counts.get(phase.name, 0) + 1
        return {
            name: PhaseTotal(
                name=name, host_s=host[name], device_s=device[name], steps=counts[name]
            )
            for name in host
        }

    def hotspots(self, limit: int | None = None) -> tuple[Hotspot, ...]:
        """Phases ranked by removable time, largest first.

        By overhead and not by host time, which is the one design decision in this
        method and the one that changes the answer. `forward` is always the longest
        phase and is mostly device time that no Python optimisation reaches;
        `sample` is half its length and almost entirely host. Ranking by host time
        sends you to optimise a matmul; ranking by overhead sends you to the token
        readback, which is where the seconds actually are.

        Ties break on name, so two phases that cost the same come out in a stable
        order rather than in dictionary order.
        """
        totals = self.phase_totals()
        total_overhead = self.total_overhead_s
        ranked = sorted(totals.values(), key=lambda p: (-p.overhead_s, p.name))
        out: list[Hotspot] = []
        running = 0.0
        for entry in ranked:
            share = 0.0 if total_overhead <= 0.0 else entry.overhead_s / total_overhead
            running += share
            out.append(
                Hotspot(
                    name=entry.name,
                    host_s=entry.host_s,
                    device_s=entry.device_s,
                    overhead_s=entry.overhead_s,
                    share=share,
                    cumulative_share=running,
                    per_step_s=entry.overhead_s / self.steps,
                )
            )
        return tuple(out) if limit is None else tuple(out[:limit])


# --- the models -----------------------------------------------------------------------


def amdahl(fraction: float, factor: float) -> float:
    """Speedup of the whole when `fraction` of it is made `factor` times faster.

    `1 / ((1 - f) + f/k)`, and the reason it is here rather than done by hand is
    that it is the sanity check on every optimisation this week proposes. Its two
    limits are the useful ones. With `k` infinite the answer is `1/(1-f)`: the part
    you did not touch is a floor, and no amount of work on the rest goes under it.
    With `f` equal to 1 the answer is `k`, which is the only case where the headline
    speedup of a component is also the speedup of the program, and it never happens.

    An 80% Python-overhead step made infinitely cheap to launch is 5x, and that 5x
    is the ceiling on a week of work, available from a profile taken in an
    afternoon.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"a fraction of the runtime is between 0 and 1; got {fraction}")
    if factor <= 0.0:
        raise ValueError(f"a speedup factor is positive; got {factor}")
    if math.isinf(factor):
        return math.inf if _close(fraction, 1.0) else 1.0 / (1.0 - fraction)
    return 1.0 / ((1.0 - fraction) + fraction / factor)


def step_time_under(overhead_s: float, device_s: float, model: str = "serial") -> float:
    """How host overhead and device time combine into one step's wall.

    `serial` adds them: the host cannot run ahead because something in the step
    reads a device tensor back, so every microsecond of Python is a microsecond the
    GPU is idle and vice versa. This is nanoserve today, because sampling returns
    Python ints.

    `overlapped` takes the larger: launches are asynchronous, the host queues work
    and moves on, and whichever side is slower sets the pace. Overhead that fits
    under the kernels costs nothing at all in this model, which is why the same
    profile can support "we can be 2.2x faster" and "we can be 1.0x faster"
    depending on a fact about the code rather than about the timings.
    """
    if model not in MODELS:
        raise ValueError(f"unknown overlap model {model!r}; expected one of {MODELS}")
    if overhead_s < 0.0 or device_s < 0.0:
        raise ValueError(f"a step time is not negative; got {overhead_s} and {device_s}")
    return overhead_s + device_s if model == "serial" else max(overhead_s, device_s)


def recommended_model(profile: StepProfile) -> str:
    """Which overlap model this profile's own phases say applies.

    Read off `Phase.syncs`, so it is a statement about the code that was profiled
    and not a knob. One sync per step is enough to serialise the whole thing: the
    host stops at the readback until the queue drains, so the work it queued after
    the previous readback had one step's device time to hide under and no more.
    """
    return "serial" if profile.sync_points > 0 else "overlapped"


def speedup_if(
    profile: StepProfile,
    *,
    eliminate: Sequence[str] = (),
    overhead_factor: float = 1.0,
    device_factor: float = 1.0,
    model: str = "serial",
) -> float:
    """The step speedup an optimisation would buy, from the profile it starts from.

    eliminate:       phases whose *overhead* goes to zero. Their device time stays,
                     because deleting a phase's Python does not delete its kernels.
    overhead_factor: how much cheaper the remaining overhead gets. `inf` is the
                     ceiling: zero Python, zero dispatch, a captured graph replayed
                     by the driver.
    device_factor:   how much faster the kernels get. A different optimisation
                     entirely (a better kernel, a bigger tensor core), kept here so
                     the two can be compared on the same profile.
    model:           which overlap model to price it under. `check_model_applies`
                     is the guard against picking the flattering one.

    Denominated in host time rather than wall, so a profile with unattributed time
    in it gives an answer about the part that was attributed. `check_accounted` is
    what keeps that from mattering.
    """
    names = set(profile.phase_totals())
    missing = [n for n in eliminate if n not in names]
    if missing:
        raise ProfileUnsound(
            f"{profile.name}: {', '.join(sorted(missing))} not in this profile; "
            f"it has {', '.join(sorted(names))}"
        )
    if overhead_factor <= 0.0 or device_factor <= 0.0:
        raise ValueError(
            f"speedup factors are positive; got {overhead_factor} and {device_factor}"
        )

    totals = profile.phase_totals()
    kept_overhead = sum(t.overhead_s for name, t in totals.items() if name not in set(eliminate))
    before = step_time_under(profile.total_overhead_s, profile.total_device_s, model)
    after = step_time_under(
        0.0 if math.isinf(overhead_factor) else kept_overhead / overhead_factor,
        0.0 if math.isinf(device_factor) else profile.total_device_s / device_factor,
        model,
    )
    if after <= 0.0:
        return math.inf
    return before / after


#: The phases whose host time is arithmetic rather than driving. Everything else in
#: a step is the loop around the model, and the loop is what this week is about.
COMPUTE_PHASES = ("forward",)


def loop_overhead_s(profile: StepProfile, *, compute: Sequence[str] = COMPUTE_PHASES) -> float:
    """Per-step host time spent outside the named compute phases.

    The measurement that survives having no accelerator. Without a device there is
    no second clock, so `device_s` is zero everywhere and `overhead_s` is the whole
    step, which is useless. But the *loop* is still separable from the *arithmetic*
    by name: scheduling, syncing the block tables, building two small tensors,
    sampling and appending tokens are the same host work whatever the forward runs
    on, while `forward` is the part that a faster device makes faster.

    So a CPU profile can still answer the question this week actually asks, which is
    not "how long is my step" but "how much of my step would still be there if the
    arithmetic were free".
    """
    totals = profile.phase_totals()
    missing = [name for name in compute if name not in totals]
    if missing:
        raise ProfileUnsound(
            f"{profile.name}: {', '.join(missing)} is not a phase of this profile, so the "
            f"loop cannot be separated from the compute; it has {', '.join(totals)}"
        )
    outside = sum(t.host_s for name, t in totals.items() if name not in set(compute))
    return outside / profile.steps


def step_with_compute(
    profile: StepProfile, compute_s: float, *, compute: Sequence[str] = COMPUTE_PHASES
) -> float:
    """The step this loop would take if its forward cost `compute_s` instead.

    The transfer from the box that was profiled to the box that matters. A CPU
    forward of this model is tens of milliseconds and a GPU forward of it is a
    couple, and the loop around it does not get faster when the forward does: it is
    the same Python either way. Substituting one number and keeping the rest is
    therefore a much better estimate of a GPU step than scaling the whole CPU step
    down, which is the estimate that gets made by default and is wrong in the
    direction that hides the problem.
    """
    if compute_s < 0.0:
        raise ValueError(f"a compute time is not negative; got {compute_s}")
    return loop_overhead_s(profile, compute=compute) + compute_s


# --- the gates ------------------------------------------------------------------------


def check_accounted(profile: StepProfile, *, min_share: float = _MIN_ACCOUNTED) -> None:
    """Refuse a profile whose phases do not add up to its steps.

    A hotspot table is a claim that these phases are where the time went, and it is
    only that claim if the phases are all of the time. Unattributed seconds are not
    small by nature: the phase nobody wrapped is exactly the phase nobody was
    thinking about, which is a decent description of where a surprise hotspot lives.
    """
    wall = profile.total_wall_s
    if wall <= 0.0:
        raise ProfileUnsound(f"{profile.name}: the profile has no measured wall clock")
    share = profile.total_host_s / wall
    if share < min_share:
        raise ProfileUnsound(
            f"{profile.name}: phases account for {share:.1%} of the step "
            f"({wall - profile.total_host_s:.6f}s per run unaccounted); a hotspot table "
            "from this ranks the places a timer was put, not the expensive ones"
        )


def check_warmed_up(profile: StepProfile, *, factor: float = _WARMUP_FACTOR) -> None:
    """Refuse a profile whose first retained step still looks like a warmup.

    Compared against the median of the rest rather than the mean, so one genuinely
    slow step later in the run cannot mask the first one. Needs at least three steps
    to have a median worth the name, and says so instead of guessing.
    """
    if profile.steps < 3:
        raise ProfileUnsound(
            f"{profile.name}: too few steps ({profile.steps}) to tell a warmup from a step"
        )
    first = profile.samples[0].host_s
    rest = _median([s.host_s for s in profile.samples[1:]])
    if rest > 0.0 and first > factor * rest:
        raise ProfileUnsound(
            f"{profile.name}: the first step is {first / rest:.1f}x the median of the rest "
            f"({first * 1e3:.2f}ms against {rest * 1e3:.2f}ms), so the warmup is still in the "
            "profile; context creation, allocator growth and autotuning happen once"
        )


def check_attributable(profile: StepProfile) -> None:
    """Refuse a profile whose host timings do not contain the work they launched.

    Pure arithmetic, no flags: a phase whose device time exceeds its host time is a
    phase the host did not wait for, so its host time is a launch cost. The compute
    is still paid, by whichever later phase synchronises, and it shows up there. That
    is the mechanism behind every profile that reports a 0.1ms forward and a 30ms
    argmax, and it is a wrong *conclusion* drawn from correct measurements.
    """
    for sample in profile.samples:
        for phase in sample.phases:
            if phase.hidden:
                raise ProfileUnsound(
                    f"{profile.name}: step {sample.index} phase {phase.name!r} launched "
                    f"{phase.device_s * 1e3:.2f}ms of device work in {phase.host_s * 1e3:.2f}ms "
                    "of host time; the host did not wait for it, so this time is launch cost "
                    "and the compute will be charged to whatever synchronises next"
                )


def check_device_timed(profile: StepProfile) -> None:
    """Refuse to price an optimisation on a profile with no device times in it.

    Nothing here needs a GPU to *run*: the phase breakdown of a CPU step is real,
    useful and is where the per-step Python cost is visible. What a CPU profile
    cannot do is the arithmetic the rest of the module is for. With every
    `device_s` at zero, the whole step is "overhead" by definition, `idle_fraction`
    is 1.0, and `speedup_if(overhead_factor=inf)` comes out infinite, which is not a
    ceiling, it is a missing measurement wearing one. The matmuls did not stop being
    compute because nobody timed them separately.
    """
    if profile.total_device_s <= 0.0:
        raise ProfileUnsound(
            f"{profile.name}: no phase recorded any device time, so every second in this "
            "profile counts as overhead and the ceiling comes out infinite; rank the phases "
            "from it if you like, but do not price an optimisation on it"
        )


def check_model_applies(profile: StepProfile, model: str) -> None:
    """Refuse an overlap claim the profiled code cannot make.

    Only ever fires one way. `serial` is always safe: it is the pessimistic model
    and a step that could overlap and does not is merely slower than it needs to be.
    `overlapped` on a step that synchronises is the flattering error, and it makes
    launch overhead look free at exactly the moment somebody is deciding whether to
    spend a week removing it.
    """
    if model not in MODELS:
        raise ValueError(f"unknown overlap model {model!r}; expected one of {MODELS}")
    if model == "overlapped" and profile.sync_points > 0:
        offenders = sorted(
            {p.name for s in profile.samples for p in s.phases if p.syncs}
        )
        raise ProfileUnsound(
            f"{profile.name}: {', '.join(offenders)} synchronises every step, so the host "
            "cannot run ahead of the device and the overlapped model does not apply; "
            "price this under 'serial' or move the readback off the step"
        )


# --- drawing it -----------------------------------------------------------------------


def bar(overhead_s: float, device_s: float, *, per_cell_s: float, width: int = 40) -> str:
    """`#` per cell of host overhead then `=` per cell of device time.

    One rounding rule worth stating: any nonzero quantity gets at least one cell, so
    an empty stretch means exactly zero and never "too small to draw". A phase that
    costs 0.2ms next to a phase that costs 5ms should be visible as small rather
    than absent, because absent is what the reader will believe.
    """
    if per_cell_s <= 0.0:
        raise ValueError(f"per_cell_s has to be positive; got {per_cell_s}")
    if overhead_s < 0.0 or device_s < 0.0:
        raise ValueError(f"a bar is not drawn from negative seconds; got {overhead_s}, {device_s}")

    def cells(seconds: float) -> int:
        if seconds <= 0.0:
            return 0
        return max(1, int(round(seconds / per_cell_s)))

    out = "#" * cells(overhead_s) + "=" * cells(device_s)
    return out[:width]


def render(profile: StepProfile, *, width: int = 40, title: str | None = None) -> str:
    """The step as a table with a bar per phase, in execution order.

    Execution order rather than ranked order, because in that order the table *is*
    the step's timeline and the shape of the step is visible in the column of bars.
    `hotspots()` is the ranked view and it is a different question. `#` is host
    overhead and `=` is device time, so a row that is all hashes is a phase the GPU
    slept through.
    """
    totals = profile.phase_totals()
    if not totals:
        raise ProfileUnsound(f"{profile.name}: nothing to draw")
    steps = profile.steps
    per_cell = max(t.host_s for t in totals.values()) / steps / width

    lines: list[str] = []
    lines.append(title or f"{profile.name}: {profile.mean_step_s * 1e3:.3f} ms/step")
    lines.append(
        f"  {steps} steps ({profile.dropped} warmup dropped), "
        f"{profile.mean_batch_size:.1f} rows, "
        f"{profile.device_utilisation:.0%} device, "
        f"{profile.bound}-bound, {profile.sync_points} sync/step"
    )
    lines.append(f"  {'phase':<14}{'host ms':>9}{'device':>9}{'overhead':>10}  bar")
    for name, entry in totals.items():
        host = entry.host_s / steps
        device = entry.device_s / steps
        lines.append(
            f"  {name:<14}{host * 1e3:>9.3f}{device * 1e3:>9.3f}{(host - device) * 1e3:>10.3f}"
            f"  {bar(host - device, device, per_cell_s=per_cell, width=width)}"
        )
    lines.append(
        f"  {'total':<14}{profile.total_host_s / steps * 1e3:>9.3f}"
        f"{profile.total_device_s / steps * 1e3:>9.3f}"
        f"{profile.total_overhead_s / steps * 1e3:>10.3f}"
    )
    return "\n".join(line.rstrip() for line in lines)


# --- the bridge back to Day 45's curve ------------------------------------------------


def project_point(
    point: OperatingPoint,
    speedup: float,
    *,
    latency_factor: float | None = None,
    label: str | None = None,
) -> OperatingPoint:
    """Where a measured operating point would move if the step got `speedup` faster.

    A projection and not a measurement, which is why it takes the speedup as an
    argument instead of computing one: the number comes from `speedup_if`, which is
    a ceiling, and the point this returns inherits that. Redrawing Day 45's curve
    with these is the right way to see what a week of optimisation is worth *before*
    spending it, and the wrong thing to publish as a result.

    `latency_factor` defaults to the speedup, which is only right for a run with no
    queue in it: a faster step shortens a decode-bound latency proportionally, but
    an overloaded server's latency is mostly waiting, and a faster step there drains
    the queue instead, which moves latency by much more. Pass 1.0 to project the
    throughput and hold the latency still, which is the conservative reading.
    """
    if speedup <= 0.0:
        raise ValueError(f"a speedup is positive; got {speedup}")
    divisor = speedup if latency_factor is None else latency_factor
    if divisor <= 0.0:
        raise ValueError(f"a latency factor is positive; got {divisor}")
    return OperatingPoint(
        label=label or f"{point.label} (projected {speedup:.2f}x)",
        knob=point.knob,
        knob_value=point.knob_value,
        throughput_tps=point.throughput_tps * speedup,
        latency_s=point.latency_s / divisor,
    )


# --- collecting it from a live loop ---------------------------------------------------


class DeviceTimer(Protocol):
    """Whatever can say how long the accelerator spent on one bracketed region.

    Two methods, so that the module needs no `torch` import and the tests need no
    GPU. `start` marks the beginning of a region and `stop` returns its device time
    in seconds, having waited for it.
    """

    def start(self) -> None: ...

    def stop(self) -> float: ...


class CudaDeviceTimer:
    """A `DeviceTimer` built from a pair of CUDA events.

    Events are recorded *into the stream*, so what they measure is the interval on
    the device between the two markers, not on the host: the launch cost of the
    kernels in between is deliberately not in this number, and that is the entire
    reason it is a separate measurement from the clock around the same region.

    `stop` synchronises, which is the honest cost of the instrument. It has to: the
    elapsed time is not readable until the second event has actually happened, and
    without the wait the host clock around this region would return a launch time
    while the device clock returned a compute time, which is precisely the mismatch
    `check_attributable` exists to reject. So a profiled step is a serialised step,
    the profile's own step time is a little longer than the uninstrumented one, and
    the *shape* it reports is the thing to trust rather than its total.
    """

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start.record()

    def stop(self) -> float:
        self._end.record()
        self._end.synchronize()
        return self._start.elapsed_time(self._end) / 1e3


def device_timer_for(device: str) -> DeviceTimer | None:
    """A timer for this device, or `None` when the device has no separate clock.

    On CPU there is no second clock to read: the matmuls run on the same core the
    Python does, so "device time" is not a smaller quantity hiding inside the host
    time, it does not exist. Returning `None` rather than a zero timer is the point,
    because a zero would make every CPU phase look like pure overhead.
    """
    if not device.startswith("cuda"):
        return None
    return CudaDeviceTimer()



@dataclass
class _PhaseScope:
    """The handle a timed region hands back, so the caller can attach a device time.

    Mutable and short-lived on purpose. The host time is the recorder's business and
    the device time is the caller's, because only the caller knows which CUDA events
    bracketed the kernels, and there is no way to ask torch afterwards.
    """

    name: str
    device_s: float = 0.0
    syncs: bool = False


@dataclass
class _StepScope:
    """One step being recorded. Hands out phase scopes and collects their timings.

    `kind` and `batch_size` are mutable because the engine does not know either of
    them when the step starts: the scheduler decides during the first phase whether
    this iteration is a prefill or a decode and how many rows it carries, and the
    clock has to already be running by then or the schedule itself goes untimed.
    `describe` is how the caller fills them in once it knows.
    """

    kind: str
    batch_size: int
    clock: Callable[[], float]
    started_s: float
    phases: list[Phase] = field(default_factory=list)
    dropped: bool = False
    device_timer: DeviceTimer | None = None

    def describe(self, kind: str, batch_size: int) -> None:
        """Name this step, once the caller knows what it turned out to be."""
        self.kind = kind
        self.batch_size = batch_size

    def drop(self) -> None:
        """Discard this step instead of recording it.

        For the iteration where the scheduler admitted nothing: it carries no rows,
        emits no tokens and is not a step of the loop being profiled, so averaging it
        in would divide real work by a step count that includes idling.
        """
        self.dropped = True

    @contextmanager
    def phase(
        self, name: str, *, syncs: bool = False, device: bool = False
    ) -> Iterator[_PhaseScope]:
        """Time one region of the step.

        `device=True` brackets the region with the recorder's device timer, which is
        how a phase gets a device time at all. Without a timer installed it is a
        no-op and the phase records zero, which `check_device_timed` refuses to price
        anything from. A caller who has its own number can set `.device_s` on the
        yielded scope instead.
        """
        scope = _PhaseScope(name=name, syncs=syncs)
        timer = self.device_timer if device else None
        if timer is not None:
            timer.start()
        start = self.clock()
        try:
            yield scope
        finally:
            if timer is not None:
                scope.device_s = timer.stop()
            self.phases.append(
                Phase(
                    name=name,
                    host_s=self.clock() - start,
                    device_s=scope.device_s,
                    syncs=scope.syncs,
                )
            )


class StepRecorder:
    """Collects `StepSample`s from a running engine loop.

    The clock is injectable so the tests can advance it by hand and assert exact
    durations rather than ranges, which is the only way a timing module gets tested
    at all. `time.perf_counter` is the default and is the right one: it is monotonic
    and it counts the sleeping the host does while waiting on the device, which is
    precisely the quantity this module is about.

    The step's own wall is taken around the whole `with` block, so time inside the
    step that no phase claimed shows up as `unaccounted_s` instead of vanishing.
    That is what `check_accounted` reads, and it is the difference between a profile
    that adds up and one that merely looks like it does.

    A step that raises is dropped rather than recorded half-finished: a partial step
    would be a short one, and a short step in a profile pulls the mean the wrong way
    while looking like good news.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        device_timer: DeviceTimer | None = None,
    ) -> None:
        self.clock = clock
        self.device_timer = device_timer
        self._samples: list[StepSample] = []

    @property
    def samples(self) -> tuple[StepSample, ...]:
        return tuple(self._samples)

    @contextmanager
    def step(self, kind: str = "step", *, batch_size: int = 1) -> Iterator[_StepScope]:
        scope = _StepScope(
            kind=kind,
            batch_size=batch_size,
            clock=self.clock,
            started_s=self.clock(),
            device_timer=self.device_timer,
        )
        yield scope
        wall = self.clock() - scope.started_s
        if scope.dropped or not scope.phases:
            return
        self._samples.append(
            StepSample(
                index=len(self._samples),
                kind=scope.kind,
                batch_size=scope.batch_size,
                phases=tuple(scope.phases),
                wall_s=wall,
            )
        )

    def profile(self, name: str | None = None, *, warmup: int = 0) -> StepProfile:
        return StepProfile.from_samples(name or "steps", self._samples, warmup=warmup)


# --- the off switch -------------------------------------------------------------------


class _NullPhase:
    """A timed region that is not timed. Accepts a device time and forgets it."""

    __slots__ = ("device_s", "syncs")

    def __init__(self) -> None:
        self.device_s = 0.0
        self.syncs = False

    def __enter__(self) -> _NullPhase:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _NullStep:
    """A step that is not recorded. Every method is the shape of the real one."""

    __slots__ = ()

    def phase(self, name: str, *, syncs: bool = False, device: bool = False) -> _NullPhase:
        return _NULL_PHASE

    def describe(self, kind: str, batch_size: int) -> None:
        return None

    def drop(self) -> None:
        return None

    def __enter__(self) -> _NullStep:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class NullRecorder:
    """The recorder an engine carries when nobody is profiling it.

    Written as plain `__enter__`/`__exit__` classes rather than
    `contextlib.nullcontext` or a generator, because the engine enters seven of
    these per step and a generator-based context manager costs about a microsecond
    each while these cost about a tenth of that. It is a small number either way
    against a step measured in milliseconds, and it is the kind of small number this
    week exists to stop being casual about: instrumentation that changes the thing
    it measures is the oldest failure mode a profiler has.

    Singletons, so the whole engine allocates nothing per step when profiling is
    off. The phase object is shared and mutable, which is safe only because nothing
    ever reads `device_s` back off it, and that is exactly what "null" means here.
    """

    __slots__ = ()

    def step(self, kind: str = "step", *, batch_size: int = 1) -> _NullStep:
        return _NULL_STEP


_NULL_PHASE = _NullPhase()
_NULL_STEP = _NullStep()

#: The default an uninstrumented `Engine` holds. One object for the whole process.
NULL_RECORDER = NullRecorder()
