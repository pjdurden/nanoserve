"""Output handling: the sampled token, and the journey home. Week 13, Day 47.

Day 46 put a stopwatch inside one decode step and the shape of the answer was not
the one the code suggests. `forward` is the longest phase and almost none of it is
removable Python. `sample` is a fraction of its length and holds **86% to 89% of
the entire host loop** at every batch size measured. The reason is one line of
return type. `BatchedSampler.sample_batch` handed back `list[int]`, and every one
of those ints was its own `int(tensor)`: a value read out of device memory, one row
at a time.

On a CPU that is a memcpy and some dispatch. On a GPU it is the thing the whole
week is about. CUDA launches are **asynchronous**: the host queues kernels and runs
on, so Python that fits under the arithmetic costs nothing. Reading a value back
ends that, because the number does not exist until the kernels that produce it have
finished. So a readback is a **synchronisation**, and one per sampled row per step
means the host stops N times a step and every microsecond of the next step's launch
overhead lands on the critical path in full. That is exactly why Day 46's
`recommended_model` returns `serial` for this engine.

This module is about the **count**, and it is built on four ideas.

**A readback costs a journey, not a payload.** One step's tokens are `[rows]`
int64: 64 bytes at eight rows, on a bus that moves gigabytes a second. Essentially
all of the cost is fixed per call (synchronise, small DMA, resume), so `readback_s`
prices syncs and ignores bytes, and the quantity worth reducing is how many times
you go rather than how much you carry.

**Moving a synchronisation is not removing one.** After today the tokens come back
as a tensor and `sample` no longer syncs, but the engine still has to turn them
into ints to append them and apply a stop rule, so `collect` syncs instead. The
step still has exactly one. `syncs_per_step` names all three positions the engine
can be in (`per_row` before today, `one_transfer` after it, `deferred` next) and
the arithmetic prices them on the same footing, because the flattering version of
this day is to report the first change as if it were the second.

**The optimisation is worth nothing on the path the engine runs by default, and
that has to be said out loud.** `SamplingParams()` is greedy, and greedy rows were
*already* one batched `argmax` and one `.tolist()`. So for an all-greedy batch the
sync count was one before and is one after: `saving_per_step` returns zero and the
test that pins it is called
`test_an_all_greedy_batch_saves_nothing_by_batching_the_readback`. What today buys
on that path is Python, not synchronisations: no gather of `[rows, vocab]` floats,
no dict, no per-row loop. Two different savings that a benchmark will happily
conflate into one number.

**A CPU box cannot price any of it.** `check_measurable` refuses a `cpu` device for
the same reason Day 46's `check_device_timed` refuses a profile with no device
clock in it: the quantity being asked about does not exist here, and a plausible
number is worse than none.

vLLM's version of the next step is called async output processing: the tokens for
step N are resolved while step N+1 is already in flight, which costs one token of
overshoot past a stop condition and removes the last synchronisation from the
decode loop. `syncs_per_step(..., strategy="deferred", window=k)` is the arithmetic
of that, and it is arithmetic rather than code because the engine does not do it
yet.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

import torch

from .profiler import StepProfile, speedup_if

#: Where a step's readback can be. The three are ordered by how much of it is left.
#:
#:   per_row       Day 40's `sample_batch`: one `.tolist()` for the greedy block,
#:                 plus one `int(...)` per sampled row.
#:   one_transfer  Day 47: one `.tolist()` for the whole step, whatever it holds.
#:   deferred      one every `window` steps, resolved while the next step runs.
STRATEGIES = ("per_row", "one_transfer", "deferred")


class OutputUnsound(AssertionError):
    """A claim about the output path that the numbers do not support.

    An `AssertionError` for the same reason `ProfileUnsound` is: every one of these
    is a measurement that would otherwise be reported, and a wrong number that
    looks right is worse than a crash. The gates below raise it.
    """


# --- the instrument -------------------------------------------------------------


class Readback:
    """Counts the journeys home. The whole point of the day, made countable.

    A synchronisation leaves no trace in a return value. It shows up as the host
    waiting, and on a box with no accelerator it does not show up at all, so
    "the sampler no longer reads back" is not a claim any assertion about tokens
    can check. Routing every deliberate readback through one object turns it into
    an integer, which the gates read and the tests fail on.

    Two counters, because the cost model has two terms and only one of them
    matters. `transfers` is the number of times the host stopped; `elements` is how
    much it carried. At `[rows]` int64 the second is nothing, and keeping it
    anyway is what makes that visible rather than assumed.

    `__slots__` because the engine touches this once per step and Day 46 is the day
    this project stopped being casual about per-step allocation.
    """

    __slots__ = ("transfers", "elements")

    def __init__(self) -> None:
        self.transfers = 0
        self.elements = 0

    def tolist(self, tensor: torch.Tensor) -> list:
        """Bring a tensor home, and count it. The only sanctioned way to do that."""
        self.transfers += 1
        self.elements += tensor.numel()
        return tensor.tolist()

    def reset(self) -> None:
        """Start counting again. For a benchmark that discards its warmup."""
        self.transfers = 0
        self.elements = 0


# --- one step's tokens ----------------------------------------------------------


class TokenBatch:
    """One step's sampled tokens, still on the device, plus who owns each row.

    tokens:      `[rows]` integer tensor, one token id per row, in row order.
    request_ids: the request that owns each row, in the same order.

    The ids are the load-bearing half and the reason this is not just a tensor. A
    `[rows]` tensor is anonymous, and between sampling and collecting the engine
    releases finished rows and admits new ones; a row order that shifted in between
    hands one caller another caller's token, and the lengths still agree, so no
    shape check sees it. `OutputProcessor.apply` compares the ids and refuses.

    Resolution is lazy and memoised, which is the difference between one
    synchronisation per step and one per caller who looks. A plain `ids` property
    recomputed on access is the version of this that quietly costs a round trip
    every time somebody logs it.
    """

    __slots__ = ("tokens", "request_ids", "_ids")

    def __init__(self, tokens: torch.Tensor, request_ids: Sequence[str]) -> None:
        if tokens.dim() != 1:
            raise ValueError(
                f"a token batch is one dimension, [rows]; got shape {tuple(tokens.shape)}. "
                "`multinomial` returns [rows, 1] and that is not one token per row"
            )
        if tokens.is_floating_point() or tokens.is_complex():
            raise ValueError(f"a token id is an integer index; got dtype {tokens.dtype}")
        if tokens.numel() != len(request_ids):
            raise ValueError(
                f"{tokens.numel()} tokens for {len(request_ids)} rows: a row is a "
                "request, so they must line up"
            )
        self.tokens = tokens
        self.request_ids = tuple(request_ids)
        self._ids: list[int] | None = None

    @classmethod
    def empty(cls, device: torch.device | str | None = None) -> TokenBatch:
        """A step that sampled nobody. A prefill-only iteration still collects."""
        return cls(torch.empty(0, dtype=torch.long, device=device), ())

    @property
    def num_rows(self) -> int:
        return len(self.request_ids)

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def is_resolved(self) -> bool:
        """Whether the host has already paid for this batch."""
        return self._ids is not None

    def resolve(self, readback: Readback | None = None) -> list[int]:
        """The token ids as Python ints, at a cost of at most one transfer, ever.

        `readback` is the instrument to charge it to; without one the transfer still
        happens and simply goes uncounted, which is the right default for callers
        outside the engine loop.
        """
        if self._ids is None:
            self._ids = (
                readback.tolist(self.tokens) if readback is not None else self.tokens.tolist()
            )
        return self._ids

    def adopt_ids(self, ids: Sequence[int]) -> None:
        """Take resolved ids that came home in somebody else's transfer. Day 48.

        The one thing a batch cannot do for itself. Deferred output processing
        concatenates several steps' token tensors and brings the whole lot back in
        a single `.tolist()`, so each batch's slice of that list arrives from
        outside rather than from its own `resolve`. Handing it in here is what
        keeps `is_resolved` and the memoisation true afterwards: a batch that was
        filled this way must never pay for a second journey.

        Refuses a batch that already has ids, because that is the caller draining
        the same step twice, and refuses a wrong length, because a slice taken at
        the wrong offset hands every row after it another row's token and every
        answer stays grammatical.
        """
        if self._ids is not None:
            raise ValueError(
                f"this batch of {self.num_rows} rows is already resolved: adopting "
                "again means a step was drained twice"
            )
        if len(ids) != self.num_rows:
            raise ValueError(
                f"{len(ids)} ids for {self.num_rows} rows: a slice taken at the wrong "
                "offset hands every row after it somebody else's token"
            )
        self._ids = list(ids)


class OutputProcessor:
    """Turns a `TokenBatch` into appended tokens, once, and counts what it cost.

    The engine's whole collect phase, and the only place in the loop that is
    allowed to synchronise after today. Keeping it in one object is what makes
    `check_single_transfer` a real check: a run whose transfer count exceeds its
    step count has grown a second readback somewhere, which is what one helpful
    debug line does.

    It deliberately does not own the stop rules. `Request.append_token` applies
    those, exactly as it did on Day 30, because the alternative is a second copy of
    the state machine living in the fast path.
    """

    def __init__(self, readback: Readback | None = None) -> None:
        self.readback = readback if readback is not None else Readback()
        self.steps = 0
        self.tokens = 0

    @property
    def transfers(self) -> int:
        return self.readback.transfers

    @property
    def transfers_per_step(self) -> float:
        """The number this day exists to move. One, and one is the floor until the
        readback is deferred entirely."""
        return self.transfers / self.steps if self.steps else 0.0

    def apply(self, batch: TokenBatch, requests: Sequence) -> list[int]:
        """Resolve one step's tokens and hand each row's to its request.

        An empty batch is not a step: a prefill-only iteration reaches here with
        nothing to collect, and counting it would divide real transfers by a step
        count inflated with steps that never read anything back.
        """
        check_in_row_order(batch, requests)
        if batch.num_rows == 0:
            return []
        ids = batch.resolve(self.readback)
        for request, token in zip(requests, ids):
            request.append_token(token)
        self.steps += 1
        self.tokens += len(ids)
        return ids


# --- what a synchronisation costs, and how many there are -----------------------


def syncs_per_step(
    num_rows: int,
    *,
    num_sampled: int = 0,
    strategy: str = "one_transfer",
    window: int = 1,
) -> float:
    """How many times the host stops per step, under one of the three strategies.

    num_rows:    rows in the step's forward.
    num_sampled: how many of them are not greedy. Greedy rows are one batched
                 `argmax` and share a single readback; sampled rows each had their
                 own `int(multinomial(...))`.
    window:      for `deferred` only: how many steps one readback covers.

    A float, because `deferred` is a rate rather than a count. That is the honest
    unit: no individual step under deferral costs 0.125 synchronisations, but eight
    steps cost one.
    """
    if num_rows < 0:
        raise ValueError(f"a step has zero or more rows; got {num_rows}")
    if not 0 <= num_sampled <= num_rows:
        raise ValueError(
            f"{num_sampled} sampled rows out of {num_rows}: the sampled rows are a "
            "subset of the rows"
        )
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; the three are {STRATEGIES}")
    if window < 1:
        raise ValueError(f"a deferral window is at least one step; got {window}")
    if num_rows == 0:
        return 0.0
    if strategy == "per_row":
        # One journey for the greedy block, if there was one, plus one per sampled row.
        return float(num_sampled + (1 if num_rows > num_sampled else 0))
    if strategy == "one_transfer":
        return 1.0
    return 1.0 / window


def readback_s(syncs: float, *, latency_s: float) -> float:
    """What that many synchronisations cost, at a measured per-transfer latency.

    Deliberately not a function of the payload. One step's tokens are `[rows]`
    int64, which is 64 bytes at eight rows; the bandwidth term is five or six
    orders of magnitude below the fixed cost of stopping the host, draining a
    queue and starting it again. So the model is `count x latency`, and the
    optimisation it points at is fewer journeys rather than smaller ones.
    """
    if syncs < 0.0:
        raise ValueError(f"a synchronisation count is not negative; got {syncs}")
    if latency_s < 0.0:
        raise ValueError(f"a transfer latency is not negative; got {latency_s}")
    return syncs * latency_s


def saving_per_step(
    num_rows: int,
    *,
    num_sampled: int = 0,
    latency_s: float,
    before: str = "per_row",
    after: str = "one_transfer",
    window: int = 1,
) -> float:
    """Seconds per step that moving from one strategy to another gives back.

    Zero is a real answer and the one an all-greedy batch gets: the greedy path
    already read its block back in a single call, so batching the readback removes
    no synchronisation there at all. The Python it removes is a separate saving,
    measured in a profile rather than derived here, and adding the two together is
    the arithmetic that turns a modest day into an overclaim.
    """
    was = syncs_per_step(num_rows, num_sampled=num_sampled, strategy=before, window=window)
    now = syncs_per_step(num_rows, num_sampled=num_sampled, strategy=after, window=window)
    return readback_s(max(was - now, 0.0), latency_s=latency_s)


def strategy_speedup(step_s: float, saved_s: float) -> float:
    """The step speedup that saving `saved_s` out of a `step_s` step would be.

    One division, and it is worth doing before the work rather than after: a
    saving of 80us out of a 20ms step is 1.004x, which is a good reason to spend
    the day on something else.
    """
    if saved_s < 0.0:
        raise ValueError(f"a saving is not negative; got {saved_s}")
    if step_s <= 0.0:
        raise ValueError(f"a step takes time; got {step_s}")
    if saved_s >= step_s:
        raise ValueError(
            f"a saving of {saved_s:.6f}s out of a {step_s:.6f}s step is the whole step "
            "or more: that is arithmetic, not a measurement"
        )
    return step_s / (step_s - saved_s)


def sample_ceiling(profile: StepProfile, *, phase: str = "sample") -> float:
    """The most that removing a phase's host overhead could ever buy. Day 46's tool.

    The number Day 46 left as the one to beat, pointed at the phase this day is
    about. It is a ceiling and not a forecast: it prices the phase's Python going
    to *zero* while its kernels stay, which no change to a return type achieves.
    Comparing what actually landed against this is how the day gets an honest
    fraction rather than a speedup with no scale attached.
    """
    return speedup_if(profile, eliminate=(phase,))


# --- the gates ------------------------------------------------------------------


def check_single_transfer(processor: OutputProcessor, *, limit: float = 1.0) -> None:
    """Refuse a run that went home more than once a step.

    The regression this catches is not a rewrite, it is one line: a log statement,
    an assertion, a stray `int(...)` in a branch that only fires under preemption.
    The token values stay perfectly correct and the sync count doubles, which is
    invisible in every test that checks output.
    """
    if processor.steps == 0:
        return
    rate = processor.transfers_per_step
    if rate > limit:
        raise OutputUnsound(
            f"the output path read back {rate:.2f} times per step over "
            f"{processor.steps} steps, above {limit:.2f}: something outside "
            "`OutputProcessor.apply` is bringing tensors home"
        )


def check_device_resident(tokens, logits: torch.Tensor) -> None:
    """Refuse a sampler that came home. The day, as one type check and one compare.

    Two ways to fail. Returning a `list` is the old path, and it is the whole
    optimisation undone. Returning a tensor on a different device from the logits
    is the subtle one: a `.cpu()` added to make a print work synchronises exactly
    as hard as a `.tolist()` and leaves a tensor behind to look innocent.
    """
    if not isinstance(tokens, torch.Tensor):
        raise OutputUnsound(
            f"the sampler returned {type(tokens).__name__}, not a tensor: the tokens "
            "were read back to the host inside the sample phase"
        )
    if tokens.device != logits.device:
        raise OutputUnsound(
            f"the tokens are on {tokens.device} and the logits are on {logits.device}: "
            "something moved them off the device the forward ran on"
        )


def check_in_row_order(batch: TokenBatch, requests: Iterable) -> None:
    """Refuse a batch whose rows are not the requests about to be handed them.

    The failure mode is silent and it corrupts output rather than crashing: two
    rows swapped means two callers get each other's next token, and both answers
    stay grammatical.
    """
    ids = tuple(r.request_id for r in requests)
    if ids != batch.request_ids:
        raise OutputUnsound(
            f"row order changed between sampling and collecting: sampled "
            f"{batch.request_ids}, collecting {ids}"
        )


def check_measurable(device: torch.device | str) -> None:
    """Refuse to price a transfer from a box with nothing to transfer across.

    Day 46's `check_device_timed`, in the currency of synchronisations. On a CPU a
    `.tolist()` is a memcpy out of the same RAM the interpreter runs in: no bus, no
    queue to drain, no host that was ever running ahead of anything. A latency
    measured here is a real number about a different question, and quoting it as
    what a card would save is the mistake that makes an optimisation look free.
    """
    kind = torch.device(device).type
    if kind == "cpu":
        raise OutputUnsound(
            "there is no device on this box, so there is no readback to price: a "
            "`.tolist()` here is a memcpy and never a synchronisation. Measure the "
            "count on a CPU and the seconds on a card"
        )


# --- measuring it ---------------------------------------------------------------


def measure_transfer_s(
    device: torch.device | str = "cpu", *, rows: int = 8, repeats: int = 200
) -> float:
    """Median seconds for one `[rows]` int64 readback on this device.

    The median rather than the mean, for the reason every timing in this project
    takes the median: one descheduled iteration is a large positive outlier and
    there is no corresponding negative one, so a mean of a hundred transfers is a
    measurement of the operating system.

    On CUDA the first call also drains whatever the queue is holding, which is the
    honest thing to include: that wait is the cost of reading back, not an
    artefact of measuring it. A warmup iteration runs first so the cost being timed
    is a transfer rather than a context.
    """
    if repeats < 1:
        raise ValueError(f"a timing needs at least one repeat: repeats was {repeats}")
    tokens = torch.zeros(rows, dtype=torch.long, device=device)
    tokens.tolist()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        tokens.tolist()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


# --- the table ------------------------------------------------------------------


def render(
    num_rows: int,
    *,
    num_sampled: int = 0,
    latency_s: float,
    window: int = 8,
    step_s: float | None = None,
    title: str | None = None,
) -> str:
    """The three strategies side by side, in syncs and in seconds.

    `step_s` turns the seconds into a speedup, which is the only form of this that
    is safe to quote: 4 x 20us saved is a big number until you put it next to a
    20ms step, and then it is 1.004x.
    """
    lines = [title] if title else []
    header = f"{'strategy':<14}{'syncs/step':>12}{'seconds/step':>15}"
    if step_s is not None:
        header += f"{'speedup':>10}"
    lines.append(header)
    baseline = readback_s(
        syncs_per_step(num_rows, num_sampled=num_sampled, strategy="per_row"),
        latency_s=latency_s,
    )
    for strategy in STRATEGIES:
        syncs = syncs_per_step(
            num_rows, num_sampled=num_sampled, strategy=strategy, window=window
        )
        seconds = readback_s(syncs, latency_s=latency_s)
        row = f"{strategy:<14}{syncs:>12.2f}{seconds * 1e6:>13.1f}us"
        if step_s is not None:
            row += f"{strategy_speedup(step_s, baseline - seconds):>10.3f}x"
        lines.append(row)
    return "\n".join(lines)
