"""Deferred output processing: one journey home for k steps. Week 13, Day 48.

Day 47 said the honest thing about itself. It moved the decode step's readback
from `sample` to `collect`; it did not remove one. The host still stops once a
step, because `Request.append_token` applies the stop rules and a stop rule needs
a Python int, and that int does not exist until the kernels that produced it have
finished. `sync_points` stayed at 1 and `recommended_model` stayed `serial`.

There is exactly one way to stop fewer times, and it is not to stop faster. It is
to **stop later**: keep step N's tokens on the device, launch step N+1 out of that
same tensor, and bring several steps home in one journey. vLLM calls this async
output processing. Three separate things fall out of it and this module keeps them
apart, because a benchmark that adds them together is the reason this day would
otherwise look twice as good as it is.

**A window of one changes the input, not the count.** With one step of lag the
newest batch is still on the device when the next decode builds its inputs, and
that batch *is* the input: `[rows]` unsqueezed to `[rows, 1]`, a view, no copy, no
`torch.tensor([[r.output_token_ids[-1]] ...])` built out of Python lists and
pushed across the bus every step. The transfer count is unchanged at one per step.
It has to be said out loud, because "deferred" sounds like "removed" and on a
single stream it is not: `Tensor.tolist` issues its copy on the current stream and
waits for everything queued ahead of it, so a resolve placed after the next
forward waits for that forward too. Hiding a synchronisation behind a kernel needs
a second stream and an event, and this engine has one stream.

**A window of k removes k-1 journeys, and that part is real on any stream.** k
held batches concatenate with `torch.cat` (a launch, not a synchronisation) and go
home in a single `.tolist()`. `syncs_per_step(..., strategy="deferred",
window=k)` was written on Day 47 as arithmetic for something the engine did not do;
`DeferredOutputProcessor` is that arithmetic made true, and `transfers_per_step`
is it measured rather than derived.

**The price is rows nobody keeps.** A request whose stop token is still on the
device is still, as far as the scheduler can tell, running: it stays in the batch
and the engine forwards a row for it that will be thrown away. That is
`waste_fraction`, the Day-29 number continuous batching drove to zero, coming back
off zero by up to `window` rows per finished request. So the window has a best
value rather than a large one, and `best_window` is where the two sides meet.

One structural hazard comes with all of it and it is not about tokens.
`request.num_tokens` lags the cache row's length by however many tokens are still
held, so `Scheduler.blocks_needed_for` would top a running request up to blocks it
has already outgrown. `BlockTable.append` does not raise on that: it reaches the
allocator itself, the pool is booked twice for one sequence, and the release frees
only the blocks the request knows about. The fix is `Scheduler(lookahead=k)`,
which buys the headroom at the top of the iteration where every other block in
this engine is bought.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch

from .output import (
    OutputProcessor,
    OutputUnsound,
    Readback,
    TokenBatch,
    check_in_row_order,
    readback_s,
    syncs_per_step,
)
from .scheduler import RequestState

#: Windows a sweep walks by default. Powers of two because the saving is `1 - 1/k`
#: and everything interesting in it happens in the first few doublings.
WINDOWS = (1, 2, 4, 8, 16, 32)


# --- what a window costs, in rows ------------------------------------------------


def overshoot_bound(window: int) -> int:
    """Rows a finished request costs past its stop, worst case.

    A batch is drained at most `window` steps after it was sampled, and every step
    in between forwards a row for a request that has already emitted its stop
    token. So the worst case is the whole window: a stop that lands in the oldest
    of k batches going home together is seen k steps late.

    Note that it is not zero at `window=1`. One step of lag still means the stop is
    read after the next forward has been launched, which is the whole idea, and
    that forward has a row in it for a request that was already done.
    """
    if window < 1:
        raise ValueError(f"a deferral window is at least one step; got {window}")
    return window


def expected_overshoot(window: int) -> float:
    """Rows a finished request costs on average: `(window + 1) / 2`.

    The stop token is equally likely to land in any of the `window` batches that go
    home in one journey. Landing in the oldest costs `window` rows; landing in the
    newest costs one, because even then the next step has already been launched.
    The mean of 1..window is `(window + 1) / 2`.

    A float and an average, which is the right unit: no individual request
    overshoots by 4.5 rows, and a hundred of them cost 450.
    """
    if window < 1:
        raise ValueError(f"a deferral window is at least one step; got {window}")
    return (window + 1) / 2.0


def overshoot_waste_fraction(window: int, output_tokens: int) -> float:
    """Share of a request's forwarded rows that nobody keeps, from deferral alone.

    Day 29 measured static batching at 79% here and continuous batching drove it to
    zero by construction: a row is in the forward only while it still wants a
    token. Deferral is the first thing in this engine to put any of it back, and
    the share is entirely a function of how long the request ran. A window of 8 on
    a request that kept 100 tokens is 4.3%; the same window on a request that kept
    5 is 47%, which is why a server with short generations should not defer far.
    """
    if output_tokens < 1:
        raise ValueError(
            f"a waste share needs output tokens to divide by; got {output_tokens}"
        )
    over = expected_overshoot(window)
    return over / (output_tokens + over)


# --- the two sides of the window -------------------------------------------------


def deferred_saving_per_step(window: int, *, latency_s: float) -> float:
    """Seconds per step that a window of k gives back, in synchronisations alone.

    One transfer a step becomes `1 / k`, so the saving is `(1 - 1/k) x latency`.
    Zero at `k = 1`, which is the number this function exists to keep honest: one
    step of lag removes no journey and the input-side saving it does buy is Python,
    measured in a profile rather than derived here.

    It saturates. However large the window gets, there was only ever one transfer
    per step to remove, so the saving is bounded by `latency_s` and every extra
    step of lag past that buys a rapidly shrinking fraction of a fixed prize.
    """
    was = syncs_per_step(1, strategy="one_transfer")
    now = syncs_per_step(1, strategy="deferred", window=window)
    return readback_s(was - now, latency_s=latency_s)


def overshoot_cost_per_step(window: int, *, row_s: float, output_tokens: int) -> float:
    """Seconds per step the wasted rows cost, spread over the run that wasted them.

    `row_s` is what one row of one decode forward costs. A request that kept
    `output_tokens` tokens ran that many steps and threw away
    `expected_overshoot(window)` rows, so the cost per step is the second divided
    by the first.

    Linear in the window, which is the asymmetry that makes this a trade at all:
    the saving above saturates and this does not.
    """
    if row_s < 0.0:
        raise ValueError(f"a row cost is not negative; got {row_s}")
    if output_tokens < 1:
        raise ValueError(f"a per-step cost needs steps to spread over; got {output_tokens}")
    return expected_overshoot(window) * row_s / output_tokens


def net_saving_per_step(
    window: int, *, latency_s: float, row_s: float, output_tokens: int
) -> float:
    """What deferring k steps is actually worth: journeys saved minus rows wasted.

    Negative is a real answer and the one a short generation on a fast bus gets.
    Reporting the first term on its own is how a day like this becomes an
    overclaim, so the subtraction is in the function rather than in the prose.
    """
    return deferred_saving_per_step(window, latency_s=latency_s) - overshoot_cost_per_step(
        window, row_s=row_s, output_tokens=output_tokens
    )


def best_window(
    *,
    latency_s: float,
    row_s: float,
    output_tokens: int,
    windows: Sequence[int] = WINDOWS,
) -> int:
    """The window with the largest net saving. A search over candidates, not a root.

    The maximum is real: the saving is `latency x (1 - 1/k)` and the cost is linear
    in `k`, so the difference has one interior peak. It is solved by walking the
    candidates rather than by differentiating, because the answer has to be an
    integer number of steps and the list is six long.

    Ties go to the smaller window, which is the conservative direction: a window
    buys a bounded prize and costs unbounded overshoot, so when two are worth the
    same the one that lags less is the one to run.
    """
    best, best_value = None, None
    for window in windows:
        value = net_saving_per_step(
            window, latency_s=latency_s, row_s=row_s, output_tokens=output_tokens
        )
        if best_value is None or value > best_value:
            best, best_value = window, value
    if best is None:
        raise ValueError("a window has to be chosen from at least one candidate")
    return best


# --- the processor ---------------------------------------------------------------


class DeferredOutputProcessor(OutputProcessor):
    """Day 47's collect phase, holding its tokens back for up to `window` steps.

    window:   how many steps one journey home covers. 1 is a one-step lag and the
              same transfer count as Day 47; k is one transfer per k steps.
    readback: the Day-47 instrument. Shared with the engine so the counter is one
              number for the whole loop.

    The rule is one line: **after every step, keep the newest batch and drain
    everything older, together.** The newest is kept because it is the next step's
    decode input and draining it is exactly the readback being removed. Everything
    older is drained in a single transfer, which is what makes the rate `1 / k`
    rather than `1` however many batches are held.

    Three counters, and they are three because the reasons a token does not reach
    its request are three:

      `tokens`             applied, the ones a caller gets.
      `overshoot_tokens`   the request had already finished when they arrived. The
                           price of the window, and the thing `waste_fraction`
                           sees.
      `abandoned_tokens`   the request went back to the waiting queue in between.
                           Not overshoot: that token was really wanted and the
                           recompute will sample it again. Charging it to the
                           window would blame deferral for a cost preemption
                           already had.

    `deferred_tokens` is the total that went in, and `check_tokens_conserved` is
    the identity that says none of them went anywhere else.
    """

    def __init__(self, window: int = 1, readback: Readback | None = None) -> None:
        super().__init__(readback)
        if window < 1:
            raise ValueError(f"a deferral window is at least one step; got {window}")
        self.window = window
        self._held: deque[tuple[TokenBatch, tuple]] = deque()
        self.deferred_tokens = 0
        self.overshoot_tokens = 0
        self.abandoned_tokens = 0

    # --- what is on the device right now -------------------------------------

    @property
    def held(self) -> int:
        """Batches sampled and not yet brought home."""
        return len(self._held)

    @property
    def held_batches(self) -> tuple[TokenBatch, ...]:
        return tuple(batch for batch, _ in self._held)

    @property
    def newest(self) -> TokenBatch | None:
        """The last batch sampled, or None. The next decode step's input, if the
        rows have not changed under it."""
        return self._held[-1][0] if self._held else None

    # --- the loop's two calls -------------------------------------------------

    def defer(self, batch: TokenBatch, requests: Sequence) -> None:
        """Take one step's tokens without looking at them. Costs no transfer.

        The row-order check happens here rather than at the drain, on purpose. The
        requests are captured now, so what is compared later is a tuple against a
        tuple; the moment worth refusing is the one where a batch is paired with
        rows it was not sampled for, and that moment is this one.

        An empty batch is not held. A prefill-only iteration that sampled nobody
        would otherwise sit in the deque as a row set matching nothing, and force a
        drain on the next step for no tokens at all.
        """
        check_in_row_order(batch, requests)
        if batch.num_rows == 0:
            return
        self._held.append((batch, tuple(requests)))
        self.deferred_tokens += batch.num_rows

    def settle(self) -> list[int]:
        """Bring everything but the newest home, in one journey. After the launch.

        Called once per iteration, after both forwards have been queued, which is
        the placement the whole idea rests on: the host has already given the
        device the next step's work before it stops to look at the last step's
        answer. On one stream that is a reordering rather than a saving, and the
        saving is the `len(self._held) - 1` batches that go home together.
        """
        if len(self._held) <= self.window:
            return []
        due = [self._held.popleft() for _ in range(len(self._held) - 1)]
        return self._resolve(due)

    def flush(self) -> list[int]:
        """Bring everything home, newest included. The end of a run, or a row change.

        The second caller is the important one. Deferral is only safe while the
        batch is the batch that was sampled, so a step whose rows differ from the
        held ones cannot use the held tensor as its input and needs Python ints it
        does not have. It pays here, at the top of that step, and goes back to
        deferring on the next one.
        """
        due = list(self._held)
        self._held.clear()
        return self._resolve(due)

    # --- the journey ----------------------------------------------------------

    def _resolve(self, due: list[tuple[TokenBatch, tuple]]) -> list[int]:
        """One `torch.cat`, one `.tolist()`, however many steps are due.

        The concatenation is the part worth naming. It is a device-side copy of a
        few `[rows]` int64 tensors, which the host hands to the driver and walks
        away from; it costs one launch and it is what turns k synchronisations into
        one. Trading a kernel for a stop is the right direction whenever the stop
        is a stop, and it is exactly the wrong direction on a CPU, where there was
        never a stop and the copy is pure loss.
        """
        if not due:
            return []
        tensors = [batch.tokens for batch, _ in due]
        flat = tensors[0] if len(tensors) == 1 else torch.cat(tensors)
        ids = self.readback.tolist(flat)
        applied: list[int] = []
        at = 0
        for batch, requests in due:
            rows = batch.num_rows
            batch.adopt_ids(ids[at : at + rows])
            at += rows
            applied.extend(self._apply_held(batch, requests))
            self.steps += 1
        return applied

    def _apply_held(self, batch: TokenBatch, requests: tuple) -> list[int]:
        """Hand each row's token to its request, or account for why it cannot.

        The state check is not defensive coding, it is the semantics of the day. A
        request that finished while its next token was in flight must not be given
        that token: `append_token` would raise, and appending past a stop rule is
        how a deferred engine returns one more token than it promised.
        """
        kept: list[int] = []
        for request, token in zip(requests, batch.resolve()):
            if request.state is RequestState.RUNNING:
                request.append_token(token)
                self.tokens += 1
                kept.append(token)
            elif request.is_finished:
                self.overshoot_tokens += 1
            else:
                self.abandoned_tokens += 1
        return kept


# --- the gates -------------------------------------------------------------------


def check_window_respected(processor: DeferredOutputProcessor) -> None:
    """Refuse a processor holding more steps than it said it would.

    Held batches are unbounded memory and unbounded overshoot, and the way this
    grows is a step that defers and forgets to settle. The tokens stay correct the
    whole time, which is what makes it invisible: the run simply ends with a queue
    of answers nobody ever asked for.
    """
    if processor.held > processor.window:
        raise OutputUnsound(
            f"the output path is holding {processor.held} steps on a window of "
            f"{processor.window}: a step deferred without settling"
        )


def check_overshoot_bounded(
    processor: DeferredOutputProcessor, *, finished_requests: int
) -> None:
    """Refuse a run that threw away more rows than the window can explain.

    `window` rows per finished request is the whole budget, and anything above it
    is not deferral, it is rows being forwarded for requests the scheduler should
    already have reaped.
    """
    limit = overshoot_bound(processor.window) * finished_requests
    if processor.overshoot_tokens > limit:
        raise OutputUnsound(
            f"the run overshot by {processor.overshoot_tokens} rows past "
            f"{finished_requests} finished requests, above the {limit} a window of "
            f"{processor.window} can explain"
        )


def check_tokens_conserved(processor: DeferredOutputProcessor) -> None:
    """Refuse a run whose deferred tokens do not add up. Four places, one total.

    Every token that went in is applied, overshot, abandoned, or still on the
    device. The failure this catches is a batch dropped on the floor by a drain
    that took the wrong slice, and the symptom without this check is a request that
    is one token short of its budget and finishes anyway.
    """
    held = sum(batch.num_rows for batch in processor.held_batches)
    total = processor.tokens + processor.overshoot_tokens + processor.abandoned_tokens + held
    if total != processor.deferred_tokens:
        raise OutputUnsound(
            f"{processor.deferred_tokens} tokens were deferred and {total} are "
            f"conserved ({processor.tokens} applied, {processor.overshoot_tokens} "
            f"overshot, {processor.abandoned_tokens} abandoned, {held} held): a "
            "batch went somewhere unaccounted"
        )


def check_input_is_held(input_ids: torch.Tensor, batch: TokenBatch) -> None:
    """Refuse a decode input that was rebuilt on the host. The day, as one compare.

    The saving on the input side is that `[rows, 1]` is a *view* of the tensor the
    sampler left, so it costs no allocation, no host to device copy and no Python
    list at all. A tensor with the same values and a different storage is the old
    path wearing the new path's shape, and it is what an innocent-looking
    `torch.tensor(batch.tokens.tolist())` produces.
    """
    if input_ids.dim() != 2 or input_ids.shape[1] != 1:
        raise OutputUnsound(
            f"a decode input is [rows, 1], one column of one token each; got shape "
            f"{tuple(input_ids.shape)}"
        )
    if input_ids.data_ptr() != batch.tokens.data_ptr():
        raise OutputUnsound(
            "the decode input does not share storage with the tokens the sampler "
            "left: it was rebuilt on the host, which is the copy this day removes"
        )


# --- the table -------------------------------------------------------------------


def render(
    *,
    latency_s: float,
    row_s: float,
    output_tokens: int,
    windows: Sequence[int] = WINDOWS,
    title: str | None = None,
) -> str:
    """The windows side by side: what each saves, what each wastes, and the net.

    Four columns because the day has four numbers and three of them are routinely
    quoted alone. The syncs column is what a paper reports, the overshoot column is
    what it costs, and the net column is the only one that answers the question.
    """
    lines = [title] if title else []
    lines.append(
        f"{'window':>7}{'syncs/step':>12}{'saved':>10}{'overshoot':>11}"
        f"{'waste':>8}{'net':>12}"
    )
    for window in windows:
        syncs = syncs_per_step(1, strategy="deferred", window=window)
        saved = deferred_saving_per_step(window, latency_s=latency_s)
        over = expected_overshoot(window)
        waste = overshoot_waste_fraction(window, output_tokens)
        net = net_saving_per_step(
            window, latency_s=latency_s, row_s=row_s, output_tokens=output_tokens
        )
        lines.append(
            f"{window:>7}{syncs:>12.3f}{saved * 1e6:>8.1f}us{over:>10.1f}r"
            f"{waste * 100:>7.1f}%{net * 1e6:>10.1f}us"
        )
    return "\n".join(lines)
