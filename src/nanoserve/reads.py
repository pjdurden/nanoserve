"""Which read a decode step runs, and what that read has held. Day 60.

Day 59 wrote `paged_attention_batched_kernel`, proved it equal to
`paged_attention_batched_reference` on toy pools, and stopped. Nothing called it, so
every decode step in the engine still went through the read that builds a
`[rows, heads, 1, ctx]` score rectangle before it masks: 268 MB of one intermediate
at 256 rows, 32 heads and an 8192-token width, the tensor Day 54's `workspace_bytes`
prices and the reason Day 52 had to bucket the width axis at all.

This module is the seam between the two. One object, `PagedRead`, holds the choice
and the dispatch, and the cache owns one of them instead of naming a function.

**The default is the rectangle, and that is the day's one load-bearing decision.**
The streamed read is a tlsim loop in Python: correct to a few ulps and roughly an
order of magnitude slower per call than the torch path it replaces, measured on this
box. A flag that defaulted on would make every engine in the repo correct and
unusable, and it would do it silently, because nothing downstream of the read can
tell which one ran. So `streamed_read=False` everywhere until the loop is Triton, and
the switch is a flag an operator types, not a default anybody inherits.

**Day 62 makes the streamed branch a dispatch, and that changes what the flag means
without changing the flag.** `STREAMED` now calls `paged_attention_batched`, which
launches the Triton kernel on a CUDA tensor and falls back to the Day-59 tlsim loop
everywhere else. The two backends compute the same attention to a few ulps, so the
choice is a speed decision; but it is the speed decision this whole stretch has been
waiting on, and the paragraph above is true on exactly one of the two. Day 60
measured the tlsim loop 8x to 67x slower per call than the torch rectangle it
replaces, and the kernel is meant to be the other side of that, which no test in this
repo can say because no box in this repo has a card. The default stays off because
this box is the one where it is a Python loop, and a default that is right on a card
and ruinous on a laptop is not a default.

**So the counters grow a third thing to witness.** `mode` says which read the
operator asked for and `backend` says which one the process could actually run, and
those are different questions the moment a dispatch exists: a server launched
`--streamed-read` on a box with no Triton reports `streamed` and means `tlsim`, which
is correct, slow, and indistinguishable from the fast thing in every other field of
the payload. An empty `backend` means no read has run yet, which is its own answer
and not a missing one.

**Which is exactly why the counters are here.** Two servers on the same weights, the
same pool and the same scheduler, one with the flag and one without, return the same
tokens with the same latency profile and the same everything else a client can
observe. The wiring has no witness at all unless the process publishes one. So a read
counts what it did: how many calls, how many rows, how many score cells a rectangle
read *would* have materialised, and how many this read actually held. The ratio of
the last two is the saving a live server got, which is a different claim from
`streambench.py`'s, and a weaker and more useful one: that table is arithmetic about
shapes the bench invented, and this one is a quotient of two things that happened.

Both cell counts are free. `q.shape` and `slot_mapping.shape` are static, so the
accounting is host arithmetic over numbers the caller already has, and nothing here
reads a tensor's *contents*. That matters more than it looks: the obvious richer
counter is tiles walked, which is `cdiv` over `context_lens`, and getting at those on
a card is `int(tensor)` per call per layer, which is the synchronisation Day 48 found
the hard way and the graph break Day 49 spent a day removing. A counter that costs a
sync is a counter that changes the thing it measures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .captured import score_cells, streamed_score_cells
from .compiled import DecodeShape
from .kernels.paged_attention import paged_attention_batched_reference
from .kernels.triton_batched_attention import paged_attention_batched, select_backend

#: The Day-28 read: gather the whole `[rows, max_ctx]` mapping, score it into a
#: rectangle, mask the result. The default, and the oracle the other one is graded
#: against.
RECTANGLE = "rectangle"

#: Day 59's read: one program per `(row, query head)`, walking that row's own history
#: a tile of keys at a time and folding each tile into an online softmax. No gather
#: and no rectangle, at the price of a Python loop until it is Triton.
STREAMED = "streamed"

READS = (RECTANGLE, STREAMED)

#: Keys per score tile, when nobody says otherwise. A pure performance knob: every
#: value returns the same attention. It is *not* `block_size`, and the two are
#: unrelated: this is a tile of the score, that is a tile of the pool.
DEFAULT_BLOCK = 32

#: The backend the rectangle read runs on. It has no tile and no loop, so it has no
#: choice to make: it is one gather and two matmuls, which is torch either way. The
#: streamed read's backends are `select_backend`'s, "triton" or "tlsim".
TORCH = "torch"


class ReadUnwitnessed(AssertionError):
    """A harness could not find out which read a server ran.

    Its own class because the two ways to fail here are different mornings. A gate
    that refuses because the counters say "rectangle" has observed a server and
    disagreed with it. A gate that refuses because there were no counters has
    observed nothing, and a run that reports the second as the first is a benchmark
    quietly describing a process it never reached.
    """


@dataclass(frozen=True)
class ReadStats:
    """One reading of what the decode read has done, without the read. Day 60.

    mode:        `RECTANGLE` or `STREAMED`. In the payload for the reason
                 `CaptureStats.mode` is: a saving of 1.0x means two different
                 things, and only the mode next to it says whether this server is
                 doing what it was launched to do.
    block:       the score tile the streamed read folds, and 0 on the rectangle,
                 which has no tile because it has no loop.
    backend:     what the last read actually ran on: "torch" for the rectangle,
                 "triton" or "tlsim" for the streamed one, and "" when nothing has
                 run yet. Day 62. `mode` is what the operator asked for and this is
                 what the box could give them, and a server that asked for the
                 streamed read on a machine with no Triton differs from one that got
                 the kernel in no other field of this payload.
    calls:       reads issued. One per layer per decode step, so it is roughly
                 `steps * num_hidden_layers` and not `steps`.
    rows:        sequences summed over those calls, which is the batch size
                 integrated over the run. `rows / calls` is the average batch a
                 decode step actually had, and that is the one number here a
                 scheduler person wants.
    score_cells: entries in the `[rows, heads, 1, ctx]` rectangle these calls were
                 *charged*, whether or not this read built it.
    held_cells:  entries this read really had alive at once, summed the same way.

    Frozen, because a reading is a moment, and cumulative, because a counter is.
    Every claim about a *run* is therefore a subtraction, which is what `since` is
    for: the same shape Day 57 gave `CaptureStats` and for the same reason.
    """

    mode: str = RECTANGLE
    block: int = 0
    backend: str = ""
    calls: int = 0
    rows: int = 0
    score_cells: int = 0
    held_cells: int = 0

    @property
    def saving(self) -> float:
        """How many times the rectangle covers what this read held. 1.0x on the
        default, and that is a measurement and not a missing number: a read that
        materialises the row it scores holds every cell it was charged.

        **Day 61 makes this a tautology on a streamed server, and the note is the
        useful part.** A streamed bucket set reads at `max_model_len` on every step,
        so `score_cells` is charged a constant width and `held_cells` is a constant
        tile, and the quotient is exactly `max_model_len / block` however long the run
        was. It is not wrong, it just stopped being a measurement of anything the
        traffic did: Day 60's table came off a cache that still bucketed the width,
        where the quotient really was integrated over a growing rectangle. The honest
        numerator would be the width a rectangle read would have rounded this step to,
        which is `int(context_lens.max())`, which is Day 48's synchronisation once per
        call per layer to make a log read better. So the number stays and this
        paragraph is the price tag on it."""
        if not self.held_cells:
            return 1.0
        return self.score_cells / self.held_cells

    @property
    def rows_per_call(self) -> float:
        """The average batch a decode step had, over this reading's window."""
        if not self.calls:
            return 0.0
        return self.rows / self.calls

    def since(self, earlier: ReadStats) -> ReadStats:
        """This reading minus an earlier one: what happened in between.

        Refuses two readings that disagree about the mode. A process picks its read
        when the cache is built and cannot change it, so a mismatch is two processes
        or one payload parsed wrong, and subtracting them would produce a perfectly
        plausible window over nothing.

        Refuses a change of backend on the same grounds, with one allowance: an
        earlier reading taken before the first read has an empty backend, and that is
        a reading of a process that had not yet found out, not of a different one. So
        "" to "triton" is a legal window and "tlsim" to "triton" is not.
        """
        if self.mode != earlier.mode:
            raise ValueError(
                f"these two readings are not of the same read ({earlier.mode} then "
                f"{self.mode}): a process picks its read at construction, so a "
                "window across a change of mode is a window across two processes"
            )
        if earlier.backend and self.backend and self.backend != earlier.backend:
            raise ValueError(
                f"these two readings are not of the same backend ({earlier.backend} "
                f"then {self.backend}): a device does not acquire Triton mid-run, so "
                "this window spans two processes and its per-call costs are a blend"
            )
        deltas = {
            name: getattr(self, name) - getattr(earlier, name)
            for name in ("calls", "rows", "score_cells", "held_cells")
        }
        negative = sorted(name for name, value in deltas.items() if value < 0)
        if negative:
            raise ValueError(
                f"the earlier reading counted more than the later one ({', '.join(negative)}): "
                "these counters only rise, so this window is the wrong way round"
            )
        return replace(self, **deltas)

    @classmethod
    def from_dict(cls, payload: dict) -> ReadStats:
        """Rebuild a reading from the JSON it went over the wire as.

        Every field has a default, so a payload from an older process arrives with
        counters missing rather than unreadable. The mode is the one that is not
        guessed at: absent, it is the rectangle, which is the reading that makes a
        gate about the streamed read refuse rather than pass.
        """
        return cls(
            mode=payload.get("mode", RECTANGLE),
            block=int(payload.get("block", 0)),
            backend=payload.get("backend", ""),
            calls=int(payload.get("calls", 0)),
            rows=int(payload.get("rows", 0)),
            score_cells=int(payload.get("score_cells", 0)),
            held_cells=int(payload.get("held_cells", 0)),
        )

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "block": self.block,
            "backend": self.backend,
            "calls": self.calls,
            "rows": self.rows,
            "score_cells": self.score_cells,
            "held_cells": self.held_cells,
        }

    def render(self) -> str:
        tile = f" block {self.block}" if self.block else ""
        on = f" on {self.backend}" if self.backend else ""
        return (
            f"{self.mode}{tile}{on}: {self.calls} reads, {self.rows_per_call:.1f} rows "
            f"each, held {self.held_cells} of {self.score_cells} cells "
            f"({self.saving:.1f}x)"
        )


class PagedRead:
    """The decode read a cache runs, and the tally of what it has held. Day 60.

    Two branches and one contract. Both take exactly the arguments
    `paged_attention_batched_reference` takes, both refuse exactly what it refuses,
    and both return `[batch, n_q, 1, d]`. The refusals matter as much as the output:
    a dispatch that softened one branch's guard would make the two reads differ in
    what they *accept* rather than in what they hold, and the first symptom of that
    is a crash that only happens under one flag.

    `validated` and `context_bounds` are passed straight through for the same
    reason. The plan path hands `validated=True` because Day 50 checked the lengths
    on the host when the plan was built, and that has to mean the same thing on both
    branches or Day 50's whole argument holds for one flag and not the other.

    Not a function, because it has to remember. The counters are the only way anybody
    outside the process can tell the two reads apart (see the module docstring), and
    a closure over a mutable box would be the same object with the tally hidden.
    """

    def __init__(self, mode: str = RECTANGLE, block: int = DEFAULT_BLOCK):
        if mode not in READS:
            raise ValueError(f"unknown decode read {mode!r}; expected one of {READS}")
        if block < 1:
            raise ValueError(f"a tile holds at least one key; got {block}")
        self.mode = mode
        self.block = block
        self.backend = ""
        self.calls = 0
        self.rows = 0
        self.score_cells = 0
        self.held_cells = 0

    @property
    def streamed(self) -> bool:
        return self.mode == STREAMED

    def __call__(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        n_rep: int,
        scale: float | None = None,
        *,
        context_bounds: tuple[int, int] | None = None,
        validated: bool = False,
    ) -> torch.Tensor:
        """Run the picked read, record what ran it, and charge it for what it held.

        The backend is read off the device rather than stored at construction, and it
        is recorded *after* the call rather than before. A `PagedRead` is built before
        anything has been moved anywhere, so the only moment the answer is knowable is
        the moment a tensor arrives, and a call that was refused ran on nothing.
        `select_backend` is a string compare on `q.device.type`, so asking every call
        costs nothing and removes the failure where the cache was built on the host
        and the engine later moved to a card.
        """
        if self.streamed:
            backend = select_backend(q.device)
            out = paged_attention_batched(
                q,
                k_pool,
                v_pool,
                slot_mapping,
                context_lens,
                n_rep,
                scale,
                block=self.block,
                context_bounds=context_bounds,
                validated=validated,
            )
        else:
            backend = TORCH
            out = paged_attention_batched_reference(
                q,
                k_pool,
                v_pool,
                slot_mapping,
                context_lens,
                n_rep,
                scale,
                context_bounds=context_bounds,
                validated=validated,
            )
        self.backend = backend
        self._charge(q, slot_mapping)
        return out

    def _charge(self, q: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Both cell counts for one call, from shapes and never from contents.

        Charged after the read rather than before it, so a call that was refused is
        not counted: a reading is what this read *did*, and a rejected call did
        nothing. `DecodeShape` is reused rather than open-coded because
        `score_cells` and `streamed_score_cells` are Day 54's and Day 59's
        definitions of the same two quantities, and a second copy of the
        multiplication here is a second copy that can drift from the one the capture
        list sizes its pool with.
        """
        rows, heads = q.shape[0], q.shape[1]
        shape = DecodeShape(rows=rows, context_width=slot_mapping.shape[1])
        charged = score_cells(shape, heads)
        self.calls += 1
        self.rows += rows
        self.score_cells += charged
        self.held_cells += (
            streamed_score_cells(shape, heads, self.block) if self.streamed else charged
        )

    def stats(self) -> ReadStats:
        """A frozen reading of the tally so far. See `ReadStats`."""
        return ReadStats(
            mode=self.mode,
            block=self.block if self.streamed else 0,
            backend=self.backend,
            calls=self.calls,
            rows=self.rows,
            score_cells=self.score_cells,
            held_cells=self.held_cells,
        )

    def as_dict(self) -> dict:
        return self.stats().as_dict()

    def render(self) -> str:
        return self.stats().render()
