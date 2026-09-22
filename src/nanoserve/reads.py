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

**Day 65 adds the third read, and it is the first one that cannot run on its own.**
The rectangle needs a pool and a mapping. The streamed read needs those and a tile
width, which it carries itself because a tile is an integer. The split needs an arena
somebody else allocated, at a width somebody else fixed, and Day 64 is why: three
`torch.empty` calls inside the read are legal under capture and unpriced by the plan,
so the workspace became a `SplitWorkspace` the plan owns. A `PagedRead` is built when
the cache is, which is before anything has been moved to a device, so the arena
cannot be a constructor argument on the boot path. `attach` is that gap written down:
between construction and the first decode step there is a window where the mode is
set and the workspace is not, and a split read that quietly allocated its own there
would undo the whole of the previous day on exactly the path that matters.

**So `streamed` stops being the question the width axis turns on.** Day 61 collapsed
the bucket set for a read that holds a tile instead of a rectangle, and the split
holds tiles too: more of them, at once, one per chunk per row per head. `tiled` is
that question and `streamed` goes back to meaning "is exactly Day 59's read", because
a single flag doing both jobs would either refuse the split read or wave it through
the gate, and both of those are wrong in the same file.

**And the saving means a third thing.** A split does not make the tile smaller, it
makes more of them live: the split count is a grid axis, so `held_cells` is the
streamed read's multiplied by the chunks and the quotient is divided by them. A
server reporting 16x next to one reporting 256x is not broken, it is split. What this
counter does *not* hold is the arena, and that is deliberate rather than an omission:
the partials are not scores, they are constant for the life of the process, and they
are priced once at boot by `split_workspace_bytes`. A per-call counter that added
them would restate a boot number once per layer per step and make the quotient
untraceable to either.

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

from .captured import score_cells, split_score_cells, streamed_score_cells
from .compiled import DecodeShape
from .kernels.flash_decoding import SplitUnsound
from .kernels.paged_attention import paged_attention_batched_reference
from .kernels.triton_batched_attention import paged_attention_batched, select_backend
from .partials import SplitWorkspace

#: The Day-28 read: gather the whole `[rows, max_ctx]` mapping, score it into a
#: rectangle, mask the result. The default, and the oracle the other one is graded
#: against.
RECTANGLE = "rectangle"

#: Day 59's read: one program per `(row, query head)`, walking that row's own history
#: a tile of keys at a time and folding each tile into an online softmax. No gather
#: and no rectangle, at the price of a Python loop until it is Triton.
STREAMED = "streamed"

#: Day 63's read, wired on Day 65: Day 59's loop cut into `splits` chunks per row,
#: each chunk its own program with its own partial softmax, folded by a second pass.
#: The only read that needs a workspace it does not own, because the chunks hand
#: numbers to each other through memory. See `nanoserve.partials`.
SPLIT = "split"

READS = (RECTANGLE, STREAMED, SPLIT)

#: The reads that hold a tile rather than a rectangle, which is what the width axis
#: turns on. Day 61 asked this question as `streamed`, when there were two reads and
#: the answer happened to coincide. See `check_read_matches`.
TILED = (STREAMED, SPLIT)

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

    mode:        `RECTANGLE`, `STREAMED` or `SPLIT`. In the payload for the reason
                 `CaptureStats.mode` is: a saving of 1.0x means two different
                 things, and only the mode next to it says whether this server is
                 doing what it was launched to do.
    block:       the score tile a tiled read folds, and 0 on the rectangle, which
                 has no tile because it has no loop.
    splits:      chunks per row, on a split read that has been handed its arena.
                 0 everywhere else, and *also* 0 on a split read that has not, which
                 is the one ambiguity in this payload and the cheapest place to leave
                 it: a process in that state has completed no decode step, so
                 `calls` is 0 next to it and the pair says which. Day 65. It is here
                 because it is the field a memory budget turns on: two servers both
                 reporting `split` on `triton` can hold arenas a factor of sixteen
                 apart, and `rows * heads * splits * (head_dim + 2)` is why.
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
    splits: int = 0
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
        paragraph is the price tag on it.

        **Day 65 gives it a third reading, and this one is not a tautology.** A split
        read holds the streamed read's tile once per chunk, because the split count is
        a grid axis and every program on it is scoring, so the quotient is the
        streamed one divided by `splits`. That is the trade stated in the only
        currency a live server publishes: the split shortens the tail by spreading one
        row over more programs and pays for it in score cells held at once and in an
        arena this number does not contain. A server reporting 16x beside one
        reporting 256x is not a broken streamed server."""
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

        Refuses a change of split count for the mode's reason rather than the
        backend's, so it gets no allowance. An arena is allocated once, at a size,
        before the first decode step: a process cannot acquire chunks the way it can
        discover a backend, so 0 to 16 is two processes and not a process finding out.
        Day 65.
        """
        if self.mode != earlier.mode:
            raise ValueError(
                f"these two readings are not of the same read ({earlier.mode} then "
                f"{self.mode}): a process picks its read at construction, so a "
                "window across a change of mode is a window across two processes"
            )
        if self.splits != earlier.splits:
            raise ValueError(
                f"these two readings are not of the same arena ({earlier.splits} then "
                f"{self.splits} chunks): a split read is handed its workspace before "
                "its first step and holds it for the life of the process, so a window "
                "across a change of chunks is a window across two processes"
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
            splits=int(payload.get("splits", 0)),
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
            "splits": self.splits,
            "backend": self.backend,
            "calls": self.calls,
            "rows": self.rows,
            "score_cells": self.score_cells,
            "held_cells": self.held_cells,
        }

    def render(self) -> str:
        tile = f" block {self.block}" if self.block else ""
        cut = f" x {self.splits} splits" if self.splits else ""
        on = f" on {self.backend}" if self.backend else ""
        return (
            f"{self.mode}{tile}{cut}{on}: {self.calls} reads, "
            f"{self.rows_per_call:.1f} rows each, held {self.held_cells} of "
            f"{self.score_cells} cells ({self.saving:.1f}x)"
        )


class PagedRead:
    """The decode read a cache runs, and the tally of what it has held. Day 60.

    Three branches and one contract. All of them take exactly the arguments
    `paged_attention_batched_reference` takes, all of them refuse exactly what it
    refuses, and all of them return `[batch, n_q, 1, d]`. The refusals matter as much
    as the output: a dispatch that softened one branch's guard would make the reads
    differ in what they *accept* rather than in what they hold, and the first symptom
    of that is a crash that only happens under one flag.

    The third branch has one precondition the other two do not, and it is the shape
    of Day 65: a split read holds a `SplitWorkspace` it did not allocate. Between
    construction and `attach` it knows its mode and has no arena, and a call in that
    window is refused rather than served, because the only other options are to
    allocate one (Day 64's whole argument, undone) or to serve the wrong read (a flag
    that silently means something else).

    `validated` and `context_bounds` are passed straight through for the same
    reason. The plan path hands `validated=True` because Day 50 checked the lengths
    on the host when the plan was built, and that has to mean the same thing on both
    branches or Day 50's whole argument holds for one flag and not the other.

    Not a function, because it has to remember. The counters are the only way anybody
    outside the process can tell the two reads apart (see the module docstring), and
    a closure over a mutable box would be the same object with the tally hidden.
    """

    def __init__(
        self,
        mode: str = RECTANGLE,
        block: int = DEFAULT_BLOCK,
        workspace: SplitWorkspace | None = None,
    ):
        if mode not in READS:
            raise ValueError(f"unknown decode read {mode!r}; expected one of {READS}")
        if block < 1:
            raise ValueError(f"a tile holds at least one key; got {block}")
        self.mode = mode
        self.block = block
        self.workspace: SplitWorkspace | None = None
        self.backend = ""
        self.calls = 0
        self.rows = 0
        self.score_cells = 0
        self.held_cells = 0
        if workspace is not None:
            self.attach(workspace)

    @property
    def streamed(self) -> bool:
        """Exactly Day 59's read, and not "does not build a rectangle"."""
        return self.mode == STREAMED

    @property
    def split(self) -> bool:
        return self.mode == SPLIT

    @property
    def tiled(self) -> bool:
        """Whether this read holds a tile instead of the row it is scoring.

        The question the width axis actually turns on, which `check_read_matches` asked
        as `streamed` while there were only two reads. A split read gathers no more
        than a streamed one does, so a width bucket buys it nothing either.
        """
        return self.mode in TILED

    @property
    def splits(self) -> int:
        """Chunks per row, or 0 for a read with no arena. See `ReadStats.splits`."""
        return 0 if self.workspace is None else self.workspace.splits

    def attach(self, workspace: SplitWorkspace) -> None:
        """Take the arena the plan allocated. Day 65.

        A method rather than a constructor argument because of when each is knowable.
        The mode is a flag an operator typed and the cache is built from it; the arena
        is device memory, and on the boot path nothing has been moved to a device yet
        when the cache is constructed. So the two arrive at different times and the
        window between them is real, which is what the refusal in `__call__` is for.

        Idempotent on the *same* workspace and refused on a different one, and that
        asymmetry is the capture argument rather than tidiness. A recorded graph is
        bound to the addresses it was recorded with, so a workspace swapped under a
        read that has already been captured leaves the replay writing into storage
        nothing reads and reading storage nothing writes. The answer that comes out is
        finite, plausible and one step stale, which is the failure this repo has spent
        three weeks learning to refuse instead of debug.
        """
        if not self.split:
            raise SplitUnsound(
                f"a {self.mode} read does not address a split workspace: the arena is "
                "an accumulator per (row, head, chunk) and this read has no chunks, so "
                "attaching one reserves device memory nothing in the process reads"
            )
        if workspace.block != self.block:
            raise SplitUnsound(
                f"this read folds a {self.block}-key score tile and the arena was "
                f"partitioned in {workspace.block}-key tiles: the chunk bounds are "
                "`split * keys_per_split` and the chunk is a whole number of the "
                "arena's tiles, so the read would walk a partition it was not cut for"
            )
        if self.workspace is not None and self.workspace is not workspace:
            raise SplitUnsound(
                "this read already holds an arena and a second one would move the "
                "addresses out from under anything that captured the first: a replay "
                "is bound to the pointers it recorded, so the read would keep "
                "returning the step before it, plausibly"
            )
        self.workspace = workspace

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
        if self.split:
            if self.workspace is None:
                raise SplitUnsound(
                    "this read is a split and holds no workspace: the arena is "
                    "allocated once by the plan and handed over before the first "
                    "decode step, so a launch here would have to allocate its own, "
                    "which is the unpriced `torch.empty` the previous day removed"
                )
            backend = select_backend(q.device)
            out = self.workspace.read(
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
        elif self.streamed:
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
        `score_cells`, `streamed_score_cells` and `split_score_cells` are Day 54's,
        Day 59's and Day 64's definitions of the same quantities, and a second copy of
        the multiplication here is a second copy that can drift from the one the
        capture list sizes its pool with. The split arm charges the split count it
        reads off the *arena* rather than one it was told, because the arena is what
        the launch will actually be shaped by.
        """
        rows, heads = q.shape[0], q.shape[1]
        shape = DecodeShape(rows=rows, context_width=slot_mapping.shape[1])
        charged = score_cells(shape, heads)
        if self.split:
            held = split_score_cells(shape, heads, self.block, self.splits)
        elif self.streamed:
            held = streamed_score_cells(shape, heads, self.block)
        else:
            held = charged
        self.calls += 1
        self.rows += rows
        self.score_cells += charged
        self.held_cells += held

    def stats(self) -> ReadStats:
        """A frozen reading of the tally so far. See `ReadStats`."""
        return ReadStats(
            mode=self.mode,
            block=self.block if self.tiled else 0,
            splits=self.splits,
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
