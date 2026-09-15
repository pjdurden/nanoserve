"""The persistent batch: keep the running rows a prefix of the table. Day 58.

Day 57 found the bug and then measured the hole it left. A recorded decode reads two
things and both of them are indexed from zero: the persistent input buffers, which a
step writes in *batch* order, and the read rectangle, which is the window
`slots[:rows, :width]` on the slot table in *cache row* order. Those are the same
index only while the scheduler's rows are `(0, 1, ... n-1)`. `rows_are_a_prefix` is
the gate that made a misaligned step run the forward instead of replaying a graph
over somebody else's history, and `scattered_calls` is how often that happened:

    slots  requests    gen lengths   replayed   scattered   share
        8        16             16         30           0    100%
        8        16           8/16         14          17     45%
        8        16      4/8/16/32         12          35     26%

Four generation lengths over eight slots and three quarters of the decode loop could
not use any graph in the list. More slots made it worse, because a wider batch has
more chances to be something other than a contiguous run from zero.

**The scheduler's freedom to hand out any free row is what costs that.** It is a
reasonable design: a heap of free slots, lowest index wins, O(1) per admission, and
nothing ever moves a running request. The bill arrives three weeks later, as the
reason a warm capture list covers a quarter of the steps it was recorded for. The
moment one request in a batch finishes before its neighbour, the survivor is left in
row 1 and the batch is one row wide, and there is no graph anywhere that reads row 1.

So this module takes that freedom away, which is what vLLM and SGLang do and what
they mean by a persistent batch: **after every schedule, the running rows are
`(0, 1, ... n-1)`**, so the question a replay asks has one answer forever.

Three things make it cheap, and they are worth separating because only the first is
obvious:

  1. **A move is a relabel.** The K/V does not move. A block id is a name for a
     place in the pool, and what changes hands is the `BlockTable` holding those
     names, the slot table's row of addressing, and the request's slot id. A
     2,000-token row costs 2,000 int64 of device copy and none of the megabytes its
     context actually occupies.
  2. **A move is paid for by a completion, not by a step.** A hole is made by a
     release, and one move fills it, so `moves <= releases` over any run:
     `check_moves_amortised`. A decode step that finishes nobody pays one tuple
     comparison on the host and copies nothing.
  3. **The number of moves is forced.** Every occupied row at or past the running
     count has to leave and every hole below it has to be filled, so a plan is
     minimal by counting and the only freedom left is which stray takes which hole.
     `plan_compaction` pairs them sorted against sorted, which keeps a batch in the
     relative order it was already in and makes two consecutive traces readable next
     to each other.

**And one thing makes it dangerous, which is where it runs.** The row this copies
into is storage a captured graph holds the address of and reads on every replay. The
only safe moment to write it is one where no step is in flight, and that moment is
inside `Scheduler.schedule`, after the reap and the growth and before admission.
After the growth because a preemption releases a row halfway through and those holes
have to be swept too. Before admission because a newcomer admitted into a hole would
still leave the survivor above it out of place, so the second order pays a move for a
row that had only just arrived.

There is no tensor in this file. It plans moves over integers and calls back for the
physical half, the same split `Scheduler.on_release` has: the scheduler owns row
lifetime and has no tensors, and the engine is the only object that knows what a row
means physically. See `BatchedPagedKVCache.move_row` and `SlotTable.move_row`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

#: What a compactor may be told to be. "off" is every day from 31 to 57 and is still
#: the default: a scheduler that starts moving rows because it was constructed
#: differently would silently change what a slot id means to everything holding one.
MODES = ("off", "on")


class CompactionUnsound(AssertionError):
    """A batch that is not a prefix, or a plan that is not the cheapest one.

    An `AssertionError` for the same reason `PlanUnsound` and `SlotsUnsound` are:
    every one of these is a statement the code believes about itself, checked in a
    test or on a benchmark rather than on the step of a serving loop.
    """


@dataclass(frozen=True)
class RowMove:
    """One row's worth of compaction: the history in `src` is to live in `dst`.

    Frozen and tiny, because it travels: it is planned here, applied by a callback
    that owns tensors, reported on the `SchedulerOutput` so a trace can show what a
    step rearranged, and read back by the gates below.
    """

    src: int
    dst: int

    def __post_init__(self) -> None:
        if self.src < 0 or self.dst < 0:
            raise ValueError(f"a row index is not negative; got {self.src} -> {self.dst}")
        if self.src == self.dst:
            raise ValueError(
                f"row {self.src} cannot move to the same row: a move is a copy into a "
                "row nobody is holding, and a self-move would be a no-op wearing the "
                "name of a real one"
            )

    def render(self) -> str:
        return f"row {self.src} -> row {self.dst}"


# --- what a compact batch is ----------------------------------------------------------


def _rows(rows) -> tuple[int, ...]:
    """Normalise a row selection. Order is the caller's; this module wants the set."""
    rows = tuple(int(r) for r in rows)
    if len(set(rows)) != len(rows):
        raise ValueError(f"rows must be distinct, got {list(rows)}")
    if any(r < 0 for r in rows):
        raise ValueError(f"a row index is not negative, got {list(rows)}")
    return rows


def is_compact(rows) -> bool:
    """Whether these rows are `(0, 1, ... n-1)` as a set, in any order.

    The primitive the whole module is about, and the same question
    `captured.rows_are_a_prefix` asks of a plan: a recorded read is a window from
    row zero, so a batch occupying `{0, 1, 2}` is one every graph in the list
    addresses and a batch occupying `{1, 2}` is one none of them do.
    """
    rows = _rows(rows)
    return set(rows) == set(range(len(rows)))


def holes(rows) -> tuple[int, ...]:
    """Rows under the running count that nobody is holding, lowest first."""
    rows = _rows(rows)
    held = set(rows)
    return tuple(r for r in range(len(rows)) if r not in held)


def strays(rows) -> tuple[int, ...]:
    """Occupied rows at or past the running count, lowest first.

    There are always exactly as many of these as there are holes, which is why a
    compaction is a permutation and never an allocation: `len(rows)` rows spread over
    a range wider than `len(rows)` leaves the same number of gaps below the line as
    there are occupants above it, by counting alone.
    """
    rows = _rows(rows)
    return tuple(sorted(r for r in rows if r >= len(rows)))


def moves_needed(rows) -> int:
    """How many rows have to move before this batch is a prefix."""
    return len(strays(rows))


def plan_compaction(rows) -> tuple[RowMove, ...]:
    """The fewest moves that turn `rows` into `(0, 1, ... n-1)`.

    Minimal by construction rather than by search: a stray has to vacate, a hole has
    to be filled, and one move does both, so the count is forced and the only choice
    is the pairing. Sorted against sorted is the choice, and it is a readability
    decision rather than a cost one: it leaves the batch in the same relative order
    it was in, so a request that was ahead of another this step is ahead of it next
    step too.
    """
    return tuple(
        RowMove(src, dst) for src, dst in zip(strays(rows), holes(rows), strict=True)
    )


def move_cells(moves: Iterable[RowMove], lengths: Mapping[int, int]) -> int:
    """Slot table cells a plan copies: each moved row's own length.

    The whole price of a compaction, and the reason it is affordable. `lengths` is
    keyed by source row because that is where the tokens are when the question is
    asked. Nothing in the K/V pool is read or written, so this is int64 of addressing
    and not bytes of context.
    """
    return sum(int(lengths.get(move.src, 0)) for move in moves)


# --- doing it -------------------------------------------------------------------------


class RowCompactor:
    """Plans compactions, applies them through a callback, and counts what it cost.

    `on_move(src, dst)` is the physical half, installed by whoever owns tensors and
    called once per move in plan order, before the caller relabels anything. That
    ordering is `on_release`'s and is load-bearing for the same reason: the callback
    is told which physical row is moving where while the scheduler still knows, and
    the row it is told about is the row nobody has been handed yet.

    Three counters, and the distance between them is the claim. `calls` is how often
    the question was asked, which is every schedule; `compactions` is how often the
    answer was anything but nothing; `moves` is rows actually copied. On a run of
    decode steps that finish nobody, the first grows and the other two do not.
    """

    def __init__(self, *, mode: str = "off", on_move: Callable[[int, int], None] | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.on_move = on_move
        self.calls = 0
        self.compactions = 0
        self.moves = 0

    def compact(self, rows) -> tuple[RowMove, ...]:
        """Plan and apply. Returns the moves, so the caller can relabel and report.

        Switched off, this is `()` without planning: the rows are whatever the
        scheduler made them, which is what every day before this one did and is
        correct, just not a prefix.
        """
        if self.mode == "off":
            return ()
        self.calls += 1
        moves = plan_compaction(rows)
        if not moves:
            return ()
        self.compactions += 1
        self.moves += len(moves)
        if self.on_move is not None:
            for move in moves:
                self.on_move(move.src, move.dst)
        return moves

    @property
    def moves_per_compaction(self) -> float:
        return self.moves / self.compactions if self.compactions else 0.0

    def render(self) -> str:
        """One line for a log: how often it looked, how often it had to act."""
        return (
            f"compaction {self.mode}: {self.calls} calls, {self.compactions} "
            f"compactions, {self.moves} moves "
            f"({self.moves_per_compaction:.1f} per compaction)"
        )


# --- gates ----------------------------------------------------------------------------


def _plan_rows(rows) -> tuple[int, ...]:
    """Accept a row tuple or anything carrying one, e.g. a `DecodePlan`."""
    return _rows(getattr(rows, "rows", rows))


def check_batch_is_persistent(rows) -> None:
    """Refuse a batch whose rows are not `(0, 1, ... n-1)`. The day's claim.

    Takes a plan as readily as a tuple, the same way Day 54's capture gates take a
    reading as readily as the object: the caller holding a `DecodePlan` should not
    have to reach into it to ask the one question the plan exists to answer.

    The message names the lowest hole rather than printing the set, because that is
    the row the reader wants: it is where the window a graph was recorded against
    starts, and it is empty.
    """
    rows = _plan_rows(rows)
    if is_compact(rows):
        return
    missing = holes(rows)
    raise CompactionUnsound(
        f"this batch holds rows {sorted(rows)} and row {missing[0]} is empty: a "
        "recorded read is the window `slots[:rows]`, so a batch with a hole under it "
        "is one no graph in the capture list addresses, and the step falls through to "
        "the forward"
    )


def check_moves_minimal(rows, moves) -> None:
    """Refuse a plan that left a hole, or filled them by copying more than it had to.

    Both halves matter and they fail differently. Too few moves is a batch that is
    still not a prefix, which is a correctness claim about the next step. Too many is
    a batch that is a prefix and paid for rows that were already where they belong,
    which is pure cost on the one path this module exists to keep cheap.
    """
    rows = _rows(rows)
    moves = tuple(moves)
    where = {move.src: move.dst for move in moves}
    after = tuple(where.get(r, r) for r in rows)
    if not is_compact(after):
        raise CompactionUnsound(
            f"applying {[m.render() for m in moves]} to rows {sorted(rows)} leaves "
            f"{sorted(after)}, which is not compact: a plan that does not finish the "
            "job is a compaction cost paid for a batch that still cannot replay"
        )
    needed = moves_needed(rows)
    if len(moves) != needed:
        raise CompactionUnsound(
            f"{len(moves)} move(s) to compact rows {sorted(rows)}, which needs "
            f"{needed}: every stray has to leave and every hole has to be filled, so "
            "anything above that count copied a row that was already in place"
        )


def check_moves_amortised(compactor: RowCompactor, *, releases: int, limit: float = 1.0) -> None:
    """Refuse a run that moved more rows than it gave back. The cost gate.

    The invariant is one sentence: a hole is made by a release and filled by one
    move, so moves can never outnumber releases over any run. That makes the cost of
    a persistent batch proportional to *completions* rather than to steps, which is
    the difference between a fixed overhead on a decode loop and a per-request tidy-up
    nobody notices.

    A gate rather than an assertion inside `compact` because it is a statement about
    a whole run: any single call can legitimately move several rows (a preemption
    cascade releases several at once), and it is the sum that has to stay bounded.
    """
    if releases < 0:
        raise ValueError(f"a release count is not negative; got {releases}")
    if limit <= 0.0:
        raise ValueError(f"a limit is a positive ratio of moves to releases; got {limit}")
    if compactor.moves > releases * limit:
        raise CompactionUnsound(
            f"{compactor.moves} row move(s) against {releases} release(s), past the "
            f"{limit:g} per release this run allows: a hole is made by a release and "
            "filled by one move, so more moves than releases means something other "
            "than a completion is rearranging the batch"
        )
