"""The decode step's shape, rounded into a small closed set. Week 13, Day 52.

Three days have been spent making a decode step something a compiler, and then a
capture, can hold on to. Day 49 removed the graph breaks and found the guard that
mattered was on `table.num_tokens`, a Python integer. Day 50 moved the addressing
out of the forward so the only thing left to guard on is a tensor shape. Day 51
made that tensor a window on a persistent buffer, so the *address* stopped moving.

The shape is what is still moving, and it moves on both axes. The rectangle is
`[rows, max_ctx]`: `rows` is whatever the scheduler admitted this iteration, and
`max_ctx` is the longest history in the batch, which grows by one every step for
the whole run. So a 100-step generation presents 100 distinct shapes, `static` mode
asks for 100 builds, and dynamo abandons the frame after eight of them.

**Bucketing closes the set by rounding both axes up.** The batch is padded to the
next row bucket and the rectangle is read at the next width multiple, so a run over
one bucket presents one shape, forever. That is the whole module and the arithmetic
was written on Day 49 (`bucket_for`, `round_up`, `bucketed` in `nanoserve.compiled`);
what is here is the *policy* (which buckets exist for a given cache), the price, and
the gates. `DecodeBuckets` is deliberately constructed from the cache's own two
limits, `max_batch_size` and `max_model_len`, because a bucket the cache has no row
for and a width the slot table cannot hold are both a crash rather than a padding
decision.

**Closed is not the same as small, and `count` is the number that says which one
you got.** The set is `len(rows) * len(widths)`, a product, and the width axis is
unbounded in a way the row axis is not: 256 rows gives 9 row buckets, and 8192
tokens at a 128-token multiple gives 64 widths, so the "closed" set is 576 shapes.
That is worse than the open one, because every member is a real compile. The width
multiple is the knob, and making it coarse buys the budget back in padding:
2048-token widths over the same model is 4 of them and 36 shapes total. vLLM does
not pay this at all, because its capture list is over batch sizes only: its kernel
walks a block table and reads the length from a tensor, so the context axis never
reaches a guard. nanoserve buckets both axes because its reference read gathers a
`[rows, width]` rectangle of K/V before it masks, which makes the width a real
tensor dimension. `check_capture_budget` is that difference, priced.

**A padded row has to be inert in two directions and only one of them is obvious.**
Reading is easy: its `context_lens` entry is 0, so the mask covers its whole row and
the softmax runs over `finfo.min` everywhere, which is finite (Day 27's choice of
`finfo.min` over `-inf`, paying off in a place it was not written for) and is thrown
away by the caller anyway. *Writing* is the half that can corrupt: a decode write is
one `index_put` over the whole padded batch, so a padded row writes K/V somewhere,
and "somewhere" must not be a slot a sequence owns. So the cache keeps a **sink
slot**, one row past the end of the pool the allocator knows about: a legal address
that no block ever maps to and therefore no `BlockTable` can ever name. Slicing the
write instead (`k[:real_rows]`) would work and would put a Python integer that
changes every step back inside the traced region, which is exactly the guard Day 50
existed to remove.

**And the padding is not free.** A padded row attends over a padded history and the
kernel computes every cell of it. `waste` is the fraction in the same currency as
Day 29's `waste_fraction`, Day 34's `prefill_padding_waste` and Day 50's
`padding_share`: the fourth time this engine has bought a fixed rectangle and paid
for the corners. The trade is real work per step against a graph that does not have
to be rebuilt, and it is only worth taking when something downstream can actually
reuse the graph.
"""

from __future__ import annotations

from collections.abc import Sequence

from .compiled import RECOMPILE_LIMIT, ROW_BUCKETS, WIDTH_MULTIPLE, DecodeShape, bucket_for, round_up

#: How much of a bucketed rectangle may be cells nobody asked for before the
#: padding is the step. Deliberately loose: a decode batch is ragged anyway, and
#: this gate is aimed at the pathological corner (a 12-token context rounded up to
#: a 2048-token width) rather than at the ordinary overshoot.
DEFAULT_WASTE_LIMIT = 0.9


class BucketsUnsound(AssertionError):
    """A bucketed step is not the closed, inert thing bucketing promises.

    An `AssertionError` for the same reason `PlanUnsound` and `SlotsUnsound` are:
    these are checks a test suite runs and a benchmark asserts. None of them raise
    on their own. A shape outside the set is a silent recompile, and a padded row
    that is not inert writes a made-up token's K/V into a real sequence's history
    and the run keeps producing plausible text.
    """


class DecodeBuckets:
    """The shapes one cache is allowed to present, and the rounding that gets there.

    max_batch_size: the cache's row count. It is a hard ceiling and not a rounding
                    target: there is no row to pad into past it, so a batch bigger
                    than this is refused rather than clamped.
    max_model_len:  the slot table's width, for the same reason. The largest width
                    bucket is `max_model_len` itself even when that is not a
                    multiple, because a rectangle wider than the table it is a
                    window on does not exist.
    row_buckets:    candidate row counts, filtered to those the cache has rows for.
                    Powers of two, which is vLLM's cudagraph capture list and for
                    the same reason: the batch is bounded and small, so doubling
                    keeps the set logarithmic.
    width_multiple: the context axis rounds to a multiple instead, because widths
                    run to thousands and doubling would pad a 600-token history to
                    1024. A multiple bounds the overshoot by the multiple itself
                    whatever the length, and it is the knob that decides whether
                    `count` is a capture list or a compile bill.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_model_len: int,
        *,
        row_buckets: Sequence[int] = ROW_BUCKETS,
        width_multiple: int = WIDTH_MULTIPLE,
    ):
        if max_batch_size < 1:
            raise ValueError(f"a batch has at least one row; got {max_batch_size}")
        if max_model_len < 1:
            raise ValueError(f"a row holds at least one token; got {max_model_len}")
        if width_multiple < 1:
            raise ValueError(f"a width multiple is at least one; got {width_multiple}")
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        self.width_multiple = width_multiple
        # The candidates that fit, plus the cache's own row count. The second half
        # matters: `max_batch_size` is often not a power of two (a server picks 6),
        # and without it a full batch would have no bucket at all.
        self.rows = tuple(
            sorted({int(b) for b in row_buckets if 0 < int(b) <= max_batch_size} | {max_batch_size})
        )
        widths = []
        width = width_multiple
        while width < max_model_len:
            widths.append(width)
            width += width_multiple
        widths.append(max_model_len)
        self.widths = tuple(widths)

    # --- rounding -------------------------------------------------------------

    def row_bucket(self, rows: int) -> int:
        """The batch this many rows is padded up to."""
        return bucket_for(rows, self.rows)

    def width_bucket(self, width: int) -> int:
        """The rectangle width this context is read at, never past the table."""
        if width < 1:
            raise ValueError(f"a rectangle is at least one column wide; got {width}")
        if width > self.max_model_len:
            raise ValueError(
                f"a context of {width} is longer than max_model_len "
                f"{self.max_model_len}: this is the table's width and not a rounding "
                "decision"
            )
        return min(round_up(width, self.width_multiple), self.max_model_len)

    def shape_for(self, rows: int, width: int, query_len: int = 1) -> DecodeShape:
        """The shape a step over `rows` rows and `width` of context really runs."""
        return DecodeShape(
            rows=self.row_bucket(rows),
            context_width=self.width_bucket(width),
            query_len=query_len,
        )

    # --- what it costs --------------------------------------------------------

    @property
    def shapes(self) -> tuple[DecodeShape, ...]:
        """Every shape this cache can present. The capture list, in full."""
        return tuple(
            DecodeShape(rows=r, context_width=w) for r in self.rows for w in self.widths
        )

    @property
    def count(self) -> int:
        """How many graphs the closed set implies. A product, and that is the trap."""
        return len(self.rows) * len(self.widths)

    def render(self) -> str:
        """One line for a log: the two axes and what they multiply out to."""
        return (
            f"{len(self.rows)} row buckets {list(self.rows)} x {len(self.widths)} "
            f"width buckets of {self.width_multiple} up to {self.max_model_len} = "
            f"{self.count} shapes"
        )


# --- what a run looks like through them ---------------------------------------------


def bucket_run(shapes: Sequence[DecodeShape], buckets: DecodeBuckets) -> tuple[DecodeShape, ...]:
    """The shapes a run really presents once every step is rounded up.

    The one measurement the day is about: hand it a shape history off a benchmark
    or off `CompiledDecode.shapes`, and the number of distinct entries is how many
    graphs the same run asks for with bucketing on.
    """
    return tuple(buckets.shape_for(s.rows, s.context_width, s.query_len) for s in shapes)


def padded_cells(shapes: Sequence[DecodeShape], buckets: DecodeBuckets) -> int:
    """Cells the kernel computes over a run, padding included."""
    return sum(s.cells for s in bucket_run(shapes, buckets))


def waste(shapes: Sequence[DecodeShape], buckets: DecodeBuckets) -> float:
    """Share of the computed cells that only exist to keep the shape constant.

    Same currency as Day 29's `waste_fraction` and Day 34's
    `prefill_padding_waste`, and it is the price of the closed set stated in the
    only unit that is comparable across days: work the machine did that nobody
    wanted.
    """
    padded = padded_cells(shapes, buckets)
    if not padded:
        return 0.0
    real = sum(s.cells for s in shapes)
    return (padded - real) / padded


# --- gates ---------------------------------------------------------------------------


def check_shape_in_set(shape: DecodeShape, buckets: DecodeBuckets) -> None:
    """Refuse a step whose shape is not one the bucket set promised.

    The gate that says bucketing is actually on. A shape outside the set is a build
    nobody budgeted for, and under a capture it is worse than a build: there is no
    recorded graph with those dimensions, so the step falls back to eager and the
    only symptom is that the run got slower.
    """
    want = buckets.shape_for(shape.rows, shape.context_width, shape.query_len)
    if shape != want:
        raise BucketsUnsound(
            f"{shape.rows} x {shape.context_width} is not one of this cache's "
            f"{buckets.count} bucketed shapes: it rounds to {want.rows} x "
            f"{want.context_width}, so this step was planned without buckets"
        )


def check_run_closed(
    shapes: Sequence[DecodeShape], buckets: DecodeBuckets, *, limit: int = RECOMPILE_LIMIT
) -> None:
    """Refuse a run that presented an unbucketed shape or too many bucketed ones.

    Two failures in one gate because they are two halves of one claim. Every shape
    has to be in the set, or bucketing did not happen; and the number of distinct
    shapes has to stay under the compiler's cache size, or it happened and did not
    help. Day 49's `falls_back` predicts the second half off an open set; this is
    the same prediction once the set is closed on purpose.
    """
    if limit < 1:
        raise ValueError(f"a recompile limit is at least one; got {limit}")
    for shape in shapes:
        check_shape_in_set(shape, buckets)
    distinct = len(set(shapes))
    if distinct > limit:
        raise BucketsUnsound(
            f"this run presented {distinct} distinct bucketed shapes against a limit "
            f"of {limit}: the set is closed and still bigger than the compiler will "
            "hold, so widen the width multiple or narrow the row buckets"
        )


def check_capture_budget(buckets: DecodeBuckets, *, limit: int = RECOMPILE_LIMIT) -> None:
    """Refuse a bucket set that is closed and too big to be worth closing.

    `count` is a product, so the width axis decides it: 9 row buckets against a
    128-token multiple over 8192 tokens is 576 graphs, each one a real compile or a
    real capture with its own memory. Closing an unbounded set into a large finite
    one is not obviously progress, and this is the line that says where the two
    stop being the same thing.
    """
    if limit < 1:
        raise ValueError(f"a capture budget is at least one shape; got {limit}")
    if buckets.count > limit:
        raise BucketsUnsound(
            f"this cache's bucket set is {buckets.count} shapes ({buckets.render()}) "
            f"against a budget of {limit}: a closed set this large is a compile bill, "
            "not a capture list. The width multiple is the knob, and it is paid for "
            "in padding"
        )


def check_waste_bounded(
    shapes: Sequence[DecodeShape],
    buckets: DecodeBuckets,
    *,
    limit: float = DEFAULT_WASTE_LIMIT,
) -> None:
    """Refuse a run where the rounding computes far more than the run asked for."""
    if not 0 < limit < 1:
        raise ValueError(f"a waste limit is a fraction; got {limit}")
    share = waste(shapes, buckets)
    if share > limit:
        raise BucketsUnsound(
            f"{share:.0%} of the bucketed cells are padding against a limit of "
            f"{limit:.0%}: {padded_cells(shapes, buckets)} cells computed for "
            f"{sum(s.cells for s in shapes)} that were wanted"
        )


def check_pad_inert(plan) -> None:
    """Refuse a bucketed plan whose padded rows can touch anything real.

    The correctness gate of the day, and it has two halves because a padded row has
    two ways to matter. It must read nothing, which is `context_lens == 0`: the
    whole row is masked, so what its rectangle points at never reaches a softmax.
    And it must write nowhere, which is the sink slot: a decode write is one
    `index_put` over the padded batch, so the row writes K/V unconditionally and the
    only safe destination is an address no `BlockTable` can name.

    The third clause is the one that catches a wiring mistake rather than a design
    one: no *real* row may write to the sink either. A real row pointed at the sink
    loses its token, the same step's read looks for it in the pool, and the sequence
    quietly attends over a stale slot.
    """
    pad = getattr(plan, "pad_rows", 0)
    real = plan.batch_size
    if pad and plan.sink_slot is None:
        raise BucketsUnsound(
            f"this plan pads {pad} row(s) and names no sink slot: a padded row's "
            "write has to go somewhere, and without a sink it goes into the pool"
        )
    for i in range(real, real + pad):
        length = int(plan.context_lens[i])
        if length:
            raise BucketsUnsound(
                f"padded row {i} claims {length} cached tokens: a padded row exists "
                "to keep the shape constant and must attend over nothing"
            )
        slot = int(plan.write_slots[i])
        if slot != plan.sink_slot:
            raise BucketsUnsound(
                f"padded row {i} writes to slot {slot} and the sink is "
                f"{plan.sink_slot}: this row's made-up token would land in the pool, "
                "in whatever sequence owns that slot"
            )
    if plan.sink_slot is not None:
        for i in range(real):
            if int(plan.write_slots[i]) == plan.sink_slot:
                raise BucketsUnsound(
                    f"row {plan.rows[i]} writes to the sink slot {plan.sink_slot}: its "
                    "token would be thrown away and the same step's read would look "
                    "for it in the pool"
                )
