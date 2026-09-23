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

**Day 61: the width axis was a property of the read, and one read does not have
it.** Everything above is true of the Day-28 rectangle and of nothing else. That read
gathers a `[rows, width]` mapping and scores it into a `[rows, heads, 1, width]`
tensor, so the width is a dimension of two real allocations and a wider one costs
real memory: that is the entire reason it is in the set. Day 59's streamed read walks
`cdiv(context_lens[row], block)` tiles of its own row and holds one tile, so handing
it a wider mapping changes nothing it touches and nothing it holds. `streamed=True`
therefore collapses the width axis to a single bucket, the table's own
`max_model_len`, and the set stops being a product: 576 shapes at 256 rows and 8192
tokens become 9, which is exactly vLLM's capture list and for exactly its reason.

Two consequences are worth having in the head before reading the code. The *price*
has to be restated: `waste` measured against a rectangle nobody builds would report
99% on a read that touches every cell it is charged, so `cells_for` asks the bucket
set which read it belongs to and the currency follows. And the two halves have to
agree, which is what `check_read_matches` is: a streamed set under a rectangle read
hands the rectangle a full-width mapping from the first token onwards, which is
correct, silent, and the whole memory bound this module exists to impose, gone.
"""

from __future__ import annotations

from collections.abc import Sequence

from .compiled import RECOMPILE_LIMIT, ROW_BUCKETS, WIDTH_MULTIPLE, DecodeShape, bucket_for, round_up

#: How much of a bucketed rectangle may be cells nobody asked for before the
#: padding is the step. Deliberately loose: a decode batch is ragged anyway, and
#: this gate is aimed at the pathological corner (a 12-token context rounded up to
#: a 2048-token width) rather than at the ordinary overshoot.
DEFAULT_WASTE_LIMIT = 0.9


def row_axis(max_batch_size: int, row_buckets: Sequence[int] = ROW_BUCKETS) -> tuple[int, ...]:
    """The row buckets a cache of this size really has. Day 65, lifted out of the set.

    `DecodeBuckets` has computed this inline since Day 52. It comes out here because
    the split count is planned *over* the row axis and has to be known *before* the
    set is built, `plan_splits` being a max over `choose_splits` of every bucket in
    the list. Building a throwaway set to read its rows off would work and would also
    be a second construction whose arguments could drift from the real one's.

    The candidates that fit, plus the cache's own row count. The second half matters:
    `max_batch_size` is often not a power of two (a server picks 6), and without it a
    full batch would have no bucket at all.
    """
    if max_batch_size < 1:
        raise ValueError(f"a batch has at least one row; got {max_batch_size}")
    return tuple(
        sorted({int(b) for b in row_buckets if 0 < int(b) <= max_batch_size} | {max_batch_size})
    )


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
                    `count` is a capture list or a compile bill. Ignored entirely
                    when `streamed`, because then there is no axis to round.
    streamed:       Day 61. Whether the read this set belongs to is Day 59's, which
                    holds a tile instead of a rectangle. True collapses the width
                    axis to `max_model_len` alone and `count` stops being a product.
                    It is a fact about the *read*, passed in rather than inferred,
                    and `check_read_matches` is what stops the two from drifting.
    block:          the score tile that read folds, and the unit the price is stated
                    in once the width is gone. No default: a set that guessed a tile
                    would disagree with the read that has one, and the disagreement
                    would be a number in a table rather than a crash.
    splits:         Day 65. Chunks per row, when the read this set belongs to is the
                    split one, and 0 when it is not. It sits here for `block`'s
                    reason and one more. `block` is here because the price of a
                    streamed set is stated in tiles; `splits` is here because the
                    price of a split set is stated in tiles *and* in an arena, and
                    the arena is `rows * heads * splits * (head_dim + 2)` reserved
                    once at boot. It is a fact about the read, passed in rather than
                    inferred, and it requires `streamed` for the reason
                    `read_workspace_bytes` requires a tile: a split is a partition of
                    the streamed read's tiles, and a set that still buckets the width
                    has no such thing to partition.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_model_len: int,
        *,
        row_buckets: Sequence[int] = ROW_BUCKETS,
        width_multiple: int = WIDTH_MULTIPLE,
        streamed: bool = False,
        block: int = 0,
        splits: int = 0,
    ):
        if max_batch_size < 1:
            raise ValueError(f"a batch has at least one row; got {max_batch_size}")
        if max_model_len < 1:
            raise ValueError(f"a row holds at least one token; got {max_model_len}")
        if width_multiple < 1:
            raise ValueError(f"a width multiple is at least one; got {width_multiple}")
        if block < 0:
            raise ValueError(f"a tile holds at least one key; got {block}")
        if streamed and block < 1:
            raise ValueError(
                "a streamed bucket set is priced in score tiles and will not guess "
                f"one; got block={block}. Pass the same tile the read was built with"
            )
        if splits < 0:
            raise ValueError(f"a row is cut into at least one chunk; got {splits}")
        if splits and not streamed:
            raise ValueError(
                f"this set is priced for a split read of {splits} chunks and still "
                "buckets the width: a split is a partition of the streamed read's "
                "per-program tile, and a set that gathers a rectangle has no tile to "
                "partition. Pass streamed=True with it, or drop the chunks"
            )
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        self.width_multiple = width_multiple
        self.streamed = bool(streamed)
        self.block = int(block) if streamed else 0
        self.splits = int(splits)
        self.rows = row_axis(max_batch_size, row_buckets)
        if self.streamed:
            # One width, and it is the table's. Not "no width": a mapping is still a
            # 2-D tensor and a shape still has a second number, so what changes is
            # that the number is a constant of the cache rather than a function of
            # how far into the run this step is.
            self.widths = (max_model_len,)
        else:
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
        """The rectangle width this context is read at, never past the table.

        The refusals are kept on both arms and they are not symmetry for its own
        sake. A context longer than the table is a row the slot table cannot address
        whichever read is running, and a streamed read that quietly accepted it would
        walk tiles off the end of a mapping it was handed. What changes under
        `streamed` is only the answer: every legal context is read at the table's
        full width, because a wider mapping is free to a reader that never gathers.
        """
        if width < 1:
            raise ValueError(f"a rectangle is at least one column wide; got {width}")
        if width > self.max_model_len:
            raise ValueError(
                f"a context of {width} is longer than max_model_len "
                f"{self.max_model_len}: this is the table's width and not a rounding "
                "decision"
            )
        if self.streamed:
            return self.max_model_len
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
        """How many graphs the closed set implies. A product, and that is the trap.

        Still written as a product under `streamed`, where the second factor is 1.
        Special-casing it to `len(self.rows)` would hide the one thing this number is
        for, which is saying *why* a set is the size it is.
        """
        return len(self.rows) * len(self.widths)

    def cells_for(self, rows: int, width: int, query_len: int = 1) -> int:
        """Cells the read really touches for a step over `rows` rows and `width`.

        Day 61, and it exists because the old answer stopped being an answer. A
        bucketed rectangle touches `row_bucket * width_bucket` cells: every padded row
        and every padded column is gathered and scored before the mask throws it away,
        which is what `DecodeShape.cells` says and what `waste` has priced since Day
        52. A streamed read touches neither kind of padding. A padded row's
        `context_lens` entry is 0, so its program walks zero tiles; a padded *column*
        does not exist, because the loop bound is the row's own length and not the
        mapping's width.

        So the streamed charge is `rows * round_up(width, block)`: the real rows, each
        walking whole tiles over its own history. `round_up` and not `min` is the
        difference between work and memory, and both are real: `streamed_score_cells`
        prices the tile a program *holds*, which is one tile, and this prices every
        tile it *walks*, which is the last one rounded up. A four-token row under a
        32-key tile costs one whole tile of scoring, masked.

        The granularity is the shape's, so a ragged batch is charged its longest row
        on every row. That over-states the streamed read and it is the honest place
        to leave it: a `DecodeShape` has one width in it, and a per-row charge would
        need `context_lens` on the host, which is Day 48's synchronisation.
        """
        if not self.streamed:
            return self.shape_for(rows, width, query_len).cells
        if rows < 1:
            raise ValueError(f"a decode step has at least one row; got {rows}")
        # Validated for the same refusals the bucketed arm gets, and thrown away: the
        # streamed charge is over the real width, not the bucketed one.
        self.width_bucket(width)
        return rows * query_len * round_up(width, self.block)

    def render(self) -> str:
        """One line for a log: the axes, and what they multiply out to."""
        if self.streamed:
            cut = f" x {self.splits} splits" if self.splits else ""
            read = "split" if self.splits else "streamed"
            return (
                f"{len(self.rows)} row buckets {list(self.rows)} x 1 width of "
                f"{self.max_model_len}{cut} (the {read} read at a {self.block}-key "
                f"tile has no width axis) = {self.count} shapes"
            )
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
    """Cells the kernel computes over a run, padding included.

    Through `cells_for` since Day 61, so the answer is in the currency of the read
    this set belongs to. On the default that is exactly what it always was, the
    bucketed rectangle summed over the run; on a streamed set it is tiles walked,
    which is the only number that describes a read holding no rectangle at all.
    """
    return sum(buckets.cells_for(s.rows, s.context_width, s.query_len) for s in shapes)


def waste(shapes: Sequence[DecodeShape], buckets: DecodeBuckets) -> float:
    """Share of the computed cells that only exist to keep the shape constant.

    Same currency as Day 29's `waste_fraction` and Day 34's
    `prefill_padding_waste`, and it is the price of the closed set stated in the
    only unit that is comparable across days: work the machine did that nobody
    wanted.

    Day 61 is why this reads `padded_cells` rather than inlining the rectangle. A
    streamed set's width bucket is `max_model_len` on every step, so a waste computed
    against `DecodeShape.cells` would report 99% for a read that walks the same tiles
    it would have walked unbucketed. The rounding did not get worse; the rectangle it
    was being measured against stopped being built.
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


def check_read_matches(buckets: DecodeBuckets, read, *, armed: bool = True) -> None:
    """Refuse a bucket set and a decode read that disagree about the width axis.

    Day 61's correctness gate, and the only one this week that guards against
    corruption of a *server* rather than of a number. Everything else here is an
    assertion about shapes; this one is about memory.

    **A streamed set under a rectangle read is the failure that matters.** The set
    rounds every context to `max_model_len`, so `plan_decode` reads at the table's
    full width on the very first decode step of every request. The rectangle read
    then gathers all of it and scores a `[rows, heads, 1, max_model_len]` tensor: at
    256 rows, 32 heads and 8192 tokens that is the 268 MB Day 54 priced, materialised
    for a batch whose longest history is nine tokens, and it is materialised on step
    one rather than at the end of a long run. Nothing raises. The attention is right,
    the tokens are right, and the process is holding two orders of magnitude more
    workspace than the bucket set was there to bound.

    **The other direction is only waste, and it is still refused.** A rectangle set
    under a streamed read gives the streamed read a width axis it has no use for:
    every width bucket becomes its own graph, so the capture list is a product again
    and each member records the same tile loop. Harmless per step and a boot that
    spends 64x the startup seconds it needed to.

    **Day 65 adds the split read and the question changes shape twice.** First, the
    width clause stops keying on `streamed` and keys on `tiled`: a split read gathers
    no more than a streamed one, so a width bucket buys it nothing either, and a gate
    that asked the narrower question would refuse every legitimate split server. And
    second, the split adds an axis a bucket set is the only place to state. The arena
    is `rows * heads * splits * (head_dim + 2)`, reserved once at boot; a set priced
    for sixteen chunks under a read that runs one has bought fifteen sixteenths of a
    workspace nobody addresses, and a split read under a set priced in whole tiles is
    a launch nobody sized. Neither raises on its own. Both are a memory number that is
    wrong in a direction the process cannot see.

    Duck-typed on purpose, the way `check_pad_inert` takes a plan. `nanoserve.reads`
    imports `nanoserve.captured`, which imports this module, so a real import here
    would be a cycle; and the only thing this gate needs is four attributes any read
    that wants to be checkable can carry.
    """
    tiled = bool(getattr(read, "tiled", getattr(read, "streamed", False)))
    split = bool(getattr(read, "split", False))
    splits = int(getattr(read, "splits", 0))
    block = int(getattr(read, "block", 0))
    if buckets.streamed and not tiled:
        raise BucketsUnsound(
            f"this bucket set dropped its width axis and reads every step at "
            f"{buckets.max_model_len} tokens, and the read is the rectangle: it would "
            f"gather and score a full-width rectangle from the first decode step of "
            "every request, which is correct and unbounded. Build the cache with "
            "streamed_read on both halves, or bucket the width"
        )
    if tiled and not buckets.streamed:
        raise BucketsUnsound(
            f"this read is {getattr(read, 'mode', 'tiled')} and holds a {block}-key "
            f"tile, and the bucket set "
            f"still has {len(buckets.widths)} width buckets in it: the width buys this "
            f"read nothing, so the set is {buckets.count} graphs where {len(buckets.rows)} "
            "would do. Build the bucket set with streamed=True"
        )
    if tiled and block != buckets.block:
        raise BucketsUnsound(
            f"the read folds a {block}-key score tile and this set is priced in "
            f"{buckets.block}-key tiles: every cell count here would be a quotient of "
            "one read's numerator and another's denominator"
        )
    if buckets.splits and not split:
        raise BucketsUnsound(
            f"this set is priced for a split read of {buckets.splits} chunks and the "
            f"read is the {getattr(read, 'mode', 'unsplit')} one: the plan reserved "
            "an accumulator per (row, head, chunk) for the life of the process and "
            "nobody addresses it, so the arena is device memory this server will "
            "never read and the boot line that priced it is the only witness"
        )
    if split and not buckets.splits:
        raise BucketsUnsound(
            "this read is a split and the set is priced in whole tiles: the arena is "
            "the largest thing a split server reserves and the set is where its size "
            "is decided, so this is a launch nobody sized. Build the bucket set with "
            "the same splits the workspace was allocated for"
        )
    # Day 66. The one clause that is about *time* rather than configuration, so the
    # one a caller may excuse: an engine is built before its arena exists, and
    # `armed=False` is how its construction-time gate says "not yet" without
    # waiving the six clauses that are wrong at any time.
    if split and splits < 1 and armed:
        raise BucketsUnsound(
            f"this read is a split against a set of {buckets.splits} chunks and holds "
            "no workspace: the arena is allocated once and handed over before the "
            "first decode step, so this is a boot path that stopped one call short "
            "rather than a plan that disagrees with itself"
        )
    if split and splits and splits != buckets.splits:
        raise BucketsUnsound(
            f"the read was handed an arena of {splits} chunks and this set is priced "
            f"for {buckets.splits}: the chunk count is a launch constant baked into "
            "the compiled kernel, so one of these two numbers sized a workspace the "
            "grid will not match"
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
