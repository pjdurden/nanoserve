"""Flash-decoding: the tail becomes parallelism. Week 14, Day 63.

Day 62 put a number on the thing this module exists to fix. A batched decode read
launches one program per `(row, query head)`, each walking its own row's history, and
`LaunchWork` says what that costs two ways: `work` is the sum over programs, which is
what an oversubscribed grid collects, and `wave` is the max over them, which is all a
fully resident grid gets. They differ by exactly the load imbalance, `tail / mean`,
and on a long-tail decode batch that is 4.27x: three quarters of the grid finished
and then waited for one row.

The wait is structural, not a scheduling accident. A row's online softmax is a
sequential fold, so the program that owns a four-thousand-token row has four thousand
tokens of serial work no matter how empty the card is around it.

Flash-decoding is the answer vLLM and SGLang ship, and it is one idea: stop making
one program own a whole row. Cut the history into fixed chunks, give each chunk its
own program with its own running max, denominator and weighted-V accumulator, and
reduce the partials in a second pass. The grid grows a third axis, the longest
program shrinks by the split count, and the same tiles get walked by more of the
machine at once.

Three things below are the whole day.

**The partition is in whole tiles, and that is a refusal and not a convention.** A
program's inner loop is `arange(0, BLOCK_N)` from its chunk's start, so a chunk that
begins at key 48 of a 32-key tiling begins in the middle of a tile another program
also owns. `partition_width` rounds the chunk up to a tile boundary, and a
caller-supplied `keys_per_split` that is not a multiple of the block is refused
rather than rounded, because rounding it would silently change which keys a program
reads.

**The split count comes from the mapping's width, which is a shape, and never from
the batch's longest row, which is a tensor.** Asking a tensor for its max is Day 48's
synchronisation and Day 49's graph break; worse, a split count that changes with the
batch is a different grid on every step, which is a different graph on every step,
which is Day 61's capture list back again. The width is a Python int the caller
already has, and under Day 61's streamed bucket set it is `max_model_len` for the
life of the process. So `choose_splits` takes `context_width` and the grid and
nothing else, and the split count is a launch constant.

**The partials are a real workspace, and this is the first thing since Day 59 that
gives memory back.** Day 59 took the `[rows, heads, 1, max_ctx]` score rectangle away,
268 MB at serving size. A split launch needs `[rows, heads, splits]` of a max, a
denominator and a `head_dim`-wide accumulator, in fp32 because another program is
about to rescale them by `exp(m - M)`. `SplitPlan.partial_bytes` prices it, and it
grows linearly in exactly the number the tail shrinks by, so the split count is a
memory decision as much as a latency one.

The layering is Day 62's, for Day 62's reason: every integer the kernel turns into a
pointer or a loop bound is ordinary Python here, tested on any box, and the two
`@triton.jit` bodies are held to `paged_attention_batched_reference` by tests gated on
a real device. `paged_attention_split_kernel` is the same two passes as tlsim
programs, which is the model the jitted version transcribes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paged_attention import cdiv
from .tlsim import arange, launch, load, store
from .triton_batched_attention import DEFAULT_BLOCK, check_batched_inputs, slot_index_dtype
from .triton_paged_attention import has_triton, next_power_of_2, select_backend

try:  # Triton rides along with the Linux GPU torch wheel; a CPU wheel has no module.
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - exercised by whichever box you are on
    triton = None
    tl = None

#: The smallest chunk worth giving its own program, in keys. Below it the second pass
#: and the workspace cost more than the shortened tail is worth: a program that walks
#: two tiles still pays a launch, a query load and a partial store. vLLM's paged
#: attention uses a 512-key partition for the same trade; this is that number, named.
DEFAULT_PARTITION = 512

#: A stand-in for how many programs the card holds resident at once, which is what
#: decides whether a split buys anything. The real number is per-device (SMs times
#: occupancy) and this module has no device, so it is one constant, stated rather
#: than hidden, and `choose_splits` takes an override.
DEFAULT_TARGET_PROGRAMS = 2048

#: The partials' dtype, and it is not the pool's. A partial is a max and two running
#: sums that a second program rescales by `exp(m - M)`; doing that in fp16 is where
#: the accuracy of the whole read would go.
PARTIAL_DTYPE = torch.float32

__all__ = [
    "DEFAULT_PARTITION",
    "DEFAULT_TARGET_PROGRAMS",
    "PARTIAL_DTYPE",
    "SplitPlan",
    "SplitUnsound",
    "check_partials",
    "choose_splits",
    "partial_splits",
    "paged_attention_split",
    "paged_attention_split_kernel",
    "paged_attention_split_triton",
    "partition_width",
    "plan_splits",
    "reduce_partials",
    "split_plan",
]


class SplitUnsound(AssertionError):
    """A split launch was handed a workspace it cannot address.

    An assertion and not a `ValueError`, for the reason `CaptureUnsound` is one:
    every failure here is a thing the caller believed about a buffer that is not
    true of the buffer, and none of them raises anything on its own. Both kernels
    address the partials flat, so a workspace with the wrong stride, the wrong
    dtype or too few slots does not fault. It reads and writes somebody else's
    slot, the reduction folds it, and the read returns a finite, plausible vector
    that is not the attention over this row.
    """


def partition_width(
    context_width: int,
    block: int,
    splits: int | None = None,
    keys_per_split: int | None = None,
) -> tuple[int, int]:
    """Cut a mapping's width into chunks of whole tiles: `(splits, keys_per_split)`.

    context_width:  the mapping's row stride, the widest history any row can hold.
                    A shape, deliberately: see the module docstring.
    block:          keys folded per program step, the SRAM tile.
    splits:         how many chunks to aim for. The chunk is `cdiv(width_tiles,
                    splits) * block`, so the count that comes back can be *smaller*
                    than the one asked for: four tiles cut three ways is two chunks
                    of two tiles, because the floor on a chunk is one tile and the
                    chunk is rounded up rather than the last one left short.
    keys_per_split: the chunk directly, which is the vLLM shape (a fixed partition,
                    and the split count falls out of the width). Must be a positive
                    multiple of `block`.

    Exactly one of `splits` and `keys_per_split`: they name the same partition, and
    only one of them can be the reason for it.

    Returns the pair the launch really uses. The chunk is a multiple of the tile
    because a program's inner loop starts at `chunk * s` and ramps `arange(0,
    BLOCK_N)` from there; a chunk of 48 keys would put split 1's first tile halfway
    through split 0's last one, and the two programs would both fold the same keys.
    """
    if block < 1:
        raise ValueError(f"a tile holds at least one key; got {block}")
    if context_width < 1:
        raise ValueError(f"a mapping is at least one key wide; got {context_width}")
    if (splits is None) == (keys_per_split is None):
        raise ValueError(
            "a partition is named either by a split count or by a chunk size, not by "
            "both and not by neither: `splits` asks for a number of programs per row "
            "and `keys_per_split` asks for a chunk, and each implies the other"
        )
    if keys_per_split is not None:
        if keys_per_split < 1:
            raise ValueError(f"a chunk holds at least one key; got {keys_per_split}")
        if keys_per_split % block:
            raise ValueError(
                f"a chunk is a whole number of tiles: keys_per_split {keys_per_split} is "
                f"not a multiple of the tile {block}, so one split would begin partway "
                "through a tile another split also walks"
            )
        return cdiv(context_width, keys_per_split), keys_per_split
    if splits < 1:
        raise ValueError(f"a row is cut into at least one split; got {splits}")
    chunk = cdiv(cdiv(context_width, block), splits) * block
    return cdiv(context_width, chunk), chunk


def choose_splits(
    rows: int,
    n_q: int,
    context_width: int,
    block: int = DEFAULT_BLOCK,
    target_programs: int = DEFAULT_TARGET_PROGRAMS,
    partition: int = DEFAULT_PARTITION,
) -> int:
    """How many ways to cut a row, from the grid and the width and nothing else.

    rows, n_q:       the unsplit launch grid. Their product is how many programs the
                     read already has, which is the only thing that says whether more
                     would help.
    context_width:   the mapping's width. A shape, not `context_lens.max()`: see the
                     module docstring for why that distinction is the whole design.
    block:           the SRAM tile, the floor a chunk cannot go below.
    target_programs: roughly how many programs keep the card busy.
    partition:       the smallest chunk worth its own program.

    Two bounds, and the answer is the smaller of them.

    *The grid bound.* A batch of 256 rows on 32 heads is 8192 programs and
    oversubscribes anything; there the hardware is already a queue, a skipped tile is
    a skipped wave, and splitting adds a workspace and a second pass for no
    parallelism at all. `cdiv(target_programs, rows * n_q)` is how many splits it
    takes to fill the grid, and it is 1 for exactly that batch.

    *The length bound.* A chunk shorter than `partition` pays a launch, a query load
    and a partial store to walk a handful of tiles, so `context_width // partition`
    is a floor on the chunk expressed as a ceiling on the count. A 256-token width is
    below it and is never split.

    This is why vLLM ships both kernels rather than replacing one: the split is a win
    for a long context on a small batch, and a loss for a short context on a large
    one, and which of those a server is in changes minute to minute.
    """
    if rows < 1:
        raise ValueError(f"a decode batch has at least one row; got {rows}")
    if n_q < 1:
        raise ValueError(f"a launch needs at least one query head; got {n_q}")
    if context_width < 1:
        raise ValueError(f"a mapping is at least one key wide; got {context_width}")
    if block < 1:
        raise ValueError(f"a tile holds at least one key; got {block}")
    if target_programs < 1:
        raise ValueError(f"a launch fills at least one program; got {target_programs}")
    if partition < block:
        raise ValueError(
            f"a partition is at least one tile: partition {partition} < tile {block}, "
            "and a chunk smaller than a tile is a tile walked by two programs"
        )
    by_length = max(1, context_width // partition)
    by_grid = cdiv(target_programs, rows * n_q)
    return max(1, min(by_length, by_grid))


def plan_splits(
    row_buckets,
    n_q: int,
    context_width: int,
    block: int = DEFAULT_BLOCK,
    target_programs: int = DEFAULT_TARGET_PROGRAMS,
    partition: int = DEFAULT_PARTITION,
) -> int:
    """One split count for a whole capture list. Day 64.

    row_buckets: the row axis of the bucket set this list is planned over, which is
                 every batch size a decode step can round to.
    n_q, context_width, block, target_programs, partition: `choose_splits`', because
                 this is that function asked once per bucket.

    `choose_splits` answers per launch, and a capture list is not one launch: it is a
    graph per shape, all of them replaying against *one* workspace, and a workspace is
    allocated once at a size. So the list has to collapse the per-bucket answers to a
    single number, and the max is the only direction that keeps the split.

    Rounding a bucket's count *up* costs it chunks it does not fill, and Day 63 made
    that free: a chunk past a row's end walks nothing, stores `-inf, 0, 0`, and drops
    out of the reduction with no branch. Rounding *down* is the expensive direction,
    because the bucket that wanted the most splits is the one-row batch, which is the
    only batch a split was ever for.

    The two maxima are at opposite ends of the list and that is the day's arithmetic
    surprise. The split *axis* is sized by the narrowest bucket, because a small grid
    is what needs filling. The *arena* is `rows * n_q * splits * (head_dim + 2)` and
    is sized by the widest, because `choose_splits` is a ceiling division of a program
    target, so `rows * splits(rows)` rises to that target and flattens rather than
    falling. Allocating the rectangle of both maxima therefore buys strictly more than
    any single shape in the list needs, and the surplus is the price of one arena.
    """
    rows = tuple(int(r) for r in row_buckets)
    if not rows:
        raise ValueError(
            "a capture list has at least one row bucket; got none. A split count for "
            "an empty list is a workspace for a launch that cannot happen"
        )
    return max(
        choose_splits(r, n_q, context_width, block, target_programs, partition) for r in rows
    )


def partial_splits(partials) -> int:
    """The split axis of a handed-in workspace, which is what the launch must use.

    A planned buffer is a launch constant, so it *answers* the split question rather
    than being graded against an answer from somewhere else: a kernel handed a
    workspace and no `splits` takes the count the plan allocated. Naming both is
    still allowed and still checked, because a caller who says 8 and hands a
    four-wide arena has one of the two numbers wrong and should be told which.
    """
    tensors = tuple(partials)
    if len(tensors) != 3:
        raise SplitUnsound(
            f"a split workspace is three buffers, a max, a denominator and an "
            f"accumulator; got {len(tensors)}"
        )
    if tensors[0].ndim != 3:
        raise SplitUnsound(
            f"part_max has shape {tuple(tensors[0].shape)} and a workspace's max is "
            "[rows, heads, splits]: there is no split axis to read a count off"
        )
    return int(tensors[0].shape[2])


def check_partials(
    partials,
    *,
    rows: int,
    n_q: int,
    splits: int,
    head_dim: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refuse a handed-in workspace the launch about to happen cannot address. Day 64.

    partials: `(part_max, part_denom, part_acc)`, already narrowed to this launch's
              rows. `[rows, n_q, splits]` twice and `[rows, n_q, splits, head_dim]`.
    rows, n_q, splits, head_dim, device: the launch, exactly.

    Returns the three tensors, so a caller can use the checked thing rather than the
    thing it passed in.

    **Contiguity is the load-bearing clause and the reason this is a gate.** Both
    passes address the workspace flat, `(row * n_q + head) * splits + split`, because
    that is the only form `tl.store` takes. A buffer allocated at the list's widest
    split count and narrowed to a smaller one with `buffer[:, :, :splits]` is a
    perfectly good tensor of exactly the right shape whose stride is the *allocated*
    split count, and the flat arithmetic does not know that. Every program past the
    first would then read and write a slot belonging to some other `(row, head)`, and
    nothing would fault: the reduction folds whatever it finds and returns a finite
    vector. A window on the outermost axis, `buffer[:rows]`, is the one narrowing the
    flat form still addresses, which is why the row axis is the only one a launch is
    allowed to take less than all of.

    The dtype clause is the other half of Day 63's fp32 decision, moved to where a
    caller can trip over it. A partial is rescaled by `exp(m - M)` in a *different*
    program, so it makes a round trip through memory between the two, and a buffer
    someone allocated in the pool's dtype to save bytes would lose the range that trip
    needs.
    """
    tensors = tuple(partials)
    if len(tensors) != 3:
        raise SplitUnsound(
            f"a split workspace is three buffers, a max, a denominator and an "
            f"accumulator; got {len(tensors)}"
        )
    part_max, part_denom, part_acc = tensors
    want = {
        "part_max": (part_max, (rows, n_q, splits)),
        "part_denom": (part_denom, (rows, n_q, splits)),
        "part_acc": (part_acc, (rows, n_q, splits, head_dim)),
    }
    device = torch.device(device)
    for name, (tensor, shape) in want.items():
        if tuple(tensor.shape) != shape:
            raise SplitUnsound(
                f"{name} has shape {tuple(tensor.shape)} and this launch addresses "
                f"{shape}: the grid is (rows, heads, splits) and every program stores "
                "one slot of it, so a workspace that is not exactly that shape is "
                "either overrun or read past"
            )
        if not tensor.is_contiguous():
            raise SplitUnsound(
                f"{name} is not contiguous, so its stride is not the one both passes "
                "address it with: the partials are indexed flat as "
                "`(row * n_q + head) * splits + split`, and a narrowing of any axis "
                "but the rows keeps the buffer's stride and silently moves every "
                "program onto another program's slot"
            )
        if tensor.dtype != PARTIAL_DTYPE:
            raise SplitUnsound(
                f"{name} is {tensor.dtype} and a partial is {PARTIAL_DTYPE}: pass two "
                "rescales this value by `exp(m - M)` after it has been through memory, "
                "and that round trip is the one place in the read that needs the range"
            )
        if tensor.device != device:
            raise SplitUnsound(
                f"{name} is on {tensor.device} and this launch runs on {device}: a "
                "kernel takes a pointer, and a pointer into host memory is not a "
                "slower read, it is a fault or a wrong answer"
            )
    return part_max, part_denom, part_acc


@dataclass(frozen=True)
class SplitPlan:
    """What a split launch asks for, and what it gives back. Day 63.

    rows, n_q:      the unsplit grid's two axes.
    head_dim:       channels in one head, which is what makes an accumulator a row of
                    the workspace rather than a scalar.
    block:          keys folded per program step.
    splits:         chunks per row, the grid's third axis. What the launch really
                    uses, which may be fewer than were asked for: see
                    `partition_width`.
    keys_per_split: the chunk, always a whole number of tiles.
    context_width:  the mapping's width, which is what was partitioned.
    split_tiles:    `[rows][splits]` tiles each chunk of each row actually walks.
                    Zero is a real entry and the interesting one: a chunk past a short
                    row's end launches, loads its bound, finds nothing, stores an
                    empty partial and retires.

    Frozen, like `BatchedGeometry` and `LaunchWork`, because it describes a launch
    that is about to happen and a description that can be edited afterwards is a
    description of nothing.
    """

    rows: int
    n_q: int
    head_dim: int
    block: int
    splits: int
    keys_per_split: int
    context_width: int
    split_tiles: tuple[tuple[int, ...], ...]

    @property
    def programs(self) -> int:
        """Programs the first pass starts: `rows * n_q * splits`, one partial each."""
        return self.rows * self.n_q * self.splits

    @property
    def unsplit_programs(self) -> int:
        """What Day 62's grid was: one program per (row, query head)."""
        return self.rows * self.n_q

    @property
    def tiles(self) -> int:
        """Tiles walked, summed over every program. Identical to the unsplit count.

        Conservation, and it is why the split is free in work terms: the same keys
        are read either way, by more programs. Every other number here is about *when*
        the tiles are walked and never about how many.
        """
        return self.n_q * sum(sum(row) for row in self.split_tiles)

    @property
    def tail_tiles(self) -> int:
        """Tiles the slowest program walks, which is now the longest *chunk*."""
        return max(max(row) for row in self.split_tiles)

    @property
    def unsplit_tail_tiles(self) -> int:
        """Tiles the slowest program would walk unsplit: the longest *row*."""
        return max(sum(row) for row in self.split_tiles)

    @property
    def mean_tiles(self) -> float:
        """Tiles the average program walks, counting the empty ones.

        An idle program is still a scheduled program, so it belongs in the mean. That
        makes the post-split imbalance look worse than the tail alone suggests, and
        it should: `idle_fraction` is the other half of what the split cost.
        """
        return self.tiles / self.programs

    @property
    def imbalance(self) -> float:
        """How many times the slowest program outlasts the average one, after the cut.

        The number Day 62's `LaunchWork.imbalance` is, recomputed on the new grid. It
        does not go to 1.0, because cutting the width uniformly does not cut a ragged
        batch uniformly: the short rows' chunks are empty, not short.
        """
        return self.tail_tiles / self.mean_tiles

    @property
    def wave_speedup(self) -> float:
        """Unsplit tail over split tail: what a fully resident grid gets back.

        1.0 at `splits=1`, and capped by how many tiles the longest row holds rather
        than by the split count: cutting a 1024-wide mapping 32 ways does nothing for
        a batch whose longest row is two tiles, because the other 30 chunks are empty.
        """
        return self.unsplit_tail_tiles / self.tail_tiles

    @property
    def idle_programs(self) -> int:
        """Programs that walk no tiles at all, and still launch and still store.

        The cost the grid axis imposes. A split is an axis of the launch, so every row
        gets `splits` programs whether its history reaches them or not, and on a
        long-tail batch most of them do not.
        """
        return self.n_q * sum(1 for row in self.split_tiles for tiles in row if tiles == 0)

    @property
    def idle_fraction(self) -> float:
        """`idle_programs / programs`, the share of the grid that was manufactured."""
        return self.idle_programs / self.programs

    @property
    def partial_dtype(self) -> torch.dtype:
        """fp32, whatever the pool holds. See `PARTIAL_DTYPE`."""
        return PARTIAL_DTYPE

    @property
    def partial_elements(self) -> int:
        """Numbers the workspace holds: one max, one denominator, one accumulator row
        per `(row, head, split)`."""
        return self.programs * (self.head_dim + 2)

    @property
    def partial_bytes(self) -> int:
        """The workspace in bytes, at fp32."""
        return self.partial_elements * torch.finfo(PARTIAL_DTYPE).bits // 8

    @property
    def partial_mib(self) -> float:
        """The same number a capture plan would print."""
        return self.partial_bytes / 2**20

    def render(self) -> str:
        """One line, for a bench table: the trade rather than the win."""
        return (
            f"{self.rows} rows x {self.n_q} heads x {self.splits} splits of "
            f"{self.keys_per_split} keys: {self.programs} programs, tail "
            f"{self.tail_tiles} of {self.unsplit_tail_tiles} tiles "
            f"(wave {self.wave_speedup:.2f}x, {self.idle_fraction:.0%} idle, "
            f"{self.partial_mib:.2f} MiB partials)"
        )


def split_plan(
    context_lens,
    n_q: int,
    head_dim: int,
    block: int = DEFAULT_BLOCK,
    splits: int | None = None,
    keys_per_split: int | None = None,
    context_width: int | None = None,
) -> SplitPlan:
    """What a split launch walks, per row and per chunk. Host arithmetic only.

    context_lens:  per-row history lengths, as a tensor (what the planned decode path
                   holds) or any sequence of ints.
    n_q:           query heads. It multiplies the program count and the tile total
                   and cancels out of every ratio, which is exactly why it is worth
                   printing next to them.
    head_dim:      channels in one head, for the workspace.
    block:         keys folded per program step.
    splits /
    keys_per_split: the partition, exactly one of them; see `partition_width`.
    context_width: the mapping's width; defaults to the longest row, which is what an
                   unbucketed mapping is.

    Walks no memory and touches no pool, so a bench can ask it about a 256-row batch
    at an 8192-token width on a laptop. This is the arithmetic half of the day, and
    the half a box with no card can check.
    """
    if n_q < 1:
        raise ValueError(f"a launch needs at least one query head; got {n_q}")
    if head_dim < 1:
        raise ValueError(f"a head has at least one channel; got {head_dim}")
    if block < 1:
        raise ValueError(f"a tile holds at least one key; got {block}")
    lens = [int(n) for n in context_lens]
    if not lens:
        raise ValueError("a decode batch has at least one row; got no context lengths")
    if min(lens) < 1:
        raise ValueError(
            "context_lens must be at least 1 for every row: a query with no visible "
            "key softmaxes over nothing, and here no split of it walks a tile either, "
            "so the reduction would divide zero by zero"
        )
    longest = max(lens)
    width = longest if context_width is None else context_width
    if longest > width:
        raise ValueError(
            f"a row cannot hold more history than the mapping's width: longest "
            f"{longest} > context_width {width}"
        )
    count, chunk = partition_width(width, block, splits=splits, keys_per_split=keys_per_split)
    tiles = tuple(
        tuple(cdiv(max(0, min(s * chunk + chunk, ctx) - s * chunk), block) for s in range(count))
        for ctx in lens
    )
    return SplitPlan(
        rows=len(lens),
        n_q=n_q,
        head_dim=head_dim,
        block=block,
        splits=count,
        keys_per_split=chunk,
        context_width=width,
        split_tiles=tiles,
    )


def reduce_partials(
    part_max: torch.Tensor,
    part_denom: torch.Tensor,
    part_acc: torch.Tensor,
) -> torch.Tensor:
    """Fold a row's partial softmaxes into the softmax they were cut from. Day 63.

    part_max:   [rows, n_q, splits] each chunk's running max, `-inf` where the chunk
                was empty.
    part_denom: [rows, n_q, splits] each chunk's unnormalised denominator.
    part_acc:   [rows, n_q, splits, head_dim] each chunk's unnormalised weighted-V
                sum, in the same scale as its own max.

    Returns [rows, n_q, 1, head_dim], the attention output, in fp32.

    The second pass, and it is the online softmax's renormalisation hoisted one level
    out. Each chunk's numbers are in its own exponent scale, so they cannot be added:
    take the joint max `M`, rescale every chunk by `exp(m_s - M)`, then sum. That is
    the same `alpha` the inner loop applies when a tile raises the running max, done
    once across programs instead of once across tiles, and it is the whole reason a
    split is exact rather than approximate.

    **The empty chunk is the case worth reading twice.** A split past a short row's
    end walks nothing and still launches and still stores, because a program cannot
    decline to write its slot of a workspace another program is about to read. It
    stores `-inf`, `0`, `0`, and `exp(-inf - M)` is exactly zero for any finite `M`,
    so it falls out of both sums with no branch anywhere. The reduction never learns
    which chunks were real.

    The only way that breaks is a row where *every* chunk was empty: then `M` is
    `-inf`, the rescale is `-inf - -inf`, and the answer is NaN rather than an error.
    Every row is required to have at least one key upstream, so it is unreachable,
    which is exactly why it is checked here (one comparison, on the host) and not in
    the kernel (where it would be a branch on every program).

    fp32 throughout, for the reason `PARTIAL_DTYPE` gives, and the max comes out
    first for the reason every flash kernel takes it out first: two chunks whose
    scores differ by 60 would overflow one and flush the other to zero.
    """
    if part_max.ndim != 3 or part_denom.ndim != 3 or part_acc.ndim != 4:
        raise ValueError(
            "the partial workspace is [rows, heads, splits] maxima and denominators "
            f"and [rows, heads, splits, head_dim] accumulators; got "
            f"{tuple(part_max.shape)}, {tuple(part_denom.shape)}, {tuple(part_acc.shape)}"
        )
    if part_max.shape != part_denom.shape:
        raise ValueError(
            "every split stores the same three numbers, so the maxima and the "
            f"denominators have the same shape; got {tuple(part_max.shape)} and "
            f"{tuple(part_denom.shape)}"
        )
    if part_acc.shape[:3] != part_max.shape:
        raise ValueError(
            "one accumulator per split: the accumulators' leading axes must match the "
            f"maxima; got {tuple(part_acc.shape)} against {tuple(part_max.shape)}"
        )
    joint = part_max.to(PARTIAL_DTYPE).amax(dim=2, keepdim=True)  # [rows, n_q, 1]
    if not bool(torch.isfinite(joint).all()):
        raise ValueError(
            "a row whose every partial max is -inf means no split saw a key, so the "
            "rescale is (-inf) - (-inf) and the output is NaN. Every row of a decode "
            "batch has at least one key; this workspace does not describe one"
        )
    alpha = torch.exp(part_max.to(PARTIAL_DTYPE) - joint)  # [rows, n_q, splits]
    denom = (alpha * part_denom.to(PARTIAL_DTYPE)).sum(dim=2)  # [rows, n_q]
    acc = (alpha[..., None] * part_acc.to(PARTIAL_DTYPE)).sum(dim=2)  # [rows, n_q, head_dim]
    return (acc / denom[..., None])[:, :, None, :]


def check_reduce_workspace(partials) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refuse a workspace pass two cannot fold, reading the launch off the workspace. Day 69.

    `check_partials` grades a workspace against a launch somebody described. Pass two
    on its own has no launch to grade against: its grid *is* the workspace's leading
    axes, so the accumulator's shape is taken as the claim and the other two buffers
    are held to it. Every clause of Day 64's gate still applies, contiguity first,
    because the reduce addresses the partials with the same flat
    `(row * n_q + head) * splits + split` that pass one stored them with.
    """
    tensors = tuple(partials)
    if len(tensors) != 3:
        raise SplitUnsound(
            f"a split workspace is three buffers, a max, a denominator and an "
            f"accumulator; got {len(tensors)}"
        )
    part_acc = tensors[2]
    if part_acc.ndim != 4:
        raise SplitUnsound(
            f"part_acc has shape {tuple(part_acc.shape)} and a workspace's accumulator "
            "is [rows, heads, splits, head_dim]: pass two reads its grid off that shape"
        )
    rows, n_q, splits, head_dim = part_acc.shape
    return check_partials(
        tensors, rows=rows, n_q=n_q, splits=splits, head_dim=head_dim, device=part_acc.device
    )


def split_reduce_kernel(partials, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Pass two alone, as tlsim programs: a workspace in, the attention out. Day 69.

    partials:  `(part_max, part_denom, part_acc)`, `[rows, n_q, splits]` twice and
               `[rows, n_q, splits, head_dim]`, fp32 and contiguous.
    out_dtype: what the output is stored as; the read passes the pool's dtype.
               `None` keeps the partials' fp32.

    Returns [rows, n_q, 1, head_dim].

    This is the reduce Day 63 wrote inside `paged_attention_split_kernel`, lifted out
    unchanged, and the read now calls it. The reason is a test and not a refactor.
    Reached only through the read, pass two is exercised on exactly the rows the test
    fed the read, and Day 68 showed what that means: a row shorter than one chunk
    leaves every other chunk `-inf`, `0`, `0`, a reduce over one live partial is the
    identity, and a reduce that keeps only the winning chunk passes. Handed a
    workspace directly, a test decides how many chunks saw keys without having to
    build a 514-token row to get there.
    """
    part_m, part_denom, part_acc = check_reduce_workspace(partials)
    rows, n_q, count, d = part_acc.shape
    dtype = PARTIAL_DTYPE if out_dtype is None else out_dtype
    m_flat = part_m.reshape(-1)
    denom_flat = part_denom.reshape(-1)
    acc_flat = part_acc.reshape(-1)
    chan = arange(0, d)
    out_flat = torch.zeros(rows * n_q * d, dtype=dtype)

    def reduce_kernel(prog, m_buf, denom_buf, acc_buf, dst) -> None:
        # This program owns one (row, head) and reduces its `count` partials.
        i = prog.program_id(0)
        h = prog.program_id(1)
        offs_s = arange(0, count)
        base = (i * n_q + h) * count
        m_s = load(m_buf, base + offs_s)  # [count]
        denom_s = load(denom_buf, base + offs_s)  # [count]
        acc_s = load(acc_buf, (base + offs_s)[:, None] * d + chan[None, :])  # [count, d]
        joint = m_s.max()
        alpha = torch.exp(m_s - joint)  # exactly 0 for an empty chunk
        out = (alpha[:, None] * acc_s).sum(dim=0) / (alpha * denom_s).sum()
        store(dst, (i * n_q + h) * d + chan, out.to(dtype))

    launch((rows, n_q), reduce_kernel, m_flat, denom_flat, acc_flat, out_flat)
    return out_flat.reshape(rows, n_q, 1, d)


def split_reduce_triton(partials, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Launch `_split_reduce_fwd` alone on a workspace already on the card. Day 69.

    Same contract as `split_reduce_kernel`, and held to `reduce_partials` by tests
    gated on a device. `paged_attention_split_triton` calls this for its second pass,
    so a test of this function is a test of the read's reduce and not of a copy.

    Raises `ValueError` on host tensors and `RuntimeError` when Triton is missing,
    like every jitted entry point here.
    """
    tensors = tuple(partials)
    if tensors and tensors[0].device.type != "cuda":
        raise ValueError(
            f"the Triton reduce reads device memory; the partials are on "
            f"{tensors[0].device}. Use `split_reduce` for a dispatch that falls back "
            "to the CPU model"
        )
    if not has_triton():
        raise RuntimeError("the triton package is not installed; `split_reduce` falls back")
    part_m, part_denom, part_acc = check_reduce_workspace(tensors)
    rows, n_q, count, d = part_acc.shape
    dtype = PARTIAL_DTYPE if out_dtype is None else out_dtype
    out = torch.empty((rows, n_q, d), dtype=dtype, device=part_acc.device)
    _split_reduce_fwd[(rows, n_q)](
        part_m,
        part_denom,
        part_acc,
        out,
        n_q * d,
        SPLITS=count,
        HEAD_DIM=d,
        BLOCK_D=next_power_of_2(d),
        BLOCK_S=next_power_of_2(count),
        num_warps=4,
    )
    return out[:, :, None, :]


def split_reduce(partials, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Pass two on whichever backend the workspace's device can run. Day 69.

    The same dispatch as `paged_attention_split`, keyed on where the partials live,
    because that is the only tensor pass two touches.
    """
    tensors = tuple(partials)
    if tensors and select_backend(tensors[0].device) == "triton":
        return split_reduce_triton(tensors, out_dtype=out_dtype)
    return split_reduce_kernel(tensors, out_dtype=out_dtype)


def paged_attention_split_kernel(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
    block: int = DEFAULT_BLOCK,
    splits: int | None = None,
    context_bounds: tuple[int, int] | None = None,
    validated: bool = False,
    partials=None,
) -> torch.Tensor:
    """The split decode read as two grids of tlsim programs. The model, Day 63.

    Same contract as `paged_attention_batched_reference`, and held to it: the split
    count is a performance knob and every value returns the same attention.

    q, k_pool, v_pool, slot_mapping, context_lens, n_rep, scale, context_bounds,
    validated: exactly as in the unsplit read, including the refusals, because a read
    that accepts inputs its oracle rejects cannot be compared to it.
    block:  keys folded per program step.
    splits: chunks per row. `None` means the handed-in workspace's split axis if
            there is one, and otherwise `choose_splits` on the mapping's *width* and
            the grid, which is the capture-safe default: no argument may be a reason
            to read the lengths tensor.
    partials: Day 64. `(part_max, part_denom, part_acc)` owned by somebody else,
            already narrowed to this launch's rows, written in place. `None` keeps
            the old behaviour and allocates three tensors per call.

    Returns [batch, n_q, 1, d].

    Two launches, which is the shape of the thing and not an implementation detail.

    *Pass one*, grid `(row, head, split)`. Each program reads its row's length, works
    out which slice of the history its chunk owns, walks that slice `block` keys at a
    time exactly as Day 59's loop does, and stores its three partials. A chunk that
    starts past the row's end walks zero tiles and stores `-inf`, `0`, `0`: it cannot
    decline to write, because pass two is going to read its slot either way.

    *Pass two*, grid `(row, head)`. Each program loads its row's `splits` partials,
    takes the joint max, rescales and sums. That is `reduce_partials` written as a
    program, and the plain-torch version is the oracle it is checked against.

    Still a model: this is tlsim on the CPU and slower than the read it replaces by a
    wide margin. What it pins is the addressing, the chunk arithmetic and the empty
    partial, so the jitted version has a fixed target.
    """
    geom = check_batched_inputs(
        q, k_pool, v_pool, slot_mapping, context_lens, n_rep, context_bounds, validated
    )
    if block < 1:
        raise ValueError(f"a tile holds at least one key; got {block}")
    if splits is None and partials is not None:
        splits = partial_splits(partials)
    if splits is None:
        splits = choose_splits(geom.batch, geom.n_q, geom.max_ctx, block)
    count, chunk = partition_width(geom.max_ctx, block, splits=splits)
    if scale is None:
        scale = geom.head_dim**-0.5

    batch, n_q, d = geom.batch, geom.n_q, geom.head_dim
    channels = geom.channels

    slots = slot_mapping.to(torch.long).reshape(batch * geom.max_ctx)
    lens = context_lens.to(torch.long)
    k_flat = k_pool.reshape(geom.num_slots * channels)
    v_flat = v_pool.reshape(geom.num_slots * channels)
    q_rows = q[:, :, 0, :].to(PARTIAL_DTYPE)  # [batch, n_q, d]
    head_kv = arange(0, n_q) // n_rep  # query head -> its KV head (GQA)
    chan = arange(0, d)  # the channel ramp inside one head of one token

    # The workspace, flat, exactly as the jitted version addresses it: one max, one
    # denominator and one `d`-wide accumulator per (row, head, split). A handed-in
    # one is checked and then flattened, which is free and stays a view because the
    # gate has already refused anything that is not contiguous. Day 63 filled the
    # owned buffers with `-inf` and zeros; that was never load-bearing, because every
    # program stores its slot whether or not it walked a tile, and taking the fill
    # away is what makes a buffer reusable across steps.
    if partials is None:
        part_m = torch.empty(batch * n_q * count, dtype=PARTIAL_DTYPE)
        part_denom = torch.empty(batch * n_q * count, dtype=PARTIAL_DTYPE)
        part_acc = torch.empty(batch * n_q * count * d, dtype=PARTIAL_DTYPE)
    else:
        held = check_partials(
            partials, rows=batch, n_q=n_q, splits=count, head_dim=d, device=q.device
        )
        part_m, part_denom, part_acc = (t.reshape(-1) for t in held)

    def split_kernel(prog, s_buf, len_buf, k_buf, v_buf, m_buf, denom_buf, acc_buf) -> None:
        # This program owns one chunk of one row's one query head.
        i = prog.program_id(0)
        h = prog.program_id(1)
        s = prog.program_id(2)
        ctx = int(load(len_buf, arange(i, i + 1))[0])  # the row's own dynamic bound
        lo = s * chunk  # where this chunk starts in the row's history
        hi = min(lo + chunk, ctx)  # ...and where it stops, which may be before it starts
        slot = (i * n_q + h) * count + s  # this program's slot of the workspace

        qi = q_rows[i, h]  # [d]
        head_cols = int(head_kv[h]) * d + chan  # this query head's channels of a token
        base = i * geom.max_ctx  # where this row's mapping starts in the flat buffer
        m = torch.tensor(float("-inf"))
        denom = torch.zeros(())
        acc = torch.zeros(d)
        for b in range(max(0, cdiv(hi - lo, block))):
            pos = lo + b * block + arange(0, block)  # positions in *this row's* history
            valid = pos < hi  # guards the ragged last tile of the chunk
            row_slots = load(s_buf, base + pos, mask=valid, other=0)
            ptr = row_slots[:, None] * channels + head_cols[None, :]
            k_tile = load(k_buf, ptr, mask=valid[:, None], other=0.0).to(PARTIAL_DTYPE)
            v_tile = load(v_buf, ptr, mask=valid[:, None], other=0.0).to(PARTIAL_DTYPE)
            s_row = scale * (k_tile * qi[None, :]).sum(dim=1)
            s_row = torch.where(valid, s_row, torch.full_like(s_row, float("-inf")))
            m_new = torch.maximum(m, s_row.max())
            alpha = torch.exp(m - m_new)
            p = torch.exp(s_row - m_new)
            denom = denom * alpha + p.sum()
            acc = acc * alpha + (p[:, None] * v_tile).sum(dim=0)
            m = m_new
        # An empty chunk stores -inf, 0, 0, and it stores them: pass two reads this
        # slot whether or not a key was ever in it.
        store(m_buf, arange(slot, slot + 1), m)
        store(denom_buf, arange(slot, slot + 1), denom)
        store(acc_buf, slot * d + chan, acc)

    launch(
        (batch, n_q, count),
        split_kernel,
        slots,
        lens,
        k_flat,
        v_flat,
        part_m,
        part_denom,
        part_acc,
    )

    # Pass two is its own entry point since Day 69, and this is its only caller in
    # the read. The flat buffers go back to their shapes as views: no copy, and the
    # gate it runs is the same one a handed-in workspace already passed.
    return split_reduce_kernel(
        (
            part_m.reshape(batch, n_q, count),
            part_denom.reshape(batch, n_q, count),
            part_acc.reshape(batch, n_q, count, d),
        ),
        out_dtype=q.dtype,
    )


if triton is not None:  # pragma: no cover - compiled and run only on a GPU box

    @triton.jit
    def _paged_attention_split_fwd(
        q_ptr,  # [batch, n_q, head_dim] contiguous
        k_ptr,  # [num_slots * channels] the layer's flat physical K pool, shared
        v_ptr,  # [num_slots * channels] the layer's flat physical V pool, shared
        slot_ptr,  # [batch * max_ctx] logical position -> physical slot, per row
        ctx_ptr,  # [batch] how many of a row's slots are real
        m_ptr,  # [batch, n_q, splits] partial maxima
        denom_ptr,  # [batch, n_q, splits] partial denominators
        acc_ptr,  # [batch, n_q, splits, head_dim] partial weighted-V sums
        scale,
        stride_map,  # max_ctx: elements from one row's mapping to the next
        stride_slot,  # channels: elements from one pool slot to the next
        stride_q_row,  # n_q * head_dim
        SPLITS: tl.constexpr,  # chunks per row, the grid's third axis
        KEYS_PER_SPLIT: tl.constexpr,  # the chunk, a whole number of tiles
        HEAD_DIM: tl.constexpr,
        N_REP: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Pass one: row `i`, query head `h`, chunk `s`, folded into a partial softmax.

        Day 62's body with two lines changed. The walk starts at `s * KEYS_PER_SPLIT`
        instead of at 0 and stops at the smaller of the chunk's end and the row's
        length, and the result goes to a workspace instead of to the output, without
        the final divide: the denominator belongs to this chunk alone and only pass
        two knows the joint one.

        `ctx` is still loaded, not computed, for the reason it was yesterday. The
        chunk bounds are `constexpr` arithmetic on a launch constant, which is what
        makes the split count safe to capture: it is baked into the compiled kernel
        and identical on every step, while the thing that varies per step stays a
        load.

        An empty chunk (`lo >= ctx`) runs the loop zero times and stores `-inf`, `0`,
        `0`. It stores them rather than returning early because pass two reads this
        slot unconditionally, and a workspace slot nobody wrote holds whatever the
        allocator left there.
        """
        i = tl.program_id(0)
        h = tl.program_id(1)
        s = tl.program_id(2)
        kv_head = h // N_REP
        ctx = tl.load(ctx_ptr + i)

        lo = s * KEYS_PER_SPLIT
        hi = tl.minimum(lo + KEYS_PER_SPLIT, ctx)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM
        q_off = i * stride_q_row + h * HEAD_DIM + offs_d
        q = tl.load(q_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

        m = tl.full([1], float("-inf"), dtype=tl.float32)
        denom = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        base = i * stride_map
        for b in range(0, tl.cdiv(tl.maximum(hi - lo, 0), BLOCK_N)):
            offs_n = lo + b * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = offs_n < hi
            slots = tl.load(slot_ptr + base + offs_n, mask=valid, other=0)
            kv_off = slots[:, None] * stride_slot + kv_head * HEAD_DIM + offs_d[None, :]
            tile_mask = valid[:, None] & mask_d[None, :]
            k = tl.load(k_ptr + kv_off, mask=tile_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + kv_off, mask=tile_mask, other=0.0).to(tl.float32)

            sc = tl.sum(q[None, :] * k, axis=1) * scale
            sc = tl.where(valid, sc, float("-inf"))

            m_new = tl.maximum(m, tl.max(sc, axis=0))
            alpha = tl.exp(m - m_new)
            p = tl.exp(sc - m_new)
            denom = denom * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m = m_new

        slot = (i * tl.num_programs(1) + h) * SPLITS + s
        tl.store(m_ptr + slot, m)
        tl.store(denom_ptr + slot, denom)
        tl.store(acc_ptr + slot * HEAD_DIM + offs_d, acc, mask=mask_d)

    @triton.jit
    def _split_reduce_fwd(
        m_ptr,  # [batch, n_q, splits]
        denom_ptr,  # [batch, n_q, splits]
        acc_ptr,  # [batch, n_q, splits, head_dim]
        out_ptr,  # [batch, n_q, head_dim]
        stride_q_row,  # n_q * head_dim
        SPLITS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_S: tl.constexpr,  # SPLITS padded up to a power of two
    ):
        """Pass two: one program per (row, head), folding that row's partials.

        `reduce_partials` as a program. The splits ramp is padded to a power of two
        the same way the channel ramp is, and the surplus lanes load `-inf` and `0`,
        which is exactly what an empty chunk stores: the padding and the empty chunks
        take the same path, and neither needs a branch.

        The joint max comes out first for the reason it always does. Two chunks whose
        scores differ by sixty would overflow one exponent and flush the other to
        zero; rescaled to `M` both survive, and the sum is the softmax the row was
        cut from.
        """
        i = tl.program_id(0)
        h = tl.program_id(1)
        n_q = tl.num_programs(1)

        offs_s = tl.arange(0, BLOCK_S)
        mask_s = offs_s < SPLITS
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        base = (i * n_q + h) * SPLITS
        m_s = tl.load(m_ptr + base + offs_s, mask=mask_s, other=float("-inf"))
        denom_s = tl.load(denom_ptr + base + offs_s, mask=mask_s, other=0.0)
        acc_off = (base + offs_s)[:, None] * HEAD_DIM + offs_d[None, :]
        acc_s = tl.load(acc_ptr + acc_off, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

        joint = tl.max(m_s, axis=0)
        alpha = tl.exp(m_s - joint)  # exactly 0 for an empty chunk and for the padding
        denom = tl.sum(alpha * denom_s, axis=0)
        acc = tl.sum(alpha[:, None] * acc_s, axis=0)

        out = acc / denom
        out_off = i * stride_q_row + h * HEAD_DIM + offs_d
        tl.store(out_ptr + out_off, out.to(out_ptr.dtype.element_ty), mask=mask_d)


def paged_attention_split_triton(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
    block: int = DEFAULT_BLOCK,
    splits: int | None = None,
    context_bounds: tuple[int, int] | None = None,
    validated: bool = False,
    partials=None,
) -> torch.Tensor:
    """Launch the two split passes. Same contract as the oracle, every tensor on CUDA.

    Returns [batch, n_q, 1, d], matching `paged_attention_batched_reference` to about
    1e-3. The split reassociates the exponent sums once more than the unsplit kernel
    already does, so it is close and not bit-identical to that either: the accuracy
    trade every flash-decoding kernel makes, with the partials in fp32 whatever the
    pool holds.

    **`partials` is Day 64, and it is the caveat Day 63 wrote, corrected.** That day
    said a captured region cannot allocate. It can: a `torch.empty` under capture is
    served from the graph's pool and its address is baked into the replay like every
    other intermediate, and the launch is perfectly legal. What is wrong with it is
    quieter. That allocation is charged to the shared arena, once per graph, at the
    largest shape in the list, and *nothing prices it*: `CapturePlan.pool_bytes` was
    the score term alone, so a process running this read reserved an arena it had
    under-reported by the partials and found out from the allocator. Handing the
    buffers in moves the number into `split_workspace_bytes`, where a plan can be
    wrong about it out loud. Outside a graph it is the plainer thing it looks like:
    three allocations per read per layer per step.

    Raises `ValueError` on host tensors and `RuntimeError` when Triton is missing,
    rather than letting a launch crash somewhere unreadable.
    """
    geom = check_batched_inputs(
        q, k_pool, v_pool, slot_mapping, context_lens, n_rep, context_bounds, validated
    )
    if block < 1:
        raise ValueError(f"a tile holds at least one key; got {block}")
    if q.device.type != "cuda":
        raise ValueError(
            f"the Triton kernel reads device memory; q is on {q.device}. "
            "Use `paged_attention_split` for a dispatch that falls back to the CPU model"
        )
    if not has_triton():
        raise RuntimeError(
            "the triton package is not installed; `paged_attention_split` falls back"
        )
    if k_pool.device != q.device or slot_mapping.device != q.device:
        raise ValueError("q, the pools, and slot_mapping must live on the same device")
    if context_lens.device != q.device:
        raise ValueError("context_lens must live on the same device as q: the kernel loads it")

    if splits is None and partials is not None:
        splits = partial_splits(partials)
    if splits is None:
        splits = choose_splits(geom.batch, geom.n_q, geom.max_ctx, block)
    count, chunk = partition_width(geom.max_ctx, block, splits=splits)
    if scale is None:
        scale = geom.head_dim**-0.5

    q_rows = q[:, :, 0, :].contiguous()  # [batch, n_q, head_dim]
    k_flat = k_pool.contiguous().reshape(-1)
    v_flat = v_pool.contiguous().reshape(-1)
    slots = slot_mapping.to(slot_index_dtype(geom)).contiguous().reshape(-1)
    lens = context_lens.to(torch.int32).contiguous()

    shape = (geom.batch, geom.n_q, count)
    if partials is None:
        part_m = torch.empty(shape, dtype=PARTIAL_DTYPE, device=q.device)
        part_denom = torch.empty(shape, dtype=PARTIAL_DTYPE, device=q.device)
        part_acc = torch.empty((*shape, geom.head_dim), dtype=PARTIAL_DTYPE, device=q.device)
    else:
        part_m, part_denom, part_acc = check_partials(
            partials,
            rows=geom.batch,
            n_q=geom.n_q,
            splits=count,
            head_dim=geom.head_dim,
            device=q.device,
        )

    _paged_attention_split_fwd[(geom.batch, geom.n_q, count)](
        q_rows,
        k_flat,
        v_flat,
        slots,
        lens,
        part_m,
        part_denom,
        part_acc,
        scale,
        geom.max_ctx,
        geom.channels,
        geom.stride_q_row,
        SPLITS=count,
        KEYS_PER_SPLIT=chunk,
        HEAD_DIM=geom.head_dim,
        N_REP=n_rep,
        BLOCK_D=next_power_of_2(geom.head_dim),
        BLOCK_N=next_power_of_2(block),
        num_warps=4,
    )
    # Day 69: pass two through its own entry point, so the function the reduce tests
    # launch is the function the read launches.
    return split_reduce_triton((part_m, part_denom, part_acc), out_dtype=q.dtype)


def paged_attention_split(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
    block: int = DEFAULT_BLOCK,
    splits: int | None = None,
    context_bounds: tuple[int, int] | None = None,
    validated: bool = False,
    partials=None,
) -> torch.Tensor:
    """The split decode read, on whichever backend this device can actually run.

    Day 62's dispatch, one read further along: a CUDA tensor with Triton installed
    gets the two jitted passes, anything else gets the two tlsim passes. Both compute
    the same attention to a few ulps, so the backend is a speed decision and never a
    numerics one, which is the only claim that makes a fallback honest.

    `partials` goes to whichever backend runs, because the workspace is a property
    of the plan and not of the box: a `SplitWorkspace` allocated on CPU feeds the
    tlsim passes and one allocated on a card feeds the jitted ones, and the gate that
    refuses a mismatch is the same gate.

    Not yet wired to `PagedRead`. The split is a third read alongside the rectangle
    and the stream, and it earns its place on a number this box cannot produce: the
    tail it shortens is measured in `SplitPlan.wave_speedup` and paid for in
    `SplitPlan.partial_bytes`, and which of those wins is a property of a card.
    """
    if select_backend(q.device) == "triton":
        return paged_attention_split_triton(
            q,
            k_pool,
            v_pool,
            slot_mapping,
            context_lens,
            n_rep,
            scale,
            block=block,
            splits=splits,
            context_bounds=context_bounds,
            validated=validated,
            partials=partials,
        )
    return paged_attention_split_kernel(
        q,
        k_pool,
        v_pool,
        slot_mapping,
        context_lens,
        n_rep,
        scale,
        block=block,
        splits=splits,
        context_bounds=context_bounds,
        validated=validated,
        partials=partials,
    )
