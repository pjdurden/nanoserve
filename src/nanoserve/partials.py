"""The split read's workspace, allocated once and handed to the kernel. Week 14, Day 64.

Day 63 wrote flash-decoding and stopped one step short of it being usable. Both
passes need a place to put `[rows, heads, splits]` of a running max, a denominator
and a `head_dim`-wide accumulator, and the kernel made that place with three
`torch.empty` calls inside the read. Per layer. Per step.

**The caveat that day wrote is not quite the true one, and the difference is the
reason this file exists.** Day 63 said a captured region cannot allocate. It can:
an allocation made while a CUDA graph is recording is served from the graph's memory
pool, the address is baked into the replay exactly like every other intermediate's,
and the launch is legal and fast. What it is not is *priced*. `CapturePlan.pool_bytes`
was `shared_pool_bytes` over the score term alone, so a process running this read
reserves an arena it has under-reported by the partials, once per graph, at the
largest shape in the list, and finds out the real number from the allocator rather
than from the plan. Outside a graph it is the plainer thing it looks like: three
allocations per read, which on a 16-layer model at 8 splits is 48 allocator calls a
step to hold four megabytes that could have been held for the life of the process.

So the partials become what the mapping already is. Day 51 made the read rectangle a
window on a persistent slot table; this is the same move on the workspace, and the
same three properties fall out of it: one allocation, a fixed address, and a window
per step that is storage rather than a copy.

Three things are the day.

**One split count for a whole capture list.** `choose_splits` answers per launch, and
a capture list is not one launch: it is a graph per shape, all of them replaying
against one arena. `plan_splits` collapses the per-bucket answers with a max, because
that is the direction that keeps the split. A bucket handed more chunks than it asked
for gets empty ones, and Day 63 made an empty chunk free: it walks nothing, stores
`-inf, 0, 0`, and drops out of the reduction with no branch. Rounding the other way
would take the splits off the one-row batch, which is the only batch a split was ever
for.

**The two maxima sit at opposite ends of the list.** The split *axis* is sized by the
narrowest row bucket, because a small grid is what needs filling. The *arena* is
`rows * heads * splits * (head_dim + 2)` and is sized by the widest, because
`choose_splits` is a ceiling division of a program target, so `rows * splits(rows)`
climbs to that target and then flattens instead of falling. The rectangle of both
maxima is therefore strictly larger than any shape in the list needs, and that
surplus is what one shared arena costs. It is the same shape of finding as Day 54's
pool sharing: the number you save is not the length of the list.

**Only the row axis narrows.** Both passes address the workspace flat, as
`(row * n_q + head) * splits + split`, because that is the form `tl.store` takes. A
window on the outermost axis of a contiguous buffer is still contiguous and the flat
arithmetic still holds; a window on the split axis is a tensor of exactly the right
shape whose stride is the *allocated* count, and every program past the first would
then write another program's slot without faulting. `SplitWorkspace.rows` is the
legal window and `check_partials` is the refusal for everything that resembles it.

And nothing here is initialised, which is Day 63's empty-chunk design paying out a
day later. Every program stores its slot whether or not it walked a tile, so no slot
the reduction reads is a slot pass one skipped, so a buffer reused across a thousand
steps carries nothing from the step before it. The `-inf` fill the tlsim path had was
decorative, and taking the allocation away is what made that testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kernels.flash_decoding import (
    DEFAULT_PARTITION,
    DEFAULT_TARGET_PROGRAMS,
    PARTIAL_DTYPE,
    SplitUnsound,
    paged_attention_split,
    partition_width,
    plan_splits,
)
from .kernels.triton_batched_attention import DEFAULT_BLOCK

__all__ = [
    "SplitWorkspace",
    "allocate_for",
    "allocate_partials",
    "check_workspace_covers",
]


@dataclass(frozen=True, eq=False)
class SplitWorkspace:
    """The three buffers a split read writes, owned by the plan and not by the read.

    part_max:      [max_rows, n_q, splits] each chunk's running max.
    part_denom:    [max_rows, n_q, splits] each chunk's unnormalised denominator.
    part_acc:      [max_rows, n_q, splits, head_dim] each chunk's weighted-V sum.
    keys_per_split: the chunk, always a whole number of tiles.
    context_width: the mapping width the partition was computed from. Not decoration:
                   the chunk bounds are `s * keys_per_split`, so a step whose mapping
                   is a different width is a different partition, and the arena would
                   be sized for a launch that is not the one happening.
    block:         the score tile the inner loop folds, carried for the same reason.

    `eq=False` because two workspaces are the same workspace when they are the same
    storage, and a dataclass `__eq__` over tensors is a truth value of a tensor.
    Frozen for `SplitPlan`'s reason: it describes an allocation that has already
    happened, and one that can be edited afterwards describes nothing.
    """

    part_max: torch.Tensor
    part_denom: torch.Tensor
    part_acc: torch.Tensor
    keys_per_split: int
    context_width: int
    block: int

    @property
    def max_rows(self) -> int:
        """The row ceiling the arena was sized for: the widest batch it can hold."""
        return int(self.part_max.shape[0])

    @property
    def n_q(self) -> int:
        """Query heads. Not a ceiling: a model has the number it has."""
        return int(self.part_max.shape[1])

    @property
    def splits(self) -> int:
        """Chunks per row, the grid's third axis. A launch constant, not a ceiling.

        A narrower launch cannot take fewer of these the way it takes fewer rows,
        because the split axis is not the outermost one: see the module docstring.
        """
        return int(self.part_max.shape[2])

    @property
    def head_dim(self) -> int:
        return int(self.part_acc.shape[3])

    @property
    def device(self) -> torch.device:
        return self.part_max.device

    @property
    def dtype(self) -> torch.dtype:
        """fp32, whatever the pool holds. See `PARTIAL_DTYPE`."""
        return self.part_max.dtype

    @property
    def programs(self) -> int:
        """Programs pass one starts at the arena's widest batch, one partial each."""
        return self.max_rows * self.n_q * self.splits

    @property
    def cells(self) -> int:
        """Numbers held: a max, a denominator and a `head_dim` row per program."""
        return self.programs * (self.head_dim + 2)

    @property
    def bytes(self) -> int:
        """What the arena weighs. `SplitPlan.partial_bytes` of the widest shape."""
        return self.cells * torch.finfo(self.dtype).bits // 8

    @property
    def mib(self) -> float:
        return self.bytes / 2**20

    @property
    def addresses(self) -> tuple[int, int, int]:
        """The three pointers a captured replay would be bound to.

        The witness, and it is the same one `check_addresses_stable` reads off the
        decode inputs: a workspace is doing its job exactly when this tuple is the
        same before and after a step.
        """
        return (
            self.part_max.data_ptr(),
            self.part_denom.data_ptr(),
            self.part_acc.data_ptr(),
        )

    def rows(self, rows: int):
        """The window this step's launch addresses: `[:rows]` of all three buffers.

        The one narrowing the flat addressing survives. A slice of the outermost axis
        of a contiguous tensor is contiguous and begins at the buffer's own address,
        so `(row * n_q + head) * splits + split` means the same thing over the window
        as it does over the arena. Every other axis would keep the arena's stride and
        quietly move each program onto its neighbour's slot.
        """
        if rows < 1:
            raise SplitUnsound(f"a decode step has at least one row; got {rows}")
        if rows > self.max_rows:
            raise SplitUnsound(
                f"this arena holds {self.max_rows} rows and the step has {rows}: a "
                "window cannot be wider than the buffer it is a window on, and the "
                "row ceiling came from the scheduler's slot count at plan time"
            )
        return self.part_max[:rows], self.part_denom[:rows], self.part_acc[:rows]

    def is_window(self, tensor: torch.Tensor) -> bool:
        """Whether `tensor` is one of these buffers' storage rather than a copy.

        `SlotTable.is_window` for the workspace, and the question it answers is the
        capture one: a copy computes the same attention and replays against storage
        that is not there any more.
        """
        mine = {t.untyped_storage().data_ptr() for t in (self.part_max, self.part_denom, self.part_acc)}
        return tensor.untyped_storage().data_ptr() in mine

    def read(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        n_rep: int,
        scale: float | None = None,
        context_bounds: tuple[int, int] | None = None,
        validated: bool = False,
    ) -> torch.Tensor:
        """One split decode read over this arena. Same contract as the oracle.

        The reason this is a method and not a free function: the split count, the
        tile and the chunk are all properties of the allocation, so a caller that
        could pass its own would be able to launch a grid the arena was not sized
        for. Here there is one place those numbers come from, and it is the plan.
        """
        check_workspace_covers(
            self,
            rows=int(q.shape[0]),
            n_q=int(q.shape[1]),
            head_dim=int(q.shape[-1]),
            context_width=int(slot_mapping.shape[-1]),
            device=q.device,
        )
        return paged_attention_split(
            q,
            k_pool,
            v_pool,
            slot_mapping,
            context_lens,
            n_rep,
            scale,
            block=self.block,
            splits=self.splits,
            context_bounds=context_bounds,
            validated=validated,
            partials=self.rows(int(q.shape[0])),
        )

    def as_dict(self) -> dict:
        """What a health payload would say about the arena, next to the read's mode."""
        return {
            "max_rows": self.max_rows,
            "heads": self.n_q,
            "splits": self.splits,
            "keys_per_split": self.keys_per_split,
            "context_width": self.context_width,
            "block": self.block,
            "partial_bytes": self.bytes,
        }

    def render(self) -> str:
        """One line, for a boot log: the trade rather than the size."""
        return (
            f"split workspace: {self.max_rows} rows x {self.n_q} heads x "
            f"{self.splits} splits of {self.keys_per_split} keys over a "
            f"{self.context_width}-wide mapping, {self.mib:.2f} MiB, allocated once"
        )


def allocate_partials(
    *,
    max_rows: int,
    n_q: int,
    head_dim: int,
    context_width: int,
    block: int = DEFAULT_BLOCK,
    splits: int | None = None,
    row_buckets=None,
    target_programs: int = DEFAULT_TARGET_PROGRAMS,
    partition: int = DEFAULT_PARTITION,
    device=None,
    dtype: torch.dtype = PARTIAL_DTYPE,
) -> SplitWorkspace:
    """Reserve the split read's workspace for the life of the process.

    max_rows:      the row ceiling, which is the scheduler's slot count unless a flag
                   is lower. The same number `CapturePlan.max_rows` holds.
    n_q:           query heads.
    head_dim:      channels in one head, which is what makes an accumulator a row.
    context_width: the mapping's width, and under Day 61's streamed bucket set that
                   is `max_model_len` for the life of the process.
    block:         the score tile.
    splits:        the count, when the caller has already decided it.
    row_buckets:   the list's row axis, when they have not: the count is
                   `plan_splits` over these, which is the max and not the widest.
    device, dtype: where it lives and what it holds. The dtype is an argument so a
                   test can name it and not so a deployment can lower it.

    Keyword-only on purpose. Six integers in a row, four of which are plausible
    values for each other, is a call whose arguments cannot be read at the call site,
    and this one is made once at boot where getting it wrong is an arena that is
    silently the wrong size.

    `torch.empty` and not `zeros`, and that is a claim rather than a saving. Every
    program in pass one stores its slot whether it walked a tile or not, so there is
    no slot pass two reads that pass one did not write, on this step or on any later
    one. A fill would say the opposite: that some slot's prior contents matter.
    """
    if max_rows < 1:
        raise ValueError(f"an arena holds at least one row; got {max_rows}")
    if n_q < 1:
        raise ValueError(f"a launch needs at least one query head; got {n_q}")
    if head_dim < 1:
        raise ValueError(f"a head has at least one channel; got {head_dim}")
    if context_width < 1:
        raise ValueError(f"a mapping is at least one key wide; got {context_width}")
    if block < 1:
        raise ValueError(f"a tile holds at least one key; got {block}")
    if splits is None:
        buckets = (max_rows,) if row_buckets is None else row_buckets
        splits = plan_splits(buckets, n_q, context_width, block, target_programs, partition)
    count, chunk = partition_width(context_width, block, splits=splits)
    shape = (max_rows, n_q, count)
    return SplitWorkspace(
        part_max=torch.empty(shape, dtype=dtype, device=device),
        part_denom=torch.empty(shape, dtype=dtype, device=device),
        part_acc=torch.empty((*shape, head_dim), dtype=dtype, device=device),
        keys_per_split=chunk,
        context_width=context_width,
        block=block,
    )


def allocate_for(capture, head_dim: int, device=None, **kwargs) -> SplitWorkspace:
    """Build the arena a `CapturePlan` already decided the size of.

    Duck-typed on five fields rather than imported, because `launch` prices this
    workspace and `launch` is the top of the boot path: a plan is a record of
    decisions and this is the one call that spends them, so the dependency runs this
    way and not the other.

    `head_dim` is the one number a capture plan does not carry, because a score
    rectangle has no channel axis: it is `rows * heads * query * width`, and the
    channels were summed away before the workspace existed. The split's accumulator
    is the first intermediate in this engine that is `head_dim` wide, which is a
    small thing that says what the two reads actually hold.
    """
    splits = int(getattr(capture, "splits", 0) or 0)
    if splits < 1:
        raise ValueError(
            "this capture plan did not plan a split: `splits` is 0, which is the "
            "rectangle or the streamed read, and an arena for a launch nothing in "
            "the process makes is reserved device memory nobody addresses"
        )
    return allocate_partials(
        max_rows=capture.max_rows,
        n_q=capture.num_heads,
        head_dim=head_dim,
        context_width=capture.max_width,
        block=capture.block,
        splits=splits,
        device=device,
        **kwargs,
    )


def check_workspace_covers(
    workspace: SplitWorkspace,
    *,
    rows: int,
    n_q: int,
    head_dim: int,
    context_width: int,
    device=None,
) -> None:
    """Refuse a step the arena was not planned for, before it becomes a pointer.

    `check_partials` is the gate on the *buffers*, and it runs inside the kernel over
    whatever it was handed. This is the gate on the *plan*, and it runs here because
    the interesting mismatches are plan-shaped: a mapping of a width the partition
    was not computed from, a model with more heads than the arena has, a batch wider
    than the row ceiling. None of them is a shape error in the kernel's sense, and
    all of them are an arena describing a launch that is not the one about to happen.

    The width clause is the one that would otherwise be silent. The chunk bounds are
    `s * keys_per_split` in both passes, so a narrower mapping does not overrun
    anything: it makes chunks past the mapping's end, each of which walks nothing and
    stores an empty partial, and the read returns the attention over a *prefix* of
    each row. Finite, plausible, and not the answer.
    """
    if rows > workspace.max_rows:
        raise SplitUnsound(
            f"this step has {rows} rows and the arena holds {workspace.max_rows}: the "
            "row ceiling came from the scheduler's slot count at plan time, so a "
            "wider batch means the plan and the scheduler disagree"
        )
    if n_q != workspace.n_q:
        raise SplitUnsound(
            f"this step has {n_q} query heads and the arena has {workspace.n_q}: the "
            "head count is not a ceiling, it is the model's, so this arena belongs "
            "to some other model"
        )
    if head_dim != workspace.head_dim:
        raise SplitUnsound(
            f"this step's heads are {head_dim} channels wide and the arena's "
            f"accumulator is {workspace.head_dim}"
        )
    if context_width != workspace.context_width:
        raise SplitUnsound(
            f"this step's mapping is {context_width} wide and the partition was "
            f"computed from {workspace.context_width}: the chunk bounds are "
            "`split * keys_per_split` on both passes, so a different width is a "
            "different partition, and the read would return the attention over a "
            "prefix of every row rather than raise"
        )
    if device is not None and torch.device(device) != workspace.device:
        raise SplitUnsound(
            f"this step runs on {torch.device(device)} and the arena is on "
            f"{workspace.device}"
        )
