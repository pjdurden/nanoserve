"""The batched decode read as a real Triton kernel. Week 14, Day 62.

Day 59 wrote `paged_attention_batched_kernel`: one program per `(row, query head)`,
each walking its own row's `cdiv(ctx, block)` tiles through the block table and
folding them into a flash-attention online softmax, with no gather and no
`[rows, heads, 1, ctx]` score rectangle anywhere. Day 60 put it behind a flag and
counted what it held. Day 61 spent the memory that frees: the width axis left the
capture list and 576 graphs became 9. Every one of those three days ended on the
same sentence, that the loop is `tlsim` in Python and nothing arrived faster. This
module is that loop on hardware.

It is Day 23's `triton_paged_attention` one axis wider, and the layering is the
same on purpose: every integer the kernel turns into a pointer is computed by
ordinary Python in `check_batched_inputs` / `batched_launch_grid` /
`slot_index_dtype`, which are tested on any box, and the jitted body is held to
`paged_attention_batched_reference` by tests gated on a real device. Addressing bugs
live in the arithmetic, and arithmetic you can single step is arithmetic you can fix.

Three things are genuinely new below, and all three are about the batch axis.

**The loop bound is data, not a program id.** Day 23's kernel knows its extent
before it loads anything: query `i` sees `past + i + 1` keys, and `past` is a host
integer baked into the launch. Here the extent is `context_lens[i]`, which the
program has to `tl.load` from device memory before it can decide how many times to
go round. That is the ragged walk, and it is also the reason this survives CUDA-graph
capture unchanged: the number that varies per step varies *inside* the graph, in a
buffer the launcher overwrites, and never in a Python `range` the tracer would bake.
A host-side `int(context_lens.max())` to size the loop would be Day 48's
synchronisation and Day 49's graph break in one line.

**`max_ctx` appears exactly once, as a stride.** The mapping's row stride is its
width, which under Day 61's streamed bucket set is `max_model_len` on every step of
every request. It is an address multiplier here and nothing else: not a loop bound,
not an allocation, not a shape. That is the whole reason a streamed capture list can
collapse to the row axis, stated as a line of pointer arithmetic instead of as a
table of graph counts.

**The address arithmetic has a width, and 32 bits is not always enough.** The kernel
forms `slot * channels + kv_head * head_dim + d`, and if the slot index is int32 that
multiply is an int32 multiply, which wraps silently. `slot_index_dtype` picks the
narrow type only when the whole flat pool addresses inside `INT32_MAX`, because the
failure on the other side of that line is not a fault: a wrapped offset is a legal
offset somewhere else, and the softmax over it is finite and plausible and wrong.

What is *not* here is the split that production kernels do. vLLM and SGLang partition
a long sequence across several programs and reduce their partial softmaxes
afterwards (flash-decoding), because a grid of `(row, head)` gives one program per
row and a batch with one long row waits for it. `launch_work` below is the arithmetic
that says how much that costs, and it is the honest half of today's claim: tiles
walked is a sum, wall clock is a max, and the two only coincide when the grid is
large enough to queue.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paged_attention import cdiv, paged_attention_batched_kernel
from .triton_paged_attention import has_triton, next_power_of_2, select_backend

try:  # Triton rides along with the Linux GPU torch wheel; a CPU wheel has no module.
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - exercised by whichever box you are on
    triton = None
    tl = None

#: The largest value a signed 32-bit offset holds. Past it the kernel's pointer
#: arithmetic has to run in 64 bits; see `slot_index_dtype`.
INT32_MAX = 2**31 - 1

#: Keys folded per program step when nobody says otherwise, matching
#: `paged_attention_batched_kernel`. A pure performance knob: every value returns the
#: same attention. It is *not* `block_size`, and the two are unrelated.
DEFAULT_BLOCK = 32

__all__ = [
    "DEFAULT_BLOCK",
    "INT32_MAX",
    "BatchedGeometry",
    "LaunchWork",
    "batched_launch_grid",
    "check_batched_inputs",
    "has_triton",
    "launch_work",
    "paged_attention_batched",
    "paged_attention_batched_triton",
    "select_backend",
    "slot_index_dtype",
]


@dataclass(frozen=True)
class BatchedGeometry:
    """Every integer the batched kernel turns into a pointer, derived once on the host.

    batch, n_q:   rows decoding together, and query heads. The launch grid is exactly
                  their product, one independent online softmax each.
    n_kv:         compact KV heads; `n_q == n_kv * n_rep` under GQA.
    head_dim:     channels in one head, the extent the `BLOCK_D` ramp is masked to.
    num_slots:    slots in the shared pool, which every row addresses into.
    channels:     `n_kv * head_dim`, the stride from one pool slot to the next. This
                  is what a block table's slot gets multiplied by; get it wrong and a
                  program reads a neighbouring token's key and nothing objects.
    max_ctx:      the mapping's row stride, which is its *width* and not the longest
                  row in the batch. Under Day 61's streamed bucket set it is
                  `max_model_len` forever, and it is deliberately the only place that
                  number reaches the kernel at all.
    stride_q_row: `n_q * head_dim`, the stride from one row's queries to the next in
                  the flattened `[batch, n_q, head_dim]` buffer, and the same stride
                  for the output.

    Frozen because it is a description of a launch that is about to happen, and a
    description that can be edited after the launch is a description of nothing.
    """

    batch: int
    n_q: int
    n_kv: int
    head_dim: int
    num_slots: int
    channels: int
    max_ctx: int
    stride_q_row: int

    @property
    def pool_elements(self) -> int:
        """Length of the flat K (or V) pool: `num_slots * channels`.

        The largest offset any program forms is one less than this, so this single
        product decides whether the kernel's addresses fit in 32 bits.
        """
        return self.num_slots * self.channels

    @property
    def addresses_fit_int32(self) -> bool:
        """True when every offset into the pool fits a signed 32-bit integer."""
        return self.pool_elements - 1 <= INT32_MAX

    @property
    def programs(self) -> int:
        """Programs this geometry launches: `batch * n_q`."""
        return self.batch * self.n_q


def check_batched_inputs(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    n_rep: int,
    context_bounds: tuple[int, int] | None = None,
    validated: bool = False,
) -> BatchedGeometry:
    """Validate the batched decode inputs and derive the kernel's geometry.

    Same arguments as `paged_attention_batched_reference`, including Day 50's
    `validated` / `context_bounds` pair, and it makes every refusal that reference
    makes: a kernel that accepts inputs its oracle rejects is a kernel that cannot be
    compared to it. Returns a `BatchedGeometry`; raises `ValueError` on anything the
    kernel would otherwise turn into a wrong pointer.

    Two of the checks are stricter than the oracle's, and both are strictness the
    oracle gets for free from torch. A head_dim that disagrees with the pool dies in
    the oracle's matmul with torch's own message; here it would be `slot * channels +
    kv_head * head_dim`, two plausible numbers producing a real key belonging to
    another head. An `n_rep` that does not tile the query heads is the same story one
    axis over. A jitted body has no shapes, only pointers and integers, so every
    invariant that keeps the arithmetic inside the buffer is stated once, here.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be [batch, n_q, 1, d]; got {tuple(q.shape)}")
    if q.shape[2] != 1:
        raise ValueError(
            f"the batched read is the decode read: one new token per row, got "
            f"seq_q={q.shape[2]}. A ragged prefill goes through the dense masked path"
        )
    if k_pool.ndim != 3 or k_pool.shape != v_pool.shape:
        raise ValueError(
            "k_pool and v_pool must have the same shape [num_slots, n_kv, d]: a token's "
            f"key and value share a slot; got {tuple(k_pool.shape)} and {tuple(v_pool.shape)}"
        )
    if slot_mapping.ndim != 2:
        raise ValueError(
            f"slot_mapping must be [batch, max_ctx]; got {tuple(slot_mapping.shape)}"
        )
    batch, n_q, _, head_dim = q.shape
    num_slots, n_kv, pool_dim = k_pool.shape
    max_ctx = int(slot_mapping.shape[1])
    if slot_mapping.shape[0] != batch or tuple(context_lens.shape) != (batch,):
        raise ValueError(
            f"slot_mapping and context_lens must have one row each per sequence "
            f"(batch={batch}); got {tuple(slot_mapping.shape)} and "
            f"{tuple(context_lens.shape)}"
        )
    if head_dim != pool_dim:
        raise ValueError(f"head_dim disagrees with the pool: q has {head_dim}, pool has {pool_dim}")
    if n_rep < 1 or n_kv * n_rep != n_q:
        raise ValueError(
            f"n_rep must tile the query heads: n_q ({n_q}) != n_kv ({n_kv}) * n_rep ({n_rep}); "
            "the kernel maps query head h to KV head h // n_rep and cannot check the range"
        )
    if validated and context_bounds is not None:
        raise ValueError(
            "a read is either validated by its caller or handed bounds to validate "
            "with, not both: validated=True says the check already happened, and "
            "context_bounds says do it here with these two numbers"
        )
    if not validated:
        shortest, longest = (
            context_bounds
            if context_bounds is not None
            else (int(context_lens.min()), int(context_lens.max()))
        )
        if shortest < 1:
            raise ValueError(
                "context_lens must be at least 1 for every row: a query with no visible "
                "key softmaxes over nothing (0/0). Here the program's loop does not run "
                "at all, so the division is 0/0 in a lane that never faulted"
            )
        if longest > max_ctx:
            raise ValueError(
                f"context_lens claims more history than the mapping holds: max "
                f"{longest} > max_ctx {max_ctx}"
            )
    return BatchedGeometry(
        batch=batch,
        n_q=n_q,
        n_kv=n_kv,
        head_dim=head_dim,
        num_slots=num_slots,
        channels=n_kv * head_dim,
        max_ctx=max_ctx,
        stride_q_row=n_q * head_dim,
    )


def batched_launch_grid(batch: int, n_q: int) -> tuple[int, int]:
    """The launch grid: one program per (row, query head).

    The grid vLLM's paged kernel launches, and the reason is the accumulator. One
    program keeps a running max, a running denominator and one `head_dim`-wide
    weighted-V sum, and those live in registers. A row's query heads share the slot
    lookup and nothing else, since each reads its own KV head's channels out of the
    tile, so handing one program all of them would multiply the register state for no
    saved traffic. Nothing is reduced across programs, so the grid is the full
    product and the launch needs no second pass.
    """
    return (batch, n_q)


def slot_index_dtype(geom: BatchedGeometry) -> torch.dtype:
    """The integer width the kernel's pool addresses have to be computed in.

    Triton takes the type of an offset from the types it was built out of, so
    `slots[:, None] * stride_slot` with an int32 `slots` is an int32 multiply. Past
    `INT32_MAX` that multiply wraps, and the result is not a fault: it is a negative
    offset, which is a legal offset into whatever the allocator happened to put in
    front of the pool, under a mask that says the lane is valid. The softmax over it
    comes back finite and plausible.

    So the rule is the product and not a guess: int32 while `num_slots * channels`
    addresses inside the ceiling, int64 the moment it does not. Widening costs
    address registers on every lane of every tile, which is why it is not simply
    always on, and the boundary is exact rather than a safety margin because the
    arithmetic is exact.
    """
    return torch.int32 if geom.addresses_fit_int32 else torch.int64


@dataclass(frozen=True)
class LaunchWork:
    """What a batch asks of the grid, as a sum and as a max. Day 62.

    Day 59's `StreamedWork` counts tiles walked against the tiles a rectangle of the
    same width implies, and that ratio is the read's *work* saving. A grid does not
    bill in work. It bills in waves: programs are resident until they retire, and a
    batch is finished when its slowest program is. Those two numbers are equal only
    when the grid is big enough that skipped tiles turn into skipped scheduling, and
    on a decode batch of a few dozen rows it usually is not.

    rows, n_q:     the launch grid's two axes.
    block:         keys folded per program step, the SRAM tile.
    context_width: the mapping's width, which is what a rectangle read would walk for
                   every row. Defaults to the longest row, which is what an
                   unbucketed mapping is; under Day 61's streamed bucket set it is
                   `max_model_len` and that is where the wave saving comes from.
    tiles:         tiles walked, summed over every program in the grid.
    tail_tiles:    tiles walked by the slowest program, which is the longest row.

    Pure host arithmetic over Python ints, like `streamed_work`: no pool, no device,
    no tensor contents. Frozen because it is a measurement.
    """

    rows: int
    n_q: int
    block: int
    context_width: int
    tiles: int
    tail_tiles: int

    @property
    def programs(self) -> int:
        """Programs the launch starts: `rows * n_q`, one softmax each."""
        return self.rows * self.n_q

    @property
    def rectangle_tiles(self) -> int:
        """The same count for a read that gives every program the whole width."""
        return self.programs * cdiv(self.context_width, self.block)

    @property
    def mean_tiles(self) -> float:
        """Tiles the average program walks. The grid's work, per lane of parallelism."""
        return self.tiles / self.programs

    @property
    def imbalance(self) -> float:
        """How many times the slowest program outlasts the average one.

        1.0 on a uniform batch, and that is the good case: every program retires
        together and there is nothing a split could recover. It rises with the
        batch's length spread, which is the same spread `ragged_saving` turns into a
        win, so the two numbers move in opposite directions on the same batch. That
        is not a contradiction, it is the reason flash-decoding exists.
        """
        return self.tail_tiles / self.mean_tiles

    @property
    def work_saving(self) -> float:
        """Rectangle tiles over walked tiles: what an oversubscribed grid collects.

        When there are far more programs than the machine can hold at once, the
        hardware is a queue and tiles not walked are waves not run, so the whole
        ragged saving is real. This is Day 59's `ragged_saving` with the head axis
        multiplied through, which changes nothing because it multiplies both sides.
        """
        return self.rectangle_tiles / self.tiles

    @property
    def wave_saving(self) -> float:
        """Width tiles over tail tiles: what a fully resident grid collects.

        The other end of the same batch. If every program is resident, the launch
        takes as long as its slowest program, so the only thing the ragged walk saved
        is the difference between the longest row and the mapping's width. On an
        unbucketed mapping those are the same number and this is exactly 1.0x: a read
        that saves a great deal of memory and no time at all.
        """
        return cdiv(self.context_width, self.block) / self.tail_tiles

    def render(self) -> str:
        """One line, for a bench table."""
        return (
            f"{self.programs} programs x {self.block}-key tiles: {self.tiles} walked "
            f"of {self.rectangle_tiles}, tail {self.tail_tiles} "
            f"(work {self.work_saving:.2f}x, wave {self.wave_saving:.2f}x, "
            f"imbalance {self.imbalance:.2f}x)"
        )


def launch_work(
    context_lens,
    n_q: int,
    block: int,
    context_width: int | None = None,
) -> LaunchWork:
    """Count what a batch asks of the grid: tiles summed, and tiles at the tail.

    context_lens:  per-row history lengths, as a tensor (what the planned decode path
                   holds) or any sequence of ints.
    n_q:           query heads, the grid's second axis. Every head of a row walks the
                   same tiles, so it multiplies both the program count and the total
                   and cancels out of every ratio except the absolute counts.
    block:         keys folded per program step, as in the kernel.
    context_width: the mapping's width; defaults to the longest row.

    Host arithmetic only, so a bench can ask it about a 256-row batch at an
    8192-token width on a laptop.
    """
    if n_q < 1:
        raise ValueError(f"a launch needs at least one query head; got {n_q}")
    if block < 1:
        raise ValueError(f"a tile holds at least one key; got {block}")
    lens = [int(n) for n in context_lens]
    if not lens:
        raise ValueError("a decode batch has at least one row; got no context lengths")
    if min(lens) < 1:
        raise ValueError(
            "context_lens must be at least 1 for every row: a query with no visible "
            "key softmaxes over nothing, and here it also walks no tiles"
        )
    longest = max(lens)
    width = longest if context_width is None else context_width
    if longest > width:
        raise ValueError(
            f"a row cannot hold more history than the mapping's width: longest "
            f"{longest} > context_width {width}"
        )
    return LaunchWork(
        rows=len(lens),
        n_q=n_q,
        block=block,
        context_width=width,
        tiles=n_q * sum(cdiv(n, block) for n in lens),
        tail_tiles=cdiv(longest, block),
    )


if triton is not None:  # pragma: no cover - compiled and run only on a GPU box

    @triton.jit
    def _paged_attention_batched_fwd(
        q_ptr,  # [batch, n_q, head_dim] contiguous, one new query per row and head
        k_ptr,  # [num_slots * channels] the layer's flat physical K pool, shared
        v_ptr,  # [num_slots * channels] the layer's flat physical V pool, shared
        slot_ptr,  # [batch * max_ctx] logical position -> physical slot, per row
        ctx_ptr,  # [batch] how many of a row's slots are real
        out_ptr,  # [batch, n_q, head_dim] contiguous
        scale,  # softmax scale, usually head_dim ** -0.5
        stride_map,  # max_ctx: elements from one row's mapping to the next
        stride_slot,  # channels = n_kv * head_dim: elements from one slot to the next
        stride_q_row,  # n_q * head_dim: elements from one row's queries to the next
        HEAD_DIM: tl.constexpr,  # the true head dimension
        N_REP: tl.constexpr,  # GQA repeat: query heads per KV head
        BLOCK_D: tl.constexpr,  # HEAD_DIM padded up to a power of two
        BLOCK_N: tl.constexpr,  # keys folded per step, the SRAM tile
    ):
        """One program: row `i`'s query head `h`, streamed over that row's own history.

        Day 59's loop, unchanged in structure and one axis wider than Day 23's. The
        program loads its single query row into registers, reads its own row's
        context length, then walks the causally complete history `BLOCK_N` keys at a
        time. Each tile is read through this row's slice of the block table, so the
        rows may interleave in the shared pool however the allocator left them, and
        folded into an online softmax, so the `[ctx, head_dim]` history is never
        materialized and neither is the `[n_q, ctx]` score row.

        The loop bound is loaded, not computed. `ctx` comes out of `ctx_ptr` on the
        device, which is what makes the walk ragged (row 0 may go round once while
        row 3 goes round a hundred times out of one launch) and what makes the whole
        thing safe under CUDA-graph capture: the number that changes every step
        changes inside a buffer, never in a Python `range`.
        """
        i = tl.program_id(0)  # which row of the batch
        h = tl.program_id(1)  # which query head
        kv_head = h // N_REP  # GQA: this query head's compact KV head
        ctx = tl.load(ctx_ptr + i)  # this row's own extent: the dynamic loop bound

        # The channel ramp, padded to a power of two. Lanes past HEAD_DIM are masked
        # everywhere: they load 0.0, contribute 0 to the dot product, and are never
        # stored. This is what makes a head_dim of 48 legal.
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        q_off = i * stride_q_row + h * HEAD_DIM + offs_d
        q = tl.load(q_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

        # Online-softmax state, held as 1-element tensors so the loop-carried types
        # never change between iterations (Triton requires that). These are registers,
        # and not one of them has max_ctx in it.
        m = tl.full([1], float("-inf"), dtype=tl.float32)  # running max
        denom = tl.zeros([1], dtype=tl.float32)  # running softmax denominator
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # running weighted-V sum

        base = i * stride_map  # where this row's mapping starts in the flat buffer
        for b in range(0, tl.cdiv(ctx, BLOCK_N)):
            offs_n = b * BLOCK_N + tl.arange(0, BLOCK_N)  # positions in *this* history
            valid = offs_n < ctx  # guards the ragged last tile
            # The paged read: each key's physical slot, then that slot's K/V channels
            # for this KV head. Masked lanes never index either buffer, so the padding
            # past `ctx` is not read, and may be -1 naming a slot that holds NaN.
            slots = tl.load(slot_ptr + base + offs_n, mask=valid, other=0)
            kv_off = slots[:, None] * stride_slot + kv_head * HEAD_DIM + offs_d[None, :]
            tile_mask = valid[:, None] & mask_d[None, :]
            k = tl.load(k_ptr + kv_off, mask=tile_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + kv_off, mask=tile_mask, other=0.0).to(tl.float32)

            s = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_N] scores
            # A masked load returned zeros, and a query scores a real, finite weight
            # against a zero key. Force the phantom lanes to -inf so exp gives exactly
            # zero: out of bounds and out of the softmax are two different jobs.
            s = tl.where(valid, s, float("-inf"))

            # Fold the tile in: renormalize the state to the new running max, then add.
            m_new = tl.maximum(m, tl.max(s, axis=0))
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new)  # [BLOCK_N], unnormalized tile weights
            denom = denom * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m = m_new

        out = acc / denom  # the deferred normalization: one divide, at the very end
        tl.store(out_ptr + q_off, out.to(out_ptr.dtype.element_ty), mask=mask_d)


def paged_attention_batched_triton(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
    block: int = DEFAULT_BLOCK,
    context_bounds: tuple[int, int] | None = None,
    validated: bool = False,
) -> torch.Tensor:
    """Launch the batched paged-attention kernel. Same contract as the oracle.

    q, k_pool, v_pool, slot_mapping, context_lens, n_rep, scale, context_bounds,
    validated: exactly as in `paged_attention_batched_reference`, but every tensor
    must live on the same CUDA device.
    block:   keys folded per program step, the SRAM tile. Rounded up to a power of
             two for `tl.arange`; a pure performance knob, since the mask makes any
             extent return the same attention.

    Returns [batch, n_q, 1, d], matching the oracle to about 1e-4. Streaming the
    softmax reassociates the exponent sums, so it is close and not bit-identical:
    the accuracy trade every flash-attention kernel makes, and the accumulators run
    in fp32 whatever the pool's dtype is.

    Raises `ValueError` on host tensors and `RuntimeError` when Triton is missing,
    rather than letting a launch crash somewhere unreadable. `paged_attention_batched`
    picks this path only when both hold; call it directly when you mean to demand the
    GPU, which is what a kernel test does.
    """
    geom = check_batched_inputs(
        q, k_pool, v_pool, slot_mapping, context_lens, n_rep, context_bounds, validated
    )
    if block < 1:
        raise ValueError(f"a tile holds at least one key; got {block}")
    if q.device.type != "cuda":
        raise ValueError(
            f"the Triton kernel reads device memory; q is on {q.device}. "
            "Use `paged_attention_batched` for a dispatch that falls back to the CPU model"
        )
    if not has_triton():
        raise RuntimeError(
            "the triton package is not installed; `paged_attention_batched` falls back"
        )
    if k_pool.device != q.device or slot_mapping.device != q.device:
        raise ValueError("q, the pools, and slot_mapping must live on the same device")
    if context_lens.device != q.device:
        raise ValueError("context_lens must live on the same device as q: the kernel loads it")

    if scale is None:
        scale = geom.head_dim**-0.5

    # Flatten to the 1-D buffers the kernel addresses: a token's K/V is `channels`
    # contiguous elements at `slot * channels`, and row i's mapping is the `max_ctx`
    # entries at `i * max_ctx`. `.contiguous()` is what makes that arithmetic true,
    # so it is not optional even when it is usually a no-op.
    q_rows = q[:, :, 0, :].contiguous()  # [batch, n_q, head_dim]
    k_flat = k_pool.contiguous().reshape(-1)
    v_flat = v_pool.contiguous().reshape(-1)
    slots = slot_mapping.to(slot_index_dtype(geom)).contiguous().reshape(-1)
    lens = context_lens.to(torch.int32).contiguous()
    out = torch.empty_like(q_rows)

    _paged_attention_batched_fwd[batched_launch_grid(geom.batch, geom.n_q)](
        q_rows,
        k_flat,
        v_flat,
        slots,
        lens,
        out,
        scale,
        geom.max_ctx,
        geom.channels,
        geom.stride_q_row,
        HEAD_DIM=geom.head_dim,
        N_REP=n_rep,
        BLOCK_D=next_power_of_2(geom.head_dim),
        BLOCK_N=next_power_of_2(block),
        num_warps=4,
    )
    return out[:, :, None, :]  # [batch, n_q, 1, d], the shape o_proj expects


def paged_attention_batched(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
    block: int = DEFAULT_BLOCK,
    context_bounds: tuple[int, int] | None = None,
    validated: bool = False,
) -> torch.Tensor:
    """The streamed decode read, on whichever backend this device can actually run.

    The public entry point, and what `PagedRead` calls under `STREAMED`. On a CUDA
    tensor with Triton installed it launches the kernel above; anywhere else it runs
    Day 59's `paged_attention_batched_kernel`, the tlsim model of the same loop. Both
    read K/V through the block table, both walk each row its own length, neither
    builds the score rectangle, and both agree with
    `paged_attention_batched_reference` to a few ulps. So the backend is a speed
    decision and never a numerics one, which is the only claim that makes a fallback
    honest rather than convenient.

    The fallback is the model, not a slow path that happens to be correct. It is the
    loop in Python doing a masked gather per tile, roughly an order of magnitude
    slower per call than the torch rectangle it replaces, and it exists so the engine
    runs on a laptop while the kernel runs on the card.
    """
    if select_backend(q.device) == "triton":
        return paged_attention_batched_triton(
            q,
            k_pool,
            v_pool,
            slot_mapping,
            context_lens,
            n_rep,
            scale,
            block=block,
            context_bounds=context_bounds,
            validated=validated,
        )
    return paged_attention_batched_kernel(
        q,
        k_pool,
        v_pool,
        slot_mapping,
        context_lens,
        n_rep,
        scale,
        block=block,
        context_bounds=context_bounds,
        validated=validated,
    )
