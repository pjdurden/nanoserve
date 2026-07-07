"""A tiny CPU model of the Triton programming model. Week 6, Day 21.

Day 20 measured the number the paged-attention kernel has to beat. Before writing
that kernel, this module makes its *shape* concrete on a box with no GPU and no
Triton installed. A Triton kernel is not ordinary array code: you `triton.jit` a
function, launch it over a *grid* of programs, and each program runs the identical
body with only its `tl.program_id` different. Inside, you do not slice tensors; you
compute flat integer offsets from the program id, form block pointers, and move
data with masked `tl.load` / `tl.store`. The mask is the load-bearing idea: it is
what lets the last, not-full tile of a sequence read and write without walking off
the end of a buffer.

`tlsim` is a faithful, deliberately slow model of exactly those primitives, so the
paged read's inner loop can be *written and tested* in plain terms first:

* `launch(grid, kernel, *args)` runs `kernel(prog, *args)` once per program id, the
  SPMD loop the hardware runs in parallel and we run in order.
* `Program.program_id(axis)` / `.num_programs(axis)` are the only things that differ
  between two runs of the same body, the way each program finds its own tile.
* `load(buf, offsets, mask, other)` / `store(buf, offsets, value, mask)` are the
  masked gather/scatter over a flat 1-D buffer: masked-off lanes read `other` and
  are never written, and crucially never index the buffer, so an out-of-bounds
  offset under a false mask cannot fault (the property the ragged tail relies on).
* `arange(start, end)` and `cdiv(a, b)` build the offset ramp and size the grid.

None of this is fast; a masked `torch.where` gather is slower than the plain index
it models. The point is the mental model. `paged_gather` below rewrites the Day-18
`k_pool[slot_mapping]` read as a grid of programs, each gathering a tile of
positions through the block table with the tail guarded by a mask, and it returns
the byte-identical history. Next week the same loop becomes a `triton.jit` kernel;
this is that loop on understood ground rather than as a copied incantation.

Batch and axes: the sim supports a multi-dimensional launch grid (`itertools`
product of the ranges), which is all the paged kernel needs. Scope is one sequence,
the same scope as `PagedKVCache`; per-sequence batching arrives in Phase 3.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import torch


class Program:
    """A single program in the launch grid: the SPMD unit, like a Triton instance.

    Every program runs the same kernel body; the only thing that separates two
    programs is the coordinate this handle reports. A kernel reads `program_id(0)`
    to decide which tile of the work it owns, and `num_programs(0)` to know the grid
    size (used to place the ragged last tile). The coordinate is fixed at launch and
    read-only, mirroring `tl.program_id`.
    """

    def __init__(self, pid: tuple[int, ...], grid: tuple[int, ...]) -> None:
        self._pid = pid
        self._grid = grid

    def program_id(self, axis: int = 0) -> int:
        """This program's coordinate along `axis` (0-based), like `tl.program_id`."""
        return self._pid[axis]

    def num_programs(self, axis: int = 0) -> int:
        """The grid extent along `axis`, like `tl.num_programs`."""
        return self._grid[axis]


def launch(grid: int | tuple[int, ...], kernel: Callable[..., None], *args) -> None:
    """Run `kernel(prog, *args)` once for every program id in `grid`.

    grid:   an int for a 1-D launch, or a tuple for a multi-dimensional one. The
            paged read uses a 1-D grid (one program per tile of positions); the
            tuple form is here so the model matches Triton's up-to-3-D launch.
    kernel: the "jitted" body. It takes a `Program` as its first argument and reads
            `program_id`/`num_programs` off it; it does its work by mutating buffers
            passed in `args` through `store` (kernels return nothing, like Triton).

    On hardware the programs run in parallel in an unspecified order; here they run
    sequentially in row-major (lexicographic) program-id order, which is why a
    correct kernel must give each program a disjoint tile: order must not matter.
    """
    if isinstance(grid, int):
        grid = (grid,)
    for pid in itertools.product(*(range(g) for g in grid)):
        kernel(Program(pid, grid), *args)


def arange(start: int, end: int) -> torch.Tensor:
    """The integer ramp `[start, start+1, ..., end-1]`, like `tl.arange`.

    This is how a program turns its `program_id` into the block of offsets it owns:
    `program_id(0) * BLOCK + arange(0, BLOCK)`. Always `long`, because these values
    index a buffer.
    """
    return torch.arange(start, end, dtype=torch.long)


def cdiv(a: int, b: int) -> int:
    """Ceiling division, like `triton.cdiv`: the grid size for `a` items in `b`-tiles.

    Rounds up so the last, partial tile still gets its own program; that program's
    out-of-range lanes are handled by the load/store mask, not by shrinking the grid.
    """
    return -(-a // b)


def load(
    buffer: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor | None = None,
    other: float = 0.0,
) -> torch.Tensor:
    """Masked gather from a flat buffer: `tl.load(buffer + offsets, mask, other)`.

    buffer:  a 1-D tensor, the flat memory the pointers address.
    offsets: a long tensor of any shape; the returned tensor has this shape.
    mask:    a bool tensor broadcastable to `offsets`; where false, that lane reads
             `other` instead of the buffer. `None` means every lane reads.
    other:   the value returned for masked-off lanes (Triton's `other`, default 0).

    The masked-off lanes are never used to index `buffer`, so an out-of-bounds
    offset under a false mask is safe (it cannot fault). That is the exact guarantee
    a kernel leans on to read the ragged last tile of a sequence: point the tail
    lanes anywhere, mask them off, and they quietly return `other`.
    """
    if mask is None:
        return buffer[offsets]
    # Neutralize the masked-off (possibly out-of-bounds) offsets before indexing,
    # then overwrite those lanes with `other`. torch.where broadcasts the mask.
    safe = torch.where(mask, offsets, torch.zeros_like(offsets))
    gathered = buffer[safe]
    return torch.where(mask.expand_as(gathered), gathered, torch.full_like(gathered, other))


def store(
    buffer: torch.Tensor,
    offsets: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    """Masked scatter into a flat buffer: `tl.store(buffer + offsets, value, mask)`.

    buffer:  a 1-D tensor, mutated in place.
    offsets: a long tensor of any shape addressing `buffer`.
    value:   a tensor broadcastable to `offsets`, the data to write.
    mask:    a bool tensor broadcastable to `offsets`; where false, that slot is
             left exactly as it was. `None` writes every lane.

    Masked-off lanes are neither read from `value` nor written to `buffer`, so the
    ragged tail's out-of-range slots stay untouched, the write-side mirror of the
    load mask.
    """
    if mask is None:
        buffer[offsets] = value.expand_as(offsets).to(buffer.dtype)
        return
    m = mask.expand_as(offsets)
    buffer[offsets[m]] = value.expand_as(offsets)[m].to(buffer.dtype)


def paged_gather(pool: torch.Tensor, slot_mapping: torch.Tensor, block: int) -> torch.Tensor:
    """The paged read, `pool[slot_mapping]`, rewritten as a grid of programs.

    pool:         [num_slots, n_kv, d], a layer's flat physical K (or V) pool, the
                  same tensor `PagedKVCache.k_pool[layer]` holds. A "slot" is the flat
                  index `block_id * block_size + offset`.
    slot_mapping: [seq_total] long tensor; `slot_mapping[p]` is the physical slot of
                  logical position p (oldest first), i.e. `[table.slot(p) for p ...]`.
    block:        the tile size, how many positions one program gathers. A pure
                  performance knob: any value returns the identical history.

    Returns [seq_total, n_kv, d], the ordered history the Day-18 reference reads
    before it scores, byte-identical to `pool[slot_mapping]`. This is the read the
    Triton kernel will do block by block; here it is a grid of `cdiv(seq_total,
    block)` programs, each gathering `block` positions' K/V through their slots with
    a mask over the ragged tail.

    Each token's K/V is `n_kv * d` contiguous elements, so the pool and the output
    are viewed as flat [n, C] matrices (C = n_kv * d) and a program moves a
    [block, C] tile: rows are positions (masked on the tail), columns are the C
    channels of one token.
    """
    num_slots, n_kv, d = pool.shape
    channels = n_kv * d
    seq_total = int(slot_mapping.shape[0])

    slot_buf = slot_mapping.to(torch.long)
    pool_flat = pool.reshape(num_slots * channels)
    out_flat = torch.zeros(seq_total * channels, dtype=pool.dtype)

    def kernel(prog, slots, src_flat, dst_flat) -> None:
        # This program owns positions [pid*block, pid*block+block); rows past the
        # end of the history are masked off (the ragged last tile).
        rows = prog.program_id(0) * block + arange(0, block)  # [block]
        row_mask = rows < seq_total
        # Each valid row's physical slot; masked rows read slot 0 (unused).
        row_slots = load(slots, rows, mask=row_mask, other=0)  # [block]
        cols = arange(0, channels)  # [channels]
        # Block pointers: [block, channels] flat offsets into the pool and output.
        src = row_slots[:, None] * channels + cols[None, :]
        dst = rows[:, None] * channels + cols[None, :]
        tile = load(src_flat, src, mask=row_mask[:, None], other=0.0)
        store(dst_flat, dst, tile, mask=row_mask[:, None])

    launch(cdiv(seq_total, block), kernel, slot_buf, pool_flat, out_flat)
    return out_flat.reshape(seq_total, n_kv, d)
