"""Day 21 tests: the tiny CPU model of the Triton programming model.

Week 6's headline is a hand-written Triton paged-attention kernel, and Day 20
measured the number it has to beat. But a Triton kernel is not ordinary Python:
it launches a *grid* of programs, each one owning a tile of the work, and it does
not slice tensors the way torch does. It computes flat integer offsets, forms
block pointers, and reads and writes through masked `tl.load` / `tl.store`, where
the mask is what keeps the ragged last tile from reading past the end of a buffer.
Get that mental model wrong and the kernel faults or reads garbage no reference
can catch.

`nanoserve.kernels.tlsim` is a stdlib-plus-torch model of exactly those
primitives, run on the CPU with no GPU and no Triton installed: a `launch` that
runs a kernel once per program id, a `Program` handle that answers `program_id`
and `num_programs`, and masked `load` / `store` over a flat buffer with an `arange`
and a `cdiv` to build the offsets. Nothing here is fast. Its whole job is to make
the paged read's inner loop expressible and *testable* in plain terms, so the real
`triton.jit` kernel next week is a translation of an understood loop rather than a
copied incantation.

The payoff test is `paged_gather`: the same `k_pool[slot_mapping]` read the Day-18
reference does, rewritten as a grid of programs each gathering a tile of positions
through the block table, the ragged tail guarded by a mask. It must reproduce the
torch index byte-for-byte and feed the exact reference softmax, so the loop the
kernel will run is pinned before a line of Triton is written.
"""

from __future__ import annotations

import torch

from nanoserve.cache import BlockAllocator, BlockTable
from nanoserve.config import ModelConfig
from nanoserve.kernels.paged_attention import paged_attention_reference
from nanoserve.kernels.tlsim import arange, cdiv, launch, load, paged_gather, store
from nanoserve.layers import repeat_kv


# --- the launch grid and the program handle --------------------------------


def test_launch_runs_the_kernel_once_per_program_id_in_order():
    """A grid of N launches N programs, program ids 0..N-1, each exactly once.

    This is the whole SPMD contract: the same kernel body runs many times, and the
    only thing that differs between runs is `program_id`. The sim runs them in
    order so a test can read the trace, but real hardware may run them in any
    order, which is exactly why each program must own a disjoint tile.
    """
    seen = []

    def kernel(prog):
        seen.append((prog.program_id(0), prog.num_programs(0)))

    launch(5, kernel)
    assert [pid for pid, _ in seen] == [0, 1, 2, 3, 4]
    assert all(n == 5 for _, n in seen)


def test_launch_accepts_an_int_or_a_one_tuple_grid():
    """`launch(5, ...)` and `launch((5,), ...)` describe the same grid."""
    a, b = [], []
    launch(3, lambda prog: a.append(prog.program_id(0)))
    launch((3,), lambda prog: b.append(prog.program_id(0)))
    assert a == b == [0, 1, 2]


# --- masked load and store, the primitives that keep the tail in bounds -----


def test_load_without_a_mask_is_a_plain_gather():
    """No mask means every lane reads: `load(buf, offs)` is `buf[offs]`."""
    buf = torch.tensor([10.0, 20.0, 30.0, 40.0])
    offs = torch.tensor([3, 1, 0])
    assert torch.equal(load(buf, offs), buf[offs])


def test_masked_load_returns_other_where_the_mask_is_false():
    """The ragged-tail lesson: a masked-off lane reads `other`, never the buffer.

    The offset 99 is out of bounds; because its mask lane is false the load must
    return `other` for it and must not fault. That is precisely how a Triton kernel
    reads the last, not-full tile of a history without walking off the end of the
    KV pool.
    """
    buf = torch.tensor([10.0, 20.0, 30.0])
    offs = torch.tensor([0, 1, 99, 2])  # 99 would be out of bounds
    mask = torch.tensor([True, True, False, True])
    out = load(buf, offs, mask=mask, other=-1.0)
    assert torch.equal(out, torch.tensor([10.0, 20.0, -1.0, 30.0]))


def test_masked_store_writes_only_the_valid_lanes():
    """A masked store leaves the masked-off slots exactly as they were.

    Store into a sentinel buffer with half the lanes masked off; the untouched
    slots keep their sentinel. This is the write side of the same tail guard: the
    last tile writes only its real rows.
    """
    buf = torch.full((5,), -1.0)
    offs = torch.tensor([0, 1, 2, 3])
    value = torch.tensor([10.0, 20.0, 30.0, 40.0])
    mask = torch.tensor([True, False, True, False])
    store(buf, offs, value, mask=mask)
    assert torch.equal(buf, torch.tensor([10.0, -1.0, 30.0, -1.0, -1.0]))


def test_arange_builds_the_offset_ramp():
    """`arange(a, b)` is the 0..n ramp offsets are built from, like `tl.arange`."""
    assert torch.equal(arange(0, 4), torch.tensor([0, 1, 2, 3]))
    assert torch.equal(arange(4, 7), torch.tensor([4, 5, 6]))
    assert arange(0, 4).dtype == torch.long


def test_cdiv_rounds_the_grid_up():
    """`cdiv` sizes the grid so the last, partial tile still gets a program."""
    assert cdiv(8, 4) == 2  # exact
    assert cdiv(10, 4) == 3  # ragged: 2 full tiles plus a tail
    assert cdiv(1, 4) == 1
    assert cdiv(0, 4) == 0


# --- paged_gather: the paged read written as a grid of programs -------------


def test_paged_gather_reproduces_the_torch_index():
    """The headline: the gridded gather equals `pool[slot_mapping]`, tail and all.

    Ten positions in tiles of four means two full tiles and a ragged tail of two,
    so the mask path is exercised. The result must be byte-identical to the plain
    torch index the Day-18 reference uses; the kernel just reads it a tile at a
    time instead of all at once.
    """
    torch.manual_seed(0)
    num_slots, n_kv, d = 16, 2, 4
    pool = torch.randn(num_slots, n_kv, d)
    seq_total = 10
    slot_mapping = torch.randperm(num_slots)[:seq_total]

    out = paged_gather(pool, slot_mapping, block=4)
    assert out.shape == (seq_total, n_kv, d)
    assert torch.equal(out, pool[slot_mapping])


def test_paged_gather_is_transparent_to_block_size():
    """SPMD tiling is invisible: every block size gives the identical gather.

    Block sizes that divide the history, that leave a ragged tail, and that are
    larger than the whole history (one partial tile) must all produce the same
    answer. The tile size is a performance knob, never a correctness one.
    """
    torch.manual_seed(1)
    num_slots, n_kv, d = 32, 2, 4
    pool = torch.randn(num_slots, n_kv, d)
    seq_total = 13  # prime, so most block sizes leave a tail
    slot_mapping = torch.randperm(num_slots)[:seq_total]

    truth = pool[slot_mapping]
    for block in (1, 2, 3, 4, 8, 13, 20):
        assert torch.equal(paged_gather(pool, slot_mapping, block), truth)


def test_paged_gather_reads_through_physically_scattered_blocks():
    """The paged point: scattered physical blocks give the same gathered history.

    A fragmented block table places the history on non-contiguous physical blocks,
    exactly the layout the real cache produces. The gridded gather follows each
    position's slot regardless, so the ordered history it returns is the same as a
    plain index over the scattered pool. Paging moved where the K/V live, not what
    the read returns.
    """
    alloc = BlockAllocator(num_blocks=8, block_size=4)
    warm = BlockTable(alloc)
    warm.append(8)
    warm.free()  # hand two blocks back so the next table reuses them LIFO
    table = BlockTable(alloc)
    seq_total = 10
    table.append(seq_total)
    assert table.block_ids != sorted(table.block_ids)  # genuinely scattered

    slot_mapping = torch.tensor([table.slot(p) for p in range(seq_total)], dtype=torch.long)
    num_slots = alloc.num_blocks * alloc.block_size

    torch.manual_seed(2)
    pool = torch.randn(num_slots, 2, 4)
    out = paged_gather(pool, slot_mapping, block=4)
    assert torch.equal(out, pool[slot_mapping])


def test_paged_gather_feeds_the_reference_softmax():
    """End to end: gather K and V with the sim, then the reference math matches.

    The gathered history is exactly what the Day-18 reference reads before it
    scores. So building attention from the sim-gathered K/V and comparing to
    `paged_attention_reference` over the same pools must agree to the byte: the
    kernel's job is only to produce this history through masked loads, the softmax
    on top is unchanged. This is the seam the Triton kernel slots into.
    """
    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups
    seq_total = 7

    torch.manual_seed(3)
    q = torch.randn(1, n_q, 1, d)  # one decode query over the whole history
    k_pool = torch.randn(seq_total, n_kv, d)
    v_pool = torch.randn(seq_total, n_kv, d)
    slot_mapping = torch.arange(seq_total, dtype=torch.long)

    # Gather the ordered history through the sim, then run the same causal softmax
    # the reference runs, and demand agreement with the reference over the pools.
    k_hist = paged_gather(k_pool, slot_mapping, block=4).transpose(0, 1)[None]
    v_hist = paged_gather(v_pool, slot_mapping, block=4).transpose(0, 1)[None]
    k = repeat_kv(k_hist, n_rep)
    v = repeat_kv(v_hist, n_rep)
    scores = torch.matmul(q, k.transpose(2, 3)) * (d**-0.5)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    mine = torch.matmul(weights, v)

    truth = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert torch.equal(mine, truth)
