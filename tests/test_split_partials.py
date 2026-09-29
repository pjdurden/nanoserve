"""Day 70 tests: pass one gets its own entry point, and every chunk is checked by name.

Day 69 lifted pass two out of the split read so a test could hand it partials. This
is the same move one pass earlier. `_paged_attention_split_fwd` could only be reached
through the read, and the read only shows a test the *reduced* answer: whatever pass
one stored in a chunk's slot is folded with every other chunk before anyone looks.

That hides a specific mistake. `_overrun` below is pass one with its chunk end
dropped, `hi = ctx` instead of `min(lo + chunk, ctx)`, so every chunk walks from its
own start to the row's end and the later keys are folded more than once. On a row
that fits in one chunk it is exact, because there is only one chunk and it stops at
the row's end either way. That is the same shape as Day 68's mutant: right whenever
exactly one chunk is live, which is every request shorter than the partition.

So pass one now has a door too. `split_partials_kernel` is the tlsim pass one, lifted
out of `paged_attention_split_kernel` unchanged; `split_partials_triton` launches
`_paged_attention_split_fwd` alone; `split_partials` dispatches. Each returns the
workspace, `(part_max, part_denom, part_acc)`, and a test compares one chunk's slot
against a direct softmax over exactly that chunk's keys. The oracle below is that
softmax, written chunk by chunk in plain torch, and the mutant is the same oracle
with one flag flipped, so the control and the thing it controls cannot drift apart.

The device half skips on this box, like Day 69's.
"""

from __future__ import annotations

import pytest
import torch
from reference import requires_triton_gpu

from nanoserve.kernels import flash_decoding
from nanoserve.kernels.flash_decoding import (
    PARTIAL_DTYPE,
    SplitUnsound,
    paged_attention_split_kernel,
    reduce_partials,
    split_partials,
    split_partials_kernel,
    split_partials_triton,
)
from nanoserve.kernels.paged_attention import paged_attention_batched_reference


def _case(lens, width, n_q=2, n_rep=1, d=8, num_slots=96, seed=0, device="cpu"):
    """A decode batch over a scattered pool: `(q, k_pool, v_pool, mapping, lens)`."""
    generator = torch.Generator().manual_seed(seed)
    n_kv = n_q // n_rep
    k_pool = torch.randn(num_slots, n_kv, d, generator=generator)
    v_pool = torch.randn(num_slots, n_kv, d, generator=generator)
    q = torch.randn(len(lens), n_q, 1, d, generator=generator)
    mapping = torch.zeros(len(lens), width, dtype=torch.long)
    for i, n in enumerate(lens):
        mapping[i, :n] = torch.randperm(num_slots, generator=generator)[:n]
    tensors = (q, k_pool, v_pool, mapping, torch.tensor(lens))
    return tuple(t.to(device) for t in tensors)


def _chunk_partials(q, k_pool, v_pool, mapping, lens, n_rep, chunk, count, overrun=False):
    """Each chunk's partial softmax, computed directly from the keys it owns.

    Chunk `s` of row `i` owns positions `[s * chunk, min(s * chunk + chunk, ctx))`.
    Its max is the largest score among them, its denominator is `sum(exp(s - m))`
    and its accumulator is the same weights on the values, unnormalised. A chunk
    that owns nothing is `-inf, 0, 0`. `overrun=True` is the mutant: the chunk's end
    is dropped and it walks to the row's end.
    """
    rows, n_q, _, d = q.shape
    scale = d**-0.5
    part_max = torch.full((rows, n_q, count), float("-inf"))
    part_denom = torch.zeros(rows, n_q, count)
    part_acc = torch.zeros(rows, n_q, count, d)
    for i in range(rows):
        ctx = int(lens[i])
        for h in range(n_q):
            for s in range(count):
                lo = s * chunk
                hi = ctx if overrun else min(lo + chunk, ctx)
                if lo >= hi:
                    continue
                slots = mapping[i, lo:hi].cpu()
                k = k_pool.cpu()[slots, h // n_rep].float()
                v = v_pool.cpu()[slots, h // n_rep].float()
                sc = scale * (k @ q[i, h, 0].cpu().float())
                m = sc.max()
                p = torch.exp(sc - m)
                part_max[i, h, s] = m
                part_denom[i, h, s] = p.sum()
                part_acc[i, h, s] = (p[:, None] * v).sum(dim=0)
    return part_max, part_denom, part_acc


def _overrun_differs(case, n_rep, chunk, count):
    """True when the mutant's partials differ from the honest ones on this case."""
    honest = _chunk_partials(*case, n_rep, chunk, count)
    wrong = _chunk_partials(*case, n_rep, chunk, count, overrun=True)
    return any(not torch.allclose(a, b, atol=1e-5, equal_nan=True) for a, b in zip(honest, wrong))


# --- the controls are held to the mutant first ----------------------------------------


def test_a_batch_of_rows_shorter_than_a_chunk_is_a_control_the_overrun_survives():
    """Every row fits in chunk 0, so chunk 0 stops at the row's end either way and the
    later chunks own nothing. The mutant is exact here, which is what makes it a
    control and not a test."""
    case = _case([5, 3, 8], width=24)
    assert not _overrun_differs(case, n_rep=1, chunk=8, count=3)


def test_a_row_that_crosses_a_chunk_is_one_the_overrun_gets_wrong():
    case = _case([20, 3], width=24)
    assert _overrun_differs(case, n_rep=1, chunk=8, count=3)


def test_the_overrun_is_invisible_to_the_reduced_answer_on_short_rows():
    """Why pass one needs its own door: a test that only sees the reduced output
    cannot tell the mutant from the kernel until some row crosses a chunk."""
    case = _case([5, 3, 8], width=24)
    wrong = reduce_partials(*_chunk_partials(*case, 1, chunk=8, count=3, overrun=True))
    oracle = paged_attention_batched_reference(*case, n_rep=1)
    assert torch.allclose(wrong, oracle, atol=1e-5)


# --- the tlsim pass one, by name -----------------------------------------------------


@pytest.mark.parametrize(
    "lens, width, block, splits, n_rep",
    [
        ([20, 3], 24, 4, 3, 1),  # one row crosses two chunk boundaries
        ([17, 16, 1], 32, 8, 2, 1),  # 17 is one key into chunk 1; 16 ends on the boundary
        ([40, 9], 40, 8, 3, 2),  # GQA: two query heads read one KV head
    ],
)
def test_the_model_pass_one_stores_each_chunk_s_own_softmax(lens, width, block, splits, n_rep):
    case = _case(lens, width=width, n_q=2, n_rep=n_rep, seed=1)
    part_max, part_denom, part_acc = split_partials_kernel(
        *case, n_rep=n_rep, block=block, splits=splits
    )
    count = part_max.shape[2]
    chunk = flash_decoding.partition_width(width, block, splits=splits)[1]
    want = _chunk_partials(*case, n_rep, chunk=chunk, count=count)
    assert part_acc.shape == (len(lens), 2, count, 8)
    assert torch.equal(torch.isinf(part_max), torch.isinf(want[0]))
    assert torch.allclose(part_max, want[0], atol=1e-5, equal_nan=True)
    assert torch.allclose(part_denom, want[1], atol=1e-5)
    assert torch.allclose(part_acc, want[2], atol=1e-5)


def test_an_empty_chunk_stores_minus_inf_zero_zero():
    """A chunk past its row's end still writes its slot, and what it writes is what
    the reduce folds to nothing."""
    case = _case([3, 20], width=24, seed=2)
    part_max, part_denom, part_acc = split_partials_kernel(*case, n_rep=1, block=8, splits=3)
    assert torch.isinf(part_max[0, :, 1:]).all() and (part_max[0, :, 1:] < 0).all()
    assert (part_denom[0, :, 1:] == 0).all()
    assert (part_acc[0, :, 1:] == 0).all()
    assert torch.isfinite(part_max[1]).all()


def test_the_model_pass_one_s_partials_reduce_to_the_oracle():
    case = _case([20, 3, 11], width=24, seed=3)
    partials = split_partials_kernel(*case, n_rep=1, block=4, splits=3)
    got = reduce_partials(*partials)
    oracle = paged_attention_batched_reference(*case, n_rep=1)
    assert torch.allclose(got, oracle, atol=1e-5)


def test_a_handed_in_workspace_is_the_workspace_that_comes_back():
    """Written in place, and returned as itself: the caller's buffers are the answer."""
    case = _case([20, 3], width=24, seed=4)
    part = (
        torch.full((2, 2, 3), 1234.5, dtype=PARTIAL_DTYPE),
        torch.full((2, 2, 3), 1234.5, dtype=PARTIAL_DTYPE),
        torch.full((2, 2, 3, 8), 1234.5, dtype=PARTIAL_DTYPE),
    )
    got = split_partials_kernel(*case, n_rep=1, block=8, partials=part)
    assert all(a is b for a, b in zip(got, part))
    want = _chunk_partials(*case, 1, chunk=8, count=3)
    assert torch.allclose(part[2], want[2], atol=1e-5)


def test_pass_one_refuses_a_workspace_narrowed_on_the_split_axis():
    case = _case([20, 3], width=24, seed=5)
    wide = (
        torch.zeros(2, 2, 6, dtype=PARTIAL_DTYPE),
        torch.zeros(2, 2, 6, dtype=PARTIAL_DTYPE),
        torch.zeros(2, 2, 6, 8, dtype=PARTIAL_DTYPE),
    )
    narrowed = tuple(t[:, :, :3] for t in wide)
    with pytest.raises(SplitUnsound, match="not contiguous"):
        split_partials_kernel(*case, n_rep=1, block=8, partials=narrowed)


def test_pass_one_refuses_what_the_read_refuses():
    q, k_pool, v_pool, mapping, lens = _case([20, 3], width=24, seed=6)
    with pytest.raises(ValueError):
        split_partials_kernel(q, k_pool, v_pool, mapping, lens, n_rep=1, block=0, splits=2)
    with pytest.raises(ValueError):
        split_partials_kernel(q, k_pool, v_pool, mapping, torch.tensor([20, 0]), n_rep=1)


def test_the_read_s_pass_one_is_the_function_this_file_tests(monkeypatch):
    """Swap the overrun into the module and the read breaks on a row that crosses a
    chunk and stays exact on one that does not. That is how the read is shown to call
    `split_partials_kernel`, rather than a private copy of it."""
    case = _case([40, 3], width=40, seed=7)
    honest = paged_attention_split_kernel(*case, n_rep=1, block=8, splits=3)

    calls = []

    def mutant(q, k_pool, v_pool, slot_mapping, context_lens, n_rep, scale=None, **kw):
        calls.append(kw["splits"])
        return _chunk_partials(
            q, k_pool, v_pool, slot_mapping, context_lens, n_rep, 16, 3, overrun=True
        )

    monkeypatch.setattr(flash_decoding, "split_partials_kernel", mutant)
    broken = paged_attention_split_kernel(*case, n_rep=1, block=8, splits=3)
    assert calls == [3]
    assert not torch.allclose(broken[0], honest[0], atol=1e-3)  # 40 keys, three chunks
    assert torch.allclose(broken[1], honest[1], atol=1e-5)  # 3 keys, one live chunk


def test_the_jitted_pass_one_refuses_host_tensors_rather_than_crashing():
    case = _case([20, 3], width=24, seed=8)
    with pytest.raises((RuntimeError, ValueError), match="cuda|CUDA|Triton|triton"):
        split_partials_triton(*case, n_rep=1, block=8, splits=3)


def test_the_dispatch_runs_host_tensors_through_the_model():
    case = _case([20, 3], width=24, seed=9)
    got = split_partials(*case, n_rep=1, block=8, splits=3)
    want = _chunk_partials(*case, 1, chunk=8, count=3)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-5, equal_nan=True)


# --- the device half, gated -----------------------------------------------------------


@requires_triton_gpu
@pytest.mark.parametrize(
    "lens, width, block, splits, n_rep",
    [
        ([20, 3], 24, 4, 3, 1),
        ([17, 16, 1], 32, 8, 2, 1),
        ([40, 9], 40, 8, 3, 2),
    ],
)
def test_the_jitted_pass_one_stores_each_chunk_s_own_softmax(lens, width, block, splits, n_rep):
    """`_paged_attention_split_fwd` alone, graded chunk by chunk, on rows that cross a
    chunk so the overrun cannot pass it."""
    case = _case(lens, width=width, n_q=2, n_rep=n_rep, seed=10, device="cuda")
    chunk = flash_decoding.partition_width(width, block, splits=splits)[1]
    count = flash_decoding.partition_width(width, block, splits=splits)[0]
    assert _overrun_differs(case, n_rep, chunk, count)
    got = split_partials_triton(*case, n_rep=n_rep, block=block, splits=splits)
    want = _chunk_partials(*case, n_rep, chunk=chunk, count=count)
    for a, b in zip(got, want):
        assert torch.allclose(a.cpu(), b, atol=1e-4, equal_nan=True)


@requires_triton_gpu
def test_the_jitted_pass_one_writes_the_workspace_it_was_handed():
    case = _case([20, 3], width=24, seed=11, device="cuda")
    part = (
        torch.full((2, 2, 3), 1234.5, dtype=PARTIAL_DTYPE, device="cuda"),
        torch.full((2, 2, 3), 1234.5, dtype=PARTIAL_DTYPE, device="cuda"),
        torch.full((2, 2, 3, 8), 1234.5, dtype=PARTIAL_DTYPE, device="cuda"),
    )
    got = split_partials_triton(*case, n_rep=1, block=8, partials=part)
    assert all(a is b for a, b in zip(got, part))
    want = _chunk_partials(*case, 1, chunk=8, count=3)
    assert torch.allclose(part[2].cpu(), want[2], atol=1e-4)
