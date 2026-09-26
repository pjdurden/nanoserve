"""Day 69 tests: pass two gets its own entry point, and a test that owns its partials.

Day 68 ended on a mutant that survived. `_split_reduce_fwd`, the jitted second pass,
was changed to keep only the winning chunk, and every test in the suite passed,
because on this box the split runs the tlsim passes and the Triton reduce has never
launched. Day 68's other lesson said why that is not the whole story: even on a card,
an end-to-end split test only exercises the reduce on rows that cross a chunk, and a
row that crosses a 512-key chunk is a 514-token request.

So the reduce stops being reachable only through the read. `split_reduce_kernel` is
the tlsim pass two, lifted out of `paged_attention_split_kernel` unchanged;
`split_reduce_triton` launches `_split_reduce_fwd` alone; `split_reduce` picks one the
way every dispatch here does. Each takes a partial workspace and returns the attention
it folds to, so a test can build the partials by hand, and a hand-built partial is
live by construction. No row has to be long, because there are no rows: there are
three tensors, and the test decides how many of their chunks saw a key.

The controls are Day 68's, moved one level down. `_keep_the_winner` is the mutant,
written as a function, and each fixture here is asserted to be one the mutant gets
wrong before it is used to show the real reduce gets it right. A fixture the mutant
survives is a fixture that tests nothing, and this file refuses to hold one.

The jitted half is gated like every other device test. It will skip on this box,
which is the honest outcome: what changes today is that the day a card exists, the
reduce is checked by name and not only as a side effect of a read.
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
    split_reduce,
    split_reduce_kernel,
    split_reduce_triton,
)


def _workspace(live, rows=2, n_q=3, d=8, seed=0, spread=4.0, device="cpu"):
    """A `[rows, n_q, splits]` partial workspace where `live[s]` says if chunk s saw keys.

    Each live chunk is the honest partial softmax of its own random keys and values,
    scored against a shared query per (row, head), so the reduce has a known answer:
    the softmax over the union of the live chunks. A dead chunk stores exactly what
    pass one stores past a row's end, `-inf`, `0`, `0`. `spread` scales the scores,
    which is how far apart the chunks' maxima land.

    Returns `(part_max, part_denom, part_acc, whole)`, `whole` being
    `[rows, n_q, 1, d]`.
    """
    generator = torch.Generator().manual_seed(seed)
    splits = len(live)
    part_max = torch.full((rows, n_q, splits), float("-inf"))
    part_denom = torch.zeros(rows, n_q, splits)
    part_acc = torch.zeros(rows, n_q, splits, d)
    whole = torch.zeros(rows, n_q, 1, d)
    for i in range(rows):
        for h in range(n_q):
            scores, values = [], []
            for s, alive in enumerate(live):
                if not alive:
                    continue
                n = 3 + s  # chunks of different lengths, so no two denominators match
                sc = spread * torch.randn(n, generator=generator)
                v = torch.randn(n, d, generator=generator)
                m = sc.max()
                p = torch.exp(sc - m)
                part_max[i, h, s] = m
                part_denom[i, h, s] = p.sum()
                part_acc[i, h, s] = (p[:, None] * v).sum(dim=0)
                scores.append(sc)
                values.append(v)
            weights = torch.softmax(torch.cat(scores), dim=0)
            whole[i, h, 0] = (weights[:, None] * torch.cat(values)).sum(dim=0)
    return (
        part_max.to(device),
        part_denom.to(device),
        part_acc.to(device),
        whole.to(device),
    )


def _keep_the_winner(part_max, part_denom, part_acc):
    """Day 68's mutant: `alpha = (m_s == joint)` in place of `exp(m_s - joint)`.

    Keeps the chunk that holds the row's largest score and drops every other one. It
    is exact whenever one chunk is live, which is the whole trouble with it.
    """
    joint = part_max.amax(dim=2, keepdim=True)
    alpha = (part_max == joint).to(PARTIAL_DTYPE)
    denom = (alpha * part_denom).sum(dim=2)
    acc = (alpha[..., None] * part_acc).sum(dim=2)
    return (acc / denom[..., None])[:, :, None, :]


def _discriminates(fixture):
    """True when the mutant gets this fixture wrong, which is what makes it a test."""
    part_max, part_denom, part_acc, whole = fixture
    wrong = _keep_the_winner(part_max, part_denom, part_acc)
    return not torch.allclose(wrong, whole, atol=1e-2)


# --- the fixtures are held to the mutant before they hold anything else --------------


def test_a_workspace_with_one_live_chunk_is_a_control_the_mutant_survives():
    """Day 68's short rows, rebuilt as tensors: one live chunk and the rest `-inf`.

    The mutant answers it exactly, so a reduce test built on it would pass either
    reduce. It stays here as the control that says so.
    """
    fixture = _workspace([True, False, False, False], seed=1)
    part_max, part_denom, part_acc, whole = fixture
    assert torch.allclose(_keep_the_winner(part_max, part_denom, part_acc), whole, atol=1e-5)
    assert not _discriminates(fixture)


@pytest.mark.parametrize(
    "live",
    [
        [True, True],
        [True, True, True, False],
        [True, False, True, True, False],  # five: the jitted ramp pads it to eight
    ],
)
def test_a_workspace_with_two_live_chunks_is_one_the_mutant_gets_wrong(live):
    assert _discriminates(_workspace(live, seed=2))


# --- the tlsim pass two, by name ------------------------------------------------------


@pytest.mark.parametrize(
    "live",
    [
        [True, True],
        [True, True, True, False],
        [True, False, True, True, False],
        [False, False, True],  # the live chunk is not the first one
    ],
)
def test_the_model_reduce_is_the_softmax_its_partials_were_cut_from(live):
    part_max, part_denom, part_acc, whole = _workspace(live, seed=3)
    got = split_reduce_kernel((part_max, part_denom, part_acc))
    assert got.shape == whole.shape
    assert torch.allclose(got, whole, atol=1e-5)
    assert torch.allclose(got, reduce_partials(part_max, part_denom, part_acc), atol=1e-6)


def test_the_model_reduce_survives_chunks_whose_maxima_are_far_apart():
    """Maxima more than sixty apart. Rescaled against anything but the joint max, one
    chunk's weight overflows or the other's flushes to zero; against `M` both survive."""
    part_max, part_denom, part_acc, whole = _workspace([True, True, True], seed=4, spread=40.0)
    assert float((part_max.amax(dim=2) - part_max.amin(dim=2)).max()) > 60
    got = split_reduce_kernel((part_max, part_denom, part_acc))
    assert torch.isfinite(got).all()
    assert torch.allclose(got, whole, atol=1e-4)


def test_the_model_reduce_writes_the_dtype_it_is_asked_for():
    """The read's output is the pool's dtype, and pass two is where the fp32
    partials turn back into it. The default is the partials' own dtype."""
    part_max, part_denom, part_acc, whole = _workspace([True, True], seed=5)
    assert split_reduce_kernel((part_max, part_denom, part_acc)).dtype == PARTIAL_DTYPE
    half = split_reduce_kernel((part_max, part_denom, part_acc), out_dtype=torch.bfloat16)
    assert half.dtype == torch.bfloat16
    assert torch.allclose(half.float(), whole, atol=3e-2)


def test_the_read_s_pass_two_is_the_function_this_file_tests(monkeypatch):
    """A test of `split_reduce_kernel` is a test of the read only if the read calls
    it. Swapping in the mutant through the module and watching the read go wrong on a
    row that crosses a chunk is how that is checked, rather than assumed."""
    lens = [40, 3]
    torch.manual_seed(6)
    k_pool = torch.randn(64, 1, 8)
    v_pool = torch.randn(64, 1, 8)
    q = torch.randn(2, 2, 1, 8)
    mapping = torch.zeros(2, 40, dtype=torch.long)
    mapping[0] = torch.randperm(64)[:40]
    mapping[1, :3] = torch.randperm(64)[:3]
    context_lens = torch.tensor(lens)
    honest = paged_attention_split_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, block=8, splits=3
    )

    calls = []

    def mutant(partials, out_dtype=None):
        calls.append(tuple(partials[0].shape))
        wrong = _keep_the_winner(*partials)
        return wrong.to(out_dtype or PARTIAL_DTYPE)

    monkeypatch.setattr(flash_decoding, "split_reduce_kernel", mutant)
    broken = paged_attention_split_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, block=8, splits=3
    )
    assert calls == [(2, 2, 3)]
    assert not torch.allclose(broken[0], honest[0], atol=1e-3)  # 40 keys, three chunks
    assert torch.allclose(broken[1], honest[1], atol=1e-6)  # 3 keys, one live chunk


# --- the gate is the workspace gate ----------------------------------------------------


def test_the_reduce_refuses_a_workspace_narrowed_on_the_split_axis():
    """Pass two addresses the partials flat, like pass one, so the Day 64 window that
    keeps the allocated stride is the same wrong answer here."""
    part_max, part_denom, part_acc, _ = _workspace([True, True, True, True], seed=7)
    narrowed = (part_max[:, :, :2], part_denom[:, :, :2], part_acc[:, :, :2])
    with pytest.raises(SplitUnsound, match="not contiguous"):
        split_reduce_kernel(narrowed)


def test_the_reduce_refuses_partials_that_disagree_about_the_shape():
    part_max, part_denom, part_acc, _ = _workspace([True, True], seed=8)
    with pytest.raises(SplitUnsound, match="part_denom has shape"):
        split_reduce_kernel((part_max, part_denom[:1].contiguous(), part_acc))


def test_the_reduce_refuses_partials_in_the_pool_s_dtype():
    part_max, part_denom, part_acc, _ = _workspace([True, True], seed=9)
    with pytest.raises(SplitUnsound, match="a partial is"):
        split_reduce_kernel((part_max, part_denom, part_acc.to(torch.bfloat16)))


def test_the_jitted_reduce_refuses_host_tensors_rather_than_crashing():
    part_max, part_denom, part_acc, _ = _workspace([True, True], seed=10)
    with pytest.raises((RuntimeError, ValueError), match="cuda|CUDA|Triton|triton"):
        split_reduce_triton((part_max, part_denom, part_acc))


def test_the_dispatch_folds_host_partials_with_the_model():
    part_max, part_denom, part_acc, whole = _workspace([True, False, True], seed=11)
    got = split_reduce((part_max, part_denom, part_acc))
    assert torch.allclose(got, whole, atol=1e-5)


# --- the device half, gated -----------------------------------------------------------


@requires_triton_gpu
@pytest.mark.parametrize(
    "live",
    [
        [True, True],
        [True, True, True, False],
        [True, False, True, True, False],
        [False, False, True],
    ],
)
def test_the_jitted_reduce_is_reduce_partials_on_every_live_pattern(live):
    """The test Day 68's mutant needed: `_split_reduce_fwd` alone, on partials where
    more than one chunk is live, so `keep the winner` cannot pass it."""
    fixture = _workspace(live, seed=12, device="cuda")
    part_max, part_denom, part_acc, whole = fixture
    if sum(live) > 1:
        assert _discriminates(fixture)
    got = split_reduce_triton((part_max, part_denom, part_acc))
    assert torch.allclose(got, reduce_partials(part_max, part_denom, part_acc), atol=1e-5)
    assert torch.allclose(got, whole, atol=1e-5)


@requires_triton_gpu
def test_the_jitted_reduce_survives_chunks_whose_maxima_are_far_apart():
    part_max, part_denom, part_acc, whole = _workspace(
        [True, True, True], seed=13, spread=40.0, device="cuda"
    )
    got = split_reduce_triton((part_max, part_denom, part_acc))
    assert torch.isfinite(got).all()
    assert torch.allclose(got, whole, atol=1e-4)


@requires_triton_gpu
def test_the_jitted_reduce_writes_the_pool_s_dtype():
    part_max, part_denom, part_acc, whole = _workspace([True, True], seed=14, device="cuda")
    got = split_reduce_triton((part_max, part_denom, part_acc), out_dtype=torch.bfloat16)
    assert got.dtype == torch.bfloat16
    assert torch.allclose(got.float(), whole, atol=3e-2)
