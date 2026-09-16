"""Day 59 tests: the batched decode read as a streaming kernel, not a rectangle.

Day 28 wrote `paged_attention_batched_reference`: the decode read for a whole
batch, one new token per row, each row over its own scattered slots. It is the
oracle and it is honest about being the slow path. It gathers the entire
`[batch, max_ctx]` mapping into K/V before it masks, and it scores that gather
into a `[batch, heads, 1, max_ctx]` rectangle, which is the tensor Day 54's
`workspace_bytes` prices and the single largest live intermediate a captured
decode region holds.

`paged_attention_batched_kernel` is the same read with the rectangle removed. It
is the Day-22 single-sequence streaming loop one axis wider, expressed on the same
tlsim primitives, with a 2-D launch grid over `(row, query head)` because that is
the grid vLLM's paged kernel launches. Each program owns one row's one head, walks
*that row's own* context a `block` of keys at a time, and folds each tile into an
online softmax. Peak state per program is one `[block, d]` tile plus a running max,
a denominator and a `d`-wide accumulator, and none of those has `max_ctx` in it.

Three properties separate it from the oracle and each gets its own section here:

* **It agrees.** Row for row, tile size for tile size, over ragged lengths and
  physically scattered blocks, to a few ulps (streaming reassociates the exponent
  sums the way every flash kernel does).
* **It never reads the padding.** The oracle gathers a padded entry and then kills
  it with a mask, so the pad slot must hold a *legal* index and its contents are
  read. The kernel masks the load, so the pad may be -1 and may hold NaN, and both
  are tested, because that difference is what "the read is ragged" actually means.
* **It does not pay for another row's history.** `streamed_work` counts the tiles
  walked against the tiles a rectangle of the same width implies, and the gap is
  the batch's length spread. A row of 4 tokens next to a row of 2,000 costs one
  tile here and 2,000 columns of rectangle there.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.kernels.paged_attention import (
    StreamedWork,
    paged_attention_batched_kernel,
    paged_attention_batched_reference,
    paged_attention_reference,
    streamed_work,
)

N_KV, D = 2, 4
N_REP = 4
N_Q = N_KV * N_REP


def _pools(num_slots: int, seed: int = 0, n_kv: int = N_KV, d: int = D):
    """A layer's flat physical K/V pools filled with noise."""
    torch.manual_seed(seed)
    return torch.randn(num_slots, n_kv, d), torch.randn(num_slots, n_kv, d)


def _q(batch: int, seed: int = 1, n_q: int = N_Q, d: int = D):
    """This step's rotated queries: one new token per row."""
    torch.manual_seed(seed)
    return torch.randn(batch, n_q, 1, d)


def _mapping(rows: list[list[int]], width: int | None = None, pad: int = 0):
    """Pack per-row slot lists into the `[batch, max_ctx]` rectangle plus lengths.

    `width` wider than the longest row is a bucketed mapping (Day 52): the rectangle
    the oracle pays for is the bucket, not the batch's longest history.
    """
    lens = [len(r) for r in rows]
    w = width if width is not None else max(lens)
    packed = [r + [pad] * (w - len(r)) for r in rows]
    return (
        torch.tensor(packed, dtype=torch.long),
        torch.tensor(lens, dtype=torch.long),
    )


def _both(rows, *, width=None, block=8, seed=0, n_rep=N_REP, pad=0, num_slots=None):
    """Run the oracle and the kernel on the same inputs; return `(kernel, oracle)`."""
    slots = max(max(r) for r in rows) + 1
    k_pool, v_pool = _pools(num_slots or slots + 1, seed=seed)
    mapping, lens = _mapping(rows, width=width, pad=pad)
    q = _q(len(rows), seed=seed + 1)
    out = paged_attention_batched_kernel(
        q, k_pool, v_pool, mapping, lens, n_rep, block=block
    )
    ref = paged_attention_batched_reference(q, k_pool, v_pool, mapping, lens, n_rep)
    return out, ref


# --- it agrees with the oracle -------------------------------------------------


def test_kernel_matches_the_oracle_on_a_uniform_batch():
    """Every row the same length: the plainest decode step there is."""
    out, ref = _both([[3, 1, 4], [2, 5, 0]])
    assert tuple(out.shape) == tuple(ref.shape) == (2, N_Q, 1, D)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_matches_the_oracle_on_ragged_lengths():
    """The shape a real batch is in: four rows, four different histories."""
    out, ref = _both([[1], [7, 3], [0, 4, 6, 2, 9], [5, 8, 11]])
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_matches_the_oracle_over_scattered_slots():
    """Rows interleave in the shared pool; only the mapping keeps them apart."""
    out, ref = _both([[13, 2, 9, 0], [1, 14, 6], [8, 4, 11, 3, 15]], block=4)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_matches_the_oracle_on_one_row():
    """Batch of one is still the batched read, not the single-sequence one."""
    out, ref = _both([[4, 0, 2, 7, 1, 6]], block=4)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_matches_the_oracle_on_a_long_history():
    """Many tiles per row, including a ragged last one."""
    rows = [list(range(37)), list(range(40, 90)), list(range(100, 111))]
    out, ref = _both(rows, block=16, num_slots=128)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_matches_the_oracle_without_gqa():
    """n_rep=1: every query head has its own KV head, no head expansion."""
    k_pool, v_pool = _pools(12)
    mapping, lens = _mapping([[3, 1, 4, 9], [2, 5]])
    q = _q(2, n_q=N_KV)
    out = paged_attention_batched_kernel(q, k_pool, v_pool, mapping, lens, 1, block=4)
    ref = paged_attention_batched_reference(q, k_pool, v_pool, mapping, lens, 1)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_matches_the_oracle_under_a_bucketed_width():
    """A mapping padded out to a bucket (Day 52) is the oracle's bill, not the kernel's."""
    out, ref = _both([[3, 1, 4], [9, 2]], width=16, block=8)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_honours_the_scale():
    """An explicit softmax scale goes through both paths identically."""
    k_pool, v_pool = _pools(10)
    mapping, lens = _mapping([[1, 4, 7], [2, 9]])
    q = _q(2)
    out = paged_attention_batched_kernel(
        q, k_pool, v_pool, mapping, lens, N_REP, scale=0.37, block=2
    )
    ref = paged_attention_batched_reference(
        q, k_pool, v_pool, mapping, lens, N_REP, scale=0.37
    )
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_each_row_equals_the_single_sequence_reference_on_its_own_slots():
    """Batching is a throughput change and never a behaviour change, one axis in."""
    rows = [[9, 2, 5, 11], [0, 7], [3, 12, 1, 6, 8]]
    k_pool, v_pool = _pools(14)
    mapping, lens = _mapping(rows)
    q = _q(3)

    out = paged_attention_batched_kernel(q, k_pool, v_pool, mapping, lens, N_REP, block=4)

    for row, slots in enumerate(rows):
        alone = paged_attention_reference(
            q[row : row + 1], k_pool, v_pool, torch.tensor(slots, dtype=torch.long), N_REP
        )
        torch.testing.assert_close(out[row : row + 1], alone, atol=1e-5, rtol=1e-4)


def test_the_output_dtype_follows_the_query():
    """The accumulators are fp32; what comes back is the dtype the model is in."""
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 3], [5, 2, 6]])
    q = _q(2)
    out = paged_attention_batched_kernel(q, k_pool, v_pool, mapping, lens, N_REP)
    assert out.dtype == q.dtype == torch.float32


# --- the block is a performance knob, never a correctness one ------------------


@pytest.mark.parametrize("block", [1, 2, 3, 5, 8, 64])
def test_any_block_size_returns_the_same_attention(block):
    """Including a block larger than the whole history, and one that never divides it."""
    rows = [[2, 9, 4, 0, 13], [7, 1], [11, 5, 8]]
    out, ref = _both(rows, block=block)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_block_sizes_agree_with_each_other_to_the_ulp_bar():
    """Two tilings of the same row reassociate differently and must still agree."""
    rows = [list(range(23)), list(range(30, 47))]
    small, _ = _both(rows, block=2, num_slots=64)
    large, _ = _both(rows, block=32, num_slots=64)
    torch.testing.assert_close(small, large, atol=1e-5, rtol=1e-4)


def test_a_block_must_hold_at_least_one_key():
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="at least one key"):
        paged_attention_batched_kernel(
            _q(2), k_pool, v_pool, mapping, lens, N_REP, block=0
        )


# --- it never reads the padding ------------------------------------------------


def test_padding_may_be_an_illegal_index():
    """The difference the mask makes, stated as the thing the oracle cannot do.

    The reference gathers the whole rectangle before it masks, so `slot_mapping`
    has to be padded with a legal slot: a -1 would wrap onto the last slot of the
    pool rather than erroring. A masked load never indexes the buffer on a false
    lane, so the kernel takes -1 and returns the same attention.
    """
    k_pool, v_pool = _pools(10)
    legal, lens = _mapping([[1, 4, 7], [2, 9]], pad=0)
    illegal, _ = _mapping([[1, 4, 7], [2, 9]], pad=-1)
    q = _q(2)

    out = paged_attention_batched_kernel(q, k_pool, v_pool, illegal, lens, N_REP, block=2)
    ref = paged_attention_batched_reference(q, k_pool, v_pool, legal, lens, N_REP)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_a_poisoned_pad_slot_is_never_loaded():
    """NaN in the padding is the sharpest form of "that load did not happen".

    A masked-off gather still returns `other` for the lane, so a value that would
    contaminate any arithmetic it touched proves the tile was skipped rather than
    scored and discarded. The oracle cannot pass this test: it reads the slot,
    multiplies it, and NaN survives `masked_fill`.
    """
    k_pool, v_pool = _pools(10)
    mapping, lens = _mapping([[1, 4, 7], [2, 9]], width=6, pad=5)
    q = _q(2)
    clean = paged_attention_batched_kernel(q, k_pool, v_pool, mapping, lens, N_REP, block=4)

    k_pool[5] = float("nan")
    v_pool[5] = float("nan")
    poisoned = paged_attention_batched_kernel(
        q, k_pool, v_pool, mapping, lens, N_REP, block=4
    )

    assert torch.isfinite(poisoned).all()
    torch.testing.assert_close(clean, poisoned)


def test_a_huge_pad_value_cannot_dominate_the_softmax():
    """The Day-28 test of the oracle's mask, asked of the kernel's mask instead."""
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 3, 4], [2, 7, 7]], pad=7)
    lens = torch.tensor([3, 1], dtype=torch.long)
    q = _q(2)
    before = paged_attention_batched_kernel(q, k_pool, v_pool, mapping, lens, N_REP, block=2)
    k_pool[7] = 1e4
    v_pool[7] = 1e4
    after = paged_attention_batched_kernel(q, k_pool, v_pool, mapping, lens, N_REP, block=2)
    torch.testing.assert_close(before, after)


def test_a_row_does_not_see_another_rows_history():
    """Lengthening row 1 leaves row 0's output bit for bit where it was."""
    k_pool, v_pool = _pools(64, seed=3)
    q = _q(2)
    short, short_lens = _mapping([[1, 4, 7], [9, 2]], width=40)
    long_rows, long_lens = _mapping([[1, 4, 7], list(range(20, 60))], width=40)

    a = paged_attention_batched_kernel(q, k_pool, v_pool, short, short_lens, N_REP, block=8)
    b = paged_attention_batched_kernel(
        q, k_pool, v_pool, long_rows, long_lens, N_REP, block=8
    )

    torch.testing.assert_close(a[0], b[0])


# --- the refusals, which are the oracle's refusals ------------------------------


def test_the_batched_kernel_is_the_decode_read():
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="decode"):
        paged_attention_batched_kernel(
            torch.randn(2, N_Q, 2, D), k_pool, v_pool, mapping, lens, N_REP
        )


def test_q_must_be_four_dimensional():
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match=r"\[batch, n_q, 1, d\]"):
        paged_attention_batched_kernel(
            torch.randn(2, N_Q, D), k_pool, v_pool, mapping, lens, N_REP
        )


def test_the_two_pools_must_match():
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="same shape"):
        paged_attention_batched_kernel(
            _q(2), k_pool, v_pool[:4], mapping, lens, N_REP
        )


def test_the_mapping_must_be_a_rectangle():
    k_pool, v_pool = _pools(8)
    _, lens = _mapping([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match=r"\[batch, max_ctx\]"):
        paged_attention_batched_kernel(
            _q(2), k_pool, v_pool, torch.tensor([1, 2, 3]), lens, N_REP
        )


def test_every_row_needs_a_mapping_and_a_length():
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="one row each"):
        paged_attention_batched_kernel(
            _q(3), k_pool, v_pool, mapping, lens, N_REP
        )


def test_a_row_with_no_visible_key_is_refused():
    """A query that softmaxes over nothing is 0/0, and here it is also zero tiles."""
    k_pool, v_pool = _pools(8)
    mapping, _ = _mapping([[1, 2], [3, 4]])
    lens = torch.tensor([2, 0], dtype=torch.long)
    with pytest.raises(ValueError, match="at least 1"):
        paged_attention_batched_kernel(_q(2), k_pool, v_pool, mapping, lens, N_REP)


def test_a_length_past_the_mapping_is_refused():
    k_pool, v_pool = _pools(8)
    mapping, _ = _mapping([[1, 2], [3, 4]])
    lens = torch.tensor([2, 5], dtype=torch.long)
    with pytest.raises(ValueError, match="more history than the mapping holds"):
        paged_attention_batched_kernel(_q(2), k_pool, v_pool, mapping, lens, N_REP)


def test_bounds_and_validated_are_the_same_claim():
    """Day 50's refusal, inherited: only one of the two can be the reason."""
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="not both"):
        paged_attention_batched_kernel(
            _q(2), k_pool, v_pool, mapping, lens, N_REP,
            context_bounds=(2, 2), validated=True,
        )


def test_given_bounds_are_trusted_and_the_lengths_are_not_read():
    """The Day-49 contract: two ints in, no readback, and the claim is the caller's."""
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 2], [3, 4]])
    out = paged_attention_batched_kernel(
        _q(2), k_pool, v_pool, mapping, lens, N_REP, context_bounds=(2, 2)
    )
    ref = paged_attention_batched_reference(
        _q(2), k_pool, v_pool, mapping, lens, N_REP, context_bounds=(2, 2)
    )
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_validated_skips_the_check_the_planned_path_already_did():
    k_pool, v_pool = _pools(8)
    mapping, lens = _mapping([[1, 2], [3, 4]])
    out = paged_attention_batched_kernel(
        _q(2), k_pool, v_pool, mapping, lens, N_REP, validated=True
    )
    assert tuple(out.shape) == (2, N_Q, 1, D)


# --- what it walks, against what a rectangle would ------------------------------


def test_streamed_work_counts_each_rows_own_tiles():
    work = streamed_work([4, 17, 33], block=16)
    assert work.tiles == 1 + 2 + 3
    assert work.rows == 3
    assert work.context_width == 33
    assert work.block == 16


def test_the_rectangle_pays_the_widest_row_for_every_row():
    work = streamed_work([4, 17, 33], block=16)
    assert work.rectangle_tiles == 3 * 3
    assert work.wasted_tiles == 9 - 6


def test_a_uniform_batch_wastes_nothing():
    """No length spread, no saving: the ragged win is the spread and only the spread."""
    work = streamed_work([32, 32, 32], block=16)
    assert work.tiles == work.rectangle_tiles == 6
    assert work.wasted_tiles == 0
    assert work.ragged_saving == pytest.approx(1.0)


def test_the_saving_is_the_ratio_of_the_two():
    work = streamed_work([4, 2048], block=128)
    assert work.tiles == 1 + 16
    assert work.rectangle_tiles == 32
    assert work.ragged_saving == pytest.approx(32 / 17)


def test_a_bucketed_width_is_charged_to_the_rectangle_alone():
    """Day 52 pads the mapping out to a bucket; the kernel never walks the padding."""
    tight = streamed_work([30, 40], block=16)
    bucketed = streamed_work([30, 40], block=16, context_width=128)
    assert bucketed.tiles == tight.tiles == 2 + 3
    assert bucketed.rectangle_tiles == 2 * 8
    assert bucketed.ragged_saving > tight.ragged_saving


def test_streamed_work_accepts_a_tensor_of_lengths():
    """`context_lens` is what the planned path holds, so that is what it takes."""
    work = streamed_work(torch.tensor([4, 17], dtype=torch.long), block=16)
    assert work.tiles == 3


def test_streamed_work_renders_one_line():
    line = streamed_work([4, 2048], block=128).render()
    assert "17" in line and "32" in line


def test_streamed_work_refuses_an_empty_batch():
    with pytest.raises(ValueError, match="at least one row"):
        streamed_work([], block=16)


def test_streamed_work_refuses_a_row_with_no_history():
    with pytest.raises(ValueError, match="at least 1"):
        streamed_work([4, 0], block=16)


def test_streamed_work_refuses_a_width_narrower_than_the_longest_row():
    with pytest.raises(ValueError, match="more history than"):
        streamed_work([4, 40], block=16, context_width=32)


def test_streamed_work_refuses_an_empty_block():
    with pytest.raises(ValueError, match="at least one key"):
        streamed_work([4], block=0)


def test_streamed_work_is_frozen():
    work = streamed_work([4], block=16)
    assert isinstance(work, StreamedWork)
    with pytest.raises(AttributeError):
        work.tiles = 99


def test_the_walked_tiles_match_what_the_kernel_actually_runs():
    """The accounting and the loop are the same number or the accounting is a story."""
    rows = [[1], [2, 3, 4, 5, 6], [7, 8]]
    _, lens = _mapping(rows)
    work = streamed_work(lens, block=2)
    assert work.tiles == 1 + 3 + 1
    out, ref = _both(rows, block=2, num_slots=16)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)
