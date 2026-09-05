"""Day 50 tests: the decode step's addressing, worked out before the forward runs.

Day 49 pointed `torch.compile` at the decode forward and got one graph with no
breaks and a forty-six-fold regression. The breaks were gone and the *guard* was
not, and the guard dynamo kept failing was not on a shape:

    kwargs['cache'].cache.tables[0].num_tokens == 13
      # [table.slot(p) for p in range(start, table.num_tokens)], cache.py:729

`num_tokens` is a plain Python int on a plain Python object, read inside the traced
region, and a tracer specialises on its *value*. It grows by one every decode step,
so the graph was invalidated every step, rebuilt every step, and after eight
rebuilds abandoned to the interpreter for the rest of the process.

This file is the fix and its price. A `DecodePlan` is everything the decode forward
needs to know about *where* things live, computed on the host before the forward is
called and handed in as tensors: this step's write slot per row, the `[rows, ctx]`
read rectangle, the context lengths, and the absolute position of each row's new
token. With one in hand the forward reads no Python attribute of the cache at all,
which is what `test_a_planned_forward_never_reads_a_block_table` pins by making
`BlockTable.num_tokens` raise for the duration of the call.

Four claims, and the last two are the interesting ones:

  1. **A planned decode computes exactly what an unplanned one did.** Moving the
     addressing changes when it is worked out and nothing about what it says. Two
     caches in the same state, one planned and one not, agree to the bit.
  2. **The plan is what grows the tables, so the forward does not.** Growth is a
     host mutation, and a mutation inside a traced region is either a break or a
     lie. It happens once, in `plan_decode`, atomically across the batch.
  3. **A planned forward is a function again, and an unplanned one never was.**
     Day 49's first equality test failed by whole units because calling a cached
     forward twice attends over a history one token longer the second time. With
     the addressing fixed up front, the same plan replayed writes the same slots
     and returns the same logits, which is what makes a compiled forward
     comparable to an eager one at all.
  4. **The plan is not free, and its cost is quadratic.** The read rectangle is
     `[rows, max_ctx]` int64 rebuilt from scratch every step, and `max_ctx` grows
     by one per step, so a 512-step generation rebuilds a quarter of a million
     cells per row to append 512. `rebuild_cells` against `appended_cells` is that
     ratio, and `check_rebuild_bounded` is the gate that says when it stops being
     affordable. The engine that fixes it keeps a persistent device-side table and
     writes one entry per step; this day prices the version that does not.

Two tiers as usual: pure tests on tiny weights, plus the handful that need the real
compiler and say so in their names.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.batch import pad_prompts
from nanoserve.cache import (
    BatchedPagedKVCache,
    BlockAllocator,
    BlockTable,
    KVCacheExhausted,
)
from nanoserve.config import ModelConfig
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.plan import (
    DecodePlan,
    PlanUnsound,
    appended_cells,
    check_plan_addressing,
    check_plan_current,
    check_plan_rows,
    check_rebuild_bounded,
    mapping_bytes,
    mapping_cells,
    plan_decode,
    rebuild_cells,
    rebuild_ratio,
)


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _model(seed: int = 0) -> tuple[LlamaModel, ModelConfig]:
    torch.manual_seed(seed)
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg)), cfg


def _kv(cfg: ModelConfig, batch: int, seq: int, seed: int = 1):
    torch.manual_seed(seed)
    shape = (batch, cfg.num_key_value_heads, seq, cfg.head_dim)
    return torch.randn(*shape), torch.randn(*shape)


def _prefilled(prompts, *, num_blocks=32, block_size=4, batch_size=None, seed=1):
    """A batched cache holding one prefill per row, ready to decode."""
    cfg = _tiny_config()
    batch_size = batch_size if batch_size is not None else len(prompts)
    cache = BatchedPagedKVCache(
        cfg, BlockAllocator(num_blocks=num_blocks, block_size=block_size), batch_size
    )
    batch = pad_prompts(list(prompts), pad_id=0)
    for layer in range(cfg.num_hidden_layers):
        k, v = _kv(cfg, len(prompts), batch.max_length, seed=seed + layer)
        cache.write(layer, k, v, batch.attention_mask, rows=tuple(range(len(prompts))))
    return cfg, cache


# --- what a plan says ------------------------------------------------------------


def test_a_plan_holds_one_write_slot_per_row():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)

    assert plan.rows == (0, 1, 2)
    assert plan.batch_size == 3
    assert tuple(plan.write_slots.shape) == (3,)


def test_the_positions_are_the_lengths_from_before_the_step():
    """The new token's absolute position is the count of tokens that precede it."""
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)

    assert plan.positions.tolist() == [[3], [1], [2]]
    assert tuple(plan.positions.shape) == (3, 1)


def test_the_context_lengths_are_the_lengths_from_after_the_step():
    """The read includes this step's own token: it is written before it is attended."""
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)

    assert plan.context_lens.tolist() == [4, 2, 3]
    assert (plan.min_ctx, plan.max_ctx) == (2, 4)


def test_the_rectangle_is_as_wide_as_the_longest_history():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)

    assert tuple(plan.slot_mapping.shape) == (3, 4)
    assert plan.width == 4


def test_the_write_slot_is_the_rows_last_real_entry_in_the_rectangle():
    """The one invariant tying the two halves together: write here, read it there."""
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)

    for row in range(plan.batch_size):
        last = int(plan.context_lens[row]) - 1
        assert int(plan.slot_mapping[row, last]) == int(plan.write_slots[row])


def test_the_padding_in_a_plans_rectangle_is_slot_zero():
    """Same choice `slot_mapping` made and for the same reason: a legal index."""
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)

    assert plan.slot_mapping[1, 2:].tolist() == [0, 0]


def test_every_real_slot_in_a_plan_is_distinct():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)

    live = [
        int(s)
        for row, n in enumerate(plan.context_lens.tolist())
        for s in plan.slot_mapping[row, :n]
    ]
    assert len(live) == len(set(live)) == 9


def test_a_plan_reports_the_cells_it_built_and_the_ones_that_are_padding():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)

    assert plan.cells == 12  # 3 rows x 4 wide
    assert plan.real_cells == 9
    assert plan.padding_cells == 3
    assert plan.padding_share == pytest.approx(0.25)


def test_a_plan_prices_itself_in_bytes():
    """int64 addressing: the rectangle is eight bytes a cell, rebuilt every step."""
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache)

    assert plan.mapping_bytes == 12 * 8


def test_a_plan_renders_its_shape_for_a_log():
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    text = plan.render()

    assert "2" in text and "4" in text  # rows and width
    assert "rows" in text


def test_a_plan_is_frozen():
    """It is addressing that has already been handed to a forward; nothing edits it."""
    _, cache = _prefilled([[1, 2]])
    plan = plan_decode(cache)

    with pytest.raises(Exception):
        plan.max_ctx = 99


# --- what planning does to the cache ---------------------------------------------


def test_planning_is_what_grows_the_tables():
    _, cache = _prefilled([[1, 2, 3], [4]])
    assert cache.seq_lens == [3, 1]

    plan_decode(cache)

    assert cache.seq_lens == [4, 2]


def test_planning_invalidates_the_caches_own_mapping():
    """The tables moved, so the read addressing the cache was holding is stale."""
    _, cache = _prefilled([[1, 2, 3], [4]])
    cache.slot_mapping()
    assert cache._mapping is not None

    plan_decode(cache)

    assert cache._mapping is None


def test_a_plan_the_pool_cannot_take_leaves_every_table_untouched():
    """Atomic across rows, the same promise `_reserve` makes for a write."""
    _, cache = _prefilled([[1, 2, 3, 4], [5, 6, 7, 8]], num_blocks=2, block_size=4)
    assert cache.seq_lens == [4, 4]
    before = cache.allocator.num_free

    with pytest.raises(KVCacheExhausted):
        plan_decode(cache)

    assert cache.seq_lens == [4, 4]
    assert cache.allocator.num_free == before


def test_planning_a_row_that_holds_nothing_is_refused():
    """A decode over an empty row would attend over its own token and no prompt."""
    _, cache = _prefilled([[1, 2, 3]], batch_size=2)

    with pytest.raises(ValueError, match="prefill"):
        plan_decode(cache, rows=(0, 1))


def test_planning_names_its_rows_and_grows_no_others():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])

    plan = plan_decode(cache, rows=(0, 2))

    assert plan.rows == (0, 2)
    assert cache.seq_lens == [4, 1, 3]
    assert plan.context_lens.tolist() == [4, 3]


def test_planning_over_a_view_plans_the_views_rows():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    view = cache.view((1, 2))

    plan = view.plan_decode()

    assert plan.rows == (1, 2)
    assert cache.seq_lens == [3, 2, 3]


def test_planning_refuses_a_row_that_does_not_exist():
    _, cache = _prefilled([[1, 2, 3]], batch_size=2)

    with pytest.raises(ValueError, match="out of range"):
        plan_decode(cache, rows=(5,))


def test_planning_refuses_a_repeated_row():
    """One row twice would write two tokens into one slot and call it one step."""
    _, cache = _prefilled([[1, 2, 3]])

    with pytest.raises(ValueError, match="distinct"):
        plan_decode(cache, rows=(0, 0))


def test_a_plan_lands_on_the_device_it_was_asked_for():
    _, cache = _prefilled([[1, 2, 3]])
    plan = plan_decode(cache, device=torch.device("cpu"))

    assert plan.slot_mapping.device == torch.device("cpu")
    assert plan.write_slots.dtype == torch.long
    assert plan.slot_mapping.dtype == torch.long
    assert plan.context_lens.dtype == torch.long
    assert plan.positions.dtype == torch.long


# --- the forward with a plan in hand ---------------------------------------------


def test_a_planned_write_scatters_this_steps_kv_to_the_plans_slots():
    cfg, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    k, v = _kv(cfg, 2, 1, seed=7)

    cache.write(0, k, v, plan=plan)

    for row in range(2):
        slot = int(plan.write_slots[row])
        torch.testing.assert_close(cache.k_pool[0][slot], k[row, :, 0, :])
        torch.testing.assert_close(cache.v_pool[0][slot], v[row, :, 0, :])


def test_a_planned_write_does_not_grow_the_tables():
    """The plan already did, once, for the whole step. A second growth is a bug."""
    cfg, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    assert cache.seq_lens == [4, 2]
    k, v = _kv(cfg, 2, 1, seed=7)

    cache.write(0, k, v, plan=plan)
    cache.write(1, k, v, plan=plan)

    assert cache.seq_lens == [4, 2]


def test_a_planned_write_needs_no_layer_zero_first():
    """The layer-0 handshake existed because layer 0 was what grew the tables."""
    cfg, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    k, v = _kv(cfg, 2, 1, seed=7)

    cache.write(1, k, v, plan=plan)  # no raise: nothing to hand shake about

    slot = int(plan.write_slots[0])
    torch.testing.assert_close(cache.k_pool[1][slot], k[0, :, 0, :])


def test_a_planned_write_refuses_a_plan_for_other_rows():
    cfg, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache, rows=(0, 1))
    k, v = _kv(cfg, 2, 1, seed=7)

    with pytest.raises(ValueError, match="rows"):
        cache.write(0, k, v, rows=(1, 2), plan=plan)


def test_a_planned_write_refuses_a_batch_that_is_not_one_token_per_row():
    cfg, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    k, v = _kv(cfg, 2, 3, seed=7)

    with pytest.raises(ValueError, match="one token"):
        cache.write(0, k, v, plan=plan)


def test_a_planned_slot_mapping_is_the_plans_own_tensors():
    """Not a copy and not a rebuild: the read is handed what the host already built."""
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)

    slots, lens = cache.slot_mapping(plan=plan)

    assert slots is plan.slot_mapping
    assert lens is plan.context_lens


def test_a_planned_decode_read_equals_an_unplanned_one():
    """The oracle. Two caches in the same state, one planned, one on the old path."""
    cfg, planned = _prefilled([[1, 2, 3], [4], [5, 6]])
    _, eager = _prefilled([[1, 2, 3], [4], [5, 6]])
    k, v = _kv(cfg, 3, 1, seed=9)
    q = torch.randn(3, cfg.num_attention_heads, 1, cfg.head_dim)

    plan = plan_decode(planned)
    got = planned.paged_attention(0, k, v, q, cfg.num_kv_groups, plan=plan)
    want = eager.paged_attention(0, k, v, q, cfg.num_kv_groups)

    torch.testing.assert_close(got, want)
    assert planned.seq_lens == eager.seq_lens


def test_a_planned_model_forward_equals_an_unplanned_one():
    model, cfg = _model()
    _, planned = _prefilled([[1, 2, 3], [4]])
    _, eager = _prefilled([[1, 2, 3], [4]])
    ids = torch.tensor([[7], [8]], dtype=torch.long)

    plan = plan_decode(planned)
    got = model.forward(ids, plan.positions, cache=planned.view((0, 1), plan=plan))
    want = model.forward(
        ids, torch.tensor([[3], [1]], dtype=torch.long), cache=eager.view((0, 1))
    )

    torch.testing.assert_close(got, want)


def test_a_planned_forward_replayed_over_one_plan_returns_the_same_logits():
    """The property that makes a compiled forward comparable to an eager one.

    Day 49's first equality test called both and compared, and it failed by whole
    units: a forward over a KV cache is not a function, because the paged read
    writes this step's K/V into the row before it attends, so the second call sees
    a history one token longer. With the addressing fixed before the call, the
    replay writes the same slots and reads the same rectangle.
    """
    model, cfg = _model()
    _, cache = _prefilled([[1, 2, 3], [4]])
    ids = torch.tensor([[7], [8]], dtype=torch.long)
    plan = plan_decode(cache)
    view = cache.view((0, 1), plan=plan)

    first = model.forward(ids, plan.positions, cache=view)
    second = model.forward(ids, plan.positions, cache=view)

    torch.testing.assert_close(first, second)


def test_an_unplanned_forward_replayed_does_not_return_the_same_logits():
    """The control, and the reason claim 3 is worth a test rather than a comment."""
    model, cfg = _model()
    _, cache = _prefilled([[1, 2, 3], [4]])
    ids = torch.tensor([[7], [8]], dtype=torch.long)
    positions = torch.tensor([[3], [1]], dtype=torch.long)
    view = cache.view((0, 1))

    first = model.forward(ids, positions, cache=view)
    second = model.forward(ids, positions, cache=view)

    assert not torch.allclose(first, second)


def test_a_planned_forward_never_reads_a_block_table():
    """The day's central claim, made falsifiable.

    `BlockTable.num_tokens` becomes a property that raises for the duration of the
    call. A data descriptor on the class wins over the instance attribute, so every
    read of it anywhere under the forward is an exception. This is the whole point
    of a plan: the traced region touches tensors and nothing else.
    """
    model, cfg = _model()
    _, cache = _prefilled([[1, 2, 3], [4]])
    ids = torch.tensor([[7], [8]], dtype=torch.long)
    plan = plan_decode(cache)
    view = cache.view((0, 1), plan=plan)

    with _no_table_reads():
        model.forward(ids, plan.positions, cache=view)


def test_an_unplanned_forward_does_read_a_block_table():
    """The control. Without a plan the addressing is built inside the forward."""
    model, cfg = _model()
    _, cache = _prefilled([[1, 2, 3], [4]])
    ids = torch.tensor([[7], [8]], dtype=torch.long)
    positions = torch.tensor([[3], [1]], dtype=torch.long)
    view = cache.view((0, 1))

    with pytest.raises(AssertionError, match="num_tokens"):
        with _no_table_reads():
            model.forward(ids, positions, cache=view)


class _no_table_reads:
    """Make every read of `BlockTable.num_tokens` raise, for one block."""

    def __enter__(self):
        def _boom(_self):
            raise AssertionError("num_tokens was read inside the forward")

        self._saved = BlockTable.__dict__.get("num_tokens")
        BlockTable.num_tokens = property(_boom)
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            del BlockTable.num_tokens
        else:
            BlockTable.num_tokens = self._saved
        return False


# --- the kernel's side of the trade ----------------------------------------------


def test_the_kernel_can_be_told_its_bounds_were_already_checked():
    """`validated=True` and not `context_bounds=(lo, hi)`, which is the whole point.

    Day 49 moved the two ints out of the kernel and handed them down instead. Passing
    them *in* is the same specialisation wearing a different hat: they are Python
    ints that change every step, so a tracer guards on their values and rebuilds. A
    bool that is True for the whole run is a guard that holds for the whole run.
    """
    cfg, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    k, v = _kv(cfg, 2, 1, seed=9)
    cache.write(0, k, v, plan=plan)
    q = torch.randn(2, cfg.num_attention_heads, 1, cfg.head_dim)

    got = paged_attention_batched_reference(
        q,
        cache.k_pool[0],
        cache.v_pool[0],
        plan.slot_mapping,
        plan.context_lens,
        cfg.num_kv_groups,
        validated=True,
    )
    want = paged_attention_batched_reference(
        q,
        cache.k_pool[0],
        cache.v_pool[0],
        plan.slot_mapping,
        plan.context_lens,
        cfg.num_kv_groups,
        context_bounds=(plan.min_ctx, plan.max_ctx),
    )
    torch.testing.assert_close(got, want)


def test_a_validated_read_is_trusted_and_does_not_check():
    """The honest reading of `validated`: it buys speed by believing the caller."""
    cfg, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    k, v = _kv(cfg, 2, 1, seed=9)
    cache.write(0, k, v, plan=plan)
    q = torch.randn(2, cfg.num_attention_heads, 1, cfg.head_dim)
    lying = torch.tensor([0, 2], dtype=torch.long)  # row 0 claims no history at all

    paged_attention_batched_reference(
        q, cache.k_pool[0], cache.v_pool[0], plan.slot_mapping, lying,
        cfg.num_kv_groups, validated=True,
    )

    with pytest.raises(ValueError, match="at least 1"):
        paged_attention_batched_reference(
            q, cache.k_pool[0], cache.v_pool[0], plan.slot_mapping, lying,
            cfg.num_kv_groups,
        )


def test_a_read_cannot_be_both_validated_and_handed_bounds():
    cfg, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    k, v = _kv(cfg, 2, 1, seed=9)
    cache.write(0, k, v, plan=plan)
    q = torch.randn(2, cfg.num_attention_heads, 1, cfg.head_dim)

    with pytest.raises(ValueError, match="validated"):
        paged_attention_batched_reference(
            q, cache.k_pool[0], cache.v_pool[0], plan.slot_mapping, plan.context_lens,
            cfg.num_kv_groups, context_bounds=(1, 4), validated=True,
        )


# --- the gates -------------------------------------------------------------------


def test_plan_unsound_is_an_assertion_error():
    """Same choice `CompileUnsound` made: a gate failing is a failed assertion."""
    assert issubclass(PlanUnsound, AssertionError)


def test_check_plan_addressing_passes_on_a_real_plan():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    check_plan_addressing(plan_decode(cache))


def test_check_plan_addressing_catches_a_write_slot_that_is_not_in_the_rectangle():
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    broken = DecodePlan(
        rows=plan.rows,
        positions=plan.positions,
        write_slots=torch.tensor([99, 99], dtype=torch.long),
        slot_mapping=plan.slot_mapping,
        context_lens=plan.context_lens,
        min_ctx=plan.min_ctx,
        max_ctx=plan.max_ctx,
    )

    with pytest.raises(PlanUnsound, match="write slot"):
        check_plan_addressing(broken)


def test_check_plan_addressing_catches_two_rows_pointed_at_one_slot():
    """The invariant a shared pool lives on, checked on the addressing not the pool."""
    _, cache = _prefilled([[1, 2], [3, 4]])
    plan = plan_decode(cache)
    mapping = plan.slot_mapping.clone()
    mapping[1] = mapping[0]
    broken = DecodePlan(
        rows=plan.rows,
        positions=plan.positions,
        write_slots=mapping[:, -1].clone(),
        slot_mapping=mapping,
        context_lens=plan.context_lens,
        min_ctx=plan.min_ctx,
        max_ctx=plan.max_ctx,
    )

    with pytest.raises(PlanUnsound, match="share"):
        check_plan_addressing(broken)


def test_check_plan_rows_accepts_the_rows_it_was_built_for():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache, rows=(0, 2))
    check_plan_rows(plan, (0, 2))


def test_check_plan_rows_refuses_a_forward_over_different_rows():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]])
    plan = plan_decode(cache, rows=(0, 2))

    with pytest.raises(PlanUnsound, match="rows"):
        check_plan_rows(plan, (0, 1))


def test_check_plan_current_accepts_a_plan_nothing_has_moved_under():
    _, cache = _prefilled([[1, 2, 3], [4]])
    check_plan_current(plan_decode(cache), cache)


def test_check_plan_current_catches_a_plan_the_tables_have_outgrown():
    """A plan is addressing for the state it was built in. Plan twice, use the first,
    and the forward writes where the second step's token belongs."""
    _, cache = _prefilled([[1, 2, 3], [4]])
    stale = plan_decode(cache)
    plan_decode(cache)

    with pytest.raises(PlanUnsound, match="stale"):
        check_plan_current(stale, cache)


# --- what the plan costs ---------------------------------------------------------


def test_the_rectangle_is_rows_times_width_cells():
    assert mapping_cells(4, 200) == 800
    assert mapping_bytes(4, 200) == 6400
    assert mapping_bytes(4, 200, itemsize=4) == 3200


def test_rebuilding_the_rectangle_every_step_is_quadratic_in_the_steps():
    """The price of this design, and the reason it is not the last word.

    Step i rebuilds `rows * (start + i)` cells, so N steps is O(N^2) host work to
    append N tokens.
    """
    assert rebuild_cells(steps=3, rows=2, start_ctx=10) == 2 * (11 + 12 + 13)
    assert rebuild_cells(steps=0, rows=2, start_ctx=10) == 0


def test_appending_one_entry_a_step_is_linear():
    """What a persistent device-side table would touch: one cell per row per step."""
    assert appended_cells(steps=3, rows=2) == 6
    assert appended_cells(steps=512, rows=4) == 2048


def test_the_ratio_between_them_grows_with_the_run():
    short = rebuild_ratio(steps=8, rows=4, start_ctx=16)
    long = rebuild_ratio(steps=512, rows=4, start_ctx=16)

    assert long > short > 1.0
    assert long == pytest.approx(rebuild_cells(512, 4, 16) / appended_cells(512, 4))


def test_check_rebuild_bounded_passes_a_short_run():
    check_rebuild_bounded(steps=8, rows=4, start_ctx=16, limit=32.0)


def test_check_rebuild_bounded_refuses_a_run_that_rebuilds_far_more_than_it_appends():
    with pytest.raises(PlanUnsound, match="rebuild"):
        check_rebuild_bounded(steps=512, rows=4, start_ctx=16, limit=32.0)


def test_the_cost_arithmetic_refuses_nonsense():
    with pytest.raises(ValueError):
        mapping_cells(0, 10)
    with pytest.raises(ValueError):
        rebuild_cells(steps=-1, rows=2, start_ctx=4)
    with pytest.raises(ValueError):
        appended_cells(steps=2, rows=0)
    with pytest.raises(ValueError):
        check_rebuild_bounded(steps=2, rows=2, start_ctx=4, limit=0.0)


# --- the engine ------------------------------------------------------------------


def test_the_engine_plans_its_decode_step_before_it_calls_the_forward():
    from nanoserve.engine import Engine
    from nanoserve.scheduler import Request

    model, cfg = _model()
    engine = Engine.build(model, num_blocks=64, block_size=4, max_batch_size=2)
    engine.add_request(Request(request_id="a", prompt_token_ids=[1, 2, 3], max_new_tokens=3))
    engine.step()  # prefill

    seen = {}
    original = engine.model.forward

    def spy(input_ids, positions=None, **kwargs):
        seen["plan"] = getattr(kwargs.get("cache"), "plan", None)
        seen["positions"] = positions
        return original(input_ids, positions, **kwargs)

    engine.model.forward = spy
    engine.step()  # decode

    assert isinstance(seen["plan"], DecodePlan)
    assert seen["positions"] is seen["plan"].positions


def test_an_engine_run_still_generates_what_it_generated_before_the_plan():
    """The regression that matters: the addressing moved, the tokens did not."""
    from nanoserve.engine import Engine

    model, cfg = _model()
    prompts = [[1, 2, 3], [4, 5]]
    want = model.greedy_generate_batch(prompts, max_new_tokens=4, block_size=4)

    engine = Engine.build(model, num_blocks=64, block_size=4, max_batch_size=2)

    assert engine.generate(prompts, max_new_tokens=4) == want


# --- the shape a compiler is shown -------------------------------------------------


def test_the_shape_probe_reads_a_plans_width_rather_than_predicting_it():
    """Day 49's probe added one to `seq_lens`, and a plan makes that one too many.

    `decode_shape` predicted the width as "the longest history plus this step's
    token", because the read built its rectangle after the write and the probe ran
    before either. With a plan the tables have *already* grown by the time the
    wrapper sees the call, so the same arithmetic overshoots by one. The plan is not
    a prediction: it is holding the rectangle, and `max_ctx` is its width.
    """
    from nanoserve.compiled import decode_shape

    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)
    view = cache.view((0, 1), plan=plan)

    shape = decode_shape(torch.tensor([[7], [8]], dtype=torch.long), view)

    assert plan.max_ctx == 4
    assert shape.context_width == 4


def test_the_shape_probe_still_predicts_when_there_is_no_plan():
    """The control: unplanned, the write happens inside the forward, so add one."""
    from nanoserve.compiled import decode_shape

    _, cache = _prefilled([[1, 2, 3], [4]])
    view = cache.view((0, 1))

    shape = decode_shape(torch.tensor([[7], [8]], dtype=torch.long), view)

    assert shape.context_width == 4  # longest history 3, plus this step's token


# --- the compiler, for real --------------------------------------------------------


def _compile_engine(mode: str, *, prompts, max_new_tokens):
    """One tiny engine driven to completion under a real `torch.compile`."""
    import torch._dynamo

    from nanoserve.compiled import dynamo_frames_compiled
    from nanoserve.engine import Engine
    from nanoserve.scheduler import Request

    model, _ = _model()
    torch._dynamo.reset()  # Day 49's lesson: the counter is per process, not per run
    before = dynamo_frames_compiled()
    engine = Engine.build(
        model, num_blocks=128, block_size=4, max_batch_size=len(prompts),
        compile_decode=mode,
    )
    for i, prompt in enumerate(prompts):
        engine.add_request(
            Request(
                request_id=f"r{i}", prompt_token_ids=list(prompt),
                max_new_tokens=max_new_tokens,
            )
        )
    engine.run_to_completion()
    return engine, dynamo_frames_compiled() - before


def test_a_planned_decode_reuses_its_graph_where_an_unplanned_one_rebuilt_it():
    """The day's payoff, measured against Day 49's own gate.

    Day 49 ran 9 decode steps in `dynamic` mode and dynamo built 8 graphs, because
    the guard that kept failing was `tables[0].num_tokens == 13` and a symbolic
    tensor dimension has nothing to say about a Python int. With the addressing
    handed in as tensors the guard that fails is a *size*, which is the thing
    `dynamic=True` exists for, so the graph is built and then kept.
    """
    from nanoserve.compiled import check_graph_reused

    engine, builds = _compile_engine("dynamic", prompts=[[1, 2, 3], [4, 5]], max_new_tokens=14)

    assert engine.decode_forward.calls >= 12
    assert builds <= 2
    check_graph_reused(engine.decode_forward.calls, builds)


def test_a_planned_decode_still_traces_to_one_graph_with_no_breaks():
    """Day 49's other gate, which the plan had every chance to undo and did not."""
    from nanoserve.compiled import check_single_graph, explain_forward

    model, _ = _model()
    _, cache = _prefilled([[1, 2, 3], [4]])
    plan = plan_decode(cache)

    report = explain_forward(
        model.forward,
        torch.tensor([[7], [8]], dtype=torch.long),
        plan.positions,
        cache=cache.view((0, 1), plan=plan),
    )

    check_single_graph(report)
    assert (report.graphs, report.breaks) == (1, 0)
