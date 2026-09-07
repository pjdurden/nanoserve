"""Day 52 tests: the decode step's shape closed into a small set.

Day 50 moved the addressing out of the forward so a tracer guards on tensor shapes
instead of on `table.num_tokens`. Day 51 made the rectangle a window on a
persistent buffer, so the address stopped moving. What is still moving is the
*shape*: `[rows, max_ctx]` with `max_ctx` growing by one every step and `rows`
being whatever the scheduler admitted, so a run presents one new shape per step and
`static` mode walks past dynamo's recompile limit in eight of them.

This file closes it. `DecodeBuckets` rounds the row count up to a bucket and the
context width up to a multiple, the plan pads the batch to the bucketed row count
and reads at the bucketed width, and the forward sees one of a small set of shapes
at one address.

Four claims:

  1. **The set is closed, and the arithmetic says how big.** `count` is
     `len(rows) * len(widths)`, and it is the number that decides whether closing
     the set helped. Nine row buckets against a 128-multiple over 8192 tokens is
     576 shapes, which is closed and useless.
  2. **A padded row is inert.** Its `context_lens` entry is 0, so it attends over
     nothing and the mask throws its whole row away; its write goes to a sink slot
     one past the end of the pool, which is a legal address the allocator can never
     hand to a sequence. Both halves are needed: a padded row that wrote into the
     pool would put garbage K/V in somebody's history.
  3. **Padding does not cost the window.** `slots[:graph_rows, :width]` is the same
     basic slice Day 51 relied on, so a bucketed rectangle is still the buffer's own
     storage at the buffer's own address.
  4. **The padding is not free and is priced here.** A bucketed rectangle computes
     cells nobody wanted, `waste` is the fraction, and the row axis and the width
     axis waste for different reasons.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nanoserve.batch import pad_prompts
from nanoserve.buckets import (
    DEFAULT_WASTE_LIMIT,
    BucketsUnsound,
    DecodeBuckets,
    bucket_run,
    check_capture_budget,
    check_pad_inert,
    check_run_closed,
    check_shape_in_set,
    check_waste_bounded,
    padded_cells,
    waste,
)
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.compiled import RECOMPILE_LIMIT, DecodeShape, decode_shape, shape_history
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.plan import PlanUnsound, check_plan_addressing, check_plan_current, plan_decode
from nanoserve.scheduler import Request
from nanoserve.slots import check_mapping_is_window


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


def _prefilled(prompts, *, num_blocks=32, block_size=4, batch_size=None, seed=1, **kw):
    cfg = _tiny_config()
    batch_size = batch_size if batch_size is not None else len(prompts)
    cache = BatchedPagedKVCache(
        cfg, BlockAllocator(num_blocks=num_blocks, block_size=block_size), batch_size, **kw
    )
    _prefill_rows(cache, cfg, prompts, tuple(range(len(prompts))), seed=seed)
    return cfg, cache


def _prefill_rows(cache, cfg, prompts, rows, seed=1):
    batch = pad_prompts(list(prompts), pad_id=0)
    for layer in range(cfg.num_hidden_layers):
        k, v = _kv(cfg, len(prompts), batch.max_length, seed=seed + layer)
        cache.write(layer, k, v, batch.attention_mask, rows=rows)


# --- the bucket set ----------------------------------------------------------------


def test_the_row_buckets_stop_at_the_batch_the_cache_has_rows_for():
    buckets = DecodeBuckets(max_batch_size=6, max_model_len=256)

    assert buckets.rows == (1, 2, 4, 6)


def test_a_batch_size_that_is_already_a_bucket_is_not_repeated():
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=256)

    assert buckets.rows == (1, 2, 4, 8)


def test_the_widths_are_multiples_and_the_last_one_is_the_model_length():
    buckets = DecodeBuckets(max_batch_size=4, max_model_len=300, width_multiple=128)

    assert buckets.widths == (128, 256, 300)


def test_a_model_shorter_than_one_multiple_has_a_single_width():
    buckets = DecodeBuckets(max_batch_size=4, max_model_len=64, width_multiple=128)

    assert buckets.widths == (64,)


def test_a_width_that_lands_exactly_on_a_multiple_is_not_repeated():
    buckets = DecodeBuckets(max_batch_size=2, max_model_len=256, width_multiple=128)

    assert buckets.widths == (128, 256)


def test_the_closed_set_is_the_product_of_the_two_axes():
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=512, width_multiple=128)

    assert buckets.count == len(buckets.rows) * len(buckets.widths) == 4 * 4
    assert len(buckets.shapes) == buckets.count
    assert len(set(buckets.shapes)) == buckets.count


def test_a_run_rounds_up_on_both_axes():
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=512, width_multiple=128)

    assert buckets.row_bucket(3) == 4
    assert buckets.width_bucket(130) == 256
    assert buckets.shape_for(3, 130) == DecodeShape(rows=4, context_width=256)


def test_a_width_is_never_rounded_past_the_table_that_holds_it():
    buckets = DecodeBuckets(max_batch_size=2, max_model_len=300, width_multiple=128)

    assert buckets.width_bucket(257) == 300
    assert buckets.width_bucket(300) == 300


def test_a_batch_wider_than_the_cache_has_no_bucket():
    buckets = DecodeBuckets(max_batch_size=4, max_model_len=256)

    with pytest.raises(ValueError, match="4"):
        buckets.row_bucket(5)


def test_a_context_longer_than_the_model_length_has_no_bucket():
    buckets = DecodeBuckets(max_batch_size=4, max_model_len=256)

    with pytest.raises(ValueError, match="256"):
        buckets.width_bucket(257)


def test_a_bucket_set_needs_a_shape():
    with pytest.raises(ValueError):
        DecodeBuckets(max_batch_size=0, max_model_len=128)
    with pytest.raises(ValueError):
        DecodeBuckets(max_batch_size=4, max_model_len=0)
    with pytest.raises(ValueError):
        DecodeBuckets(max_batch_size=4, max_model_len=128, width_multiple=0)


def test_a_bucket_set_renders_one_line_for_a_log():
    line = DecodeBuckets(max_batch_size=8, max_model_len=512).render()

    assert "4 row" in line and "4 width" in line and "16" in line


# --- what bucketing does to a run's shape count -------------------------------------


def test_a_hundred_step_run_presents_a_hundred_shapes_and_four_buckets():
    shapes = shape_history([4] * 100, range(20, 120))
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=512, width_multiple=128)

    assert len(set(shapes)) == 100
    assert len(set(bucket_run(shapes, buckets))) == 1


def test_a_run_that_crosses_a_multiple_gets_a_second_shape():
    shapes = shape_history([2] * 40, range(110, 150))
    buckets = DecodeBuckets(max_batch_size=4, max_model_len=512, width_multiple=128)

    assert len(set(bucket_run(shapes, buckets))) == 2


def test_a_run_whose_row_count_moves_within_a_bucket_keeps_one_shape():
    shapes = shape_history([3, 4, 3, 4, 4], [20] * 5)
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=512, width_multiple=128)

    assert len(set(bucket_run(shapes, buckets))) == 1


def test_the_run_gate_defaults_to_the_limit_dynamo_actually_has():
    """The set may be closed and still bigger than the cache the compiler keeps."""
    buckets = DecodeBuckets(max_batch_size=256, max_model_len=8192, width_multiple=128)
    shapes = bucket_run(shape_history([4] * 20, range(100, 2100, 100)), buckets)

    assert len(set(shapes)) == RECOMPILE_LIMIT + 8
    with pytest.raises(BucketsUnsound, match="distinct"):
        check_run_closed(shapes, buckets)


def test_bucketing_brings_a_run_back_under_the_recompile_limit():
    shapes = shape_history([4] * 40, range(20, 60))
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=512, width_multiple=128)

    with pytest.raises(BucketsUnsound, match="not one of"):
        check_run_closed(shapes, buckets)
    check_run_closed(bucket_run(shapes, buckets), buckets)


def test_a_closed_set_that_is_still_bigger_than_the_limit_is_refused():
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=1024, width_multiple=128)
    shapes = bucket_run(shape_history([1, 2, 4, 8] * 3, [100, 300, 600, 900] * 3), buckets)

    with pytest.raises(BucketsUnsound, match="distinct"):
        check_run_closed(shapes, buckets, limit=3)


def test_check_shape_in_set_refuses_a_shape_that_was_never_bucketed():
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=512, width_multiple=128)

    check_shape_in_set(DecodeShape(rows=4, context_width=256), buckets)
    with pytest.raises(BucketsUnsound, match="not one of"):
        check_shape_in_set(DecodeShape(rows=3, context_width=256), buckets)
    with pytest.raises(BucketsUnsound, match="not one of"):
        check_shape_in_set(DecodeShape(rows=4, context_width=130), buckets)


# --- what the closed set costs ------------------------------------------------------


def test_the_capture_budget_is_the_product_and_the_width_axis_is_what_blows_it():
    wide = DecodeBuckets(max_batch_size=256, max_model_len=8192, width_multiple=128)

    assert wide.count == 9 * 64 == 576
    with pytest.raises(BucketsUnsound, match="576"):
        check_capture_budget(wide, limit=64)


def test_a_coarser_width_multiple_buys_the_budget_back_with_waste():
    coarse = DecodeBuckets(max_batch_size=256, max_model_len=8192, width_multiple=2048)

    assert coarse.count == 9 * 4
    check_capture_budget(coarse, limit=64)


def test_padding_is_counted_in_cells_the_kernel_computes():
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=512, width_multiple=128)
    shapes = shape_history([3], [130])

    assert padded_cells(shapes, buckets) == 4 * 256
    assert shapes[0].cells == 3 * 130
    assert waste(shapes, buckets) == pytest.approx(1 - (3 * 130) / (4 * 256))


def test_the_two_axes_waste_for_different_reasons():
    rows_only = DecodeBuckets(max_batch_size=8, max_model_len=512, width_multiple=1)
    width_only = DecodeBuckets(max_batch_size=8, max_model_len=512, row_buckets=(8,), width_multiple=128)
    shapes = shape_history([5], [130])

    assert waste(shapes, rows_only) == pytest.approx(1 - 5 / 8)
    assert waste(shapes, width_only) == pytest.approx(1 - (5 * 130) / (8 * 256))


def test_waste_of_an_empty_run_is_zero():
    buckets = DecodeBuckets(max_batch_size=8, max_model_len=512)

    assert waste((), buckets) == 0.0


def test_check_waste_bounded_refuses_a_rectangle_that_is_mostly_padding():
    buckets = DecodeBuckets(max_batch_size=256, max_model_len=8192, width_multiple=2048)
    shapes = shape_history([1], [12])

    check_waste_bounded(shape_history([8], [256]), buckets, limit=0.9)
    with pytest.raises(BucketsUnsound, match="padding"):
        check_waste_bounded(shapes, buckets, limit=DEFAULT_WASTE_LIMIT)


# --- the sink slot ------------------------------------------------------------------


def test_the_sink_is_one_slot_past_every_slot_the_allocator_can_hand_out():
    _, cache = _prefilled([[1, 2], [3]], num_blocks=8, block_size=4)

    assert cache.sink_slot == 8 * 4


def test_the_pool_is_one_row_taller_than_the_pool_the_allocator_knows_about():
    cfg, cache = _prefilled([[1, 2], [3]], num_blocks=8, block_size=4)

    assert cache.k_pool[0].shape[0] == 8 * 4 + 1
    assert cache.v_pool[0].shape[0] == 8 * 4 + 1


def test_the_sink_is_not_a_block_and_so_never_shows_up_in_the_ledger():
    _, cache = _prefilled([[1, 2], [3]], num_blocks=8, block_size=4)
    plan_decode(cache, buckets=DecodeBuckets(2, cache.max_model_len))

    assert cache.allocator.num_free + len(cache.allocator.allocated_blocks) == 8


# --- a bucketed plan ----------------------------------------------------------------


def test_a_bucketed_plan_pads_the_batch_up_to_a_row_bucket():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    buckets = DecodeBuckets(4, cache.max_model_len, width_multiple=8)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=buckets)

    assert plan.rows == (0, 1, 2)
    assert plan.batch_size == 3
    assert plan.pad_rows == 1
    assert plan.graph_rows == 4
    assert tuple(plan.slot_mapping.shape) == (4, 8)
    assert tuple(plan.context_lens.shape) == (4,)
    assert tuple(plan.write_slots.shape) == (4,)
    assert tuple(plan.positions.shape) == (4, 1)


def test_a_padded_row_attends_over_nothing():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))

    assert plan.context_lens.tolist() == [4, 2, 3, 0]


def test_a_padded_row_writes_to_the_sink_and_not_into_the_pool():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))

    assert plan.sink_slot == cache.sink_slot
    assert int(plan.write_slots[-1]) == cache.sink_slot
    assert all(int(s) != cache.sink_slot for s in plan.write_slots[:3])
    check_pad_inert(plan)


def test_check_pad_inert_catches_a_padded_row_pointed_at_the_pool():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))
    slots = plan.write_slots.clone()
    slots[-1] = 0
    with pytest.raises(BucketsUnsound, match="sink"):
        check_pad_inert(replace(plan, write_slots=slots))


def test_check_pad_inert_catches_a_padded_row_that_would_attend():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))
    lens = plan.context_lens.clone()
    lens[-1] = 2
    with pytest.raises(BucketsUnsound, match="attend"):
        check_pad_inert(replace(plan, context_lens=lens))


def test_a_padded_rows_position_is_zero_and_not_a_neighbours():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))

    assert plan.positions.squeeze(1).tolist() == [3, 1, 2, 0]


def test_a_full_batch_is_padded_on_the_width_axis_alone():
    _, cache = _prefilled([[1, 2, 3], [4]], batch_size=2)
    plan = plan_decode(cache, buckets=DecodeBuckets(2, cache.max_model_len, width_multiple=8))

    assert plan.pad_rows == 0
    assert plan.graph_rows == 2
    assert plan.graph_width == 8
    assert plan.max_ctx == 4


def test_an_unbucketed_plan_is_exactly_what_day_51_built():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2))

    assert plan.pad_rows == 0
    assert plan.sink_slot is None
    assert not plan.is_bucketed
    assert plan.graph_rows == 3
    assert plan.graph_width == plan.max_ctx == 4


def test_a_bucketed_rectangle_is_still_a_window_on_the_persistent_table():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))

    assert cache.slot_table.is_window(plan.slot_mapping)
    check_mapping_is_window(cache.slot_table, plan)


def test_the_padded_rows_of_a_window_come_off_the_end_of_the_table():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))

    assert plan.slot_mapping[3].tolist() == cache.slot_table.slots[3, :8].tolist()


def test_a_bucketed_plan_over_scattered_rows_is_a_gather_and_still_correct():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))

    assert plan.graph_rows == 2
    assert plan.context_lens.tolist() == [4, 3]
    assert not cache.slot_table.is_window(plan.slot_mapping)


def test_a_bucketed_plan_still_passes_the_day_50_gates():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))

    check_plan_addressing(plan)
    check_plan_current(plan, cache)


def test_check_plan_addressing_still_catches_a_real_row_pointed_somewhere_else():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))
    slots = plan.write_slots.clone()
    slots[0] = 31
    with pytest.raises(PlanUnsound, match="does not look"):
        check_plan_addressing(replace(plan, write_slots=slots))


def test_check_plan_addressing_refuses_a_padded_row_that_claims_a_history():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))
    lens = plan.context_lens.clone()
    lens[-1] = 1
    with pytest.raises(PlanUnsound, match="padded"):
        check_plan_addressing(replace(plan, context_lens=lens))


def test_a_bucketed_plan_prices_the_cells_nobody_wanted():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))

    assert plan.cells == 3 * 4
    assert plan.graph_cells == 4 * 8
    assert plan.pad_cells == 32 - 12
    assert plan.pad_share == pytest.approx(20 / 32)


def test_a_bucketed_plan_says_so_when_it_renders():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=DecodeBuckets(4, cache.max_model_len, width_multiple=8))

    assert "4 x 8" in plan.render()


# --- the table's side of it ---------------------------------------------------------


def test_the_table_counts_the_cells_it_handed_out_as_padding():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    buckets = DecodeBuckets(4, cache.max_model_len, width_multiple=8)
    plan_decode(cache, rows=(0, 1, 2), buckets=buckets)
    plan_decode(cache, rows=(0, 1, 2), buckets=buckets)

    assert cache.slot_table.pads == 2
    assert cache.slot_table.padded_cells == 2 * 1 * 8


def test_padding_does_not_change_how_many_cells_a_step_writes():
    _, cache = _prefilled([[1, 2, 3], [4], [5, 6]], batch_size=4)
    buckets = DecodeBuckets(4, cache.max_model_len, width_multiple=8)
    for _ in range(5):
        plan_decode(cache, rows=(0, 1, 2), buckets=buckets)

    assert cache.slot_table.appended_cells == 15
    assert cache.slot_table.length(3) == 0


def test_a_read_cannot_pad_past_the_rows_the_table_has():
    _, cache = _prefilled([[1, 2], [3]], batch_size=2)

    with pytest.raises(ValueError, match="2"):
        cache.slot_table.read((0, 1), 4, pad_rows=1)


def test_a_negative_pad_is_refused():
    _, cache = _prefilled([[1, 2], [3]], batch_size=2)

    with pytest.raises(ValueError):
        cache.slot_table.read((0,), 4, pad_rows=-1)


# --- through the forward -------------------------------------------------------------


def test_a_bucketed_forward_returns_the_same_logits_for_the_real_rows():
    model, cfg = _model()
    plain = Engine.build(model, num_blocks=32, block_size=4, max_batch_size=4, max_model_len=32)
    padded = Engine.build(
        model, num_blocks=32, block_size=4, max_batch_size=4, max_model_len=32, bucket_decode=True
    )
    want = got = None
    for engine, keep in ((plain, "want"), (padded, "got")):
        cache = engine.cache
        _prefill_rows(cache, cfg, [[1, 2, 3], [4], [5, 6]], (0, 1, 2), seed=3)
        buckets = engine.cache.decode_buckets
        plan = plan_decode(cache, rows=(0, 1, 2), buckets=buckets)
        rows = 3 if buckets is None else plan.graph_rows
        ids = torch.zeros(rows, 1, dtype=torch.long)
        ids[:3, 0] = torch.tensor([7, 8, 9])
        logits = model.forward(ids, plan.positions, cache=cache.view((0, 1, 2), plan=plan))
        if keep == "want":
            want = logits[:3]
        else:
            got = logits[:3]
    assert torch.allclose(want, got, atol=1e-5)


def test_a_padded_row_leaves_no_trace_in_the_pool():
    model, cfg = _model()
    engine = Engine.build(
        model, num_blocks=32, block_size=4, max_batch_size=4, max_model_len=32, bucket_decode=True
    )
    cache = engine.cache
    _prefill_rows(cache, cfg, [[1, 2, 3], [4], [5, 6]], (0, 1, 2), seed=3)
    plan = plan_decode(cache, rows=(0, 1, 2), buckets=cache.decode_buckets)
    before = cache.k_pool[0][: cache.sink_slot].clone()
    ids = torch.zeros(plan.graph_rows, 1, dtype=torch.long)
    model.forward(ids, plan.positions, cache=cache.view((0, 1, 2), plan=plan))
    changed = (cache.k_pool[0][: cache.sink_slot] != before).flatten(1).any(dim=1)

    assert sorted(int(i) for i in changed.nonzero().flatten()) == sorted(
        int(s) for s in plan.write_slots[:3]
    )


def test_forty_steps_of_a_fixed_batch_present_one_bucketed_shape():
    _, cache = _prefilled([[1, 2, 3, 4], [5], [6, 7]], batch_size=4, num_blocks=64)
    buckets = DecodeBuckets(4, cache.max_model_len, width_multiple=64)
    ids = torch.zeros(4, 1, dtype=torch.long)
    seen = []
    for _ in range(40):
        plan = plan_decode(cache, rows=(0, 1, 2), buckets=buckets)
        seen.append(decode_shape(ids, cache.view((0, 1, 2), plan=plan)))

    assert len(set(seen)) == 1
    assert set(seen) == {DecodeShape(rows=4, context_width=64)}
    check_run_closed(seen, buckets)


def test_the_same_forty_steps_unbucketed_present_forty():
    _, cache = _prefilled([[1, 2, 3, 4], [5], [6, 7]], batch_size=4, num_blocks=64)
    ids = torch.zeros(3, 1, dtype=torch.long)
    seen = [
        decode_shape(ids, cache.view((0, 1, 2), plan=plan_decode(cache, rows=(0, 1, 2))))
        for _ in range(40)
    ]

    assert len(set(seen)) == 40


def _run(engine, requests):
    for request in requests:
        engine.add_request(request)
    done = {}
    while engine.has_unfinished():
        out = engine.step()
        for request in out.finished:
            done[request.request_id] = list(request.output_token_ids)
    return done


def test_an_engine_run_under_buckets_stays_inside_the_bucket_set():
    model, _ = _model()
    engine = Engine.build(
        model,
        num_blocks=64,
        block_size=4,
        max_batch_size=4,
        max_model_len=64,
        bucket_decode=True,
        compile_decode="off",
    )
    _run(
        engine,
        [Request("a", [1, 2, 3, 4], max_new_tokens=12), Request("b", [5, 6], max_new_tokens=12)],
    )
    shapes = engine.decode_forward.shapes

    assert len(shapes) <= 2  # one per row bucket the run passed through
    check_run_closed(shapes, engine.cache.decode_buckets)


def test_the_same_run_without_buckets_presents_one_shape_per_step():
    model, _ = _model()
    engine = Engine.build(
        model, num_blocks=64, block_size=4, max_batch_size=4, max_model_len=64, compile_decode="off"
    )
    _run(
        engine,
        [Request("a", [1, 2, 3, 4], max_new_tokens=12), Request("b", [5, 6], max_new_tokens=12)],
    )

    assert len(engine.decode_forward.shapes) > 8


def test_the_two_engines_generate_the_same_tokens():
    model, _ = _model()
    out = []
    for bucket in (False, True):
        engine = Engine.build(
            model,
            num_blocks=64,
            block_size=4,
            max_batch_size=4,
            max_model_len=64,
            bucket_decode=bucket,
        )
        out.append(
            _run(
                engine,
                [
                    Request("a", [1, 2, 3, 4], max_new_tokens=10),
                    Request("b", [5, 6], max_new_tokens=10),
                ],
            )
        )
    assert out[0] == out[1]


def test_an_engine_without_buckets_is_untouched():
    model, _ = _model()
    engine = Engine.build(model, num_blocks=32, block_size=4, max_batch_size=4)

    assert engine.cache.decode_buckets is None
