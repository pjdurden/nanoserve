"""Day 61 tests: the capture list loses its width axis.

Day 52 bucketed both axes of the decode shape because both of them moved, and the
price was a product: `len(rows) * len(widths)`, which at 256 rows and 8192 tokens on
a 128-token multiple is 576 graphs. The width axis was in the set for exactly one
reason, and it is a property of the *read*: the Day-28 rectangle gathers a
`[rows, width]` mapping and scores it into a `[rows, heads, 1, width]` tensor, so the
width is a real dimension of a real allocation and a wider one costs real memory.

Day 59 wrote a read that never builds either of those, Day 60 wired it to a flag, and
this is the day that spends it. A streamed read walks `cdiv(context_lens[row], block)`
tiles of its own row and holds one tile, so the mapping's width buys nothing: hand it
the whole table every step and it touches exactly the same cells. The width stops
being a thing to round and becomes a constant, the set collapses to the row axis, and
the three things that fall out of that are what this file checks.

Five claims:

  1. **A streamed bucket set has one width, and it is the table's.** 576 shapes
     become 9, and `count` stops being a product.
  2. **The price is stated in the read's currency or it is a lie.** `cells_for` is
     the cells the read really touches, which is a bucketed rectangle on the default
     and a tile per row on the streamed one. Routing `waste` through it is what keeps
     the number comparable across the two arms instead of reporting a 99% waste
     against a rectangle that is never built.
  3. **The two halves have to agree, and the gate is not optional.** A streamed
     bucket set under a rectangle read hands the rectangle a full-width mapping on
     every step from the very first token: correct, and 64x the memory the bucketing
     was there to bound. That is the one combination in this day that corrupts a
     server rather than slowing it, so it is refused.
  4. **The shared arena is the tile.** `shared_pool_bytes` sized by a score rectangle
     is the wrong number for a read that never builds one.
  5. **The budget stops being the binding constraint.** `plan_capture` turns three
     things into a width ceiling, and none of them is a width ceiling any more: a
     streamed workspace has no width in it, so a budget either holds the tile or it
     does not, and the graph limit becomes a statement about rows.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.batch import pad_prompts
from nanoserve.buckets import (
    BucketsUnsound,
    DecodeBuckets,
    check_capture_budget,
    check_pad_inert,
    check_read_matches,
    check_run_closed,
    check_shape_in_set,
    waste,
)
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.captured import (
    CaptureUnsound,
    check_capture_ready,
    pool_sharing_ratio,
    private_pool_bytes,
    shared_pool_bytes,
    streamed_workspace_bytes,
    workspace_bytes,
)
from nanoserve.compiled import DecodeShape
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.kernels.flash_decoding import plan_splits
from nanoserve.launch import CaptureTooSmall, KVPoolPlan, plan_capture
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.partials import allocate_for, allocate_partials
from nanoserve.plan import plan_decode
from nanoserve.reads import RECTANGLE, SPLIT, STREAMED, PagedRead
from nanoserve.scheduler import Request
from nanoserve.warmup import warm_shapes

#: The deployment every arithmetic table in this week is stated over, so a number
#: here can be read next to Day 52's and Day 54's without converting anything.
SERVING_ROWS = 256
SERVING_LEN = 8192
SERVING_HEADS = 32


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
    batch = pad_prompts(list(prompts), pad_id=0)
    rows = tuple(range(len(prompts)))
    for layer in range(cfg.num_hidden_layers):
        k, v = _kv(cfg, len(prompts), batch.max_length, seed=seed + layer)
        cache.write(layer, k, v, batch.attention_mask, rows=rows)
    return cfg, cache


def _run(engine, requests):
    for request in requests:
        engine.add_request(request)
    done = {}
    while engine.has_unfinished():
        out = engine.step()
        for request in out.finished:
            done[request.request_id] = list(request.output_token_ids)
    return done


# --- the set loses an axis ----------------------------------------------------------


def test_a_streamed_bucket_set_has_one_width_and_it_is_the_table():
    buckets = DecodeBuckets(4, 300, streamed=True, block=32)

    assert buckets.widths == (300,)


def test_the_width_multiple_is_not_consulted_when_the_read_is_streamed():
    coarse = DecodeBuckets(4, 300, width_multiple=128, streamed=True, block=32)
    fine = DecodeBuckets(4, 300, width_multiple=8, streamed=True, block=32)

    assert coarse.widths == fine.widths == (300,)


def test_the_count_stops_being_a_product():
    buckets = DecodeBuckets(SERVING_ROWS, SERVING_LEN, streamed=True, block=32)

    assert buckets.count == len(buckets.rows) == 9


def test_the_serving_set_goes_from_576_shapes_to_9():
    rectangle = DecodeBuckets(SERVING_ROWS, SERVING_LEN)
    streamed = DecodeBuckets(SERVING_ROWS, SERVING_LEN, streamed=True, block=32)

    assert rectangle.count == 576
    assert streamed.count == 9


def test_the_rows_are_the_same_rows():
    rectangle = DecodeBuckets(6, 512)
    streamed = DecodeBuckets(6, 512, streamed=True, block=32)

    assert streamed.rows == rectangle.rows == (1, 2, 4, 6)


def test_every_legal_width_rounds_to_the_table():
    buckets = DecodeBuckets(4, 512, streamed=True, block=32)

    assert buckets.width_bucket(1) == 512
    assert buckets.width_bucket(129) == 512
    assert buckets.width_bucket(512) == 512


def test_a_width_past_the_table_is_still_refused():
    buckets = DecodeBuckets(4, 512, streamed=True, block=32)

    with pytest.raises(ValueError, match="longer than max_model_len"):
        buckets.width_bucket(513)


def test_a_width_under_one_is_still_refused():
    buckets = DecodeBuckets(4, 512, streamed=True, block=32)

    with pytest.raises(ValueError, match="at least one column wide"):
        buckets.width_bucket(0)


def test_shape_for_pins_the_width_and_still_buckets_the_rows():
    buckets = DecodeBuckets(8, 512, streamed=True, block=32)

    assert buckets.shape_for(3, 17) == DecodeShape(rows=4, context_width=512)


def test_a_streamed_set_needs_a_tile_and_will_not_guess_one():
    with pytest.raises(ValueError, match="tile"):
        DecodeBuckets(4, 512, streamed=True)


def test_a_negative_tile_is_refused():
    with pytest.raises(ValueError, match="at least one key"):
        DecodeBuckets(4, 512, streamed=True, block=-1)


def test_the_rectangle_set_is_untouched():
    buckets = DecodeBuckets(4, 300, width_multiple=128)

    assert buckets.widths == (128, 256, 300)
    assert buckets.streamed is False
    assert buckets.block == 0


def test_render_names_the_read():
    buckets = DecodeBuckets(6, 512, streamed=True, block=32)

    assert "streamed" in buckets.render()
    assert "4 row buckets" in buckets.render()


def test_a_shape_read_at_a_multiple_is_out_of_a_streamed_set():
    buckets = DecodeBuckets(4, 512, streamed=True, block=32)

    with pytest.raises(BucketsUnsound, match="rounds to"):
        check_shape_in_set(DecodeShape(rows=4, context_width=128), buckets)


def test_a_streamed_run_is_one_shape_per_row_bucket():
    buckets = DecodeBuckets(4, 512, streamed=True, block=32)
    shapes = [DecodeShape(rows=2, context_width=w) for w in range(8, 400)]

    check_run_closed([buckets.shape_for(s.rows, s.context_width) for s in shapes], buckets)


def test_the_capture_budget_that_refused_the_rectangle_holds_the_streamed_set():
    rectangle = DecodeBuckets(SERVING_ROWS, SERVING_LEN)
    streamed = DecodeBuckets(SERVING_ROWS, SERVING_LEN, streamed=True, block=32)

    with pytest.raises(BucketsUnsound, match="compile bill"):
        check_capture_budget(rectangle, limit=64)
    check_capture_budget(streamed, limit=64)


# --- the price, in the read's own currency ------------------------------------------


def test_cells_for_is_the_bucketed_rectangle_on_the_default_read():
    buckets = DecodeBuckets(8, 512, width_multiple=128)

    assert buckets.cells_for(3, 130) == 4 * 256


def test_cells_for_is_a_tile_per_row_when_streamed():
    buckets = DecodeBuckets(8, 512, streamed=True, block=32)

    # Three real rows walking one tile of 32 keys each: the padded fourth row has a
    # context length of 0 and therefore no tiles at all.
    assert buckets.cells_for(3, 17) == 3 * 32


def test_a_context_that_spans_two_tiles_is_charged_two():
    buckets = DecodeBuckets(8, 512, streamed=True, block=32)

    assert buckets.cells_for(2, 33) == 2 * 64


def test_a_streamed_charge_does_not_grow_with_the_table():
    narrow = DecodeBuckets(8, 512, streamed=True, block=32)
    wide = DecodeBuckets(8, 8192, streamed=True, block=32)

    assert narrow.cells_for(4, 100) == wide.cells_for(4, 100)


def test_a_rectangle_charge_does_grow_with_the_multiple():
    fine = DecodeBuckets(8, 8192, width_multiple=128)
    coarse = DecodeBuckets(8, 8192, width_multiple=2048)

    assert coarse.cells_for(4, 100) == 16 * fine.cells_for(4, 100)


def test_the_width_padding_is_free_and_the_row_padding_is_not():
    buckets = DecodeBuckets(8, 512, streamed=True, block=32)
    shapes = [DecodeShape(rows=5, context_width=64)]

    # 5 rows x 64 keys is exactly two tiles each and no row is padded into a charge,
    # so the streamed read touches precisely what the run asked for.
    assert waste(shapes, buckets) == 0.0


def test_a_ragged_tile_is_still_charged_whole():
    buckets = DecodeBuckets(8, 512, streamed=True, block=32)
    shapes = [DecodeShape(rows=4, context_width=33)]

    assert 0.4 < waste(shapes, buckets) < 0.6


def test_the_rectangle_waste_is_unchanged():
    buckets = DecodeBuckets(8, 8192, width_multiple=2048)
    shapes = [DecodeShape(rows=4, context_width=100)]

    assert waste(shapes, buckets) == pytest.approx(1 - 400 / 8192)


def test_a_streamed_run_wastes_far_less_than_a_bucketed_rectangle_one():
    shapes = [DecodeShape(rows=6, context_width=w) for w in range(20, 400)]
    rectangle = DecodeBuckets(8, 8192, width_multiple=2048)
    streamed = DecodeBuckets(8, 8192, streamed=True, block=32)

    assert waste(shapes, rectangle) > 0.9
    assert waste(shapes, streamed) < 0.2


# --- the two halves have to agree ---------------------------------------------------


def test_a_matching_pair_passes():
    check_read_matches(
        DecodeBuckets(4, 512, streamed=True, block=32), PagedRead(STREAMED, block=32)
    )
    check_read_matches(DecodeBuckets(4, 512), PagedRead(RECTANGLE))


def test_a_streamed_set_under_a_rectangle_read_is_refused():
    buckets = DecodeBuckets(4, 8192, streamed=True, block=32)

    with pytest.raises(BucketsUnsound, match="rectangle"):
        check_read_matches(buckets, PagedRead(RECTANGLE))


def test_that_refusal_says_how_wide_the_rectangle_would_be():
    buckets = DecodeBuckets(4, 8192, streamed=True, block=32)

    with pytest.raises(BucketsUnsound, match="8192"):
        check_read_matches(buckets, PagedRead(RECTANGLE))


def test_a_rectangle_set_under_a_streamed_read_is_refused():
    buckets = DecodeBuckets(4, 8192, width_multiple=2048)

    with pytest.raises(BucketsUnsound, match="width buckets"):
        check_read_matches(buckets, PagedRead(STREAMED, block=32))


def test_a_tile_mismatch_is_refused():
    buckets = DecodeBuckets(4, 512, streamed=True, block=32)

    with pytest.raises(BucketsUnsound, match="tile"):
        check_read_matches(buckets, PagedRead(STREAMED, block=8))


def test_check_capture_ready_runs_the_same_gate():
    cfg, cache = _prefilled(
        [[1, 2, 3, 4]], bucket_decode=True, persist_inputs=True, streamed_read=True, read_block=8
    )
    cache.read = PagedRead(RECTANGLE)

    with pytest.raises(BucketsUnsound):
        check_capture_ready(cache)


def test_check_capture_ready_still_refuses_an_unbucketed_cache():
    cfg, cache = _prefilled([[1, 2, 3, 4]], persist_inputs=True)

    with pytest.raises(CaptureUnsound, match="bucket_decode"):
        check_capture_ready(cache)


# --- the cache builds both halves from one flag -------------------------------------


def test_a_streamed_cache_gets_a_streamed_bucket_set():
    cfg, cache = _prefilled(
        [[1, 2, 3, 4]], bucket_decode=True, streamed_read=True, read_block=8
    )

    assert cache.decode_buckets.streamed is True
    assert cache.decode_buckets.block == 8


def test_a_default_cache_keeps_its_width_axis():
    cfg, cache = _prefilled(
        [[1, 2, 3, 4]], num_blocks=256, bucket_decode=True, max_model_len=512
    )

    assert cache.decode_buckets.streamed is False
    assert cache.decode_buckets.widths == (128, 256, 384, 512)
    assert cache.decode_buckets.count > len(cache.decode_buckets.rows)


def test_the_cache_it_built_agrees_with_the_read_it_built():
    cfg, cache = _prefilled(
        [[1, 2, 3, 4]], bucket_decode=True, streamed_read=True, read_block=8
    )

    check_read_matches(cache.decode_buckets, cache.read)


def test_a_streamed_plan_reads_at_the_full_table_width():
    cfg, cache = _prefilled(
        [[1, 2, 3, 4], [5, 6]],
        bucket_decode=True,
        max_model_len=64,
        streamed_read=True,
        read_block=8,
    )
    plan = plan_decode(cache)

    assert plan.graph_width == 64
    assert plan.slot_mapping.shape == (2, 64)


def test_the_padded_rows_of_a_streamed_plan_are_still_inert():
    cfg, cache = _prefilled(
        [[1, 2, 3, 4], [5, 6], [7]],
        batch_size=4,
        bucket_decode=True,
        max_model_len=64,
        streamed_read=True,
        read_block=8,
    )
    plan = plan_decode(cache, rows=(0, 1, 2))

    assert plan.pad_rows == 1
    check_pad_inert(plan)


def test_a_streamed_engine_presents_one_shape_for_a_whole_run():
    model, _ = _model()
    engine = Engine.build(
        model,
        num_blocks=64,
        block_size=4,
        max_batch_size=2,
        max_model_len=64,
        bucket_decode=True,
        streamed_read=True,
        read_block=8,
    )
    _run(engine, [Request("a", [1, 2, 3, 4], max_new_tokens=12)])

    assert engine.cache.decode_buckets.count == 2


def test_the_two_bucket_sets_generate_the_same_tokens():
    out = []
    for streamed in (False, True):
        model, _ = _model()
        engine = Engine.build(
            model,
            num_blocks=64,
            block_size=4,
            max_batch_size=4,
            max_model_len=64,
            bucket_decode=True,
            streamed_read=streamed,
            read_block=8,
        )
        out.append(
            _run(
                engine,
                [
                    Request("a", [1, 2, 3, 4], max_new_tokens=8),
                    Request("b", [5, 6], max_new_tokens=8),
                ],
            )
        )
    assert out[0] == out[1]


def test_the_health_saving_becomes_arithmetic_under_a_streamed_set():
    """Day 60's counter degenerates here, and the point is to know that it does.

    `PagedRead` charges `score_cells` off `slot_mapping.shape[1]`, which under a
    streamed bucket set is `max_model_len` on every call, and it holds
    `min(block, width)`, which is `block` on every call. So the quotient is exactly
    `max_model_len / block` whatever the run did: a server that answered one
    twelve-token request reports the same saving as one that streamed for an hour.

    It cannot be fixed and that is the interesting half. The honest numerator is the
    width a *rectangle* read would have bucketed this step to, which is the longest
    real history in the batch, which is `int(context_lens.max())` on the device: Day
    48's synchronisation, once per call per layer, to make a number in a log nicer.
    """
    model, _ = _model()
    engine = Engine.build(
        model,
        num_blocks=256,
        block_size=4,
        max_batch_size=4,
        max_model_len=512,
        bucket_decode=True,
        streamed_read=True,
        read_block=8,
    )
    _run(engine, [Request("a", [1, 2, 3, 4], max_new_tokens=6)])
    short = engine.cache.read.stats()

    engine2 = Engine.build(
        _model()[0],
        num_blocks=256,
        block_size=4,
        max_batch_size=4,
        max_model_len=512,
        bucket_decode=True,
        streamed_read=True,
        read_block=8,
    )
    _run(engine2, [Request("a", [1, 2, 3, 4], max_new_tokens=40)])
    long = engine2.cache.read.stats()

    assert short.calls < long.calls
    assert short.saving == long.saving == 512 / 8


# --- the arena is the tile ----------------------------------------------------------


def test_the_shared_pool_is_sized_by_the_tile_when_a_block_is_given():
    shapes = (DecodeShape(rows=8, context_width=8192),)

    assert shared_pool_bytes(shapes, SERVING_HEADS, block=32) == streamed_workspace_bytes(
        shapes[0], SERVING_HEADS, 32
    )


def test_the_tile_pool_is_smaller_than_the_rectangle_pool_by_the_width():
    shape = DecodeShape(rows=8, context_width=8192)

    assert workspace_bytes(shape, SERVING_HEADS) == 256 * shared_pool_bytes(
        (shape,), SERVING_HEADS, block=32
    )


def test_private_pools_shrink_the_same_way():
    shapes = tuple(DecodeShape(rows=r, context_width=8192) for r in (1, 2, 4))

    assert private_pool_bytes(shapes, SERVING_HEADS, block=32) == sum(
        streamed_workspace_bytes(s, SERVING_HEADS, 32) for s in shapes
    )


def test_the_sharing_ratio_is_the_row_axis_alone():
    buckets = DecodeBuckets(8, 8192, streamed=True, block=32)

    # 1 + 2 + 4 + 8 rows over a max of 8, and nothing else is left to vary.
    assert pool_sharing_ratio(buckets.shapes, SERVING_HEADS, block=32) == pytest.approx(
        15 / 8
    )


def test_a_pool_with_no_shapes_is_still_free():
    assert shared_pool_bytes((), SERVING_HEADS, block=32) == 0
    assert private_pool_bytes((), SERVING_HEADS, block=32) == 0


# --- the budget stops being the binding constraint ----------------------------------


def _capture_engine(**kw):
    model, _ = _model()
    return Engine.build(
        model,
        num_blocks=256,
        block_size=4,
        max_batch_size=4,
        max_model_len=512,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        **kw,
    )


def _pool_plan(engine) -> KVPoolPlan:
    return KVPoolPlan(
        num_blocks=256,
        block_size=4,
        max_batch_size=4,
        max_model_len=512,
        bytes_per_block=1024,
        budget_bytes=1 << 20,
        dtype=torch.float32,
    )


def test_a_streamed_capture_list_is_bound_by_the_read():
    engine = _capture_engine(streamed_read=True, read_block=8)
    capture = plan_capture(engine, _pool_plan(engine), budget_bytes=1 << 20)

    assert capture.width_bound_by == "read"
    assert capture.max_width == 512


def test_the_budget_that_would_have_bitten_a_rectangle_does_not_bite_the_tile():
    rectangle = _capture_engine()
    streamed = _capture_engine(streamed_read=True, read_block=8)
    # Exactly a 256-token score rectangle at 4 rows and 8 heads in fp32, which is
    # half the context this deployment sells.
    tight = 4 * 8 * 256 * 4

    rect_plan = plan_capture(rectangle, _pool_plan(rectangle), budget_bytes=tight)
    assert (rect_plan.width_bound_by, rect_plan.max_width) == ("budget", 256)

    tile_plan = plan_capture(streamed, _pool_plan(streamed), budget_bytes=tight)
    assert (tile_plan.width_bound_by, tile_plan.max_width) == ("read", 512)


def test_a_budget_too_small_for_one_tile_is_refused():
    engine = _capture_engine(streamed_read=True, read_block=8)

    with pytest.raises(CaptureTooSmall, match="tile"):
        plan_capture(engine, _pool_plan(engine), budget_bytes=16)


def test_a_streamed_capture_plan_carries_its_tile():
    engine = _capture_engine(streamed_read=True, read_block=8)
    capture = plan_capture(engine, _pool_plan(engine))

    assert capture.block == 8
    assert capture.as_dict()["read_block"] == 8


def test_the_pool_bytes_of_a_streamed_plan_is_the_tile():
    engine = _capture_engine(streamed_read=True, read_block=8)
    capture = plan_capture(engine, _pool_plan(engine))

    assert capture.pool_bytes == shared_pool_bytes(capture.shapes, capture.num_heads, block=8)


def test_the_streamed_list_is_the_row_axis():
    engine = _capture_engine(streamed_read=True, read_block=8)
    capture = plan_capture(engine, _pool_plan(engine))

    assert capture.count == len(engine.cache.decode_buckets.rows) == 3


def test_more_row_buckets_than_graphs_is_refused_on_the_row_axis():
    engine = _capture_engine(streamed_read=True, read_block=8)

    with pytest.raises(CaptureTooSmall, match="no width axis left"):
        plan_capture(engine, _pool_plan(engine), limit=2)


def test_describe_names_the_read():
    engine = _capture_engine(streamed_read=True, read_block=8)
    capture = plan_capture(engine, _pool_plan(engine))

    assert "streamed read" in capture.describe()


def test_warm_shapes_over_a_streamed_set_is_the_row_axis():
    buckets = DecodeBuckets(8, 512, streamed=True, block=32)
    shapes = warm_shapes(buckets)

    assert len(shapes) == len(buckets.rows)
    assert shapes[0].rows == 8
    assert {s.context_width for s in shapes} == {512}


# --- Day 64: the split's arena, planned here because the list is ---------------------
#
# The split read is a partition of the streamed read's tiles, so everything above
# holds and one number is added to it. `plan_capture(split_read=True)` calls
# `plan_splits` once, here, over the row buckets this list covers and the width the
# set already fixed, and the plan carries the count so the arena can be allocated
# from it instead of from a tensor on some later step.


def _split_engine(**kw):
    """A deployment wide enough to be worth splitting: `DEFAULT_PARTITION` is 512
    keys, so a 512-token context is one chunk however small the batch is."""
    model, _ = _model()
    return Engine.build(
        model,
        num_blocks=1024,
        block_size=4,
        max_batch_size=4,
        max_model_len=2048,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        streamed_read=True,
        read_block=8,
        **kw,
    )


def _split_pool_plan() -> KVPoolPlan:
    return KVPoolPlan(
        num_blocks=1024,
        block_size=4,
        max_batch_size=4,
        max_model_len=2048,
        bytes_per_block=1024,
        budget_bytes=1 << 24,
        dtype=torch.float32,
    )


def test_a_split_plan_decides_its_count_once_over_the_whole_list():
    engine = _split_engine()
    capture = plan_capture(engine, _split_pool_plan(), split_read=True)
    buckets = engine.cache.decode_buckets

    assert capture.splits == plan_splits(buckets.rows, capture.num_heads, 2048, 8)
    assert capture.splits > 1


def test_a_context_under_the_partition_floor_is_not_split_at_all():
    """`choose_splits` refuses to cut a row into chunks shorter than a partition, so
    a small deployment gets `splits=1`: one chunk, one program per (row, head), and
    a second pass over a single partial. The plan says 1 rather than 0, because 0
    means some other read and 1 means this read declining to cut."""
    engine = _capture_engine(streamed_read=True, read_block=8)
    capture = plan_capture(engine, _pool_plan(engine), split_read=True)

    assert capture.max_width == 512
    assert capture.splits == 1


def test_a_split_plan_carries_the_head_dim_the_other_two_reads_never_needed():
    engine = _split_engine()
    capture = plan_capture(engine, _split_pool_plan(), split_read=True)

    assert capture.head_dim == engine.model.config.head_dim


def test_the_arena_of_a_split_plan_is_the_tiles_plus_the_partials():
    """The number Day 63 priced and nothing charged for, now in `pool_bytes`."""
    engine = _split_engine()
    capture = plan_capture(engine, _split_pool_plan(), split_read=True)

    assert capture.pool_bytes == shared_pool_bytes(
        capture.shapes,
        capture.num_heads,
        block=8,
        splits=capture.splits,
        head_dim=capture.head_dim,
    )


def test_a_split_arena_is_strictly_bigger_than_the_streamed_one_it_partitions():
    engine = _split_engine()
    streamed = plan_capture(engine, _split_pool_plan())
    split = plan_capture(engine, _split_pool_plan(), split_read=True)

    assert split.pool_bytes > streamed.pool_bytes
    assert streamed.splits == 0


def test_a_split_plan_says_so_in_the_payload_and_in_the_line():
    engine = _split_engine()
    capture = plan_capture(engine, _split_pool_plan(), split_read=True)

    assert capture.as_dict()["read_splits"] == capture.splits
    assert f"{capture.splits}-way split" in capture.describe()


def test_a_split_over_the_rectangle_read_is_refused():
    """There is no tile to partition, and a split of a materialised score rectangle
    is a second pass over an intermediate that was already the whole answer."""
    engine = _capture_engine()

    with pytest.raises(ValueError, match="streamed"):
        plan_capture(engine, _pool_plan(engine), split_read=True)


def test_a_budget_that_holds_the_tiles_but_not_the_partials_is_refused():
    """The failure Day 63 could not have: the arena is now priced, so a budget that
    would have passed on the tile alone is told which term overran it."""
    engine = _split_engine()
    streamed = plan_capture(engine, _split_pool_plan())
    tight = streamed.pool_bytes * 2

    plan_capture(engine, _split_pool_plan(), budget_bytes=tight)
    with pytest.raises(CaptureTooSmall, match="partials"):
        plan_capture(engine, _split_pool_plan(), budget_bytes=tight, split_read=True)


def test_a_planned_arena_is_allocated_from_the_plan_and_from_nothing_else():
    engine = _split_engine()
    capture = plan_capture(engine, _split_pool_plan(), split_read=True)

    workspace = allocate_for(capture, head_dim=capture.head_dim)

    assert workspace.max_rows == capture.max_rows
    assert workspace.splits == capture.splits
    assert workspace.context_width == capture.max_width
    assert workspace.block == capture.block


# --- the third read's half of the same agreement ------------------------------------
#
# Day 65. `check_read_matches` has asked one question since Day 61: does this set
# still have a width axis, and does the read still want one. A split read makes that
# question insufficient rather than wrong. It wants exactly what the streamed read
# wants of the width, and it wants one more thing the set is the only place to state:
# the chunk count. The arena is `rows * heads * splits * (head_dim + 2)` and it is
# reserved once at boot, so a set priced for sixteen chunks under a read that runs
# one has bought fifteen sixteenths of a workspace nobody addresses, and the reverse
# is a launch nobody sized.


def _split_set(rows=4, width=512, block=32, splits=None):
    if splits is None:
        splits = plan_splits(DecodeBuckets(rows, width).rows, 8, width, block)
    return DecodeBuckets(rows, width, streamed=True, block=block, splits=splits)


def _split_read(block=32, splits=4, n_q=8, head_dim=8, rows=4, width=512):
    read = PagedRead(SPLIT, block=block)
    read.attach(
        allocate_partials(
            max_rows=rows, n_q=n_q, head_dim=head_dim, context_width=width,
            block=block, splits=splits,
        )
    )
    return read


def test_a_split_set_needs_the_width_axis_gone_first():
    """A split partitions the streamed read's tiles, so a set that still buckets the
    width and also names a chunk count is describing two reads at once."""
    with pytest.raises(ValueError, match="partition"):
        DecodeBuckets(4, 512, splits=4)


def test_a_split_set_and_a_split_read_that_agree_pass():
    check_read_matches(_split_set(splits=4), _split_read(splits=4))


def test_a_split_set_under_a_plain_streamed_read_is_refused():
    with pytest.raises(BucketsUnsound, match="nobody addresses"):
        check_read_matches(_split_set(splits=4), PagedRead(STREAMED, block=32))


def test_a_split_read_under_a_plain_streamed_set_is_refused():
    buckets = DecodeBuckets(4, 512, streamed=True, block=32)

    with pytest.raises(BucketsUnsound, match="nobody sized"):
        check_read_matches(buckets, _split_read(splits=4))


def test_a_split_read_that_never_got_its_arena_fails_the_set_it_was_planned_for():
    """Distinct from a count mismatch, and the message has to be, because the fix is
    different: one is a plan that disagrees with itself and the other is a boot path
    that stopped one call short."""
    with pytest.raises(BucketsUnsound, match="no workspace"):
        check_read_matches(_split_set(splits=4), PagedRead(SPLIT, block=32))


def test_two_chunk_counts_that_disagree_are_refused():
    with pytest.raises(BucketsUnsound, match="chunks"):
        check_read_matches(_split_set(splits=4), _split_read(splits=2))


def test_a_split_set_says_what_it_is_priced_in():
    line = _split_set(splits=4).render()

    assert "4 splits" in line


def test_a_split_cache_builds_both_halves_from_one_flag():
    cfg, cache = _prefilled(
        [[1, 2, 3, 4]], bucket_decode=True, split_read=True, read_block=8,
        max_model_len=64,
    )

    assert cache.decode_buckets.streamed is True
    assert cache.decode_buckets.splits == cache.read_splits > 0
    cache.allocate_split_workspace()
    check_read_matches(cache.decode_buckets, cache.read)


def test_check_capture_ready_runs_the_split_gate_too():
    cfg, cache = _prefilled(
        [[1, 2, 3, 4]], bucket_decode=True, persist_inputs=True, split_read=True,
        read_block=8, max_model_len=64,
    )

    with pytest.raises(BucketsUnsound, match="no workspace"):
        check_capture_ready(cache)
