"""Day 55 tests: the capture list recorded before the server accepts anything.

Day 54 got the decode step into a graph and did it in the wrong place. The first
sighting of every shape takes a recording, mid-run, in front of a client who is
waiting for a token, and a run walks the width axis for as long as it generates, so
the stall comes back every `width_multiple` tokens. vLLM records its list at
startup off dummy batches. This is that.

The whole day turns on one question: what is a decode batch that presents a shape
and owns nothing? The answer was already written on Day 52 and I did not see it. A
*padded* row reads nothing (`context_lens` 0) and writes to the sink slot, which is
one past the pool the allocator hands out. So a warm batch is a step whose rows are
**all** padding: `rows=()`, `pad_rows=graph_rows`, every write at the sink. It
touches no sequence's slots because there is no sequence.

Four claims:

  1. **A warm batch names no rows and writes nowhere.** `warmup_plan` builds a plan
     with an empty `rows` tuple, and every one of Day 54's six preconditions passes
     on it unchanged, including `check_pad_inert`, whose "no real row writes to the
     sink" clause is vacuous when there are no real rows.
  2. **A graph recorded on a warm batch replays real steps correctly**, and that is
     not luck: every tensor the recorded call holds is a window on storage a real
     step writes through, so the frozen plan reads this step's numbers. The engine
     that warms produces the same tokens as the engine that does not.
  3. **A warm graph is unbound.** Day 54's `check_replay_rows` exists because a
     mid-run capture freezes step 1's row tuple. A warm capture freezes the empty
     tuple, so there is no set of sequences it can be wrong about. Warming up front
     deletes the one field that was not storage.
  4. **The stall is a rate, and the budget has a number now.** `lazy_captures` says
     how many recordings a lazy run takes (one per width bucket it crosses),
     `stall_seconds` prices them into somebody's inter-token latency, and
     `warm_budget_bytes` is Day 54's `check_pool_budget` finally connected to
     `nanoserve.launch`'s memory probe.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.batch import pad_prompts
from nanoserve.buckets import DecodeBuckets, check_pad_inert
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.captured import (
    ACTIVATION_ITEMSIZE,
    CaptureUnsound,
    CapturedDecode,
    check_capture_preconditions,
    eager_recorder,
    workspace_bytes,
)
from nanoserve.compiled import DecodeShape
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine, Request
from nanoserve.inputs import check_step_inputs_persistent
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.plan import check_plan_addressing, plan_decode
from nanoserve.sampling import SamplingParams
from nanoserve.slots import check_mapping_is_window
from nanoserve.warmup import (
    DEFAULT_WARM_TOKEN,
    WarmupReport,
    WarmupUnsound,
    check_all_warm,
    check_no_cold_captures,
    check_table_stable,
    check_warm_budget,
    check_warm_graphs_unbound,
    check_warm_rows_empty,
    check_warm_touches_nothing,
    check_warm_writes_sink,
    lazy_captures,
    stall_seconds,
    warm_budget_bytes,
    warm_decode,
    warm_shapes,
    warmup_batch,
    warmup_plan,
    warmup_seconds,
    width_ceiling,
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


def _cache(batch_size=4, *, num_blocks=32, block_size=4, width_multiple=4, **kwargs):
    """An empty batched cache with both of Day 55's preconditions switched on.

    The bucket set is rebuilt with a small `width_multiple` so a tiny cache has more
    than one width bucket to warm: the default is 128, which over a 128-token table
    is a capture list of exactly one column.
    """
    cfg = _tiny_config()
    kwargs.setdefault("bucket_decode", True)
    kwargs.setdefault("persist_inputs", True)
    cache = BatchedPagedKVCache(
        cfg, BlockAllocator(num_blocks=num_blocks, block_size=block_size), batch_size, **kwargs
    )
    if cache.decode_buckets is not None:
        cache.decode_buckets = DecodeBuckets(
            batch_size, cache.max_model_len, width_multiple=width_multiple
        )
    return cfg, cache


def _prefilled(prompts, *, num_blocks=32, block_size=4, batch_size=None, seed=1, **kwargs):
    """A batched cache holding one prefill per row, ready to decode."""
    cfg, cache = _cache(
        batch_size if batch_size is not None else len(prompts),
        num_blocks=num_blocks,
        block_size=block_size,
        **kwargs,
    )
    batch = pad_prompts(list(prompts), pad_id=0)
    for layer in range(cfg.num_hidden_layers):
        k, v = _kv(cfg, len(prompts), batch.max_length, seed=seed + layer)
        cache.write(layer, k, v, batch.attention_mask, rows=tuple(range(len(prompts))))
    return cfg, cache


def _buckets(cache) -> DecodeBuckets:
    return cache.decode_buckets


def _step(cache, rows=None):
    """One planned decode step's arguments: `(input_ids, plan, view)`."""
    rows = tuple(range(cache.batch_size)) if rows is None else tuple(rows)
    plan = plan_decode(cache, rows=rows)
    view = cache.view(rows, plan=plan)
    ids = cache.decode_inputs.set_input_ids([1] * plan.batch_size, window=plan.graph_rows)
    return ids, plan, view


def _double(t, *args, **kwargs):
    return t * 2


# --- what a warm batch is ------------------------------------------------------------


def test_a_warm_plan_names_no_rows():
    """The day's whole trick: a batch that is all padding owns nothing."""
    _, cache = _cache(4)
    shape = DecodeShape(rows=2, context_width=cache.decode_buckets.width_bucket(8))

    plan = warmup_plan(cache, shape)

    assert plan.rows == ()
    assert plan.batch_size == 0
    assert plan.pad_rows == 2
    assert plan.graph_rows == 2


def test_a_warm_plan_presents_exactly_the_shape_it_was_asked_for():
    _, cache = _cache(8)
    shape = DecodeShape(rows=4, context_width=cache.decode_buckets.width_bucket(30))

    plan = warmup_plan(cache, shape)

    assert plan.graph_rows == shape.rows
    assert plan.graph_width == shape.context_width
    assert tuple(plan.slot_mapping.shape) == (shape.rows, shape.context_width)


def test_every_warm_row_writes_to_the_sink_and_reads_nothing():
    _, cache = _cache(4)
    shape = DecodeShape(rows=4, context_width=cache.decode_buckets.width_bucket(5))

    plan = warmup_plan(cache, shape)

    assert plan.sink_slot == cache.sink_slot
    assert plan.write_list == [cache.sink_slot] * 4
    assert plan.context_list == [0] * 4


def test_a_warm_rectangle_is_a_window_on_the_persistent_table():
    _, cache = _cache(4)
    shape = DecodeShape(rows=2, context_width=cache.decode_buckets.width_bucket(9))

    plan = warmup_plan(cache, shape)

    assert cache.slot_table.is_window(plan.slot_mapping)
    check_mapping_is_window(cache.slot_table, plan)


def test_a_warm_plan_writes_its_inputs_into_the_persistent_buffers():
    _, cache = _cache(4)
    shape = DecodeShape(rows=2, context_width=cache.decode_buckets.width_bucket(9))

    batch = warmup_batch(cache, shape)

    check_step_inputs_persistent(batch.input_ids, batch.plan, cache.decode_inputs)


def test_a_warm_batch_forwards_one_token_per_row():
    _, cache = _cache(4)
    shape = DecodeShape(rows=2, context_width=cache.decode_buckets.width_bucket(9))

    batch = warmup_batch(cache, shape)

    assert tuple(batch.input_ids.shape) == (2, 1)
    assert batch.input_ids.tolist() == [[DEFAULT_WARM_TOKEN], [DEFAULT_WARM_TOKEN]]
    assert batch.view.rows == ()
    assert batch.rows == 0


def test_a_warm_shape_wider_than_the_table_is_refused():
    _, cache = _cache(4)

    with pytest.raises(WarmupUnsound, match="max_model_len"):
        warmup_plan(cache, DecodeShape(rows=2, context_width=cache.slot_table.max_model_len + 1))


def test_a_warm_shape_with_more_rows_than_the_cache_is_refused():
    _, cache = _cache(4)

    with pytest.raises(WarmupUnsound, match="rows"):
        warmup_plan(cache, DecodeShape(rows=5, context_width=4))


def test_a_warm_plan_needs_a_cache_with_the_last_two_days_switched_on():
    _, cache = _cache(4, bucket_decode=False, persist_inputs=False)

    with pytest.raises(CaptureUnsound):
        warmup_plan(cache, DecodeShape(rows=2, context_width=4))


# --- the payoff: Day 54's gates pass on a batch made of nothing ----------------------


def test_every_capture_precondition_passes_on_a_warm_batch():
    """The point of writing the six gates as functions, one day later."""
    _, cache = _cache(4)
    shape = DecodeShape(rows=4, context_width=cache.decode_buckets.width_bucket(7))

    batch = warmup_batch(cache, shape)

    check_capture_preconditions(batch.input_ids, batch.plan, batch.view)


def test_pad_inert_is_vacuous_on_a_warm_plan_rather_than_special_cased():
    """Its third clause is "no real row writes to the sink", and there are none."""
    _, cache = _cache(4)
    shape = DecodeShape(rows=2, context_width=cache.decode_buckets.width_bucket(7))

    check_pad_inert(warmup_plan(cache, shape))


def test_plan_addressing_passes_on_a_warm_plan():
    _, cache = _cache(4)
    shape = DecodeShape(rows=2, context_width=cache.decode_buckets.width_bucket(7))

    check_plan_addressing(warmup_plan(cache, shape))


# --- warming touches nothing ---------------------------------------------------------


def test_warming_grows_no_row():
    cfg, cache = _prefilled([[1, 2, 3], [4, 5]], batch_size=4)
    before = list(cache.seq_lens)

    warm_decode(_capture(cfg), cache, warm_shapes(_buckets(cache), max_width=8))

    assert cache.seq_lens == before


def test_warming_takes_no_block_out_of_the_pool():
    cfg, cache = _prefilled([[1, 2, 3]], batch_size=4)
    free = cache.allocator.num_free

    warm_decode(_capture(cfg), cache, warm_shapes(_buckets(cache), max_width=8))

    assert cache.allocator.num_free == free


def test_warming_writes_only_to_the_sink_slot():
    """The correctness claim of the day, read straight off the pool."""
    cfg, cache = _prefilled([[1, 2, 3], [4, 5]], batch_size=4)
    _, plan, view = _step(cache, rows=(0, 1))
    model, _ = _model()
    model.forward(cache.decode_inputs.set_input_ids([1, 1], window=plan.graph_rows),
                  plan.positions, cache=view)
    before = [pool.clone() for pool in cache.k_pool]

    warm_decode(_capture(cfg, model=model), cache, warm_shapes(_buckets(cache), max_width=8))

    for was, now in zip(before, cache.k_pool):
        assert torch.equal(was[: cache.sink_slot], now[: cache.sink_slot])


def test_check_warm_touches_nothing_catches_a_row_that_moved():
    _, cache = _prefilled([[1, 2, 3]], batch_size=4)
    before = tuple(cache.seq_lens)
    plan_decode(cache, rows=(0,))

    with pytest.raises(WarmupUnsound, match="grew"):
        check_warm_touches_nothing(cache, before)


def test_check_warm_writes_sink_catches_a_real_write_slot():
    _, cache = _prefilled([[1, 2, 3]], batch_size=4)
    _, plan, _ = _step(cache, rows=(0,))

    with pytest.raises(WarmupUnsound, match="sink"):
        check_warm_writes_sink(plan)


def test_check_warm_rows_empty_catches_a_real_plan():
    _, cache = _prefilled([[1, 2, 3]], batch_size=4)
    _, plan, _ = _step(cache, rows=(0,))

    with pytest.raises(WarmupUnsound, match="names rows"):
        check_warm_rows_empty(plan)


# --- the list ------------------------------------------------------------------------


def test_warm_shapes_is_the_bucket_set():
    buckets = DecodeBuckets(4, 16, row_buckets=(1, 2, 4), width_multiple=8)

    assert set(warm_shapes(buckets)) == set(buckets.shapes)


def test_warm_shapes_records_the_biggest_first():
    """A shared pool is sized by its largest capture, so record that one first and
    every later graph fits inside an arena that is already the right size."""
    buckets = DecodeBuckets(4, 16, row_buckets=(1, 2, 4), width_multiple=8)

    shapes = warm_shapes(buckets)

    assert shapes[0] == DecodeShape(rows=4, context_width=16)
    assert [s.cells for s in shapes] == sorted((s.cells for s in shapes), reverse=True)


def test_warm_shapes_trims_to_what_a_server_will_present():
    buckets = DecodeBuckets(8, 32, row_buckets=(1, 2, 4, 8), width_multiple=8)

    shapes = warm_shapes(buckets, max_rows=2, max_width=16)

    assert {s.rows for s in shapes} == {1, 2}
    assert {s.context_width for s in shapes} == {8, 16}


def test_warm_shapes_refuses_an_empty_list():
    buckets = DecodeBuckets(8, 32, row_buckets=(1, 2, 4, 8), width_multiple=8)

    with pytest.raises(WarmupUnsound, match="no shape"):
        warm_shapes(buckets, max_width=4)


# --- warming a capture ---------------------------------------------------------------


def _capture(cfg, *, model=None, mode="capture") -> CapturedDecode:
    model = model if model is not None else _model()[0]
    return CapturedDecode(model.forward, mode=mode, recorder=eager_recorder, warmup=0)


def test_warming_records_one_graph_per_shape():
    cfg, cache = _cache(4)
    captured = _capture(cfg)
    shapes = warm_shapes(_buckets(cache), max_width=8)

    report = warm_decode(captured, cache, shapes)

    assert captured.count == len(shapes)
    assert report.captures == len(shapes)
    assert report.cold == ()


def test_warming_is_idempotent():
    cfg, cache = _cache(4)
    captured = _capture(cfg)
    shapes = warm_shapes(_buckets(cache), max_width=8)
    warm_decode(captured, cache, shapes)

    again = warm_decode(captured, cache, shapes)

    assert again.captures == 0
    assert captured.count == len(shapes)


def test_a_warmed_engine_replays_its_first_real_step():
    """The whole point: no client ever waits for a recording."""
    cfg, cache = _prefilled([[1, 2, 3], [4, 5]], batch_size=4)
    captured = _capture(cfg)
    warmed = warm_decode(captured, cache, warm_shapes(_buckets(cache), max_width=8))

    ids, _, view = _step(cache, rows=(0, 1))
    captured(ids, view.plan.positions, cache=view)

    assert captured.captures == warmed.captures
    check_no_cold_captures(captured, warmed=warmed.captures)


def test_a_real_step_outside_the_warmed_list_is_a_cold_capture():
    cfg, cache = _prefilled([[1, 2, 3], [4, 5]], batch_size=4)
    captured = _capture(cfg)
    warmed = warm_decode(captured, cache, warm_shapes(_buckets(cache), max_rows=1, max_width=8))

    ids, _, view = _step(cache, rows=(0, 1))
    captured(ids, view.plan.positions, cache=view)

    with pytest.raises(WarmupUnsound, match="after the list was warmed"):
        check_no_cold_captures(captured, warmed=warmed.captures)


def test_check_all_warm_names_the_shape_nobody_recorded():
    cfg, cache = _cache(4)
    captured = _capture(cfg)
    shapes = warm_shapes(_buckets(cache), max_width=8)
    warm_decode(captured, cache, shapes[1:])

    with pytest.raises(WarmupUnsound, match="never recorded"):
        check_all_warm(captured, shapes)


def test_warming_a_capture_that_is_off_is_refused():
    cfg, cache = _cache(4)

    with pytest.raises(ValueError, match="off"):
        warm_decode(_capture(cfg, mode="off"), cache, warm_shapes(_buckets(cache), max_width=8))


def test_a_report_prices_the_walk():
    cfg, cache = _cache(4)
    clock = iter([1.0, 1.5]).__next__
    shapes = warm_shapes(_buckets(cache), max_width=8)

    report = warm_decode(_capture(cfg), cache, shapes, clock=clock)

    assert isinstance(report, WarmupReport)
    assert report.seconds == pytest.approx(0.5)
    assert report.per_capture_s == pytest.approx(0.5 / len(shapes))
    assert "graphs" in report.render()


# --- a warm graph is unbound ---------------------------------------------------------


def test_a_warm_graph_carries_no_row_tuple():
    """Day 54's one non-storage field, emptied by construction."""
    cfg, cache = _cache(4)
    captured = _capture(cfg)

    warm_decode(captured, cache, warm_shapes(_buckets(cache), max_width=8))

    assert all(graph.rows == () for graph in captured.graphs.values())
    check_warm_graphs_unbound(captured)


def test_check_warm_graphs_unbound_catches_a_graph_recorded_mid_run():
    cfg, cache = _prefilled([[1, 2, 3], [4, 5]], batch_size=4)
    captured = _capture(cfg)
    ids, _, view = _step(cache, rows=(0, 1))
    captured(ids, view.plan.positions, cache=view)

    with pytest.raises(WarmupUnsound, match="rows"):
        check_warm_graphs_unbound(captured)


# --- a warm graph replays real steps -------------------------------------------------


def test_a_graph_recorded_on_nothing_computes_a_real_step():
    """Not luck. Every tensor the recorded call holds is a window a real step writes
    through, so the frozen plan reads this step's numbers."""
    cfg, cache = _prefilled([[1, 2, 3], [4, 5]], batch_size=4, seed=3)
    model, _ = _model()
    captured = _capture(cfg, model=model)
    warm_decode(captured, cache, warm_shapes(_buckets(cache), max_width=8))

    ids, plan, view = _step(cache, rows=(0, 1))
    replayed = captured(ids, plan.positions, cache=view).clone()
    expected = model.forward(ids, plan.positions, cache=view)

    assert torch.allclose(replayed, expected, atol=1e-5)


def test_a_warmed_engine_generates_what_an_unwarmed_one_does():
    model, _ = _model(seed=5)
    prompt = [3, 1, 4, 1, 5]

    plain = _run(Engine.build(model, num_blocks=64, block_size=4, max_batch_size=4), prompt)
    warm = Engine.build(
        model,
        num_blocks=64,
        block_size=4,
        max_batch_size=4,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
    )
    warm.warm_decode()

    assert _run(warm, prompt) == plain


def _run(engine: Engine, prompt, max_tokens=6) -> list[int]:
    request = engine.add_request(
        Request("r", prompt, max_new_tokens=max_tokens, sampling=SamplingParams(temperature=0.0))
    )
    while engine.has_unfinished():
        engine.step()
    return list(request.output_token_ids)


# --- the engine's own door -----------------------------------------------------------


def test_the_engine_warms_its_whole_bucket_set():
    model, _ = _model()
    engine = Engine.build(
        model,
        num_blocks=16,
        block_size=4,
        max_batch_size=2,
        max_model_len=16,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
    )

    report = engine.warm_decode()

    assert report.captures == engine.decode_graphs.count == len(report.shapes)
    check_all_warm(engine.decode_graphs, report.shapes)


def test_an_engine_with_no_capture_refuses_to_warm():
    model, _ = _model()
    engine = Engine.build(model, num_blocks=16, block_size=4, max_batch_size=2)

    with pytest.raises(ValueError, match="capture_decode"):
        engine.warm_decode()


def test_a_warmed_engine_records_nothing_while_it_serves():
    model, _ = _model()
    engine = Engine.build(
        model,
        num_blocks=32,
        block_size=4,
        max_batch_size=2,
        max_model_len=16,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
    )
    report = engine.warm_decode()

    engine.add_request(
        Request("r", [1, 2, 3], max_new_tokens=5, sampling=SamplingParams(temperature=0.0))
    )
    while engine.has_unfinished():
        engine.step()

    check_no_cold_captures(engine.decode_graphs, warmed=report.captures)


# --- the table nobody was guarding ---------------------------------------------------


def test_check_table_stable_catches_a_table_that_moved():
    """Day 54 guards the four input buffers and not the 16 MB table the rectangle is
    a window on. Warming is what makes the hole reachable: it is the first thing in
    the process to hand out a window, so it is the first thing that can hand one out
    on the wrong device."""
    _, cache = _cache(4)
    table = cache.slot_table
    address = table.address
    table.slots = table.slots.clone()

    with pytest.raises(WarmupUnsound, match="slot table"):
        check_table_stable(table, address)


def test_a_table_that_did_not_move_passes():
    _, cache = _cache(4)

    warm_decode(_capture(_tiny_config()), cache, warm_shapes(_buckets(cache), max_width=8))

    check_table_stable(cache.slot_table, cache.slot_table.address)


# --- what a lazy capture costs -------------------------------------------------------


def test_a_short_run_inside_one_width_bucket_records_once():
    assert lazy_captures(steps=64, start_width=16, width_multiple=128) == 1


def test_a_run_that_crosses_a_width_bucket_records_again():
    """Day 54's capturebench, exactly: a 16-token prompt and 128 generated tokens
    reaches 144, which is over the 128 boundary, so it takes two recordings."""
    assert lazy_captures(steps=128, start_width=16, width_multiple=128) == 2


def test_a_long_run_records_once_per_width_multiple():
    assert lazy_captures(steps=1024, start_width=0, width_multiple=128) == 8


def test_a_stall_is_priced_into_somebody_s_inter_token_latency():
    assert stall_seconds(
        steps=1024, start_width=0, width_multiple=128, per_capture_s=0.05
    ) == pytest.approx(0.4)


def test_warming_up_front_pays_for_every_shape_and_stalls_nobody():
    assert warmup_seconds(36, 0.05) == pytest.approx(1.8)


def test_a_run_takes_at_least_one_step():
    with pytest.raises(ValueError, match="at least one step"):
        lazy_captures(steps=0, start_width=0, width_multiple=128)


# --- the budget, finally connected ---------------------------------------------------


def test_the_budget_is_what_is_free_after_the_weights_and_the_pool():
    free, total = 8 << 30, 24 << 30

    budget = warm_budget_bytes(
        "cuda:0", reserved_bytes=1 << 30, utilization=0.90, probe=lambda d: (free, total)
    )

    assert budget == free - int(total * 0.10) - (1 << 30)


def test_the_budget_cannot_be_probed_on_a_cpu():
    with pytest.raises(RuntimeError, match="no VRAM"):
        warm_budget_bytes("cpu")


def test_a_budget_picks_a_width_ceiling():
    """A shared pool is sized by its largest shape, so the budget is a statement
    about one number: how wide the widest capture may be."""
    budget = workspace_bytes(DecodeShape(rows=8, context_width=2048), 32)

    assert width_ceiling(rows=8, num_heads=32, budget_bytes=budget) == 2048


def test_a_budget_too_small_for_one_column_is_a_ceiling_of_zero():
    assert width_ceiling(rows=8, num_heads=32, budget_bytes=1) == 0


def test_check_warm_budget_refuses_a_list_whose_widest_shape_does_not_fit():
    shapes = (DecodeShape(rows=8, context_width=8192),)
    free, total = 1 << 30, 24 << 30

    with pytest.raises(CaptureUnsound, match="capture pool"):
        check_warm_budget(
            shapes, 32, device="cuda:0", probe=lambda d: (free, total), reserved_bytes=free
        )


def test_check_warm_budget_passes_a_list_that_fits():
    shapes = (DecodeShape(rows=2, context_width=128),)
    free, total = 8 << 30, 24 << 30

    check_warm_budget(shapes, 32, device="cuda:0", probe=lambda d: (free, total))


def test_the_widest_shape_is_the_one_the_budget_is_about():
    small = DecodeShape(rows=1, context_width=128)
    wide = DecodeShape(rows=8, context_width=8192)

    assert workspace_bytes(wide, 32, ACTIVATION_ITEMSIZE) > workspace_bytes(
        small, 32, ACTIVATION_ITEMSIZE
    )
