"""Day 54 tests: the decode step recorded once and replayed, and the pool it costs.

Five days have been spent making a decode step something a capture can hold. Day 49
removed the graph breaks, Day 50 moved the addressing out of the forward, Day 51 made
the read rectangle a window on a persistent table, Day 52 closed the shape into a
bucket set, Day 53 put the four inputs into buffers that do not move. Every one of
those days ended in a gate, written as a function rather than as a comment, and this
is the day that spends them: they are the preconditions a capture checks before it
records anything.

What a recorded graph is, in one sentence: a list of kernel launches bound to the
addresses they were launched with. Replaying takes no arguments. It re-runs those
kernels over whatever those addresses now hold, and writes into whatever address the
output was at.

Four claims:

  1. **A shape is recorded once and replayed after that.** `captures` counts the
     first sighting of each bucketed shape and stops growing; `replays` counts every
     step after. On an unbucketed run the two are the same number, which is the
     failure the last two days existed to prevent.
  2. **The preconditions are the five days, run in order.** A step that is not
     bucketed, whose rectangle is a fresh gather, whose inputs are their own
     allocations, or whose plan carries no snapshot, is refused before a graph
     exists rather than found out afterwards in the text.
  3. **The output is a fixed buffer too, and that half is the one nobody warns you
     about.** Every replay writes into the same storage, so a captured step's result
     has a lifetime of exactly one step. `check_output_not_held` is that stated as a
     gate, and Day 48's deferred window is the thing it is aimed at.
  4. **One memory pool, shared by every graph, and it is not the saving it looks
     like.** The capture list is geometric, so the biggest shape is most of the bill:
     sharing the pool over 36 shapes saves about 5x, not 36x. The number that
     actually hurts is the absolute one, and it comes from the reference read
     materialising a `[rows, heads, 1, ctx]` score rectangle rather than from
     anything the capture does.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nanoserve.batch import pad_prompts
from nanoserve.buckets import BucketsUnsound, DecodeBuckets
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.captured import (
    ACTIVATION_ITEMSIZE,
    DEFAULT_CAPTURE_LIMIT,
    DEFAULT_WARMUP,
    CaptureUnsound,
    CapturedDecode,
    CapturedGraph,
    capture_breakeven_steps,
    capture_cost_s,
    check_all_shapes_captured,
    check_capture_preconditions,
    check_capture_ready,
    check_output_not_held,
    check_pool_budget,
    check_pool_shared,
    check_replay_rows,
    check_replays_dominate,
    eager_recorder,
    launch_saving_s,
    pool_sharing_ratio,
    private_pool_bytes,
    score_cells,
    shared_pool_bytes,
    workspace_bytes,
)
from nanoserve.compiled import DecodeShape
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine, Request
from nanoserve.inputs import InputsUnsound
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.plan import plan_decode
from nanoserve.slots import SlotsUnsound


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


def _prefilled(prompts, *, num_blocks=32, block_size=4, batch_size=None, seed=1, **kwargs):
    """A batched cache holding one prefill per row, ready to decode."""
    cfg = _tiny_config()
    batch_size = batch_size if batch_size is not None else len(prompts)
    cache = BatchedPagedKVCache(
        cfg,
        BlockAllocator(num_blocks=num_blocks, block_size=block_size),
        batch_size,
        **kwargs,
    )
    batch = pad_prompts(list(prompts), pad_id=0)
    for layer in range(cfg.num_hidden_layers):
        k, v = _kv(cfg, len(prompts), batch.max_length, seed=seed + layer)
        cache.write(layer, k, v, batch.attention_mask, rows=tuple(range(len(prompts))))
    return cfg, cache


def _ready(prompts, **kwargs):
    """A model and a cache with every precondition of a capture already switched on."""
    model, _ = _model()
    _, cache = _prefilled(prompts, bucket_decode=True, persist_inputs=True, **kwargs)
    return model, cache


def _step(cache, rows=None):
    """One planned decode step's arguments: `(input_ids, plan, view)`."""
    rows = tuple(range(cache.batch_size)) if rows is None else tuple(rows)
    plan = plan_decode(cache, rows=rows)
    view = cache.view(rows, plan=plan)
    ids = cache.decode_inputs.set_input_ids([1] * plan.batch_size, window=plan.graph_rows)
    return ids, plan, view


def _double(t, *args, **kwargs):
    """A stand-in forward: one tensor in, one tensor out, deterministic."""
    return t * 2


# --- what a recorded graph is -------------------------------------------------------


def test_a_recorder_hands_back_a_replay_an_output_and_a_pool():
    buffer = torch.tensor([1, 2, 3, 4])

    replay, output, pool = eager_recorder(_double, (buffer,), {}, pool=None, warmup=1)

    assert callable(replay)
    assert output.tolist() == [2, 4, 6, 8]
    assert pool is not None


def test_a_replay_re_reads_the_buffer_it_was_recorded_against():
    """The whole property, in five lines. Nothing is passed to `replay()`."""
    buffer = torch.tensor([1, 2, 3, 4])
    replay, output, _ = eager_recorder(_double, (buffer,), {}, pool=None, warmup=1)

    buffer.copy_(torch.tensor([5, 5, 5, 5]))
    replay()

    assert output.tolist() == [10, 10, 10, 10]


def test_a_replay_cannot_see_a_tensor_that_was_replaced_instead_of_written():
    """The failure the last two days exist to prevent, made to happen on purpose.

    Rebinding the name is what a per-step build does: the old storage is still what
    the graph holds, so the replay computes last step's answer and nothing raises."""
    buffer = torch.tensor([1, 2, 3, 4])
    replay, output, _ = eager_recorder(_double, (buffer,), {}, pool=None, warmup=1)

    buffer = torch.tensor([5, 5, 5, 5])  # noqa: F841 - the point is that it is ignored
    replay()

    assert output.tolist() == [2, 4, 6, 8]


def test_every_replay_writes_into_the_same_output_storage():
    buffer = torch.tensor([1, 2, 3, 4])
    replay, output, _ = eager_recorder(_double, (buffer,), {}, pool=None, warmup=1)
    before = output.data_ptr()

    for _ in range(5):
        replay()

    assert output.data_ptr() == before


def test_a_recorder_reuses_the_pool_it_is_handed():
    first, _, pool = eager_recorder(_double, (torch.tensor([1]),), {}, pool=None, warmup=1)
    _, _, again = eager_recorder(_double, (torch.tensor([2]),), {}, pool=pool, warmup=1)

    assert again is pool
    assert callable(first)


def test_a_recorder_warms_up_before_it_records():
    """Not decoration: the first call into a kernel allocates workspaces and picks
    algorithms, and recording that records the allocation into the graph."""
    calls = []

    def counted(t):
        calls.append(1)
        return t * 2

    eager_recorder(counted, (torch.tensor([1]),), {}, pool=None, warmup=3)

    assert len(calls) == 4  # three warmups plus the recorded call


def test_a_captured_graph_counts_its_replays_and_hands_back_the_output():
    buffer = torch.tensor([1, 2, 3])
    replay, output, pool = eager_recorder(_double, (buffer,), {}, pool=None, warmup=1)
    graph = CapturedGraph(
        shape=DecodeShape(rows=3, context_width=8),
        replay=replay,
        output=output,
        input_addresses=(buffer.data_ptr(),),
        pool=pool,
    )

    result = graph.run()

    assert result is output
    assert graph.replays == 1
    assert "3 x 8" in graph.render()


def test_a_captured_graph_knows_its_own_output_storage():
    buffer = torch.tensor([1, 2, 3])
    replay, output, pool = eager_recorder(_double, (buffer,), {}, pool=None, warmup=1)
    graph = CapturedGraph(
        shape=DecodeShape(rows=3, context_width=8),
        replay=replay,
        output=output,
        input_addresses=(),
        pool=pool,
    )

    assert graph.owns(output)
    assert graph.owns(output[:2])
    assert not graph.owns(output.clone())


# --- capturing a decode step --------------------------------------------------------


def test_capture_off_hands_the_call_straight_through():
    captured = CapturedDecode(_double, mode="off")

    assert captured(torch.tensor([2])).tolist() == [4]
    assert captured.captures == 0
    assert captured.calls == 1


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        CapturedDecode(_double, mode="sometimes")


def test_the_first_step_of_a_shape_is_recorded():
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    ids, _, view = _step(cache)

    captured(ids, view.plan.positions, cache=view)

    assert captured.captures == 1
    assert captured.count == 1


def test_the_second_step_of_the_same_shape_is_replayed():
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)

    for _ in range(6):
        ids, _, view = _step(cache)
        captured(ids, view.plan.positions, cache=view)

    assert captured.captures == 1
    assert captured.replays == 6  # the recording step replays once too
    assert captured.count == 1


def test_a_bucketed_run_records_one_graph_and_an_unbucketed_one_records_every_step():
    """The two days' worth of shape work, priced in graphs."""
    model, _ = _model()
    counts = {}
    for bucketed in (False, True):
        _, cache = _prefilled(
            [[1, 2, 3], [4]], bucket_decode=bucketed, persist_inputs=True, num_blocks=64
        )
        captured = CapturedDecode(
            model.forward,
            mode="capture",
            recorder=eager_recorder,
            precheck=False,
        )
        for _ in range(6):
            ids, _, view = _step(cache)
            captured(ids, view.plan.positions, cache=view)
        counts[bucketed] = captured.captures

    assert counts[True] == 1
    assert counts[False] == 6


def test_a_run_past_the_capture_limit_is_refused():
    model, _ = _model()
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True, num_blocks=64)
    captured = CapturedDecode(
        model.forward,
        mode="capture",
        recorder=eager_recorder,
        precheck=False,
        limit=3,
    )

    with pytest.raises(CaptureUnsound, match="limit of 3"):
        for _ in range(6):
            ids, _, view = _step(cache)
            captured(ids, view.plan.positions, cache=view)


def test_a_step_with_no_plan_falls_through_to_eager():
    """A prefill, or a caller who forgot. Not an error and not a capture: there is
    nothing fixed about an unplanned step's addressing to record."""
    captured = CapturedDecode(_double, mode="capture", recorder=eager_recorder)

    out = captured(torch.tensor([[2]]))

    assert out.tolist() == [[4]]
    assert captured.captures == 0
    assert captured.eager_calls == 1


def test_a_replay_returns_the_graphs_own_output_buffer():
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)

    ids, _, view = _step(cache)
    first = captured(ids, view.plan.positions, cache=view)
    ids, _, view = _step(cache)
    second = captured(ids, view.plan.positions, cache=view)

    assert first.data_ptr() == second.data_ptr()


def test_a_replayed_step_computes_what_an_eager_one_would():
    """The equality the day rests on. Same plan, same buffers, same logits."""
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)

    ids, _, view = _step(cache)
    captured(ids, view.plan.positions, cache=view)
    ids, _, view = _step(cache)
    replayed = captured(ids, view.plan.positions, cache=view).clone()
    eager = model.forward(ids, view.plan.positions, cache=view)

    assert torch.allclose(replayed, eager, atol=1e-5)


# --- the pool -----------------------------------------------------------------------


def test_every_graph_shares_one_memory_pool():
    """Three shapes, one arena. The first capture makes the pool and every later one
    is handed it, which is the whole of vLLM's `graph_pool_handle` policy."""
    model, cache = _ready([[1, 2, 3], [4], [5, 6]], batch_size=4, num_blocks=64)
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)

    for rows in ((0,), (0, 1), (0, 1, 2)):
        ids, _, view = _step(cache, rows)
        captured(ids, view.plan.positions, cache=view)

    assert captured.count == 3
    check_pool_shared(captured)
    assert len({id(g.pool) for g in captured.graphs.values()}) == 1


def test_check_pool_shared_catches_a_graph_that_made_its_own():
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    ids, _, view = _step(cache)
    captured(ids, view.plan.positions, cache=view)
    graph = next(iter(captured.graphs.values()))
    captured.graphs[DecodeShape(rows=99, context_width=8)] = replace(graph, pool=object())

    with pytest.raises(CaptureUnsound, match="pool"):
        check_pool_shared(captured)


def test_check_pool_shared_is_happy_with_nothing_captured():
    check_pool_shared(CapturedDecode(_double, mode="off"))


# --- the preconditions --------------------------------------------------------------


def test_the_preconditions_pass_on_a_step_the_five_days_prepared():
    model, cache = _ready([[1, 2, 3], [4]])
    ids, plan, view = _step(cache)

    check_capture_preconditions(ids, plan, view)


def test_an_unbucketed_step_is_refused_before_anything_is_recorded():
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    ids, plan, view = _step(cache)

    with pytest.raises(CaptureUnsound, match="bucket"):
        check_capture_preconditions(ids, plan, view)


def test_a_step_over_freshly_built_inputs_is_refused():
    _, cache = _prefilled([[1, 2, 3], [4]], bucket_decode=True)
    plan = plan_decode(cache)
    view = cache.view((0, 1), plan=plan)
    ids = torch.tensor([[1], [1]])

    with pytest.raises(CaptureUnsound, match="buffer"):
        check_capture_preconditions(ids, plan, view)


def test_a_step_whose_tokens_are_not_a_window_is_refused():
    model, cache = _ready([[1, 2, 3], [4]])
    ids, plan, view = _step(cache)

    with pytest.raises(InputsUnsound, match="input_ids"):
        check_capture_preconditions(ids.clone(), plan, view)


def test_a_plan_with_a_freshly_gathered_rectangle_is_refused():
    model, cache = _ready([[1, 2, 3], [4]])
    ids, plan, view = _step(cache)
    gathered = replace(plan, slot_mapping=plan.slot_mapping.clone())

    with pytest.raises(SlotsUnsound, match="window"):
        check_capture_preconditions(ids, gathered, cache.view((0, 1), plan=gathered))


def test_a_plan_that_kept_no_snapshot_is_refused():
    model, cache = _ready([[1, 2, 3], [4]])
    ids, plan, view = _step(cache)
    stripped = replace(plan, context_snapshot=(), write_snapshot=())

    with pytest.raises(InputsUnsound, match="snapshot"):
        check_capture_preconditions(ids, stripped, cache.view((0, 1), plan=stripped))


def test_a_padded_row_that_names_no_sink_is_refused():
    """Day 52's correctness gate, doing its job as a precondition. Three rows round
    up to a bucket of four, and the invented row writes K/V whether anyone wants it
    to or not."""
    model, cache = _ready([[1, 2, 3], [4], [5, 6]], batch_size=4, num_blocks=64)
    ids, plan, view = _step(cache, (0, 1, 2))

    assert plan.pad_rows == 1
    check_capture_preconditions(ids, plan, view)

    lying = replace(plan, sink_slot=None)
    with pytest.raises(BucketsUnsound, match="sink"):
        check_capture_preconditions(ids, lying, cache.view((0, 1, 2), plan=lying))


def test_check_replay_rows_refuses_a_replay_over_a_different_batch():
    """The only thing a graph carries that is not storage. Everything else in the
    held plan is a window and says whatever this step wrote."""
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    ids, plan, view = _step(cache)
    captured(ids, view.plan.positions, cache=view)
    graph = next(iter(captured.graphs.values()))

    check_replay_rows(graph, plan)

    with pytest.raises(CaptureUnsound, match="recorded over rows"):
        check_replay_rows(graph, replace(plan, rows=(2, 3)))


def test_check_capture_ready_names_the_switch_that_is_off():
    _, plain = _prefilled([[1, 2]])
    with pytest.raises(CaptureUnsound, match="bucket_decode"):
        check_capture_ready(plain)

    _, half = _prefilled([[1, 2]], bucket_decode=True)
    with pytest.raises(CaptureUnsound, match="persist_inputs"):
        check_capture_ready(half)

    _, ready = _ready([[1, 2]])
    check_capture_ready(ready)


def test_the_capture_path_runs_the_preconditions_itself():
    model, _ = _model()
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True)
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    ids, _, view = _step(cache)

    with pytest.raises(CaptureUnsound, match="bucket"):
        captured(ids, view.plan.positions, cache=view)

    assert captured.captures == 0


def test_a_buffer_that_moved_between_capture_and_replay_is_caught():
    """The gate Day 53 wrote, doing the job it was written for. Nothing about a
    replay over freed storage raises on its own."""
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    ids, _, view = _step(cache)
    captured(ids, view.plan.positions, cache=view)

    cache.decode_inputs.positions.buffer = torch.zeros(2, 1, dtype=torch.long)
    ids, _, view = _step(cache)

    with pytest.raises(InputsUnsound, match="positions"):
        captured(ids, view.plan.positions, cache=view)


# --- the output is a buffer too -----------------------------------------------------


def test_a_captured_step_owns_the_logits_it_hands_back():
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    ids, _, view = _step(cache)

    logits = captured(ids, view.plan.positions, cache=view)

    assert captured.owns_output(logits)
    assert captured.owns_output(logits[:, -1])
    assert not captured.owns_output(logits.clone())


def test_check_output_not_held_refuses_a_value_carried_into_the_next_step():
    """The half nobody warns you about. Day 48 keeps sampled tokens on the device
    across step boundaries, and a tensor that is a graph's output buffer holds this
    step's answer for exactly as long as the next replay takes to start."""
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    ids, _, view = _step(cache)
    logits = captured(ids, view.plan.positions, cache=view)

    check_output_not_held(captured, logits.clone())

    with pytest.raises(CaptureUnsound, match="one step"):
        check_output_not_held(captured, logits)


def test_a_held_output_really_does_change_under_the_holder():
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    ids, _, view = _step(cache)
    held = captured(ids, view.plan.positions, cache=view)
    before = held.clone()

    ids, _, view = _step(cache)
    captured(ids, view.plan.positions, cache=view)

    assert not torch.allclose(held, before)


# --- what it costs ------------------------------------------------------------------


def test_score_cells_is_the_rectangle_the_reference_read_materialises():
    assert score_cells(DecodeShape(rows=4, context_width=128), num_heads=8) == 4 * 8 * 128


def test_workspace_bytes_prices_that_rectangle():
    shape = DecodeShape(rows=4, context_width=128)

    assert workspace_bytes(shape, 8) == 4 * 8 * 128 * ACTIVATION_ITEMSIZE
    assert workspace_bytes(shape, 8, itemsize=2) == 4 * 8 * 128 * 2


def test_workspace_bytes_refuses_nonsense():
    shape = DecodeShape(rows=4, context_width=128)
    with pytest.raises(ValueError, match="at least one head"):
        workspace_bytes(shape, 0)
    with pytest.raises(ValueError, match="at least one byte"):
        workspace_bytes(shape, 8, itemsize=0)


def test_a_shared_pool_is_sized_by_the_largest_shape_and_a_private_one_by_all_of_them():
    shapes = (
        DecodeShape(rows=1, context_width=128),
        DecodeShape(rows=4, context_width=128),
        DecodeShape(rows=4, context_width=256),
    )

    assert shared_pool_bytes(shapes, 8) == workspace_bytes(shapes[2], 8)
    assert private_pool_bytes(shapes, 8) == sum(workspace_bytes(s, 8) for s in shapes)


def test_pool_sharing_ratio_is_far_under_the_number_of_shapes():
    """The finding of the day's arithmetic. The capture list is geometric on both
    axes, so the largest shape is most of the bill and sharing the pool over 36 of
    them saves about 5x rather than 36x."""
    buckets = DecodeBuckets(256, 8192, width_multiple=2048)
    shapes = buckets.shapes

    assert buckets.count == 36
    assert 4.0 < pool_sharing_ratio(shapes, 32) < 6.0


def test_an_empty_capture_list_costs_nothing():
    assert shared_pool_bytes((), 8) == 0
    assert private_pool_bytes((), 8) == 0
    assert pool_sharing_ratio((), 8) == 1.0


def test_capture_cost_is_linear_in_the_graphs():
    assert capture_cost_s(36, 0.05) == pytest.approx(1.8)
    assert capture_cost_s(0, 0.05) == 0.0

    with pytest.raises(ValueError, match="not negative"):
        capture_cost_s(-1, 0.05)


def test_launch_saving_is_every_launch_but_one():
    """What a replay actually buys: one submission where there were n."""
    assert launch_saving_s(100, 5e-6) == pytest.approx(99 * 5e-6)
    assert launch_saving_s(1, 5e-6) == 0.0

    with pytest.raises(ValueError, match="at least one"):
        launch_saving_s(0, 5e-6)


def test_capture_breakeven_is_infinite_when_a_replay_saves_nothing():
    assert capture_breakeven_steps(1.8, 0.0) == float("inf")
    assert capture_breakeven_steps(1.8, 1e-4) == pytest.approx(18000.0)


# --- gates --------------------------------------------------------------------------


def test_check_all_shapes_captured_refuses_a_run_that_fell_through():
    model, cache = _ready([[1, 2, 3], [4]])
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    ids, _, view = _step(cache)
    captured(ids, view.plan.positions, cache=view)
    check_all_shapes_captured(captured)

    captured(torch.tensor([[1]]))

    with pytest.raises(CaptureUnsound, match="eager"):
        check_all_shapes_captured(captured)


def test_check_replays_dominate_refuses_a_run_that_recorded_every_step():
    model, _ = _model()
    _, cache = _prefilled([[1, 2, 3], [4]], persist_inputs=True, num_blocks=64)
    captured = CapturedDecode(
        model.forward, mode="capture", recorder=eager_recorder, precheck=False
    )
    for _ in range(6):
        ids, _, view = _step(cache)
        captured(ids, view.plan.positions, cache=view)

    with pytest.raises(CaptureUnsound, match="reuse"):
        check_replays_dominate(captured)


def test_check_replays_dominate_accepts_a_bucketed_run():
    model, cache = _ready([[1, 2, 3], [4]], num_blocks=64)
    captured = CapturedDecode(model.forward, mode="capture", recorder=eager_recorder)
    for _ in range(8):
        ids, _, view = _step(cache)
        captured(ids, view.plan.positions, cache=view)

    check_replays_dominate(captured)


def test_check_replays_dominate_needs_a_run():
    with pytest.raises(ValueError, match="at least one"):
        check_replays_dominate(CapturedDecode(_double, mode="off"))


def test_check_pool_budget_refuses_a_capture_list_the_device_cannot_hold():
    shapes = DecodeBuckets(256, 8192, width_multiple=2048).shapes

    check_pool_budget(shapes, 32, budget_bytes=1 << 30)

    with pytest.raises(CaptureUnsound, match="pool"):
        check_pool_budget(shapes, 32, budget_bytes=1 << 20)


def test_check_pool_budget_wants_a_positive_budget():
    with pytest.raises(ValueError, match="positive"):
        check_pool_budget((), 32, budget_bytes=0)


# --- the engine ---------------------------------------------------------------------


def _run(engine, requests):
    for request in requests:
        engine.add_request(request)
    done = {}
    while engine.has_unfinished():
        out = engine.step()
        for request in out.finished:
            done[request.request_id] = list(request.output_token_ids)
    return done


def test_an_engine_without_capture_records_nothing():
    model, _ = _model()
    engine = Engine.build(model, num_blocks=32, block_size=4, max_batch_size=4)

    assert engine.decode_graphs.mode == "off"
    assert engine.decode_graphs.captures == 0


def test_an_engine_that_captures_needs_the_five_days_switched_on():
    model, _ = _model()

    with pytest.raises(ValueError, match="bucket_decode"):
        Engine.build(
            model, num_blocks=32, block_size=4, max_batch_size=4, capture_decode=True
        )


def test_an_engine_over_captured_decode_generates_the_same_tokens():
    """The day ships or it does not: replaying a recorded step has to produce the
    text an eager one did, or every one of the last five days bought nothing."""
    model, _ = _model()
    out = []
    for extra in ({}, {"bucket_decode": True, "persist_inputs": True, "capture_decode": True}):
        engine = Engine.build(
            model,
            num_blocks=64,
            block_size=4,
            max_batch_size=4,
            max_model_len=64,
            **extra,
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


def test_a_long_run_records_a_handful_of_graphs_and_replays_the_rest():
    model, _ = _model()
    engine = Engine.build(
        model,
        num_blocks=64,
        block_size=4,
        max_batch_size=4,
        max_model_len=64,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
    )

    _run(engine, [Request("a", [1, 2, 3, 4], max_new_tokens=20)])

    graphs = engine.decode_graphs
    assert graphs.captures <= 2
    assert graphs.replays >= 18
    check_all_shapes_captured(graphs)
    check_replays_dominate(graphs)
    check_pool_shared(graphs)


def test_the_engines_sampled_tokens_are_not_the_graphs_output():
    """Why Day 48's deferred window survives a capture: the sampler runs outside the
    recorded region, so what crosses a step boundary is its allocation and not the
    graph's one output buffer."""
    model, _ = _model()
    engine = Engine.build(
        model,
        num_blocks=64,
        block_size=4,
        max_batch_size=4,
        max_model_len=64,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        defer_window=4,
    )

    engine.add_request(Request("a", [1, 2, 3, 4], max_new_tokens=12))
    seen = 0
    while engine.has_unfinished():
        engine.step()
        for batch in engine.output.held_batches:
            check_output_not_held(engine.decode_graphs, batch.tokens)
            seen += 1

    assert seen > 0


def test_the_default_warmup_is_more_than_zero():
    assert DEFAULT_WARMUP >= 1
    assert DEFAULT_CAPTURE_LIMIT >= 1
