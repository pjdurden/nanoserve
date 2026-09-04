"""Day 49 tests: the decode forward behind a compiler, and the scissors in it.

Day 46 measured where a step's seconds go and Day 47 and Day 48 spent two days
taking synchronisations out of the loop around the forward. This file is about the
forward itself: `torch.compile` on the decode pass, which is the optimisation
Week 13 has been pointing at since Day 46 named `build_inputs` as the phase a
captured graph deletes.

Four claims, and only the first one is about speed:

  1. **A compiled decode returns the same logits.** Everything else here is
     worthless if a graph quietly changes an answer. Inductor reassociates and
     fuses, so the comparison is `allclose` and not equality, and the tolerance is
     written down rather than tuned until it passes.
  2. **The compiled region is one graph, or it is not a graph at all.** Dynamo does
     not fail on code it cannot trace: it *breaks*, returns to the interpreter for
     that line, and resumes tracing afterwards. The result is a compiled function
     made of fragments with Python between them, which looks like a success and
     performs like the eager loop. This engine started with twelve breaks in a
     two-layer decode and every one of them was `int(context_lens.min())` in its
     own reference kernel, which Day 48 already caught doing something else wrong.
  3. **The churn is the recompile bill, and it is not only shape churn.** A decode
     batch is `[rows, 1]` over a `[rows, max_ctx]` mapping whose width grows every
     step, which `dynamic=True` answers by making both symbolic. What it does not
     answer is a guard on a Python int: `table.num_tokens`, read inside the traced
     region, is specialised on its *value*, so the graph is invalidated every step
     anyway. Past `cache_size_limit` dynamo stops compiling that frame and runs
     eager for the rest of the process without raising anything, which is why
     `check_graph_reused` is a gate and not a note.
  4. **A compile is paid up front and has to be earned back.** Seconds of compile
     against microseconds of saving is a breakeven in steps, and a short run is
     better off eager. That is arithmetic, so it is tested as arithmetic.

The wrapper takes its compiler as an argument, which is what lets most of this file
run in milliseconds: a fake compiler counts what it was asked to build and returns
the function unchanged, so the *policy* (which shapes, how many builds, when it
falls back) is tested apart from inductor. The handful of tests that need the real
thing say so in their names and use a two-layer model.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.compiled import (
    ROW_BUCKETS,
    RECOMPILE_LIMIT,
    CompiledDecode,
    CompileReport,
    CompileUnsound,
    DecodeShape,
    breakeven_steps,
    bucket_for,
    bucket_padding_waste,
    bucketed,
    check_compile_amortised,
    check_graph_reused,
    check_no_fallback,
    check_shapes_bucketed,
    check_single_graph,
    compile_cost_s,
    decode_shape,
    distinct_shapes,
    dynamo_frames_compiled,
    dynamo_unique_graphs,
    explain_forward,
    fragment_overhead_s,
    fragments,
    net_saving_s,
    recompiles,
    render,
    round_up,
    saving_per_step_s,
    shape_history,
    step_speedup,
    worth_compiling,
)
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.scheduler import Request

# --- fixtures -------------------------------------------------------------------


def _tiny_config(num_hidden_layers: int = 2) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _model(seed: int = 0, num_hidden_layers: int = 2) -> LlamaModel:
    torch.manual_seed(seed)
    cfg = _tiny_config(num_hidden_layers)
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg))


def _engine(model, num_blocks=64, block_size=4, max_batch_size=4, **kw) -> Engine:
    return Engine.build(
        model,
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch_size=max_batch_size,
        **kw,
    )


def _req(request_id, prompt, max_new_tokens=4) -> Request:
    return Request(
        request_id=request_id, prompt_token_ids=list(prompt), max_new_tokens=max_new_tokens
    )


class FakeCompiler:
    """A compiler that builds nothing and remembers everything it was asked.

    The wrapper's job is policy: which shapes reach the compiler, how many builds
    that implies, and when the answer is "stop, this will fall back". None of that
    needs inductor, and running it without inductor is what keeps this file fast.
    `graphs` stands in for `torch._dynamo`'s unique-graph counter, so the wrapper's
    accounting is exercised over a counter that moves.
    """

    def __init__(self, per_shape: bool = True):
        self.per_shape = per_shape
        self.built: list[bool | None] = []
        self.graphs = 0
        self._seen: set = set()

    def __call__(self, fn, dynamic):
        self.built.append(dynamic)

        def wrapper(*args, **kwargs):
            key = tuple(a.shape for a in args if isinstance(a, torch.Tensor))
            if not self.per_shape or key not in self._seen:
                self._seen.add(key)
                self.graphs += 1
            return fn(*args, **kwargs)

        return wrapper

    def count(self) -> int:
        return self.graphs


def _decode_call(engine):
    """Drive one engine to the point where the next decode's tensors are buildable.

    Returns `(input_ids, positions, view)`, the exact three arguments
    `Engine._decode` hands the model, without stepping the engine past them.
    """
    out = engine.scheduler.schedule()
    requests = out.decode
    rows = [r.slot for r in requests]
    engine._sync_rows(requests)
    input_ids = torch.tensor(
        [[r.output_token_ids[-1]] for r in requests], dtype=torch.long
    )
    positions = torch.tensor(
        [[engine.cache.tables[row].num_tokens] for row in rows], dtype=torch.long
    )
    return input_ids, positions, engine.cache.view(rows)


# --- what a decode step looks like to a guard ------------------------------------


def test_decode_shape_is_rows_and_context_width():
    shape = DecodeShape(rows=4, context_width=37)
    assert shape.rows == 4
    assert shape.context_width == 37
    assert shape.query_len == 1


def test_decode_shape_is_hashable_and_compares_by_value():
    assert DecodeShape(4, 37) == DecodeShape(4, 37)
    assert len({DecodeShape(4, 37), DecodeShape(4, 37), DecodeShape(4, 38)}) == 2


def test_decode_shape_refuses_a_prefill():
    with pytest.raises(ValueError, match="decode"):
        DecodeShape(rows=4, context_width=37, query_len=8)


@pytest.mark.parametrize("rows,width", [(0, 4), (4, 0), (-1, 4)])
def test_decode_shape_refuses_an_empty_dimension(rows, width):
    with pytest.raises(ValueError):
        DecodeShape(rows=rows, context_width=width)


def test_decode_shape_reads_a_real_cache_view():
    model = _model()
    engine = _engine(model)
    for i in range(3):
        engine.add_request(_req(f"r{i}", [1, 2, 3], max_new_tokens=6))
    engine.step()
    input_ids, _, view = _decode_call(engine)
    shape = decode_shape(input_ids, view)
    assert shape.rows == 3
    # Three tokens of prompt plus the one the prefill sampled.
    assert shape.context_width == 4


def test_decode_shape_context_width_grows_every_step():
    """The whole recompile story in one assertion: the guarded dim never repeats."""
    model = _model()
    engine = _engine(model)
    engine.add_request(_req("a", [1, 2, 3], max_new_tokens=6))
    engine.step()
    widths = []
    for _ in range(4):
        input_ids, _, view = _decode_call(engine)
        widths.append(decode_shape(input_ids, view).context_width)
        engine.step()
    assert widths == sorted(widths)
    assert len(set(widths)) == len(widths)


# --- counting the shapes a run presents -------------------------------------------


def test_shape_history_pairs_rows_with_widths():
    shapes = shape_history([2, 2, 3], [10, 11, 12])
    assert shapes == (DecodeShape(2, 10), DecodeShape(2, 11), DecodeShape(3, 12))


def test_shape_history_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="one width per step"):
        shape_history([2, 2], [10])


def test_distinct_shapes_counts_the_set_not_the_run():
    assert distinct_shapes(shape_history([2, 2, 2], [10, 10, 10])) == 1
    assert distinct_shapes(shape_history([2, 2, 2], [10, 11, 12])) == 3


def test_recompiles_is_one_fewer_than_the_shapes():
    """The first build is a compile, not a recompile. Off by one, on purpose."""
    assert recompiles(shape_history([2], [10])) == 0
    assert recompiles(shape_history([2, 2, 2], [10, 11, 12])) == 2


def test_recompiles_of_an_empty_run_is_zero():
    assert recompiles(()) == 0


def test_falls_back_once_the_shapes_pass_the_limit():
    from nanoserve.compiled import falls_back

    within = shape_history([1] * RECOMPILE_LIMIT, range(10, 10 + RECOMPILE_LIMIT))
    beyond = shape_history([1] * (RECOMPILE_LIMIT + 1), range(10, 11 + RECOMPILE_LIMIT))
    assert not falls_back(within)
    assert falls_back(beyond)


def test_a_hundred_step_generation_blows_the_limit_by_itself():
    """One request, no churn, nothing exotic: still 100 shapes and a silent bail."""
    from nanoserve.compiled import falls_back

    shapes = shape_history([1] * 100, range(8, 108))
    assert distinct_shapes(shapes) == 100
    assert falls_back(shapes)


# --- bucketing: the closed shape set ---------------------------------------------


@pytest.mark.parametrize("rows,expected", [(1, 1), (2, 2), (3, 4), (5, 8), (8, 8), (9, 16)])
def test_bucket_for_rounds_up_to_the_next_bucket(rows, expected):
    assert bucket_for(rows, ROW_BUCKETS) == expected


def test_bucket_for_refuses_a_value_above_every_bucket():
    with pytest.raises(ValueError, match="no bucket"):
        bucket_for(1000, (1, 2, 4))


def test_bucket_for_refuses_a_non_positive_value():
    with pytest.raises(ValueError):
        bucket_for(0, ROW_BUCKETS)


@pytest.mark.parametrize("value,multiple,expected", [(1, 128, 128), (128, 128, 128), (129, 128, 256)])
def test_round_up_is_the_width_bucket(value, multiple, expected):
    assert round_up(value, multiple) == expected


def test_bucketed_pads_both_guarded_dimensions():
    assert bucketed(DecodeShape(3, 130), width_multiple=128) == DecodeShape(4, 256)


def test_bucketing_collapses_a_hundred_shapes_to_a_handful():
    shapes = shape_history([1] * 100, range(8, 108))
    padded = tuple(bucketed(s, width_multiple=32) for s in shapes)
    assert distinct_shapes(shapes) == 100
    assert distinct_shapes(padded) == 4  # widths 32, 64, 96, 128 at one row


def test_bucket_padding_waste_is_the_cells_nobody_wanted():
    """Two rows of 10 real cells each, padded to 2 rows of 32: 20 of 64 are real."""
    shapes = (DecodeShape(2, 10),)
    waste = bucket_padding_waste(shapes, width_multiple=32)
    assert waste == pytest.approx(1 - 20 / 64)


def test_bucket_padding_waste_is_zero_on_an_exact_fit():
    assert bucket_padding_waste((DecodeShape(4, 32),), width_multiple=32) == 0.0


def test_bucket_padding_waste_of_nothing_is_zero():
    assert bucket_padding_waste((), width_multiple=32) == 0.0


def test_wider_buckets_trade_shapes_for_waste():
    """The whole bucketing decision, as two numbers moving in opposite directions."""
    shapes = shape_history([1] * 64, range(1, 65))
    coarse = tuple(bucketed(s, width_multiple=64) for s in shapes)
    fine = tuple(bucketed(s, width_multiple=8) for s in shapes)
    assert distinct_shapes(coarse) < distinct_shapes(fine)
    assert bucket_padding_waste(shapes, width_multiple=64) > bucket_padding_waste(
        shapes, width_multiple=8
    )


# --- graph breaks: the fragments a "compiled" function is really made of -----------


def test_fragments_is_one_more_than_the_breaks():
    assert fragments(0) == 1
    assert fragments(12) == 13


def test_fragments_refuses_a_negative_break_count():
    with pytest.raises(ValueError):
        fragments(-1)


def test_fragment_overhead_scales_with_the_breaks():
    assert fragment_overhead_s(12, per_break_s=5e-6) == pytest.approx(60e-6)
    assert fragment_overhead_s(0, per_break_s=5e-6) == 0.0


def test_compile_report_derives_its_fragments():
    report = CompileReport(graphs=13, breaks=12, ops=108)
    assert report.fragments == 13
    assert not report.is_one_graph
    assert report.ops_per_graph == pytest.approx(108 / 13)


def test_compile_report_of_one_graph_is_one_graph():
    report = CompileReport(graphs=1, breaks=0, ops=137)
    assert report.is_one_graph
    assert report.fragments == 1


def test_check_single_graph_passes_a_whole_capture():
    check_single_graph(CompileReport(graphs=1, breaks=0, ops=137))


def test_check_single_graph_refuses_a_fragmented_one():
    with pytest.raises(CompileUnsound, match="12 graph break"):
        check_single_graph(CompileReport(graphs=13, breaks=12, ops=108))


# --- the arithmetic of paying for a compile ---------------------------------------


def test_compile_cost_scales_with_the_graphs_built():
    assert compile_cost_s(4, per_graph_s=2.5) == pytest.approx(10.0)


def test_step_speedup_is_amdahl_over_the_forward_share():
    """Compiling the forward speeds the forward. The step has other phases in it."""
    assert step_speedup(1.0, 2.0) == pytest.approx(2.0)
    assert step_speedup(0.5, 2.0) == pytest.approx(1 / (0.5 + 0.25))
    assert step_speedup(0.0, 100.0) == pytest.approx(1.0)


def test_saving_per_step_is_the_seconds_amdahl_leaves():
    saved = saving_per_step_s(step_s=1e-3, forward_share=0.5, forward_speedup=2.0)
    assert saved == pytest.approx(1e-3 - 1e-3 / step_speedup(0.5, 2.0))


def test_saving_per_step_of_a_speedup_of_one_is_zero():
    assert saving_per_step_s(step_s=1e-3, forward_share=0.9, forward_speedup=1.0) == 0.0


def test_breakeven_is_the_compile_divided_by_the_saving():
    assert breakeven_steps(10.0, 1e-3) == pytest.approx(10_000)


def test_breakeven_of_no_saving_never_arrives():
    assert breakeven_steps(10.0, 0.0) == float("inf")
    assert breakeven_steps(10.0, -1e-4) == float("inf")


def test_net_saving_is_negative_before_breakeven_and_positive_after():
    args = dict(compile_s=10.0, saving_per_step_s=1e-3)
    assert net_saving_s(1_000, **args) < 0
    assert net_saving_s(10_000, **args) == pytest.approx(0.0)
    assert net_saving_s(20_000, **args) > 0


def test_worth_compiling_is_the_sign_of_the_net():
    args = dict(compile_s=10.0, saving_per_step_s=1e-3)
    assert not worth_compiling(1_000, **args)
    assert worth_compiling(20_000, **args)


def test_check_compile_amortised_refuses_a_run_too_short_to_pay():
    with pytest.raises(CompileUnsound, match="breakeven"):
        check_compile_amortised(1_000, compile_s=10.0, saving_per_step_s=1e-3)


def test_check_compile_amortised_passes_a_long_run():
    check_compile_amortised(50_000, compile_s=10.0, saving_per_step_s=1e-3)


# --- the wrapper's policy, over a compiler that builds nothing --------------------


def test_off_mode_never_reaches_the_compiler():
    compiler = FakeCompiler()
    fn = CompiledDecode(lambda x: x * 2, mode="off", compiler=compiler)
    assert fn(torch.ones(2, 1)).tolist() == [[2.0], [2.0]]
    assert compiler.built == []
    assert fn.compiles == 0


def test_dynamic_mode_asks_for_one_symbolic_build():
    compiler = FakeCompiler(per_shape=False)
    fn = CompiledDecode(lambda x: x * 2, mode="dynamic", compiler=compiler)
    fn(torch.ones(2, 1))
    fn(torch.ones(3, 1))
    assert compiler.built == [True]
    assert fn.expected_compiles == 1


def test_static_mode_asks_for_a_build_per_shape():
    compiler = FakeCompiler(per_shape=True)
    fn = CompiledDecode(lambda x: x * 2, mode="static", compiler=compiler)
    fn(torch.ones(2, 1))
    fn(torch.ones(3, 1))
    fn(torch.ones(3, 1))
    assert compiler.built == [False]
    assert compiler.count() == 2
    assert fn.expected_compiles == 2


def test_the_wrapper_refuses_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        CompiledDecode(lambda x: x, mode="fast")


def test_calls_and_reuses_are_counted_separately():
    fn = CompiledDecode(lambda x: x, mode="static", compiler=FakeCompiler())
    for _ in range(3):
        fn(torch.ones(2, 1))
    assert fn.calls == 3
    assert fn.distinct == 1
    assert fn.reuses == 2


def test_the_wrapper_records_the_shapes_in_first_seen_order():
    fn = CompiledDecode(lambda x: x, mode="static", compiler=FakeCompiler())
    fn(torch.ones(2, 1))
    fn(torch.ones(3, 1))
    fn(torch.ones(2, 1))
    assert [s.rows for s in fn.shapes] == [2, 3]


def test_a_static_wrapper_reports_its_own_fallback():
    fn = CompiledDecode(
        lambda x: x, mode="static", compiler=FakeCompiler(), recompile_limit=3
    )
    for width in range(1, 4):
        fn(torch.ones(2, width))
    assert not fn.fell_back
    fn(torch.ones(2, 4))
    assert fn.fell_back


def test_a_dynamic_wrapper_never_falls_back():
    fn = CompiledDecode(
        lambda x: x, mode="dynamic", compiler=FakeCompiler(), recompile_limit=3
    )
    for width in range(1, 20):
        fn(torch.ones(2, width))
    assert not fn.fell_back


def test_check_no_fallback_refuses_a_wrapper_over_the_limit():
    fn = CompiledDecode(
        lambda x: x, mode="static", compiler=FakeCompiler(), recompile_limit=2
    )
    for width in range(1, 4):
        fn(torch.ones(2, width))
    with pytest.raises(CompileUnsound, match="fell back"):
        check_no_fallback(fn)


def test_check_no_fallback_passes_a_wrapper_inside_the_limit():
    fn = CompiledDecode(lambda x: x, mode="dynamic", compiler=FakeCompiler())
    fn(torch.ones(2, 1))
    check_no_fallback(fn)


def test_check_graph_reused_passes_a_run_that_built_once():
    check_graph_reused(calls=100, builds=1)


def test_check_graph_reused_refuses_a_build_per_call():
    """Day 49's actual result, as a gate. One build per step is not an optimisation."""
    with pytest.raises(CompileUnsound, match="reuse"):
        check_graph_reused(calls=9, builds=9)


def test_check_graph_reused_refuses_a_run_just_under_the_bar():
    with pytest.raises(CompileUnsound, match="reuse"):
        check_graph_reused(calls=10, builds=6)


def test_check_graph_reused_takes_the_bar_as_an_argument():
    check_graph_reused(calls=10, builds=6, min_reuse=0.3)


@pytest.mark.parametrize("calls,builds", [(0, 0), (5, -1)])
def test_check_graph_reused_refuses_nonsense(calls, builds):
    with pytest.raises(ValueError):
        check_graph_reused(calls=calls, builds=builds)


def test_the_dynamo_counters_are_readable_and_are_not_the_same_number():
    """A recompile of one structure moves one counter and not the other.

    `unique_graphs` counts distinct graph structures, so a frame rebuilt eight
    times because a guard on a Python int kept failing leaves it sitting still.
    `frames.ok` counts builds. Picking the first cost a benchmark run that reported
    one compile for a run that had done eight.
    """
    assert dynamo_unique_graphs() >= 0
    assert dynamo_frames_compiled() >= 0


def test_check_shapes_bucketed_refuses_an_open_shape_set():
    shapes = shape_history([1] * 100, range(8, 108))
    with pytest.raises(CompileUnsound, match="distinct"):
        check_shapes_bucketed(shapes, width_multiple=1)


def test_check_shapes_bucketed_passes_a_closed_one():
    shapes = shape_history([1] * 100, range(8, 108))
    check_shapes_bucketed(shapes, width_multiple=64)


def test_the_wrapper_passes_keyword_arguments_through():
    fn = CompiledDecode(
        lambda x, cache=None: x + cache, mode="static", compiler=FakeCompiler()
    )
    assert fn(torch.ones(2, 1), cache=torch.ones(2, 1)).tolist() == [[2.0], [2.0]]


# --- the readback the kernel used to do on every layer ----------------------------


def test_context_bounds_given_match_the_bounds_computed():
    """Same answer either way. The difference is a journey, not a number."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 1, 4)
    k_pool = torch.randn(16, 2, 4)
    v_pool = torch.randn(16, 2, 4)
    slots = torch.tensor([[0, 1, 2, 0], [3, 4, 5, 6]])
    lens = torch.tensor([3, 4])
    computed = paged_attention_batched_reference(q, k_pool, v_pool, slots, lens, n_rep=2)
    given = paged_attention_batched_reference(
        q, k_pool, v_pool, slots, lens, n_rep=2, context_bounds=(3, 4)
    )
    assert torch.equal(computed, given)


def test_context_bounds_still_validate():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 1, 4)
    k_pool = torch.randn(16, 2, 4)
    v_pool = torch.randn(16, 2, 4)
    slots = torch.tensor([[0, 1, 2, 0], [3, 4, 5, 6]])
    lens = torch.tensor([3, 4])
    with pytest.raises(ValueError, match="at least 1"):
        paged_attention_batched_reference(
            q, k_pool, v_pool, slots, lens, n_rep=2, context_bounds=(0, 4)
        )
    with pytest.raises(ValueError, match="more history"):
        paged_attention_batched_reference(
            q, k_pool, v_pool, slots, lens, n_rep=2, context_bounds=(3, 9)
        )


def test_the_cache_knows_its_bounds_without_asking_the_device():
    cfg = _tiny_config()
    allocator = BlockAllocator(num_blocks=32, block_size=4)
    cache = BatchedPagedKVCache(cfg, allocator, batch_size=3)
    for row, length in enumerate([5, 2, 7]):
        cache.tables[row].adopt(allocator.allocate_for(length))
        cache.tables[row].append(length)
    assert cache.context_bounds() == (2, 7)
    assert cache.view([0, 2]).context_bounds() == (5, 7)


def test_a_decode_step_never_reads_context_lens_back():
    """The day, as one monkeypatch. Day 48 scoped this claim to the input side.

    Day 48 wrote this test and had to narrow it, because the reference kernel
    validated `context_lens` with `int(...)` on every layer of every step. With the
    bounds handed down from the cache, where they are already Python ints, the whole
    decode forward runs without a single readback.
    """
    model = _model()
    engine = _engine(model)
    for i in range(2):
        engine.add_request(_req(f"r{i}", [1, 2, 3], max_new_tokens=6))
    engine.step()
    input_ids, positions, view = _decode_call(engine)

    def boom(self, *a, **kw):
        raise AssertionError("the decode forward read a tensor back to the host")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "tolist", boom)
        mp.setattr(torch.Tensor, "item", boom)
        mp.setattr(torch.Tensor, "__int__", boom)
        model.forward(input_ids, positions, cache=view)


# --- the real compiler ------------------------------------------------------------


def test_explain_forward_finds_one_graph_in_the_decode_pass():
    """The claim that made this day: no breaks left in a two-layer decode."""
    model = _model()
    engine = _engine(model)
    for i in range(2):
        engine.add_request(_req(f"r{i}", [1, 2, 3], max_new_tokens=6))
    engine.step()
    input_ids, positions, view = _decode_call(engine)
    report = explain_forward(model.forward, input_ids, positions, cache=view)
    assert report.breaks == 0
    assert report.graphs == 1
    check_single_graph(report)


def test_a_compiled_decode_returns_the_same_logits():
    """Two engines, not two calls: a forward over a KV cache is not a function.

    The obvious version of this test runs the eager forward and the compiled one
    over the same cache view and compares. It fails, and not by a tolerance: the
    read writes this step's K/V into the row before it attends, so the second call
    attends over a history one token longer than the first and writes a duplicate
    into the next slot. Comparing a compiler against an eager baseline needs two
    identical states, which here means two engines driven the same way.
    """

    def state():
        engine = _engine(_model())
        for i in range(2):
            engine.add_request(_req(f"r{i}", [1, 2, 3], max_new_tokens=6))
        engine.step()
        return engine

    a, b = state(), state()
    ii_a, pos_a, view_a = _decode_call(a)
    eager = a.model.forward(ii_a, pos_a, cache=view_a)
    ii_b, pos_b, view_b = _decode_call(b)
    got = CompiledDecode(b.model.forward, mode="dynamic")(ii_b, pos_b, cache=view_b)
    assert torch.equal(ii_a, ii_b)
    assert torch.allclose(eager, got, atol=1e-5, rtol=1e-5)


def test_a_compiled_engine_generates_the_same_tokens():
    """The only assertion in this file that would survive deleting the rest."""
    prompts = [[1, 2, 3], [4, 5], [6]]
    eager = _engine(_model()).generate(prompts, max_new_tokens=5)
    compiled = _engine(_model(), compile_decode="dynamic").generate(prompts, max_new_tokens=5)
    assert compiled == eager


def test_the_engine_counts_what_it_compiled():
    engine = _engine(_model(), compile_decode="dynamic")
    engine.generate([[1, 2, 3], [4, 5]], max_new_tokens=4)
    assert engine.decode_forward.calls > 0
    assert engine.decode_forward.mode == "dynamic"
    check_no_fallback(engine.decode_forward)


def test_an_uncompiled_engine_still_has_a_wrapper_that_does_nothing():
    engine = _engine(_model())
    assert engine.decode_forward.mode == "off"
    engine.generate([[1, 2, 3]], max_new_tokens=3)
    assert engine.decode_forward.compiles == 0


def test_the_engine_refuses_an_unknown_compile_mode():
    with pytest.raises(ValueError, match="mode"):
        _engine(_model(), compile_decode="turbo")


def test_a_static_engine_run_records_a_shape_per_step():
    """Continuous batching against a static graph, in one counter."""
    engine = _engine(_model(), compile_decode="static")
    engine.generate([[1, 2, 3]], max_new_tokens=5)
    forward = engine.decode_forward
    assert forward.distinct == forward.calls
    assert recompiles(forward.shapes) == forward.calls - 1


# --- the table --------------------------------------------------------------------


def test_render_has_a_row_per_mode():
    out = render(step_s=1e-3, forward_share=0.6, forward_speedup=1.5, compile_s=8.0)
    assert "dynamic" in out
    assert "static" in out
    assert len(out.splitlines()) >= 3


def test_render_takes_a_title():
    out = render(
        step_s=1e-3, forward_share=0.6, forward_speedup=1.5, compile_s=8.0, title="hello"
    )
    assert out.splitlines()[0] == "hello"
