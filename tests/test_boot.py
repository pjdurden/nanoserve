"""Day 56 tests: the boot path, and the second sizing decision it has to make.

Six days of work sit behind flags nobody turns on. Day 52's bucket set, Day 53's
persistent inputs, Day 54's capture and Day 55's warm-up are all reachable from
`Engine.build` and none of them are reachable from `serve.py`, because
`build_engine` does not pass them through. A real server started from the command
line today runs the Day-47 eager loop. This is the wiring, and wiring turns out to
be where the last constraint was hiding.

Four claims:

  1. **The flags thread through, and they thread through as a group.** `build_engine`
     takes the three and hands them to `Engine.build`, which already refuses two out
     of three loudly. The CLI is allowed to bundle them behind one `--cuda-graphs`;
     the constructor is not, because a capture over an open shape set is not a slower
     engine, it is a wrong one.
  2. **The capture list is a sizing decision like the pool, and it is made later.**
     `plan_capture` is to the graph pool what `plan_kv_pool` is to the K/V pool: a
     record a human can read, computed from numbers that arrive from three different
     places. The rows come from the scheduler, the width from the served context, and
     the budget from a probe taken *after* the K/V pool is spoken for.
  3. **Three ceilings, one number.** Day 55 found that a byte budget is not a
     statement about the length of the capture list but about a single width, because
     a shared arena is sized by its largest member. The graph limit turns out to be
     the same kind of statement: a list is `rows x widths`, so a cap on the count is a
     cap on how far up the width axis it may go. Served context, byte budget and
     graph limit all reduce to one width ceiling, and `width_bound_by` says which of
     them bit.
  4. **Warming happens before the door opens, and the boot payload proves it.** The
     warm-up writes the shared input buffers and the slot table, so a walk taken while
     the `AsyncEngine` loop is running races a real step over the same storage.
     `build_app` cannot get that wrong because it warms before the bridge exists;
     `check_warm_before_serving` is for everybody else. `/health` then carries both
     decisions, because "are this server's graphs warm" is a 3am question and a
     restart is not an answer.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import torch

from nanoserve.buckets import DecodeBuckets
from nanoserve.captured import DEFAULT_CAPTURE_LIMIT, eager_recorder, shared_pool_bytes
from nanoserve.config import ModelConfig
from nanoserve.launch import (
    BootUnsound,
    CapturePlan,
    CaptureTooSmall,
    boot_info,
    boot_lines,
    build_app,
    build_engine,
    check_boot_info,
    check_capture_limit,
    check_capture_matches_cache,
    check_warm_before_serving,
    kv_bytes_per_block,
    plan_capture,
    warm_engine,
    width_from_limit,
)
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.serving import AsyncEngine
from nanoserve.warmup import WarmupReport, stall_seconds


def _tiny_config(**kw) -> ModelConfig:
    base = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )
    base.update(kw)
    return ModelConfig(**base)


def _tiny_weights(config: ModelConfig, seed: int = 0) -> Weights:
    torch.manual_seed(seed)
    tensors = {n: torch.randn(*s) for n, s in expected_shapes(config).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return Weights(tensors, config)


def _build(**kw):
    """A launched engine over the tiny random model, on the CPU, with no weights dir.

    The same injection every launcher test since Day 38 uses: the wiring is what is
    under test and the weights are furniture. `capture_recorder` is the Day-54 seam,
    so a capture on a box with no CUDA records a stand-in with a capture's semantics
    and says so.
    """
    cfg = _tiny_config()
    defaults = dict(
        weights_dir="unused",
        device="cpu",
        dtype="float32",
        block_size=4,
        max_batch_size=4,
        max_model_len=16,
        kv_cache_bytes=kv_bytes_per_block(cfg, 4, torch.float32) * 32,
        load=lambda _dir, **_kw: _tiny_weights(cfg),
        read_config=lambda _dir: cfg,
    )
    defaults.update(kw)
    return build_engine(**defaults)


def _graphed(**kw):
    """The same launch with all three of the week's flags on."""
    defaults = dict(
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
    )
    defaults.update(kw)
    return _build(**defaults)


def _wide(**kw):
    """A launch whose served context spans several width buckets.

    `max_model_len=1024` at a width multiple of 128 gives eight widths and three row
    buckets, which is the smallest set where trimming one axis is visible in the
    other. The pool has to hold one request of that length, which is what the block
    count is for.
    """
    defaults = dict(max_model_len=1024, num_blocks=300, kv_cache_bytes=None)
    defaults.update(kw)
    return _graphed(**defaults)


# --- threading the flags through the launcher -----------------------------------


def test_a_plain_launch_has_no_buckets_no_buffers_and_no_capture():
    """What `serve.py` built yesterday, and the reason today exists."""
    engine, _ = _build()
    assert engine.cache.decode_buckets is None
    assert engine.cache.decode_inputs is None
    assert engine.decode_graphs.mode == "off"


def test_bucket_decode_reaches_the_cache_as_a_shape_set():
    engine, plan = _build(bucket_decode=True)
    assert engine.cache.decode_buckets is not None
    assert engine.cache.decode_buckets.max_batch_size == plan.max_batch_size


def test_the_bucket_set_is_sized_from_the_plan_not_from_the_pool():
    """The Day-51 number, one more time. The widths run to the served context, not
    to the pool's capacity: a set sized off `num_blocks * block_size` would hold
    buckets no row can ever be that wide."""
    engine, plan = _build(bucket_decode=True)
    assert engine.cache.decode_buckets.max_model_len == plan.max_model_len
    assert plan.max_model_len < plan.capacity_tokens


def test_persist_inputs_reaches_the_cache_as_a_buffer_set():
    engine, _ = _build(persist_inputs=True)
    assert engine.cache.decode_inputs is not None


def test_capture_decode_reaches_the_engine_as_a_recording_wrapper():
    engine, _ = _graphed()
    assert engine.decode_graphs.mode == "capture"


def test_a_launcher_cannot_switch_on_the_capture_alone():
    """`Engine.build` refuses two out of three and the launcher does not soften it.
    The whole point of the refusal is that the missing halves are not performance
    settings: a graph replayed over an input allocated per step reads storage
    nobody wrote."""
    with pytest.raises(ValueError, match="bucket_decode and persist_inputs"):
        _build(capture_decode=True)


def test_the_capture_recorder_is_threaded_so_a_cpu_launch_can_be_tested():
    engine, _ = _graphed()
    assert engine.decode_graphs.recorder is eager_recorder


def test_a_graphed_engine_still_generates_the_same_tokens():
    """The claim the whole week rests on, asserted at the level a server is built."""
    plain, _ = _build()
    graphed, _ = _graphed()
    assert plain.generate([[1, 2, 3]], max_new_tokens=5) == graphed.generate(
        [[1, 2, 3]], max_new_tokens=5
    )


# --- planning the capture list --------------------------------------------------


def test_the_row_ceiling_is_the_scheduler_s_slot_count():
    """Nobody else knows it. The cache has rows because the scheduler hands out
    slots, so the widest batch that can ever be presented is `max_batch_size`."""
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    assert capture.max_rows == plan.max_batch_size == 4
    assert max(s.rows for s in capture.shapes) == 4


def test_the_width_ceiling_defaults_to_the_context_this_server_sells():
    engine, plan = _wide()
    capture = plan_capture(engine, plan)
    assert capture.max_width == plan.max_model_len == 1024
    assert capture.width_bound_by == "served"


def test_the_whole_untrimmed_set_is_reported_next_to_the_trimmed_one():
    engine, plan = _wide()
    capture = plan_capture(engine, plan)
    assert capture.count == capture.full_count == 3 * 8
    assert capture.trimmed == 0


def test_a_width_flag_trims_the_list_and_says_that_it_did():
    engine, plan = _wide()
    capture = plan_capture(engine, plan, max_width=256)
    assert capture.max_width == 256
    assert capture.width_bound_by == "flag"
    assert capture.count == 3 * 2
    assert capture.trimmed == 3 * 6


def test_a_rows_flag_trims_the_other_axis():
    engine, plan = _wide()
    capture = plan_capture(engine, plan, max_rows=2)
    assert capture.max_rows == 2
    assert max(s.rows for s in capture.shapes) == 2
    assert capture.count == 2 * 8


def test_a_byte_budget_binds_the_width_and_not_the_length():
    """Day 55's lesson, applied at boot. The arena is sized by the largest member,
    so a budget cannot be spent by dropping shapes off the end of the list. It is a
    statement about one number and that number is a width."""
    engine, plan = _wide()
    # 4 rows x 8 heads x 4 bytes a cell, so 512 cells of width per byte-block.
    budget = 4 * 8 * 4 * 300
    capture = plan_capture(engine, plan, budget_bytes=budget)
    assert capture.width_bound_by == "budget"
    assert capture.max_width == 256  # 300 rounds down to the 128 multiple below it
    assert capture.pool_bytes <= budget


def test_a_ceiling_is_snapped_down_to_a_bucket_the_list_actually_holds():
    """A budget that buys 300 tokens buys the 256 bucket. Reporting 300 would name
    a context no graph in the list was recorded at, and the three ceilings only
    compare against each other once they are all in the same units."""
    engine, plan = _wide()
    capture = plan_capture(engine, plan, max_width=300)
    assert capture.max_width == 256
    assert max(s.context_width for s in capture.shapes) == 256


def test_the_tightest_of_the_ceilings_is_the_one_reported():
    engine, plan = _wide()
    capture = plan_capture(engine, plan, max_width=512, budget_bytes=4 * 8 * 4 * 300)
    assert capture.max_width == 256
    assert capture.width_bound_by == "budget"


def test_a_ceiling_that_does_not_bite_is_not_reported_as_the_one_that_did():
    """A flag set to exactly the served context changed nothing, and a boot line
    that blamed it would send somebody to the wrong knob."""
    engine, plan = _wide()
    capture = plan_capture(engine, plan, max_width=1024)
    assert capture.width_bound_by == "served"


def test_the_list_comes_back_biggest_first():
    """Day 55's ordering, preserved through the launcher: the arena is allocated
    once at its final size rather than grown once per shape on the way up."""
    engine, plan = _wide()
    shapes = plan_capture(engine, plan).shapes
    assert shapes[0].cells == max(s.cells for s in shapes)
    assert list(shapes) == sorted(shapes, key=lambda s: s.cells, reverse=True)


def test_the_workspace_is_the_widest_shape_and_not_the_sum():
    engine, plan = _wide()
    capture = plan_capture(engine, plan)
    heads = engine.model.config.num_attention_heads
    assert capture.pool_bytes == shared_pool_bytes(capture.shapes, heads)


def test_a_budget_under_the_smallest_shape_refuses_the_boot():
    """Not a degraded server. The first real decode would record the same graph in
    front of a client and want the same bytes, so a budget that cannot hold the
    narrowest bucket is an out-of-memory error scheduled for later."""
    engine, plan = _wide()
    with pytest.raises(CaptureTooSmall, match="128"):
        plan_capture(engine, plan, budget_bytes=16)


def test_planning_a_capture_list_for_an_engine_that_has_no_bucket_set_is_refused():
    engine, plan = _build()
    with pytest.raises(ValueError, match="bucket_decode"):
        plan_capture(engine, plan)


# --- the third ceiling: how many graphs this process will hold ------------------


def test_a_graph_limit_is_a_width_ceiling_too():
    """The list is `rows x widths`, so a cap on how many graphs may be held is a cap
    on how far up the width axis the list may go. Same shape of statement as the
    byte budget, arrived at from a completely different direction."""
    buckets = DecodeBuckets(16, 2048)
    assert len(buckets.rows) == 5 and len(buckets.widths) == 16
    # 64 graphs over 5 row buckets is 12 widths, and the twelfth is 1536.
    assert width_from_limit(buckets, row_count=5, limit=64) == 1536


def test_the_limit_keeps_whole_rows_of_the_width_axis():
    """Truncating mid-row would warm a shape for four rows and not for eight at the
    same context, which is a capture list with a hole in it at exactly the moment
    the scheduler admits one more request."""
    buckets = DecodeBuckets(16, 2048)
    width = width_from_limit(buckets, row_count=5, limit=64)
    kept = [w for w in buckets.widths if w <= width]
    assert 5 * len(kept) <= 64


def test_a_limit_with_room_for_every_width_does_not_trim():
    buckets = DecodeBuckets(4, 1024)
    assert width_from_limit(buckets, row_count=3, limit=64) == buckets.widths[-1] == 1024


def test_a_limit_below_one_full_row_bucket_is_refused():
    buckets = DecodeBuckets(16, 2048)
    with pytest.raises(CaptureTooSmall, match="row bucket"):
        width_from_limit(buckets, row_count=5, limit=4)


def test_the_default_serving_shape_lands_exactly_on_the_graph_limit():
    """The number that made this a wiring day with a finding in it. `serve.py`'s
    defaults are 8 slots and 2048 tokens, which is 4 row buckets x 16 widths = 64
    shapes, and `DEFAULT_CAPTURE_LIMIT` is 64. The default config fits with nothing
    to spare."""
    buckets = DecodeBuckets(8, 2048)
    assert len(buckets.shapes) == DEFAULT_CAPTURE_LIMIT
    assert width_from_limit(buckets, row_count=4, limit=DEFAULT_CAPTURE_LIMIT) == 2048


def test_doubling_the_slots_overruns_the_limit_and_the_width_pays_for_it():
    """`--max-batch-size 16` adds one row bucket, which multiplies the list by
    5/4 and puts it over. Nothing about the width changed and the width is what
    gives, because the count is a product and only one of its factors is a knob a
    deployment can move without changing what it sells."""
    buckets = DecodeBuckets(16, 2048)
    assert len(buckets.shapes) == 80 > DEFAULT_CAPTURE_LIMIT
    width = width_from_limit(buckets, row_count=5, limit=DEFAULT_CAPTURE_LIMIT)
    assert width == 1536


def test_the_limit_reaches_the_plan_as_the_ceiling_it_is():
    engine, plan = _wide()
    capture = plan_capture(engine, plan, limit=6)
    assert capture.width_bound_by == "limit"
    assert capture.count <= 6
    assert capture.max_width == 256


def test_a_planned_list_never_exceeds_the_capture_s_own_limit():
    engine, plan = _wide()
    capture = plan_capture(engine, plan)
    assert capture.limit == engine.decode_graphs.limit
    assert capture.count <= capture.limit


# --- warming at boot ------------------------------------------------------------


def test_warming_records_every_shape_in_the_list_and_leaves_nothing_cold():
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    report = warm_engine(engine, capture)
    assert isinstance(report, WarmupReport)
    assert report.captures == capture.count == 3
    assert report.cold == ()
    assert engine.decode_graphs.count == 3


def test_warming_twice_records_nothing_the_second_time():
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    warm_engine(engine, capture)
    assert warm_engine(engine, capture).captures == 0


def test_warming_spends_no_blocks_and_grows_no_row():
    """A warm batch is every row padding, so it writes at the sink and owns
    nothing. Asserted here rather than trusted, because this is the first call in
    the process and a leak of one block at boot is a leak for the life of it."""
    engine, plan = _graphed()
    free = engine.allocator.num_free
    lens = list(engine.cache.seq_lens)
    warm_engine(engine, plan_capture(engine, plan))
    assert engine.allocator.num_free == free
    assert list(engine.cache.seq_lens) == lens


def test_the_first_real_decode_after_warming_is_a_replay():
    """The whole day, in two counters. Yesterday this number was one recording per
    shape the run reached, taken mid-stream in front of whoever was waiting."""
    engine, plan = _graphed()
    warm_engine(engine, plan_capture(engine, plan))
    recorded = engine.decode_graphs.captures
    engine.generate([[1, 2, 3]], max_new_tokens=5)
    assert engine.decode_graphs.captures == recorded
    assert engine.decode_graphs.replays > 0
    assert engine.decode_graphs.eager_calls == 0


def test_a_warmed_engine_and_a_cold_one_agree_token_for_token():
    """Not luck. Every tensor the recorded call holds is a window on storage a real
    step writes through, so a graph recorded over no sequences at all still reads
    this step's numbers."""
    cold, _ = _graphed()
    warm, plan = _graphed()
    warm_engine(warm, plan_capture(warm, plan))
    assert cold.generate([[1, 2, 3]], max_new_tokens=5) == warm.generate(
        [[1, 2, 3]], max_new_tokens=5
    )


def test_warming_an_engine_that_records_nothing_is_refused():
    """The two halves without the third. Walking the bucket set through an engine
    whose capture is off runs every shape and records none of them, which is a
    startup cost with nothing bought."""
    graphed, plan = _graphed()
    eager, _ = _build(bucket_decode=True, persist_inputs=True)
    with pytest.raises(ValueError, match="capture_decode"):
        warm_engine(eager, plan_capture(graphed, plan))


def test_a_capture_list_from_another_cache_is_refused():
    """The plan and the engine are two arguments and nothing in the types stops
    them being from different processes. A shape this cache cannot present is a
    graph recorded on a rectangle wider than the slot table it is a window on."""
    engine, plan = _graphed()
    other, other_plan = _wide()
    with pytest.raises(BootUnsound, match="bucket set"):
        warm_engine(engine, plan_capture(other, other_plan))


def test_the_report_prices_a_graph_and_that_price_is_what_nobody_waited_for():
    """The trade stated in one assertion: seconds moved out of an inter-token
    latency and into a startup nobody is timing."""
    engine, plan = _graphed()
    report = warm_engine(engine, plan_capture(engine, plan))
    assert report.per_capture_s > 0
    avoided = stall_seconds(
        steps=512, start_width=16, width_multiple=128, per_capture_s=report.per_capture_s
    )
    assert avoided == pytest.approx(5 * report.per_capture_s)


def test_warming_while_the_serving_loop_runs_is_refused():
    """The race the boot order exists to avoid. The warm-up writes the shared input
    buffers and the slot table; a real step in flight reads the same addresses, and
    the failure is a token, not a traceback."""
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    serving = AsyncEngine(engine)

    async def scenario():
        await serving.start()
        try:
            with pytest.raises(BootUnsound, match="already running"):
                warm_engine(engine, capture, serving=serving)
        finally:
            await serving.stop()

    _run(scenario())


def test_a_stopped_bridge_is_a_bridge_that_may_be_warmed_behind():
    engine, plan = _graphed()
    serving = AsyncEngine(engine)
    assert serving.running is False
    warm_engine(engine, plan_capture(engine, plan), serving=serving)


def test_a_bridge_says_whether_its_loop_exists():
    engine, _ = _build()
    serving = AsyncEngine(engine)

    async def scenario():
        assert serving.running is False
        await serving.start()
        assert serving.running is True
        await serving.stop()
        assert serving.running is False

    _run(scenario())


# --- the gates ------------------------------------------------------------------


def test_check_warm_before_serving_passes_on_a_bridge_that_has_not_started():
    engine, _ = _build()
    check_warm_before_serving(AsyncEngine(engine))


def test_check_capture_matches_cache_passes_on_the_list_that_cache_produced():
    engine, plan = _wide()
    check_capture_matches_cache(plan_capture(engine, plan), engine.cache)


def test_check_capture_matches_cache_refuses_a_row_count_the_cache_has_no_rows_for():
    engine, plan = _wide()
    capture = plan_capture(engine, plan)
    wider = CapturePlan(
        shapes=capture.shapes,
        max_rows=capture.max_rows * 4,
        max_width=capture.max_width,
        width_bound_by="flag",
        num_heads=capture.num_heads,
        full_count=capture.full_count,
    )
    with pytest.raises(BootUnsound, match="rows"):
        check_capture_matches_cache(wider, engine.cache)


def test_check_capture_limit_refuses_a_list_longer_than_the_process_will_hold():
    engine, plan = _wide()
    capture = plan_capture(engine, plan)
    engine.decode_graphs.limit = 4
    with pytest.raises(BootUnsound, match="limit"):
        check_capture_limit(capture, engine.decode_graphs)


def test_check_capture_limit_counts_graphs_already_held():
    """Warming is not always the first thing to record. A process that captured
    something before the warm list was planned has that many fewer slots."""
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    engine.decode_graphs.limit = 3
    check_capture_limit(capture, engine.decode_graphs)
    warm_engine(engine, capture)
    check_capture_limit(capture, engine.decode_graphs)  # the same three, not six


# --- what the process publishes about itself ------------------------------------


def test_the_boot_payload_carries_the_pool_decision_unchanged():
    _, plan = _graphed()
    info = boot_info(plan)
    assert info["num_blocks"] == plan.num_blocks
    assert info["capacity_tokens"] == plan.capacity_tokens


def test_a_launch_without_graphs_says_nothing_about_graphs():
    _, plan = _build()
    assert "cuda_graphs" not in boot_info(plan)


def test_the_capture_decision_is_reported_beside_the_pool_and_not_inside_it():
    """Two sizing decisions, made at different times against different numbers.
    Flattening them into one dict would put `max_width` next to `max_model_len` and
    invite the reading that one is derived from the other."""
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    info = boot_info(plan, capture)
    assert info["cuda_graphs"]["shapes"] == capture.count
    assert info["cuda_graphs"]["max_rows"] == capture.max_rows
    assert info["cuda_graphs"]["width_bound_by"] == "served"
    assert "num_blocks" not in info["cuda_graphs"]


def test_the_payload_reports_what_warming_actually_cost():
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    report = warm_engine(engine, capture)
    graphs = boot_info(plan, capture, report)["cuda_graphs"]
    assert graphs["graphs_held"] == 3
    assert graphs["cold"] == 0
    assert graphs["warmup_seconds"] >= 0


def test_check_boot_info_passes_on_a_real_boot():
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    check_boot_info(boot_info(plan, capture, warm_engine(engine, capture)))


def test_check_boot_info_refuses_a_payload_that_claims_warm_graphs_with_a_cold_shape():
    """The lie a health check can tell that nothing else would catch: the server is
    up, the graphs are on, and some shape is still going to be recorded in front of
    a client."""
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    report = warm_engine(engine, capture)
    info = boot_info(plan, capture, WarmupReport(report.shapes, 2, 2, 0.1, report.shapes[:1]))
    with pytest.raises(BootUnsound, match="cold"):
        check_boot_info(info)


def test_check_boot_info_refuses_a_payload_with_no_pool_in_it():
    with pytest.raises(BootUnsound, match="pool"):
        check_boot_info({"status": "ok"})


def test_the_boot_lines_name_both_decisions_and_what_they_cost():
    engine, plan = _graphed()
    capture = plan_capture(engine, plan)
    report = warm_engine(engine, capture)
    lines = boot_lines(plan, capture, report)
    assert len(lines) == 3
    assert "KV pool" in lines[0]
    assert "CUDA graphs" in lines[1] and "the context this server sells" in lines[1]
    assert "recorded" in lines[2]


def test_a_launch_without_graphs_prints_one_line():
    _, plan = _build()
    assert len(boot_lines(plan)) == 1


# --- the app ---------------------------------------------------------------------


ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."


class ByteTokenizer:
    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [ALPHABET.index(ch) for ch in text]

    def decode(self, token_ids) -> str:
        return "".join(ALPHABET[i] for i in token_ids)


def _app(**kw):
    cfg = _tiny_config()
    defaults = dict(
        weights_dir="unused",
        device="cpu",
        dtype="float32",
        block_size=4,
        max_batch_size=4,
        max_model_len=16,
        kv_cache_bytes=kv_bytes_per_block(cfg, 4, torch.float32) * 32,
        load=lambda _dir, **_kw: _tiny_weights(cfg),
        read_config=lambda _dir: cfg,
        tokenizer=ByteTokenizer(),
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
    )
    defaults.update(kw)
    return build_app(**defaults)


def _run(coro, timeout: float = 20.0):
    async def guarded():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(guarded())


def test_the_app_is_warm_before_it_is_built():
    """`build_app` cannot get the order wrong, and that is the argument for doing it
    there: the bridge it would have to race does not exist yet when it warms."""
    app = _app()
    assert app.state.warmup.captures == 3
    assert app.state.capture.count == 3
    assert app.state.engine.decode_graphs.count == 3


def _get(app, path: str):
    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://nano") as client:
            async with app.router.lifespan_context(app):
                return await client.get(path)

    return _run(scenario())


def _complete(app, prompt: str = "abc", max_tokens: int = 4):
    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://nano") as client:
            async with app.router.lifespan_context(app):
                return await client.post(
                    "/v1/completions",
                    json={"model": "nanoserve", "prompt": prompt, "max_tokens": max_tokens},
                )

    return _run(scenario())


def test_health_reports_both_sizing_decisions():
    payload = _get(_app(), "/health").json()
    assert payload["status"] == "ok"
    assert payload["num_blocks"] == 32
    assert payload["cuda_graphs"]["graphs_held"] == 3
    assert payload["cuda_graphs"]["cold"] == 0


def test_a_server_launched_without_graphs_reports_none():
    app = _app(bucket_decode=False, persist_inputs=False, capture_decode=False)
    assert app.state.capture is None
    assert app.state.warmup is None
    assert "cuda_graphs" not in _get(app, "/health").json()


def test_warming_can_be_declined_and_the_graphs_stay_lazy():
    """Not a flag a deployment should want, and it is the control the benchmark
    needs: the same engine, the same list, recorded the Day-54 way."""
    app = _app(warm=False)
    assert app.state.capture is not None
    assert app.state.warmup is None
    assert app.state.engine.decode_graphs.count == 0


def test_the_warm_flags_reach_the_plan():
    app = _app(warm_rows=2)
    assert app.state.capture.max_rows == 2
    assert app.state.capture.count == 2


def test_a_launched_graphed_server_answers_a_completion():
    response = _complete(_app())
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 4


def test_serving_through_a_warm_engine_records_nothing_more():
    """The counter a production process would watch. Every decode the server ran
    replayed a graph the boot recorded, so no client paid for a recording."""
    app = _app()
    _complete(app)
    graphs = app.state.engine.decode_graphs
    assert graphs.captures == 3
    assert graphs.replays > 0
    assert graphs.eager_calls == 0


def test_the_served_text_does_not_depend_on_whether_the_graphs_were_warmed():
    """The end of the arc, checked where a user would feel it. Six days of shape
    rounding, fixed buffers, recordings and a warm-up, and the bytes on the wire are
    the ones Day 47 produced."""
    plain = _app(bucket_decode=False, persist_inputs=False, capture_decode=False)
    expected = _complete(plain).json()["choices"][0]["text"]
    assert _complete(_app()).json()["choices"][0]["text"] == expected
