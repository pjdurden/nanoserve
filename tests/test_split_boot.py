"""Day 66: the split read reaches the command line, and the arena gets a caller.

Day 65 stopped at the cache. `BatchedPagedKVCache(split_read=True)` decides the mode
and `allocate_split_workspace` hands the arena over, and nothing above the cache
called either: `Engine.build` did not take the flag, `build_engine` did not pass it,
and `build_app` had no step that turned a planned split into reserved memory. This
file is that wiring, and the questions it raises are all about order:

  1. **Can an engine be built at all?** `Engine.__init__` runs `check_capture_ready`
     when the graphs are on, and Day 65's read-matches gate refuses a split read with
     no arena. On the boot path the engine is built before anything is on a device,
     so the gate as written refused every split server that asked for CUDA graphs,
     at construction, before the one call that would have armed it.
  2. **Is the arena there before the first thing that reads it?** The warm-up is a
     decode step. A split read with no arena refuses a decode step. So the arena is
     allocated between the capture plan and the warm-up, and not a line later.
  3. **Do the two numbers a split server reserves appear together?** The capture's
     arena and the partials are both held for the life of the process and priced at
     different moments, and a boot line that prints one of them is half an answer.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.captured import CaptureUnsound, check_capture_ready, eager_recorder
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.launch import (
    BootUnsound,
    arm_split_read,
    boot_info,
    boot_lines,
    build_app,
    build_engine,
    check_arena_matches_capture,
    check_boot_info,
    kv_bytes_per_block,
    plan_capture,
)
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.reads import RECTANGLE, SPLIT, STREAMED
from nanoserve.scheduler import Request


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


def _weights(seed: int = 0) -> Weights:
    cfg = _tiny_config()
    torch.manual_seed(seed)
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return Weights(tensors, cfg)


def _model(seed: int = 0) -> LlamaModel:
    return LlamaModel(_tiny_config(), _weights(seed))


#: The three switches `--cuda-graphs` bundles, spelled out, because the question in
#: most of this file is what happens when the split read is asked for beside them.
GRAPHS = dict(bucket_decode=True, persist_inputs=True, capture_decode=True)


def _engine(**kw) -> Engine:
    defaults = dict(num_blocks=256, block_size=4, max_batch_size=4, max_model_len=1024,
                    read_block=4, capture_recorder=eager_recorder)
    defaults.update(kw)
    return Engine.build(_model(), **defaults)


def _launch_kwargs(**kw):
    cfg = _tiny_config()
    defaults = dict(
        weights_dir="unused",
        device="cpu",
        dtype="float32",
        block_size=4,
        max_batch_size=4,
        max_model_len=1024,
        read_block=4,
        kv_cache_bytes=kv_bytes_per_block(cfg, 4, torch.float32) * 256,
        load=lambda _dir, **_kw: _weights(),
        read_config=lambda _dir: cfg,
    )
    defaults.update(kw)
    return defaults


class ByteTokenizer:
    eos_token_id = None

    def encode(self, text, add_special_tokens=True):
        return [b % 64 for b in text.encode()]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids)


def _app(**kw):
    defaults = dict(tokenizer=ByteTokenizer(), capture_recorder=eager_recorder,
                    **_launch_kwargs())
    defaults.update(kw)
    return build_app(**defaults)


# --- Engine.build takes the flag ---------------------------------------------------------


def test_an_engine_asked_for_the_split_read_gets_it_unarmed():
    """The mode arrives with the engine and the arena does not, which is Day 65's
    two-step arrival seen from one layer up."""
    engine = _engine(bucket_decode=True, split_read=True)
    assert engine.cache.read.mode == SPLIT
    assert engine.cache.read.workspace is None
    # 1024 keys over 4-key tiles on 32 programs: wide enough that the planner really
    # cuts the row, so every test below is about a split and not a 1-way degenerate.
    assert engine.cache.read_splits == 2


def test_an_engine_passes_the_split_refusals_through_unsoftened():
    with pytest.raises(ValueError, match="bucket_decode"):
        _engine(split_read=True)
    with pytest.raises(ValueError, match="one read"):
        _engine(bucket_decode=True, split_read=True, streamed_read=True)


def test_a_split_engine_with_graphs_on_can_be_built_before_its_arena_exists():
    """The bug the wiring found. `Engine.__init__` runs `check_capture_ready` so a
    capture over an open shape set is refused at construction, and since Day 65 that
    gate also asks whether a split read holds an arena. On the boot path it cannot
    yet: the engine is built first and the arena is reserved after the capture plan
    prices it. So the construction-time gate asks about configuration only, and the
    arena question moves to the moment something is about to read it."""
    engine = _engine(split_read=True, **GRAPHS)
    assert engine.decode_graphs.mode == "capture"
    assert engine.cache.read.workspace is None


def test_build_engine_threads_the_split_read():
    engine, _ = build_engine(**_launch_kwargs(split_read=True, bucket_decode=True))
    assert engine.cache.read.mode == SPLIT
    assert engine.cache.read.block == 4


# --- check_capture_ready: configuration now, arena later ----------------------------------


def test_a_split_cache_with_no_arena_is_not_ready_to_capture():
    """Named for the call that fixes it rather than for the mismatch, because the
    mismatch is not a disagreement between two plans: it is a boot path that stopped
    one call short, and the operator needs the name of that call."""
    engine = _engine(split_read=True, **GRAPHS)
    with pytest.raises(CaptureUnsound, match="allocate_split_workspace"):
        check_capture_ready(engine.cache)


def test_a_split_cache_with_no_arena_passes_the_configuration_half():
    engine = _engine(split_read=True, **GRAPHS)
    check_capture_ready(engine.cache, armed=False)


def test_an_armed_split_cache_is_ready_to_capture():
    engine = _engine(split_read=True, **GRAPHS)
    engine.cache.allocate_split_workspace()
    check_capture_ready(engine.cache)


def test_the_configuration_half_still_refuses_a_misassembled_set():
    """`armed=False` excuses exactly one clause. A split read under a set priced in
    whole tiles is not an arena that has not arrived yet, it is a launch nobody sized,
    and it is refused whichever half is asking."""
    from nanoserve.buckets import BucketsUnsound, DecodeBuckets

    engine = _engine(split_read=True, **GRAPHS)
    engine.cache.decode_buckets = DecodeBuckets(4, 1024, streamed=True, block=4)
    with pytest.raises(BucketsUnsound, match="nobody sized"):
        check_capture_ready(engine.cache, armed=False)


def test_a_warm_up_before_the_arena_is_refused_with_the_boot_path_s_words():
    """The warm-up is a decode step, so this is the first moment the arena is needed,
    and the refusal comes from the capture gate rather than from inside the kernel."""
    engine = _engine(split_read=True, **GRAPHS)
    with pytest.raises(CaptureUnsound, match="allocate_split_workspace"):
        engine.warm_decode(device="cpu")


# --- arm_split_read: the one call on the boot path ---------------------------------------


def test_arming_a_non_split_engine_is_a_no_op():
    for kw in ({}, {"streamed_read": True}):
        engine = _engine(**GRAPHS, **kw)
        assert arm_split_read(engine) is None
        assert engine.cache.read.mode in (RECTANGLE, STREAMED)


def test_arming_a_split_engine_hands_its_read_the_cache_s_own_arena():
    engine = _engine(bucket_decode=True, split_read=True)
    workspace = arm_split_read(engine)
    assert engine.cache.read.workspace is workspace
    assert workspace.splits == engine.cache.read_splits
    assert workspace.max_rows == engine.cache.batch_size
    assert workspace.context_width == engine.cache.max_model_len


def test_arming_puts_the_arena_where_the_weights_are():
    engine = _engine(bucket_decode=True, split_read=True)
    workspace = arm_split_read(engine)
    assert workspace.device == engine.model.weights[EMBED].device


def test_arming_twice_is_refused_rather_than_re_reserved():
    engine = _engine(bucket_decode=True, split_read=True)
    arm_split_read(engine)
    with pytest.raises(BootUnsound, match="already"):
        arm_split_read(engine)


def test_arming_against_the_capture_plan_that_priced_it_passes():
    engine = _engine(split_read=True, **GRAPHS)
    capture = plan_capture(engine, _pool_plan(engine), split_read=True, device="cpu")
    workspace = arm_split_read(engine, capture)
    check_arena_matches_capture(capture, workspace)


def _pool_plan(engine):
    from nanoserve.launch import plan_kv_pool

    return plan_kv_pool(
        engine.model.config,
        block_size=4,
        max_batch_size=engine.cache.batch_size,
        max_model_len=engine.cache.max_model_len,
        dtype=torch.float32,
        num_blocks=256,
    )


def test_a_capture_plan_priced_for_a_different_split_is_refused():
    """The plan priced one arena and the cache allocated another. Both are right
    about their own inputs, the boot line would print the plan's number, and the
    process would hold the cache's. Refused before the warm-up records a graph
    against either."""
    from dataclasses import replace

    engine = _engine(split_read=True, **GRAPHS)
    capture = plan_capture(engine, _pool_plan(engine), split_read=True, device="cpu")
    workspace = engine.cache.allocate_split_workspace()
    with pytest.raises(BootUnsound, match="chunks"):
        check_arena_matches_capture(replace(capture, splits=capture.splits * 2), workspace)
    with pytest.raises(BootUnsound, match="tile"):
        check_arena_matches_capture(replace(capture, block=capture.block * 2), workspace)
    with pytest.raises(BootUnsound, match="wide"):
        check_arena_matches_capture(replace(capture, max_width=capture.max_width // 2),
                                    workspace)
    with pytest.raises(BootUnsound, match="rows"):
        check_arena_matches_capture(replace(capture, max_rows=workspace.max_rows + 1),
                                    workspace)


def test_a_capture_plan_that_priced_no_split_is_refused_against_an_arena():
    engine = _engine(split_read=True, **GRAPHS)
    capture = plan_capture(engine, _pool_plan(engine), split_read=True, device="cpu")
    from dataclasses import replace

    workspace = engine.cache.allocate_split_workspace()
    with pytest.raises(BootUnsound, match="priced no split"):
        check_arena_matches_capture(replace(capture, splits=0, head_dim=0), workspace)


def test_a_capture_list_under_fewer_rows_than_the_arena_holds_is_fine():
    """`--warm-rows` trims what is recorded, not what is served. The arena is sized
    for the scheduler's slot count because an unrecorded shape still runs, eagerly,
    against the same arena."""
    from dataclasses import replace

    engine = _engine(split_read=True, **GRAPHS)
    capture = plan_capture(engine, _pool_plan(engine), split_read=True, device="cpu")
    workspace = engine.cache.allocate_split_workspace()
    check_arena_matches_capture(replace(capture, max_rows=1), workspace)


# --- build_app: plan, arm, warm, in that order -------------------------------------------


def test_a_split_server_with_graphs_boots_warm_and_armed():
    app = _app(split_read=True, **GRAPHS)
    engine = app.state.engine
    assert engine.cache.read.workspace is app.state.workspace
    assert app.state.capture.splits == app.state.workspace.splits
    assert app.state.warmup is not None and not app.state.warmup.cold


def test_a_split_server_without_graphs_is_still_armed():
    """The arena is a property of the read and not of the capture. A split server
    with the graphs off still runs a split read on every decode step, and a boot path
    that only armed it inside the capture branch would serve its first request with a
    refusal."""
    app = _app(split_read=True, bucket_decode=True)
    assert app.state.capture is None
    assert app.state.engine.cache.read.workspace is app.state.workspace


def test_a_server_on_either_other_read_holds_no_arena():
    for kw in ({}, {"streamed_read": True}):
        app = _app(**GRAPHS, **kw)
        assert app.state.workspace is None


def test_a_split_server_answers_the_same_tokens_as_the_rectangle():
    """Greedy, through the engine's own step, the arena armed by the boot path rather
    than by the test. Not over a socket: that is the acceptance run's third arm and a
    separate decision (see the day's log)."""

    def generate(app):
        engine = app.state.engine
        requests = [
            engine.add_request(Request(f"r{i}", prompt, max_new_tokens=6))
            for i, prompt in enumerate(([1, 2, 3, 4, 5], [7, 8], [9, 10, 11]))
        ]
        while engine.has_unfinished():
            engine.step()
        return [list(r.output_token_ids) for r in requests]

    split_app = _app(split_read=True, **GRAPHS)
    split = generate(split_app)
    rectangle = generate(_app(**GRAPHS))
    assert split == rectangle
    stats = split_app.state.engine.cache.read.stats()
    assert stats.mode == SPLIT and stats.calls > 0 and stats.splits == 2


# --- the boot line and the health payload -------------------------------------------------


def test_the_health_payload_carries_the_arena_next_to_the_capture():
    app = _app(split_read=True, **GRAPHS)
    info = boot_info(app.state.plan, app.state.capture, app.state.warmup,
                     app.state.workspace)
    assert info["split_workspace"] == app.state.workspace.as_dict()
    assert info["cuda_graphs"]["read_splits"] == info["split_workspace"]["splits"]
    check_boot_info(info)


def test_the_boot_lines_print_the_arena_right_after_the_capture():
    """The two numbers a split server reserves, adjacent, because they are priced at
    two different moments and a reader adding up what the card is holding needs both."""
    app = _app(split_read=True, **GRAPHS)
    lines = boot_lines(app.state.plan, app.state.capture, app.state.warmup,
                       app.state.workspace)
    graphs = next(i for i, line in enumerate(lines) if line.startswith("CUDA graphs"))
    assert lines[graphs + 1] == app.state.workspace.render()


def test_a_split_server_without_graphs_still_prints_its_arena():
    app = _app(split_read=True, bucket_decode=True)
    lines = boot_lines(app.state.plan, app.state.capture, app.state.warmup,
                       app.state.workspace)
    assert app.state.workspace.render() in lines


def test_a_payload_that_priced_a_split_and_holds_no_arena_is_refused():
    """The capture section says a split was planned and the payload names no arena:
    the plan ran and the call that spends it did not. The server would refuse its
    first decode, so the health check should refuse first."""
    app = _app(split_read=True, **GRAPHS)
    info = boot_info(app.state.plan, app.state.capture, app.state.warmup)
    with pytest.raises(BootUnsound, match="arena"):
        check_boot_info(info)


def test_a_payload_whose_arena_disagrees_with_its_plan_is_refused():
    app = _app(split_read=True, **GRAPHS)
    info = boot_info(app.state.plan, app.state.capture, app.state.warmup,
                     app.state.workspace)
    info["split_workspace"] = dict(info["split_workspace"], splits=99)
    with pytest.raises(BootUnsound, match="99"):
        check_boot_info(info)
