"""Day 60: the decode read becomes a choice, and the choice is reported.

Day 59 wrote `paged_attention_batched_kernel` and proved it equal to the oracle on
toy pools. Nothing called it. This file is the wiring and the three questions it
raises, in the order they fail quietly:

  1. **Does the picked read return the same attention?** Asked of the cache rather
     than of the kernel, because the cache is where the plan path, the `validated`
     flag and the gathered mapping meet, and none of those exist in Day 59's tests.
  2. **Does a reader outside the process know which read ran?** A server whose flag
     did not reach the cache and a server running the streamed read report the same
     tokens, the same latency and the same everything else. Only a counter separates
     them, so the counter is the wiring's only witness.
  3. **Does the same engine answer the same bytes over a socket?** The one that
     matters. A read swapped under a live engine sits inside the forward, under the
     sampler, under the detokeniser and under SSE framing, and an output that is
     right to a few ulps is an output that can still round to a different token.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from nanoserve.acceptance import live_server
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.graphbench import (
    check_arm_read,
    check_arm_was_crowded,
    check_same_answers,
    paired_plans,
    read_from_health,
    run_arm,
)
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.launch import build_app, build_engine, kv_bytes_per_block
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.reads import (
    RECTANGLE,
    STREAMED,
    PagedRead,
    ReadStats,
)
from nanoserve.server import health_payload
from nanoserve.servebench import burst_arrivals
from nanoserve.serving import AsyncEngine


# --- ReadStats: one reading of what the read has held ---------------------------------


def test_a_fresh_reading_is_the_rectangle_and_has_done_nothing():
    stats = ReadStats()
    assert stats.mode == RECTANGLE
    assert stats.calls == 0
    assert stats.saving == 1.0


def test_the_saving_is_the_two_cell_counts_and_nothing_else():
    stats = ReadStats(mode=STREAMED, block=32, calls=4, rows=16, score_cells=8192, held_cells=256)
    assert stats.saving == 32.0


def test_a_rectangle_reading_saves_exactly_nothing_and_says_so():
    """1.0x is the report, not a missing number. A read that materialises the row it
    scores holds every cell it was charged, and rounding that to "no data" would make
    the default configuration look unmeasured rather than measured at parity."""
    stats = ReadStats(mode=RECTANGLE, calls=2, rows=8, score_cells=4096, held_cells=4096)
    assert stats.saving == 1.0


def test_a_window_is_the_difference_of_two_readings():
    before = ReadStats(mode=STREAMED, block=32, calls=10, rows=40, score_cells=100, held_cells=10)
    after = ReadStats(mode=STREAMED, block=32, calls=25, rows=90, score_cells=250, held_cells=25)
    window = after.since(before)
    assert (window.calls, window.rows) == (15, 50)
    assert (window.score_cells, window.held_cells) == (150, 15)
    assert window.mode == STREAMED and window.block == 32


def test_a_window_across_two_different_reads_is_refused():
    """A process picks its read at construction and cannot change it, so two readings
    that disagree about the mode are two processes, or one payload that was parsed
    wrong. Subtracting them would produce a plausible window over nothing."""
    with pytest.raises(ValueError, match="same read"):
        ReadStats(mode=STREAMED, calls=5).since(ReadStats(mode=RECTANGLE, calls=1))


def test_a_window_cannot_run_backwards():
    with pytest.raises(ValueError, match="earlier"):
        ReadStats(calls=1).since(ReadStats(calls=9))


def test_a_reading_survives_the_wire():
    stats = ReadStats(mode=STREAMED, block=64, calls=7, rows=21, score_cells=900, held_cells=84)
    assert ReadStats.from_dict(stats.as_dict()) == stats


def test_a_payload_with_no_mode_reads_as_the_rectangle():
    """The reading that makes a gate refuse rather than pass: an older process, or a
    section that did not survive a merge, is the default read and not the new one."""
    assert ReadStats.from_dict({"calls": 3}).mode == RECTANGLE


def test_a_reading_renders_the_mode_the_calls_and_the_saving():
    line = ReadStats(mode=STREAMED, block=32, calls=4, rows=16, score_cells=8192,
                     held_cells=256).render()
    assert "streamed" in line and "4" in line and "32.0x" in line


# --- the backend a reading ran on (Day 62) --------------------------------------------


def test_a_reading_that_has_run_nothing_names_no_backend():
    """"" is its own answer. A read is constructed before any tensor exists, so
    there is a window in which the process genuinely does not know which backend it
    will get, and reporting a guess for it would be the one thing this payload is
    supposed to stop."""
    assert ReadStats().backend == ""
    assert "on" not in ReadStats(calls=3).render()


def test_a_reading_renders_the_backend_it_ran_on():
    """Day 62 splits what was asked for from what the box could give. A `streamed`
    server on a machine with no Triton is `tlsim`, which is correct and three orders
    of magnitude slower, and it differs from the kernel in no other field here."""
    line = ReadStats(mode=STREAMED, block=32, backend="tlsim", calls=4, rows=16,
                     score_cells=8192, held_cells=256).render()
    assert "streamed" in line and "on tlsim" in line


def test_the_backend_survives_the_wire():
    stats = ReadStats(mode=STREAMED, block=64, backend="triton", calls=7, rows=21,
                      score_cells=900, held_cells=84)
    assert ReadStats.from_dict(stats.as_dict()) == stats


def test_a_payload_with_no_backend_reads_as_unknown_rather_than_as_the_fallback():
    """An older process did not publish the field, and "tlsim" would be a claim about
    it. The empty string is the only honest default: `mode` can be guessed because
    the rectangle is what a process that never heard of the flag was running, and a
    backend cannot, because both of them predate the field."""
    assert ReadStats.from_dict({"calls": 3, "mode": STREAMED}).backend == ""


def test_a_window_across_two_backends_is_refused():
    """A device does not acquire Triton mid-run. Two readings that disagree are two
    processes, and their per-call costs would average into a number describing
    neither."""
    with pytest.raises(ValueError, match="same backend"):
        ReadStats(mode=STREAMED, backend="triton", calls=5).since(
            ReadStats(mode=STREAMED, backend="tlsim", calls=1)
        )


def test_a_window_from_before_the_first_read_is_allowed():
    """The one allowance, and it is the common case: a harness takes its baseline the
    moment the server is up, which is before any decode step has run, so the earlier
    reading has no backend to disagree with."""
    window = ReadStats(mode=STREAMED, backend="tlsim", calls=9).since(
        ReadStats(mode=STREAMED, calls=0)
    )
    assert window.calls == 9 and window.backend == "tlsim"


# --- PagedRead: the dispatch ----------------------------------------------------------


def _pool(slots=32, n_kv=2, d=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(slots, n_kv, d), torch.randn(slots, n_kv, d)


def _batch(rows, width, lens, n_q=8, d=8, seed=1):
    torch.manual_seed(seed)
    mapping = torch.stack(
        [torch.tensor([(r * 7 + i) % 32 for i in range(width)]) for r in range(rows)]
    )
    return torch.randn(rows, n_q, 1, d), mapping, torch.tensor(lens)


def test_an_unknown_read_is_refused_by_name():
    with pytest.raises(ValueError, match="rectangle"):
        PagedRead(mode="flash")


def test_a_tile_holds_at_least_one_key():
    with pytest.raises(ValueError, match="at least one key"):
        PagedRead(mode=STREAMED, block=0)


def test_the_rectangle_read_is_the_oracle_bit_for_bit():
    """Not "to a few ulps". The default read *is* `paged_attention_batched_reference`,
    so anything less than equality here means the dispatch changed the arithmetic on
    the path that was supposed to be untouched."""
    k, v = _pool()
    q, mapping, lens = _batch(3, 6, [6, 4, 2])
    got = PagedRead()(q, k, v, mapping, lens, n_rep=4)
    want = paged_attention_batched_reference(q, k, v, mapping, lens, n_rep=4)
    assert torch.equal(got, want)


@pytest.mark.parametrize("block", [1, 3, 8, 64])
def test_the_streamed_read_agrees_with_the_rectangle_one(block):
    k, v = _pool()
    q, mapping, lens = _batch(4, 7, [7, 5, 1, 3])
    got = PagedRead(mode=STREAMED, block=block)(q, k, v, mapping, lens, n_rep=4)
    want = PagedRead()(q, k, v, mapping, lens, n_rep=4)
    assert torch.allclose(got, want, atol=1e-5)


def test_the_rectangle_read_holds_every_cell_it_is_charged():
    k, v = _pool()
    q, mapping, lens = _batch(3, 6, [6, 4, 2])
    read = PagedRead()
    read(q, k, v, mapping, lens, n_rep=4)
    stats = read.stats()
    assert stats.calls == 1 and stats.rows == 3
    assert stats.score_cells == 3 * 8 * 6
    assert stats.held_cells == stats.score_cells
    assert stats.saving == 1.0


def test_the_streamed_read_holds_a_tile_and_is_charged_the_row():
    """Both numbers on every call, which is the whole point of counting here rather
    than in `streambench.py`: the saving a live server got is a ratio of two things
    that happened, not a ratio the bench computed about shapes it made up."""
    k, v = _pool()
    q, mapping, lens = _batch(4, 8, [8, 8, 8, 8])
    read = PagedRead(mode=STREAMED, block=2)
    read(q, k, v, mapping, lens, n_rep=4)
    stats = read.stats()
    assert stats.score_cells == 4 * 8 * 8
    assert stats.held_cells == 4 * 8 * 2
    assert stats.saving == 4.0


def test_a_read_on_host_tensors_reports_the_cpu_model_it_actually_ran():
    """The dispatch is real: a CPU tensor cannot reach a jitted kernel whatever is
    installed, so a streamed read on this box is `tlsim` and says so. This is the
    assertion that separates "the flag reached the cache" from "the kernel ran"."""
    k, v = _pool()
    q, mapping, lens = _batch(3, 6, [6, 4, 2])
    read = PagedRead(mode=STREAMED, block=4)
    read(q, k, v, mapping, lens, n_rep=4)
    assert read.stats().backend == "tlsim"


def test_the_rectangle_read_reports_torch_because_it_has_no_choice_to_make():
    """One gather and two matmuls, which is torch on a card and torch on a laptop.
    Naming it anyway keeps the field non-empty on the default path, so an empty
    backend means "nothing has run" and never "this read does not report"."""
    k, v = _pool()
    q, mapping, lens = _batch(3, 6, [6, 4, 2])
    read = PagedRead()
    read(q, k, v, mapping, lens, n_rep=4)
    assert read.stats().backend == "torch"


def test_a_read_that_was_refused_names_no_backend():
    """Recorded after the call, like the counters and for the same reason: a call
    that raised ran on nothing, and a payload that claims otherwise would make a
    server that never completed a decode step look like one that did."""
    k, v = _pool()
    q, mapping, lens = _batch(2, 4, [4, 9])  # a row claiming more history than it has
    read = PagedRead(mode=STREAMED, block=2)
    with pytest.raises(ValueError, match="more history"):
        read(q, k, v, mapping, lens, n_rep=4)
    assert read.stats().backend == ""
    assert read.stats().calls == 0


def test_a_tile_wider_than_the_context_is_charged_the_context():
    k, v = _pool()
    q, mapping, lens = _batch(2, 4, [4, 3])
    read = PagedRead(mode=STREAMED, block=512)
    read(q, k, v, mapping, lens, n_rep=4)
    assert read.stats().saving == 1.0


def test_the_counters_are_cumulative_over_calls():
    k, v = _pool()
    q, mapping, lens = _batch(3, 6, [6, 4, 2])
    read = PagedRead()
    for _ in range(3):
        read(q, k, v, mapping, lens, n_rep=4)
    assert read.stats().calls == 3
    assert read.stats().rows == 9


def test_a_reading_is_a_moment_and_does_not_move_afterwards():
    k, v = _pool()
    q, mapping, lens = _batch(3, 6, [6, 4, 2])
    read = PagedRead()
    read(q, k, v, mapping, lens, n_rep=4)
    snapshot = read.stats()
    read(q, k, v, mapping, lens, n_rep=4)
    assert snapshot.calls == 1 and read.stats().calls == 2


@pytest.mark.parametrize("mode", [RECTANGLE, STREAMED])
def test_both_reads_refuse_the_inputs_the_oracle_refuses(mode):
    """Day 59's argument, moved up a layer. A dispatch that softened a refusal on one
    branch would make the two reads differ in what they accept rather than in what
    they hold, and the first symptom would be a crash on one flag only."""
    k, v = _pool()
    q, mapping, lens = _batch(3, 6, [6, 4, 2])
    read = PagedRead(mode=mode)
    with pytest.raises(ValueError):
        read(q.expand(3, 8, 2, 8), k, v, mapping, lens, n_rep=4)


@pytest.mark.parametrize("mode", [RECTANGLE, STREAMED])
def test_validated_skips_the_bounds_check_on_both_reads(mode):
    """The plan path hands `validated=True` because the lengths were checked on the
    host when the plan was built. It has to mean the same thing on both branches or
    Day 50's whole argument holds for one flag and not the other."""
    k, v = _pool()
    q, mapping, lens = _batch(3, 6, [6, 4, 2])
    read = PagedRead(mode=mode)
    out = read(q, k, v, mapping, lens, n_rep=4, validated=True)
    assert out.shape == (3, 8, 1, 8)


# --- the cache picks one --------------------------------------------------------------


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


def _cache(**kw) -> BatchedPagedKVCache:
    cfg = _tiny_config()
    return BatchedPagedKVCache(
        cfg, BlockAllocator(num_blocks=16, block_size=4), batch_size=3, **kw
    )


def test_a_cache_defaults_to_the_rectangle_read():
    """The default is the day's one load-bearing decision. A tlsim loop in Python is
    two orders of magnitude slower than the torch path it replaces, so a flag that
    defaulted on would make every engine in the repo correct and unusable."""
    assert _cache().read.mode == RECTANGLE


def test_a_cache_asked_for_the_streamed_read_gets_it():
    cache = _cache(streamed_read=True, read_block=64)
    assert cache.read.mode == STREAMED
    assert cache.read.block == 64


def _decode(cache, layer_k, layer_v, q, rows=(0, 1, 2)):
    return cache.paged_attention(0, layer_k, layer_v, q, n_rep=4, rows=rows)


def _prefill(cache, lengths):
    cfg = _tiny_config()
    width = max(lengths)
    torch.manual_seed(2)
    shape = (len(lengths), cfg.num_key_value_heads, width, cfg.head_dim)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    for layer in range(cfg.num_hidden_layers):
        cache.write(layer, torch.randn(*shape), torch.randn(*shape), mask)
    return mask


def test_a_decode_step_reads_the_same_attention_either_way():
    """The cache-level version of Day 59's agreement table, and it is a different
    claim: this one goes through `write`, through the gathered `[rows, max_ctx]`
    mapping and through `context_bounds`, none of which the kernel's own tests have."""
    cfg = _tiny_config()
    torch.manual_seed(3)
    k = torch.randn(3, cfg.num_key_value_heads, 1, cfg.head_dim)
    v = torch.randn(3, cfg.num_key_value_heads, 1, cfg.head_dim)
    q = torch.randn(3, cfg.num_attention_heads, 1, cfg.head_dim)
    outs = []
    for streamed in (False, True):
        cache = _cache(streamed_read=streamed, read_block=4)
        _prefill(cache, [6, 3, 1])
        outs.append(_decode(cache, k, v, q))
    assert torch.allclose(outs[0], outs[1], atol=1e-5)


def test_a_decode_step_counts_itself_on_the_cache_s_read():
    cache = _cache(streamed_read=True, read_block=4)
    _prefill(cache, [6, 3, 1])
    cfg = _tiny_config()
    torch.manual_seed(3)
    k = torch.randn(3, cfg.num_key_value_heads, 1, cfg.head_dim)
    v = torch.randn(3, cfg.num_key_value_heads, 1, cfg.head_dim)
    q = torch.randn(3, cfg.num_attention_heads, 1, cfg.head_dim)
    _decode(cache, k, v, q)
    stats = cache.read.stats()
    assert stats.calls == 1 and stats.rows == 3
    assert stats.saving > 1.0


def test_the_planned_path_goes_through_the_same_read():
    """The branch that would be easy to miss. `paged_attention` has two arms and the
    planned one is the one a served decode step actually takes, so a wiring that
    reached only the unplanned arm would pass every test above and change nothing in
    a server."""
    cache = _cache(streamed_read=True, read_block=4)
    _prefill(cache, [6, 3, 1])
    cfg = _tiny_config()
    torch.manual_seed(3)
    k = torch.randn(3, cfg.num_key_value_heads, 1, cfg.head_dim)
    v = torch.randn(3, cfg.num_key_value_heads, 1, cfg.head_dim)
    q = torch.randn(3, cfg.num_attention_heads, 1, cfg.head_dim)
    plan = cache.plan_decode(rows=(0, 1, 2))
    cache.paged_attention(0, k, v, q, n_rep=4, plan=plan)
    assert cache.read.stats().calls == 1


def test_a_row_view_reads_through_the_cache_s_choice():
    cache = _cache(streamed_read=True, read_block=4)
    _prefill(cache, [6, 3, 1])
    cfg = _tiny_config()
    torch.manual_seed(3)
    k = torch.randn(2, cfg.num_key_value_heads, 1, cfg.head_dim)
    v = torch.randn(2, cfg.num_key_value_heads, 1, cfg.head_dim)
    q = torch.randn(2, cfg.num_attention_heads, 1, cfg.head_dim)
    view = cache.view((0, 2))
    view.paged_attention(0, k, v, q, n_rep=4)
    assert cache.read.stats().rows == 2


# --- the flag travels -----------------------------------------------------------------


def test_engine_build_threads_the_read_to_its_cache():
    engine = Engine.build(_model(), num_blocks=16, block_size=4, streamed_read=True,
                          read_block=16)
    assert engine.cache.read.mode == STREAMED
    assert engine.cache.read.block == 16


def test_an_engine_built_without_the_flag_is_on_the_rectangle():
    assert Engine.build(_model(), num_blocks=16, block_size=4).cache.read.mode == RECTANGLE


def _launch_kwargs(**kw):
    cfg = _tiny_config()
    defaults = dict(
        weights_dir="unused",
        device="cpu",
        dtype="float32",
        block_size=4,
        max_batch_size=4,
        max_model_len=32,
        kv_cache_bytes=kv_bytes_per_block(cfg, 4, torch.float32) * 64,
        load=lambda _dir, **_kw: _weights(),
        read_config=lambda _dir: cfg,
    )
    defaults.update(kw)
    return defaults


def test_build_engine_threads_the_read():
    engine, _ = build_engine(**_launch_kwargs(streamed_read=True, read_block=8))
    assert engine.cache.read.mode == STREAMED
    assert engine.cache.read.block == 8


def test_an_engine_publishes_which_read_it_is_on():
    """The wiring's only witness. Two servers with the same weights, the same pool
    and the same flags except this one answer identically and take different code
    paths, so `/health` is where the difference becomes observable at all."""
    engine, _ = build_engine(**_launch_kwargs(streamed_read=True, read_block=8))
    section = AsyncEngine(engine).stats()["paged_read"]
    assert section["mode"] == STREAMED and section["block"] == 8


def test_the_default_read_is_published_too_rather_than_left_out():
    """Unlike the capture, which publishes nothing when it is off. A read is not
    optional: every decode step runs one, so a missing section would mean the
    reporting broke, and there is no configuration it could legitimately mean."""
    engine, _ = build_engine(**_launch_kwargs())
    assert AsyncEngine(engine).stats()["paged_read"]["mode"] == RECTANGLE


def test_the_read_section_reaches_the_health_payload():
    engine, _ = build_engine(**_launch_kwargs(streamed_read=True))
    payload = health_payload("nanoserve", {}, AsyncEngine(engine).stats())
    assert read_from_health(payload).mode == STREAMED


def test_a_payload_from_a_server_that_reports_no_read_is_refused():
    """Not read as the rectangle. Every nanoserve process publishes this section, so
    its absence is a harness pointed at something else, and defaulting it would make
    that arm silently pass a gate about a read it never observed."""
    with pytest.raises(Exception, match="no read"):
        read_from_health({"status": "ok"})


# --- the acceptance run, over two real servers ----------------------------------------


class ByteTokenizer:
    """Bytes as ids, so a prompt is deterministic without a tokenizer on disk."""

    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [b % 64 for b in text.encode()]

    def decode(self, ids, **kw) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


def _app(**kw):
    defaults = dict(
        tokenizer=ByteTokenizer(),
        **_launch_kwargs(),
    )
    defaults.update(kw)
    return build_app(**defaults)


#: Six clients over four slots, generating long enough that the crowd is still a
#: crowd when `/health` is polled. Two constraints pull against each other here. The
#: read under test is a tlsim loop in Python, so the streamed arm pays roughly ten
#: times the oracle's time per call and a long run is expensive; but `peak_running`
#: is *sampled*, so a run that finishes between two polls reports a crowd of zero and
#: `check_arm_was_crowded` refuses it, which is what the first draft of this constant
#: did. `paired_plans` rather than a hand-written list because it rotates sampling on
#: 3: a read that agrees to 1e-5 can still shift a draw, and the greedy requests
#: alone would not find that.
PLANS = paired_plans(6, prompts=("abc", "abcde"), max_tokens=(12, 16))


def test_the_same_engine_answers_the_same_bytes_on_either_read():
    """Day 60's acceptance claim, and the reason it is over a socket.

    Every test above compares tensors, and a read that agrees to 1e-5 is a read that
    can still put a different token on the wire: the sampler takes an argmax over
    logits that came out of this attention, and two floats a ulp apart on either side
    of a tie choose different words. So the comparison that settles it is the one
    downstream of the sampler, the detokeniser and the SSE framing, which is the
    text a client actually received.

    Two arms, one difference. Same weights, same pool, same scheduler, same
    tokenizer, and `--streamed-read` on one of them, so a disagreement is
    attributable to the read and to nothing else.
    """

    async def scenario():
        arms = []
        for name, app in (("graphs", _app(streamed_read=True, read_block=16)),
                          ("eager", _app())):
            with live_server(app) as server:
                arms.append(
                    await run_arm(server.base_url, PLANS, burst_arrivals(len(PLANS)),
                                  name=name)
                )
        return arms

    streamed, rectangle = asyncio.run(asyncio.wait_for(scenario(), 300.0))

    check_arm_was_crowded(streamed)
    check_same_answers(streamed, rectangle)

    check_arm_read(streamed, STREAMED)
    check_arm_read(rectangle, RECTANGLE)
    assert streamed.read.saving > 1.0
    assert rectangle.read.saving == 1.0


def test_a_server_on_the_default_read_fails_the_streamed_gate():
    """The control, and the reason the assertion above is a measurement. Same file,
    same claim, the flag left off: the answers are still right and the gate still
    refuses, so `check_arm_read` is asking about a flag rather than describing an
    engine."""

    async def scenario():
        with live_server(_app()) as server:
            return await run_arm(server.base_url, PLANS, burst_arrivals(len(PLANS)),
                                 name="eager")

    arm = asyncio.run(asyncio.wait_for(scenario(), 300.0))

    check_arm_was_crowded(arm)
    with pytest.raises(Exception, match="streamed"):
        check_arm_read(arm, STREAMED)
    assert arm.read.calls > 0
