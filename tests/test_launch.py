"""Day 38 tests: turning a card and a directory into an engine that can be curled.

Everything before today took `num_blocks` as an argument. Every test, every
benchmark, every fixture picked a number that made the scenario work, and the
engine believed it. A launcher does not get to do that. It is handed a directory
of weights and a device, and it has to answer one question in bytes: **how many
blocks fit?**

Three things make that question harder than the division it looks like.

1. **A block's size is a property of the config, and GQA is a factor of four in
   it.** K and V, for every layer, for `block_size` tokens, over the *KV* heads.
   Size it off the query heads and the pool comes out a quarter of what the card
   could hold, and nothing anywhere raises.
2. **The budget is what is left after the weights land, minus what a forward
   will transiently want.** The second term is not small: at the worst shape this
   server admits it is larger than the weights, and its biggest line is a tensor
   the engine throws away.
3. **A pool that cannot hold one request is a server that refuses everything.**
   Day 33's `Scheduler.add_request` rejects a request whose worst case exceeds the
   whole pool, so a launcher that boots under that floor has built a process that
   answers 400 to every caller and 200 to its own health check.

Nothing here needs `./weights` or a GPU. The memory probe is injected, the peak
allocator stats are injected, and the model is the same tiny random one the engine
and server tests use, which is the point: sizing is arithmetic over a config, and
arithmetic can be checked exactly.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import torch

from nanoserve.config import ModelConfig
from nanoserve.launch import (
    ActivationEstimate,
    KVPoolPlan,
    PoolTooSmall,
    build_app,
    build_engine,
    estimate_activation_bytes,
    kv_budget_bytes,
    kv_bytes_per_block,
    measure_activation_bytes,
    place_weights,
    plan_kv_pool,
    profile_prefill_bytes,
    resolve_device,
    resolve_dtype,
    weights_bytes,
)
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel

GIB = 1024**3
MIB = 1024**2


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
    # The tie, exactly as `load_weights` builds it: one storage, two names.
    tensors[LM_HEAD] = tensors[EMBED]
    return Weights(tensors, config)


def _tiny_model(seed: int = 0) -> LlamaModel:
    cfg = _tiny_config()
    return LlamaModel(cfg, _tiny_weights(cfg, seed))


def _probe(free: int, total: int):
    """Stand in for `torch.cuda.mem_get_info`, which this box cannot answer."""

    def probe(device):
        return free, total

    return probe


# --- what one block costs -------------------------------------------------------


def test_a_block_holds_k_and_v_for_every_layer():
    cfg = _tiny_config()
    # 2 (K and V) * 2 layers * 4 tokens * 2 kv heads * 4 head_dim * 4 bytes
    assert kv_bytes_per_block(cfg, block_size=4, dtype=torch.float32) == 512


def test_a_block_is_sized_off_the_kv_heads_not_the_query_heads():
    """The GQA trap. 32 query heads share 8 KV heads, so a block is 4x smaller
    than the query-head arithmetic says, and the difference is silent: the pool is
    a quarter the size it could be and every test still passes."""
    gqa = _tiny_config(num_attention_heads=8, num_key_value_heads=2)
    mha = _tiny_config(num_attention_heads=8, num_key_value_heads=8)
    wrong = kv_bytes_per_block(mha, block_size=4, dtype=torch.float32)
    right = kv_bytes_per_block(gqa, block_size=4, dtype=torch.float32)
    assert wrong == right * gqa.num_kv_groups == right * 4


def test_a_block_scales_with_the_dtype_it_will_hold():
    cfg = _tiny_config()
    fp32 = kv_bytes_per_block(cfg, block_size=4, dtype=torch.float32)
    bf16 = kv_bytes_per_block(cfg, block_size=4, dtype=torch.bfloat16)
    assert fp32 == 2 * bf16


def test_a_llama_3_2_1b_block_is_exactly_half_a_mebibyte():
    """The real number this launcher divides by: 16 layers, 8 KV heads, 64 head
    dim, 16 tokens, bf16. It comes out round, which makes the pool easy to read:
    one gibibyte of KV is 2048 blocks, and 2048 blocks is 32768 tokens."""
    assert kv_bytes_per_block(ModelConfig(), block_size=16, dtype=torch.bfloat16) == 512 * 1024
    plan = plan_kv_pool(
        ModelConfig(),
        budget_bytes=GIB,
        block_size=16,
        max_batch_size=8,
        max_model_len=2048,
        dtype=torch.bfloat16,
    )
    assert plan.num_blocks == 2048
    assert plan.capacity_tokens == 32768


# --- what a forward transiently wants -------------------------------------------


def test_the_activation_estimate_names_its_three_terms():
    est = estimate_activation_bytes(
        _tiny_config(), max_batch_size=2, max_model_len=8, dtype=torch.float32
    )
    assert isinstance(est, ActivationEstimate)
    assert est.scores_bytes == 2 * 8 * 8 * 8 * 4  # batch * heads * len * len * itemsize
    assert est.mlp_bytes == 3 * 2 * 8 * 48 * 4  # gate, up and their product, all live
    assert est.logits_bytes == 2 * 8 * 64 * 4
    assert est.total_bytes == est.scores_bytes + est.mlp_bytes + est.logits_bytes


def test_the_scores_term_is_quadratic_in_the_prefill_length():
    short = estimate_activation_bytes(
        _tiny_config(), max_batch_size=1, max_model_len=8, dtype=torch.float32
    )
    long = estimate_activation_bytes(
        _tiny_config(), max_batch_size=1, max_model_len=16, dtype=torch.float32
    )
    assert long.scores_bytes == 4 * short.scores_bytes
    assert long.logits_bytes == 2 * short.logits_bytes


def test_the_logits_rectangle_is_the_biggest_line_on_a_real_prefill():
    """The finding that pays for this function. `LlamaModel.forward` returns logits
    for every position so a prefill can be compared to HF token for token, and at
    8x2048 over a 128k vocab that rectangle is larger than the weights. The engine
    reads exactly one row of it."""
    est = estimate_activation_bytes(
        ModelConfig(), max_batch_size=8, max_model_len=2048, dtype=torch.bfloat16
    )
    assert est.logits_bytes > est.scores_bytes > est.mlp_bytes
    assert est.logits_bytes > 3 * GIB
    assert est.total_bytes > 6 * GIB


def test_measure_reports_the_transient_and_not_what_was_already_resident():
    """The peak allocator stat is cumulative, so the resident weights are inside it
    unless you subtract the level at the reset."""
    marks = []
    stats = {"peak": 900}

    def reset():
        marks.append("reset")
        stats["peak"] = 900  # the weights, already on the card

    def peak():
        return stats["peak"]

    def run():
        marks.append("run")
        stats["peak"] = 1400

    assert measure_activation_bytes(run, reset=reset, peak=peak) == 500
    assert marks == ["reset", "run"]


def test_the_profile_forward_runs_the_worst_rectangle_this_server_admits():
    seen = {}
    model = _tiny_model()
    real_forward = model.forward

    def spy(input_ids, *a, **kw):
        seen["shape"] = tuple(input_ids.shape)
        return real_forward(input_ids, *a, **kw)

    model.forward = spy
    stats = {"peak": 0}
    measured = profile_prefill_bytes(
        model,
        max_batch_size=3,
        max_model_len=5,
        device=torch.device("cpu"),
        reset=lambda: stats.update(peak=100),
        peak=lambda: stats["peak"],
    )
    assert seen["shape"] == (3, 5)
    assert measured == 0  # the fake peak never moved; the shape is the assertion


# --- the budget -----------------------------------------------------------------


def test_the_budget_is_what_is_free_minus_the_reserve_minus_the_activations():
    budget = kv_budget_bytes(
        torch.device("cuda"),
        activation_bytes=2 * GIB,
        utilization=0.90,
        probe=_probe(free=20 * GIB, total=24 * GIB),
    )
    # 20 free, minus 10% of the card's 24 held back, minus 2 for the forward.
    assert budget == 20 * GIB - int(0.10 * 24 * GIB) - 2 * GIB


def test_the_reserve_is_measured_against_the_card_not_against_what_is_free():
    """`gpu_memory_utilization` is a promise about the whole device, so a co-tenant
    comes out of your budget rather than out of the headroom. Two probes with the
    same free memory and different totals do not get the same budget."""
    small = kv_budget_bytes(
        torch.device("cuda"), activation_bytes=0, utilization=0.90, probe=_probe(8 * GIB, 8 * GIB)
    )
    shared = kv_budget_bytes(
        torch.device("cuda"), activation_bytes=0, utilization=0.90, probe=_probe(8 * GIB, 24 * GIB)
    )
    assert small > shared
    assert shared == 8 * GIB - int(0.10 * 24 * GIB)


def test_a_budget_that_comes_out_negative_is_reported_as_zero():
    assert (
        kv_budget_bytes(
            torch.device("cuda"),
            activation_bytes=30 * GIB,
            utilization=0.90,
            probe=_probe(20 * GIB, 24 * GIB),
        )
        == 0
    )


def test_a_cpu_launch_has_no_vram_to_divide_and_says_so():
    with pytest.raises(RuntimeError, match="cpu"):
        kv_budget_bytes(torch.device("cpu"), activation_bytes=0)


# --- the plan -------------------------------------------------------------------


def test_the_plan_takes_every_whole_block_the_budget_holds_and_no_partial_one():
    cfg = _tiny_config()
    per_block = kv_bytes_per_block(cfg, block_size=4, dtype=torch.float32)
    plan = plan_kv_pool(
        cfg,
        budget_bytes=per_block * 10 + per_block // 2,
        block_size=4,
        max_batch_size=2,
        max_model_len=8,
        dtype=torch.float32,
    )
    assert isinstance(plan, KVPoolPlan)
    assert plan.num_blocks == 10
    assert plan.pool_bytes == per_block * 10
    assert plan.pool_bytes <= plan.budget_bytes
    assert plan.budget_bytes - plan.pool_bytes < per_block


def test_the_plan_reports_how_many_max_length_requests_it_can_hold_at_once():
    cfg = _tiny_config()
    per_block = kv_bytes_per_block(cfg, block_size=4, dtype=torch.float32)
    plan = plan_kv_pool(
        cfg,
        budget_bytes=per_block * 10,
        block_size=4,
        max_batch_size=4,
        max_model_len=8,
        dtype=torch.float32,
    )
    assert plan.blocks_per_request == 2  # 8 tokens / 4 per block
    assert plan.concurrency_at_max_len == 5  # 10 blocks / 2, so the slots bind first
    assert plan.capacity_tokens == 40


def test_a_pool_that_cannot_hold_one_max_length_request_is_refused_at_boot():
    """Day 33's `Scheduler.add_request` rejects a request whose worst case exceeds
    the whole pool. Booting under that floor produces a process that is healthy and
    answers 400 to everybody, which is worse than not booting."""
    cfg = _tiny_config()
    per_block = kv_bytes_per_block(cfg, block_size=4, dtype=torch.float32)
    with pytest.raises(PoolTooSmall) as exc:
        plan_kv_pool(
            cfg,
            budget_bytes=per_block * 3,
            block_size=4,
            max_batch_size=2,
            max_model_len=64,
            dtype=torch.float32,
        )
    # It says what it could have served, so the flag to change is obvious.
    assert "12" in str(exc.value)  # 3 blocks * 4 tokens
    assert "64" in str(exc.value)


def test_a_budget_too_small_for_a_single_block_is_refused_too():
    cfg = _tiny_config()
    with pytest.raises(PoolTooSmall):
        plan_kv_pool(
            cfg,
            budget_bytes=8,
            block_size=4,
            max_batch_size=1,
            max_model_len=4,
            dtype=torch.float32,
        )


def test_the_serving_length_is_clamped_to_what_the_model_was_trained_for():
    cfg = _tiny_config(max_position_embeddings=16)
    per_block = kv_bytes_per_block(cfg, block_size=4, dtype=torch.float32)
    plan = plan_kv_pool(
        cfg,
        budget_bytes=per_block * 100,
        block_size=4,
        max_batch_size=2,
        max_model_len=10_000,
        dtype=torch.float32,
    )
    assert plan.max_model_len == 16


def test_the_plan_prints_as_a_line_a_human_can_check():
    plan = plan_kv_pool(
        ModelConfig(),
        budget_bytes=8 * GIB,
        block_size=16,
        max_batch_size=8,
        max_model_len=2048,
        dtype=torch.bfloat16,
    )
    line = plan.describe()
    assert "16384 blocks" in line
    assert "262144 tokens" in line


# --- device and dtype -----------------------------------------------------------


def test_auto_device_falls_back_to_cpu_when_there_is_no_gpu():
    assert resolve_device("auto", cuda_available=lambda: False) == torch.device("cpu")
    assert resolve_device("auto", cuda_available=lambda: True) == torch.device("cuda")
    assert resolve_device("cpu", cuda_available=lambda: True) == torch.device("cpu")


def test_auto_dtype_is_the_configs_on_a_gpu_and_float32_on_a_cpu():
    """bfloat16 is what the weights were published in and what the card wants. On a
    CPU it is a correctness-neutral, performance-terrible choice, and every earlier
    day in this project loaded fp32 on CPU for exactly that reason."""
    cfg = ModelConfig()  # torch_dtype "bfloat16"
    assert resolve_dtype("auto", torch.device("cuda"), cfg) is torch.bfloat16
    assert resolve_dtype("auto", torch.device("cpu"), cfg) is torch.float32
    assert resolve_dtype("bfloat16", torch.device("cpu"), cfg) is torch.bfloat16


def test_an_unknown_dtype_is_refused_by_name():
    with pytest.raises(ValueError, match="fp8"):
        resolve_dtype("fp8", torch.device("cpu"), ModelConfig())


# --- placing the weights --------------------------------------------------------


def test_weights_bytes_counts_a_shared_storage_once():
    cfg = _tiny_config()
    w = _tiny_weights(cfg)
    assert w[LM_HEAD] is w[EMBED]
    embed = w[EMBED].numel() * w[EMBED].element_size()
    # Every name added up, which is what a naive walk over the bag reports.
    naive = sum(w[name].numel() * w[name].element_size() for name in w.keys())
    assert naive - weights_bytes(w) == embed


def test_placing_the_weights_keeps_the_tied_head_tied():
    """The whole point of the function. `{k: v.to(device) for k, v in ...}` is the
    obvious move and it hands `lm_head` and `embed_tokens` two separate copies,
    which on Llama-3.2-1B is 501 MiB of KV pool spent on a duplicate."""
    cfg = _tiny_config()
    w = _tiny_weights(cfg)
    naive = {name: w[name].to(torch.float64) for name in w.keys()}
    assert naive[LM_HEAD] is not naive[EMBED]

    placed = place_weights(w, torch.device("cpu"), dtype=torch.float64)
    assert placed[LM_HEAD] is placed[EMBED]
    assert weights_bytes(placed) == 2 * weights_bytes(w)  # fp32 -> fp64, and nothing else


def test_placing_the_weights_moves_every_tensor():
    cfg = _tiny_config()
    placed = place_weights(_tiny_weights(cfg), torch.device("cpu"), dtype=torch.bfloat16)
    assert all(placed[name].dtype is torch.bfloat16 for name in placed.keys())
    assert placed.config is cfg


# --- wiring it together ---------------------------------------------------------


def _build(**kw):
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


def test_build_engine_sizes_the_pool_from_the_budget_it_was_given():
    engine, plan = _build()
    assert plan.num_blocks == 32
    assert engine.allocator.num_blocks == 32
    assert engine.allocator.block_size == 4


def test_build_engine_gives_the_scheduler_and_the_cache_one_pool():
    engine, _ = _build()
    assert engine.cache.allocator is engine.scheduler.allocator
    assert engine.cache.batch_size == engine.scheduler.max_batch_size == 4


def test_an_explicit_block_count_overrides_the_sizing_entirely():
    engine, plan = _build(num_blocks=7, kv_cache_bytes=None)
    assert plan.num_blocks == 7
    assert engine.allocator.num_blocks == 7


def test_a_cpu_launch_without_a_byte_budget_is_refused_rather_than_guessed():
    with pytest.raises(RuntimeError, match="kv_cache_bytes"):
        _build(kv_cache_bytes=None)


def test_the_engine_that_comes_back_actually_generates():
    engine, _ = _build()
    out = engine.generate([[1, 2, 3]], max_new_tokens=4)[0]
    assert len(out) == 7


# --- the app --------------------------------------------------------------------


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
    )
    defaults.update(kw)
    return build_app(**defaults)


def _run(coro, timeout: float = 20.0):
    async def guarded():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(guarded())


def test_the_launched_app_answers_a_completion():
    app = _app()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://nano") as client:
            async with app.router.lifespan_context(app):
                return await client.post(
                    "/v1/completions",
                    json={"model": "nanoserve", "prompt": [1, 2, 3], "max_tokens": 4},
                )

    response = _run(scenario())
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 4


def test_health_reports_the_pool_the_launcher_chose():
    app = _app()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://nano") as client:
            async with app.router.lifespan_context(app):
                return await client.get("/health")

    payload = _run(scenario()).json()
    assert payload["status"] == "ok"
    assert payload["num_blocks"] == 32
    assert payload["max_model_len"] == 16
    assert payload["block_size"] == 4


def test_the_vocab_guard_is_wired_from_the_config_the_launcher_read():
    """The launcher knows the vocab; a caller sending an id outside it should get a
    400 rather than an IndexError in the embedding lookup."""
    app = _app()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://nano") as client:
            async with app.router.lifespan_context(app):
                return await client.post(
                    "/v1/completions",
                    json={"model": "nanoserve", "prompt": [99_999], "max_tokens": 2},
                )

    response = _run(scenario())
    assert response.status_code == 400
    assert "vocab" in response.json()["detail"]
