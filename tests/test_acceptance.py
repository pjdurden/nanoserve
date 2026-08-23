"""Day 41 tests: the Week 11 acceptance test, over a real socket.

Every day of Phase 4 asserted something and none of them asserted all of it at
once. Day 37 proved one caller's future resolves with that caller's tokens. Day 39
proved a stream carries the same ids the unary endpoint would. Day 40 proved a
seeded request is byte-identical whether it ran alone or shared a batch with three
others. All three ran through `httpx.ASGITransport`, which calls the app as a
Python coroutine: no TCP, no HTTP parser, and no client that can actually vanish.

This file is the crowd. A real uvicorn on a real port, a dozen concurrent clients
mixing streaming and unary, greedy and seeded, four-token answers and
twenty-four-token ones, against a pool too small to hold them all, with some of
them hanging up halfway. Four claims, which are the ones the phase owes:

  1. **No request receives another request's tokens.** Checked as the strongest
     version of itself: every client's crowd answer is byte-identical to the answer
     the same server gave that same request when it ran alone.
  2. **A seed means the same thing in a crowd.** The same claim, but it is Day 40's
     per-request generator that has to hold it up, and this is the first time it is
     asked over HTTP with real concurrency rather than with a scripted batch.
  3. **The pool comes back.** When the crowd leaves, every block is free, every slot
     is free, and Day 35's ledger audit passes. Including the blocks of the clients
     that hung up, which is the half nobody frees for you.
  4. **The process survives clients that disappear.** A dropped connection is one
     caller's problem, not the server's, and every other client in the same crowd
     finishes normally.

The harness is `nanoserve.acceptance`, and it is tested here too: a check nobody
has watched fail is a comment, so each of the three checks is fired on purpose
before the acceptance run is allowed to mean anything.

Nothing here needs `./weights`. The tiny random model is the same one Day 37
onwards used, which is the point: this test is about the serving layer's behaviour
under concurrency, and a 1B model would only make it slower.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import torch

from nanoserve.acceptance import (
    AcceptanceFailure,
    ClientPlan,
    ClientResult,
    CrowdReport,
    check_answers,
    check_pool_returned,
    check_survivors,
    live_server,
    mixed_crowd,
    run_client,
    run_crowd,
    run_solo,
    wait_until_idle,
)
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.server import create_app
from nanoserve.serving import AsyncEngine

# --- the same tiny model and byte tokenizer every serving test uses -------------

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."


class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ALPHABET.index(ch) for ch in text]

    def decode(self, token_ids) -> str:
        return "".join(ALPHABET[i] for i in token_ids)


TOKENIZER = ByteTokenizer()


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


def _model(seed: int = 0) -> LlamaModel:
    torch.manual_seed(seed)
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg))


def _engine(num_blocks=32, block_size=4, max_batch_size=3) -> Engine:
    """Three slots, so a dozen clients queue for them rather than each getting one."""
    return Engine.build(
        _model(), num_blocks=num_blocks, block_size=block_size, max_batch_size=max_batch_size
    )


def _crowded_engine() -> Engine:
    """Three slots and 48 tokens of K/V, for a crowd that wants several hundred.

    Sized by measurement, not by feel. At 16 blocks the same crowd never preempts
    and the acceptance run is a concurrency test; at 12 it recomputes 37 positions,
    which means Day 33's eviction path is running underneath live HTTP connections
    while the answers are being compared. That is the version worth running, because
    "your tokens do not depend on who else was in the batch" is a much weaker claim
    when nobody was ever thrown out of it.

    Still above the admission floor: the longest plan here is 8 prompt tokens plus
    24 generated, which is 8 blocks, and `Scheduler.add_request` refuses anything
    whose worst case exceeds the whole pool. A pool of 11 would turn this file into
    a study of 400s.
    """
    return _engine(num_blocks=12)


def _app(engine: Engine):
    serving = AsyncEngine(engine)
    app = create_app(serving, TOKENIZER, vocab_size=64, eos_token_id=None)
    app.state.engine = engine
    app.state.serving = serving
    return app


def run(coro, timeout: float = 120.0):
    """`asyncio.run` with a deadline, so a wedged server fails one test not the suite."""

    async def guarded():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(guarded())


def _result(client_id="c0", status=200, text="abc", **kw) -> ClientResult:
    return ClientResult(client_id=client_id, status=status, text=text, **kw)


# --- the harness: a real uvicorn on a real port ---------------------------------


def test_live_server_binds_a_real_port_and_answers_health():
    """The whole difference from every earlier serving test is in this assertion.

    `base_url` is `http://127.0.0.1:<port>`, the port was chosen by the kernel, and
    the request below went out over a socket and came back through uvicorn's HTTP
    parser. Nothing in the app changed; what changed is that there is now a
    connection that a client can drop.
    """
    with live_server(_app(_engine())) as server:
        assert server.port > 0
        assert server.base_url == f"http://127.0.0.1:{server.port}"
        payload = httpx.get(f"{server.base_url}/health", timeout=10.0).json()
    assert payload["status"] == "ok"
    assert payload["loop_running"] is True


def test_live_server_starts_the_engine_loop_and_stops_it_on_exit():
    """The lifespan is uvicorn's job and the context manager has to prove it ran.

    A server whose loop never started answers `/health` and hangs every completion;
    a server whose loop outlives the process is a daemon thread nothing can join.
    Both are invisible unless something asserts on the flag at both ends.
    """
    app = _app(_engine())
    serving = app.state.serving
    with live_server(app):
        assert serving.running
    assert not serving.running


def test_live_server_frees_the_port_when_it_leaves():
    """Two servers in a row, same code, no address-already-in-use.

    The reason this is a test and not an assumption: a socket the harness forgot to
    close makes the *next* test in the file fail, which is the worst kind of
    failure to read. Binding twice proves the shutdown joined the thread.
    """
    with live_server(_app(_engine())) as first:
        first_port = first.port
    with live_server(_app(_engine())) as second:
        assert second.port > 0
    assert first_port > 0


# --- what a client plan is -------------------------------------------------------


def test_a_plan_becomes_the_body_the_server_expects():
    plan = ClientPlan(client_id="c1", prompt="abc", max_tokens=7, temperature=0.8, seed=3)
    body = plan.body("nanoserve")
    assert body == {
        "model": "nanoserve",
        "prompt": "abc",
        "max_tokens": 7,
        "stream": False,
        "temperature": 0.8,
        "top_k": 0,
        "top_p": 1.0,
        "seed": 3,
    }


def test_a_greedy_plan_sends_no_seed():
    """`seed: null` is a legal body and a lie about intent, so it is left out.

    The default plan is what every client of this server sent before Day 40, and
    the body it produces has to be exactly that body, or the acceptance test is
    quietly testing the sampled path for requests that asked for greedy.
    """
    body = ClientPlan(client_id="c1", prompt="abc").body("nanoserve")
    assert "seed" not in body
    assert body["temperature"] == 0.0
    assert not ClientPlan(client_id="c1", prompt="abc").is_sampled


def test_a_seeded_plan_says_so():
    plan = ClientPlan(client_id="c1", prompt="abc", temperature=0.9, seed=1)
    assert plan.is_sampled
    assert plan.hangs_up is False


def test_a_mixed_crowd_actually_mixes():
    """The crowd's own thermometer: a crowd of one flavour tests one flavour.

    Asserted here rather than in the acceptance run because a change to
    `mixed_crowd` that quietly dropped every stream would leave the acceptance test
    green and meaningless.
    """
    plans = mixed_crowd(12, prompts=["ab", "abcd", "abcdefgh"])
    assert len(plans) == 12
    assert len({p.client_id for p in plans}) == 12
    assert any(p.stream for p in plans)
    assert any(not p.stream for p in plans)
    assert any(p.is_sampled for p in plans)
    assert any(not p.is_sampled for p in plans)
    assert any(p.hangs_up for p in plans)
    assert len({p.max_tokens for p in plans}) > 1


def test_a_crowd_can_be_asked_for_no_hang_ups():
    """The baseline pass needs a crowd whose every client finishes.

    Same plans, same ids, same sampling, minus the disconnects, so a test that
    wants to isolate "did concurrency change an answer" is not also testing "did a
    disconnect change somebody else's answer".
    """
    plans = mixed_crowd(8, prompts=["ab", "abcd"], hang_up_every=0)
    assert not any(p.hangs_up for p in plans)


def test_every_client_in_a_crowd_gets_a_distinct_id():
    plans = mixed_crowd(40, prompts=["ab", "abcd", "abcdefgh"])
    assert len({p.client_id for p in plans}) == 40


# --- one client at a time --------------------------------------------------------


def test_a_unary_client_gets_the_offline_answer():
    """The socket changes nothing about the tokens, which is worth one assertion."""
    engine = _engine()
    prompt = "abcd"
    offline = _engine().generate([TOKENIZER.encode(prompt)], max_new_tokens=6)[0]
    expected = TOKENIZER.decode(offline[len(prompt) :])

    async def go():
        with live_server(_app(engine)) as server:
            async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
                return await run_client(client, ClientPlan("c0", prompt, max_tokens=6))

    result = run(go())
    assert result.ok
    assert result.text == expected
    assert result.usage["completion_tokens"] == 6


def test_a_streaming_client_reassembles_the_same_text():
    """Frames concatenate to the unary answer, over TCP this time.

    Day 39 asserted this through an ASGI transport, where the frames are yielded
    straight into the caller's coroutine. Here they are chunk-encoded, written to a
    socket, and reassembled by a real HTTP client, which is the path a user's curl
    takes and the one that can lose a delimiter.
    """
    engine = _engine()
    prompt = "abcd"

    async def go():
        with live_server(_app(engine)) as server:
            async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
                unary = await run_client(client, ClientPlan("c0", prompt, max_tokens=6))
                await wait_until_idle(client)
                streamed = await run_client(
                    client, ClientPlan("c1", prompt, max_tokens=6, stream=True)
                )
                return unary, streamed

    unary, streamed = run(go())
    assert streamed.ok
    assert streamed.text == unary.text
    assert streamed.frames > 1
    assert streamed.usage == unary.usage


def test_a_client_that_hangs_up_is_recorded_as_having_hung_up():
    """Not an error. A disconnect is a client's decision and the report says so.

    Folding it into `error` would make the survivors check unable to tell "the
    caller left" from "the server broke", which is the one distinction the fourth
    claim of this file rests on.
    """
    engine = _engine()

    async def go():
        with live_server(_app(engine)) as server:
            async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
                result = await run_client(
                    client,
                    ClientPlan("c0", "abcd", max_tokens=24, stream=True, hang_up_after_frames=2),
                )
                await wait_until_idle(client)
                return result

    result = run(go())
    assert result.hung_up
    assert result.error is None
    assert result.frames == 2


def test_a_hung_up_client_gives_its_blocks_back():
    """The abort path, driven by a socket closing rather than by a cancelled task.

    This is the assertion the ASGI transport could not make. There, "the client
    went away" is a Python generator being closed; here it is a FIN on a TCP
    connection that uvicorn notices, turns into an ASGI disconnect, and delivers as
    a cancellation into the handler, which is three more places for it to be lost.
    """
    engine = _engine()

    async def go():
        with live_server(_app(engine)) as server:
            async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
                await run_client(
                    client,
                    ClientPlan("c0", "abcd", max_tokens=64, stream=True, hang_up_after_frames=2),
                )
                await wait_until_idle(client)

    run(go())
    check_pool_returned(engine)


def test_a_unary_client_that_times_out_also_gives_its_blocks_back():
    """The other half of the same claim, and the one that was false. Day 41.

    A unary caller has no frames to count, so the only handle it has on "leave
    halfway" is a deadline. What the server does with that is a different code path
    from the streaming one, and until today it did nothing at all: uvicorn delivers
    a disconnect as a message on `receive`, and a handler parked on
    `await serving.generate(...)` never reads messages. Day 37's abort was real and
    nothing ever triggered it over a socket.
    """
    engine = _engine(num_blocks=256)

    async def go():
        with live_server(_app(engine)) as server:
            async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
                result = await run_client(
                    client,
                    ClientPlan("c0", "abcd", max_tokens=800, hang_up_after_seconds=0.2),
                )
                await wait_until_idle(client)
                return result, engine

    result, engine = run(go())
    assert result.hung_up
    # The assertion that found the bug. Before Day 41 this read 800: the socket was
    # gone at 0.2s and the engine kept the row and spent the whole budget on it,
    # because nothing was listening for the disconnect. `check_pool_returned` passed
    # even then, since the request eventually finished normally and gave everything
    # back, which is exactly why a counter had to be asserted on and not just a
    # ledger.
    assert engine.collected_tokens < 800
    check_pool_returned(engine)


def test_a_streaming_client_that_leaves_stops_costing_the_engine():
    """The half that already worked, asserted so a regression cannot hide behind it.

    Starlette watches for `http.disconnect` while it is producing a response body,
    so a stream has always been cancelled promptly. That asymmetry is the reason the
    unary bug survived: the streaming tests proved disconnects were handled, and
    "handled" was true of exactly one of the two endpoints.
    """
    engine = _engine()

    async def go():
        with live_server(_app(engine)) as server:
            async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
                await run_client(
                    client,
                    ClientPlan("c0", "abcd", max_tokens=64, stream=True, hang_up_after_frames=2),
                )
                await wait_until_idle(client)

    run(go())
    assert engine.collected_tokens < 64
    check_pool_returned(engine)


def test_a_refused_request_is_reported_as_a_failure_not_a_hang_up():
    """A 400 is an answer. The report keeps the status so a check can tell them apart."""
    engine = _engine()

    async def go():
        with live_server(_app(engine)) as server:
            async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
                return await run_client(client, ClientPlan("c0", "abcd", max_tokens=4), model="nope")

    result = run(go())
    assert result.status == 404
    assert not result.ok
    assert not result.hung_up
    assert result.error


# --- the checks, fired on purpose ------------------------------------------------


def test_check_answers_passes_when_every_crowd_answer_matches_its_solo_run():
    solo = {"c0": _result("c0", text="xy"), "c1": _result("c1", text="zw")}
    crowd = CrowdReport(results={"c0": _result("c0", text="xy"), "c1": _result("c1", text="zw")})
    check_answers(crowd, solo)


def test_check_answers_fires_when_one_client_got_somebody_elses_tokens():
    """The failure this whole file exists to catch, staged by hand.

    `c1` comes back with `c0`'s text, which is what a shared sampler, a mis-indexed
    row, or a cache row addressing the wrong block table produces, and which every
    other assertion in the suite would happily call a 200.
    """
    solo = {"c0": _result("c0", text="xy"), "c1": _result("c1", text="zw")}
    crowd = CrowdReport(results={"c0": _result("c0", text="xy"), "c1": _result("c1", text="xy")})
    with pytest.raises(AcceptanceFailure, match="c1"):
        check_answers(crowd, solo)


def test_check_answers_ignores_clients_that_hung_up():
    """A client that left has no answer to compare, and a partial one is not a bug."""
    solo = {"c0": _result("c0", text="xy", hung_up=True, status=None)}
    crowd = CrowdReport(results={"c0": _result("c0", text="x", hung_up=True, status=None)})
    check_answers(crowd, solo)


def test_check_answers_fires_when_a_client_has_no_solo_run_to_compare_with():
    """Silence is not agreement: a missing baseline would make the check vacuous."""
    crowd = CrowdReport(results={"c0": _result("c0", text="xy")})
    with pytest.raises(AcceptanceFailure, match="no solo"):
        check_answers(crowd, {})


def test_check_survivors_passes_when_only_the_hang_ups_are_missing():
    report = CrowdReport(
        results={
            "c0": _result("c0", text="xy"),
            "c1": _result("c1", text="", status=None, hung_up=True),
        }
    )
    check_survivors(report)


def test_check_survivors_fires_when_a_client_that_stayed_did_not_get_an_answer():
    """One caller's disconnect must not become another caller's 500."""
    report = CrowdReport(
        results={
            "c0": _result("c0", text="xy"),
            "c1": _result("c1", text="", status=503, error="loop stopped"),
        }
    )
    with pytest.raises(AcceptanceFailure, match="c1"):
        check_survivors(report)


def test_check_pool_returned_fires_on_a_block_that_never_came_back():
    """A leaked block changes no answer, which is why it needs its own assertion."""
    engine = _engine()
    engine.allocator.allocate()
    with pytest.raises(AcceptanceFailure, match="block"):
        check_pool_returned(engine)


def test_check_pool_returned_fires_on_a_slot_that_never_came_back():
    engine = _engine()
    engine.scheduler._free_slots.pop()
    with pytest.raises(AcceptanceFailure, match="slot"):
        check_pool_returned(engine)


def test_check_pool_returned_reports_a_broken_ledger_as_an_acceptance_failure():
    """Day 35's audit, re-raised in this layer's vocabulary rather than swallowed."""
    engine = _engine()
    engine.allocator._free.append(0)
    with pytest.raises(AcceptanceFailure):
        check_pool_returned(engine)


# --- the acceptance run ----------------------------------------------------------


def test_the_crowd_runs_concurrently_at_all():
    """Before believing the crowd proved anything, prove there was a crowd.

    `peak_running` is sampled from `/health` while the clients are in flight, and it
    is this file's version of Day 35's `max_preemptions`: a run that never had two
    rows in one batch has tested the serving layer one request at a time, which
    every earlier day already did.
    """
    engine = _crowded_engine()
    plans = mixed_crowd(12, prompts=["ab", "abcd", "abcdefgh"], hang_up_every=0)

    async def go():
        with live_server(_app(engine)) as server:
            return await run_crowd(server.base_url, plans)

    report = run(go())
    assert report.peak_running >= 2
    assert report.peak_active >= 2


@pytest.mark.parametrize("hang_up_every", [0, 5])
def test_the_week_11_acceptance_run(hang_up_every):
    """The whole phase, in one test, twice: a clean crowd and a crowd that leaves.

    Solo first, one client at a time on the same server, which is the baseline every
    later comparison is against and which is only honest because it is the *same*
    engine, the same weights and the same tokenizer. Then the same plans all at once.
    Then the three checks.

    Parametrised on the disconnects rather than split into two tests because the
    interesting question is whether the second run differs from the first, and a
    hang-up that corrupted a neighbour's answer would show up as `check_answers`
    failing only for `hang_up_every=4`.
    """
    engine = _crowded_engine()
    plans = mixed_crowd(12, prompts=["ab", "abcd", "abcdefgh"], hang_up_every=hang_up_every)

    async def go():
        with live_server(_app(engine)) as server:
            solo = await run_solo(server.base_url, plans)
            crowd = await run_crowd(server.base_url, plans)
            async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
                await wait_until_idle(client)
                health = (await client.get("/health")).json()
            return solo, crowd, health

    solo, crowd, health = run(go())

    # 1 and 2: nobody got anybody else's tokens, seeded or greedy.
    check_answers(crowd, solo)
    # 4: the only clients without an answer are the ones that left.
    check_survivors(crowd)
    # 3: everything came back.
    check_pool_returned(engine)

    assert health["status"] == "ok"
    assert health["running"] == 0 and health["waiting"] == 0
    assert crowd.peak_running >= 2
    # The run was hard enough to be worth believing: rows really were evicted and
    # replayed while these sockets were open, which is the condition claim 1 is
    # least likely to survive and the one no earlier serving test ever created.
    assert engine.recomputed_tokens > 0
    assert len(crowd.completed) + len(crowd.hung_up) == len(plans)
    assert bool(crowd.hung_up) == (hang_up_every > 0)


def test_a_seeded_request_is_byte_identical_alone_and_in_a_crowd():
    """Day 40's claim, made over HTTP by clients that really are concurrent.

    The scripted version of this test batched a request with three others by
    calling `step` itself. Here the batch composition is whatever uvicorn and the
    kernel scheduler produce, which is exactly the variable a seed is supposed to
    be independent of, and which no test can pin down by construction.
    """
    engine = _crowded_engine()
    seeded = ClientPlan("seeded", "abcd", max_tokens=12, temperature=0.9, top_p=0.9, seed=7)
    noise = mixed_crowd(9, prompts=["ab", "abcdefgh"], hang_up_every=0, start=1)

    async def go():
        with live_server(_app(engine)) as server:
            alone = await run_solo(server.base_url, [seeded])
            crowd = await run_crowd(server.base_url, [seeded, *noise])
            return alone, crowd

    alone, crowd = run(go())
    assert crowd.results["seeded"].ok
    assert crowd.results["seeded"].text == alone["seeded"].text
    assert crowd.peak_running >= 2


def test_the_crowd_leaves_the_server_able_to_serve_again():
    """A pool that came back is worth nothing if the next request cannot use it.

    The failure this catches is a pool that is arithmetically free and practically
    poisoned: a stale block table, a cache row still addressing a departed request's
    blocks, a slot on the free list that a row still owns. All of those pass
    `check_pool_returned` and answer the next caller with somebody else's K/V.
    """
    engine = _crowded_engine()
    plans = mixed_crowd(12, prompts=["ab", "abcd", "abcdefgh"], hang_up_every=3)
    prompt = "abcd"
    offline = _engine().generate([TOKENIZER.encode(prompt)], max_new_tokens=8)[0]
    expected = TOKENIZER.decode(offline[len(prompt) :])

    async def go():
        with live_server(_app(engine)) as server:
            await run_crowd(server.base_url, plans)
            async with httpx.AsyncClient(base_url=server.base_url, timeout=30.0) as client:
                await wait_until_idle(client)
                return await run_client(client, ClientPlan("after", prompt, max_tokens=8))

    result = run(go())
    assert result.ok
    assert result.text == expected
