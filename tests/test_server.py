"""Day 37 tests: the OpenAI-shaped surface over the bridge.

The engine speaks token ids and the wire speaks JSON, so this layer is a
translation and a set of refusals. The translation is small enough to read in one
sitting: a prompt in, `max_tokens`, one choice out, a usage block. The refusals
are the part worth testing, because every one of them is a decision about what a
server owes a caller it cannot serve.

The rule this file enforces everywhere: **a parameter that would change the answer
is either honoured or refused, never accepted and ignored.** Day 37 wrote that rule
with `temperature=0.7` as its example, because an engine that only knows `argmax`
cannot serve it and a 200 that pretends otherwise is a wrong answer with a good
status code on it. Day 39 turned `stream` into a feature and Day 40 turns
`temperature`, `top_k`, `top_p` and `seed` into features, so what is left of the
rule here is the shape of the refusals that remain: `n`, an unknown model, a token
id outside the vocab, and a sampling parameter whose *value* is nonsense.

Nothing here needs `./weights`: the tiny random model from the engine tests, plus
a 64-symbol byte tokenizer, which is enough to prove that what comes out of the
socket is exactly what the offline engine emits for the same prompt.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from nanoserve.cache import KVCacheExhausted
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.server import (
    ClientGone,
    await_client_disconnect,
    create_app,
    run_until_client_leaves,
)
from nanoserve.serving import AsyncEngine

# --- the same tiny model, plus a tokenizer small enough to fit its vocab --------

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."


class ByteTokenizer:
    """A 64-symbol alphabet, so a test can read the model's output as text.

    The real server is handed a Llama tokenizer. Nothing in `server.py` depends on
    which one: it calls `encode` and `decode` and that is the whole contract, which
    is what lets these tests run without a 2GB download.
    """

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


def _engine(num_blocks=64, block_size=4, max_batch_size=4) -> Engine:
    return Engine.build(
        _model(), num_blocks=num_blocks, block_size=block_size, max_batch_size=max_batch_size
    )


def _offline(prompt: list[int], max_new_tokens: int, **kw) -> list[int]:
    """The tokens a fresh offline engine emits for this prompt, generation only."""
    full = _engine(**kw).generate([list(prompt)], max_new_tokens=max_new_tokens)[0]
    return full[len(prompt) :]


def run(coro, timeout: float = 20.0):
    """`asyncio.run` with a deadline, so a wedged loop fails one test not the suite."""

    async def guarded():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(guarded())


def _app(serving: AsyncEngine, **kw):
    return create_app(serving, TOKENIZER, vocab_size=64, **kw)


def _client(app) -> httpx.AsyncClient:
    """Speak HTTP to the app in-process: real routing and validation, no socket."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://nanoserve"
    )


async def _serving(engine=None, **app_kw):
    serving = AsyncEngine(engine if engine is not None else _engine())
    await serving.start()
    return serving, _app(serving, **app_kw)


def _body(**kw) -> dict:
    body = {"model": "nanoserve", "prompt": [1, 2, 3], "max_tokens": 4}
    body.update(kw)
    return body


# --- the translation ------------------------------------------------------------


def test_a_completion_is_exactly_what_the_offline_engine_emits():
    expected = TOKENIZER.decode(_offline([1, 2, 3], 4))

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body())

    response = run(scenario())
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == expected


def test_the_response_has_the_openai_completion_shape():
    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body())

    payload = run(scenario()).json()
    assert payload["object"] == "text_completion"
    assert payload["id"].startswith("cmpl-")
    assert isinstance(payload["created"], int)
    assert payload["model"] == "nanoserve"
    choice = payload["choices"][0]
    assert choice["index"] == 0
    assert choice["logprobs"] is None
    assert choice["finish_reason"] == "length"
    assert payload["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }


def test_a_string_prompt_goes_through_the_tokenizer():
    prompt = "hello"
    expected = TOKENIZER.decode(_offline(TOKENIZER.encode(prompt), 4))

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(prompt=prompt))

    payload = run(scenario()).json()
    assert payload["choices"][0]["text"] == expected
    assert payload["usage"]["prompt_tokens"] == len(prompt)


def test_the_stop_token_ends_the_answer_and_stays_out_of_the_text():
    """`append_token` keeps EOS because the cache really holds it, and says the
    decision to show it belongs to the detokenizer. This is that decision."""
    first_token = _offline([1, 2, 3], 1)[0]

    async def scenario():
        serving, app = await _serving(eos_token_id=first_token)
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(max_tokens=16))

    payload = run(scenario()).json()
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["text"] == ""
    # Counted, though: the model really did compute it.
    assert payload["usage"]["completion_tokens"] == 1


def test_health_reports_the_loop_and_the_engine():
    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            await client.post("/v1/completions", json=_body())
            return await client.get("/health")

    payload = run(scenario()).json()
    assert payload["status"] == "ok"
    assert payload["model"] == "nanoserve"
    assert payload["iterations"] >= 4
    assert payload["loop_running"] is True
    assert payload["active"] == 0


# --- the refusals ---------------------------------------------------------------


def test_streaming_is_no_longer_refused():
    """Day 37 answered `stream=true` with a 400 that named Week 11. This is that
    week: the refusal is gone and the same body returns an event stream. What it
    contains is `tests/test_stream.py`'s business; what matters here is that a
    parameter stopped being a promise and started being a feature."""

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(stream=True))

    response = run(scenario())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_a_sampling_temperature_is_no_longer_refused():
    """Day 37 made this a 400 that named itself. Day 40 is when it stops being one.

    Both halves are asserted: the request is served, and it is actually *sampled*.
    A 200 that quietly returned argmax tokens would be the exact failure the 400
    was protecting against, so the test compares five seeds against the greedy
    answer and demands that they are not all the same text.
    """

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            cold = await client.post("/v1/completions", json=_body(max_tokens=8))
            hot = [
                await client.post(
                    "/v1/completions",
                    json=_body(max_tokens=8, temperature=2.0, seed=seed),
                )
                for seed in range(5)
            ]
            return cold, hot

    cold, hot = run(scenario())
    assert cold.status_code == 200
    assert all(r.status_code == 200 for r in hot)
    greedy = cold.json()["choices"][0]["text"]
    assert any(r.json()["choices"][0]["text"] != greedy for r in hot)


def test_more_than_one_choice_is_refused():
    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(n=2))

    response = run(scenario())
    assert response.status_code == 400
    assert "n" in response.json()["detail"]


def test_an_unknown_model_is_404():
    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(model="gpt-4"))

    response = run(scenario())
    assert response.status_code == 404


def test_an_empty_prompt_is_400_and_never_reaches_the_engine():
    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            empty_list = await client.post("/v1/completions", json=_body(prompt=[]))
            empty_text = await client.post("/v1/completions", json=_body(prompt=""))
            return empty_list, empty_text, serving.engine.iterations

    empty_list, empty_text, iterations = run(scenario())
    assert empty_list.status_code == 400
    assert empty_text.status_code == 400
    assert iterations == 0


def test_a_token_id_outside_the_vocab_is_400_not_500():
    """Without the check this is an IndexError inside the embedding lookup, which
    is a 500 for something the caller got wrong and could fix."""

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(prompt=[1, 999]))

    response = run(scenario())
    assert response.status_code == 400
    assert "vocab" in response.json()["detail"]


def test_a_budget_too_large_for_the_pool_is_400():
    """The refusal comes back from inside the shared loop, on this caller's future,
    and has to become this caller's status code."""

    async def scenario():
        serving, app = await _serving(engine=_engine(num_blocks=8, block_size=4))
        async with serving, _client(app) as client:
            refused = await client.post("/v1/completions", json=_body(max_tokens=500))
            served = await client.post("/v1/completions", json=_body(max_tokens=4))
            return refused, served

    refused, served = run(scenario())
    assert refused.status_code == 400
    assert "blocks" in refused.json()["detail"]
    assert served.status_code == 200


def test_a_missing_field_is_422():
    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json={"model": "nanoserve"})

    assert run(scenario()).status_code == 422


# --- the reason any of this exists ----------------------------------------------


def test_concurrent_requests_are_batched_not_queued():
    """Four sockets, one loop, one forward per iteration covering all four."""
    expected = TOKENIZER.decode(_offline([1, 2, 3], 5))

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            responses = await asyncio.gather(
                *(
                    client.post("/v1/completions", json=_body(max_tokens=5))
                    for _ in range(4)
                )
            )
            return responses, serving.engine.iterations

    responses, iterations = run(scenario())
    assert all(r.status_code == 200 for r in responses)
    assert all(r.json()["choices"][0]["text"] == expected for r in responses)
    # Serialised, four identical requests would be 4 * (1 prefill + 4 decodes).
    assert iterations <= 8


def test_a_client_that_hangs_up_stops_costing_gpu():
    """The disconnect path. A cancelled handler must take its request with it, or
    the engine keeps a slot and its blocks busy for an answer nobody will read."""

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            task = asyncio.ensure_future(
                client.post("/v1/completions", json=_body(max_tokens=64))
            )
            for _ in range(5000):
                if serving.engine.iterations >= 3:
                    break
                await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(5000):
                if serving.parked:
                    break
                await asyncio.sleep(0.001)
            return serving.num_active, serving.engine.has_unfinished()

    active, unfinished = run(scenario())
    assert active == 0
    assert not unfinished


def test_the_lifespan_starts_and_stops_the_loop():
    """The uvicorn path: nobody calls `start` by hand in production, the app does."""
    serving = AsyncEngine(_engine())
    app = _app(serving)
    assert not serving.running
    with TestClient(app) as client:
        assert serving.running
        assert client.post("/v1/completions", json=_body()).status_code == 200
    assert not serving.running


# --- Day 40: the sampling parameters, honoured and policed ----------------------


def test_the_same_seed_gives_the_same_completion_twice():
    """What a seed is for, and the only reason a sampled endpoint is testable."""

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            body = _body(max_tokens=8, temperature=1.5, top_p=0.9, seed=1234)
            first = await client.post("/v1/completions", json=body)
            second = await client.post("/v1/completions", json=body)
            return first, second

    first, second = run(scenario())
    assert first.status_code == 200
    assert first.json()["choices"][0]["text"] == second.json()["choices"][0]["text"]


def test_different_seeds_give_different_completions():
    """The converse, stated over several seeds so one collision cannot pass it."""

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return [
                await client.post(
                    "/v1/completions",
                    json=_body(max_tokens=8, temperature=1.5, seed=seed),
                )
                for seed in range(6)
            ]

    texts = {r.json()["choices"][0]["text"] for r in run(scenario())}
    assert len(texts) > 1


def test_a_negative_temperature_is_a_400_that_names_it():
    """A refusal is only useful if the caller can tell which field broke."""

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(temperature=-1.0))

    response = run(scenario())
    assert response.status_code == 400
    assert "temperature" in response.json()["detail"]


def test_top_p_outside_its_range_is_a_400_that_names_it():
    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return [
                await client.post("/v1/completions", json=_body(top_p=bad))
                for bad in (0.0, 1.5)
            ]

    for response in run(scenario()):
        assert response.status_code == 400
        assert "top_p" in response.json()["detail"]


def test_a_negative_top_k_is_a_400_that_names_it():
    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(top_k=-3))

    response = run(scenario())
    assert response.status_code == 400
    assert "top_k" in response.json()["detail"]


def test_a_body_with_no_sampling_fields_is_still_greedy():
    """Forty days of clients that send nothing but `prompt` keep their answers."""
    expected = TOKENIZER.decode(_offline([1, 2, 3], 6))

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(max_tokens=6))

    assert run(scenario()).json()["choices"][0]["text"] == expected


def test_top_k_one_is_greedy_by_another_road():
    """A useful sanity check on the whole path: with one candidate the draw is forced.

    `top_k=1` masks everything but the argmax, so the multinomial has exactly one
    outcome whatever the temperature and whatever the seed. If this does not match
    the greedy answer, the filters are being applied to the wrong rows.
    """
    expected = TOKENIZER.decode(_offline([1, 2, 3], 6))

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post(
                "/v1/completions",
                json=_body(max_tokens=6, temperature=1.7, top_k=1),
            )

    assert run(scenario()).json()["choices"][0]["text"] == expected


# --- the disconnect a unary caller makes, which Day 41 found the server missing --


class _Receiver:
    """An ASGI `receive` a test can drive, because uvicorn's cannot be reached here.

    A real disconnect is a FIN that uvicorn turns into an `http.disconnect` message
    delivered on `receive`. `ASGITransport` never produces one, which is exactly how
    Day 37's cancel path passed its tests for four days while doing nothing on a
    real socket. So the message is injected instead: this blocks the way uvicorn's
    does, on a body that is already complete, until the test says the caller left.
    """

    def __init__(self, messages=()):
        self._queue = list(messages)
        self._event = asyncio.Event()

    def push(self, message: dict) -> None:
        self._queue.append(message)
        self._event.set()

    def disconnect(self) -> None:
        self.push({"type": "http.disconnect"})

    async def __call__(self) -> dict:
        while not self._queue:
            self._event.clear()
            await self._event.wait()
        return self._queue.pop(0)


def test_await_client_disconnect_returns_when_the_connection_closes():
    async def scenario():
        receiver = _Receiver()
        watcher = asyncio.create_task(await_client_disconnect(receiver))
        await asyncio.sleep(0)
        assert not watcher.done()
        receiver.disconnect()
        await watcher

    run(scenario())


def test_await_client_disconnect_keeps_waiting_through_other_messages():
    """A pipelined request on the same connection is not a disconnect.

    uvicorn's `receive` returns `http.request` for anything that arrives on the
    socket, so a watcher that resolves on the first message it sees would report
    every keep-alive client as gone and abort their request mid-generation.
    """

    async def scenario():
        receiver = _Receiver([{"type": "http.request", "body": b"", "more_body": False}])
        watcher = asyncio.create_task(await_client_disconnect(receiver))
        await asyncio.sleep(0)
        assert not watcher.done()
        receiver.disconnect()
        await watcher

    run(scenario())


def test_run_until_client_leaves_returns_the_answer_when_nobody_leaves():
    async def scenario():
        async def work():
            return "the answer"

        return await run_until_client_leaves(work(), _Receiver())

    assert run(scenario()) == "the answer"


def test_run_until_client_leaves_lets_the_works_own_failure_through():
    """A 400 must stay a 400. The race must not turn an admission error into a 499."""

    async def scenario():
        async def work():
            raise KVCacheExhausted("too big for the pool")

        with pytest.raises(KVCacheExhausted):
            await run_until_client_leaves(work(), _Receiver())

    run(scenario())


def test_run_until_client_leaves_cancels_the_work_when_the_client_goes():
    """The point of the whole thing: the generation is cancelled, not merely ignored.

    Ignoring it is what the server did before Day 41. The coroutine kept running, the
    request kept a slot and its blocks, and the engine spent a row of every forward
    on a caller who had closed the socket, right up to `max_tokens`.
    """

    async def scenario():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def work():
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        receiver = _Receiver()
        task = asyncio.create_task(run_until_client_leaves(work(), receiver))
        await started.wait()
        receiver.disconnect()
        with pytest.raises(ClientGone):
            await task
        assert cancelled.is_set()

    run(scenario())


def test_run_until_client_leaves_waits_for_the_cancellation_to_land():
    """The abort has to have happened before this returns, not eventually.

    `AsyncEngine.generate` frees the request inside its own `except CancelledError`,
    which only runs once the cancellation is *delivered*. Cancelling the task and
    returning would leave the handler finished and the request still running, which
    is the same leak with a shorter window.
    """

    async def scenario():
        freed = []

        async def work():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                freed.append("aborted")
                raise

        receiver = _Receiver()
        task = asyncio.create_task(run_until_client_leaves(work(), receiver))
        await asyncio.sleep(0.01)
        receiver.disconnect()
        with pytest.raises(ClientGone):
            await task
        assert freed == ["aborted"]

    run(scenario())


def _scope(body: bytes) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "raw_path": b"/v1/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"nanoserve"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 51234),
        "server": ("127.0.0.1", 8000),
    }


async def _call_asgi(app, body: dict, receiver: _Receiver) -> list[dict]:
    """Call the app as the ASGI callable it is, so the test owns `receive`.

    `ASGITransport` would be easier and is the thing that cannot express this test:
    it drives the app with a receive that only ever yields the body. Speaking ASGI
    directly is the smallest way to hand the handler a caller that goes away.
    """
    raw = json.dumps(body).encode()
    receiver.push({"type": "http.request", "body": raw, "more_body": False})
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    await app(_scope(raw), receiver, send)
    return sent


def test_a_unary_caller_that_disconnects_gets_its_request_aborted():
    """End of the story, at the ASGI seam: the request is gone before its budget is.

    `max_tokens=64` on an engine nobody else is using would run 64 iterations. The
    assertion is that it did not, which is the whole difference between a server
    that notices a dropped connection and one that finds out when the budget runs
    out.
    """

    async def scenario():
        engine = _engine()
        serving, app = await _serving(engine)
        receiver = _Receiver()
        async with serving:
            task = asyncio.create_task(
                _call_asgi(app, _body(max_tokens=64), receiver)
            )
            while engine.collected_tokens < 2:
                await asyncio.sleep(0.001)
            receiver.disconnect()
            sent = await task
            while serving.num_active or engine.scheduler.num_running:
                await asyncio.sleep(0.001)
        return engine, sent

    engine, sent = run(scenario())
    assert engine.collected_tokens < 64
    assert engine.scheduler.num_running == 0
    assert engine.allocator.num_free == engine.allocator.num_blocks
    assert sent[0]["status"] == 499


def test_a_unary_caller_that_stays_is_untouched_by_the_watcher():
    """The other half: a connection that never closes must still get its 200.

    A watcher that resolved early, or a race decided the wrong way under load, would
    turn every slow completion into a 499, and this is the assertion that would see
    it.
    """
    expected = TOKENIZER.decode(_offline([1, 2, 3], 6))

    async def scenario():
        serving, app = await _serving()
        async with serving:
            return await _call_asgi(app, _body(max_tokens=6), _Receiver())

    sent = run(scenario())
    assert sent[0]["status"] == 200
    payload = json.loads(b"".join(m.get("body", b"") for m in sent[1:]))
    assert payload["choices"][0]["text"] == expected
