"""Day 37 tests: the OpenAI-shaped surface over the bridge.

The engine speaks token ids and the wire speaks JSON, so this layer is a
translation and a set of refusals. The translation is small enough to read in one
sitting: a prompt in, `max_tokens`, one choice out, a usage block. The refusals
are the part worth testing, because every one of them is a decision about what a
server owes a caller it cannot serve.

The rule this file enforces everywhere: **a parameter that would change the answer
is either honoured or refused, never accepted and ignored.** The engine samples
with `argmax` and nothing else, so `temperature=0.7` cannot be served; accepting
it would hand back greedy tokens under a name that promises otherwise, which is a
wrong answer with a 200 on it. Week 11 adds sampling and streaming and turns two
of these 400s into features. Until then they are 400s, and each one says so.

Nothing here needs `./weights`: the tiny random model from the engine tests, plus
a 64-symbol byte tokenizer, which is enough to prove that what comes out of the
socket is exactly what the offline engine emits for the same prompt.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.server import create_app
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


def test_streaming_is_refused_rather_than_silently_batched():
    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            return await client.post("/v1/completions", json=_body(stream=True))

    response = run(scenario())
    assert response.status_code == 400
    assert "stream" in response.json()["detail"]


def test_a_sampling_temperature_is_refused_rather_than_ignored():
    """The engine is greedy. Serving `temperature=0.7` as argmax is a wrong answer
    with a 200 on it, which is worse than an error."""

    async def scenario():
        serving, app = await _serving()
        async with serving, _client(app) as client:
            hot = await client.post("/v1/completions", json=_body(temperature=0.7))
            cold = await client.post("/v1/completions", json=_body(temperature=0.0))
            return hot, cold

    hot, cold = run(scenario())
    assert hot.status_code == 400
    assert "temperature" in hot.json()["detail"]
    assert cold.status_code == 200


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
