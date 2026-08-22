"""Day 40 tests: one logits tensor, one row per request, N different samplers.

Day 10's `sample` is a function over a `[vocab]` vector, and every test it has
asks the same question: does this one draw match the reference. That was enough
for an offline generator with one knob setting per process. A server does not get
that. Under continuous batching the rows of a single `[rows, vocab]` tensor belong
to different callers, one of whom asked for greedy, one for `top_p=0.9`, one for
`temperature=1.4, top_k=40`, and all three are columns of the same forward pass.

So the batch is heterogeneous in its *parameters* and the tensor is not. That is
the whole day, and it has three failure modes worth a file of tests:

  1. **A row's token depends on who it shared a step with.** One global RNG drawn
     once per batch means the token a caller gets is a function of the batch
     composition, which is a function of whoever else happened to post at the same
     moment. Seed a request and it is still not reproducible, which is the one
     thing a seed is for. The fix is a generator per request, and the tests here
     pin the property directly: the same seeded row draws the same tokens alone
     and in a crowd.
  2. **Greedy is not temperature zero.** Not in floating point: `logits / 0` is
     `inf`, its softmax is `nan`, and `multinomial` on that raises or draws
     nonsense. Greedy short-circuits to `argmax` and, just as importantly, never
     touches the RNG, so adding a greedy row to a batch must not move any other
     row's draw.
  3. **A generator that is rebuilt per step is not a generator.** Seed at request
     start, keep the state across every token that request draws, and drop it when
     the request finishes: the state has to advance, and the dict it lives in has
     to shrink, or a long-lived server accumulates one `torch.Generator` per
     request it has ever served.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.sampling import GREEDY, BatchedSampler, SamplingParams

VOCAB = 8


def _peaked(argmax: int, vocab: int = VOCAB) -> torch.Tensor:
    """A `[vocab]` row whose largest logit is at `argmax`, by a wide margin."""
    row = torch.zeros(vocab)
    row[argmax] = 10.0
    return row


def _flat(vocab: int = VOCAB) -> torch.Tensor:
    """Uniform logits: every token equally likely, so a draw is visible as a draw."""
    return torch.zeros(vocab)


def _random(**kw) -> SamplingParams:
    """Sampling params that actually sample. Temperature 1 unless overridden."""
    return SamplingParams(temperature=kw.pop("temperature", 1.0), **kw)


# --- the parameters -------------------------------------------------------------


def test_the_default_is_greedy():
    """Every `Request` built before today gets these, so they must mean argmax."""
    assert SamplingParams().is_greedy
    assert SamplingParams().temperature == 0.0
    assert GREEDY.is_greedy


def test_any_positive_temperature_is_not_greedy():
    assert not SamplingParams(temperature=0.01).is_greedy


def test_a_negative_temperature_is_refused():
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-0.5)


def test_top_p_outside_its_range_is_refused():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(temperature=1.0, top_p=bad)


def test_a_negative_top_k_is_refused():
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(temperature=1.0, top_k=-2)


def test_the_params_are_frozen():
    """A request's sampling cannot change halfway through its own generation."""
    params = SamplingParams(temperature=1.0)
    with pytest.raises(Exception):
        params.temperature = 0.5


# --- one tensor, many samplers --------------------------------------------------


def test_every_row_is_sampled_from_its_own_logits():
    """The obvious property, and the one an index slip breaks silently."""
    logits = torch.stack([_peaked(3), _peaked(6), _peaked(1)])
    sampler = BatchedSampler(seed=0)
    rows = [("a", GREEDY), ("b", GREEDY), ("c", GREEDY)]
    assert sampler.sample_batch(logits, rows) == [3, 6, 1]


def test_a_greedy_row_is_argmax_even_in_a_sampled_batch():
    logits = torch.stack([_flat(), _peaked(5), _flat()])
    sampler = BatchedSampler(seed=0)
    rows = [("a", _random()), ("b", GREEDY), ("c", _random(top_p=0.9))]
    assert sampler.sample_batch(logits, rows)[1] == 5


def test_a_greedy_row_never_touches_the_rng():
    """Adding a greedy neighbour must not move anybody else's draw.

    Greedy is `argmax`, which needs no random numbers, so a batch of
    [greedy, sampled] and a batch of [sampled] have to hand the sampled row the
    same token from the same generator state. If greedy went through the softmax
    path it would consume a draw and every other row would shift.
    """
    row = _flat()
    with_greedy = BatchedSampler(seed=11).sample_batch(
        torch.stack([row, row]), [("g", GREEDY), ("s", _random())]
    )
    alone = BatchedSampler(seed=11).sample_batch(
        torch.stack([row]), [("s", _random())]
    )
    assert with_greedy[1] == alone[0]


def test_temperature_zero_never_reaches_a_divide():
    """The literal reason greedy is a branch and not a limit.

    `logits / 0` is `inf`, `softmax` of that is `nan`, and a draw over `nan` is
    either an exception or garbage. A finite token id out of a greedy row is the
    assertion that this path was never taken.
    """
    logits = torch.stack([_peaked(2)])
    assert BatchedSampler(seed=0).sample_batch(logits, [("a", GREEDY)]) == [2]


def test_rows_are_filtered_by_their_own_knobs():
    """`top_k=1` forces its row to the argmax; its neighbour stays free.

    The filters are per row, so two rows of identical logits and different `top_k`
    must not share a threshold. Over many steps the forced row is constant and the
    free row is not.
    """
    logits = torch.stack([_flat(), _flat()])
    logits[0][4] = 1.0
    logits[1][4] = 1.0
    sampler = BatchedSampler(seed=3)
    rows = [("forced", _random(top_k=1)), ("free", _random())]
    draws = [sampler.sample_batch(logits, rows) for _ in range(30)]
    assert {d[0] for d in draws} == {4}
    assert len({d[1] for d in draws}) > 1


def test_a_masked_token_is_never_drawn_in_a_batch():
    logits = torch.stack([torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0, 0.0, 0.0, 0.0])] * 2)
    sampler = BatchedSampler(seed=7)
    rows = [("a", _random(top_k=2)), ("b", _random(top_k=2))]
    drawn = set()
    for _ in range(50):
        drawn.update(sampler.sample_batch(logits, rows))
    assert drawn <= {1, 3}


def test_the_row_count_must_match_the_logits():
    """A misaligned batch is a caller handing one request another's distribution."""
    with pytest.raises(ValueError, match="rows"):
        BatchedSampler().sample_batch(torch.stack([_flat(), _flat()]), [("a", GREEDY)])


# --- the seed, which is the whole point -----------------------------------------


def test_a_seeded_row_draws_the_same_tokens_alone_and_in_a_crowd():
    """The property a seed promises, and the one a shared generator cannot keep.

    The same logits, the same seed, and a batch that grows from one row to four
    with the seeded row moved to the end. Every token it draws is identical, which
    is only true if its randomness is its own.
    """
    row = _flat()
    seeded = SamplingParams(temperature=1.0, seed=1234)

    alone = BatchedSampler(seed=0)
    solo = [alone.sample_batch(torch.stack([row]), [("x", seeded)])[0] for _ in range(12)]

    crowded = BatchedSampler(seed=999)
    rows = [("n1", _random()), ("n2", _random(top_p=0.5)), ("x", seeded)]
    shared = [
        crowded.sample_batch(torch.stack([row, row, row]), rows)[2] for _ in range(12)
    ]

    assert solo == shared
    assert len(set(solo)) > 1  # it really is sampling, not stuck on one token


def test_an_unseeded_row_is_coupled_to_its_batchmates():
    """The honest converse, and why `seed` is not decoration.

    Without a seed a request draws from the engine's shared generator, so its
    tokens depend on how many other sampled rows drew before it in the same step.
    Same engine seed, same logits, different neighbours, different answer.
    """
    row = _flat()
    a = BatchedSampler(seed=5)
    solo = [a.sample_batch(torch.stack([row]), [("x", _random())])[0] for _ in range(20)]
    b = BatchedSampler(seed=5)
    rows = [("n", _random()), ("x", _random())]
    shared = [
        b.sample_batch(torch.stack([row, row]), rows)[1] for _ in range(20)
    ]
    assert solo != shared


def test_two_requests_with_the_same_seed_draw_the_same_tokens():
    """Reproducibility is a property of the seed, not of the request id."""
    row = _flat()
    sampler = BatchedSampler(seed=0)
    rows = [("a", SamplingParams(temperature=1.0, seed=42)),
            ("b", SamplingParams(temperature=1.0, seed=42))]
    for _ in range(10):
        drawn = sampler.sample_batch(torch.stack([row, row]), rows)
        assert drawn[0] == drawn[1]


def test_a_generator_advances_across_steps_rather_than_restarting():
    """Seed once at the request, not once per token.

    Re-seeding every step is the subtle version of this bug: every draw comes from
    the same state, so a flat distribution emits the same token forever and the
    output looks like the model is stuck. The state has to carry.
    """
    row = _flat()
    sampler = BatchedSampler(seed=0)
    rows = [("x", SamplingParams(temperature=1.0, seed=8))]
    draws = [sampler.sample_batch(torch.stack([row]), rows)[0] for _ in range(20)]
    assert len(set(draws)) > 1


def test_a_seeded_generator_is_kept_and_released():
    """One generator per live sampled request, and none per finished one."""
    row = _flat()
    sampler = BatchedSampler(seed=0)
    sampler.sample_batch(torch.stack([row]), [("x", SamplingParams(temperature=1.0, seed=8))])
    assert sampler.num_generators == 1
    sampler.release("x")
    assert sampler.num_generators == 0


def test_an_unseeded_request_stores_no_generator():
    """Sharing the engine's generator costs nothing to keep and nothing to drop."""
    row = _flat()
    sampler = BatchedSampler(seed=0)
    sampler.sample_batch(torch.stack([row]), [("x", _random())])
    assert sampler.num_generators == 0


def test_releasing_an_unknown_request_is_not_an_error():
    """The engine releases on every finish, including requests that never sampled."""
    BatchedSampler().release("never-seen")


def test_a_greedy_request_stores_no_generator():
    row = _peaked(1)
    sampler = BatchedSampler(seed=0)
    sampler.sample_batch(torch.stack([row]), [("x", SamplingParams(seed=8))])
    assert sampler.num_generators == 0
