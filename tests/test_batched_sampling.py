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


# --- Day 47: the same draw, without coming home ---------------------------------

# Day 46 profiled the loop and found `sample` holding 86% to 89% of it at every
# batch size. Not because sampling is expensive: because `sample_batch` returns
# `list[int]`, and every one of those ints is a separate journey back from the
# device. `sample_batch_device` is the same arithmetic with the return type
# changed to a `[rows]` tensor, so the tokens stay where they were computed and
# the host reads them at most once, later, and on purpose.
#
# Two properties have to survive that change or it is not an optimisation, it is
# a rewrite of the sampler: every token must be the one the old path drew, and
# the seeded rows must still be independent of their batchmates.


def _boom(*args, **kwargs):
    raise AssertionError("the device sampler read a tensor back to the host")


def _forbid_readback(monkeypatch):
    """Make every host-side read of a tensor an error, for the duration of a test.

    This is the only direct way to assert the absence of a synchronisation. The
    cost of a readback does not show up in a return value, it shows up as the host
    waiting, and on a CPU box it does not show up at all. Removing the operations
    themselves turns "did not sync" into something a test can fail on.
    """
    for name in ("tolist", "item", "__int__", "__index__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, _boom, raising=False)


def test_the_device_sampler_returns_a_tensor_not_a_list():
    logits = torch.stack([_peaked(3), _peaked(6)])
    tokens = BatchedSampler(seed=0).sample_batch_device(logits, [("a", GREEDY), ("b", GREEDY)])
    assert isinstance(tokens, torch.Tensor)


def test_the_device_sampler_returns_one_integer_per_row():
    logits = torch.stack([_peaked(3), _peaked(6), _peaked(1)])
    rows = [("a", GREEDY), ("b", GREEDY), ("c", GREEDY)]
    tokens = BatchedSampler(seed=0).sample_batch_device(logits, rows)
    assert tokens.shape == (3,)
    assert tokens.dtype == torch.long


def test_the_tokens_come_back_on_the_logits_device():
    """The whole point: the tensor never leaves the device the forward ran on."""
    logits = torch.stack([_peaked(3), _peaked(6)])
    tokens = BatchedSampler(seed=0).sample_batch_device(logits, [("a", GREEDY), ("b", GREEDY)])
    assert tokens.device == logits.device


def test_an_all_greedy_batch_is_one_argmax_over_the_whole_tensor():
    logits = torch.stack([_peaked(3), _peaked(6), _peaked(1), _peaked(0)])
    rows = [(str(i), GREEDY) for i in range(4)]
    tokens = BatchedSampler(seed=0).sample_batch_device(logits, rows)
    assert torch.equal(tokens, logits.argmax(dim=-1))


def test_an_all_greedy_batch_never_reads_a_tensor_back(monkeypatch):
    """The common case, and the one the engine runs by default.

    `SamplingParams()` is greedy, so every request in Weeks 7 to 12 took this
    path. Before today it gathered the greedy rows into a copy, ran an argmax and
    called `.tolist()`; now it is one kernel and no journey home.
    """
    logits = torch.stack([_peaked(3), _peaked(6)])
    rows = [("a", GREEDY), ("b", GREEDY)]
    sampler = BatchedSampler(seed=0)
    _forbid_readback(monkeypatch)
    tokens = sampler.sample_batch_device(logits, rows)
    assert tokens.shape == (2,)


def test_a_mixed_batch_never_reads_a_tensor_back(monkeypatch):
    """The harder case: greedy, unseeded and seeded rows in the same tensor.

    The draw is still one `multinomial` per seeded row, because a seed belongs to
    a request. What changed is that its result is written into the token tensor
    with `index_copy_` instead of being turned into a Python int on the spot.
    """
    row = _flat()
    logits = torch.stack([row, row, row, _peaked(2)])
    rows = [
        ("g", GREEDY),
        ("u", _random()),
        ("s", SamplingParams(temperature=1.0, seed=5)),
        ("g2", GREEDY),
    ]
    sampler = BatchedSampler(seed=0)
    _forbid_readback(monkeypatch)
    tokens = sampler.sample_batch_device(logits, rows)
    assert tokens.shape == (4,)


def test_the_device_path_draws_exactly_what_the_list_path_drew():
    """Two identically seeded samplers, one down each path, token for token.

    The regression that matters. A sampler rewritten for speed that quietly
    changes which token a request gets is a behaviour change wearing an
    optimisation's clothes, and nothing downstream would catch it.
    """
    row = _flat()
    logits = torch.stack([row, _peaked(4), row, row])
    rows = [
        ("a", _random()),
        ("b", GREEDY),
        ("c", SamplingParams(temperature=1.0, seed=77)),
        ("d", _random(top_k=3)),
    ]
    listed = BatchedSampler(seed=12)
    devised = BatchedSampler(seed=12)
    for _ in range(8):
        assert listed.sample_batch(logits, rows) == devised.sample_batch_device(logits, rows).tolist()


def test_the_list_path_is_the_device_path_read_once():
    """`sample_batch` is not a second implementation, it is one `.tolist()`."""
    row = _flat()
    logits = torch.stack([row, row])
    rows = [("a", _random()), ("b", SamplingParams(temperature=1.0, seed=3))]
    a = BatchedSampler(seed=4)
    b = BatchedSampler(seed=4)
    assert a.sample_batch(logits, rows) == b.sample_batch_device(logits, rows).tolist()


def test_a_seeded_row_on_the_device_path_is_still_alone_with_its_generator():
    """Day 40's property, re-asserted through the new return type.

    Grouping and `index_copy_` are exactly the places a rewrite couples rows back
    together, so the promise is checked again rather than assumed to have carried.
    """
    row = _flat()
    seeded = SamplingParams(temperature=1.0, seed=1234)
    alone = BatchedSampler(seed=0)
    solo = [
        int(alone.sample_batch_device(torch.stack([row]), [("x", seeded)])[0])
        for _ in range(12)
    ]
    crowded = BatchedSampler(seed=999)
    rows = [("n1", _random()), ("n2", _random(top_p=0.5)), ("x", seeded)]
    shared = [
        int(crowded.sample_batch_device(torch.stack([row, row, row]), rows)[2])
        for _ in range(12)
    ]
    assert solo == shared
    assert len(set(solo)) > 1


def test_a_greedy_row_in_a_mixed_batch_is_still_the_argmax_of_its_own_row():
    logits = torch.stack([_flat(), _peaked(5), _flat()])
    rows = [("a", _random()), ("b", GREEDY), ("c", _random(top_p=0.9))]
    tokens = BatchedSampler(seed=0).sample_batch_device(logits, rows)
    assert int(tokens[1]) == 5


def test_the_device_path_refuses_a_row_count_that_does_not_match():
    with pytest.raises(ValueError, match="rows"):
        BatchedSampler().sample_batch_device(torch.stack([_flat(), _flat()]), [("a", GREEDY)])


def test_an_empty_batch_is_an_empty_tensor_and_not_an_error():
    """The scheduler can hand down zero rows on a step that only prefilled."""
    tokens = BatchedSampler().sample_batch_device(torch.zeros(0, VOCAB), [])
    assert tokens.shape == (0,)
    assert tokens.dtype == torch.long


def test_a_masked_token_is_never_drawn_on_the_device_path():
    logits = torch.stack([torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0, 0.0, 0.0, 0.0])] * 2)
    sampler = BatchedSampler(seed=7)
    rows = [("a", _random(top_k=2)), ("b", _random(top_k=2))]
    drawn = set()
    for _ in range(50):
        drawn.update(sampler.sample_batch_device(logits, rows).tolist())
    assert drawn <= {1, 3}


def test_the_device_path_still_releases_generators():
    """The leak Day 40 closed is not reopened by the new entry point."""
    row = _flat()
    sampler = BatchedSampler(seed=0)
    sampler.sample_batch_device(torch.stack([row]), [("x", SamplingParams(temperature=1.0, seed=8))])
    assert sampler.num_generators == 1
    sampler.release("x")
    assert sampler.num_generators == 0
