"""Day 12 tests: sampling wired into the cached decode loop.

`generate` is `generate_stream` drained into a tensor: the Day-11 KV cache for
speed, Day-10's `sample` for the draw. The contract is that the two compose
without either one changing the other, so the tests pin the corners where that
could break:

  - greedy is still greedy: `temperature == 0` (and the degenerate `top_k == 1`)
    must reproduce the verified `greedy_generate_cached` tokens exactly. Greedy is
    a corner of the sampling path, not a separate path that can drift.
  - sampling is reproducible: a seed makes a run deterministic, and the stream and
    the collected tensor agree token for token under that seed.
  - sampling actually samples: with temperature up, different seeds explore
    different continuations, so the loop is not silently always-argmax.
"""

from __future__ import annotations

import torch

from nanoserve.cache import BlockAllocator
from nanoserve.config import ModelConfig
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel


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


def _model() -> tuple[LlamaModel, ModelConfig]:
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg)), cfg


# --- greedy is a corner of the sampling path --------------------------------


def test_temperature_zero_matches_greedy_cached():
    """`temperature == 0` short-circuits to the argmax, so it must equal greedy.

    This is the load-bearing equivalence: greedy decode is not reimplemented, it
    is the zero-temperature corner of `sample`, run through the same cached loop
    as everything else. If this drifts, the two decode modes have diverged.
    """
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    greedy = model.greedy_generate_cached(ids, max_new_tokens=8)
    sampled = model.generate(ids, max_new_tokens=8, temperature=0.0)
    assert torch.equal(greedy, sampled)


def test_top_k_one_is_greedy_regardless_of_temperature():
    """top_k=1 keeps only the largest logit, so the draw is forced to the argmax.

    A different route to greedy than temperature=0: the filter leaves a one-hot
    distribution, so `multinomial` can only pick the mode, whatever the
    temperature or seed. Pins that the top-k filter is wired in correctly.
    """
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    greedy = model.greedy_generate_cached(ids, max_new_tokens=6)
    forced = model.generate(ids, max_new_tokens=6, temperature=1.7, top_k=1, seed=0)
    assert torch.equal(greedy, forced)


# --- sampling is reproducible -----------------------------------------------


def test_same_seed_is_deterministic():
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    a = model.generate(ids, max_new_tokens=10, temperature=1.0, seed=1234)
    b = model.generate(ids, max_new_tokens=10, temperature=1.0, seed=1234)
    assert torch.equal(a, b)


def test_stream_matches_collected_under_same_seed():
    """generate_stream and generate must yield the identical tokens for one seed.

    generate is documented as just draining the stream, so the streamed token ids
    appended to the prompt have to equal the tensor generate returns. Same seed,
    same draws, same order.
    """
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    streamed = list(
        model.generate_stream(ids, max_new_tokens=7, temperature=1.0, top_p=0.9, seed=7)
    )
    collected = model.generate(ids, max_new_tokens=7, temperature=1.0, top_p=0.9, seed=7)
    assert collected.shape == (1, 4 + 7)
    assert streamed == collected[0, 4:].tolist()


# --- sampling actually explores ---------------------------------------------


def test_different_seeds_explore_different_tokens():
    """With temperature up, varying the seed must change the continuation.

    The guard against a silent bug where the loop ignores the RNG and always takes
    the argmax (which would still pass the determinism test). Over several seeds on
    a deliberately flattened distribution we expect more than one distinct output.
    """
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    outs = {
        tuple(model.generate(ids, max_new_tokens=5, temperature=2.0, seed=s)[0, 4:].tolist())
        for s in range(8)
    }
    assert len(outs) > 1


# --- the same contracts as the greedy loop ----------------------------------


def test_generate_appends_and_stops_at_eos():
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))

    out = model.generate(ids, max_new_tokens=7, temperature=0.0)
    assert out.shape == (1, 4 + 7)
    assert torch.equal(out[:, :4], ids)

    first = model.generate(ids, max_new_tokens=1, temperature=0.0)[0, -1].item()
    stopped = model.generate(ids, max_new_tokens=10, temperature=0.0, eos_id=first)
    assert stopped.shape == (1, 5)
    assert stopped[0, -1].item() == first


def test_generate_rejects_a_real_batch():
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    try:
        model.generate(ids, max_new_tokens=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for batch > 1")


# --- Day 17: paging under the sampling path ---------------------------------
#
# Day 12 ran the sampling loop over the naive contiguous cache. Day 16 built the
# paged cache and proved it byte-for-byte identical to naive on the greedy path.
# Day 17 closes Week 5: the sampling path runs over the paged pool too, and a
# finished sequence frees its blocks back so the next one reuses them. Paging is
# a memory-layout change, not a behavior change, so nothing observable moves:
# the same seed draws the same tokens, whichever cache is underneath.


def test_paged_sampling_matches_naive_sampling_under_same_seed():
    """The headline contract, one level up from Day 16: paged == naive to sample.

    Same prompt, same seed, same temperature/top-p, one over the naive contiguous
    cache and one over the paged pool. The tokens must be `torch.equal`; the draw
    only sees the logits, and the logits only see the assembled K/V, which paging
    reproduces exactly. Any divergence is a paging bug leaking into the output.
    """
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    naive = model.generate(ids, max_new_tokens=8, temperature=1.0, top_p=0.9, seed=7)
    paged = model.generate(
        ids, max_new_tokens=8, temperature=1.0, top_p=0.9, seed=7, paged=True, block_size=4
    )
    assert torch.equal(naive, paged)


def test_paged_sampling_temperature_zero_matches_greedy_paged():
    """Greedy stays the zero-temperature corner of the paged sampling path too."""
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    greedy = model.greedy_generate_paged(ids, max_new_tokens=8, block_size=4)
    sampled = model.generate(
        ids, max_new_tokens=8, temperature=0.0, paged=True, block_size=4
    )
    assert torch.equal(greedy, sampled)


def test_paged_stream_matches_collected_under_same_seed():
    """generate_stream over the paged pool drains into the same tensor generate does."""
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    streamed = list(
        model.generate_stream(
            ids, max_new_tokens=7, temperature=1.0, top_p=0.9, seed=7, paged=True, block_size=4
        )
    )
    collected = model.generate(
        ids, max_new_tokens=7, temperature=1.0, top_p=0.9, seed=7, paged=True, block_size=4
    )
    assert streamed == collected[0, 4:].tolist()


def test_paged_generation_frees_its_blocks_on_finish():
    """A finished paged generation returns every physical block to a shared pool.

    Hand in an allocator (which selects the paged path and shares the pool) and
    the generation must leave it exactly as full as it found it: every block the
    run pulled is freed when the last token is drawn. That free-on-finish is what
    makes the pool reusable across sequences instead of leaking a block per run.
    """
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    total = 4 + 6
    alloc = BlockAllocator(num_blocks=(total + 3) // 4, block_size=4)
    full = alloc.num_free
    model.generate(ids, max_new_tokens=6, temperature=0.0, allocator=alloc)
    assert alloc.num_free == full


def test_paged_sampling_reuses_a_finished_sequences_blocks():
    """The free-and-reuse stress run: one pool, sized for one sequence, serves two.

    The allocator holds exactly enough blocks for a single prompt+generation. The
    first run fills it and frees on finish; the second only succeeds because those
    blocks came back to the pool, and it draws the same tokens a fresh pool would.
    If free-on-finish were broken, the second run would raise KVCacheExhausted.
    """
    model, cfg = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    total = 4 + 6
    shared = BlockAllocator(num_blocks=(total + 3) // 4, block_size=4)

    a_ref = model.generate(ids, max_new_tokens=6, temperature=1.0, seed=1, paged=True, block_size=4)
    b_ref = model.generate(ids, max_new_tokens=6, temperature=1.0, seed=2, paged=True, block_size=4)

    a = model.generate(ids, max_new_tokens=6, temperature=1.0, seed=1, allocator=shared)
    assert shared.num_free == shared.num_blocks  # the first run freed everything
    b = model.generate(ids, max_new_tokens=6, temperature=1.0, seed=2, allocator=shared)
    assert shared.num_free == shared.num_blocks

    assert torch.equal(a, a_ref)
    assert torch.equal(b, b_ref)
