"""Day 10 tests: the sampling transforms, verified against the HF warpers.

Two tiers, the same shape as every other component's tests:
  - Pure tests run anywhere (torch only): each transform's math on small,
    hand-checkable logits. Temperature is a plain divide; top-k keeps exactly the
    k largest and masks the rest to -inf; top-p keeps the smallest nucleus whose
    probability mass reaches `p` and never masks everything; and the `sample`
    composition does greedy at temperature 0, collapses to argmax at top_k=1, and
    only ever returns an unmasked token.
  - `requires_transformers` tests pin our masks to transformers' own
    `TemperatureLogitsWarper`, `TopKLogitsWarper`, and `TopPLogitsWarper` on
    random logits, exactly as the RoPE test pins our convention to HF's
    `apply_rotary_pos_emb`. A flipped comparison or an off-by-one nucleus fails
    loudly here without needing the gated weights.
"""

from __future__ import annotations

import math

import pytest
import torch

from nanoserve.sampling import apply_temperature, sample, top_k_filter, top_p_filter

try:
    from transformers import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    HAS_TRANSFORMERS = True
except ImportError:  # transformers is a test-time reference, not a runtime dep
    HAS_TRANSFORMERS = False

requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed (reference-only dep)"
)


def _logits_from_probs(probs: list[float]) -> torch.Tensor:
    """Logits whose softmax is exactly `probs` (up to a constant): just log them.

    softmax(log p) == p, so this lets a test state the distribution it wants and
    reason about top-p in probability space instead of guessing logit values.
    """
    return torch.log(torch.tensor(probs))


# --- temperature: a plain divide --------------------------------------------


def test_temperature_divides_logits():
    x = torch.tensor([2.0, -4.0, 6.0])
    assert torch.equal(apply_temperature(x, 2.0), x / 2.0)


def test_temperature_below_one_sharpens_above_one_flattens():
    """Cooling concentrates mass on the argmax; heating spreads it out.

    Pure and deterministic (softmax, no draw): the peak probability rises as
    temperature falls and falls as temperature rises. This is the whole reason
    temperature is the first knob, checked without touching the RNG.
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    cold = apply_temperature(x, 0.5).softmax(-1).max()
    warm = apply_temperature(x, 2.0).softmax(-1).max()
    base = x.softmax(-1).max()
    assert cold > base > warm


# --- top-k: keep exactly the k largest --------------------------------------


def test_top_k_keeps_k_largest_and_masks_the_rest():
    x = torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0])
    out = top_k_filter(x, 2)
    finite = torch.isfinite(out)
    # The two largest (5 at idx 1, 4 at idx 3) survive; everything else is -inf.
    assert finite.tolist() == [False, True, False, True, False]
    assert torch.equal(out[finite], x[finite])
    assert torch.isneginf(out[~finite]).all()


def test_top_k_zero_is_identity():
    x = torch.tensor([1.0, 5.0, 2.0])
    assert torch.equal(top_k_filter(x, 0), x)


def test_top_k_larger_than_vocab_is_identity():
    x = torch.tensor([1.0, 5.0, 2.0])
    assert torch.equal(top_k_filter(x, 99), x)


# --- top-p: keep the smallest nucleus reaching probability p ----------------


def test_top_p_keeps_the_nucleus():
    """probs [0.5, 0.3, 0.15, 0.05], p=0.8 keeps the {0.5, 0.3} nucleus.

    HF's rule masks tokens whose cumulative mass (summing from the *least* likely)
    sits at or below 1 - p = 0.2. Ascending cumsum is [0.05, 0.20, 0.50, 1.00], so
    the 0.05 and 0.15 tokens go (0.20 <= 0.20 is removed at the boundary) and the
    0.5 and 0.3 tokens stay. Two tokens survive.
    """
    x = _logits_from_probs([0.5, 0.3, 0.15, 0.05])
    out = top_p_filter(x, 0.8)
    assert torch.isfinite(out).tolist() == [True, True, False, False]


def test_top_p_keeps_at_least_the_top_token():
    """Even a vanishingly small p keeps one token; the nucleus is never empty."""
    x = _logits_from_probs([0.7, 0.2, 0.1])
    out = top_p_filter(x, 1e-9)
    finite = torch.isfinite(out)
    assert finite.sum().item() == 1
    assert finite.tolist() == [True, False, False]  # the most likely token


def test_top_p_one_is_identity():
    x = torch.tensor([1.0, 5.0, 2.0])
    assert torch.equal(top_p_filter(x, 1.0), x)


# --- sample: the composition ------------------------------------------------


def test_sample_temperature_zero_is_greedy():
    x = torch.tensor([1.0, 9.0, 3.0, 2.0])
    assert sample(x, temperature=0.0) == 1
    # Deterministic: greedy never needs the RNG, so repeats agree.
    assert {sample(x, temperature=0.0) for _ in range(5)} == {1}


def test_sample_top_k_one_collapses_to_argmax():
    """With only the top logit unmasked, the draw is forced, seed or not."""
    x = torch.tensor([1.0, 9.0, 3.0, 2.0])
    g = torch.Generator().manual_seed(0)
    assert {sample(x, temperature=1.0, top_k=1, generator=g) for _ in range(10)} == {1}


def test_sample_never_returns_a_masked_token():
    """Over many seeded draws, top-k=2 sampling only ever emits a kept token."""
    x = torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0])
    allowed = {1, 3}  # the two largest logits
    g = torch.Generator().manual_seed(7)
    drawn = {sample(x, temperature=1.0, top_k=2, generator=g) for _ in range(200)}
    assert drawn <= allowed


def test_sample_is_deterministic_for_a_fixed_generator_state():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    a = sample(x, temperature=1.0, generator=torch.Generator().manual_seed(123))
    b = sample(x, temperature=1.0, generator=torch.Generator().manual_seed(123))
    assert a == b


def test_sample_draws_the_skewed_token_most_of_the_time():
    """A peaked distribution should yield its mode the large majority of draws.

    Statistical but seeded, so it is reproducible: with one token holding ~0.9 of
    the mass, a few hundred draws land on it the overwhelming majority of the time.
    Pins that `sample` actually samples from the softmax, not from something flat.
    """
    x = _logits_from_probs([0.9, 0.05, 0.03, 0.02])
    g = torch.Generator().manual_seed(0)
    draws = [sample(x, temperature=1.0, generator=g) for _ in range(300)]
    assert draws.count(0) > 250  # ~0.9 * 300, comfortably above with this seed


# --- pinned to the transformers warpers -------------------------------------


@requires_transformers
def test_temperature_matches_hf_warper():
    torch.manual_seed(0)
    scores = torch.randn(2, 50)
    mine = apply_temperature(scores, 0.7)
    ref = TemperatureLogitsWarper(0.7)(None, scores.clone())
    assert torch.equal(mine, ref)


@requires_transformers
def test_top_k_matches_hf_warper():
    torch.manual_seed(1)
    scores = torch.randn(2, 50)
    mine = top_k_filter(scores, 5)
    ref = TopKLogitsWarper(5)(None, scores.clone())
    assert torch.equal(mine, ref)


@requires_transformers
def test_top_p_matches_hf_warper():
    torch.manual_seed(2)
    scores = torch.randn(2, 50)
    mine = top_p_filter(scores, 0.9)
    ref = TopPLogitsWarper(0.9)(None, scores.clone())
    assert torch.equal(mine, ref)


@requires_transformers
def test_full_warp_pipeline_matches_hf_order():
    """temperature -> top-k -> top-p, the same order HF's generate applies them.

    Composing all three must equal stacking the three warpers in that order, so
    the masked sets and surviving logit values agree token for token. Guards
    against a subtle reordering (top-p before top-k changes the nucleus).
    """
    torch.manual_seed(3)
    scores = torch.randn(2, 50)
    mine = top_p_filter(top_k_filter(apply_temperature(scores, 0.8), 10), 0.95)
    ref = scores.clone()
    for warper in (TemperatureLogitsWarper(0.8), TopKLogitsWarper(10), TopPLogitsWarper(0.95)):
        ref = warper(None, ref)
    assert torch.equal(mine, ref)


def test_log_probs_helper_is_exact():
    """Sanity on the test's own helper: softmax(log p) recovers p."""
    probs = [0.5, 0.3, 0.2]
    got = _logits_from_probs(probs).softmax(-1)
    assert torch.allclose(got, torch.tensor(probs), atol=1e-6)
    assert math.isclose(got.sum().item(), 1.0, rel_tol=1e-6)
