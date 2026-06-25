"""Token sampling: greedy, temperature, top-k, top-p. Week 3, Day 10.

Greedy decode (Week 2) always takes the single largest logit, so a tiny float
difference is invisible: only the *order* of the top logit matters. Sampling is
the less forgiving case the Day-9 log promised. Here the actual probabilities
matter, because the next token is *drawn* from them, so each transform has to
match the reference exactly or the distribution drifts.

The whole module is pure: a logits vector goes in, a token id comes out, and the
three knobs are independent functions composed in one fixed order. That order is
the same one HuggingFace's `generate` uses, and it is not arbitrary:

    temperature  ->  top-k  ->  top-p  ->  softmax  ->  draw

  1. **temperature** reshapes the distribution before anything is thrown away.
     It is a plain divide of the logits: `< 1` sharpens toward the argmax, `> 1`
     flattens toward uniform. It comes first because top-k and top-p decide what
     to keep based on probabilities, and temperature is what sets those.
  2. **top-k** keeps the k most likely tokens and masks the rest to `-inf`. A
     hard cap on the candidate set, regardless of how the mass is spread.
  3. **top-p** (nucleus) keeps the smallest set of most likely tokens whose
     probabilities sum to at least `p`, masking the long improbable tail. Unlike
     top-k it adapts: a confident step keeps few tokens, an uncertain one keeps
     many. It runs after top-k so the nucleus is measured over what survived.

Masking means setting a logit to `-inf` so its softmax probability is exactly 0
and `torch.multinomial` can never draw it. The filters always keep at least one
token, so the surviving distribution is never empty. The masks are pinned to
transformers' own `TopKLogitsWarper` / `TopPLogitsWarper` in the tests, the same
way Day-4 RoPE was pinned to HF's `apply_rotary_pos_emb`: match the reference,
do not reinvent the convention.

The filters operate on the last dim, so they work on a single `[vocab]` vector
or a `[batch, vocab]` batch unchanged. `sample` itself takes one `[vocab]` vector
(one sequence's next-token logits) and returns a Python int; ragged batched
sampling is a scheduler concern for later phases.
"""

from __future__ import annotations

import torch

# The logit value that means "masked": softmax sends it to exactly 0 probability.
_FILTER = float("-inf")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by 1/temperature. `< 1` sharpens, `> 1` flattens.

    A plain divide, matching HF's `TemperatureLogitsWarper`. `temperature == 0`
    is the greedy limit and is handled in `sample` (dividing by zero is not), so
    callers of this function should pass a positive temperature.
    """
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits, mask the rest to -inf. `k <= 0` is a no-op.

    `k` is clamped to the vocab size, so an oversized k keeps everything. The
    threshold is the k-th largest logit; anything strictly below it is dropped,
    exactly as HF's `TopKLogitsWarper` does it.
    """
    if k <= 0:
        return logits
    k = min(k, logits.shape[-1])
    # The smallest of the top-k logits along the last dim, kept as a column so it
    # broadcasts back against every position.
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, _FILTER)


def top_p_filter(
    logits: torch.Tensor, p: float, min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """Keep the smallest nucleus of most likely tokens reaching mass `p`.

    `p >= 1.0` keeps everything. Otherwise this mirrors HF's `TopPLogitsWarper`:
    sort ascending, take the cumulative softmax mass, and remove every token whose
    cumulative mass (counted from the least likely upward) sits at or below
    `1 - p`. At least `min_tokens_to_keep` of the most likely tokens always
    survive, so the nucleus is never empty even when `p` is tiny.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Ascending order, so "remove the unlikely tail" is the low-cumulative end.
    sorted_remove = cumulative_probs <= (1 - p)
    # The most likely tokens are at the high end; never drop the last few.
    sorted_remove[..., -min_tokens_to_keep:] = False
    # Scatter the per-rank mask back to the original token positions.
    remove = sorted_remove.scatter(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, _FILTER)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Draw the next token id from a single `[vocab]` logits vector.

    The knobs compose in HF's order: temperature, then top-k, then top-p, then a
    softmax and one `torch.multinomial` draw. `temperature == 0` is the greedy
    limit and short-circuits to the argmax (the same token Week 2's `greedy_token`
    picks), so greedy is just the zero-temperature corner of this one function and
    never touches the RNG. The defaults (`temperature=1, top_k=0, top_p=1`) are
    plain softmax sampling over the full vocabulary.

    `generator` threads an explicit `torch.Generator` through the draw so a run is
    reproducible from a seed; omit it to use the global RNG.
    """
    if temperature == 0:
        return int(logits.argmax(dim=-1))
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = logits.softmax(dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))
