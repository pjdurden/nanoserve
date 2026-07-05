"""Paged attention over scattered KV blocks. Week 6 (the headline artifact).

Weeks 4-5 built a paged KV cache: real K/V living in a fixed pool of physical
blocks, addressed through a per-sequence block table. But the *read* was
deliberately slow. Every step `PagedKVCache.append` gathers the whole history
back into one contiguous `[1, n_kv, seq, d]` tensor and hands it to a normal
attention, so the output is provably identical to the naive contiguous cache
before any real kernel exists. That gather throws away the entire point of
paging on the read side: it re-materializes the very contiguous buffer paging
was meant to avoid.

Week 6 replaces that gather with attention that reads K/V *directly* through the
block table and never assembles the contiguous history. `paged_attention_reference`
below is the first step: a plain-torch version of exactly that computation, the
correctness oracle the hand-written Triton kernel gets held to. The kernel and the
reference take the same inputs (query, the layer's flat K/V pools, the per-position
slot mapping) and must return the same output to about 1e-3, so the day the kernel
lands there is already a test that says whether it is right.

Reference, not the fast path: it still gathers K/V with a torch index rather than
streaming blocks the way the kernel will. What it proves is the *interface and the
math*, that attention can be expressed as "read each past token through its slot,
then score", so the kernel has a fixed target to match instead of a moving one.
"""

from __future__ import annotations

import torch

from ..layers import repeat_kv


def paged_attention_reference(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention for one sequence's new queries, reading K/V through the block table.

    q:            [1, n_q, seq_q, d] rotated queries for the tokens generated this
                  step (`seq_q` is the prompt length on prefill, 1 on a decode step).
    k_pool,
    v_pool:       [num_slots, n_kv, d] the layer's flat physical pool, exactly what
                  `PagedKVCache.k_pool[layer]` holds. A "slot" is the flat index
                  `block_id * block_size + offset`; the sequence's K/V may sit at
                  any scattered slots.
    slot_mapping: [seq_total] long tensor, `slot_mapping[p]` is the flat pool slot of
                  logical position p, for p in 0..seq_total-1 (oldest token first).
                  This is `[table.slot(p) for p in range(seq_total)]`. `seq_total`
                  is the whole history including the `seq_q` new tokens.
    n_rep:        GQA repeat factor (`config.num_kv_groups`): query heads per KV head.
    scale:        softmax scale; defaults to `head_dim ** -0.5`, matching
                  `gqa_attention`.

    Returns [1, n_q, seq_q, d], the attention output before o_proj, byte-identical
    to a contiguous SDPA over the same K/V. The read is paged (each token fetched
    through its slot); the math is the ordinary causal, GQA-repeated softmax.

    One sequence only (batch 1), the same scope as `PagedKVCache`; per-sequence
    batching arrives in Phase 3.
    """
    if q.shape[0] != 1:
        raise ValueError(
            "paged_attention_reference handles one sequence; batch must be 1 "
            "(per-sequence batching arrives in Phase 3)"
        )
    seq_q = q.shape[2]
    d = q.shape[3]
    if scale is None:
        scale = d**-0.5

    # Paged read: fetch each historical token's compact K/V through its slot. The
    # gather turns the scattered pool into the ordered history [seq_total, n_kv, d]
    # without the sequence ever owning a contiguous buffer; the Triton kernel will
    # do this fetch block by block instead of as one index.
    k_hist = k_pool[slot_mapping]  # [seq_total, n_kv, d]
    v_hist = v_pool[slot_mapping]
    # -> [1, n_kv, seq_total, d], the shape attention scores against.
    k = k_hist.transpose(0, 1)[None]
    v = v_hist.transpose(0, 1)[None]

    # Grow the compact KV heads to the full query-head count (a view, not a copy).
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    # Scaled scores, then the rectangular causal mask: query i (absolute position
    # `past + i`, where `past = seq_total - seq_q`) may see keys 0..past+i and no
    # further. On a single decode step past+i is the last position, so the one
    # query sees the whole history. Identical band to `gqa_attention`.
    kv_len = k.shape[2]
    past = kv_len - seq_q
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    causal = torch.full((seq_q, kv_len), float("-inf"), dtype=scores.dtype, device=scores.device)
    scores = scores + torch.triu(causal, diagonal=past + 1)

    # Softmax in fp32 then cast back, mirroring HF (and `gqa_attention`) so the
    # bf16 path will match later; in the fp32 pipeline the cast is a no-op.
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(weights, v)


# TODO(week6): triton.jit kernel for paged attention (held to the reference above)
# Day 20 microbenchmarked the gather vs the fused read (readbench.py): both are
# torch and O(history), so they tie; the CPU cost curve there is the target the
# kernel has to beat, benchmarked with the same harness once it lands.
