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
from .tlsim import arange, cdiv, launch, load, store


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


def paged_attention_batched_reference(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
) -> torch.Tensor:
    """The decode read for a whole batch: one new token per row, one table per row.

    Day 28. The single-sequence read above walks one slot mapping over one history.
    A batch of N sequences decoding together has N histories of different lengths,
    living in one shared pool, so the mapping grows a row axis and the "how far does
    this row's history go" scalar becomes a vector.

    q:            [batch, n_q, 1, d] this step's rotated queries, one new token per
                  row. Exactly one: a decode step is one token per sequence, and a
                  ragged *prefill* is the dense masked path (`gqa_attention` with an
                  attention mask), not this.
    k_pool,
    v_pool:       [num_slots, n_kv, d] the layer's flat physical pool, shared by
                  every row. Rows interleave in it freely; the mapping is what keeps
                  them apart.
    slot_mapping: [batch, max_ctx] long. `slot_mapping[i, p]` is the flat pool slot
                  of row i's logical position p, oldest first. Rows shorter than
                  `max_ctx` are padded with any *legal* index (the cache uses 0):
                  this reference gathers the whole rectangle before it masks, so a
                  padded entry really is read, and a -1 would silently wrap onto the
                  last slot of the pool instead of erroring.
    context_lens: [batch] long, how many of a row's slots are real. This is what the
                  Day-27 key mask turns into once each sequence owns its own table:
                  no pad token is ever written to the cache, so the read does not
                  need to be told which keys are fake, only how many are real.
    n_rep:        GQA repeat factor (`config.num_kv_groups`).
    scale:        softmax scale; defaults to `head_dim ** -0.5`.

    Returns [batch, n_q, 1, d], the attention output before o_proj. Row i is equal to
    `paged_attention_reference` run on row i's slots alone, which is the property the
    tests pin: batching is a throughput change and never a behaviour change.

    No causal mask appears here, and its absence is not an oversight. The single new
    query of a decode step sits at the end of its own history, so it may see every
    cached token; the only thing to exclude is the rectangle's padding, and that is
    what `context_lens` does. The bias is `finfo.min` rather than `-inf` for the
    reason Day 27 found the hard way, though here it is belt and braces: every row is
    required to have at least one real key, so no row can softmax over nothing.

    Reference, not the fast path: the gather still materializes `[batch, max_ctx]`
    tokens of K/V, which is both the contiguous buffer paging exists to avoid *and*
    a rectangle sized by the longest history in the batch. A kernel streams each
    row's own tiles and never pays for another row's length. This is the oracle that
    version gets held to.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be [batch, n_q, 1, d]; got {tuple(q.shape)}")
    if q.shape[2] != 1:
        raise ValueError(
            f"the batched read is the decode read: one new token per row, got "
            f"seq_q={q.shape[2]}. A ragged prefill goes through the dense masked path"
        )
    if k_pool.ndim != 3 or k_pool.shape != v_pool.shape:
        raise ValueError(
            "k_pool and v_pool must have the same shape [num_slots, n_kv, d]; got "
            f"{tuple(k_pool.shape)} and {tuple(v_pool.shape)}"
        )
    if slot_mapping.ndim != 2:
        raise ValueError(
            f"slot_mapping must be [batch, max_ctx]; got {tuple(slot_mapping.shape)}"
        )
    batch, n_q, _, d = q.shape
    max_ctx = slot_mapping.shape[1]
    if slot_mapping.shape[0] != batch or context_lens.shape != (batch,):
        raise ValueError(
            f"slot_mapping and context_lens must have one row each per sequence "
            f"(batch={batch}); got {tuple(slot_mapping.shape)} and "
            f"{tuple(context_lens.shape)}"
        )
    if int(context_lens.min()) < 1:
        raise ValueError(
            "context_lens must be at least 1 for every row: a query with no visible "
            "key softmaxes over nothing (0/0). A decode query always has its own token"
        )
    if int(context_lens.max()) > max_ctx:
        raise ValueError(
            f"context_lens claims more history than the mapping holds: max "
            f"{int(context_lens.max())} > max_ctx {max_ctx}"
        )
    if scale is None:
        scale = d**-0.5

    # Paged read, one gather for the whole batch: [batch, max_ctx, n_kv, d] ->
    # [batch, n_kv, max_ctx, d], the shape attention scores against.
    k = k_pool[slot_mapping].permute(0, 2, 1, 3)
    v = v_pool[slot_mapping].permute(0, 2, 1, 3)
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    scores = torch.matmul(q, k.transpose(2, 3)) * scale  # [batch, n_q, 1, max_ctx]
    beyond = torch.arange(max_ctx, device=q.device)[None, :] >= context_lens[:, None].to(q.device)
    scores = scores.masked_fill(beyond[:, None, None, :], torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(weights, v)


def paged_attention_kernel(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
    block: int = 32,
) -> torch.Tensor:
    """The fused paged attention, modeled on the CPU as a grid of tlsim programs.

    Same inputs and same output as `paged_attention_reference`, but the read, the
    score, and the softmax are folded into one streaming loop rather than a gather
    followed by a contiguous softmax. This is the loop next week's `triton.jit`
    kernel runs; here it is expressed on the Day-21 tlsim primitives so it can be
    written and pinned to the oracle on a box with no GPU.

    q, k_pool, v_pool, slot_mapping, n_rep, scale: exactly as in
    `paged_attention_reference` (one sequence, batch 1). `slot_mapping[p]` is the
    flat pool slot of logical position p, oldest first, over the whole history.
    block:   how many keys one program folds in per step. A pure performance knob
             (SRAM tile size on hardware); any value returns the same attention.

    Returns [1, n_q, seq_q, d], matching the reference to a few ulps. It is not
    bit-identical because the online softmax reassociates the exponent sums, the
    same accuracy trade every flash-attention kernel makes.

    The point paging exists for is honored here and not in the reference: the full
    `[kv_len, n_kv, d]` history is never assembled. One program owns one query
    position and walks the history a `block` of keys at a time, reading each tile
    through the block table with a masked `load`, so peak state is a single tile
    plus the per-head accumulators, not the whole past.
    """
    if q.shape[0] != 1:
        raise ValueError(
            "paged_attention_kernel handles one sequence; batch must be 1 "
            "(per-sequence batching arrives in Phase 3)"
        )
    n_q = q.shape[1]
    seq_q = q.shape[2]
    d = q.shape[3]
    if scale is None:
        scale = d**-0.5

    num_slots, n_kv, _ = k_pool.shape
    channels = n_kv * d
    seq_total = int(slot_mapping.shape[0])
    past = seq_total - seq_q  # tokens already in the cache before this step's queries

    # Flatten the pools to the 1-D buffers tlsim addresses; a token's K/V is
    # `channels` contiguous elements at `slot * channels`.
    slots = slot_mapping.to(torch.long)
    k_flat = k_pool.reshape(num_slots * channels)
    v_flat = v_pool.reshape(num_slots * channels)
    q_rows = q[0].transpose(0, 1).to(torch.float32)  # [seq_q, n_q, d], query per position
    head_kv = torch.arange(n_q, dtype=torch.long) // n_rep  # query head -> its KV head (GQA)
    cols = arange(0, channels)  # the channel ramp inside one token
    lane = arange(0, n_q * d)  # the output lanes of one query position

    out_flat = torch.zeros(seq_q * n_q * d, dtype=q.dtype)

    def kernel(prog, s_buf, k_buf, v_buf, dst) -> None:
        # This program owns query position i (absolute position past+i). Causal:
        # it may see keys 0..past+i, i.e. the first `kv_len` of the history.
        i = prog.program_id(0)
        qi = q_rows[i]  # [n_q, d]
        kv_len = past + i + 1
        # Online-softmax state, per query head: running max, denominator, and
        # weighted-V sum. These are the registers/SRAM a flash kernel keeps; the
        # history streams past them and is never held whole.
        m = torch.full((n_q,), float("-inf"))
        denom = torch.zeros(n_q)
        acc = torch.zeros(n_q, d)
        for b in range(cdiv(kv_len, block)):
            rows = b * block + arange(0, block)  # the key positions in this tile
            valid = rows < kv_len  # in-history and causally visible; guards the tail
            row_slots = load(s_buf, rows, mask=valid, other=0)  # each key's physical slot
            ptr = row_slots[:, None] * channels + cols[None, :]  # [block, channels]
            k_tile = load(k_buf, ptr, mask=valid[:, None], other=0.0).reshape(block, n_kv, d)
            v_tile = load(v_buf, ptr, mask=valid[:, None], other=0.0).reshape(block, n_kv, d)
            # Grow the compact KV heads to the query-head count (GQA), then score.
            k_exp = k_tile[:, head_kv, :].to(torch.float32)  # [block, n_q, d]
            v_exp = v_tile[:, head_kv, :].to(torch.float32)
            s = scale * torch.einsum("qd,jqd->qj", qi, k_exp)  # [n_q, block]
            # Masked-off keys score -inf so their weight is exactly zero (a masked
            # load returned zeros, which would otherwise score a spurious 0, not -inf).
            s = torch.where(valid[None, :], s, torch.full_like(s, float("-inf")))
            # Fold this tile into the accumulators: renormalize the running state to
            # the new max, then add the tile's contribution. Never holds the whole row.
            m_new = torch.maximum(m, s.max(dim=1).values)
            alpha = torch.exp(m - m_new)  # rescale factor for the old state
            p = torch.exp(s - m_new[:, None])  # [n_q, block] tile weights (unnormalized)
            denom = denom * alpha + p.sum(dim=1)
            acc = acc * alpha[:, None] + torch.einsum("qj,jqd->qd", p, v_exp)
            m = m_new
        out_i = (acc / denom[:, None]).to(q.dtype).reshape(-1)  # [n_q*d]
        store(dst, i * (n_q * d) + lane, out_i)

    launch(seq_q, kernel, slots, k_flat, v_flat, out_flat)
    return out_flat.reshape(seq_q, n_q, d).transpose(0, 1)[None]  # [1, n_q, seq_q, d]


# Day 23 translated this loop into a real `triton.jit` kernel in
# `.triton_paged_attention`, held to the reference above by tests gated on a GPU. Call
# `triton_paged_attention.paged_attention` for the dispatching entry point: it launches
# the kernel on CUDA and falls back to `paged_attention_kernel` here everywhere else.
# TODO(week6): benchmark the kernel against the Day-20 curve with the `readbench`
# harness, then wire the dispatcher into `gqa_attention`'s paged read.
