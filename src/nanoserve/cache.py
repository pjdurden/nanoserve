"""Paged KV cache and block allocator. Weeks 3-5.

Week 3 ships a naive contiguous cache that grows per step. Weeks 4-5 replace it
with a paged cache: a fixed pool of physical blocks plus a per-sequence block
table that maps logical token positions to physical blocks. This is the OS-paging
analogy at the heart of the engine.

Why a cache at all? Without one, generating token n re-runs attention over the
whole 0..n-1 prefix every step, because attention needs every past key and value.
But those keys and values only depend on tokens that are already fixed: token 3's
K and V never change once token 3 is in the sequence. So the recompute is pure
waste. The cache is the one idea that turns an O(n^2) decode (every step redoes
all the prefix work) into O(n) (every step does one token's work and reads the
rest): compute each token's K/V exactly once, keep them, and append.
"""

from __future__ import annotations

import torch


class NaiveKVCache:
    """Week 3: contiguous per-layer K/V store that grows by concatenation.

    One (K, V) pair per transformer layer, each shaped
    `[batch, num_kv_heads, seq_so_far, head_dim]`. "Naive" and "contiguous" mean
    the store is literally one growing tensor per layer: every decode step
    `torch.cat`s the new token's K/V onto the end and hands the whole thing back.
    That is the simplest thing that works and the thing Weeks 4-5 will replace,
    because a contiguous per-sequence buffer is exactly what fragments VRAM and
    forces you to pre-reserve max-length space for every request. For a single
    sequence on Week 3 it is perfect, and it makes the speedup visible before the
    paging machinery arrives.

    Crucially the cache holds the *compact* GQA K/V (8 KV heads here), not the
    repeated 32-head version: the head expansion is a cheap view applied at read
    time in attention, so storing the repeat would quadruple the cache for nothing
    and throw away the entire reason GQA exists.
    """

    def __init__(self, num_layers: int):
        self.k: list[torch.Tensor | None] = [None] * num_layers
        self.v: list[torch.Tensor | None] = [None] * num_layers

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's K/V for `layer` and return the full running K/V.

        k, v: [batch, num_kv_heads, new_seq, head_dim]. `new_seq` is the prompt
              length on the prefill call and 1 on each decode step. The first
              append for a layer seeds the store; later ones concatenate on the
              sequence axis (dim 2), oldest token first.

        Returns the complete (K, V) for the layer so attention can score the new
        query against the entire history without the model holding the past
        itself.
        """
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    @property
    def seq_len(self) -> int:
        """Tokens cached so far (read off layer 0; all layers stay in lockstep)."""
        return 0 if self.k[0] is None else self.k[0].shape[2]


class BlockAllocator:
    """Weeks 4-5: fixed pool of physical KV blocks, alloc and free."""

    def __init__(self, num_blocks, block_size):
        raise NotImplementedError("week4")


class PagedKVCache:
    """Weeks 4-5: block table mapping logical positions to physical blocks."""

    def __init__(self, config, allocator):
        raise NotImplementedError("week5")
