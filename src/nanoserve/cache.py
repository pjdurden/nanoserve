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


class KVCacheExhausted(RuntimeError):
    """Raised when the block pool cannot satisfy an allocation.

    This is the engine's out-of-memory signal, scoped to KV cache. On a single
    sequence it just means "context too long for the pool", but it is the same
    event that, once many sequences share the pool (Weeks 8-9), triggers
    scheduling decisions: stop admitting new requests, or preempt and recompute a
    running one to reclaim its blocks. Naming it now means the later scheduler can
    catch one specific thing instead of a bare RuntimeError.
    """


class BlockAllocator:
    """Week 4: a fixed pool of physical KV blocks, with alloc and free.

    This is the OS-paging analogy made literal. Physical memory is carved into
    `num_blocks` fixed-size blocks, each holding `block_size` tokens' worth of
    K/V. The allocator owns nothing about tensors yet (Week 5 hangs the real K/V
    storage off these ids); it is pure bookkeeping over integer block ids, and
    keeping it that way is the point. The hard part of paging is not the storage,
    it is never losing or double-handing-out a block, so the whole component is a
    free list plus an allocated set, and the invariant (every block is in exactly
    one of them) is what the tests pin.

    Why fixed blocks at all? The Week-3 contiguous cache forces each sequence to
    reserve one growing buffer sized to its worst case, which fragments memory and
    caps how many sequences fit. Blocks decouple a sequence's *logical* length
    from *physical* placement: a 40-token sequence with block_size 16 takes three
    blocks that can sit anywhere in the pool, and frees them independently when it
    finishes. That is what lets the pool stay packed under many concurrent
    sequences, which is the entire reason paged attention exists.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError(
                f"num_blocks and block_size must be positive, got "
                f"num_blocks={num_blocks}, block_size={block_size}"
            )
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Free ids kept as a stack. Seeded high-to-low so pop() hands out 0, 1, 2,
        # ...: not required for correctness, but it makes allocation order
        # deterministic and the tests legible. `_allocated` is the other half of
        # the invariant: a block is in exactly one of these two at all times.
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._allocated: set[int] = set()

    @property
    def num_free(self) -> int:
        """Physical blocks available right now."""
        return len(self._free)

    def blocks_for_length(self, num_tokens: int) -> int:
        """How many blocks a sequence of `num_tokens` needs (ceiling division).

        Zero tokens need zero blocks; otherwise the last partial block still costs
        a whole physical block, because a block is the unit of allocation. This is
        the only place the token count meets the block count.
        """
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        """Whether the pool can currently hold a `num_tokens` sequence."""
        return self.blocks_for_length(num_tokens) <= self.num_free

    def allocate(self) -> int:
        """Hand out one free physical block, or raise if the pool is dry."""
        if not self._free:
            raise KVCacheExhausted(
                f"block pool exhausted: all {self.num_blocks} blocks allocated"
            )
        block_id = self._free.pop()
        self._allocated.add(block_id)
        return block_id

    def allocate_for(self, num_tokens: int) -> list[int]:
        """Reserve all blocks a `num_tokens` sequence needs, atomically.

        Either every needed block is reserved and returned, or the pool is left
        exactly as it was and `KVCacheExhausted` is raised. The atomicity matters:
        a half-filled allocation would strand blocks that no sequence owns and no
        one ever frees, so the count is checked up front before any block moves.
        """
        need = self.blocks_for_length(num_tokens)
        if need > self.num_free:
            raise KVCacheExhausted(
                f"need {need} blocks for {num_tokens} tokens, only "
                f"{self.num_free} free"
            )
        return [self.allocate() for _ in range(need)]

    def free(self, block_id: int) -> None:
        """Return one block to the pool. Rejects anything not currently allocated.

        Freeing a block that is already free (a double free) or never belonged to
        this pool is always a bug in the caller, and a silent one corrupts the
        pool: the same block would be handed to two sequences. So it raises rather
        than guessing.
        """
        if block_id not in self._allocated:
            raise ValueError(
                f"block {block_id} is not allocated (double free or foreign id)"
            )
        self._allocated.discard(block_id)
        self._free.append(block_id)

    def free_all(self, block_ids: list[int]) -> None:
        """Return a whole sequence's blocks to the pool, e.g. when it finishes."""
        for block_id in block_ids:
            self.free(block_id)


class BlockTable:
    """Week 4: one sequence's map from logical token position to physical block.

    The allocator owns the pool but is blind to meaning: it hands out block ids
    and takes them back, nothing more. The block table is the per-sequence layer
    on top that remembers *which* physical block holds *which* logical token. It
    is the address-translation table of the paging analogy, the exact counterpart
    of an OS page table: logical position -> (physical block, offset within it).

    A sequence is just an ordered list of block ids, `block_ids[i]` holding
    logical positions `i*block_size .. i*block_size + block_size - 1`. Those
    physical blocks can sit anywhere in the pool and need not be contiguous (the
    allocator hands out whatever is free), which is the entire freedom paging buys
    over the Week-3 contiguous cache. Translation stays trivial regardless:
    position `p` lives in `block_ids[p // block_size]` at offset `p % block_size`.

    Still torch-free. No K/V is stored here; the table only computes *where* each
    token's K/V will go. Week 5 hangs the real per-block K/V tensors off the pool
    and writes/reads them at the `slot` this table returns.
    """

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.block_ids: list[int] = []
        self.num_tokens = 0

    @property
    def capacity(self) -> int:
        """Tokens the currently held blocks can hold (>= num_tokens)."""
        return len(self.block_ids) * self.block_size

    def append(self, num_new_tokens: int = 1) -> None:
        """Grow the sequence by `num_new_tokens`, pulling blocks only as needed.

        New blocks are reserved exactly when growth crosses a block boundary: a
        token at position `capacity` has no block yet, so one is allocated. On the
        common decode step (one token, mid-block) no allocation happens at all,
        which is the whole efficiency story: physical memory is grabbed a block at
        a time, only at the moment it is first written.

        Atomic like `allocate_for`: if the growth needs more blocks than the pool
        can supply, none are taken and `KVCacheExhausted` is raised, leaving the
        table exactly as it was. A half-grown table would strand blocks and desync
        `num_tokens` from the blocks actually held.
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be non-negative, got {num_new_tokens}")
        target = self.num_tokens + num_new_tokens
        blocks_needed = self.allocator.blocks_for_length(target) - len(self.block_ids)
        if blocks_needed > 0:
            # allocate_for is itself atomic: it reserves all or raises, taking
            # none, so a failure here cannot leave a partially grown table.
            self.block_ids.extend(self.allocator.allocate_for(blocks_needed * self.block_size))
        self.num_tokens = target

    def position(self, logical_pos: int) -> tuple[int, int]:
        """Translate a logical token position to (physical block id, offset).

        Raises `IndexError` for any position not currently live (negative or
        `>= num_tokens`), because a translation past the sequence would point at a
        block that holds no K/V for this token yet.
        """
        if logical_pos < 0 or logical_pos >= self.num_tokens:
            raise IndexError(
                f"position {logical_pos} out of range for {self.num_tokens} tokens"
            )
        block_id = self.block_ids[logical_pos // self.block_size]
        offset = logical_pos % self.block_size
        return block_id, offset

    def slot(self, logical_pos: int) -> int:
        """Flat index of a token's K/V in a `[num_blocks * block_size]` pool.

        This is the single integer Week 5's paged read/write uses: rather than
        index a `[block][offset]` tensor, flatten the pool to one axis and address
        it as `block_id * block_size + offset`. Distinct live positions always map
        to distinct slots, even when the sequence's blocks are scattered.
        """
        block_id, offset = self.position(logical_pos)
        return block_id * self.block_size + offset

    def free(self) -> None:
        """Return every block to the pool and reset to an empty, reusable table."""
        self.allocator.free_all(self.block_ids)
        self.block_ids = []
        self.num_tokens = 0


class PagedKVCache:
    """Week 5: paged K/V storage read and written through a BlockTable."""

    def __init__(self, config, allocator):
        raise NotImplementedError("week5")
