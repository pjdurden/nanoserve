"""Paged KV cache and block allocator. Weeks 3-5.

Week 3 ships a naive contiguous cache that grows per step. Weeks 4-5 replace it
with a paged cache: a fixed pool of physical blocks plus a per-sequence block
table that maps logical token positions to physical blocks. This is the OS-paging
analogy at the heart of the engine.
"""


class NaiveKVCache:
    """Week 3: contiguous per-sequence cache, grows each step."""

    def __init__(self, config):
        raise NotImplementedError("week3")


class BlockAllocator:
    """Weeks 4-5: fixed pool of physical KV blocks, alloc and free."""

    def __init__(self, num_blocks, block_size):
        raise NotImplementedError("week4")


class PagedKVCache:
    """Weeks 4-5: block table mapping logical positions to physical blocks."""

    def __init__(self, config, allocator):
        raise NotImplementedError("week5")
