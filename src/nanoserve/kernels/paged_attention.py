"""Triton paged-attention kernel. Week 6 (the headline artifact).

Reads K and V through the block table and computes attention without ever
materializing a contiguous KV tensor. Always verify against a plain torch
reference (same inputs, same output to about 1e-3) before trusting it.
"""

# TODO(week6): triton.jit kernel for paged attention
# TODO(week6): torch reference implementation for correctness checks
# TODO(week6): microbenchmark kernel vs torch path
