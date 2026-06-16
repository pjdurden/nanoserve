"""The building blocks: RMSNorm, RoPE, GQA attention, SwiGLU MLP. Weeks 1-2.

Verify each piece against the HF reference to about 1e-5 before moving on. The
common mismatch sources are RoPE application, the causal mask, and dtype.
"""

# TODO(week1): RMSNorm
# TODO(week1): rotary position embedding (RoPE) precompute + apply
# TODO(week1): SwiGLU MLP (gate, up, down)
# TODO(week1): attention for a single block (GQA: repeat KV heads)
# TODO(week2): wire blocks together in model.py
