"""Token sampling: greedy, temperature, top-k, top-p. Week 3.

Keep this pure: logits in, token id out. Test each sampler in isolation.
"""


def sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Return the next token id from a logits vector."""
    raise NotImplementedError("week3: greedy + temperature + top-k + top-p")
