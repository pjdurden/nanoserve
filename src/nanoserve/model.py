"""The transformer: stack of blocks, final norm, LM head, forward to logits. Week 2.

Goal for week 2 is greedy decode that matches HuggingFace token-for-token.
"""


class LlamaModel:
    """Llama-3.2-1B forward pass built from nanoserve.layers."""

    def __init__(self, config):
        self.config = config
        # TODO(week2): embedding, list of blocks, final RMSNorm, lm_head

    def forward(self, input_ids, kv_cache=None):
        """Return logits for the next token."""
        raise NotImplementedError("week2: full forward pass")
