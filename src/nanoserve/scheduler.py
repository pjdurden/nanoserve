"""Scheduler: static then continuous batching. Weeks 7-9.

Week 7 batches a fixed set of sequences. Week 8 adds iteration-level scheduling
with waiting and running queues so requests join and leave mid-flight. Week 9
adds preemption and eviction under KV memory pressure (recompute policy).
"""


class Request:
    """One generation request moving through the queues."""

    def __init__(self, request_id, prompt_token_ids, sampling_params):
        raise NotImplementedError("week8")


class Scheduler:
    """Decides which requests run each engine step."""

    def __init__(self, allocator):
        raise NotImplementedError("week7")

    def step(self):
        """Pick the batch for the next forward pass."""
        raise NotImplementedError("week8")
