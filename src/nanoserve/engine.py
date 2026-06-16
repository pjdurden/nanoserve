"""The engine: ties model + paged cache + scheduler into a run loop. Weeks 5-9.

This is the object the server talks to. add_request() enqueues work; step()
runs one scheduler-chosen forward pass and emits finished or streamed tokens.
"""


class Engine:
    def __init__(self, model, cache, scheduler):
        self.model = model
        self.cache = cache
        self.scheduler = scheduler

    def add_request(self, request):
        raise NotImplementedError("week8")

    def step(self):
        """Run one iteration: schedule, forward, sample, update caches."""
        raise NotImplementedError("week8")

    def generate(self, prompt, sampling_params):
        """Offline convenience path used before the server exists (weeks 2-5)."""
        raise NotImplementedError("week2")
