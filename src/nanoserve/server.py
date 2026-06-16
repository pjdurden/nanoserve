"""FastAPI server: OpenAI-compatible /v1/completions with SSE streaming. Weeks 10-11.

Bridges async HTTP requests onto the synchronous engine.step() loop. Week 10 is
the non-streaming endpoint; week 11 adds stream=true via server-sent events and
hardens concurrency. Phase 4 done = this serves Llama-3.2-1B correctly. Tag v1.0.
"""

# TODO(week10): FastAPI app, request/response schemas, /v1/completions
# TODO(week10): request queue bridging HTTP -> Engine.add_request
# TODO(week11): SSE streaming for stream=true
