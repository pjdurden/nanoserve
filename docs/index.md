---
title: nanoserve
---

A from-scratch LLM inference engine you can read in an afternoon. Think nanoGPT, but for serving instead of training.

Production inference engines like vLLM and SGLang are 100k+ lines. The ideas that make them fast (paged attention, continuous batching, iteration-level scheduling) are buried under that scale. nanoserve implements those same ideas in the smallest code that still does the real work: roughly 1.5k to 2k annotated lines, built and posted in public over 100 days. Correctness before speed: every numerical piece is checked against the HuggingFace reference to 1e-5 before anything is optimized.

<p align="center">
  <img src="diagrams/architecture-overview.svg" alt="nanoserve architecture: the life of a request" width="840">
</p>

## Documentation

- [Architecture](ARCHITECTURE.md) — what gets built, how the pieces fit, and the two ideas (paged KV cache, continuous batching) that make it an engine. With diagrams.
- [The 100-day plan](PLAN.md) — the weekly build agenda, the done line, and what is deliberately out of scope.
- [Daily build log](daily/) — one short doc per day: what was added, why, and what was learned.

## Links

- Repo and README: [github.com/pjdurden/nanoserve](https://github.com/pjdurden/nanoserve)
- Build in public on X: [@pdurdenj](https://x.com/pdurdenj)
