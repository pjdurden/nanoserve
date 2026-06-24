---
title: "Day 01: scaffolding and the plan"
parent: Daily log
nav_order: 1
---

# Day 01: scaffolding and the plan

Date: 2026-06-16 · Week 1 · Phase 0 Foundations

## What I added today
Repo skeleton for nanoserve: package layout under `src/nanoserve/`, the 100-day plan in `docs/PLAN.md`, the system design in `docs/ARCHITECTURE.md`, and three diagrams in `docs/diagrams/`. Every module is a stub that names the week it gets implemented.

## Why it matters
Setting the done line before writing a single forward pass is the whole game. v1 stops at a correct, batched, served engine running Llama-3.2-1B. Speculative decoding and tensor parallelism are explicitly out, they are the v2 teaser. The structure exists so the daily work is "fill in the next stub," never "what do I do today."

## What I learned
The hard part of an inference engine is not the transformer, it is the memory and the scheduling. The architecture doc already makes that clear: layers and model are standard, the two ideas that earn the name "engine" are the paged KV cache and continuous batching. Naming that on Day 1 sets the right focus for the next 14 weeks.

## Diagram
[architecture-overview.svg](../diagrams/architecture-overview.svg) — the life of a request.

## Tomorrow
Week 1 menu: download Llama-3.2-1B, inspect config.json, fill in `config.py`, start the safetensors name mapping in `loader.py`.

---
Post angle: I am building an AI inference engine from scratch in 100 days. Not a wrapper, the actual thing: paged KV cache, continuous batching, a Triton kernel. Day 1 is the map. Here is what the next 100 days build and why.
