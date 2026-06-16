# nanoserve

A from-scratch LLM inference engine you can read in an afternoon. Think nanoGPT, but for serving instead of training.

Built in public over 100 days. The goal is a minimal, annotated inference engine: load Llama-3.2-1B, serve an OpenAI-compatible HTTP endpoint, with a paged KV cache, continuous batching, and a hand-written Triton paged-attention kernel. Roughly 1.5k to 2k lines, every one of them readable.

## Why

Production inference engines like vLLM and SGLang are 100k+ lines. The core ideas that make them fast (paged attention, continuous batching, iteration-level scheduling) are buried under that scale. nanoserve implements those same ideas in the smallest code that still does the real work, so you can actually read them.

## What it will do (v1, by Day 100)

- Load Llama-3.2-1B weights from safetensors into hand-written layers (RMSNorm, RoPE, GQA attention, SwiGLU).
- Generate text that matches HuggingFace token-for-token under greedy decoding.
- Sample with temperature, top-k, and top-p.
- Manage KV memory with a paged cache and a block allocator.
- Run a custom Triton paged-attention kernel.
- Batch many concurrent requests with continuous (iteration-level) batching and preemption.
- Serve an OpenAI-compatible `/v1/completions` endpoint with SSE streaming.

## What it will not do (v1)

Speculative decoding, tensor parallelism, prefix caching, quantization. Those are the v2 roadmap, on purpose. v1 stops at a correct, batched, served engine.

## Status

Day 0: scaffolding. See [docs/PLAN.md](docs/PLAN.md) for the full 100-day weekly plan.

## Target hardware

One small GPU (a single 4090 or A10 is plenty for a 1B model). Logic is developed against the HuggingFace reference; performance numbers are taken on the GPU.

## Layout

```
src/nanoserve/
  config.py        model config + names
  loader.py        safetensors -> tensors
  layers.py        RMSNorm, RoPE, attention, SwiGLU
  model.py         the transformer, forward pass
  cache.py         paged KV cache + block allocator
  kernels/         Triton paged-attention kernel
  sampling.py      greedy, temperature, top-k, top-p
  scheduler.py     waiting/running queues, continuous batching
  engine.py        ties model + cache + scheduler together
  server.py        FastAPI, OpenAI-compatible endpoint
```

## License

MIT.
