# nanoserve: the 100-day plan

A from-scratch nano-vLLM, built and posted in public daily over 100 days.

- Window: 2026-06-16 to roughly 2026-09-24 (Day 100).
- Model target: Llama-3.2-1B (RoPE, RMSNorm, GQA, SwiGLU).
- Math: PyTorch ops for matmuls and norms, plus one hand-written Triton paged-attention kernel.
- Hardware: one rented small GPU (4090 or A10). Logic verified against the HuggingFace reference.
- Output: roughly 1.5k to 2k annotated lines, plus a series of written deep-dives.

## Rules of the build

1. **The streak is the product.** One push and one post every day. A "stuck on X today" post counts, never break the streak.
2. **Buffer.** Stay 2 to 3 days ahead on code so a bad day still has something to post.
3. **Two post tiers.** Roughly 6 to 8 milestone posts built for reach, the other ~92 are grind posts (a diff, a graph, a bug, one thing learned).
4. **No em dashes in posts.**
5. **The done line is Phase 4.** Paged cache plus continuous batching plus an OpenAI-compatible server running Llama-3.2-1B correctly, tagged v1.0. Phases 5 and 6 are polish only. If you finish early you optimize and document, you do not add new features. Speculative decoding, tensor parallelism, and prefix caching are the v2 teaser.

## Each week is a menu

Pick the day's task from that week's list based on where you are. Weeks are agendas, not rigid day assignments.

---

## Phase 0: Foundations

### Week 1 (Days 1-7): skeleton and a single block
- Code: create repo and manifesto README; download Llama-3.2-1B; inspect safetensors and config; load weights into your own tensors (name mapping); implement RMSNorm; implement RoPE; implement SwiGLU MLP; assemble one transformer block; verify each piece against HF to about 1e-5.
- Posts: "Day 1: building an AI inference engine from scratch in 100 days, here is why"; the weight-name-mapping rabbit hole; RoPE explained by implementing it; the first numerical-match screenshot.

## Phase 1: Correct generation

### Week 2 (Days 8-14): full forward pass and greedy
- Code: stack all layers; final norm and LM head; full forward to logits; greedy decode loop; match HF token-for-token; debug the inevitable mismatch (usually RoPE, causal mask, or dtype).
- Posts: stacking 16 blocks; the mismatch hunt; "Day 14: my engine and HuggingFace produce identical tokens" (milestone).

### Week 3 (Days 15-21): sampling and naive cache
- Code: temperature; top-k; top-p (nucleus); sampling correctness tests; naive contiguous KV cache that grows per step; measure the speedup caching gives.
- Posts: why greedy is boring, sampling visualized; top-p explained; "KV cache: the one idea that makes inference fast" (milestone).

## Phase 2: Paged KV cache

### Week 4 (Days 22-28): block allocator
- Code: design the block table; physical block pool plus allocator (alloc and free); map logical token positions to physical blocks; unit-test allocation and eviction.
- Posts: why a contiguous KV cache wastes most of your VRAM; the OS-paging analogy; the allocator design diagram.

### Week 5 (Days 29-35): paged cache wiring
- Code: rewrite attention to read and write KV through the block table (still torch math); single-sequence generation through the paged cache; verify output is identical to the naive cache.
- Posts: PagedAttention without the kernel, first; a before and after VRAM-fragmentation graph; debugging a gather bug.

### Week 6 (Days 36-42): the Triton kernel
- Code: learn Triton basics (1 to 2 days, post the learning); write the paged-attention Triton kernel; verify against the torch reference; benchmark the kernel against the torch path.
- Posts: day 1 of writing my first GPU kernel; what Triton actually is; "Day 40: I wrote a paged-attention kernel, here is why vLLM needs one" (milestone).

## Phase 3: batching and scheduler

### Week 7 (Days 43-49): static batching
- Code: batch multiple sequences in one forward; ragged and padded handling; per-sequence block tables in a batch; correctness across the batch.
- Posts: static batching and why it is not enough; head-of-line blocking explained.

### Week 8 (Days 50-56): continuous batching core
- Code: waiting and running queues; iteration-level scheduler; admit new requests mid-flight; separate prefill from decode steps.
- Posts: continuous batching, the trick behind every serving stack; the scheduler state-machine diagram; a request-timeline graphic.

### Week 9 (Days 57-63): preemption and polish
- Code: preemption and eviction under memory pressure; recompute-versus-swap policy (pick recompute for simplicity); stress-test many concurrent sequences; fix scheduler edge cases.
- Posts: what happens when the GPU runs out of KV blocks; "Day 60: many requests, one GPU, no blocking" (milestone).

## Phase 4: serving layer (the done line)

### Week 10 (Days 64-70): HTTP server
- Code: FastAPI app; request queue to scheduler bridge; OpenAI-compatible `/v1/completions` request and response schema; basic error handling.
- Posts: wiring an async server to a sync GPU loop; the OpenAI-compat schema tour.

### Week 11 (Days 71-77): streaming and DONE
- Code: SSE token streaming; the `stream=true` path; concurrency hardening; Phase 4 done, curl it, OpenAI-compatible, correct, batched; tag v1.0.
- Posts: "Day 75: curl my from-scratch engine"; a live streaming demo GIF; the core is done, the rest is speed (milestone).

## Phase 5: performance and measurement

### Week 12 (Days 78-84): benchmark harness
- Code: throughput plus latency (TTFT, ITL) measurement; baseline against HF generate; sweep batch sizes and sequence lengths; produce clean graphs.
- Posts: how to actually benchmark an inference engine; your first numbers versus HF; a latency-versus-throughput tradeoff graph.

### Week 13 (Days 85-91): optimize
- Code: CUDA graphs or torch.compile on decode; reduce per-step Python overhead; profile (Nsight or torch profiler); chase the top 2 to 3 hotspots.
- Posts: where my engine spent its time, a profiling teardown; before and after each optimization; "Day 90: Nx faster than naive" (milestone).

## Phase 6: capstone

### Week 14 (Days 92-98): make it readable
- Code: annotation pass (this is what makes it nanoGPT-grade); architecture diagrams; compile deep-dives into README chapters; final correctness and benchmark re-run.
- Posts: the annotated-code tour; read a whole inference engine in an afternoon; the diagram set.

### Week 15 (Days 99-100): ship the story
- Code: final polish; v2 roadmap (speculative decoding, tensor parallelism) as the teaser; pinned repo and thread.
- Posts: "Day 100: nanoserve" launch thread recapping the arc and linking every deep-dive; the retrospective on what you learned; the v2 teaser to keep followers.
