# Week 1 daily plan (Days 2-7): skeleton to one verified block

Phase 0 Foundations. Day 1 (scaffold + docs + diagrams) is done. This breaks the
Week 1 menu into one pushable deliverable plus one post per day.

No GPU this week. Everything here runs in fp32 on CPU and is verified against the
HuggingFace reference. Rent the GPU around Week 5-6 for the Triton kernel.

## The verification harness (set up once, Day 2)

Every "verify to ~1e-5" below means: run the HF Llama-3.2-1B for the same input,
grab the intermediate activation, compare. Easiest way to get intermediates is a
forward hook on the HF module:

```python
acts = {}
def hook(name):
    def f(m, inp, out): acts[name] = out
    return f
hf_model.model.layers[0].input_layernorm.register_forward_hook(hook("rmsnorm0"))
# run hf_model(input_ids); then compare acts["rmsnorm0"] to your tensor
```

Keep this in `tests/reference.py`. Load the HF model once (fp32, CPU), expose a
helper that returns the activation dict for a fixed prompt. Every component test
asserts `torch.allclose(mine, acts[key], atol=1e-5)`.

---

## Day 2 (2026-06-17): weights + config

- Code: implement `scripts/download_weights.py` (huggingface_hub.snapshot_download
  for meta-llama/Llama-3.2-1B into ./weights; gated repo, accept license + hf login
  first). Inspect config.json and the safetensors index. Populate every field in
  `config.py:ModelConfig` from the real config.json. Add the verification harness
  skeleton in `tests/reference.py`.
- Verify: ModelConfig shapes match what the safetensors index reports.
- Post: the config tour. What is actually inside a 1B model: 16 layers, hidden 2048,
  8 KV heads vs 32 query heads (GQA), vocab 128k. The shape of the thing you are
  about to rebuild.

## Day 3: the loader

- Code: `loader.py:load_weights`. Read safetensors, build the explicit HF-key to
  nanoserve-attribute mapping dict, load tensors in place. This is the first rabbit
  hole: fused vs split qkv, the lm_head/embed_tokens tie, naming per layer.
- Verify: every expected key is consumed, no tensor left unmapped, shapes line up.
- Post: the weight-name-mapping rabbit hole. The unglamorous truth that half of
  "loading a model" is just getting the names right.

## Day 4: RMSNorm + RoPE

- Code: `layers.py` RMSNorm (~10 lines) and RoPE precompute + apply. RoPE is the one
  people get wrong: inv_freq, the cos/sin table, how it rotates q and k.
- Verify: RMSNorm output and post-RoPE q/k both within 1e-5 of the HF hook.
- Post (the reach one this week): RoPE explained by implementing it. Why position is
  a rotation, not an added vector, in maybe 25 lines. Screenshot the 1e-5 match.

## Day 5: SwiGLU MLP

- Code: `layers.py` SwiGLU (gate, up, down; silu(gate) * up then down).
- Verify: MLP output within 1e-5 of the HF layer's mlp hook.
- Post: SwiGLU, the gated MLP every modern model uses. Short, with the diff.

## Day 6: GQA attention

- Code: `layers.py` single-block attention. Q/K/V projections, RoPE on q/k, repeat
  KV heads to match query heads (GQA), causal mask, softmax, output projection. No
  KV cache yet, this is the plain prefill math.
- Verify: attention output within 1e-5 of the HF layer's self_attn hook. The usual
  mismatch suspects: mask, head reshape, the GQA repeat.
- Post: GQA in one picture. Why 32 query heads share 8 KV heads, and what that buys
  you in memory (foreshadows the whole paged-cache arc).

## Day 7: one full block, verified

- Code: assemble the transformer block in `layers.py` / wire enough of `model.py` to
  run a single block: input_layernorm, attention, residual, post_attention_layernorm,
  MLP, residual. Run it on the real loaded weights.
- Verify: full block output within 1e-5 of HF `model.layers[0]` output. This is the
  Week 1 done line.
- Post (milestone): one transformer block, rebuilt from scratch, matches HuggingFace
  to 1e-5. Week 1 recap: the map, the weights, the four pieces, the block. Next week
  is stacking 16 of these into a real forward pass.

---

## Notes

- Buffer rule: RMSNorm (D4) and SwiGLU (D5) are short. If you have an hour spare on a
  light day, start the next component so a bad day still has something to push.
- Bad-day escape hatch: a "stuck on the GQA repeat today" post still counts. Never
  break the streak.
- No em dashes in any post.
- Each day = one commit + one post. Keep `docs/daily/day-NN.md` per the template.
