# KV Cache & Inference Optimization

> **One-liner:** Generating text is two different workloads -- a fast, parallel **prefill** over the prompt and a slow, one-token-at-a-time **decode** limited by [memory bandwidth](../glossary.md#memory-bandwidth) -- and every LLM inference optimization (the [KV cache](../glossary.md#kv-cache), cache quantization, prompt caching, batching, speculative decoding) targets one of the two.

## The Intuition

[Attention](06-attention.md) introduced the KV cache: when a model generates token 101, the keys and values of tokens 1-100 haven't changed, so it stores them instead of recomputing them. That single idea turns generation from quadratic to linear work. This page is about what comes *after* that idea: how big the cache gets, why generation speed is capped by your Mac's memory bandwidth rather than its GPU, and which of the classic speed-ups actually pay off on Apple Silicon.

The key picture is that an LLM call has two phases with opposite bottlenecks:

```
PREFILL (reading the prompt)              DECODE (writing the answer)

  all N prompt tokens in ONE pass           ONE token per pass, N passes
  big matrix-matrix products                thin matrix-vector products
  -> COMPUTE-bound                          -> MEMORY-BANDWIDTH-bound
  fills the KV cache                        reads ALL weights + the whole
                                              KV cache for every token
  measured: 123 tok/s                       measured: 16 tok/s
```

(Measured with `mlx-lm` 0.31.3 on an M2 Pro, Gemma 3 12B 4-bit, 1,802-token prompt.) The same model processes prompt tokens ~8x faster than it produces new ones, because prefill does lots of arithmetic per byte of weights it reads, and decode does almost none.

---

## How It Actually Works

### Decode speed has a hard ceiling: bandwidth ÷ bytes per token

To produce one token, the GPU must stream every weight of the model from memory (plus the KV cache). The arithmetic is trivial by comparison. So the upper bound on decode speed is:

```
  max tokens/s  ≈  memory bandwidth  /  bytes read per token

  Gemma 3 12B, 4-bit:  7.19 GB of weights
  M2 Pro:              ~200 GB/s
  ceiling:             200 / 7.19 ≈ 28 tok/s
  measured:            16 tok/s  (~60% of the ceiling -- typical)
```

This one formula explains most of what you see in practice:

- **Quantization speeds up decode** because it shrinks the bytes read per token, not because int4 math is faster (see [Quantization](10-quantization.md)).
- **An M-series Max/Ultra decodes faster than a Pro** mostly because of bandwidth (400-800 GB/s vs 200 GB/s), not GPU core count.
- **A long context slows decode down**: the KV cache is read on every token too, so at long contexts it adds to the bytes per token.

### How big the KV cache gets

Each layer stores one key and one value vector per KV head, per token:

```
  KV bytes = 2 (K and V) × layers × kv_heads × head_dim × bytes_per_value × tokens

  Gemma 3 12B: 2 × 48 × 8 × 256 × 2 bytes (bf16) = 384 KB per token

     2K tokens  ->  0.76 GB
    32K tokens  ->  12.9 GB
   128K tokens  ->  51.5 GB   (more than the 4-bit weights, 7 times over)
```

Long contexts are a memory problem before they are a speed problem. Three techniques shrink the cache, and model architectures now build some of them in:

```
SHRINKING THE KV CACHE

  Technique           What it does                          Who decides
  ---------           ------------                          -----------
  GQA                 fewer KV heads than query heads       the model
                        (Gemma 3 12B: 16 query, 8 KV)         (config: num_key_value_heads)
  Sliding window      some layers only keep the last W      the model
                        tokens (RotatingKVCache)              (config: sliding_window)
  KV quantization     store K/V in 8 or 4 bits              you (--kv-bits)
```

**Sliding window, measured.** Gemma 3 12B is a *hybrid*: 40 of its 48 layers attend only to the last 1,024 tokens; 8 layers are global. `mlx-lm` builds the right cache per layer (40 `RotatingKVCache`, 8 `KVCache`) via the model's `make_cache()`. On a 1,930-token run the measured cache was **462 MB instead of 760 MB**, exactly the hybrid formula; at 128K tokens the difference is 8.9 GB vs 51.5 GB.

**`--max-kv-size` is not the same thing.** For a model *without* its own `make_cache`, `mlx-lm`'s `--max-kv-size N` makes every layer a rotating cache that keeps the first 4 tokens plus the last N-4. That bounds memory for any model, but the model was never trained to lose old context, so answers that depend on the forgotten part degrade. For models *with* `make_cache` (like Gemma 3), the flag is ignored.

**KV quantization** (`--kv-bits 8` or `4`) stores the cache like quantized weights. Trade-offs, from the `mlx-lm` docs and code:

- quantization starts after `--quantized-kv-start` tokens (default 5,000), since short caches aren't worth it;
- attention on a quantized cache isn't the fused kernel, so it materializes a `prefill_step_size × context` score matrix -- lower `--prefill-step-size` or you can lose what you saved;
- rotating (sliding-window) caches can't be quantized yet: on Gemma 3 it raises `NotImplementedError: RotatingKVCache Quantization NYI`;
- in `mlx_lm.server`, it disables batching.

### Prompt caching: never prefill the same prefix twice

If many requests share a prefix (a long system prompt, a document you ask several questions about, a chat's history), its KV cache can be computed once and reused. Only the new suffix needs prefilling.

```bash
# Prefill a long prefix once, save its KV cache to disk
mlx_lm.cache_prompt --model <model> --prompt - --prompt-cache-file doc.safetensors < document.txt

# Reuse it: only the question is prefilled
mlx_lm.generate --model <model> --prompt-cache-file doc.safetensors --prompt "Summarize section 3."
```

`mlx_lm.server` does this automatically: it keeps an LRU cache of recent KV caches (`--prompt-cache-size`, `--prompt-cache-bytes`) and reuses the longest matching prefix of each new request. For chat, that means each turn only prefills the new message.

### Batching: sharing one weight read across requests

Decode reads all the weights to produce one token. If eight requests decode together, that same read produces eight tokens. `mlx-lm` implements **continuous batching** (`BatchGenerator`, `batch_generate`, and in `mlx_lm.server` via `--decode-concurrency` / `--prompt-concurrency`): requests join and leave the batch as they arrive and finish.

Measured on the same M2 Pro / Gemma 3 12B 4-bit, 128 tokens per request:

```
  concurrent requests   aggregate decode speed   per request
          1                  16.2 tok/s             16.2
          4                  22.4 tok/s              5.6
          8                  25.8 tok/s              3.2
```

Throughput rises, but only 1.6x at 8 requests -- far from 8x. At this model size and bandwidth, a batch of tokens is **not** free: the quantized matmuls start costing real compute as the batch grows. Batching helps a server with many users; it does nothing for a single user's latency. Batching is disabled in the server when a draft model or `--kv-bits` is used.

### Speculative decoding: guess cheaply, verify in one pass

A small **draft model** proposes the next *k* tokens; the big model checks all *k* in a single forward pass (like a tiny prefill) and keeps the longest correct run, plus one token of its own. When the draft is usually right, you get several tokens for roughly the price of one big-model pass. The output distribution is unchanged: every kept token is one the big model accepts.

```bash
mlx_lm.generate --model <big> --draft-model <small> --num-draft-tokens 2 --prompt "..."
```

The draft must share the big model's tokenizer (`mlx-lm` checks the vocabulary size and refuses otherwise).

Measured on the same setup, Gemma 3 1B 4-bit drafting for Gemma 3 12B 4-bit, 256 tokens of code:

```
  draft    k    decode speed   tokens accepted from draft
  none     -     16.5 tok/s          -
  1B       2     17.0 tok/s       158/256  (62%)
  1B       4     15.6 tok/s       190/256  (74%)
```

Acceptance is good, yet the speed-up is ~3% at k=2 and negative at k=4. The reason is the batching result above: speculative decoding assumes that verifying *k+1* tokens costs about the same as generating one. On this machine and model it doesn't, and the draft model's own passes add up. Speculative decoding pays off when the big model is strongly bandwidth-bound (bigger models, higher-bandwidth chips) and the draft is much smaller. **Measure it on your hardware before relying on it.**

---

## Why It Matters for MLX

### Porting the cache correctly

Incremental generation is where many ports break, because the cache couples three things that must stay in sync: the stored K/V, the position offset, and the attention mask.

```
KV-CACHE PORT FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  First token fine, then gibberish           RoPE not using the cache offset:
                                               new tokens encoded at position 0
                                               (see Positional Encoding)
  Model "forgets" the prompt after the        cache never updated, or ignored
    first generated token
  Matches the reference on short prompts,    sliding-window layers treated as
    drifts after ~W tokens                     global (or vice versa)
  Output differs from reference only with    causal mask built for the new
    a multi-token prefill chunk                tokens alone, not offset against
                                               the cached ones
  Memory grows much faster than the          full-size cache on layers the
    formula predicts                           config says are sliding-window,
                                               or K/V repeated to all query heads
                                               before caching (breaks GQA savings)
  Decode gets slower every token             cache concatenated each step
                                               instead of preallocated (mlx-lm's
                                               KVCache grows in 256-token steps)
```

Three rules:

1. **Reuse `mlx_lm.models.cache`.** `KVCache`, `RotatingKVCache`, `QuantizedKVCache` and `make_prompt_cache` already handle offsets, preallocation, trimming, and saving. If the architecture mixes layer types (sliding vs global, attention vs state-space), give the model a `make_cache()` that returns the right cache per layer, as Gemma 3 does.
2. **Cache the K/V heads as stored, before GQA repetition.** Cache `num_key_value_heads` heads, not `num_attention_heads`, or you lose GQA's memory savings.
3. **Test the cached path against the uncached one.** Generate greedily with the cache, then recompute the full sequence without it; the logits for each position must match (see [Testing & Validation](../03-contributing/05-testing-validation.md)).

### What to reach for, in order

```
  Goal                           First try                       Then
  ----                           ---------                       ----
  Faster decode (1 user)         smaller quantization (4-bit)    speculative decoding,
                                                                   if measured faster
  Long context fits in memory    model's own sliding window/GQA  --kv-bits 8
  Repeated long prompts          prompt cache (file or server)   --
  Many users                     mlx_lm.server batching          --
```

---

## Key Terms

| Term | Definition |
|------|------------|
| **Prefill** | The first phase of generation: processing all prompt tokens in one parallel pass and filling the KV cache; compute-bound. |
| **Decode** | The generation phase: producing one token per forward pass; memory-bandwidth-bound because every pass reads all weights and the KV cache. |
| **KV cache** | Stored keys and values of past tokens, per layer, so each new token only computes its own. |
| **Sliding-window attention** | Layers that attend only to the last W tokens, so their cache stops growing at W (`RotatingKVCache` in mlx-lm). |
| **KV cache quantization** | Storing cached keys/values in 8 or 4 bits to cut long-context memory. |
| **Prompt caching** | Saving and reusing the KV cache of a shared prefix so it is prefilled only once. |
| **Continuous batching** | Serving several requests in one decode batch, with requests joining and leaving as they arrive and finish. |
| **Speculative decoding** | A small draft model proposes several tokens that the large model verifies in one pass; same output distribution, faster only when verification is nearly free. |
| **Draft model** | The small, fast model that proposes tokens in speculative decoding; must share the large model's tokenizer. |

---

## Sources

- Measurements on this page: `mlx-lm` 0.31.3 / MLX 0.31.1, Apple M2 Pro (32 GB), `mlx-community/gemma-3-12b-it-4bit` with `mlx-community/gemma-3-1b-it-4bit` as draft. Absolute numbers vary by chip; the ratios are the lesson.
- `mlx-lm` source: `mlx_lm/models/cache.py` (cache classes, `make_prompt_cache`), `mlx_lm/generate.py` (`speculative_generate_step`, `BatchGenerator`), `mlx_lm/SERVER.md` (KV quantization and batching caveats): [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." Prefill vs decode, memory-bandwidth bounds, KV cache costs. [arxiv.org/abs/2211.05102](https://arxiv.org/abs/2211.05102)
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding." [arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)
- Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." [arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)
- Gemma Team (2025). "Gemma 3 Technical Report." Interleaved local (sliding-window) and global attention layers. [arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786)
- Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM). [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

---

## See Also

- [Attention](06-attention.md) -- prerequisite; what the KV cache stores and how GQA shrinks it
- [GPU Computing](04-gpu-computing.md) -- memory bandwidth vs compute, the reason decode is slow
- [Quantization](10-quantization.md) -- why fewer bytes per weight means faster decode
- [Positional Encoding](11-positional-encoding.md) -- the RoPE `offset` that must follow the cache
- [Lazy Evaluation](15-lazy-evaluation.md) -- evaluating once per token, and `mx.async_eval` in the generation loop
- [LLMs](../02-ecosystem/03-llms.md) and [Serving & Deployment](../02-ecosystem/09-serving-deployment.md) -- the tools that implement these techniques, CUDA vs MLX
