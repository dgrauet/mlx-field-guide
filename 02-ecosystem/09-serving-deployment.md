# Serving & Deployment: CUDA vs MLX

> **Status:** 🟢 [CUDA](../glossary.md#cuda) (Mature -- vLLM, TGI, TensorRT-LLM cover every production scenario) | 🟡 MLX (Growing -- local and single-user production covered; multi-tenant high-throughput still maturing)

## What This Domain Covers

Running a model behind an API, serving it to multiple concurrent users, shipping it inside a desktop or mobile app, or distributing [inference](../glossary.md#inference) across multiple devices. This page is the bridge between "the model runs on my laptop" and "the model is in production."

For a developer porting a [CUDA](../glossary.md#cuda) serving setup to MLX, the question is almost never "can MLX run this model" -- it usually can. The question is: "what does the equivalent of vLLM look like for Apple Silicon?" The short answer: there isn't a one-to-one equivalent yet, but there are several good-enough options depending on the target scenario.

---

## The CUDA Ecosystem

### Server-Class Inference

The production standard is **vLLM**. Built around paged attention (introduced in the original vLLM paper, 2023), it manages the [KV cache](../glossary.md#kv-cache) in fixed-size pages so that concurrent requests can share GPU memory efficiently. Combined with continuous batching -- where new requests join an in-flight batch instead of waiting for it to finish -- vLLM routinely achieves 5-20x the throughput of naive request-serialized inference.

**TGI (Text Generation Inference)** from HuggingFace is the close alternative. Comparable architecture, similar throughput, slightly different ergonomics (first-class HuggingFace integration, optional Rust gRPC layer).

**TensorRT-LLM** (NVIDIA) is the compiler-first path: convert your model to a TensorRT engine, get maximum throughput on NVIDIA hardware, accept a long compile step and a locked-in architecture.

All three expose OpenAI-compatible HTTP APIs. All three require NVIDIA GPUs.

### Desktop & Embedded

**llama.cpp** runs the same [GGUF](../glossary.md#gguf) model on CUDA, Metal, or CPU. It is the most portable production-grade inference engine. On Apple Silicon it uses Metal (not MLX) -- this is important context for the comparison below.

**Ollama** wraps llama.cpp with a local HTTP API, model registry, and friendly CLI. On Apple Silicon, Ollama runs on Metal via llama.cpp. It is *not* an MLX-based stack, despite common confusion.

---

## The MLX Ecosystem

### Server-Class Inference

**`mlx_lm.server`** is the reference server. It ships with `mlx-lm`, exposes an OpenAI-compatible HTTP endpoint (chat completions, streaming, completions), and handles token streaming correctly. It does *not* yet implement paged attention or continuous batching -- for single-user or low-concurrency scenarios this is fine; for high-QPS serving, it is a bottleneck.

```bash
# Launch a server with any mlx-lm-compatible model
mlx_lm.server \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --host 0.0.0.0 \
  --port 8080

# Hit it with the OpenAI SDK
# client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
```

**`fastmlx`** (community) is an alternative OpenAI-compatible server built on top of mlx-lm. Adds some features mlx-lm.server lacks (concurrent request handling, model hot-swap). Track the repo for the current feature set.

### Multi-Device Inference

**exo** distributes inference across multiple Apple devices over LAN, slicing the model's layers across memory. Useful when a single Mac doesn't have enough RAM for the model you want to run -- two M2 Ultras can host a model neither could alone. Not a replacement for vLLM (throughput, not latency, is the target), but a unique capability the CUDA world lacks.

### Desktop GUI

**LM Studio** is a desktop app (macOS/Windows/Linux) that handles model download, [quantization](../glossary.md#quantization) selection, and an OpenAI-compatible local server. On Apple Silicon it uses MLX as a backend for supported models, falling back to llama.cpp/Metal otherwise. The closest thing to a "batteries-included" MLX serving story for non-developers.

### On-Device App Integration

**MLX-Swift** is the Swift binding for MLX. It exists so iOS and macOS apps can embed models directly, running inference on-device without a server. The primary use case is not "serving" in the HTTP-API sense -- it's in-process inference inside a shipping app. For developers with a CUDA server-side inference pipeline who want to run the same model client-side on Apple hardware, MLX-Swift is the bridge.

---

## Side-by-Side

| Use case | CUDA option | MLX option | Gap |
|----------|-------------|------------|-----|
| High-throughput multi-tenant API | vLLM, TGI | `mlx_lm.server` (no paged attention) | Significant: no MLX equivalent to paged attention yet |
| Low-QPS local API | Ollama, llama.cpp server | `mlx_lm.server`, `fastmlx`, LM Studio | Parity |
| Desktop app with GUI | LM Studio (CUDA/Metal) | LM Studio (MLX), Enchanted | Parity |
| iOS/macOS app integration | -- (no NVIDIA hardware on mobile/Mac) | MLX-Swift | MLX wins by default |
| Distributed across multiple local devices | -- (uncommon on CUDA outside datacenter) | exo | MLX wins by default |
| Maximum single-stream throughput | TensorRT-LLM | `mx.compile` + `mlx_lm.generate` | CUDA ahead on raw compile optimization |
| Speculative decoding | vLLM, TGI | Partial (mlx-lm has assist_model) | Gap: no production-grade speculative serving |

---

## What the Gap Actually Costs You

If you are porting a **single-user desktop app** from CUDA to MLX: almost no gap. LM Studio, `mlx_lm.server`, and MLX-Swift cover every reasonable scenario, often better than the CUDA equivalents (native on-device, unified memory advantages).

If you are porting a **production multi-tenant API** from vLLM to MLX: significant work. You will be serving fewer concurrent users per dollar on an M-series box than on an H100 with vLLM, because the batching and KV cache management that makes vLLM fast is not yet in the MLX stack. This is the largest open production gap in the MLX ecosystem as of early 2026.

If you are building **an on-device mobile feature** where a CUDA solution never existed: MLX-Swift is the obvious choice and has no CUDA competitor.

---

## Contributing Here

The serving gap is the single most contributor-available production gap in the MLX ecosystem. Concrete targets:

1. **Paged-attention-style KV cache management in mlx-lm.** The algorithm is well-documented (vLLM paper); the blocker is implementation effort, not research.
2. **Continuous batching in `mlx_lm.server`.** Combine with paged attention for real multi-tenant throughput.
3. **Speculative decoding serving.** mlx-lm has the primitives (`assist_model`); production serving integration is open.
4. **MLX-Swift high-level LLM APIs.** mlx-swift-examples has the building blocks; a reusable "drop-in chat API" would lower the barrier for app developers.

See [Open Opportunities](../03-contributing/03-open-opportunities.md) for how these rank against other gaps.

---

## Sources

- vLLM: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- TGI: [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)
- TensorRT-LLM: [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- llama.cpp: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- Ollama: [github.com/ollama/ollama](https://github.com/ollama/ollama)
- mlx-lm (includes `mlx_lm.server`): [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- fastmlx: [github.com/Blaizzy/fastmlx](https://github.com/Blaizzy/fastmlx)
- LM Studio: [lmstudio.ai](https://lmstudio.ai/)
- MLX-Swift: [github.com/ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift)
- exo: [github.com/exo-explore/exo](https://github.com/exo-explore/exo)
- vLLM paged attention paper: [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

---

## See Also

- [LLMs](03-llms.md) -- the broader LLM tooling landscape; mlx-lm internals; inference primitives this page builds on
- [Image Generation](04-image-generation.md) -- image model serving (mflux, diffusers); different shape from LLM serving
- [Audio](06-audio.md) -- real-time audio serving considerations
- [Tooling](08-tooling.md) -- profilers and converters that help tune a serving deployment
- [Porting Guide](../03-contributing/02-porting-guide.md) -- how to get a model working in MLX in the first place, before thinking about serving
