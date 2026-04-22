# LLMs: CUDA vs MLX

> **Status:** 🟢 [CUDA](../glossary.md#cuda) (Mature -- industry standard, massive tooling ecosystem) | 🟡 MLX (Strong -- inference and fine-tuning well covered, serving gaps remain)

## What This Domain Covers

Running, serving, and fine-tuning large language models. This includes every step from loading pre-trained weights to generating tokens at inference time, through serving many concurrent users in production, to adapting a base model to a new task via fine-tuning.

LLMs are where the CUDA/MLX gap is smallest. Apple Silicon's unified memory architecture is a genuine advantage here -- a Mac Studio with 192 GB can run models that simply do not fit on any consumer NVIDIA [GPU](../glossary.md#gpu). mlx-lm, the primary MLX library for language models, covers the most common architectures and workflows with a stable API. For a developer porting CUDA inference or fine-tuning code to MLX, the LLM domain is the most mature translation target in the MLX ecosystem.

This page answers: what does each ecosystem offer for LLM inference, serving, and fine-tuning? Where does the gap matter for local development versus production? What is missing and what can be contributed?

---

## The CUDA Ecosystem

### Inference Libraries

CUDA's LLM inference ecosystem has stratified into layers based on use case: local/developer use, high-throughput API serving, and maximum-performance compiled inference.

**llama.cpp** is the foundational local inference engine. Written in C++ with minimal dependencies, it runs on [CPU](../glossary.md#cpu), CUDA, and [Metal](../glossary.md#metal) (Apple GPU). Models are stored in [GGUF](../glossary.md#gguf) format, which packages weights and metadata in a single file. llama.cpp pioneered k-bit quantization (Q4_K_M, Q6_K, and similar formats) as a practical way to run 7B-70B models on consumer hardware.

```
LLAMA.CPP ARCHITECTURE

  GGUF model file  --[load]-->  llama.cpp engine
                                    |
                            Backend selection:
                              CUDA (NVIDIA)
                              Metal (Apple GPU)  <-- shared with MLX world
                              CPU (fallback)
                                    |
                            Sampling strategies:
                              temperature, top-p, top-k, repetition
                                    |
                            Output: token stream
```

**vLLM** is the production-grade serving engine. Designed for high-throughput multi-user inference, it introduced **paged attention** -- a key-value cache management technique that dramatically increases GPU memory utilization and concurrent user capacity.

```
PAGED ATTENTION (vLLM)

  Problem: KV cache for long sequences is large and variable-length.
           Naive approach: pre-allocate maximum possible length.
           Waste: most sequences are much shorter than the maximum.

  Paged attention:
    - KV cache divided into fixed-size "pages" (blocks)
    - Pages allocated on demand as sequences grow
    - Non-contiguous pages stitched together by the attention kernel
    - Multiple sequences share a page pool
    - Freed pages immediately returned to pool when a sequence ends

  Result: 2-4x higher throughput vs naive KV cache management
          More concurrent users per GPU
          Essential for production LLM serving

  CUDA implementation: custom CUDA kernels for block attention
  MLX equivalent: not implemented
```

**TGI (Text Generation [Inference](../glossary.md#inference))** is Hugging Face's serving framework. Supports continuous batching (rolling batch of requests rather than fixed-size batch), tensor parallelism across multiple GPUs, and quantized inference ([GPTQ](../glossary.md#gptq), AWQ). Used internally by Hugging Face's Inference API.

**TensorRT-LLM** is NVIDIA's production inference framework. It compiles models into optimized TensorRT engines at deployment time. Supports speculative decoding, inflight batching, and [INT8](../glossary.md#int8)/FP8 quantization with calibration. Achieves the highest throughput of any CUDA inference framework, at the cost of requiring a compilation step and being NVIDIA-specific.

**Ollama** wraps llama.cpp with a simple REST API, model management, and automatic backend detection (CUDA, Metal, or CPU). It is the most accessible entry point for local model serving. Because it uses llama.cpp under the hood, it runs on Apple Silicon via Metal without any MLX dependency.

### Serving Frameworks

```
CUDA SERVING STACK

  vLLM:
    Paged attention: KV cache as virtual memory
    Continuous batching: new requests fill gaps as sequences end
    Multi-GPU tensor parallelism: split model across GPUs
    OpenAI-compatible API: drop-in replacement for applications
    Throughput: thousands of tokens/sec on H100

  TGI (Text Generation Inference):
    Continuous batching + flash attention
    Tensor parallelism (multi-GPU)
    GPTQ, AWQ, EETQ quantization
    Used in Hugging Face Inference API

  Triton Inference Server (NVIDIA):
    Framework-agnostic (PyTorch, TensorRT, ONNX)
    Dynamic batching
    Model ensembles (chain multiple models)
    gRPC + HTTP endpoints
    Designed for enterprise deployment
```

### Fine-tuning

CUDA's fine-tuning ecosystem is built around Hugging Face:

**PEFT ([Parameter](../glossary.md#parameter)-Efficient Fine-Tuning)** implements LoRA, QLoRA, IA3, and other adapter methods. LoRA adds low-rank adapter matrices to frozen base model weights. QLoRA combines 4-bit quantized base weights with LoRA adapters, enabling fine-tuning of 65B+ models on a single 80 GB A100.

```
QLORA TRAINING (CUDA)

  Base model:    65B parameters, quantized to 4-bit NF4
                 Memory: ~33 GB (fits in A100 80GB with optimizer states)

  LoRA adapters: rank r=64, alpha=16
                 Memory: ~120 MB additional
                 Trainable: <0.1% of total parameters

  Combined: 65B model fine-tunable on one A100 80GB GPU
            Without QLoRA: would require 4x A100 80GB in FP16

  Training speed: ~1000 tokens/sec on A100 for 7B model
```

**Axolotl** is a training pipeline built on PEFT that handles data formatting, training loops, evaluation, and checkpoint management. Widely used for instruction tuning and domain-specific fine-tuning.

**Unsloth** achieves faster LoRA training via hand-written Triton kernels that fuse operations in the LoRA forward and backward pass. Reports 2x speed improvement over vanilla PEFT with the same GPU memory.

---

## The MLX Ecosystem

### mlx-lm

mlx-lm is the primary MLX library for language models. Part of the mlx-examples repository, maintained by Apple's MLX team. Covers the full LLM workflow: model loading, generation, quantization, fine-tuning, and serving.

```
MLX-LM CAPABILITIES (as of Q1 2025)

  Supported architectures (selection):
    LLaMA / LLaMA 2 / LLaMA 3 / LLaMA 3.1 / LLaMA 3.2
    Mistral / Mixtral (MoE)
    Qwen 2 / Qwen 2.5 / Qwen 2.5-VL
    Gemma / Gemma 2
    Phi-2 / Phi-3 / Phi-3.5
    DeepSeek-R1 / DeepSeek-V3
    Falcon
    MPT
    StarCoder / StarCoder 2
    DBRX
    OLMo
    Cohere Command R

  Quantization:
    4-bit (default): nn.QuantizedLinear, 4 bits per weight
    8-bit: 8-bit quantization via mlx_lm.convert --q-bits 8
    Group quantization: configurable group size (default 64)

  Fine-tuning:
    LoRA: full LoRA adapter training
    QLoRA: quantized base + LoRA adapters
    DoRA: weight decomposition LoRA (experimental)

  Serving:
    mlx_lm.server: OpenAI-compatible HTTP API
    Single-user optimized (no paged attention)
```

**Loading and running a model:**

```python
from mlx_lm import load, generate

# Load a model from Hugging Face or local path
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Generate tokens
messages = [{"role": "user", "content": "Explain attention mechanisms in two sentences."}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
response = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=True)
```

**Quantizing and converting a model:**

```python
# Convert from Hugging Face format to MLX + quantize
# mlx_lm.convert --hf-path meta-llama/Llama-3.2-3B-Instruct \
#                --mlx-path ./llama-3.2-3b-instruct-4bit \
#                --q-bits 4

# Or from Python:
from mlx_lm import convert
convert(
    hf_path="meta-llama/Llama-3.2-3B-Instruct",
    mlx_path="./llama-3.2-3b-instruct-4bit",
    quantize=True,
    q_bits=4
)
```

**LoRA fine-tuning:**

```python
# Fine-tune with LoRA via command line
# mlx_lm.lora \
#   --model mlx-community/Llama-3.2-3B-Instruct-4bit \
#   --train \
#   --data ./data \
#   --batch-size 4 \
#   --num-layers 16 \
#   --iters 1000 \
#   --learning-rate 1e-4

# Or load adapters for inference
from mlx_lm import load, generate
model, tokenizer = load(
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    adapter_path="./adapters"
)
```

### Memory Advantage in Practice

The unified memory architecture changes which models are even runnable:

```
MODEL SIZE vs HARDWARE (4-bit quantization, approximate)

  Model      FP16 size    4-bit size    RTX 4090 (24GB VRAM)    M3 Max (128GB)
  --------   ---------    ----------    --------------------    --------------
  7B         14 GB        3.5 GB        YES                     YES
  13B        26 GB        6.5 GB        NO (exceeds 24GB)       YES
  34B        68 GB        17 GB         NO                      YES
  70B        140 GB       35 GB         NO                      YES (~3 tok/s)
  405B       810 GB       200 GB        NO                      NO (too large)

  M2 Ultra / M3 Ultra (192 GB):
  70B at 4-bit:  35 GB   YES  (~5-8 tok/s)
  405B at 4-bit: 200 GB  YES  (~1-2 tok/s, barely)

  Practical conclusion:
    For models up to ~70B at 4-bit, MLX on Apple Silicon
    is the only consumer-grade option.
    For models above 70B, multi-GPU CUDA is required for
    reasonable throughput; MLX handles it but slowly.
```

### mlx-vlm

mlx-vlm extends mlx-lm to vision-language models -- models that accept both images and text as input.

```
MLX-VLM SUPPORTED ARCHITECTURES (selection, Q1 2025)

  LLaVA / LLaVA-1.5 / LLaVA-1.6
  Qwen2-VL / Qwen2.5-VL
  LLaMA 3.2 Vision (11B, 90B)
  InternVL2
  Phi-3 Vision / Phi-3.5 Vision
  Idefics3
  PaliGemma / PaliGemma 2
  Pixtral
  MiniCPM-V
```

Usage is similar to mlx-lm:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/Qwen2-VL-7B-Instruct-4bit")

# Image + text prompt
prompt = apply_chat_template(processor, model.config, "Describe this image.", num_images=1)
output = generate(model, processor, "path/to/image.jpg", prompt, max_tokens=200)
```

### MLX Community on Hugging Face

The mlx-community organization on Hugging Face hosts thousands of converted models. The workflow is:
1. Base model released on Hugging Face in PyTorch/safetensors format
2. Community member (or automated pipeline) converts using `mlx_lm.convert`
3. Converted model uploaded to mlx-community with standardized naming (e.g., `mlx-community/Llama-3.2-3B-Instruct-4bit`)

This conversion pipeline is well-established. New models typically appear on mlx-community within days of the base model release for major architectures (LLaMA, Mistral, Qwen, Gemma, Phi). Minor or research architectures may take weeks or never appear.

### Ollama and llama.cpp

Ollama is worth noting separately: it runs on macOS natively using llama.cpp's Metal backend, and is independent of MLX. For many use cases (simple local inference, model management), Ollama on Apple Silicon is a viable and simpler alternative to mlx-lm -- no Python setup required.

The practical difference:
- **Ollama**: simpler, uses GGUF format, llama.cpp kernels, works on any Mac
- **mlx-lm**: Python API, MLX format, Apple Silicon optimized kernels, better for Python integration and fine-tuning

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **Tokens per second (7B model)** | 🟢 H100: ~2000 tok/s (vLLM batched); RTX 4090: ~100 tok/s single-user | 🟡 M3 Max: ~60-80 tok/s single-user; M2 Ultra: ~90 tok/s | Medium for single-user; large for batched serving. Per-token latency on Apple Silicon is comparable to RTX 4090; batched throughput is not. |
| **[Model](../glossary.md#model) coverage** | 🟢 All architectures (Transformers supports everything) | 🟡 Major architectures supported; new and niche models lag by days to weeks | Small for mainstream models; noticeable for new research models |
| **Context length** | 🟢 Ring attention, sequence parallelism for very long contexts; [Flash Attention](../glossary.md#flash-attention) 2 handles 100k+ tokens efficiently | 🟡 Long contexts work but are memory-bound; no ring attention equivalent | Medium -- 32k context is fine; 128k+ contexts can be slow or OOM on lower-memory configs |
| **Paged attention / production serving** | 🟢 vLLM paged attention: serves thousands of concurrent users on one H100 | 🔴 Not implemented; mlx_lm.server is single-user | Large -- MLX cannot serve concurrent users at scale; this is a fundamental gap for production API deployments |
| **Speculative decoding** | 🟢 vLLM, TGI both support speculative decoding with draft models; 2-3x latency reduction on long outputs | 🟡 Experimental in mlx-lm; not stable | Medium -- reduces latency for long-generation tasks; useful for chatbots and code generation |
| **[Tensor](../glossary.md#tensor) parallelism (multi-device)** | 🟢 vLLM, TGI, TensorRT-LLM all support multi-GPU sharding | 🔴 Not supported -- MLX is single-device | Large for models > 70B at FP16; less critical with quantization |
| **[Quantization](../glossary.md#quantization) format compatibility** | 🟢 GPTQ, AWQ, GGUF, BitsAndBytes, TensorRT INT8/FP8 -- all interoperable with Transformers | 🟡 MLX 4-bit and 8-bit formats; GGUF via llama.cpp separately; not directly interchangeable | Medium -- MLX quantization works well; cannot load GPTQ/AWQ models directly |
| **Fine-tuning: LoRA** | 🟢 PEFT: full LoRA, QLoRA, DoRA, IA3; Axolotl pipelines; Unsloth optimized kernels | 🟡 mlx-lm LoRA and QLoRA: works well for instruction tuning; fewer optimizations | Small for basic fine-tuning; medium for large-scale or optimized training |
| **Fine-tuning: full parameter** | 🟢 DeepSpeed ZeRO, FSDP, gradient checkpointing; 7B full fine-tuning on 4x A100 | 🟡 Possible for small models; no gradient checkpointing; no distributed | Large -- full fine-tuning at scale is not practical in MLX |
| **Continuous batching** | 🟢 vLLM, TGI, TensorRT-LLM all implement continuous batching | 🔴 Not implemented in mlx-lm server | Large for production; irrelevant for personal/single-user use |
| **Flash Attention** | 🟢 Flash Attention 2 kernel: optimal memory access pattern for attention, 8-16x memory reduction, ~2x speed | 🟡 MLX attention is memory-efficient via lazy eval + fusion, but not a direct port of FA2's tiling strategy | Medium -- MLX attention is good; extreme long-context edge cases are slower |
| **Structured output / constrained generation** | 🟢 Outlines, guidance, LMQL -- schema-constrained generation via logit processing | 🟡 Basic logit processing in mlx-lm; no Outlines equivalent | Small -- manual logit processing is possible; no plug-in library |
| **Model evaluation frameworks** | 🟢 lm-evaluation-harness, HELM, EleutherAI benchmarks -- CUDA native | 🟡 lm-evaluation-harness has partial MLX support; not all benchmarks run | Small -- major benchmarks runnable; some harness integrations missing |
| **[Embedding](../glossary.md#embedding) models** | 🟢 sentence-transformers, all architectures; fast batch embedding with CUDA | 🟡 BERT-class models work; no sentence-transformers MLX integration | Small -- embeddings work; less convenient tooling |
| **MoE (Mixture of Experts) models** | 🟢 Mixtral, DeepSeek-MoE via Transformers; expert routing optimized in vLLM | 🟡 Mixtral and DeepSeek MoE supported in mlx-lm; expert routing not specially optimized | Medium -- supported but expert routing on Apple Silicon is not as tuned |

---

## Performance Deep Dive

Single-user inference performance is the metric most relevant to local development. Here is a realistic comparison for a developer running a 7B model:

```
7B MODEL INFERENCE PERFORMANCE (single user, 4-bit quantization, approximate)

  Hardware              Speed           Notes
  --------              -----           -----
  NVIDIA H100 (80GB)    ~800 tok/s      Cloud, ~$3/hr; overkill for 7B
  NVIDIA RTX 4090       ~90-120 tok/s   Consumer, CUDA; very fast
  Apple M3 Max (128GB)  ~60-80 tok/s    Local, MLX; close to RTX 4090
  Apple M4 Max (128GB)  ~80-100 tok/s   Latest, MLX; matches RTX 4090 range
  Apple M2 Ultra        ~80-100 tok/s   Mac Studio; competitive
  Apple M4 Ultra        ~120+ tok/s     (announced/estimated)

  70B MODEL (4-bit):
  NVIDIA RTX 4090:      Does not fit (24GB VRAM, 35GB needed)
  Apple M3 Max (128GB): ~3-5 tok/s    Slow but unique: no CUDA card can do this
  Apple M2 Ultra:       ~5-8 tok/s    Better memory bandwidth
```

The key takeaway: **for 7B models, Apple Silicon is within 30-50% of an RTX 4090 in single-user throughput. For 70B models, Apple Silicon is the only consumer option.** For multi-user API serving where batching matters, CUDA wins by a large margin.

---

## Contribution Opportunities

**Paged attention for mlx-lm.** The single most impactful contribution to the MLX LLM ecosystem. Implementing paged attention (or an equivalent block-based KV cache) in MLX would enable mlx-lm.server to handle concurrent requests without out-of-memory failures. This requires writing custom Metal kernels for block-attention and a Python-side scheduler, but the algorithmic design is well-documented in the vLLM paper.

**Speculative decoding stabilization.** Speculative decoding exists in mlx-lm as experimental code. Stabilizing it, adding support for more draft model configurations, and benchmarking it against vanilla generation would give MLX users a meaningful latency improvement for long-output tasks.

**Outlines-style constrained generation.** Schema-constrained generation (JSON output, regex-constrained output) is useful for tool use and structured data extraction. A lightweight logit-masking library compatible with mlx-lm's generate loop would fill this gap without requiring a port of the full Outlines library.

**Flash Attention 2 Metal kernel.** MLX's current attention implementation is efficient via kernel fusion, but does not implement FA2's blocked tiling strategy. For very long contexts (128k+), FA2-style tiling would significantly reduce peak memory and improve throughput. This requires Metal Shading Language kernel work.

**Model conversion automation.** New model architectures appear frequently. Building a CI pipeline that automatically converts new popular models (detected via Hugging Face trending) and uploads to mlx-community would keep the model library current without manual effort.

**lm-evaluation-harness full integration.** Completing mlx-lm support in the EleutherAI evaluation harness would enable reproducible benchmark comparisons across MLX model variants, which is currently difficult.

---

## Sources

- mlx-lm and mlx-examples: [github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples). The canonical source for mlx-lm implementation, supported architectures, and fine-tuning code.
- mlx-vlm: [github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Vision-language model support for MLX.
- vLLM: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm). Paged attention paper and implementation.
- TGI (Text Generation Inference): [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference). Hugging Face serving framework.
- llama.cpp: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp). GGUF format and Metal backend.
- Hugging Face PEFT: [github.com/huggingface/peft](https://github.com/huggingface/peft). LoRA, QLoRA, and other fine-tuning methods.
- Axolotl: [github.com/axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl). Fine-tuning pipeline.
- Unsloth: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth). Optimized LoRA training kernels.
- MLX Community on Hugging Face: [huggingface.co/mlx-community](https://huggingface.co/mlx-community). Pre-converted MLX models.
- QLoRA paper (Dettmers et al., 2023): The foundational paper for quantized fine-tuning; explains NF4 quantization and double quantization.
- vLLM paper (Kwon et al., 2023): "Efficient Memory Management for Large Language Model Serving with PagedAttention." Explains paged attention design.

---

## See Also

- [Transformers](../01-foundations/05-transformers.md) -- the architecture underlying every LLM on this page; attention mechanisms, positional encodings, KV cache
- [Attention](../01-foundations/06-attention.md) -- deep dive on the attention operation; explains why Flash Attention matters and why long contexts are expensive
- [Quantization](../01-foundations/10-quantization.md) -- essential for understanding why 4-bit models work, what NF4 is, and where quantization error accumulates
- [Training vs. Inference](../01-foundations/03-training-vs-inference.md) -- the compute and memory profiles are different; explains why fine-tuning gaps are larger than inference gaps
- [Frameworks](02-frameworks.md) -- PyTorch/MLX API comparison; the translation layer between CUDA ecosystem libraries and mlx-lm
- [Image Generation](04-image-generation.md) -- the adjacent domain where MLX ecosystem coverage is thinner
