# Platform: CUDA vs MLX

> **Status:** 🟢 CUDA (Mature -- 15+ years, industry standard) | 🟡 MLX (Emerging -- December 2023)

## What This Domain Covers

The fundamental hardware and software platforms that machine learning runs on. Before comparing individual frameworks, libraries, or models, you need to understand the two underlying ecosystems: the NVIDIA/CUDA world that has dominated ML research and production since 2007, and the Apple/MLX world that became viable for serious ML work in December 2023.

This page answers: what is each platform made of, where does each one excel, what is genuinely missing, and what does that mean for someone porting CUDA-based generative AI work to Apple Silicon?

---

## The CUDA Ecosystem

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform. Released in 2007, it is the foundation on which essentially all of modern ML was built. When a paper says "trained on 8 A100s" or a model card says "requires CUDA 12.1+", they mean this ecosystem.

### Hardware

NVIDIA's GPU lineup spans consumer to data center:

```
NVIDIA GPU HARDWARE (2024)

  Consumer:
    RTX 4090   -- 24 GB GDDR6X VRAM, 82.6 TFLOPS (FP16)
                  The most capable consumer GPU. Primary target for
                  local research on CUDA. ~$1,600.

    RTX 4080   -- 16 GB VRAM. Good for 13B models at float16.
    RTX 3090   -- 24 GB VRAM. Previous generation, still widely used.

  Professional / Data Center:
    A100       -- 40 or 80 GB HBM2e VRAM, 312 TFLOPS (FP16)
                  Standard for ML research clusters. ~$10,000+.

    H100       -- 80 GB HBM3 VRAM, 989 TFLOPS (FP8)
                  Current data center standard. 3x A100 performance.
                  ~$30,000+.

    B200       -- 192 GB VRAM (announced). Next generation.

  Cloud instances (A100, H100):
    Google Cloud: A2 (A100), A3 (H100) series
    AWS: p4d (A100), p5 (H100) series
    Azure: NCv4 (A100), NDv5 (H100) series
```

**The key constraint on NVIDIA hardware: VRAM is a hard ceiling.** A model that does not fit in the GPU's VRAM cannot run on that GPU, period. You cannot use system RAM as an overflow pool at any useful speed (PCIe bandwidth is roughly 64 GB/s, whereas the H100's HBM3 bandwidth is 3.35 TB/s -- a 50x difference).

### Software Stack

The CUDA software ecosystem is deep and mature:

```
CUDA SOFTWARE STACK

  Hardware: NVIDIA GPU
      |
  CUDA Toolkit: nvcc compiler, CUDA runtime, PTX (virtual ISA)
      |
  Acceleration Libraries:
    cuDNN      -- deep neural network primitives (conv, attention, BN)
    cuBLAS     -- dense linear algebra (the matmul backbone)
    cuSPARSE   -- sparse matrix operations
    cuFFT      -- fast Fourier transforms
    NCCL       -- multi-GPU communication (collective ops: all-reduce, etc.)
    TensorRT   -- inference optimization and deployment
      |
  ML Frameworks:
    PyTorch (primary), JAX, TensorFlow, PaddlePaddle
      |
  Ecosystem Libraries:
    Hugging Face (Transformers, Diffusers, PEFT)
    DeepSpeed, FSDP, Megatron-LM   -- distributed training
    Flash Attention, xFormers       -- custom kernels
    vLLM, TGI                       -- serving infrastructure
    Triton                          -- Python-based kernel writing
```

Everything in ML research traces back to CUDA. Papers publish CUDA code. Benchmarks run on CUDA. Pre-trained models are trained on CUDA. When a model card says "BF16 recommended", it means BF16 on A100/H100 hardware. The ecosystem has 15 years of accumulated tooling, documentation, and institutional knowledge.

### Community and Documentation

- **Developers:** Millions. Every major AI lab, research university, and enterprise ML team works primarily in CUDA.
- **Documentation:** Exhaustive. Stack Overflow answers. GitHub issues resolved. CUDA courses on Coursera, MIT OpenCourseWare, Fast.ai.
- **Papers:** Virtually all ML research papers include CUDA as the compute environment.
- **Pre-trained models:** Hugging Face hosts tens of thousands of models, almost all trained on CUDA hardware and tested on CUDA hardware.

---

## The MLX Ecosystem

MLX is Apple's open-source array framework for machine learning on Apple Silicon. Released December 5, 2023. "An array framework by ML researchers, for ML researchers" -- the guiding design principle is that it should feel like NumPy and allow rapid research iteration, while running efficiently on Apple's unified memory architecture.

### Hardware

Apple Silicon is a family of chips where CPU, GPU, Neural Engine, and memory are on a single package:

```
APPLE SILICON HARDWARE (2024)

  MacBook Pro / MacBook Air:
    M3 (base)        -- 10-core GPU, 8-24 GB unified memory
    M3 Pro           -- 18-core GPU, 18-36 GB unified memory
    M3 Max           -- 40-core GPU, 36-128 GB unified memory
                        400 GB/s memory bandwidth

    M4 (base)        -- 10-core GPU, 16-32 GB unified memory
    M4 Pro           -- 20-core GPU, 24-64 GB unified memory
    M4 Max           -- 40-core GPU, 48-128 GB unified memory

  Mac Studio / Mac Pro:
    M2 Ultra         -- 76-core GPU, up to 192 GB unified memory
    M3 Ultra         -- 80-core GPU, up to 192 GB unified memory
    M4 Ultra         -- (announced) up to 192 GB unified memory

  Key: there is no VRAM. CPU and GPU share one memory pool.
  The 192 GB max is 8x larger than the largest NVIDIA consumer GPU.
```

**The structural advantage: unified memory.** When you load a model in MLX, the weights go into unified memory. The GPU reads them directly. The CPU reads them directly. No copy across a PCIe bus. No VRAM limit distinct from system RAM. This is not a minor optimization -- it changes what is possible.

```
MEMORY COMPARISON

  NVIDIA RTX 4090:
    System RAM: 64-256 GB (separate from GPU)
    VRAM: 24 GB (hard limit for GPU workloads)
    Transfer: PCIe 4.0, ~64 GB/s (bottleneck)

  Apple M3 Max (128 GB config):
    Unified memory: 128 GB (used by both CPU and GPU)
    GPU bandwidth: 400 GB/s (on-package)
    Transfer cost: zero (no copy needed)

  Practical consequence:
    A 70B model at 4-bit requires ~35 GB.
    RTX 4090: does not fit in 24 GB VRAM. Cannot run.
    M3 Max (128 GB): fits. Runs at ~3-5 tokens/sec.
```

### The MLX Software Stack

MLX is younger but growing rapidly:

```
MLX SOFTWARE STACK

  Hardware: Apple Silicon (M-series)
      |
  Metal: Apple's GPU API and shading language
      |
  MLX Core (mlx):
    Array operations (NumPy-compatible API)
    Lazy evaluation + automatic kernel fusion
    Autograd (value_and_grad, grad)
    Metal kernel integration
    JIT compilation
      |
  MLX Neural Networks:
    mlx.nn: layers (Linear, Conv, Attention, etc.)
    mlx.optimizers: Adam, SGD, etc.
      |
  Ecosystem Libraries:
    mlx-lm          -- LLM inference and fine-tuning (LoRA)
    mlx-vlm         -- vision-language models
    mlx-audio       -- audio generation (e.g., Kokoro, Parler TTS)
    mlx-community   -- Hugging Face org with 1000+ converted models
    mlx-forge       -- tools for porting and kernel development
    mlx-examples    -- reference implementations (LLaMA, Stable Diffusion, etc.)
```

MLX's API is deliberately NumPy-like. If you know PyTorch, the learning curve is shallow:

```python
import mlx.core as mx
import mlx.nn as nn

# Feels like NumPy / PyTorch
x = mx.random.normal((batch, seq_len, d_model))
w = mx.random.normal((d_model, d_model))

# Operations are lazy -- nothing runs until forced to evaluate
y = x @ w
y = mx.maximum(y, 0)   # ReLU -- fused with matmul automatically
mx.eval(y)             # GPU executes here

# Autograd works like JAX
loss_fn = lambda params, x: nn.losses.cross_entropy(model(x, params), targets)
loss, grads = mx.value_and_grad(loss_fn)(params, x)
```

### Key Design Choices in MLX

**Lazy evaluation and kernel fusion.** Operations do not execute when called. MLX builds a graph of pending operations and executes them when `mx.eval()` is called. During that execution, MLX fuses compatible operations -- an add followed by a ReLU followed by a scale becomes one GPU kernel instead of three. This reduces memory bandwidth pressure automatically, without the user writing custom kernels.

**Unified memory as a first-class citizen.** MLX was designed from the start for an environment with no VRAM/RAM split. There is no `.to(device)` API, because there is only one device. Arrays live in unified memory and are accessible by both CPU and GPU operations without any copy.

**Python-first with functional style.** MLX's autograd follows JAX's functional approach (`value_and_grad` rather than `.backward()`). Models are Python objects, not a special graph. The design prioritizes researcher ergonomics over production deployment features.

**Open source and fast-moving.** MLX launched in December 2023 and has seen rapid development since. The pace of new features, contributed models, and community growth has been fast by any standard.

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **Hardware availability** | 🟢 Cloud, consumer, data center | 🟡 Mac only (no cloud, no Linux) | Large -- no cloud MLX option; must own Apple Silicon hardware |
| **Multi-GPU / distributed training** | 🟢 NCCL, FSDP, Megatron-LM, full ecosystem | 🔴 No multi-GPU support | Critical gap for large model training |
| **Community size** | 🟢 Millions of developers, 15+ years of content | 🟡 Growing, thousands of active contributors | Orders of magnitude smaller; fewer examples, answers, tutorials |
| **Documentation** | 🟢 Exhaustive at every level | 🟡 Good core docs; thin on advanced topics | Noticeable gap in advanced kernel writing, custom ops |
| **Pre-trained models** | 🟢 Tens of thousands on Hugging Face | 🟡 1000+ on mlx-community; growing | CUDA models run as-is; MLX requires conversion |
| **Custom kernel writing** | 🟢 CUDA C++, Triton (Python), extensive tooling | 🟡 Metal Shading Language; fewer examples | Steep learning curve for custom Metal kernels; no Triton equivalent |
| **Inference performance (LLMs)** | 🟢 Fastest per-token on H100/A100 | 🟡 Competitive on Apple Silicon for its class | H100 is faster; Apple Silicon wins on memory capacity per dollar |
| **Training support** | 🟢 Full: large batch, gradient checkpointing, mixed precision, distributed | 🟡 Basic: works for fine-tuning; limited for large-scale pretraining | MLX LoRA fine-tuning works well; pretraining at scale is difficult |
| **Serving / production inference** | 🟢 vLLM, TGI, Triton Inference Server, TensorRT | 🟡 mlx-lm server mode, llama.cpp (GGUF) | No equivalent to vLLM's paged attention for production serving |
| **Profiling and debugging tools** | 🟢 Nsight, nvprof, PyTorch profiler, rich ecosystem | 🟡 Instruments (Apple), `mx.metal.get_peak_memory()` | Instruments is powerful but less ML-specific than Nsight |
| **Quantization support** | 🟢 BitsAndBytes, GPTQ, AWQ, TensorRT INT8 | 🟡 4-bit and 8-bit via `nn.QuantizedLinear`; `mlx_lm.convert` | MLX covers the inference use case well; training-time quantization thin |
| **Video generation (e.g., LTX-Video)** | 🟢 Reference implementations, CUDA-optimized kernels | 🟡 Early ports (Matrix-Game-mlx, partial LTX ports) | Active area; custom temporal attention kernels not yet optimized |
| **Windows / Linux support** | 🟢 Full | 🔴 None (Apple Silicon = macOS only) | Hard constraint; MLX does not run on non-Apple hardware |

---

## Contribution Opportunities

The gap analysis above identifies where MLX needs work. If you are porting CUDA to MLX on projects like LTX-Video, Matrix-Game-mlx, or mlx-forge, these are the areas where your work has the highest leverage:

**Custom video generation kernels.** Temporal attention, 3D convolutions, causal masking for video -- none of these have well-optimized Metal implementations in the public MLX ecosystem. Porting LTX-Video's custom CUDA kernels to Metal and contributing them to mlx-forge would close a real gap.

**Benchmark and profiling tooling.** MLX has `mx.metal.get_peak_memory()` and Instruments, but no equivalent to PyTorch's `torch.profiler` or NVIDIA's Nsight Systems for kernel-level ML profiling. A profiling utility built specifically for MLX workloads would be valuable.

**Model conversion pipelines.** mlx-community has 1000+ models, but not all model architectures are covered. Systematic conversion of missing architectures (especially video models) with validation tooling would help.

**Documentation: advanced topics.** Core MLX operations are documented. Writing custom Metal kernels with MLX's C++ extension API is not. Contributions here compound -- good docs help every future porter.

**Multi-chip exploration.** Apple Silicon does not support NCCL-style multi-GPU, but some workloads could theoretically distribute across CPU and GPU within the same chip, or use the Neural Engine for specific operations. This is genuinely unexplored territory.

---

## Sources

- MLX GitHub repository: [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx). Primary source for source code, release notes, and design decisions.
- MLX documentation: [ml-explore.github.io/mlx/](https://ml-explore.github.io/mlx/). API reference, guides, and examples.
- NVIDIA CUDA Toolkit: [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). The starting point for CUDA development.
- NVIDIA GPU Specs: [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). Official hardware specifications.
- Apple Silicon memory bandwidth: [apple.com/mac/compare/](https://www.apple.com/mac/compare/). Official hardware comparison pages.
- mlx-community on Hugging Face: [huggingface.co/mlx-community](https://huggingface.co/mlx-community). Pre-converted MLX models.

---

## See Also

- [GPU Computing](../01-foundations/04-gpu-computing.md) -- the hardware-level explanation of why CUDA and Apple Silicon work the way they do; VRAM vs. unified memory, memory bandwidth, kernel execution model
- [Quantization](../01-foundations/10-quantization.md) -- the technique that makes large models runnable on Apple Silicon; essential reading before working with 4-bit MLX models
- [Frameworks](02-frameworks.md) -- compares PyTorch, JAX, and MLX as programming models, independent of the underlying hardware platform
- [LLMs](03-llms.md) -- where MLX's ecosystem is strongest; mlx-lm covers inference and LoRA fine-tuning comprehensively
- [Video Generation](05-video-generation.md) -- the frontier where LTX-Video and Matrix-Game-mlx live; where the CUDA/MLX gap is most visible
