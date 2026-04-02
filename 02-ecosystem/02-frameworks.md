# Frameworks: PyTorch vs JAX vs MLX

> **Status:** 🟢 PyTorch (Mature -- industry standard since 2016) | 🟢 JAX (Mature -- Google, TPU-native) | 🟡 MLX (Growing -- Apple, December 2023)

## What This Domain Covers

The programming frameworks that sit between your Python code and the GPU hardware. These are not competing for the same users in the same market -- PyTorch and JAX run on CUDA (and other backends), while MLX runs only on Apple Silicon. But if you are porting a CUDA-based PyTorch model to MLX, you need to understand the design decisions behind each one: why the APIs differ, where they are directly analogous, and where porting requires genuine rethinking.

This page compares PyTorch, JAX, and MLX across API design, ecosystem, performance model, hardware support, and community. By the end, you will understand not just what each framework does, but why it makes the choices it does.

---

## PyTorch

PyTorch is the dominant framework for ML research and, increasingly, production. Originally developed at Facebook AI Research (now Meta), released in 2016. Built on the idea that a neural network framework should be a Python library, not a separate compiled graph language.

### Design Philosophy

PyTorch's central bet was on **eager execution**: operations run immediately when called, like normal Python code. When you write `y = x @ w`, the matrix multiplication happens on that line. You can print intermediate values, set breakpoints, use standard Python debuggers -- the computation is transparent and Python-native.

This was a deliberate contrast with TensorFlow 1.x (the dominant framework at the time), which required you to build a computational graph first, then execute it in a separate session. PyTorch's eager mode made research iteration dramatically faster.

```python
import torch
import torch.nn as nn

# Eager execution: this runs immediately
x = torch.randn(32, 768)
w = torch.randn(768, 3072)
y = x @ w           # runs NOW
y = torch.relu(y)   # runs NOW

# Standard Python debugging works
print(y.shape)      # torch.Size([32, 3072])
print(y.mean())     # prints the actual mean value

# Autograd: track gradients with requires_grad
x = torch.randn(32, 768, requires_grad=True)
loss = (x @ w).sum()
loss.backward()     # computes gradients
print(x.grad)       # gradient is populated
```

### The Performance Evolution: torch.compile

Eager execution has a cost: each operation dispatches to a separate kernel. An operation like `relu(layernorm(x @ w + b))` launches four separate GPU kernels instead of one fused kernel. Each kernel launch has overhead; each intermediate tensor writes to and reads from global memory.

PyTorch's answer to this is `torch.compile`, introduced in PyTorch 2.0 (March 2023):

```python
# torch.compile: JIT compile a model for better performance
model = MyTransformerModel()

# Wrap the model (or any function) -- compiles on first call
compiled_model = torch.compile(model)

# First forward pass: triggers compilation (slow)
out = compiled_model(x)   # compiles here, ~5-30s delay

# Subsequent calls: uses compiled, fused kernels (fast)
out = compiled_model(x)   # runs fused compiled code
```

`torch.compile` uses TorchDynamo (Python bytecode tracing) and TorchInductor (code generation) to analyze the computation graph and generate fused CUDA kernels. In practice it achieves 1.5-3x speedups on transformer models, approaching what a hand-fused kernel would achieve.

For custom CUDA kernels that are not in PyTorch's standard set, the path is writing them in C++/CUDA directly (a `.cu` file, compiled as a Python extension) or using **Triton**, which lets you write GPU kernels in Python with NumPy-like syntax:

```python
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)
```

Triton has become the standard way to write custom GPU kernels in the CUDA world. There is no MLX equivalent.

### Hardware Support

PyTorch's primary backend is CUDA. Other backends exist but vary in maturity:

```
PYTORCH HARDWARE BACKENDS

  CUDA (NVIDIA):     🟢 Full support. Default. Every feature works.
  ROCm (AMD GPU):    🟡 Community support. Most features work.
                        Some custom ops missing.
  MPS (Apple GPU):   🟡 Available since PyTorch 1.12. Covers
                        standard operations. Custom CUDA kernels
                        (like Flash Attention) do not run on MPS.
                        Performance is generally worse than MLX
                        on Apple Silicon.
  CPU:               🟢 Full support. Slow for large models.
  XPU (Intel GPU):   🟡 Growing. Experimental.
```

PyTorch's MPS backend runs on Apple Silicon, but it is a second-class citizen. Operations that rely on hand-written CUDA kernels (Flash Attention, BitsAndBytes quantization, many custom diffusion model ops) simply do not have MPS implementations. Models that use them will fail or silently fall back to CPU. This is one of the main motivations for MLX as a separate framework: MLX is designed for Apple Silicon from the ground up, rather than treating it as an afterthought.

### Ecosystem

PyTorch's ecosystem is the largest in ML:

```
PYTORCH ECOSYSTEM (selected)

  Model hubs:
    Hugging Face Transformers   -- LLMs, vision-language, encoders
    Hugging Face Diffusers      -- Stable Diffusion, FLUX, LTX-Video
    timm                        -- vision model zoo
    torchaudio, torchvision     -- official media libraries

  Training:
    Lightning / PyTorch Lightning  -- training loop abstraction
    DeepSpeed                      -- ZeRO optimizer, large-scale
    FSDP (FullyShardedDataParallel) -- official distributed training
    Accelerate (Hugging Face)      -- multi-device training

  Fine-tuning:
    PEFT (Hugging Face)            -- LoRA, QLoRA, adapters
    Axolotl                        -- training pipeline for LLMs
    Unsloth                        -- efficient LoRA training

  Inference:
    vLLM                           -- high-throughput LLM serving
    TGI (Text Generation Inference) -- Hugging Face serving
    TensorRT-LLM                   -- NVIDIA production inference
    llama.cpp                      -- CPU + MPS backend (GGUF)

  Custom kernels:
    Triton                         -- Python-syntax GPU kernel writing
    xFormers                       -- attention variants
    Flash Attention 2              -- memory-efficient attention
```

The LTX-Video and Matrix-Game codebases are PyTorch + CUDA. When you port to MLX, you are translating out of this ecosystem.

---

## JAX

JAX is Google's numerical computing library for ML research. Released publicly in 2018. Built for TPUs first, with CUDA and CPU as additional backends. Designed around functional programming, automatic differentiation, and just-in-time compilation.

### Design Philosophy

JAX's core abstraction is different from PyTorch's in a fundamental way: **JAX is functional.** Functions are pure (no side effects), arrays are immutable, and autograd works on functions rather than on a graph built through mutation.

```python
import jax
import jax.numpy as jnp

# JAX arrays are immutable NumPy-like arrays
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

# Standard operations (NumPy-like API)
y = jnp.relu(x @ x.T)

# Autograd: differentiate a function, not a tensor
def loss_fn(params, x):
    w, b = params
    return jnp.mean((x @ w + b) ** 2)

# grad returns a function that computes the gradient
grad_fn = jax.grad(loss_fn)
grads = grad_fn(params, x)

# value_and_grad computes both
loss, grads = jax.value_and_grad(loss_fn)(params, x)
```

JAX's autograd operates on the function itself, not on tensors with `.requires_grad=True`. This functional style enables transformations that are more awkward in PyTorch: `jax.vmap` (vectorize a function over a batch axis), `jax.pmap` (parallelize across devices), `jax.jit` (JIT compile to XLA).

### XLA and JIT compilation

JAX compiles everything through **XLA** (Accelerated Linear Algebra), Google's ML-focused compiler. When you wrap a function in `jax.jit`, XLA traces the function, optimizes the computation graph (fusing operations, tiling for memory efficiency), and generates device-specific code -- for TPU, GPU, or CPU.

```python
# jit compiles and caches the function
@jax.jit
def forward(params, x):
    w1, w2 = params
    h = jnp.relu(x @ w1)
    return h @ w2

# First call: traces and compiles (~100ms)
out = forward(params, x)

# Subsequent calls: runs compiled XLA code (fast)
out = forward(params, x)
```

XLA compilation is more aggressive than `torch.compile` -- it traces the full computation including control flow, and generates highly optimized kernel sequences. The downside is that dynamic shapes (varying sequence lengths, varying batch sizes) require retracing, which can cause unexpected slowdowns in production.

### Hardware Support

JAX supports CUDA, TPU, and CPU. It does not support Apple Silicon natively.

```
JAX HARDWARE BACKENDS

  CUDA (NVIDIA):   🟢 Full support. GPU kernels via XLA + CUDA.
  TPU (Google):    🟢 Native. JAX's primary design target.
  CPU:             🟢 Full support.
  Apple Silicon:   🔴 Not supported. No Metal backend.
                      Running JAX on a Mac means running on CPU,
                      which is dramatically slower.
```

This is the key practical reason JAX is not a path for Apple Silicon development. If your workflow is CUDA -> Apple Silicon, JAX is not a stepping stone -- it is a dead end on the hardware side.

### Why JAX Matters for MLX Understanding

Even though JAX does not run on Apple Silicon, understanding JAX helps you understand MLX because MLX borrowed heavily from JAX's design:

| JAX concept | MLX equivalent |
|-------------|----------------|
| `jax.grad(f)` | `mx.grad(f)` |
| `jax.value_and_grad(f)` | `mx.value_and_grad(f)` |
| `jax.jit(f)` | operations are lazy by default; `mx.compile(f)` for explicit compilation |
| `jnp.zeros`, `jnp.ones` | `mx.zeros`, `mx.ones` |
| Immutable arrays | MLX arrays are also copy-on-write |
| Functional autograd | MLX uses the same pattern |

If you have JAX experience, MLX will feel familiar. If you have only PyTorch experience, the autograd model is the main conceptual shift.

---

## MLX

MLX is Apple's ML framework for Apple Silicon. Released December 2023. Combines a NumPy-like API with JAX-style functional autograd, lazy evaluation, and tight integration with Metal (Apple's GPU API).

### Design Philosophy

MLX occupies a deliberate niche: **fast iteration on Apple Silicon for researchers.** It is not trying to replace PyTorch in the data center. It is trying to make Apple Silicon a first-class ML research platform.

The three principles that shape every design decision:

1. **Familiar to NumPy/PyTorch users.** The array operations API mirrors NumPy closely. Standard operations have the same names and semantics. A PyTorch user can read MLX code without a manual.

2. **Unified memory is the model.** No `.to('cuda')`, no `.to('cpu')`. There is one device. MLX never copies data between CPU and GPU because on Apple Silicon, they share the same memory.

3. **Lazy evaluation enables automatic optimization.** Operations are queued, not executed. MLX fuses compatible operations automatically when evaluation is forced. The researcher writes simple sequential code; the framework generates efficient fused kernels.

### API Comparison: PyTorch vs MLX

The translation from PyTorch to MLX is largely mechanical for standard operations:

```python
# PYTORCH
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MLP(768, 3072, 768)
x = torch.randn(32, 768)
out = model(x)

# Autograd
loss = out.sum()
loss.backward()
for p in model.parameters():
    print(p.grad)
```

```python
# MLX (equivalent)
import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def __call__(self, x):
        x = mx.maximum(self.fc1(x), 0)   # relu is mx.maximum(x, 0)
        return self.fc2(x)

model = MLP(768, 3072, 768)
x = mx.random.normal((32, 768))
out = model(x)
mx.eval(out)                              # force evaluation

# Autograd: functional style (like JAX)
def loss_fn(model, x):
    return model(x).sum()

loss_and_grad = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad(model, x)
mx.eval(loss, grads)
```

Key differences visible in this example:
- `__call__` instead of `forward` (Python convention, not a special PyTorch method)
- `mx.maximum(x, 0)` instead of `torch.relu(x)` (MLX has `nn.relu` too, but `maximum` is explicit)
- `mx.eval(out)` required to force execution
- Autograd uses `nn.value_and_grad(model, loss_fn)` -- functional, model parameters are inputs

### Lazy Evaluation in Practice

MLX's lazy evaluation is the most common source of confusion when porting from PyTorch. In PyTorch, this works fine:

```python
# PyTorch: operations execute immediately
x = torch.randn(1000, 1000)
y = x @ x.T
z = torch.relu(y)
print(f"Mean: {z.mean():.4f}")  # no issue -- z is computed
```

In MLX, the same code pattern requires attention:

```python
# MLX: operations are deferred
x = mx.random.normal((1000, 1000))
y = x @ x.T
z = mx.maximum(y, 0)

# z is NOT yet computed here. It is a pending computation graph.
# Printing forces evaluation automatically:
print(f"Mean: {z.mean().item():.4f}")  # forces eval, then prints

# For benchmarking, always force eval explicitly:
import time
mx.eval(x)           # ensure inputs are materialized

start = time.time()
z = mx.maximum(x @ x.T, 0)
mx.eval(z)           # wait for GPU to finish
end = time.time()
print(f"Took {end - start:.4f}s")
```

Forgetting `mx.eval()` in benchmarks is the most common MLX pitfall. You will measure near-zero time (Python graph building) rather than actual GPU execution.

### Porting Custom Operations: the Hard Part

Standard operations (linear layers, attention, activations, normalizations) port mechanically from PyTorch to MLX. The hard cases are custom CUDA kernels -- operations that do not exist as built-in MLX primitives.

Three options, in order of effort:

**Option 1: Decompose into MLX primitives.** Most custom CUDA kernels are fusions of standard operations for performance, not algorithmic novelty. `fused_add_norm`, `flash_attention_forward`, most custom activations -- these can be decomposed into standard MLX operations, and MLX's lazy evaluation + kernel fusion will recover most of the performance automatically.

```python
# Custom CUDA kernel: fused add + RMSNorm
# In PyTorch (calling a custom CUDA extension):
out = fused_add_norm_cuda(x, residual, weight, eps)

# In MLX: decompose into primitives
# MLX will fuse these operations automatically
residual = x + residual
variance = mx.mean(residual ** 2, axis=-1, keepdims=True)
out = residual * mx.rsqrt(variance + eps) * weight
```

**Option 2: Write a Metal kernel.** For operations that genuinely cannot be efficiently expressed as a composition of primitives, you can write a kernel in Metal Shading Language and register it with MLX via its C++ extension API. This is the MLX equivalent of writing a CUDA `.cu` file. More work, but necessary for operations with complex memory access patterns (e.g., custom attention variants with non-standard masking or indexing).

**Option 3: Accept the performance cost.** For non-critical operations, running a decomposed Python implementation on MLX may be fast enough. Profile first -- Apple Silicon's unified memory and fast on-package bandwidth mean "slow" MLX code is often faster than expected.

---

## Framework Comparison

### API Design

| Aspect | PyTorch | JAX | MLX |
|--------|---------|-----|-----|
| Execution model | Eager by default; `torch.compile` for optimization | Lazy via `jit` decoration | Lazy always; `mx.eval()` to execute |
| Autograd style | Tensor-based (`.backward()`) | Functional (`grad(f)`) | Functional (`mx.grad(f)`) |
| NumPy compatibility | Partial (different API) | High (`jnp.*` mirrors `np.*`) | High (`mx.*` mirrors `np.*`) |
| Module definition | `nn.Module` + `forward()` | Pytrees (arbitrary Python) | `nn.Module` + `__call__()` |
| Data loading | DataLoader, Dataset | Manual or third-party | Manual (no official DataLoader) |
| Device management | `.to('cuda')` / `.to('cpu')` | `jax.device_put()` | Not needed (one device) |

### Ecosystem

| Aspect | PyTorch | JAX | MLX |
|--------|---------|-----|-----|
| Model zoo | 🟢 Massive (Hugging Face) | 🟡 Growing (Flax, Haiku) | 🟡 1000+ on mlx-community |
| LLM inference | 🟢 vLLM, TGI, TensorRT-LLM | 🟡 MaxText, Paxml | 🟡 mlx-lm |
| Image generation | 🟢 Diffusers (canonical) | 🟡 Limited | 🟡 mlx-examples stable diffusion |
| Video generation | 🟢 Diffusers (LTX-Video, etc.) | 🔴 None | 🔴 Early (Matrix-Game-mlx) |
| Fine-tuning | 🟢 PEFT, Axolotl, Unsloth | 🟡 Orbax + manual | 🟡 mlx-lm LoRA |
| Distributed training | 🟢 FSDP, DeepSpeed, NCCL | 🟢 pmap, multi-TPU | 🔴 Not supported |
| Custom kernels | 🟢 CUDA C++, Triton | 🟢 Custom XLA ops | 🟡 Metal Shading Language |
| Debugging tools | 🟢 PyTorch profiler, Nsight | 🟡 XLA profiler | 🟡 Apple Instruments |

### Performance Model

| Aspect | PyTorch | JAX | MLX |
|--------|---------|-----|-----|
| Eager overhead | High (per-op dispatch) | Low (all via XLA) | Low (lazy + fused) |
| Compilation | `torch.compile` (optional, 2.0+) | `jax.jit` (standard) | Automatic + `mx.compile()` |
| Kernel fusion | Via `torch.compile` or Triton | Via XLA (always) | Via lazy evaluation (always) |
| Memory efficiency | Manual with custom kernels | Good (XLA optimizes) | Good (fusion + unified memory) |
| Multi-GPU scaling | 🟢 Linear scaling with NCCL | 🟢 pmap for TPU/GPU | 🔴 Single-device only |

### Hardware Support

| Hardware | PyTorch | JAX | MLX |
|----------|---------|-----|-----|
| NVIDIA GPU (CUDA) | 🟢 Full | 🟢 Full | 🔴 None |
| Apple Silicon (Metal) | 🟡 MPS backend (limited) | 🔴 None | 🟢 Native |
| Google TPU | 🟡 Via PyTorch/XLA | 🟢 Native | 🔴 None |
| AMD GPU (ROCm) | 🟡 Community support | 🟡 Partial | 🔴 None |
| CPU | 🟢 Full | 🟢 Full | 🟢 Full |

### Community

| Aspect | PyTorch | JAX | MLX |
|--------|---------|-----|-----|
| Developers | 🟢 Millions | 🟡 Hundreds of thousands | 🟡 Thousands, growing |
| Stack Overflow answers | 🟢 100,000+ | 🟡 10,000+ | 🔴 Limited |
| GitHub stars | 🟢 ~85,000 | 🟡 ~30,000 | 🟡 ~20,000+ |
| Corporate backing | Meta (primary) + community | Google | Apple |
| Research paper usage | 🟢 Dominant | 🟡 Common (Google/DeepMind) | 🔴 Rare so far |
| Tutorials / courses | 🟢 Exhaustive | 🟡 Good (JAX 101) | 🟡 Growing |

---

## Gap Analysis

| Capability | PyTorch (CUDA) | MLX | Gap |
|-----------|---------------|-----|-----|
| **API maturity** | 🟢 Battle-tested, stable | 🟡 Core is stable; edges still changing | Minor -- core ops are reliable; some advanced APIs shift between releases |
| **Model zoo / pre-trained weights** | 🟢 Tens of thousands, all architectures | 🟡 1000+ converted models | Medium -- most popular models available; obscure or new architectures need manual conversion |
| **Distributed training** | 🟢 FSDP, DeepSpeed, Megatron, NCCL | 🔴 Not supported | Large -- MLX is single-device only; no path to multi-node |
| **Custom ops (equivalent to Triton)** | 🟢 Triton, CUDA C++, extensive docs | 🟡 Metal Shading Language (steeper curve) | Medium -- Metal is capable but documentation is sparse and community is small |
| **Debugging tools** | 🟢 PyTorch profiler, `torch.autograd.set_detect_anomaly` | 🟡 Instruments, `mx.metal.get_peak_memory()` | Medium -- basic debugging is fine; kernel-level profiling less mature |
| **Mobile / edge deployment** | 🟢 PyTorch Mobile, Core ML export | 🔴 No mobile deployment path | Large -- MLX targets Mac; no iOS/Android deployment |
| **Inference throughput (serving)** | 🟢 vLLM, continuous batching, paged attention | 🟡 mlx-lm server; no paged attention | Large for production -- MLX is not competitive for high-throughput API serving |
| **Flash Attention** | 🟢 Flash Attention 2 (hand-fused CUDA kernel) | 🟡 MLX attention is efficient but not a port of FA2 | Medium -- MLX's attention is good; not identical to FA2's tiling strategy |
| **Training stability tooling** | 🟢 Gradient clipping, loss scaling, anomaly detection | 🟡 Basic gradient operations; less tooling | Medium -- fine-tuning works; large-scale pretraining has less infrastructure |
| **PyTorch ecosystem interop** | 🟢 Native | 🟡 Weight conversion via numpy bridge | Minor for inference; models convert via `numpy()` and `mx.array(np_array)` |

---

## Contribution Opportunities

**DataLoader equivalent for MLX.** PyTorch's `DataLoader` handles batching, shuffling, multiprocessing prefetch, and augmentation pipelines. MLX has no equivalent -- you write data loading manually or use PyTorch's DataLoader and convert batches via NumPy. A proper MLX data loading utility would significantly improve training ergonomics.

**Triton-style kernel authoring for Metal.** Triton lets you write GPU kernels in Python with array-style syntax. There is no equivalent for Metal. A Python DSL that compiles to Metal Shading Language would lower the barrier to writing custom MLX operations dramatically.

**`torch.compile` style tracing for MLX.** MLX has `mx.compile()` which provides some JIT benefits, but its tracing is more limited than `torch.compile`'s TorchDynamo. Improving MLX's graph capture to handle more dynamic Python patterns would bring it closer to PyTorch's performance on complex models.

**Streaming inference with KV cache management.** vLLM's paged attention enables high-throughput LLM serving by managing the KV cache across many concurrent requests. MLX's inference is single-user. Building a paged attention or equivalent memory management system for mlx-lm would make MLX-powered Macs viable as local API servers under concurrent load.

**Hugging Face Diffusers backend for MLX.** Diffusers is the canonical library for image and video diffusion models. Every LTX-Video, FLUX, and Stable Diffusion implementation starts there. An official MLX backend for Diffusers (the way MPS is a backend today) would mean that new diffusion models could run on Apple Silicon without requiring a separate port.

---

## Sources

- PyTorch documentation: [pytorch.org/docs/](https://pytorch.org/docs/). API reference, tutorials, and design notes.
- PyTorch 2.0 release notes (torch.compile): [pytorch.org/blog/pytorch-2.0-release/](https://pytorch.org/blog/pytorch-2.0-release/)
- JAX documentation: [jax.readthedocs.io/](https://jax.readthedocs.io/). Quickstart, JAX 101 tutorial, sharp bits.
- MLX documentation: [ml-explore.github.io/mlx/](https://ml-explore.github.io/mlx/). API reference and migration guide.
- Triton documentation: [triton-lang.org/](https://triton-lang.org/). Python-syntax GPU kernel writing.
- Hugging Face Diffusers: [huggingface.co/docs/diffusers/](https://huggingface.co/docs/diffusers/). The canonical library for diffusion models, relevant for understanding what MLX needs to match.

---

## See Also

- [CUDA vs MLX Overview](01-cuda-vs-mlx-overview.md) -- the hardware layer beneath all these frameworks; unified memory vs. VRAM is why MLX exists as a separate project rather than a PyTorch backend
- [GPU Computing](../01-foundations/04-gpu-computing.md) -- kernel execution model, lazy evaluation rationale, memory hierarchy; explains why framework design choices (lazy vs. eager) have hardware consequences
- [Training vs. Inference](../01-foundations/03-training-vs-inference.md) -- PyTorch's distributed training ecosystem only matters for training; MLX's limitations are primarily in training, not inference
- [LLMs](03-llms.md) -- where MLX's framework capabilities are most developed; mlx-lm is the most complete MLX library relative to its PyTorch counterparts
- [Video Generation](05-video-generation.md) -- where the framework gap is most visible; LTX-Video and Matrix-Game exist as PyTorch + CUDA codebases that are being ported to MLX
