# Tooling: CUDA vs MLX

> **Status:** 🟢 CUDA (Mature -- comprehensive profiling, debugging, conversion, and benchmarking toolchain) | 🟡 MLX (Growing -- essential tools present; depth and integration significantly behind)

## What This Domain Covers

The developer tools that surround a framework: profiling to understand where time is spent, debugging to find correctness problems, converting models between formats, quantizing models for deployment, and benchmarking to measure performance. These tools are not the framework itself, but they determine how productive a developer can be when working with it.

For a developer porting CUDA to MLX, tooling gaps create friction at every step. Profiling a PyTorch model on CUDA gives you kernel-level timing, memory bandwidth utilization, and occupancy metrics within a single tool. The equivalent workflow on MLX requires combining Xcode Instruments, manual timing with `mx.eval`, and indirect inference. The gap is not theoretical -- it slows down every optimization cycle.

This page maps what exists in each ecosystem, where the gaps are, and what a contributor could build to close them.

---

## The CUDA Ecosystem

### Profiling: Kernel-Level Visibility

CUDA's profiling ecosystem gives developers visibility down to the hardware counter level. Understanding exactly which kernel ran, how many cycles it took, and whether memory access patterns were optimal is standard practice for CUDA performance work.

**NVIDIA Nsight Systems** is the primary profiling tool for system-level analysis. It shows a timeline of CPU activity, GPU kernel execution, CUDA API calls, memory transfers, and thread activity. For a transformer training run, you can see precisely which kernels are running, when they start and stop, and how much of the GPU is utilized.

```
NSIGHT SYSTEMS TIMELINE VIEW (schematic)

  CPU Timeline:
    [forward()]        [loss.backward()]       [optimizer.step()]
    |=======|          |==============|        |======|
    Time --->

  GPU Timeline:
    [matmul] [attn] [layernorm] [mlp]  [backward kernels...]
             |                          |
             Gap: CPU was slow         GPU idle

  CUDA API calls:
    cudaMemcpy  cudaLaunchKernel  cudaLaunchKernel ...

  Analysis: The gap between forward and backward means the CPU is
            taking too long to build the backward graph.
            torch.compile() or JIT tracing would close this gap.
```

**NVIDIA Nsight Compute** goes deeper: it profiles a single kernel at the hardware counter level. It reports:
- Memory bandwidth utilization (% of peak)
- Compute utilization (% of peak SM throughput)
- Warp efficiency (% of warps not stalled)
- Cache hit rates (L1, L2, shared memory)
- Specific stall reasons (memory latency, pipeline stall, synchronization)

This is the tool you use when a custom CUDA kernel runs slower than expected and you need to know why.

**PyTorch Profiler** is the framework-level profiler, integrated directly into PyTorch. It annotates PyTorch operations and connects them to their underlying CUDA kernels.

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = MyTransformerModel().cuda()
x = torch.randn(32, 512, 768).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        out = model(x)

# Print table sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export for TensorBoard
prof.export_chrome_trace("trace.json")
```

Output: a table of every operation with CPU time, CUDA time, memory allocated, and call count. The Chrome trace can be loaded in any browser for visual inspection.

**TensorBoard** doubles as a profiling viewer -- the TensorBoard profiler plugin shows GPU utilization, kernel timelines, and bottleneck analysis from PyTorch Profiler traces.

### Quantization Tools

Quantization reduces model size and increases inference throughput by representing weights and/or activations in lower-precision formats. The CUDA ecosystem has mature tooling for every quantization strategy.

**AutoGPTQ** implements GPTQ (Generative Pre-Trained Transformer Quantization), a post-training quantization method that minimizes quantization error layer-by-layer using a small calibration dataset. Produces INT4 or INT8 models that load directly into Transformers.

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Quantize a model with GPTQ
quantize_config = BaseQuantizeConfig(
    bits=4,               # 4-bit quantization
    group_size=128,       # group size for quantization
    desc_act=False        # disable activation reordering
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantize_config=quantize_config
)

# Calibration: run a sample of training data through the model
# (GPTQ minimizes reconstruction error on this data)
model.quantize(calibration_examples)

# Save quantized model
model.save_quantized("./llama-3.2-3b-gptq")
```

**AutoAWQ** implements AWQ (Activation-aware Weight Quantization), which identifies and protects the most salient weights from quantization error. AWQ models are generally higher quality than GPTQ at the same bit width.

**BitsAndBytes** provides on-the-fly quantization during model loading: `load_in_8bit=True` or `load_in_4bit=True` in `from_pretrained`. Used pervasively for inference and as the base for QLoRA fine-tuning.

**llama.cpp quantize** converts models to GGUF format with k-bit quantization schemes (Q2_K through Q8_0). The GGUF format bundles all quantization metadata into a single portable file.

```
CUDA QUANTIZATION FORMAT LANDSCAPE

  Format        Tool          Bit widths    Where used
  ------        ----          ----------    ----------
  GPTQ          AutoGPTQ      INT4, INT8    Transformers inference
  AWQ           AutoAWQ       INT4          Transformers inference (better quality)
  BnB NF4       BitsAndBytes  4-bit NF4     QLoRA training; HF inference
  GGUF (Q*)     llama.cpp     Q2-Q8         llama.cpp, Ollama
  TensorRT INT8 TensorRT      INT8          NVIDIA production serving
  TensorRT FP8  TensorRT      FP8           H100 production serving
```

### Conversion and Export

**Hugging Face safetensors** is the standard weight storage format: memory-mapped, zero-copy loading, secure (no arbitrary code execution unlike pickle). Most new model releases use safetensors for the weight files.

**ONNX (Open Neural Network Exchange)** is the intermediate representation for model export. A model exported to ONNX can run on any ONNX Runtime-compatible backend: CPU, CUDA, DirectML (Windows), TensorRT, CoreML.

```python
import torch

model = MyModel().cuda()

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
```

**TensorRT converter** takes an ONNX model (or PyTorch model via `torch.compile(backend="tensorrt")`) and compiles it to a TensorRT engine: a serialized, NVIDIA-specific binary that is optimized for a specific GPU and input shape. TensorRT engines achieve the highest throughput of any CUDA inference format.

**Core ML Tools (Apple)** converts PyTorch models to Core ML format for deployment on Apple devices. This is separate from MLX -- Core ML targets iOS/macOS production deployment via the Neural Engine; MLX targets research and development on Apple Silicon.

### Debugging Tools

**PyTorch hooks and anomaly detection:**

```python
# Detect NaN/Inf in autograd
torch.autograd.set_detect_anomaly(True)

# Register gradient hooks to inspect values
def grad_hook(grad):
    print(f"Gradient norm: {grad.norm():.4f}, max: {grad.max():.4f}")
    return grad

layer.weight.register_hook(grad_hook)

# Check for exploding/vanishing gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
```

**CUDA-memcheck** (now Compute Sanitizer) detects out-of-bounds memory accesses, race conditions, and memory leaks in CUDA kernels. Essential when debugging custom CUDA kernel implementations.

**cuda-gdb** is GDB extended with CUDA support, allowing you to step through CUDA kernel code, inspect thread-local variables, and set breakpoints on the GPU.

### Benchmarking

**MLPerf** is the industry-standard ML benchmarking suite. It defines specific workloads (image classification, object detection, language model training, speech recognition) and measures performance in samples/second or tokens/second under strict rules. Results are submitted and published, allowing fair comparison across hardware vendors.

**Triton benchmarking utilities** provide micro-benchmarking helpers for comparing kernel implementations:

```python
import triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["cublas", "triton"],
        line_names=["cuBLAS", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    if provider == "cublas":
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    elif provider == "triton":
        ms = triton.testing.do_bench(lambda: matmul_kernel(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms)
```

---

## The MLX Ecosystem

### Profiling: What Is Available

MLX's profiling story is built around two tools: Xcode Instruments and manual timing. Neither gives the kernel-level visibility that Nsight provides.

**Xcode Instruments -- Metal System Trace** is the most powerful profiling tool available for MLX. It shows Metal GPU work items, command buffer scheduling, CPU activity, and memory allocations on a timeline. To use it for MLX workloads:

```
PROFILING MLX WITH INSTRUMENTS

  1. Open Xcode Instruments
  2. Select "Metal System Trace" template
  3. Target: your Python process
  4. Record while running your MLX script
  5. Inspect:
     - GPU Work section: Metal command buffers, when they execute
     - CPU section: Python, MLX dispatch thread activity
     - Memory: allocation patterns

  What you see: high-level GPU timeline, memory usage
  What you do NOT see: individual kernel names, occupancy,
                       memory bandwidth utilization, cache stats
```

**mlx.metal.get_peak_memory()** and **mlx.metal.reset_peak_memory()** provide memory tracking:

```python
import mlx.core as mx

mx.metal.reset_peak_memory()

# Run your operation
x = mx.random.normal((4096, 4096))
y = x @ x.T
# Force computation before measuring: MLX is lazy
mx.eval(y)

peak_mb = mx.metal.get_peak_memory() / 1e6
print(f"Peak memory: {peak_mb:.1f} MB")
```

**Manual timing** is the standard way to benchmark MLX operations. The critical requirement: force evaluation before stopping the timer, because MLX operations are lazy -- they build a computation graph without executing it until `mx.eval` is called.

```python
import mlx.core as mx
import time

def benchmark_fn(fn, *args, warmup=5, n_runs=20):
    """Benchmark an MLX function with proper synchronization."""
    # Warmup: let Metal compile shaders and warm caches
    for _ in range(warmup):
        out = fn(*args)
        # Materialize the lazy graph; without this, timing measures
        # only the Python graph-building cost, not GPU execution
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = fn(*args)
        mx.eval(out)     # wait for GPU to finish
        end = time.perf_counter()
        times.append(end - start)

    avg_ms = (sum(times) / len(times)) * 1000
    min_ms = min(times) * 1000
    print(f"avg: {avg_ms:.2f} ms, min: {min_ms:.2f} ms")
    return times

x = mx.random.normal((2048, 2048))
benchmark_fn(lambda a: a @ a.T, x)
```

**No MLX-native kernel profiler exists.** There is no tool equivalent to Nsight Compute that shows memory bandwidth utilization or SM occupancy for Metal kernels launched by MLX. This is the most significant profiling gap.

### Quantization: mlx-lm quantize

mlx-lm provides a quantization command that converts a Hugging Face model to MLX format with group quantization:

```python
# Command line
# mlx_lm.convert \
#   --hf-path meta-llama/Llama-3.2-3B-Instruct \
#   --mlx-path ./llama-3.2-3b-4bit \
#   --q-bits 4 \
#   --q-group-size 64

# Python API
from mlx_lm import convert

convert(
    hf_path="meta-llama/Llama-3.2-3B-Instruct",
    mlx_path="./llama-3.2-3b-4bit",
    quantize=True,
    q_bits=4,             # 4 or 8
    q_group_size=64,      # weights quantized in groups of 64
)
```

What this does: loads the model weights, applies `nn.QuantizedLinear` to linear layers, and saves the quantized weights in MLX's numpy-based format (npz files + config.json).

What is missing:
- No calibration-based quantization (like GPTQ/AWQ) -- MLX quantization is post-training round-to-nearest, not error-minimizing
- No FP8 or INT8 quantization
- No fine-grained per-layer quantization configuration
- No quantization quality evaluation tooling built in

### Conversion: Weight Conversion Scripts

MLX does not have a unified conversion tool equivalent to ONNX. Model conversion is handled per-architecture by scripts in mlx-lm and mlx-examples.

```
MLX CONVERSION WORKFLOW

  Source format:  HuggingFace safetensors (PyTorch weights)
  Tool:           mlx_lm.convert (for LLMs)
                  Manual conversion scripts (for other models)
  Output format:  .npz weight files + config.json + tokenizer files

  What mlx_lm.convert handles:
    - Weight key remapping (e.g., "model.layers.0.self_attn.q_proj.weight"
      may need renaming for the MLX model class)
    - Transposing weights where PyTorch and MLX conventions differ
    - Splitting or merging weight tensors (e.g., QKV concatenation)
    - Optional 4-bit or 8-bit quantization

  What it does NOT handle:
    - Models outside mlx-lm's supported architecture list
    - GGUF -> MLX conversion (no tool exists)
    - ONNX -> MLX conversion (no tool exists)
    - Core ML -> MLX conversion (no tool exists)
```

For architectures not in mlx-lm's supported list, the conversion process is manual: write a Python script that loads the PyTorch weights with `safetensors` or `torch.load`, remaps weight names to match the MLX model class, transposes as needed, and saves with `mx.save` or `numpy.save`.

### Debugging

MLX debugging is primarily Python-level debugging. There is no GPU-side debugger.

**Standard Python tools work:**

```python
import mlx.core as mx
import mlx.nn as nn

# Check for NaN or Inf after forward pass
def check_for_nan(name, tensor):
    # Force computation before inspection
    mx.eval(tensor)
    has_nan = mx.any(mx.isnan(tensor)).item()
    has_inf = mx.any(mx.isinf(tensor)).item()
    if has_nan or has_inf:
        print(f"WARNING: {name} contains {'NaN' if has_nan else 'Inf'}")

# Gradient inspection (manual)
def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y).mean()

loss_and_grad = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad(model, x, y)

# Inspect gradient norms
for name, grad in grads.items():
    if isinstance(grad, dict):
        for k, g in grad.items():
            if g is not None:
                mx.eval(g)
                norm = mx.sqrt(mx.sum(g ** 2)).item()
                print(f"{name}.{k}: grad norm = {norm:.6f}")
```

**No equivalent to PyTorch anomaly detection (`autograd.set_detect_anomaly`).**

**No GPU memory error detection** equivalent to CUDA-memcheck. If a Metal kernel has an out-of-bounds access, the behavior is undefined and often silent.

**No kernel-level debugger.** Metal Shader Debugger in Xcode can debug Metal shaders, but MLX's Metal kernels are not easily inspectable through this path -- they are compiled from MLX's C++ layer and not directly editable.

### Benchmarking

MLX has no equivalent to MLPerf. Benchmarking is done with custom timing scripts.

The mlx-examples repository includes some benchmark scripts:

```
MLX BENCHMARKS (mlx-examples/benchmarks/)

  benchmark.py     -- basic operation benchmarks (matmul, attention, etc.)
  llm_benchmark.py -- token generation throughput for mlx-lm
  mlx_vs_torch.py  -- side-by-side comparison of common operations

  Usage:
  python benchmarks/benchmark.py --model Llama-3.2-3B-4bit
  python benchmarks/llm_benchmark.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

No standardized benchmark suite comparable to MLPerf exists for MLX. Published benchmarks are typically token/sec numbers from mlx-lm README files, which are useful but not reproducible under controlled conditions.

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **System-level profiling** | 🟢 Nsight Systems: CPU/GPU timeline, CUDA API calls, memory transfers, microsecond resolution | 🟡 Xcode Instruments Metal System Trace: timeline view, less granular | Medium -- Instruments works but shows less detail; Metal GPU work is visible but kernel names are not always readable |
| **Kernel-level profiling** | 🟢 Nsight Compute: memory bandwidth, compute utilization, warp efficiency, per-SM counters | 🔴 No equivalent; Metal GPU counters exist but are not surfaced through MLX | Large -- impossible to know if a custom Metal kernel is memory-bound or compute-bound without external tools |
| **Framework-level profiling** | 🟢 PyTorch Profiler: per-op CPU+CUDA timing, memory allocation, call stacks, TensorBoard export | 🟡 Manual timing only; no op-level profiler built into MLX | Large -- every profiling task in MLX requires writing benchmark code; no automatic tracing |
| **Memory profiling** | 🟢 PyTorch memory profiler: per-op allocation, fragmentation stats, peak tracking | 🟡 `mx.metal.get_peak_memory()`: peak only; no per-op breakdown | Medium -- peak memory is trackable; allocation hot spots are not |
| **Post-training quantization (calibration-based)** | 🟢 GPTQ (AutoGPTQ), AWQ (AutoAWQ): calibration-aware, minimizes reconstruction error | 🟡 Round-to-nearest only; no calibration step | Medium -- MLX quantization quality is lower for aggressive bit widths (2-3 bits); 4-bit quality is acceptable |
| **Quantization format breadth** | 🟢 GPTQ, AWQ, NF4, GGUF, INT8, FP8 -- all supported by different tools | 🟡 MLX 4-bit and 8-bit group quantization only | Medium -- limited to group quantization; cannot load GPTQ or AWQ models |
| **Model conversion (universal)** | 🟢 ONNX: universal intermediate format; PyTorch -> ONNX -> any backend | 🔴 No universal format; per-architecture conversion scripts only | Large for cross-framework portability; manageable for MLX-only workflows |
| **Model format compatibility** | 🟢 safetensors, GGUF, ONNX, TensorRT, pickle -- all loadable via different libraries | 🟡 safetensors loadable; GGUF not directly loadable; no ONNX runtime | Medium -- most popular models come in safetensors; less common formats require extra steps |
| **Gradient debugging** | 🟢 `autograd.set_detect_anomaly`: finds the exact operation that produced NaN; hooks for every layer | 🟡 Manual NaN checking after forward pass; no autograd anomaly detection | Medium -- NaNs are detectable with discipline; finding the source is harder |
| **GPU memory error detection** | 🟢 CUDA Compute Sanitizer: out-of-bounds, race conditions, memory leaks in kernels | 🔴 No equivalent for Metal kernels used by MLX | Large when writing custom Metal kernels; irrelevant for pure Python MLX code |
| **Kernel debugging** | 🟢 cuda-gdb: step through CUDA kernels, inspect thread variables | 🔴 Metal Shader Debugger works for hand-written shaders; MLX-generated kernels not easily debuggable | Large for custom kernel authors; irrelevant for framework users |
| **Standardized benchmarking** | 🟢 MLPerf: standardized workloads, strict methodology, published results across vendors | 🟡 Custom scripts only; no standardized benchmark suite | Medium -- real-world performance is measurable; comparative published results are sparse |
| **Visualization of computation graphs** | 🟢 TensorBoard graph view, `torchviz`, PyTorch model summary | 🔴 No graph visualization in MLX; lazy evaluation makes graph capture non-trivial | Small for daily work; useful for documentation and debugging complex architectures |
| **Model inspection tools** | 🟢 `torchinfo` / `model.summary()`: layer shapes, parameter counts, FLOPs estimation | 🟡 `model.parameters()` traversal; no built-in summary with shape inference | Small -- parameter count is easy; FLOPs estimation requires a library port |
| **Automated weight conversion pipeline** | 🟢 Hugging Face model hub conversion via Optimum; community converters for most architectures | 🟡 mlx-lm.convert covers supported LLM architectures; everything else is manual | Medium -- major LLMs convert automatically; vision/audio/custom models need manual scripting |
| **Profiling integration with training** | 🟢 PyTorch Profiler context manager inside training loop; W&B integration | 🟡 Manual timing wrappers; no profiler context manager | Medium -- workable with effort; not ergonomic for production training jobs |

---

## Practical Profiling Workflow: CUDA vs MLX

To make the tooling gap concrete, here is what a typical optimization cycle looks like in each ecosystem.

**CUDA profiling workflow:**

```
1. Baseline: run model, record tokens/second from vLLM or mlx-lm log
2. System profile: nsys profile python train.py
   -> Open nsys-rep in Nsight Systems UI
   -> Find GPU idle periods, CPU bottlenecks
3. Kernel profile: ncu python train.py
   -> Select slow kernels by CUDA time
   -> Open kernel report: "memory throughput 45% of peak"
      -> Conclusion: memory-bound kernel; need better access pattern
4. Fix: rewrite kernel in Triton with explicit memory tiling
5. Compare: run both versions through triton.testing.perf_report
```

**MLX profiling workflow:**

```
1. Baseline: run model, record tokens/second manually
2. Manual timing: add benchmark wrappers around suspected bottlenecks
   -> Identify slow operations by elapsed time
3. System profile: Instruments Metal System Trace
   -> Find GPU idle periods (command buffer gaps)
   -> Cannot identify individual kernel performance
4. Memory: mx.metal.get_peak_memory() around suspected allocations
5. Fix: refactor Python code, try mx.compile(), reduce intermediate arrays
6. Compare: run both versions with manual timers, compare millisecond numbers
```

The CUDA workflow gives you hardware-level evidence for why something is slow. The MLX workflow gives you timing evidence that something is slow, with limited ability to diagnose why.

---

## Contribution Opportunities

**MLX Profiler: op-level timing integrated into the framework.** The most impactful single tooling contribution would be a `mx.profiler` context manager that records the time spent in each MLX operation, similar to PyTorch Profiler. Metal has `MTLCaptureManager` and GPU timestamps that could power this. Exporting to Chrome trace format (JSON) would allow use of existing visualization tooling.

**Calibration-based quantization for MLX.** Implementing GPTQ or AWQ for MLX models would improve quantization quality at 3-bit and 2-bit depths. The algorithm is well-documented; the implementation requires running a calibration dataset through each layer and computing optimal quantization scales. This would make MLX quantization competitive with CUDA for aggressive compression.

**GGUF -> MLX converter.** GGUF is the most widely distributed format for open-source models (Ollama, llama.cpp). A bidirectional converter between GGUF and MLX format would mean that every model available in Ollama could also run in mlx-lm, and vice versa. The weight layout and quantization schemes differ but are well-specified.

**mlx-inspect: model summary tool.** A port of `torchinfo` or equivalent for MLX that shows layer names, input/output shapes, parameter counts, and estimated FLOPs for a given input shape. This is a small contribution with high daily-use value.

**Automated NaN detection via value_and_grad wrapping.** A context manager that wraps `nn.value_and_grad` calls and checks for NaN/Inf in outputs and gradients after each step, logging which layer's output first became NaN. This would approximate `torch.autograd.set_detect_anomaly` without requiring changes to MLX's autograd internals.

**MLPerf-style benchmark suite for MLX.** Defining a set of standardized workloads (inference throughput for 7B LLM, image classification throughput, attention benchmark) with a reproducible harness would make cross-version and cross-hardware comparison possible. Currently there is no way to know if a new MLX release is faster or slower than the previous one across a standard workload set.

---

## Sources

- NVIDIA Nsight Systems: [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems). System-level GPU profiler documentation.
- NVIDIA Nsight Compute: [developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute). Kernel-level profiler documentation.
- PyTorch Profiler documentation: [pytorch.org/tutorials/recipes/recipes/profiler_recipe.html](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
- Hugging Face safetensors: [github.com/huggingface/safetensors](https://github.com/huggingface/safetensors). Safe weight storage format.
- AutoGPTQ: [github.com/PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ). GPTQ quantization for Transformers models.
- AutoAWQ: [github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ). AWQ quantization implementation.
- GPTQ paper (Frantar et al., 2022): Layer-wise quantization minimizing reconstruction error.
- AWQ paper (Lin et al., 2023): Activation-aware weight quantization.
- MLX documentation -- memory and profiling: [ml-explore.github.io/mlx/](https://ml-explore.github.io/mlx/). `mx.metal.get_peak_memory`, `mx.metal.reset_peak_memory`.
- Apple Xcode Instruments documentation: [developer.apple.com/documentation/xcode/instruments](https://developer.apple.com/documentation/xcode/instruments). Metal System Trace profiling.

---

## See Also

- [GPU Computing](../01-foundations/04-gpu-computing.md) -- kernel execution model, memory hierarchy; explains what profiling metrics like "memory-bound" and "compute-bound" actually mean at the hardware level
- [Quantization](../01-foundations/10-quantization.md) -- the theory behind the quantization tools; why GPTQ and AWQ produce better quality than round-to-nearest
- [Frameworks](02-frameworks.md) -- MLX vs PyTorch API; `mx.eval()`, lazy evaluation, and why timing MLX code requires explicit synchronization before stopping the clock
- [Training / Fine-tuning](07-training-finetuning.md) -- tooling gaps are most visible during training; profiling a training run and debugging gradient problems are the primary use cases
