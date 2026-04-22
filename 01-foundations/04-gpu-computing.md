# GPU Computing

> **One-liner:** GPUs run thousands of simple calculations simultaneously, making them vastly faster than CPUs for the parallel math that ML requires.

## The Intuition

### A brilliant professor vs. a stadium of students

Imagine you need to grade 10,000 simple arithmetic problems. You have two options:

**Option 1: The professor.** A single expert who can solve any problem, no matter how complex. She works through the pile one problem at a time -- fast per problem, but still sequential. 10,000 problems means 10,000 turns.

**Option 2: The stadium.** A stadium filled with 10,000 students. Each student gets one problem, solves it simultaneously with all the others, and the whole pile is done in a single round.

For simple problems, the stadium wins by an enormous margin. For problems where each answer depends on the previous answer, the professor is the only option.

```
CPU (The Professor)                  GPU (The Stadium)
+---------------------------+        +---------------------------+
|                           |        |  o o o o o o o o o o o   |
|   One powerful processor  |        |  o o o o o o o o o o o   |
|                           |        |  o o o o o o o o o o o   |
|   Problem 1 -> answer     |        |  o o o o o o o o o o o   |
|   Problem 2 -> answer     |        |  o o o o o o o o o o o   |
|   Problem 3 -> answer     |        |  o o o o o o o o o o o   |
|   ...                     |        |  o o o o o o o o o o o   |
|   Problem 10000 -> answer |        |                           |
|                           |        |  All problems -> answers  |
|   Sequential, powerful    |        |  Parallel, simple         |
+---------------------------+        +---------------------------+
```

**CPUs are professors.** Fewer cores (8-32 on a modern desktop [CPU](../glossary.md#cpu)), but each core is extremely capable -- complex branch prediction, out-of-order execution, large caches, very high clock speed. Great for varied, sequential, logic-heavy work.

**GPUs are stadiums.** Thousands of cores (4,096 on an M3 Max, over 10,000 on an NVIDIA H100), but each core is simple. No complex branch prediction. All cores run the same instruction at the same time. Terrible for sequential logic. Exceptional for applying the same simple operation across massive amounts of data.

### Why ML fits the stadium model perfectly

Neural network math is almost entirely parallel. Consider matrix multiplication -- the core operation inside every linear layer, every attention mechanism, every convolution.

Given two matrices A (shape m x k) and B (shape k x n), the output C (shape m x n) has m * n cells. Here's the critical insight: **each output cell is computed independently from every other output cell.** The value at C[0,0] has nothing to do with the value at C[3,7]. They don't share intermediate results.

```
A (4x3)          B (3x5)          C (4x5) -- each cell is independent

[a00 a01 a02]   [b00 b01 b02 b03 b04]   [c00 c01 c02 c03 c04]
[a10 a11 a12] x [b10 b11 b12 b13 b14] = [c10 c11 c12 c13 c14]
[a20 a21 a22]   [b20 b21 b22 b23 b24]   [c20 c21 c22 c23 c24]
[a30 a31 a32]                            [c30 c31 c32 c33 c34]

Each cij = sum over k of (aik * bkj)
           ^-- computed independently, assign to its own thread
```

With 20 output cells, a CPU computes them one after another (20 steps). A [GPU](../glossary.md#gpu) assigns one thread to each cell and computes all 20 simultaneously (1 step). Scale this to a 4096 x 4096 matrix multiplication (over 16 million cells) and the speedup is the difference between hours and seconds.

Every layer in a neural network is this same pattern: element-wise operations (ReLU, sigmoid, softmax), reductions (layer norm, attention softmax), matrix multiplications. All massively parallel. All ideal for the stadium model.


## How It Actually Works

### Kernels: the programs that run on the GPU

A **kernel** is a function compiled to run directly on GPU hardware. Not a Python function -- a low-level compiled function that executes on hundreds or thousands of GPU cores simultaneously.

When you write `mx.matmul(a, b)` in MLX, you are not calling Python math. Behind the scenes, MLX dispatches a pre-compiled matmul kernel to the GPU. That kernel runs on the GPU hardware, operates on GPU memory, and produces a result -- all without your Python code doing anything further.

Each framework operation maps to one or more kernels:
- `mx.matmul` -> matrix multiplication kernel
- `mx.add` -> element-wise addition kernel
- `nn.LayerNorm` -> layer normalization kernel (a few operations fused into one)
- `nn.MultiHeadAttention` -> multiple kernels: QKV projection, softmax, output projection

When you write custom [CUDA](../glossary.md#cuda) code and port it to MLX, what you're actually doing is rewriting a kernel: replacing code written for CUDA's programming model (running on NVIDIA hardware via NVIDIA's compiler) with equivalent code for [Metal](../glossary.md#metal)'s programming model (running on Apple GPU hardware via Apple's compiler).

### Threads, blocks, and warps: how work is organized

A GPU doesn't just launch 10,000 individual threads loose. Work is organized in a strict hierarchy:

```
GPU
└── Grid (the full launch: e.g., compute matmul for the whole matrix)
    ├── Block 0 (a group of threads that share fast memory)
    │   ├── Warp 0 (32 threads that execute in lockstep)
    │   │   ├── Thread 0
    │   │   ├── Thread 1
    │   │   └── ... Thread 31
    │   ├── Warp 1
    │   └── ... Warp N
    ├── Block 1
    └── ... Block M
```

**Threads** are the individual workers. Each thread runs the same kernel code but on different data (different indices). [Thread](../glossary.md#thread) 0 might compute C[0,0]; thread 1 computes C[0,1]; and so on.

**Warps** (NVIDIA) are groups of 32 threads that execute in strict lockstep. All 32 threads in a warp execute the same instruction at the same clock cycle, but on different data. If threads in a warp take different branches (if/else), the warp executes both branches and masks off the results for threads that shouldn't take each branch -- this is called "warp divergence" and kills performance.

**Blocks** are groups of warps that share fast on-chip memory (shared memory). Threads in the same block can communicate and synchronize cheaply. Threads in different blocks cannot communicate during kernel execution.

**The grid** is the complete collection of blocks for one kernel launch.

Apple's GPU uses different terminology but the same concept: threadgroups (similar to CUDA blocks), SIMD-groups (similar to warps), and threads.

### Memory hierarchy: the real bottleneck

GPUs have a layered memory system. Understanding this is essential for understanding why GPU code is fast or slow.

```
+-------------------------------------------------------------------+
|  GPU MEMORY HIERARCHY (fastest to slowest, smallest to largest)   |
+-------------------------------------------------------------------+

[Registers]   Per-thread. Fastest possible. ~256 KB per SM.
              Variables local to one thread live here during execution.
              Disappears when the thread ends.

[Shared Memory / L1 Cache]   Per-block. Very fast. ~48-96 KB per SM.
              All threads in a block can read/write here.
              Used to stage data that multiple threads need.
              Key to optimized matmul kernels.

[L2 Cache]   On-chip. Fast. A few MB.
              Automatic, not programmer-controlled.
              Buffers traffic between compute units and global memory.

[Global Memory / VRAM]   Off-chip (NVIDIA). Very large. Very slow.
              The main GPU memory: 8 GB, 24 GB, 80 GB on NVIDIA cards.
              All data starts here. Must be staged to shared memory
              before the fast compute units can use it efficiently.

+-------------------------------------------------------------------+
Speed ratio: register access ~1 cycle
             shared memory ~5 cycles
             L2 cache ~30 cycles
             global memory ~300-600 cycles
+-------------------------------------------------------------------+
```

The performance consequence is dramatic: a kernel that reads from global memory on every operation is 100-600x slower than one that loads data into shared memory once and works from there. Writing fast GPU kernels is largely the art of keeping your working data in shared memory or registers.

### Memory bandwidth vs. compute: the real limit

Modern GPUs can perform trillions of floating-point operations per second (teraFLOPS). But they can only move a limited amount of data per second from global memory to the compute units (measured in GB/s).

Many ML operations are **memory-bandwidth-bound**: the GPU finishes its arithmetic faster than new data arrives from memory, and the compute units sit idle waiting. Adding 2 billion numbers is trivially fast arithmetic -- but reading those numbers from memory is the bottleneck.

This is why techniques like:
- **[Kernel](../glossary.md#kernel) fusion** -- combining multiple operations (add + ReLU, or attention's QKV projections) into a single kernel that reads data once and applies all operations before writing back
- **[Flash Attention](../glossary.md#flash-attention)** -- a rewritten attention kernel that tiles the computation to stay in fast SRAM rather than writing intermediate results to [VRAM](../glossary.md#vram)
- **[Quantization](../glossary.md#quantization)** -- smaller dtypes (float16, int8) mean less data to move from memory

... matter so much for real-world performance. They're not about doing fewer math operations -- they're about moving less data.

### CUDA cores vs. Apple GPU cores: different architectures, same principle

```
NVIDIA GPU (discrete)                Apple GPU (unified)
+----------------------------+        +----------------------------+
| CUDA Cores                 |        | GPU Cores                  |
| (floating point units)     |        | (Apple calls them ALUs)    |
|                            |        |                            |
| Tensor Cores               |        | Matrix multiply units      |
| (accelerated matmul)       |        | (AMX -- Apple Matrix Ext.) |
|                            |        |                            |
| VRAM: 8-80 GB              |        | Unified memory: shared     |
| separate from system RAM   |        | with CPU, 8-192 GB         |
|                            |        |                            |
| PCIe bus to CPU            |        | On-package, low latency    |
| ~64 GB/s bandwidth         |        | ~400+ GB/s bandwidth       |
|                            |        | (M3 Max: 400 GB/s)         |
+----------------------------+        +----------------------------+
Programming: CUDA             Programming: Metal
```

NVIDIA's CUDA (Compute Unified Device Architecture) is the programming model for NVIDIA GPUs: you write kernels in a C-like language, compile with `nvcc`, and launch them through the CUDA runtime. The majority of ML research and production infrastructure runs on CUDA.

Apple's Metal is the equivalent for Apple GPUs: kernels written in Metal Shading Language, compiled with Apple's toolchain, launched through the Metal API. MLX abstracts over Metal -- you don't write Metal kernels directly when using MLX's built-in operations, but when porting a custom CUDA kernel, you are ultimately writing the Metal equivalent.

### Unified Memory: Apple Silicon's key structural advantage

The most important architectural difference between NVIDIA GPUs and Apple Silicon is not the core count or clock speed -- it is the memory architecture.

```
NVIDIA (Discrete GPU)                Apple Silicon (Unified)
+------------------+                 +--------------------------------+
|                  |                 |                                |
|   CPU            |                 |   CPU Cores                    |
|   (system RAM)   |                 |          |                     |
|   16-256 GB      |                 |          | (same memory pool)  |
|        |         |                 |          |                     |
|   PCIe bus  <--- COPY --->         |   GPU Cores                    |
|        |         |                 |                                |
|   GPU            |                 |   Unified Memory               |
|   (VRAM)         |                 |   8 GB to 192 GB               |
|   8-80 GB        |                 |   (one pool, used by both)     |
+------------------+                 +--------------------------------+

Bottleneck: data copy across PCIe    No copy needed. Ever.
~16-64 GB/s, adds latency            ~400 GB/s on-package bandwidth
```

On NVIDIA systems, data starts in system RAM (CPU memory). Before the GPU can use it, it must be copied across the PCIe bus to VRAM. After the GPU is done, results must be copied back. For small transfers this overhead is significant. For large models, this copying imposes a fundamental latency floor.

Apple Silicon has no VRAM. The CPU and GPU share a single physical memory pool, and the memory controller can be accessed at high bandwidth by either. When you load a model in MLX, the weights go into unified memory. The GPU reads them from there directly -- no copy. The CPU can also read them -- no copy. Switching between CPU and GPU operations has zero data transfer cost.

This is why MLX exists as a framework targeting Apple Silicon specifically. The unified memory architecture makes it possible to build an efficient ML framework that moves fluidly between CPU and GPU work without copying. On NVIDIA hardware, you'd have to manage the PCIe transfers carefully. On Apple Silicon, you don't.

### VRAM limits vs. unified memory limits

One of the most common practical differences you will encounter:

**On NVIDIA:** "Out of memory" means VRAM is exhausted. A GPU with 24 GB of VRAM can only hold 24 GB worth of model weights, activations, and intermediate tensors. You cannot use system RAM as overflow during inference (in practice, the PCIe bandwidth would make it too slow). If your model doesn't fit in VRAM, your options are: quantize the model, reduce batch size, use a different GPU, or distribute across multiple GPUs.

**On Apple Silicon:** There is no fixed VRAM. The GPU can use all available unified memory. A Mac with 64 GB of unified memory can, in principle, run a model that needs 60 GB. If the model exceeds physical memory, the system will use swap (disk-backed virtual memory) -- which is dramatically slower, but at least it runs. The practical limit is system RAM, not a separate VRAM partition.

This is why running large quantized models (70B parameter models in 4-bit quantization) is feasible on high-memory M-series Macs in ways that are simply impossible on consumer NVIDIA GPUs.


## Why It Matters for MLX

### Lazy evaluation and kernel fusion

MLX does not execute operations immediately. When you write:

```python
import mlx.core as mx

a = mx.random.normal((1024, 1024))
b = mx.random.normal((1024, 1024))
c = a + b           # no computation yet
d = mx.maximum(c, 0)   # (ReLU) -- still no computation
e = d * 2.0         # still no computation
```

Nothing has run. MLX has built an internal graph of pending operations.

When you call `mx.eval(e)` (or print `e`, or use `e` in a way that requires its value), MLX examines the pending graph and decides how to execute it. In this case, instead of:
1. Launch add kernel: read a and b from memory, write c to memory
2. Launch ReLU kernel: read c from memory, write d to memory
3. Launch scale kernel: read d from memory, write e to memory

MLX can **fuse** these into a single kernel:
1. Launch fused kernel: read a and b from memory, compute add+relu+scale, write e to memory

Fusion reduces memory bandwidth by 3x. The intermediate tensors c and d never exist in global memory -- they live in registers inside the fused kernel.

`mx.eval` is how you force this deferred computation to run:

```python
mx.eval(e)  # the GPU executes the fused kernel here
print(e)    # safe to read; result is materialized in memory
```

This matters for profiling. If you time operations in MLX without calling `mx.eval`, you are not measuring GPU execution time -- you are measuring Python graph-building time (nearly zero). Always force evaluation before and after the code you are benchmarking:

```python
import time

mx.eval(x)  # ensure input is materialized

start = time.time()
y = model(x)
mx.eval(y)  # wait for the GPU to finish
end = time.time()

print(f"Inference took {end - start:.3f}s")
```

### Metal vs. CUDA: what you're translating to

When you port a custom CUDA kernel to MLX, here is what you are actually doing:

**CUDA side (what the original code looks like):**

```cuda
// CUDA kernel: element-wise ReLU
__global__ void relu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Launch: 256 threads per block, enough blocks to cover n elements
int blocks = (n + 255) / 256;
relu_kernel<<<blocks, 256>>>(d_input, d_output, n);
```

**Metal side (Apple's equivalent):**

```metal
// Metal kernel: element-wise ReLU
kernel void relu_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint idx                   [[thread_position_in_grid]]
) {
    output[idx] = max(0.0f, input[idx]);
}
```

Different syntax, different launch conventions, same parallel structure: one thread per element, each thread reads one input value, writes one output value.

MLX's built-in operations have pre-written Metal kernels. When a CUDA model uses a custom operation that MLX doesn't have, you have three options:
1. Decompose it into MLX primitives (most common for porting work)
2. Write a Metal kernel and register it with MLX
3. Accept lower performance with a CPU fallback

### Debugging GPU memory in MLX

**Checking memory usage:**

```python
import mlx.core as mx

# Force evaluation to get accurate memory figures
mx.eval(model_output)

# Check peak memory
mem = mx.metal.get_peak_memory() / (1024 ** 3)
print(f"Peak GPU memory: {mem:.2f} GB")

# Reset peak counter (useful between benchmark runs)
mx.metal.reset_peak_memory()
```

**When you hit memory limits:**

On NVIDIA, you get a hard error: `CUDA out of memory. Tried to allocate X.XX GiB.` Execution stops immediately.

On Apple Silicon, MLX will attempt to allocate from the unified memory pool. If the system is under memory pressure, macOS will page inactive memory to disk (swap). Symptoms: your inference run doesn't crash, but it becomes 10-100x slower, your Mac's fans spin up, and Activity Monitor shows swap usage climbing.

If this happens: quantize the model, reduce the context length or resolution, or close other memory-heavy applications.

### The porting mental model

When you port LTX-Video or [Matrix](../glossary.md#matrix)-Game from CUDA to MLX, here is the full picture of what is happening at the hardware level:

```
ORIGINAL (NVIDIA)                    PORTED (Apple Silicon)
+---------------------------+        +---------------------------+
| 1. Weights in system RAM  |        | 1. Weights in unified RAM |
|    (CPU memory)           |        |    (shared with GPU)      |
|         |                 |        |                           |
| 2. Copy to VRAM via PCIe  |        | 2. (no copy needed)       |
|    [latency: 100-500ms    |        |                           |
|     for large models]     |        |                           |
|         |                 |        |                           |
| 3. CUDA kernels run on    |        | 3. Metal kernels run on   |
|    NVIDIA GPU             |        |    Apple GPU              |
|    (from VRAM)            |        |    (from unified memory)  |
|         |                 |        |                           |
| 4. Copy results to RAM    |        | 4. (no copy needed)       |
|    via PCIe               |        |    CPU reads directly     |
+---------------------------+        +---------------------------+
Framework: PyTorch + CUDA            Framework: MLX + Metal
```

The weight-loading step is where Apple Silicon often feels faster than expected: there is no GPU upload step. Weights are loaded from disk into memory once, and the GPU is immediately ready to use them.


## Key Terms

| Term | Definition |
|------|------------|
| **GPU** | Graphics Processing Unit; a processor with thousands of simple cores designed to run the same operation on many data elements simultaneously. Originally for rendering graphics, now essential for ML. |
| **CPU** | Central Processing Unit; the main processor with a small number of powerful, general-purpose cores optimized for sequential, logic-heavy work. |
| **Kernel** | A function compiled to run directly on GPU hardware, executed by thousands of threads simultaneously. Every ML operation (matmul, ReLU, attention) is backed by one or more kernels. |
| **Thread** | The smallest unit of GPU execution. One thread runs the kernel code on one element (or one chunk) of data. Thousands run simultaneously. |
| **CUDA** | Compute Unified Device Architecture. NVIDIA's parallel computing platform and programming model for their GPUs. The dominant ML computing environment for research and production. |
| **Metal** | Apple's GPU programming API and shading language. The equivalent of CUDA for Apple GPUs. MLX abstracts over Metal so you don't need to write Metal kernels for standard operations. |
| **VRAM** | Video RAM. Dedicated high-speed memory on discrete NVIDIA GPUs, separate from system RAM. Typically 8-80 GB. The hard limit for model size on NVIDIA hardware. |
| **[Unified Memory](../glossary.md#unified-memory)** | Apple Silicon's memory architecture where CPU and GPU share a single physical memory pool. No data copying between CPU and GPU. The GPU can use all available system RAM. |
| **[Memory Bandwidth](../glossary.md#memory-bandwidth)** | The rate at which data can be moved between memory and the GPU's compute units, measured in GB/s. Often the real performance bottleneck for ML operations, not raw compute speed. |
| **[Parallelism](../glossary.md#parallelism)** | Doing many computations at the same time. GPU parallelism means thousands of identical simple operations on different data elements, all running simultaneously. |
| **[Shared Memory](../glossary.md#shared-memory)** | Fast on-chip memory (per block of threads) used to stage data that multiple threads need. Accessing shared memory is ~100x faster than global memory. Key to writing fast GPU kernels. |
| **[Lazy Evaluation](../glossary.md#lazy-evaluation)** | MLX's execution model: operations are not run immediately when called. Instead, MLX builds a computation graph and defers execution until `mx.eval()` is called (or output is required). Enables kernel fusion. |
| **[Kernel Fusion](../glossary.md#kernel-fusion)** | Combining multiple GPU operations into a single kernel to reduce memory bandwidth. Instead of reading/writing intermediate results to global memory between operations, fused kernels keep intermediate values in fast registers. |


## Sources

- NVIDIA Corporation. *CUDA C++ Programming Guide*. The definitive reference for CUDA programming: thread hierarchy, memory model, kernel writing. Available at [docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- Apple Developer Documentation. *Metal*. Apple's GPU API reference and programming guide, including Metal Shading Language. Available at [developer.apple.com/metal/](https://developer.apple.com/metal/)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html) -- covers unified memory implications, lazy evaluation, and `mx.eval` in the MLX-specific context
- Kirk, D. B., & Hwu, W. W. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th ed.). Morgan Kaufmann. The standard graduate-level text for GPU programming. Covers thread hierarchy, memory hierarchy, and kernel optimization with concrete examples.


## See Also

- [Tensors](01-tensors.md) -- prerequisite; GPU computing is fundamentally about applying tensor operations in parallel. The shapes, dtypes, and operations from that page are what GPU kernels operate on.
- [Transformers](05-transformers.md) -- the architecture that makes up LTX-Video and Matrix-Game; understanding GPU computing explains why attention mechanisms are so expensive and what hardware optimizations (Flash Attention, etc.) actually do.
- [CUDA vs MLX Overview](../02-ecosystem/01-cuda-vs-mlx-overview.md) -- a direct comparison of the two ecosystems from a practical porting perspective.
