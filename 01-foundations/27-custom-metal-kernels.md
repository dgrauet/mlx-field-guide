# Custom Metal Kernels

> **One-liner:** When an operation can't be expressed efficiently with MLX's built-in ops -- a rasterizer, a custom sampling kernel, an unusual gather -- `mx.fast.metal_kernel` lets you write the GPU code yourself in a few lines of Metal and call it like any MLX function. It's powerful and easy to get subtly wrong: the grid means threads (not blocks), outputs start uninitialized, non-contiguous inputs need care, and every template value is a new compile.

## The Intuition

MLX ships hundreds of [kernels](../glossary.md#kernel) -- matmuls, convolutions, `mx.fast.scaled_dot_product_attention`, reductions -- and `mx.compile` fuses chains of element-wise ops into new ones automatically ([Lazy Evaluation](15-lazy-evaluation.md)). Almost everything a model needs is covered.

What's left are operations whose structure the built-ins can't express without huge intermediates or Python loops: rasterizing triangles into pixels, a histogram, a custom scatter, a sparse attention pattern. In CUDA you'd write a `.cu` kernel. In MLX you write the **body** of a Metal function as a string, and MLX generates the signature, compiles it on first use, and runs it on MLX arrays:

```python
import mlx.core as mx

kernel = mx.fast.metal_kernel(
    name="gelu_tanh",
    input_names=["inp"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        T v = inp[i];
        out[i] = T(0.5) * v * (T(1) + metal::precise::tanh(T(0.7978845608) * (v + T(0.044715) * v * v * v)));
    """,
)

y = kernel(
    inputs=[x],
    template=[("T", mx.float32)],
    grid=(x.size, 1, 1),            # total number of THREADS
    threadgroup=(256, 1, 1),        # threads per threadgroup
    output_shapes=[x.shape],
    output_dtypes=[x.dtype],
)[0]
```

---

## How It Actually Works

### What MLX generates

From the [custom kernels documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html): MLX builds a templated `[[kernel]]` function whose parameters are your inputs (`const device T* inp`), your outputs (`device T* out`), and any Metal attributes your source uses (`thread_position_in_grid`, `threadgroup_position_in_grid`, `simdgroup_index_in_threadgroup`, ...). If the source mentions `inp_shape`, `inp_strides` or `inp_ndim`, those are passed too. `verbose=True` prints the generated code. `header=` holds helper functions placed before the kernel.

### Is it worth it? Measure against `mx.compile` first

On 16M floats, MLX 0.32.3, M2 Pro:

```
  tanh-GELU, written with MLX ops        8.48 ms
  same function under mx.compile         1.14 ms
  hand-written metal_kernel              1.15 ms    (max |diff| 4.8e-7)

  row sums of a 4096 x 4096 matrix
  mx.sum(axis=1)                          0.60 ms
  hand-written simd_sum kernel            0.58 ms    (max |diff| 4.2e-5)
```

For element-wise code, `mx.compile` already reaches what a hand-written kernel does. For standard reductions, the built-ins are already tuned. A custom kernel pays off when it **changes the algorithm**: keeping data in threadgroup memory, skipping work, or avoiding an intermediate that MLX ops would have to materialize. The rasterizer in [mlx-arsenal](https://github.com/dgrauet/mlx-arsenal) (used by the Hunyuan3D MLX port) is a good example. One threadgroup per 16×16 pixel tile walks that tile's list of triangles in chunks of 256, loaded cooperatively into threadgroup memory, testing coverage with exact integer edge functions. No combination of MLX ops expresses that.

### `grid` counts threads, not threadgroups

Metal's `dispatchThreads` takes the **total number of threads**. CUDA launches take a number of **blocks**. Bringing the CUDA habit over silently runs a fraction of the work:

```
  grid=(n, 1, 1),       threadgroup=(256, 1, 1)    all n outputs written
  grid=(n // 256, 1, 1), threadgroup=(256, 1, 1)   4 / 1,024 outputs correct (measured)
```

For kernels organized by threadgroup (one group per row, per tile), the grid is `number_of_groups × threads_per_group`, and the source uses `threadgroup_position_in_grid` to find its group. That's how the row-sum kernel above is launched: `grid=(4096 * 256, 1, 1)`.

### Outputs start uninitialized

Output buffers come from MLX's allocator, which reuses freed memory ([Memory & Metal Limits](26-memory-metal-limits.md)). A kernel that *accumulates* into its output (`+=`, atomics) must set `init_value`. Measured with an atomic histogram of 1M values, called three times in a row without `init_value`:

```
  call 1: total 1,048,576    (correct, fresh memory happened to be zero)
  call 2: total 2,097,152    (reused the previous output buffer)
  call 3: total 3,145,728
  with init_value=0: 1,048,576 every time
```

The first call passing is what makes this bug survive testing.

### Concurrent writes need atomics

Threads run in parallel, so two threads incrementing the same output element race. The same histogram without atomics:

```
  out[inp[i]] += 1;                                      total 2,792 of 1,048,576
  atomic_fetch_add_explicit(&out[inp[i]], 1,             total 1,048,576, exact
                            memory_order_relaxed);
    (with atomic_outputs=True, which declares the outputs as device atomic<T>*)
```

### Non-contiguous inputs

By default (`ensure_row_contiguous=True`) MLX copies inputs to row-contiguous memory before the kernel runs, so `inp[i]` is the i-th element in row-major order. Turning that off to avoid the copy means the kernel must follow the strides itself:

```
  transposed 512 x 512 input, ensure_row_contiguous=False
    out[i] = inp[i] * 2                         max |diff| 13.4   (reads the wrong elements)
    loc = elem_to_loc(i, inp_shape, inp_strides, inp_ndim);
    out[i] = inp[loc] * 2                       exact
  default (copy first)                          exact
```

Keep the default unless the copy shows up in a profile.

### Compilation: template values are compile-time

The first call compiles the kernel (61.7 ms for the GELU above, then 0.78 ms per call). Compiled kernels are cached by MLX, so even re-creating the same `metal_kernel` object every call cost only 0.37 ms per call. But **every distinct `template` value is a separate compile**:

```
  20 calls with 20 different scale values
    scale as a template parameter   ("SCALE", s)       54.0 ms per call: one compile each
    scale as a 1-element input array                   3.33 ms per call (one compile, amortized)
```

Use templates for things that really are fixed (dtype, tile size). Pass values that change between calls (sizes, thresholds, scalars) as small input arrays. mlx-arsenal does this with `dims` and `fparams` arrays. Create kernel objects once at module level (mlx-arsenal caches them in a `_get_kernel()` function) to keep the call path cheap.

### Precision: math modes

Kernels compile with `math_mode="safe"` by default, so special values follow IEEE rules (`exp(-inf) == 0`, which masked softmax relies on). `compile_options={"math_mode": "fast"}` or `"relaxed"` can be faster but breaks those guarantees. Use `metal::precise::` functions when matching a reference closely matters.

### Gradients

A `metal_kernel` has no automatic gradient. Wrap it in `@mx.custom_function` and define its `.vjp`, typically with a second kernel. The MLX documentation's `grid_sample` example does exactly that, with atomic outputs for the backward scatter.

---

## Why It Matters for MLX

### Port failures

```
CUSTOM KERNEL FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  Only the first few hundred outputs are     grid given in threadgroups (CUDA
    right, the rest zero or garbage            habit) instead of total threads
  Accumulated outputs grow on every call     no init_value: output buffer reused
    (correct the first time)                   from MLX's cache
  Counts / sums too low, vary between runs   concurrent writes without atomics
  Wrong results only for transposed or       ensure_row_contiguous=False without
    sliced inputs                              elem_to_loc
  Each call takes tens of ms                 a changing value passed as a
                                               template parameter: one compile per
                                               value
  NaN where the reference has 0              math_mode "fast"/"relaxed" breaking
                                               exp(-inf) == 0
  "no gradient" / wrong gradients            no custom_function + vjp
  Kernel slower than MLX ops                 element-wise work: mx.compile already
                                               fuses it; reductions: built-ins are tuned
```

### Checklist

1. **Try `mx.compile` and the `mx.fast` primitives first.** Write a kernel only if it changes the algorithm.
2. **Keep a reference implementation** in MLX ops or NumPy, and compare outputs on random inputs, including non-contiguous ones and edge sizes (not a multiple of the threadgroup size).
3. **`grid` = total threads**; guard the tail (`if (i >= n) return;`) when the grid is rounded up.
4. **Set `init_value`** for any output the kernel accumulates into; use atomics for shared writes.
5. **Templates for constants, arrays for values**; create the kernel once.
6. **Profile with a Metal capture** (`mx.metal.start_capture`, see [Memory & Metal Limits](26-memory-metal-limits.md#inspecting-what-the-gpu-did)) when performance matters.

---

## Key Terms

| Term | Definition |
|------|------------|
| **`mx.fast.metal_kernel`** | MLX API that JIT-compiles a Metal kernel from the body of a function and calls it on MLX arrays. |
| **Grid** | In Metal (and `metal_kernel`), the total number of threads launched, in up to three dimensions. |
| **Threadgroup** | A group of threads that run together and can share fast threadgroup memory and synchronize with barriers; Metal's CUDA-block equivalent. |
| **SIMD group** | 32 threads executing in lockstep; `simd_sum` and friends reduce across them without shared memory. |
| **Threadgroup memory** | Fast on-chip memory shared by one threadgroup, used to load data cooperatively. |
| **Atomic output** | An output declared `device atomic<T>*` (`atomic_outputs=True`) so concurrent threads can update it safely. |
| **init_value** | Value used to initialize outputs before the kernel runs; without it, outputs hold whatever the reused buffer contained. |
| **elem_to_loc** | Helper that converts a row-major element index to a memory offset using the input's shape and strides. |
| **Math mode** | Metal compile option (`safe`, `relaxed`, `fast`) trading IEEE special-value guarantees for speed. |

---

## Sources

- MLX documentation, "Custom Metal Kernels" (signature generation, `ensure_row_contiguous`, `elem_to_loc`, math modes, `custom_function` + `grid_sample` example): [ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- MLX `mx.fast.metal_kernel` docstring (v0.32.3).
- Apple, Metal Shading Language Specification (threads, threadgroups, SIMD groups, atomics): [developer.apple.com/metal/Metal-Shading-Language-Specification.pdf](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- mlx-arsenal rasterizer (`rasterize/_tile_raster.py`, `_binning.py`): tile/threadgroup design, cached kernels, parameters as arrays. [github.com/dgrauet/mlx-arsenal](https://github.com/dgrauet/mlx-arsenal)
- Measurements on this page: MLX 0.32.3, Apple M2 Pro.

---

## See Also

- [GPU Computing](04-gpu-computing.md) -- kernels, threads and memory bandwidth
- [Lazy Evaluation](15-lazy-evaluation.md) -- `mx.compile`, the first thing to try
- [Apple Silicon Memory & Metal Limits](26-memory-metal-limits.md) -- the allocator cache behind uninitialized outputs, command buffers, Metal captures
- [Porting Guide](../03-contributing/02-porting-guide.md) -- handling CUDA kernels when porting
- [3D Representations](20-3d-representations.md) -- the rasterization that motivated mlx-arsenal's kernels
