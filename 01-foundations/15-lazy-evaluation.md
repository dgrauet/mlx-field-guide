# Lazy Evaluation

> **One-liner:** In MLX, calling an operation only *records* it in a graph; the work happens later, when something forces [evaluation](../glossary.md#lazy-evaluation) -- and where that happens in your code decides your memory peak, your timings, and whether `mx.compile` can speed anything up.

## The Intuition

PyTorch is **eager**: `y = x @ w` computes `y` on the spot, and the next line can use it. MLX is **lazy**: `y = x @ w` returns an array that *describes* a computation ("the matmul of these two arrays") without running it. Keep writing operations and MLX keeps extending that description into a graph. Nothing touches the GPU until you ask for a result.

Think of it as the difference between a cook who prepares each ingredient the moment it is named, and one who writes down the whole recipe first and cooks it in one go when someone asks for the dish. The second cook can plan better -- but if you peek into the pan before asking for the dish, it is empty.

```
EAGER (PyTorch)                        LAZY (MLX)

  y = x @ w    -> GPU runs matmul         y = x @ w    -> graph: matmul(x, w)
  z = relu(y)  -> GPU runs relu           z = relu(y)  -> graph: relu(matmul(x, w))
  print(z)     -> already computed        print(z)     -> NOW the GPU runs both
```

[Tensors](01-tensors.md) and [GPU Computing](04-gpu-computing.md) introduced `mx.eval()` as the way to force execution. This page covers the whole model: what triggers evaluation, what laziness buys you and what it doesn't, and the port bugs that come from putting evaluation in the wrong place.

---

## How It Actually Works

### Building the graph

Every MLX operation returns an `mx.array` that carries its inputs and the primitive that produces it. An unevaluated array knows its **shape and dtype** (MLX infers them without computing anything), but has no data yet:

```python
import mlx.core as mx

x = mx.random.normal((4096, 4096))
y = mx.maximum(x * 2 + 1, 0)

y.shape   # (4096, 4096) -- known immediately, no computation
y.dtype   # float32      -- known immediately
# y's data does not exist yet
```

This is why shape errors in MLX still surface on the line that causes them: shape inference is eager, only the arithmetic is deferred.

### What forces evaluation

Evaluation happens **explicitly** when you call `mx.eval(...)`, or **implicitly** whenever Python needs actual values. In MLX 0.31, all of these trigger evaluation:

```
EXPLICIT
  mx.eval(a, b, ...)        evaluate and block until done
                            (accepts arrays, lists, dicts -- e.g. model.parameters())
  mx.async_eval(a, ...)     start evaluation, return immediately

IMPLICIT (Python needs the numbers)
  print(a) / repr(a)
  a.item(), a.tolist()
  np.array(a), np.asarray(a), memoryview(a)
  if a.sum() > 0: ...       converting to a Python bool
  mx.save / mx.save_safetensors(...)
```

Implicit evaluation is convenient but **hides synchronization points**. A value-dependent `if` in the middle of a loop, or a `.item()` used for logging, forces the GPU to finish everything pending before Python can continue. Nothing is wrong numerically -- it is just slower, and the stall is invisible in the code.

### What laziness buys -- and what it doesn't

Laziness lets MLX skip work nobody asked for (an unused branch of the graph is never computed) and lets it submit a whole batch of pending operations to the GPU at once instead of one at a time.

What laziness does **not** do on its own is **fuse** operations into a single kernel. Without `mx.compile`, each operation in the graph still runs as its own [kernel](../glossary.md#kernel), writing its intermediate result to memory. Fusion -- merging a chain of element-wise operations into one kernel whose intermediates stay in registers -- is what `mx.compile` adds.

Measured on MLX 0.31.1 (Apple Silicon), a chain of 40 element-wise operations on a 4096×4096 array:

```
                         time per call
  lazy, not compiled        29.1 ms     (40 separate kernels)
  mx.compile(f)              3.3 ms     (element-wise chain fused)
```

A ~9x difference, from one decorator. The pattern to remember: **lazy = deferred, compiled = fused.**

### `mx.compile`: tracing a function into a fused graph

`mx.compile(f)` traces `f` once with placeholder inputs, optimizes the resulting graph (fusing element-wise chains, removing redundant work), and reuses that plan on later calls:

```python
@mx.compile
def gelu_ish(x):
    return 0.5 * x * (1 + mx.tanh(0.79788456 * (x + 0.044715 * x**3)))

y = gelu_ish(x)   # first call: trace + compile
y = gelu_ish(x)   # later calls: reuse the compiled graph
```

Three rules govern it:

1. **It re-traces when input shapes (or dtypes) change.** Calling a compiled function with shapes 10, 10, 20, 30 traced it three times in our test. For variable shapes (e.g. a growing sequence), pass `mx.compile(f, shapeless=True)` -- traced once -- but only if the function's logic truly doesn't depend on the shape.
2. **No value-dependent Python control flow.** Inside a compiled function, `if x.sum() > 0:` needs a concrete value, which does not exist during tracing. MLX raises `ValueError: [eval] Attempting to eval an array during function transformations like compile or vmap is not allowed.` Replace value-dependent branches with `mx.where`.
3. **Hidden state must be declared.** A compiled function that reads or updates state outside its arguments (model parameters being trained, optimizer state, the random generator) needs that state passed through `inputs=` / `outputs=`, otherwise the traced graph bakes in stale values. See the [MLX compile docs](https://ml-explore.github.io/mlx/build/html/usage/compile.html) for the pattern.

---

## Why It Matters for MLX

### Where you evaluate decides your memory peak

A lazy graph keeps every intermediate it still needs alive until it runs. If you let a long chain of work accumulate and evaluate it all at the end, MLX may schedule it so that many intermediates coexist. Evaluating at natural boundaries keeps the peak flat.

Measured on MLX 0.31.1: accumulating `s = s + x * i` over 64 MB arrays:

```
  steps   one mx.eval at the end   mx.eval every step
    8          671 MB                  268 MB
   32         1007 MB                  268 MB
```

The deferred version grows with the number of steps; the per-step version doesn't. This is the lazy-evaluation version of a memory leak -- nothing is actually leaked, the graph just holds on to more than you expected.

The standard evaluation points, all visible in `mlx-lm` and `mlx-examples`:

```
WHERE TO PUT mx.eval

  Training loop     once per step:  mx.eval(model.parameters(), optimizer.state)
                    (without it, each step extends the previous step's graph)
  Diffusion loop    once per denoising step: mx.eval(latents)
  LLM generation    once per token (mlx-lm uses mx.async_eval to overlap
                    the next token's graph building with the current GPU work)
  Large model load  after loading / converting weights, before the first forward
  Benchmarks        before starting the timer AND before stopping it
```

Too few evaluations and the graph (and memory) balloons; too many -- say, inside every layer -- and you pay a GPU round trip each time and lose the batching benefit. Once per step of an outer loop is almost always right.

### Symptoms and causes

```
LAZY-EVALUATION PORT FAILURES

  Symptom                                   Likely cause
  -------                                   ------------
  Benchmark shows ~0 ms                     timer stopped before evaluation;
                                              you measured graph building only
                                              (see Tooling: timing MLX code)
  Memory climbs step after step,            no mx.eval inside the loop; each
    then OOM                                  step extends one giant graph
  Loop gets slower every iteration          same cause: graph construction
                                              cost grows with graph size
  Unexpectedly slow, GPU underused          hidden sync points: .item(),
                                              value-dependent `if`, logging
                                              inside the hot loop
  ValueError "[eval] ... during function    value-dependent control flow or
    transformations like compile"             .item() inside an mx.compile'd fn
  Compiled function slower than expected    recompiling on every call because
                                              the input shape keeps changing
  Compiled training step doesn't learn      model/optimizer state not passed
                                              via inputs=/outputs=
  Array looks "empty" in the debugger       it's unevaluated; call mx.eval(x)
                                              before inspecting
```

### Porting checklist

1. **Decide your evaluation points before debugging anything else.** Put one `mx.eval` per iteration of the outermost loop (training step, denoising step, generated token). Most lazy-eval bugs disappear with that single change.
2. **Remove sync points from hot loops.** Collect scalars (losses, metrics) as arrays and convert them to Python numbers once per step, not per layer.
3. **Compile the hot, shape-stable functions.** Element-wise-heavy code (activations, norms written from primitives, RoPE, sampling math) benefits most. Use `mlx.fast` primitives where they exist -- they are already fused kernels.
4. **Time correctly.** `mx.eval` inputs before starting the clock and outputs before stopping it; warm up once so compilation isn't timed.

---

## Key Terms

| Term | Definition |
|------|------------|
| **Lazy evaluation** | Execution model where operations build a graph instead of running immediately; the graph runs when evaluation is forced. |
| **Eager execution** | Execution model where each operation runs as soon as it is called (PyTorch's default). |
| **Computation graph** | The recorded chain of pending operations and their inputs that an unevaluated array represents. |
| **`mx.eval`** | Forces evaluation of the given arrays (or nested lists/dicts of arrays) and blocks until the results exist. |
| **`mx.async_eval`** | Starts evaluation without blocking, so Python can build the next graph while the GPU works. |
| **Implicit evaluation** | Evaluation triggered by Python needing values: `print`, `.item()`, `np.array`, bool conversion, saving to disk. |
| **Sync point** | A place where Python must wait for the GPU to finish all pending work; implicit evaluations create hidden ones. |
| **Kernel fusion** | Merging a chain of operations into one GPU kernel so intermediates stay in registers; in MLX this comes from `mx.compile`, not from laziness alone. |
| **`mx.compile`** | Traces a function into an optimized, fused graph that is reused across calls with the same input shapes. |
| **Shapeless compilation** | `mx.compile(f, shapeless=True)`: trace once and reuse the graph for any input shape. |

---

## Sources

- MLX documentation, "Lazy Evaluation": [ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- MLX documentation, "Compilation": [ml-explore.github.io/mlx/build/html/usage/compile.html](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- MLX API reference, `mlx.core.eval` / `mlx.core.async_eval` / `mlx.core.compile`: [ml-explore.github.io/mlx/build/html/python/transforms.html](https://ml-explore.github.io/mlx/build/html/python/transforms.html)
- `mlx-lm` generation loop (use of `mx.async_eval`): [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- Timings and memory figures on this page: measured with MLX 0.31.1 on Apple Silicon; absolute numbers vary by chip, the ratios are what matter.

---

## See Also

- [Tensors](01-tensors.md) -- prerequisite; first introduction to `mx.eval`
- [GPU Computing](04-gpu-computing.md) -- kernels, memory bandwidth, and why fusion matters
- [Numerical Stability](12-numerical-stability.md) -- the other family of "runs fine, wrong result" bugs
- [Frameworks](../02-ecosystem/02-frameworks.md) -- MLX vs PyTorch vs JAX execution models (`torch.compile`, `jax.jit`, `mx.compile`)
- [Tooling](../02-ecosystem/08-tooling.md) -- timing and profiling MLX code correctly
- [Porting Guide](../03-contributing/02-porting-guide.md) -- performance and memory management when porting a model
