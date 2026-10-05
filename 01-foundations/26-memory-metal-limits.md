# Apple Silicon Memory & Metal Limits

> **One-liner:** On a Mac, the CPU and GPU share one pool of memory, but the GPU can only use part of it, MLX keeps freed memory in a cache, and macOS kills any GPU command buffer that runs too long. Knowing these limits -- and the MLX knobs around them -- explains most of the out-of-memory errors, "memory leaks" and mysterious GPU crashes that big models hit on Apple Silicon.

## The Intuition

On a CUDA machine, the limit is obvious: the GPU has 24 or 80 GB of VRAM, and that's that. On Apple Silicon there is no separate VRAM ([GPU Computing](04-gpu-computing.md)), so it's tempting to think a 32 GB Mac gives the GPU 32 GB. It doesn't, quite:

```
  32 GB unified memory (M2 Pro)

  |<------------------------------- 34.4 GB (memory_size) -------------------------->|
  |<----------- 26.8 GB: GPU working set macOS recommends (78%) ----------->|         |
  |   model weights + KV cache + activations + MLX's buffer cache           |  macOS, |
  |                                                                         |  apps,  |
  |                                                                         |  CPU    |

  + any single GPU buffer is capped at 20.1 GB (max_buffer_length)
  + any single command buffer must finish in ~10 s (the GPU watchdog)
```

Values from `mx.device_info()` on an M2 Pro 32 GB, MLX 0.32.3. A model whose weights plus activations exceed the GPU working set doesn't fail cleanly: macOS starts swapping, everything slows to a crawl, and eventually something gets killed.

---

## How It Actually Works

### What MLX reports

```python
import mlx.core as mx

mx.device_info()          # memory_size, max_recommended_working_set_size,
                          # max_buffer_length, architecture, ...
mx.get_active_memory()    # bytes held by live arrays
mx.get_peak_memory()      # high-water mark since the last mx.reset_peak_memory()
mx.get_cache_memory()     # bytes freed by arrays but kept by MLX for reuse
```

**Resident memory (RSS) doesn't see Metal allocations.** Measured with MLX 0.32.3 while a 4.3 GB array was alive: `ps -o rss` (and therefore `psutil`'s RSS) reported **31 MB**, while `top`'s MEM column and `footprint <pid>` reported **4,112 MB**, listing the array as `IOAccelerator (graphics)`. Use MLX's own counters, `footprint`, `top` or Activity Monitor (which report the physical footprint), never RSS.

### The buffer cache: why memory doesn't go down

When an array is freed, MLX keeps its buffer in a **cache** to reuse it for the next allocation of a similar size. That avoids the cost of asking Metal for memory again, but it means freed memory isn't returned to the system:

```
  after allocating a 4.3 GB array        active 4.29 GB   cache 0.00 GB
  after `del` of that array              active 0.00 GB   cache 4.29 GB   <- still held
  after mx.clear_cache()                 active 0.00 GB   cache 0.00 GB
  with mx.set_cache_limit(0), del        active 0.00 GB   cache 0.00 GB   (freed immediately)
```

A pipeline that loads a text encoder, uses it, deletes it, then loads a large diffusion model can run out of memory **only because the encoder's memory is still in the cache**. Call `mx.clear_cache()` after dropping a large component. The Matrix-Game MLX port, for example, releases its T5 text encoder and calls `mx.clear_cache()` before loading the DiT and VAE. For tight pipelines, cap the cache with `mx.set_cache_limit(bytes)`.

### Memory limit and wired limit

```
  mx.set_memory_limit(bytes)   a guideline for graph evaluation, not a hard cap.
                                 Default on this machine: 32.6 GB (95% of RAM).
                                 Tested: with a 2 GB limit, a 4.3 GB array was still
                                 allocated. Allocation only fails when RAM (and swap)
                                 run out.

  mx.set_wired_limit(bytes)    keeps up to `bytes` of MLX memory wired (resident,
                                 never swapped). macOS 15+. Default 0. Cannot exceed
                                 the system wired limit: on this machine,
                                 set_wired_limit(32 GB) raised "Setting a wired limit
                                 larger than the maximum working set size is not
                                 allowed".
```

The **system** limit is `max_recommended_working_set_size` (26.8 GB here, about 78% of RAM). It can be raised with `sudo sysctl iogpu.wired_limit_mb=<MB>` (the setting reads `0`, meaning "default", until changed, and resets at reboot). `mlx-lm` wires the model's weights during generation (its `wired_limit` context manager in `generate.py`) for steadier decode speed. Raising the system limit lets a larger model stay resident, at the expense of memory for everything else.

**Single buffers have their own cap.** One array larger than `max_buffer_length` (20.1 GB here) can't be allocated, whatever the total memory: `RuntimeError: [metal::malloc] Attempting to allocate 20102545408 bytes which is greater than the maximum allowed buffer size of 20100448256 bytes.` A huge embedding table or one giant KV-cache tensor may need splitting.

### Command buffers and the GPU watchdog

MLX sends work to the GPU in **command buffers**, each holding a batch of kernels. It commits a buffer every `max_ops_per_buffer` operations or `max_mb_per_buffer` MB of inputs. In MLX 0.32.3 (`backend/metal/device.cpp`) that is 40/40 on base and Pro chips, 50/50 on Max and Ultra, 20/40 on phones, overridable with the `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` environment variables.

Recorded with [smeltr](https://github.com/dgrauet/smeltr) (a Metal/MLX observability tool) on a 1,200-op element-wise chain evaluated with one `mx.eval`:

```
  MLX_MAX_OPS_PER_BUFFER    command buffers committed
         400                         4
     default (40)                   38
          10                       210
```

So even a long lazy graph is split into many buffers. The danger is a **single buffer that takes too long**. macOS's GPU watchdog evicts work that keeps the GPU busy for roughly 10 seconds, to protect the display. The process then dies with errors like `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` or `MTLCommandBufferErrorInternal` (code 14), as documented in the LTX-2 MLX port. It happens with:

- a few very large kernels in one buffer: a huge matmul, attention over tens of thousands of tokens, a full-resolution VAE decode;
- many buffers queued behind other GPU users (Spotlight indexing, other apps), so each waits past the deadline;
- slower Macs: the same pipeline can be fine on an Ultra and crash on a laptop.

The fixes all make each unit of GPU work smaller:

```
  insert mx.eval between stages     LTX-2: after every Gemma layer and every 8 DiT
                                      blocks (LTX2_GEMMA_EVAL_EVERY, LTX2_DIT_EVAL_EVERY)
                                      -> ~1-2 s per command buffer
  tile the big operation            Hunyuan3D: UV rasterization in tiles so each
                                      Metal dispatch stays within the budget;
                                      tiled VAE decoding (see VAEs in Depth)
  chunk long sequences              prefill_step_size for LLM prompts
  lower MLX_MAX_OPS_PER_BUFFER /    finer buffers; costs some throughput
    MLX_MAX_MB_PER_BUFFER
```

The LTX-2 port also documents an AGX driver variable, `AGX_RELAX_CDM_CTXSTORE_TIMEOUT`, that relaxes the timeout for one process as a workaround for a macOS 26.x / MLX 0.31.x regression. The UI may stutter while it's set. Prefer splitting the work.

---

## Why It Matters for MLX

### Symptoms and causes

```
MEMORY & METAL-LIMIT FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  Mac becomes unresponsive, generation       working set above the GPU's share:
    slows to a crawl, then OOM                 swapping. Quantize, tile, chunk,
                                               free components, or raise
                                               iogpu.wired_limit_mb with care
  "Memory leak": memory never goes down      MLX's buffer cache; mx.clear_cache()
    after deleting a model                     or mx.set_cache_limit
  ps / psutil RSS shows ~0 GB for a big      Metal memory isn't in RSS: use
    model                                      mx.get_active_memory, footprint, top
  [metal::malloc] ... greater than the       one array above max_buffer_length:
    maximum allowed buffer size                split it
  set_wired_limit raises ValueError          above the system wired limit
                                               (max_recommended_working_set_size)
  kIOGPUCommandBufferCallbackError...        GPU watchdog: one command buffer ran
    ImpactingInteractivity, or                 too long. Add mx.eval between stages,
    MTLCommandBufferErrorInternal (14)         tile, chunk
  Crash only on some Macs / only when        watchdog margin: slower GPU, or
    other apps are busy                        contention from other GPU work
  Peak memory far above weights + cache      lazy graph keeps intermediates alive:
                                               evaluate per step (see Lazy Evaluation)
```

### Inspecting what the GPU did

- **`mx.metal.start_capture("run.gputrace")` / `mx.metal.stop_capture()`** (run with `MTL_CAPTURE_ENABLED=1`) records a Metal capture that opens in Xcode's Metal debugger: every dispatch, its duration, its buffers. Keep captures short: they are expensive.
- **Xcode Instruments, Metal System Trace**: a timeline of command buffers on the GPU (see [Tooling](../02-ecosystem/08-tooling.md)).
- **smeltr** records command-buffer history, memory and crash reports for a whole run (`smeltr record -- python script.py`) and lets you query them afterwards. It's how the command-buffer counts above were obtained.

### Checklist for a big pipeline on a Mac

1. **Budget against `max_recommended_working_set_size`**, not total RAM: weights + KV cache or largest activation + some headroom.
2. **Load components one at a time** and `mx.clear_cache()` after freeing each.
3. **Evaluate at stage boundaries** (per denoising step, per few blocks) for memory and for the watchdog.
4. **Tile or chunk the few giant operations** (VAE decode, long-sequence attention, rasterization).
5. **Test on the smallest Mac you support**: the watchdog and memory limits bite there first.

---

## Key Terms

| Term | Definition |
|------|------------|
| **Unified memory** | One memory pool shared by CPU and GPU on Apple Silicon; no separate VRAM. |
| **GPU working set (max recommended)** | The share of RAM macOS recommends the GPU use (about 78% on a 32 GB M2 Pro); the default system wired limit. |
| **Wired memory** | Memory locked resident (never swapped); MLX can wire model weights up to the system wired limit (`iogpu.wired_limit_mb`). |
| **Buffer cache (MLX)** | Freed buffers MLX keeps for reuse; counted by `mx.get_cache_memory()`, released by `mx.clear_cache()`. |
| **Memory limit (MLX)** | A guideline for graph evaluation set by `mx.set_memory_limit`; not a hard allocation cap. |
| **max_buffer_length** | The largest single Metal buffer the device can allocate (20.1 GB on a 32 GB M2 Pro). |
| **Command buffer** | A batch of GPU kernels submitted together; MLX commits one every 40-50 ops or MB by default. |
| **GPU watchdog** | macOS mechanism that evicts GPU work running ~10 s or more, killing the process with an "ImpactingInteractivity" or internal command-buffer error. |

---

## Sources

- MLX v0.32.3 source: `python/src/memory.cpp` (memory, cache and wired limit semantics), `mlx/backend/metal/device.cpp` (command-buffer limits per architecture, `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`): [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
- MLX documentation, "Metal Debugger" (`mx.metal.start_capture`, `MTL_CAPTURE_ENABLED`): [ml-explore.github.io/mlx/build/html/dev/metal_debugger.html](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)
- LTX-2 MLX port, "Metal Watchdog Mitigation" (error names, ~10 s window, eval cadence, `AGX_RELAX_CDM_CTXSTORE_TIMEOUT`): [github.com/dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)
- Hunyuan3D-2.1 MLX port (tiled rasterizer within the command-buffer budget): [github.com/dgrauet/Hunyuan3D-2.1-mlx](https://github.com/dgrauet/Hunyuan3D-2.1-mlx)
- smeltr, Metal/MLX observability for macOS: [github.com/dgrauet/smeltr](https://github.com/dgrauet/smeltr)
- Measurements on this page: MLX 0.32.3, Apple M2 Pro 32 GB, macOS 27.0.

---

## See Also

- [GPU Computing](04-gpu-computing.md) -- unified memory and memory bandwidth
- [Lazy Evaluation](15-lazy-evaluation.md) -- evaluation points, the main lever on memory peaks and command-buffer size
- [KV Cache & Inference Optimization](16-kv-cache-inference.md) -- KV cache sizing and chunked prefill
- [VAEs in Depth](21-vae.md) -- the decoder memory peak and tiling
- [Tooling](../02-ecosystem/08-tooling.md) -- profilers and Instruments
