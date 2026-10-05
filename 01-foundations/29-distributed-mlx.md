# Distributed MLX

> **One-liner:** When one Mac isn't enough -- too little memory for the model, or too slow -- `mx.distributed` spreads the work over several Macs connected by Ethernet or Thunderbolt. The three ways to split a model are data parallelism (copy the model, split the batch), tensor parallelism (split each layer), and pipeline parallelism (split the layers), and MLX ships helpers for all three. Everything can be tested on a single Mac with `mlx.launch -n 2`, but a single Mac can't show a speed-up.

## The Intuition

Distributing a model means running the same program as several **ranks** (processes, usually one per Mac) that exchange arrays through **collective operations**:

```
  all_sum      every rank ends with the sum of everyone's array     (the workhorse)
  all_gather   every rank ends with everyone's arrays concatenated
  all_max/min  elementwise max / min across ranks
  sum_scatter  sum, then each rank keeps one slice
  send / recv  point-to-point, for pipelines
```

Those are the collectives in `mx.distributed` (MLX 0.32.3). Notably there is **no `all_to_all`**, which some CUDA schemes rely on (see below).

How the model is split decides what gets communicated:

```
  DATA PARALLEL           TENSOR PARALLEL             PIPELINE PARALLEL

  rank 0: full model      rank 0: half of every       rank 0: layers 0-23
          batch A                 layer's weights      rank 1: layers 24-47
  rank 1: full model      rank 1: other half
          batch B
                          after each block:           activations sent from
  after backward:           all_sum of the            rank 0 to rank 1
    average gradients       partial outputs           (send / recv)
    (all_sum)

  memory: full model      memory: model / N           memory: model / N
          on each rank
  use: training           use: inference/training     use: inference of models
                            of models too big or         too big for one Mac
                            too slow for one Mac
```

---

## How It Actually Works

### Backends: how the Macs talk

```
  backend   transport                              notes
  -------   ---------                              -----
  ring      TCP over Ethernet or Thunderbolt       works everywhere; the default to start with
  jaccl     RDMA over Thunderbolt 5                macOS 26.2+; latency an order of magnitude
                                                    lower than ring; RDMA must be enabled from
                                                    macOS recovery (`rdma_ctl enable`)
  mpi       MPI (e.g. OpenMPI)                     if you already run an MPI cluster
  nccl      NVIDIA NCCL                            MLX's CUDA backend, not Macs
```

`mx.distributed.init(backend="ring")` creates the global group. `backend="any"` tries the available backends in turn. With `strict=False` (the default), a script run without a launcher gets a group of size 1, so the same code runs on one machine unchanged.

### Launching

```bash
# test on one Mac: 2 local ranks
mlx.launch -n 2 --backend ring my_script.py

# two Macs (SSH access, same Python environment and paths on both)
mlx.launch --hosts mac1,mac2 --backend ring my_script.py

# Thunderbolt setup and diagnostics, writing a hostfile for mlx.launch --hostfile
mlx.distributed_config --hosts mac1,mac2 --over thunderbolt --backend jaccl \
    --auto-setup --output-hostfile hosts.json
```

`mlx.launch` starts the script on every host, forwards their output, broadcasts stdin (so `pdb` works), and kills all ranks if one fails.

### Data parallelism: `nn.average_gradients`

Each rank computes gradients on its own slice of the batch; averaging them gives exactly the gradient of the whole batch. `nn.average_gradients(grads)` does it in a few large `all_sum` calls instead of one per parameter, because many small messages are slow. Measured with 2 local ranks (MLX 0.32.3): the averaged gradient matched the gradient computed on the gathered global batch to **6.0e-8**. After the update, every rank holds identical weights. The vjepa2-mlx port's 2-rank training test checks exactly that (cross-rank parameter difference 0.0).

### Tensor parallelism: `shard_linear`

A transformer MLP is two linear layers. Splitting the first by output columns (`"all-to-sharded"`: every rank gets the full input and computes a slice of the hidden units) and the second by input rows (`"sharded-to-all"`: every rank multiplies its slice, then one `all_sum` adds the partial results) needs **one `all_sum` per MLP**. Attention splits the same way, by heads.

```python
from mlx.nn.layers.distributed import shard_linear
group = mx.distributed.init()

up   = shard_linear(up,   "all-to-sharded", group=group)   # (8192, 2048) -> (4096, 2048) per rank
down = shard_linear(down, "sharded-to-all", group=group)   # (2048, 8192) -> (2048, 4096) per rank
y = down(nn.gelu(up(x)))                                   # all_sum happens inside `down`
```

Measured with 2 local ranks on a 2048 → 8192 → 2048 MLP: output equal to the unsharded MLP to **4.1e-6**, with half the weights per rank. In MLX 0.32.3 this also works for **4-bit quantized layers**: `shard_linear` returns `QuantizedAllToShardedLinear` / `QuantizedShardedToAllLinear`, matching the unsharded quantized MLP to 3.4e-6. Many `mlx-lm` models implement `shard()` with these helpers, and others implement pipeline parallelism (`PipelineMixin`, `pipeline()`).

The Matrix-Game MLX port uses this pattern for its video DiT: 24 attention heads and the FFN split per rank, two `all_sum` per block (~77 MB each at 720p per its README), so the number of ranks must divide both 24 and the FFN width (2, 4 or 8). The reference implementation uses Ulysses sequence parallelism instead, which needs `all_to_all`: absent from `mx.distributed`, hence the switch to head-wise tensor parallelism.

**`nn.fully_shard`** (FSDP-style: each rank stores a shard of the parameters and gathers them for the forward pass) is also available, for training models whose weights and optimizer state don't fit on one Mac.

### What one Mac can and can't tell you

Two ranks on one Mac share one GPU and one memory pool, so they are slower than one process. The same MLP forward took **105.9 ms** with 2 local tensor-parallel ranks vs **83.0 ms** unsharded. A local `all_sum` goes over loopback TCP (1 MB in 0.67 ms, 64 MB in 19.3 ms here), not representative of Thunderbolt or RDMA. Use `-n 2` to check **correctness**: shapes, sharding divisibility, that outputs match the unsharded model. Measure speed only on real hardware.

Whether distribution pays off depends on compute vs communication per step. Matrix-Game's ~150 MB of `all_sum` per block is "a few seconds per clip on Thunderbolt, negligible next to compute" according to its README, so the DiT speeds up almost linearly. An LLM decoding one token at a time does little compute per `all_sum`, so latency (JACCL's advantage) matters more than bandwidth.

---

## Why It Matters for MLX

### Failures

```
DISTRIBUTED FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  Script runs, but world size is 1           not launched with mlx.launch (init with
                                               strict=False silently returns a
                                               single-rank group)
  Hang at the first collective               ranks disagree on the order or number
                                               of collectives (rank-dependent `if`
                                               around a collective), or a firewall /
                                               wrong interface between hosts
  Shape error after sharding                 dimension not divisible by the number of
                                               ranks (heads, FFN width)
  Sharded output != unsharded output         wrong sharding direction (all-to-sharded
                                               vs sharded-to-all), or attention heads
                                               split inconsistently between q/k/v and o
  Ranks drift apart during training          gradients not averaged, or ranks seeded
                                               identically and fed the same batch
  Slow training with many tiny messages      per-parameter all_sum instead of
                                               nn.average_gradients' bucketing
  Distributed run slower than one Mac        communication dominates: ring over
                                               slow Ethernet, too little compute per
                                               collective; or testing with -n 2 on
                                               one machine
  Remote rank can't import modules           Python environment / paths differ
                                               between hosts
```

### Checklist

1. **Make the code rank-agnostic**: same collectives in the same order on every rank; only rank 0 writes outputs.
2. **Validate with `mlx.launch -n 2` on one Mac** against the unsharded model before touching a second machine.
3. **Pick the split by the bottleneck**: memory → tensor or pipeline parallel (or `fully_shard` for training); throughput of training → data parallel.
4. **Prefer Thunderbolt; JACCL where available** (macOS 26.2+, Thunderbolt 5).
5. **Measure communication time per step** before adding more Macs.

---

## Key Terms

| Term | Definition |
|------|------------|
| **Rank** | One process in a distributed run, identified by `group.rank()`; usually one per Mac. |
| **Collective operation** | An operation all ranks join, such as `all_sum`, `all_gather` or `sum_scatter`. |
| **all_sum (all-reduce)** | Every rank receives the elementwise sum of all ranks' arrays. |
| **Data parallelism** | Each rank holds the whole model and processes part of the batch; gradients are averaged. |
| **Tensor parallelism** | Each layer's weights are split across ranks; partial results are combined with collectives. |
| **Pipeline parallelism** | Consecutive layers live on different ranks; activations are passed with send/recv. |
| **Ring backend** | MLX's TCP-based backend over Ethernet or Thunderbolt. |
| **JACCL** | MLX's RDMA-over-Thunderbolt-5 backend (macOS 26.2+), with much lower latency than ring. |
| **mlx.launch** | MLX's launcher that starts and supervises the ranks on one or several hosts. |

---

## Sources

- MLX documentation, "Distributed Communication" and "Launching Distributed Programs" (backends, JACCL and RDMA, `mlx.launch`, `mlx.distributed_config`, tips): [ml-explore.github.io/mlx/build/html/usage/distributed.html](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- MLX v0.32.3 `python/mlx/nn/layers/distributed.py` (`shard_linear`, quantized sharded layers, `fully_shard`) and `nn.average_gradients`.
- mlx-lm v0.32.0: `shard()` methods and `models/pipeline.py` (`PipelineMixin`): [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (column/row tensor parallelism). [arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
- Matrix-Game MLX port (head-wise tensor parallelism, communication volume, no `all_to_all`): [github.com/dgrauet/Matrix-Game-mlx](https://github.com/dgrauet/Matrix-Game-mlx); vjepa2-mlx (2-rank distributed training): [github.com/dgrauet/vjepa2-mlx](https://github.com/dgrauet/vjepa2-mlx)
- Measurements on this page: MLX 0.32.3, 2 local ranks (`mlx.launch -n 2 --backend ring`) on one M2 Pro.

---

## See Also

- [Transformers](05-transformers.md) -- the attention and MLP blocks that tensor parallelism splits
- [Mixture of Experts](18-mixture-of-experts.md) -- experts are another natural unit to distribute
- [LoRA & Fine-Tuning in Depth](28-lora-finetuning.md) -- training memory that data parallelism and `fully_shard` address
- [Apple Silicon Memory & Metal Limits](26-memory-metal-limits.md) -- the per-Mac limits that make distribution necessary
- [Serving & Deployment](../02-ecosystem/09-serving-deployment.md) -- multi-device serving (exo, mlx-lm)
