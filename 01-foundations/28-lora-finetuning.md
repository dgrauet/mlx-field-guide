# LoRA & Fine-Tuning in Depth

> **One-liner:** LoRA fine-tunes a model by learning a small low-rank correction `scale · B·A` next to each frozen weight matrix, which makes training fit on a Mac. Most of the practical traps are in the details around it: the `scale` convention (which differs between libraries by up to 10x), the transposed adapter layout, fusing into a quantized base, and the memory spent on gradients, optimizer state and activations.

## The Intuition

Full fine-tuning updates every weight. For a 1.3B-parameter model that means storing the weights, a gradient for each, and two optimizer moments for each: tens of GB, before counting activations. [Training vs Inference](03-training-vs-inference.md) covered why training needs so much more memory than inference.

**LoRA** (low-rank adaptation) freezes the original weight `W` and learns only a small update:

```
  y = W x  +  scale · B (A x)

  W : (out, in)    frozen, can even be quantized (QLoRA)
  A : (r, in)      trainable, r = rank, typically 8-64
  B : (out, r)     trainable, initialized to ZERO so training starts from W exactly

  trainable values per layer: r · (in + out)   instead of in · out
```

For Gemma 3 1B with rank 8 on 16 layers, `mlx-lm` reports **4.0M trainable parameters out of 1,301.9M (0.31%)**.

---

## How It Actually Works

### The `scale` convention: same LoRA, different numbers

Every library multiplies the low-rank product by a scale, but they don't parameterize it the same way:

```
  library                      scale applied               default
  ---------------------------  --------------------------  ---------------------------
  Hugging Face PEFT            lora_alpha / r              alpha 8, r 8  -> 1.0
    (rsLoRA variant)           lora_alpha / sqrt(r)
  mlx-lm (tuner/lora.py)       `scale`, used as-is         scale 20.0, rank 8
  ltx-trainer (LTX-2 MLX)      alpha / rank, then passed   rank 64, alpha 64 -> 1.0
                                 to mlx-lm's LoRALinear
```

So "rank 8" means very different updates depending on where the config came from. A PEFT recipe with `r=8, lora_alpha=16` is `scale = 2`; `mlx-lm`'s default is `scale = 20`, a **10x stronger update** for the same learned `A` and `B`. The learning rate interacts with it: a scale 10x larger behaves roughly like a 10x larger learning rate on the update. Converting a recipe between libraries means converting `alpha / r` into `scale`, as ltx-trainer does.

### Adapter layout: PEFT and MLX are transposed

```
  PEFT (PyTorch)                         mlx-lm
  lora_A.weight   (r, in)                lora_a   (in, r)
  lora_B.weight   (out, r)               lora_b   (r, out)
  y += (x @ A.T @ B.T) * alpha / r       y += scale * (x @ lora_a @ lora_b)
```

Measured on a 1024 × 1024 layer with a trained-looking PEFT adapter (`r=8, alpha=16`), MLX 0.32.3 / mlx-lm 0.32.0:

```
  converted with lora_a = A.T, lora_b = B.T, scale = alpha / r = 2    max |diff| 2.1e-6
  same, but mlx-lm's default scale = 20                               max |diff| 3.19   (update 10x)
  A and B copied without transposing                                  ValueError: [matmul] shape mismatch
```

The transpose fails loudly. The scale doesn't: a wrong scale loads, runs, and produces an over- or under-applied fine-tune.

### QLoRA: training on a quantized base

Because `W` is frozen, it can stay 4-bit quantized ([Quantization](10-quantization.md)) while `A` and `B` train in higher precision. That's **QLoRA**, and it's what `mlx-lm` does automatically when the base model is quantized: `LoRALinear.from_base` wraps a `QuantizedLinear`. The memory saved on weights is what makes fine-tuning a 7-12B model possible on a 32 GB Mac.

### Fusing the adapter

For deployment, the update can be merged into the weights, `W' = W + scale · B·A`, removing the adapter's extra matmuls. On a quantized base, `mlx_lm.fuse` does this by dequantizing, adding the update, and **re-quantizing to the original bits by default** (`--dequantize` keeps the fused weights in floating point instead).

Re-quantizing rounds the fused weights to the 4-bit grid again. Whether the adapter survives depends on how big its update is compared to one quantization step:

```
  adapter                                   update rms /       fused 4-bit vs unfused
                                            4-bit step
  synthetic subtle adapter (scale 2)           0.12            error 1.33x the update itself
  Gemma 3 1B, trained 150 steps                1.14 (median,   test loss 0.752 vs 0.753
    with mlx-lm defaults (scale 20)            0.55-3.15)        (base without adapter: 7.10)
  either, fused with --dequantize               -              exact (0.0 difference)
```

With `mlx-lm`'s defaults, the trained update is about one quantization step and fusing back to 4-bit preserved the result. A subtle adapter -- small scale, small learning rate, few steps, or a converted PEFT adapter with `alpha / r = 1` -- can be **rounded away**. After fusing, always compare the evaluation loss with the unfused adapter. Fuse with `--dequantize` (and re-quantize separately if needed) when they differ. The fused-dequantized model is larger (2.5 GB vs 731 MB for Gemma 3 1B here).

### Memory: what training actually stores

```
  full fine-tune, per parameter:
                               PyTorch mixed precision     MLX, bf16 parameters
    weights                    2 (bf16) + 4 (fp32 master)  2
    gradients                  2                           2
    AdamW m and v              8 (fp32)                    4 (bf16: same dtype as
                                                              the parameter)
                              ---------                   --
                              16 bytes                    8 bytes
    1.3B params               ~21 GB                      ~10 GB      plus activations

  LoRA / QLoRA:
    frozen weights (4-bit)     ~0.56 bytes/param   (no gradient, no optimizer state)
    A, B + their grads + AdamW for 0.3% of the parameters: negligible
    activations for backprop   <- the remaining big cost
```

In MLX, `optim.AdamW` creates its moments with `zeros_like(parameter)` (checked on MLX 0.32.3: bf16 parameters get bf16 `m` and `v`). That halves optimizer memory, but low-precision moments can hurt stability on long runs; keeping trainable parameters in float32 restores the PyTorch-style budget.

Activations are what backpropagation needs to keep from the forward pass, and they grow with batch size × sequence length × layers trained. **Gradient checkpointing** (`--grad-checkpoint` in `mlx-lm`, built on `mx.checkpoint`) discards them and recomputes each layer's forward pass during the backward pass. Measured on the Gemma 3 1B QLoRA run (batch 8, 16 layers, short examples, MLX 0.32.3):

```
                              peak memory    throughput
  no checkpointing              1.97 GB       675 tok/s
  --grad-checkpoint             1.48 GB       537 tok/s     (-25% memory, -20% speed)
```

The saving grows with longer sequences and larger batches. Use it when the run doesn't fit, not by default.

---

## Why It Matters for MLX

### Port and training failures

```
LoRA FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  Fine-tune much too strong (overfits,       scale taken from mlx-lm's default (20)
    breaks the base model's behaviour)         for a recipe written for PEFT's
                                               alpha / r (often 1-2)
  Converted adapter has no visible effect    scale too small, or adapter applied to
                                               the wrong target modules / layers
  Shape error loading a PEFT adapter         A / B not transposed (PEFT: (r, in) and
                                               (out, r); mlx-lm: (in, r) and (r, out))
  Fused model behaves like the base model    subtle update rounded away when fusing
                                               back to 4-bit: compare eval loss, use
                                               --dequantize
  Training starts from a different loss      B not initialized to zero (or adapter
    than the base model                        loaded from a previous run)
  OOM in the backward pass                   activations: lower batch size or
                                               sequence length, --grad-checkpoint,
                                               fewer --num-layers
  Loss falls but outputs are unchanged at    adapter saved but not loaded
    inference                                  (adapter_path missing) or loaded on
                                               a different base model revision
```

### Checklist

1. **Write the scale explicitly** (`scale` in mlx-lm, `alpha` and `r` in PEFT) and convert it when moving recipes between libraries.
2. **Check the trainable-parameter count** printed at startup: it confirms which layers got adapters.
3. **Compare to the base model's loss at step 0**: LoRA must start exactly at the base model (B = 0).
4. **Evaluate after fusing**, against the unfused adapter.
5. **Budget memory for activations**, not weights: batch × sequence × layers, then decide on `--grad-checkpoint`.

---

## Key Terms

| Term | Definition |
|------|------------|
| **LoRA (low-rank adaptation)** | Fine-tuning method that freezes `W` and learns a low-rank update `scale · B·A` with `A: (r, in)`, `B: (out, r)`. |
| **Rank (r)** | The inner dimension of the LoRA update; sets its capacity and its parameter count `r · (in + out)`. |
| **LoRA scale / alpha** | The multiplier on `B·A`: `alpha / r` in PEFT, `scale` directly in mlx-lm (default 20). |
| **QLoRA** | LoRA on a frozen quantized base model; the adapters train in higher precision. |
| **Adapter fusing** | Merging `scale · B·A` into the base weights for deployment; re-quantizing can round small updates away. |
| **Gradient checkpointing** | Discarding forward activations and recomputing them during backpropagation, trading compute for memory (`mx.checkpoint`, `--grad-checkpoint`). |
| **Optimizer state** | Per-parameter values kept by the optimizer (AdamW: two moments), often larger than the weights in full fine-tuning. |
| **Target modules** | The layers that receive adapters (e.g. attention `q/k/v/o`), chosen by name in the training config. |

---

## Sources

- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- Kalajdzievski, D. (2023). "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA" (rsLoRA). [arxiv.org/abs/2312.03732](https://arxiv.org/abs/2312.03732)
- Chen, T., et al. (2016). "Training Deep Nets with Sublinear Memory Cost" (gradient checkpointing). [arxiv.org/abs/1604.06174](https://arxiv.org/abs/1604.06174)
- Hugging Face PEFT, LoRA reference (`lora_alpha / r`, `use_rslora`): [huggingface.co/docs/peft/package_reference/lora](https://huggingface.co/docs/peft/package_reference/lora)
- mlx-lm v0.32.0: `tuner/lora.py` (`LoRALinear`, `fuse`), `lora.py` (defaults), `tuner/trainer.py` (`grad_checkpoint`), `fuse.py` (`--dequantize`), [LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- ltx-trainer in the LTX-2 MLX port (`alpha / rank` mapped to mlx-lm's scale): [github.com/dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)
- Measurements on this page: MLX 0.32.3, mlx-lm 0.32.0, Apple M2 Pro; QLoRA on `mlx-community/gemma-3-1b-it-4bit` with a synthetic 320-example dataset.

---

## See Also

- [Training vs Inference](03-training-vs-inference.md) -- prerequisite; gradients, optimizers and why training costs more memory
- [Quantization](10-quantization.md) -- the 4-bit grid that fusing re-rounds onto
- [Checkpoints & Weight Formats](25-checkpoints-weight-formats.md) -- adapter files and loading strictly
- [Apple Silicon Memory & Metal Limits](26-memory-metal-limits.md) -- the memory budget a training run must fit in
- [Training & Fine-tuning](../02-ecosystem/07-training-finetuning.md) -- the training tools, CUDA vs MLX
