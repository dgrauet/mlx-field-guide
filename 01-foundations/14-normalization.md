# Normalization

> **One-liner:** A [normalization](../glossary.md#normalization) layer rescales activations to a controlled magnitude so they neither explode nor vanish through a deep stack -- and the small differences between the variants (LayerNorm vs RMSNorm vs GroupNorm, the `eps`, the presence of a bias) are one of the most common silent port bugs in MLX.

## The Intuition

[Neural Networks](02-neural-networks.md) introduced normalization as a layer that keeps activation magnitudes in check. This page goes deeper, because "normalization" is really four different layers that look almost identical in a weights file and behave very differently -- and porting one as if it were another produces running code with subtly wrong numbers.

The core idea is the same for all of them: take a group of values, subtract a center, divide by a spread, then re-scale with a learned parameter. The variants differ only in **which values get grouped together** and **whether the center and bias are included**. That sounds like a detail. It is exactly the detail that breaks ports.

```
THE NORMALIZATION RECIPE (shared by every variant)

  1. pick a group of activations
  2. center:  subtract the group's mean        (sometimes skipped — RMSNorm)
  3. scale:   divide by the group's spread (std or RMS), + eps for safety
  4. re-learn: multiply by gamma, optionally add beta (learned params)

  The variants differ ONLY in step 1 (the grouping) and which of
  steps 2/4 they keep.
```

---

## How It Actually Works

### LayerNorm -- normalize each token across its features

LayerNorm computes the mean and variance **across the feature dimension**, independently for every token (and every example). It centers, scales, then applies a learned `gamma` (scale) and `beta` (bias):

```
LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta

  - mean, var taken over the feature axis (e.g. the 768 hidden dims)
  - per-token: each position normalized on its own
  - has BOTH a weight (gamma) and a bias (beta)
  - the original Transformer's norm
```

### RMSNorm -- LayerNorm without the mean

Modern LLMs (LLaMA, Qwen, Mistral) replaced LayerNorm with **RMSNorm**, which drops the mean-subtraction entirely and divides by the root-mean-square. It usually has **only a scale, no bias**:

```
RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)

  - NO mean subtraction (no centering)
  - divides by RMS, not standard deviation
  - typically scale-only: gamma exists, beta does NOT
  - cheaper than LayerNorm, near-identical quality at scale
```

The difference looks tiny but matters enormously for porting. The reliable discriminant is **mean-subtraction, not the bias**: RMSNorm never centers; LayerNorm always does. (Bias alone won't tell them apart -- a LayerNorm can be configured bias-free, e.g. T5-style, and still centers.) Load an RMSNorm checkpoint into a LayerNorm implementation and you have invented a mean-subtraction the model was never trained with -- the output is wrong, but nothing crashes.

### BatchNorm -- normalize each feature across the batch

BatchNorm normalizes across the **batch (and spatial) dimension** per channel, and it carries **running statistics** accumulated during training. It behaves *differently in training vs inference*: training uses the current batch's stats and updates the running averages; inference uses the saved running stats. Common in CNNs, rare in transformers.

```
BatchNorm: per-channel, normalized over the batch + spatial positions

  - keeps running_mean / running_var as buffers (not just weight/bias)
  - train mode: use batch stats, update running stats
  - eval mode:  use the SAVED running stats
  - a porter MUST load running_mean/running_var AND run in eval mode
```

### GroupNorm -- normalize within channel groups

GroupNorm splits channels into G groups and normalizes within each group, independent of batch size. It is the workhorse of **diffusion [U-Nets](../glossary.md#u-net) and VAEs**:

```
GroupNorm: channels split into G groups, normalized within each group

  - independent of batch size (unlike BatchNorm)
  - operates on spatial feature maps -> sensitive to channel layout
  - NCHW vs NHWC mistakes here corrupt the grouping silently
```

### Where the norm sits: pre-norm vs post-norm

A transformer can place its norm *before* the sub-layer (pre-norm) or *after* it (post-norm). The original transformer was post-norm; essentially every modern LLM is **pre-norm** because it trains more stably at depth. Same layer, different position -- and getting the placement wrong relative to the reference shifts every residual stream.

---

## Why It Matters for MLX

Normalization layers are deceptively dangerous to port because they all serialize to a small `weight` (and maybe `bias`) tensor that *looks* interchangeable. Here is the symptom-to-cause table:

```
NORMALIZATION PORT FAILURES (running code, wrong numbers)

  Symptom                          Likely cause
  -------                          ------------
  Output subtly off everywhere     LayerNorm used where RMSNorm expected
                                     (phantom mean-subtraction + bias)
  Off by a small constant scale    eps value mismatch (1e-5 vs 1e-6) or
                                     eps inside vs outside the sqrt
  Garbage after the first norm      normalizing over the wrong axis
  Image port: blotchy / wrong tone GroupNorm channel layout (NCHW vs NHWC)
  CNN port: wrong from frame 1      BatchNorm running stats not loaded, or
                                     run in train mode instead of eval
  Fine at fp32, drifts at fp16     reduction (mean/var) computed in low
                                     precision instead of fp32
```

Three rules that prevent most of these:

1. **Identify the variant before porting the layer.** The decisive check is whether the mean is subtracted (LayerNorm centers, RMSNorm does not) -- *not* whether a bias is present, since a LayerNorm can be bias-free yet still center. LayerNorm-vs-RMSNorm confusion is the highest-frequency normalization bug. A no-centering layer is RMSNorm; treat it as one.

2. **Copy `eps` exactly, and match where it goes.** `eps` is a small constant (commonly `1e-5` or `1e-6`) added *inside* the square root to avoid division by zero (this is the stability `eps` from [Numerical Stability](12-numerical-stability.md)). Different architectures use different values, and the wrong one shifts every activation slightly. Also confirm it's `sqrt(var + eps)`, not `sqrt(var) + eps`.

3. **Use MLX's fused primitives.** `mlx.fast.layer_norm` and `mlx.fast.rms_norm` implement the standard formulations correctly (reduction in the right precision, `eps` in the right place), and `mlx.nn` provides `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm`, and `InstanceNorm` modules. Prefer these over hand-rolling the reduction -- hand-rolled norms are where the `eps` placement and the precision of the mean/var get silently wrong. For GroupNorm in particular, watch the channel layout: MLX favors NHWC, and a transposed grouping is the classic diffusion-port bug (see [Latent Space](07-latent-space.md)).

---

## Key Terms

| Term | Definition |
|------|------------|
| **Normalization** | A layer that rescales a group of activations to controlled magnitude, then re-scales with learned parameters. |
| **LayerNorm** | Normalizes across the feature dimension per token; centers (subtracts mean) and has both a scale (gamma) and bias (beta); the original transformer norm. |
| **RMSNorm** | LayerNorm without mean-subtraction: divides by root-mean-square, usually scale-only (no bias); the modern-LLM standard. |
| **BatchNorm** | Normalizes per channel across the batch; carries running statistics and behaves differently in train vs eval mode. |
| **GroupNorm** | Normalizes within groups of channels, independent of batch size; standard in diffusion U-Nets and VAEs; sensitive to channel layout. |
| **eps (epsilon)** | A small constant added inside the square root to prevent division by zero; its exact value must match the reference. |
| **Pre-norm / Post-norm** | Whether the norm is applied before or after a sub-layer; modern transformers are pre-norm for training stability. |

---

## Sources

- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization." [arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
- Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization" (RMSNorm). [arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
- Ioffe, S., & Szegedy, C. (2015). "Batch Normalization." [arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
- Wu, Y., & He, K. (2018). "Group Normalization." [arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)
- Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture" (pre-norm vs post-norm). [arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745)
- MLX `mlx.fast.layer_norm` / `mlx.fast.rms_norm`: [ml-explore.github.io/mlx/build/html/python/fast.html](https://ml-explore.github.io/mlx/build/html/python/fast.html)

---

## See Also

- [Neural Networks](02-neural-networks.md) -- prerequisite; introduces normalization as one of the basic layer types
- [Transformers](05-transformers.md) -- where pre-norm LayerNorm/RMSNorm sit in the transformer block; the residual stream the norm placement affects
- [Numerical Stability](12-numerical-stability.md) -- the `eps` in every norm is a division-by-zero guard; reductions are done in fp32 for the same reason
- [Latent Space](07-latent-space.md) -- GroupNorm in VAEs/U-Nets and the NCHW-vs-NHWC channel-layout trap that corrupts the grouping
- [Quantization](10-quantization.md) -- norm layers are usually kept in higher precision when a model is quantized, partly for the stability reasons above
