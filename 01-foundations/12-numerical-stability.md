# Numerical Stability

> **One-liner:** Neural networks compute with finite-precision floating point, so values can overflow to infinity, underflow to zero, or collapse into `NaN` -- and understanding *why* is the difference between fixing a broken port and randomly changing dtypes until it works.

## The Intuition

A real number like π has infinitely many digits. A computer stores it in 16 or 32 bits. Every number a model touches is therefore an *approximation*, and the approximations have limits: a largest representable value, a smallest, and a gap between adjacent numbers. Push past those limits and arithmetic stops behaving like the math you wrote.

Most of the time this is invisible -- the errors are tiny and wash out. But a few specific operations in a neural network sit right at the edges, and when a port or a precision choice nudges them over, you get the symptoms every MLX porter eventually meets: a loss that becomes `NaN`, an image that comes out pure black, logits that are all `inf`, attention weights that collapse to a single token. These are not random. Each one traces to a specific numerical event.

```
THE THREE FAILURE EVENTS

  Overflow    value too large for the dtype      -> +inf
  Underflow   value too small (rounds to zero)   -> 0  (silently)
  Invalid op  inf - inf, 0/0, log(0), sqrt(-1)   -> NaN

  NaN is contagious: any arithmetic with NaN yields NaN.
  One NaN at layer 3 poisons every layer after it.
```

---

## How It Actually Works

### Floating point has hard edges

A floating-point number is `sign × mantissa × 2^exponent`. The **exponent** sets the *range* (how big/small) and the **mantissa** sets the *precision* (how many significant digits). The three formats a model uses trade these differently:

```
FLOATING-POINT FORMATS

  Format    Exponent  Mantissa   Max value     Smallest normal   Use
  ------    --------  --------   ---------     ---------------   ---
  float32   8 bits    23 bits    ~3.40e38      ~1.2e-38          reference / safe
  float16   5 bits    10 bits    ~65504        ~6e-5             memory, but narrow range
  bfloat16  8 bits     7 bits    ~3.39e38      ~1.2e-38          common in ML: ~f32 range, coarse precision

  Key insight:
    float16  overflows easily  (max is only 65504!)
    bfloat16 shares float32's 8-bit exponent, so it overflows at
             essentially the same magnitude (~3.4e38) -- just with
             much coarser precision (7 mantissa bits, more rounding)
```

That `float16` max of **65504** is the villain in a huge fraction of stability bugs. A few large activations multiplied together blow past it and become `inf`. This is exactly why ML moved to **bfloat16**, which keeps float32's exponent range (overflowing only near ~3.4e38) and simply accepts coarser rounding -- overflow is usually worse than imprecision. (Note: MLX *arrays* default to float32; bfloat16 is a deliberate choice you make when loading or casting a model for inference, not the framework's default.)

### The dangerous operations

A handful of operations concentrate the risk:

**Softmax** -- ubiquitous in [attention](../glossary.md#attention). It computes `exp(x)`, and `exp` overflows fast: float16's max is ~65504, and `exp(x) > 65504` once `x > ln(65504) ≈ 11.1` -- so even `exp(12)` already overflows in float16. The fix is universal and you should recognize it:

```
NUMERICALLY-STABLE SOFTMAX

  Naive:   softmax(x)_i = exp(x_i) / Σ exp(x_j)
           -> exp(large) overflows to inf -> inf/inf -> NaN

  Stable:  m = max(x)
           softmax(x)_i = exp(x_i - m) / Σ exp(x_j - m)
           -> largest exponent is exp(0)=1, no overflow
           -> mathematically identical, numerically safe

  Every correct softmax (and attention kernel) does this subtraction.
  If you hand-roll attention and skip it, you get NaN on long sequences.
```

**`log`, `sqrt`, division** -- `log(0) = -inf`, `0/0 = NaN`, `sqrt` of a tiny-negative (from rounding) = `NaN`. The standard guard is a small **epsilon**: `log(x + 1e-8)`, `x / (denom + 1e-8)`, `sqrt(var + eps)`. [Normalization](../glossary.md#normalization) layers (LayerNorm, RMSNorm) all carry an `eps` for exactly this -- dividing by the square root of a variance that could be zero.

**Accumulation in low precision** -- summing thousands of small numbers in float16 loses the small ones (each addition rounds). Reductions (sums, means, norms, softmax denominators) are often computed in float32 even when the surrounding model runs in float16/bfloat16, precisely to avoid this.

### Training-only failures

[Training](../glossary.md#training) adds two classics that [inference](../glossary.md#inference)-only ports never hit, but worth knowing:

- **Exploding gradients** -- gradients grow multiplicatively through layers until they overflow; the loss prints `NaN`. Mitigated by gradient clipping and normalization.
- **Vanishing gradients** -- gradients shrink toward zero through depth, so early layers stop learning. Residual connections and normalization were invented largely to fix this.

Since "almost all MLX porting work is inference-only" (see [Training vs. Inference](03-training-vs-inference.md)), most porters meet *forward-pass* instability -- overflow and NaN -- far more than gradient issues.

---

## Why It Matters for MLX

Numerical instability is one of the most confusing classes of port bug because the code is "correct" and the failure is silent or seemingly random. Mapping the symptom to the cause is the whole game:

```
INSTABILITY SYMPTOMS IN A PORT (and where to look)

  Symptom                         Likely numerical cause
  -------                         ----------------------
  Output is all NaN               softmax/log/div without the standard guard;
                                    a NaN upstream poisoning everything after
  Image comes out pure black      values clipped/saturated to 0; underflow
  Image comes out pure white      overflow / saturation to max
  Logits all inf, then NaN        float16 overflow (max 65504) in a matmul
  Works in fp32, breaks in fp16   range problem -> try bfloat16
  Fine short, NaN on long seqs    un-stabilized softmax in hand-rolled attn
  Slightly wrong, not broken      precision (rounding) -- expected; widen tol
```

Three practical rules for porters:

1. **When in doubt, reach for bfloat16, not float16.** If a model is stable in float32 but produces `inf`/`NaN` in float16, the cause is almost always *range* (overflow), and bfloat16 -- same range as float32 -- usually fixes it without the memory cost of float32.

2. **Use the fused primitives.** `mlx.fast.scaled_dot_product_attention`, `mlx.fast.rms_norm`, `mlx.fast.layer_norm` implement the stable formulations (max-subtraction, epsilon) correctly. Hand-rolling these is where the max-subtraction or epsilon gets dropped. Prefer them.

3. **Hunt the *first* NaN, not the last.** Because NaN is contagious, the output being NaN tells you nothing about where it started. Insert checks (`mx.eval(x); assert not mx.isnan(x).any()`) layer by layer to find the *first* layer that produces a NaN -- that's the bug. Everything downstream is just the poison spreading.

This connects directly to [Quantization](10-quantization.md), where keeping certain layers (norms, the final projection) in higher precision is partly a stability decision, and to the parity-testing discipline in the [Porting Guide](../03-contributing/02-porting-guide.md).

---

## Key Terms

| Term | Definition |
|------|------------|
| **Overflow** | A value exceeds the dtype's maximum and becomes `±inf`. float16's max is only ~65504, making it overflow-prone. |
| **Underflow** | A value is smaller than the dtype's smallest representable number and silently rounds to 0. |
| **NaN** | "Not a Number"; the result of invalid operations (`inf−inf`, `0/0`, `log(0)`). Contagious -- any operation with a NaN yields NaN. |
| **Epsilon (eps)** | A tiny constant added inside `log`, `sqrt`, or division to avoid `log(0)`, `sqrt(negative)`, or divide-by-zero. |
| **Stable softmax** | Softmax computed after subtracting the max input, preventing `exp` overflow; mathematically identical to the naive form. |
| **Exploding / vanishing gradients** | Training-time instabilities where gradients grow to overflow or shrink to zero through network depth. |
| **bfloat16** | 16-bit float with float32's 8-bit exponent range but only 7 mantissa bits; the common ML inference choice because range matters more than precision for stability. |

---

## Sources

- Goldberg, D. (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic." The canonical reference. [docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- Micikevicius, P., et al. (2017). "Mixed Precision Training." Why bfloat16/float16 work and where they need float32 fallbacks. [arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)
- IEEE 754 floating-point standard (overview): [en.wikipedia.org/wiki/IEEE_754](https://en.wikipedia.org/wiki/IEEE_754)
- MLX `mlx.fast` primitives (stable fused softmax/norm/attention): [ml-explore.github.io/mlx/build/html/python/fast.html](https://ml-explore.github.io/mlx/build/html/python/fast.html)

---

## See Also

- [Tensors](01-tensors.md) -- prerequisite; dtype and precision are tensor properties, introduced there
- [Quantization](10-quantization.md) -- the deliberate-precision-reduction counterpart; why some layers must stay in higher precision is partly a stability decision
- [Attention](06-attention.md) -- home of the stable-softmax requirement; long-sequence NaNs usually trace here
- [Training vs. Inference](03-training-vs-inference.md) -- distinguishes the gradient instabilities (training) from forward-pass instabilities (inference) a porter meets
- [Porting Guide](../03-contributing/02-porting-guide.md) -- the symptom-to-cause table here mirrors the Common Port Failure Modes; the NaN-hunting discipline in practice
