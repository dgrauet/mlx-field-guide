# Testing & Validation

> **How to prove a port is correct -- and keep it correct.** The [Porting Guide](02-porting-guide.md) shows you how to translate a model; this page is the discipline that separates "it runs" from "it's right." MLX ports fail *silently*: the code executes, shapes match, an image or some tokens come out -- and they're subtly wrong. Tests are how you catch that.

## Why "It Runs" Is Not "It Works"

A ported model almost never crashes when it's wrong. A transposed [weight](../glossary.md#weight), a missing scale factor, a [NCHW vs NHWC](../glossary.md#latent-space) mix-up, a [RoPE](../glossary.md#rope) frequency computed in the wrong order -- all of these produce running code that emits plausible garbage. The model generates grammatical text, or an image that is *almost* right, and a human glance says "looks fine." It is not fine.

The only reliable signal is **numerical comparison against the reference implementation.** You set this up in the [Development Environment](04-dev-environment.md); here is how to turn it into a test suite.

```
THE VALIDATION LADDER (cheapest, most-localizing test first)

  1. Per-layer parity     run one layer, compare to PyTorch       <- start here
  2. Subgraph parity      run a block (attention, a transformer layer)
  3. Full forward parity  one input, full model, compare logits/latents
  4. End-to-end quality    full generation, judged output
  5. Quantization parity   does 4-bit still match closely enough?
  6. Regression tests      lock in correctness so it can't silently break
```

Climb from the bottom. A failure at rung 3 with no rung-1 tests tells you "something is wrong somewhere" -- useless. A failure at rung 1 tells you "this layer." Always have the lower rungs before trusting the higher ones.

---

## Rung 1: Per-Layer Parity

The workhorse test. Feed identical input to one MLX layer and its PyTorch counterpart, compare outputs:

```python
import numpy as np
import mlx.core as mx
import torch

def assert_parity(name, mlx_out, torch_out, atol=1e-4, rtol=1e-4):
    # Cast to float32 first: NumPy has no bfloat16 dtype, so np.array() on a
    # bf16 MLX (or torch) tensor raises. Always compare in float32.
    a = np.array(mlx_out.astype(mx.float32))
    b = torch_out.detach().cpu().float().numpy()
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    max_abs = np.abs(a - b).max()
    ok = np.allclose(a, b, atol=atol, rtol=rtol)
    print(f"{name}: max|Δ|={max_abs:.2e}  {'PASS' if ok else 'FAIL'}")
    assert ok, f"{name}: outputs diverge (max abs diff {max_abs:.2e})"

# Same input, same weights loaded into both layers
x = np.random.randn(1, 64, 512).astype(np.float32)
mlx_y  = mlx_layer(mx.array(x))
torch_y = torch_layer(torch.tensor(x))
assert_parity("linear_proj", mlx_y, torch_y)
```

Run this for every non-trivial layer as you port it -- not at the end. The moment a layer fails, you know exactly where the bug is, while the change is still fresh.

### Choosing Tolerances

`float32` parity should be tight (`atol≈1e-4`). MLX arrays are float32 by *default*, but model code very often loads or casts weights to `bfloat16`/`float16` for inference -- and in those dtypes accumulated rounding is real, so `1e-4` will produce false failures. Match the dtype you actually run, and loosen deliberately:

```
TOLERANCE BY DTYPE (rule of thumb)

  float32   atol 1e-5 .. 1e-4    tight; real bugs show here
  bfloat16  atol 1e-2 .. 5e-2    wide mantissa gap; expect drift
  float16   atol 1e-3 .. 1e-2    narrower range than bf16

  If you must loosen past these to pass, suspect a real bug,
  not rounding. "I widened atol until it passed" is how wrong
  ports ship.
```

---

## Rung 2-3: Subgraph and Full-Forward Parity

Once layers pass individually, test them composed -- an [attention](../01-foundations/06-attention.md) block, a full [transformer](../01-foundations/05-transformers.md) layer, then the whole model on one fixed input:

```python
# Full-forward: the saved reference from the dev-environment setup
ref = np.load("ref_output.npy")               # PyTorch output, computed once
out = np.array(full_mlx_model(mx.array(fixed_input)))
assert_parity("full_forward", out, torch.tensor(ref), atol=5e-2)  # bf16; tighten to ~1e-4 in fp32
```

Errors **compound through depth**: a 1e-3 per-layer discrepancy across 40 layers can grow large. If per-layer tests pass but full-forward fails, suspect (a) an interaction the isolated tests missed -- masking, positional encoding, residual scaling -- or (b) legitimate accumulation, distinguishable by checking whether the divergence grows smoothly with depth or jumps at one layer.

---

## Rung 4: End-to-End Quality

Numerical parity on a random input proves the math; it does not prove the *experience*. The final rung is generating real output and judging it:

- **LLMs:** generate from fixed prompts at temperature 0; the MLX output should match the reference token-for-token (or near it) for at least the first many tokens. Divergence after some tokens can be sampling/rounding; divergence at token 1 is a bug.
- **Diffusion/image:** generate with a **fixed seed and fixed scheduler** and compare images. Look specifically for the known failure signatures -- black images, cyan/inverted color, gray mush, structured-but-wrong layout -- each of which points at a specific bug class (see [Porting Guide](02-porting-guide.md) failure modes).
- **VLMs:** verify the **vision encoder** numerically (rung 1-3) *before* trusting generated descriptions; fluent wrong captions are the trap (see [Vision-Language Models](../02-ecosystem/11-vision-language-models.md)).

Visual/textual inspection is necessary here but is a *complement* to numerical tests, never a replacement. "It looks right" has shipped countless broken ports.

---

## Rung 5: Quantization Parity

[Quantization](../01-foundations/10-quantization.md) is a second place correctness silently degrades. A port that matches in `float16` can fall apart at 4-bit if group sizes or the quantization scope are wrong. Test the quantized model as its own artifact:

```python
# Compare quantized MLX output against float reference (looser tolerance)
out_q = np.array(quantized_mlx_model(mx.array(fixed_input)))
# Expect quantization error, but bounded — and output should stay coherent
assert_parity("forward_4bit", out_q, torch.tensor(ref), atol=1e-1, rtol=1e-1)
```

Quantization error is expected and *bounded*; incoherent output (garbage tokens, destroyed images) is a bug, not rounding. Watch especially for layers that should stay in higher precision (embeddings, final projection, norms) being quantized when they shouldn't be.

---

## Rung 6: Regression Tests

Once a port is correct, lock it in. A `pytest` suite with the fixed input and saved reference turns "I verified it once" into "CI verifies it on every change":

```python
# test_parity.py — runs in CI, fails the build on regression
import numpy as np, mlx.core as mx
from my_port import model

def test_full_forward_parity():
    x = np.load("fixtures/input.npy")
    ref = np.load("fixtures/ref_output.npy")
    out = np.array(model(mx.array(x)))
    assert np.allclose(out, ref, atol=5e-2), \
        f"regression: max diff {np.abs(out-ref).max():.2e}"
```

Commit the fixtures (`input.npy`, `ref_output.npy`) alongside the test -- but keep them *tiny*: one small fixed input, not a full-resolution latent or a multi-megabyte tensor, so the repo stays lean. Now a future refactor -- yours or someone else's -- that breaks numerics fails loudly instead of silently shipping. This is what makes a port *maintainable* rather than a one-time heroic effort.

---

## A Pre-PR Checklist

Before opening a pull request on a port:

- [ ] Per-layer parity passes for every non-trivial layer (`float32`, tight tolerance)
- [ ] Full-forward parity passes on a fixed input (dtype-appropriate tolerance)
- [ ] End-to-end output inspected for the known failure signatures of the domain
- [ ] Quantized variant tested separately and stays coherent
- [ ] At least one regression test + committed fixtures so CI guards it
- [ ] Tolerances are *justified by dtype*, not widened until green
- [ ] Lint/format/type checks pass (see [Development Environment](04-dev-environment.md))

---

## Sources

- NumPy `allclose` / `testing.assert_allclose`: [numpy.org/doc](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)
- pytest: [docs.pytest.org](https://docs.pytest.org/)
- mlx-examples test suites (real-world parity test patterns): [github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)
- MLX unit tests (how core tests numerics): [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

---

## See Also

- [Porting Guide](02-porting-guide.md) -- the numerical-agreement step in context, plus the "Common Port Failure Modes" each test is designed to catch
- [Development Environment](04-dev-environment.md) -- how to set up the two-stack workspace and generate the reference outputs these tests compare against
- [Quantization](../01-foundations/10-quantization.md) -- why the 4-bit model needs its own test (rung 5)
- [Vision-Language Models](../02-ecosystem/11-vision-language-models.md) -- the "verify the encoder before trusting the output" discipline, applied
- [Tooling](../02-ecosystem/08-tooling.md) -- profilers and debuggers for diagnosing *why* a parity test fails
