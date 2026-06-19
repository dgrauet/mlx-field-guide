# Development Environment Setup

> **Everything you need set up *before* the [Porting Guide](02-porting-guide.md) makes sense.** This page covers the local environment: Python, MLX, an editable checkout of the repo you're contributing to, and the PyTorch reference side you'll compare against.

## The Shape of an MLX Contribution Workspace

Almost every MLX port involves running **two stacks side by side**: the original PyTorch/[CUDA](../glossary.md#cuda) model (your source of truth for correct numbers) and the MLX version you're building. A working setup therefore has both installed, ideally in the same virtual environment so a single script can run a [tensor](../glossary.md#tensor) through both and diff the outputs.

```
THE TWO-STACK WORKSPACE

  Reference side                 MLX side
  --------------                 --------
  torch (CPU build on Mac)       mlx
  the original model's repo      mlx-lm / mlx-vlm / mflux / your fork
  produces "correct" outputs     the thing you're building
        \                           /
         \                         /
          parity script: run same input through both, compare
```

You do not need an NVIDIA GPU. PyTorch on Apple Silicon runs on CPU (or MPS) and is plenty for generating small reference outputs -- you are checking *correctness*, not benchmarking, so a slow reference pass is fine.

---

## Step 1: Python

MLX requires **Apple Silicon (M1 or later) and a native arm64 Python**. The single most common setup failure is an x86 Python running under Rosetta, which cannot import MLX. Verify before anything else:

```bash
python3 -c "import platform; print(platform.machine())"
# Must print: arm64    (NOT x86_64)
```

Use Python 3.9+ (3.10 or 3.11 recommended). A version manager (`pyenv`, `uv`, or the python.org arm64 installer) avoids fighting the system Python.

---

## Step 2: A Virtual Environment

Never install into the system Python. Create an isolated environment per project:

```bash
# Option A: venv (built in)
python3 -m venv .venv
source .venv/bin/activate

# Option B: uv (fast, recommended if you have it)
uv venv
source .venv/bin/activate
```

Confirm the environment is active and arm64 before installing.

---

## Step 3: Install MLX

```bash
pip install mlx          # core array framework
pip install mlx-lm       # if working on LLMs
pip install mlx-vlm      # if working on vision-language models
# mflux, mlx-audio, etc. as needed for your domain
```

Smoke-test the install -- this catches Rosetta and Metal problems immediately:

```bash
python3 -c "import mlx.core as mx; print(mx.array([1,2,3]).sum())"
# Should print: array(6, dtype=int32)
```

If this fails with an architecture error, your Python is not arm64 (back to Step 1).

---

## Step 4: Editable Checkout of What You're Contributing To

To *contribute* (not just use), you need the library installed from a local clone in **editable mode**, so your edits take effect without reinstalling:

```bash
# Example: contributing to mlx-vlm
git clone https://github.com/Blaizzy/mlx-vlm
cd mlx-vlm
pip install -e .          # editable: source edits are live

# Example: contributing to mlx core's examples
git clone https://github.com/ml-explore/mlx-examples
cd mlx-examples
# follow the per-example README; many are run in place
```

The `-e` (editable) install is the key difference between *using* a library and *developing* it. After it, `import mlx_vlm` resolves to your clone, and changing a file changes the imported behavior on the next run.

For changes to **mlx core itself** (the C++/Metal layer, not Python), you build from source. That is a heavier path -- a C++ toolchain and CMake -- and is documented in the mlx repo. Most contributions (new model architectures, conversion scripts, Python-level fixes) do **not** require building core from source; they live in the Python libraries above.

---

## Step 5: The Reference (PyTorch) Side

Install the original model's stack into the *same* environment so one script can drive both:

```bash
pip install torch                      # CPU/MPS build on Mac, no CUDA
pip install transformers diffusers     # whatever the source model uses
# plus the source model's own repo / requirements
```

Generate reference outputs once and save them, so the slow PyTorch pass runs only when you change inputs:

```python
import torch, numpy as np

# Fixed seed + fixed input = reproducible reference
torch.manual_seed(0)
x = torch.randn(1, 16, 256)
with torch.no_grad():
    ref = reference_model(x)
np.save("ref_output.npy", ref.cpu().numpy())   # the number to match
```

This saved array is what the [parity tests](05-testing-validation.md) compare MLX against.

---

## Step 6: Editor & Pre-commit Hygiene

- **Type/lint:** match the target repo's tooling (`ruff`, `black`, `mypy` are common). Run them before opening a PR; CI will reject violations.
- **Pre-commit hooks:** if the repo ships a `.pre-commit-config.yaml`, run `pre-commit install` once per clone. (This very guide does -- it validates internal links and glossary anchors on commit.)
- **`mx.eval` while debugging:** MLX is lazy. In a debugger or notebook, an array may be unevaluated and look "empty." Call `mx.eval(x)` before inspecting values. This trips up nearly every newcomer (see [Porting Guide](02-porting-guide.md) and the lazy-eval failure mode).

---

## A Minimal Working Setup, Start to Finish

```bash
python3 -c "import platform; assert platform.machine()=='arm64'"  # gate
python3 -m venv .venv && source .venv/bin/activate
pip install mlx mlx-lm torch transformers
git clone https://github.com/Blaizzy/mlx-vlm && cd mlx-vlm && pip install -e .
python3 -c "import mlx.core as mx, mlx_vlm; print('ok', mx.array([1,2,3]).sum())"
```

If the last line prints `ok array(6, ...)`, you are ready to start the [Porting Guide](02-porting-guide.md).

---

## Common Setup Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `import mlx` fails, architecture error | x86 Python under Rosetta | Install native arm64 Python (Step 1) |
| MLX installs but is very slow / CPU only | Old macOS or non-Metal path | Update macOS; MLX needs a recent OS for Metal |
| Edits to the library have no effect | Installed from PyPI, not `-e .` | Reinstall the clone with `pip install -e .` |
| Arrays look empty in the debugger | MLX lazy evaluation | `mx.eval(x)` before inspecting |
| Reference and MLX disagree from line 1 | Different input or unseeded RNG | Fix seed; save and reuse one reference input |

---

## Sources

- MLX installation docs: [ml-explore.github.io/mlx/build/html/install.html](https://ml-explore.github.io/mlx/build/html/install.html)
- mlx core repository (build-from-source instructions): [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
- mlx-examples: [github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)
- uv (fast Python env/installer): [github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- PyTorch on Apple Silicon: [pytorch.org](https://pytorch.org/)

---

## See Also

- [Porting Guide](02-porting-guide.md) -- the next step once your environment runs; the actual mechanics of translating a model
- [Testing & Validation](05-testing-validation.md) -- how to use the reference outputs you set up here to prove a port is correct
- [Where to Contribute](01-where-to-contribute.md) -- which repo to clone in Step 4, depending on your domain
- [Tooling](../02-ecosystem/08-tooling.md) -- profilers and debuggers that plug into this environment
