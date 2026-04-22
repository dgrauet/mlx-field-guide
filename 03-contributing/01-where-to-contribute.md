# Where to Contribute

> **The MLX open-source ecosystem is young enough that individual contributors move the needle.** The difference between "LTX-Video does not run on Apple Silicon" and "LTX-Video runs on Apple Silicon" is one person's porting work. That is the current state of this ecosystem.

## What This Page Covers

Where the code lives, who controls what, how decisions get made, how to engage, and where a newcomer has the best chance of having an impact. This is the map before you start.

---

## The Core Repositories

### ml-explore (Apple's GitHub Organization)

Apple maintains three primary repositories under the [ml-explore](https://github.com/ml-explore) GitHub organization:

**[mlx](https://github.com/ml-explore/mlx)** -- the core framework.

The C++ and [Metal](../glossary.md#metal) implementation of MLX. This is where `mlx.core` lives -- array operations, the lazy evaluation engine, automatic differentiation, JIT compilation, and the Metal kernel library. Contributing here requires C++ and, for [GPU](../glossary.md#gpu) kernels, Metal Shading Language (MSL). Apple's ML research team reviews and merges PRs. The bar is high: correctness, performance, and API consistency with the existing design.

What gets contributed here:
- New array operations (primitives that don't yet exist)
- Performance improvements to existing Metal kernels
- Bug fixes in the autograd engine or lazy evaluation
- New dtype support, new device support
- Python API additions that expose new C++ capabilities

**[mlx-examples](https://github.com/ml-explore/mlx-examples)** -- reference model implementations.

Python implementations of model architectures in MLX: LLaMA, Mistral, Stable [Diffusion](../glossary.md#diffusion), Whisper, Mamba, and many others. Also contains `mlx-lm`, the primary library for LLM inference and LoRA fine-tuning on Apple Silicon. This is the most contributor-accessible of the three repositories -- contributions are Python, the review bar is lower, and new model ports are actively welcomed.

What gets contributed here:
- New model architecture ports (the most common contribution type)
- Bug fixes in existing implementations
- Improved weight conversion scripts
- Documentation and examples
- Performance optimizations within Python-land

**[mlx-data](https://github.com/ml-explore/mlx-data)** -- data loading.

The data loading library for MLX. Handles streaming, batching, shuffling, and augmentation. Less active than mlx-examples; contributions here tend to be new data source integrations or performance improvements to the pipeline.

```
ml-explore ORGANIZATION STRUCTURE

  github.com/ml-explore/
  ├── mlx              -- C++ core + Metal kernels (high bar)
  ├── mlx-examples     -- Python model implementations (accessible)
  │   └── mlx-lm       -- LLM inference + LoRA (most active)
  └── mlx-data         -- data loading utilities
```

### Community Projects

The more active frontier for model porting lives in individual community projects, not the Apple repos:

**[mlx-audio](https://github.com/lucasnewman/mlx-audio)** (lucasnewman)
Audio generation and processing in MLX. Includes Kokoro TTS, Parler TTS, Whisper, and other audio models. One of the most complete community libraries for a specific domain. Contributors: open issues, model requests, and the maintainer is responsive.

**[mflux](https://github.com/filipstrand/mflux)** (filipstrand)
FLUX image generation on Apple Silicon. A clean, standalone MLX port of Black Forest Labs' FLUX models. Actively maintained, community around it growing. Contributions: new FLUX variants, ControlNet support, performance improvements, quantization.

**[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)** (Blaizzy)
Vision-language models on MLX. LLaVA, Qwen-VL, Phi-3 Vision, and others. One of the few places multi-modal models are being ported systematically to Apple Silicon.

**[mlx-forge](https://github.com/ml-explore/mlx-forge)** -- porting and kernel tooling.
Tools for porting [CUDA](../glossary.md#cuda) models to MLX. An opinionated directory of utilities, weight conversion helpers, and in-development Metal kernel work. If you are doing serious porting work on LTX-Video or [Matrix](../glossary.md#matrix)-Game, this is where your utilities likely belong.

**Individual model ports:**
- Matrix-Game-mlx (video generation port)
- mlx-sd-webui (Stable Diffusion WebUI integration)
- Various LLM ports that graduate into mlx-examples

---

## MLX Community on Hugging Face

[huggingface.co/mlx-community](https://huggingface.co/mlx-community) is the organization on Hugging Face that hosts converted MLX models. Over 3,000 model repositories as of early 2025, spanning:

- LLMs (Llama, Mistral, Qwen, Phi, Gemma, DeepSeek variants)
- Image generation (Stable Diffusion, FLUX variants)
- Audio models (Whisper, TTS models)
- Vision-language models
- [Embedding](../glossary.md#embedding) models

**The workflow for contributing a model conversion:**

1. Convert a model to MLX format using `mlx_lm.convert` (for LLMs) or a custom conversion script
2. Push the converted weights to a repository under your own HuggingFace account
3. If the quality is validated and the model is in demand, request to move it to the `mlx-community` organization (or push directly if you have access)

[Model](../glossary.md#model) conversions are one of the highest-leverage contributions available. A converted model is immediately usable by anyone with Apple Silicon, without requiring them to run the conversion themselves. A well-documented conversion for a popular architecture can have thousands of downloads within days of posting.

**What makes a good mlx-community model:**
- Tested on at least one M-series chip
- Includes a model card with: original model source, conversion command used, and sample generation output for verification
- Multiple quantization levels (bfloat16 for quality, 4-bit for accessibility)
- Properly configured `config.json` so `mlx-lm` loads it correctly

---

## How Decisions Get Made

Understanding the governance structure matters before you spend time on a contribution.

**Apple controls the core (mlx, mlx-examples).**

The Apple ML research team makes final decisions on what merges into `mlx` and `mlx-examples`. They are the maintainers, they set the API design direction, and they control releases. PRs that change core design decisions (API shape, fundamental behavior, naming) require alignment with the team before investment.

In practice: Apple's team is responsive on GitHub. If you open an issue describing what you want to do before starting a large PR, you will almost always get a response -- either encouragement, redirect, or an explanation of why the approach does not fit. This conversation is worth having before writing 500 lines of code.

**Community drives the ports.**

For everything outside ml-explore's repositories -- mflux, mlx-audio, mlx-vlm, individual model ports -- the respective maintainers control their projects. These maintainers are accessible (most respond to GitHub issues within days), generally welcoming of contributions, and the review bar is pragmatic: does it work, is the code readable, does it follow the project's conventions.

**Hugging Face mlx-community is open.**

Contributing to mlx-community does not require approval. You upload model weights to your own HuggingFace account; if you want them listed under the mlx-community organization, you request access or ask in the community Discord. The bar is: the model loads and runs correctly.

---

## How to Engage

### GitHub Issues

The right place to start for any non-trivial contribution:

- **ml-explore/mlx** -- [github.com/ml-explore/mlx/issues](https://github.com/ml-explore/mlx/issues). Search before opening. Label your issue (bug, feature request, question). For feature requests, describe the use case, not just the feature.
- **ml-explore/mlx-examples** -- [github.com/ml-explore/mlx-examples/issues](https://github.com/ml-explore/mlx-examples/issues). Most port requests and new model discussions happen here.

**Filter by `good first issue`** to find contributions that maintainers have explicitly marked as accessible to newcomers. These are real issues -- fixing them matters -- and they come with enough context to get started.

### GitHub Discussions

mlx uses Discussions (not Slack or Discord) for design conversations:
- [github.com/ml-explore/mlx/discussions](https://github.com/ml-explore/mlx/discussions) -- framework design, API questions
- [github.com/ml-explore/mlx-examples/discussions](https://github.com/ml-explore/mlx-examples/discussions) -- model ports, mlx-lm usage

Discussions are searchable and persistent. If you have a question about how to port a specific architecture, check Discussions first -- it may already be answered.

### Pull Requests

The standard workflow:
1. Fork the repository
2. Create a branch with a descriptive name (`port/ltx-video-vae`, `fix/attention-mask-broadcast`)
3. Write the code, following the project's existing style
4. Test it (more on this below)
5. Open a PR with a clear description: what problem does this solve, how was it tested, what are the known limitations

For mlx-examples model ports specifically, the expected PR includes:
- The model implementation file(s)
- A weight conversion script
- A README or notebook demonstrating usage
- At minimum, a smoke test showing the model produces output without error

---

## Getting Started: Three Entry Points

### Entry Point 1: Model Conversions (No CUDA porting required)

The fastest way to contribute. Pick a model on Hugging Face that does not yet exist in mlx-community:

```python
# Install mlx-lm
pip install mlx-lm

# Convert an LLM (this handles weight loading, key remapping, quantization)
mlx_lm.convert \
  --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
  --mlx-path ./mistral-7b-mlx \
  --quantize \
  --q-bits 4

# Upload to HuggingFace
# (after converting and validating)
huggingface-cli upload mlx-community/Mistral-7B-Instruct-v0.3-4bit ./mistral-7b-mlx
```

For non-LLM models, the conversion is more manual -- you write a script that loads PyTorch weights and saves them in MLX format. The [Porting Guide](02-porting-guide.md) covers this in detail.

### Entry Point 2: Documentation

MLX's documentation covers the basics well. What it does not cover well:

- Tutorials for specific use cases (fine-tuning a vision-language model, building a custom data pipeline)
- Advanced topics (writing custom Metal kernels, profiling MLX workloads)
- Worked examples of porting specific model families from PyTorch to MLX

Documentation contributions go through mlx-examples (as notebooks or markdown) or the main MLX docs (as reStructuredText in `mlx/docs/`). They are reviewed quickly and have immediate visibility -- the MLX docs are read by every new user.

### Entry Point 3: Bug Reports

Apple Silicon comes in too many configurations (M1/M2/M3/M4, Pro/Max/Ultra, 8GB/16GB/32GB/64GB/96GB/128GB/192GB) for the Apple team to test all of them. Bug reports that include:

- Exact chip model and memory size
- macOS version
- MLX version (`import mlx.core as mx; print(mx.__version__)`)
- Minimal reproducible example
- Expected vs actual behavior (including any error message in full)

...are actively useful and get attention. Especially useful: reports of models that work at float16 but fail at 4-bit, or that work on M3 Max but crash on M2 Pro.

---

## Sources

- mlx GitHub organization: [github.com/ml-explore](https://github.com/ml-explore)
- mlx core repository: [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
- mlx-examples repository: [github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)
- mlx-data repository: [github.com/ml-explore/mlx-data](https://github.com/ml-explore/mlx-data)
- mlx-forge repository: [github.com/ml-explore/mlx-forge](https://github.com/ml-explore/mlx-forge)
- mlx-community on Hugging Face: [huggingface.co/mlx-community](https://huggingface.co/mlx-community)
- mlx-audio: [github.com/lucasnewman/mlx-audio](https://github.com/lucasnewman/mlx-audio)
- mflux: [github.com/filipstrand/mflux](https://github.com/filipstrand/mflux)
- mlx-vlm: [github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
- MLX documentation: [ml-explore.github.io/mlx/](https://ml-explore.github.io/mlx/)

---

## See Also

- [Porting Guide](02-porting-guide.md) -- the mechanics of converting a PyTorch/CUDA model to MLX, including the "Common Port Failure Modes" reference you will return to repeatedly once debugging starts
- [Open Opportunities](03-open-opportunities.md) -- specific gaps ranked by impact; what to work on if you don't already have a target
- [Ecosystem Overview](../02-ecosystem/01-cuda-vs-mlx-overview.md) -- the gap analysis that identifies where MLX most needs work
- [Tooling](../02-ecosystem/08-tooling.md) -- conversion scripts, profiling utilities, and debugging tools; the practical context for contributing new tooling
