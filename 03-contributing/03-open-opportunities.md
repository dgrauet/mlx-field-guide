# Open Opportunities

> **Specific gaps you can fill today, ranked by impact and accessibility.** The MLX ecosystem has known holes -- models not yet ported, tools not yet built, documentation not yet written. This page makes those gaps concrete and points you to where the work happens.

## How to Use This Page

Each opportunity below includes:
- What the gap is
- Why it matters
- Where to start (repositories, issues, or first steps)
- Difficulty estimate

Difficulty is relative to someone who knows Python and is familiar with the PyTorch/MLX API. "Hard" means C++ and [Metal](../glossary.md#metal) Shading Language are likely required, or that the problem is genuinely novel. "Accessible" means Python-only, guided by existing examples.

---

## Tier 1: High Impact, Accessible

These opportunities have the most leverage for the time invested. They do not require C++ or Metal kernel knowledge.

---

### 1. Model Conversions to MLX Community

**What the gap is:** The mlx-community Hugging Face organization has 3,000+ model repositories, but coverage is uneven. New model releases take weeks to months to appear in MLX format. Some entire model families are missing. Non-LLM model types (video, audio, multi-modal) are especially thin.

**Why it matters:** A converted model is immediately usable by every Apple Silicon user. A conversion that takes one person 2-4 hours can save thousands of users the same work. High-download models have direct impact on how many people can run ML locally on Macs.

**High-value targets (as of 2025):**
- New LLM releases in the first week after launch (Qwen, Gemma, Llama variants)
- Multi-modal models (any new VLM not yet in mlx-vlm)
- Video generation model weights (LTX-Video 0.9.7, [Matrix](../glossary.md#matrix)-Game checkpoints)
- Audio models (new Kokoro voice variants, new TTS architectures)

**Where to start:**
- [huggingface.co/mlx-community](https://huggingface.co/mlx-community) -- see what's missing
- [github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) -- conversion scripts and instructions
- The `mlx_lm.convert` command for LLMs (see the [Porting Guide](02-porting-guide.md))
- For non-LLM models: write a custom conversion script following the patterns in mlx-examples

**Difficulty:** Accessible. LLM conversions with `mlx_lm.convert` take under an hour for supported architectures. New architectures require a custom script (moderate).

---

### 2. Documentation and Tutorials

**What the gap is:** MLX's core API documentation is good. What is missing: tutorials for specific use cases, worked examples of non-trivial tasks, and documentation for advanced topics.

The most-requested documentation gaps (from GitHub Discussions):
- How to fine-tune a vision-language model with LoRA in MLX
- How to build a streaming inference pipeline with mlx-lm
- How to profile an MLX model and interpret the results
- How to write and register a custom Metal kernel (end-to-end tutorial)
- How to build a data pipeline for training with mlx-data

**Why it matters:** Documentation contributions compound. A clear tutorial for fine-tuning VLMs removes the learning barrier for hundreds of future contributors. Every person who doesn't have to ask the same question in GitHub Discussions is a sign that the docs are working.

**Where to start:**
- [github.com/ml-explore/mlx-examples/discussions](https://github.com/ml-explore/mlx-examples/discussions) -- find frequently asked questions that don't have a written answer
- [github.com/ml-explore/mlx/issues](https://github.com/ml-explore/mlx/issues) -- filter by `documentation` label
- Write a Jupyter notebook demonstrating the use case end-to-end; submit as a PR to mlx-examples

**Difficulty:** Accessible. Writing good documentation requires understanding the subject, but it does not require C++ or systems knowledge.

---

### 3. Bug Reports on Diverse Apple Silicon Configurations

**What the gap is:** MLX is tested primarily on M1/M2/M3 Max and Pro configurations. Edge cases emerge on:
- Low-memory Macs (8 GB, 16 GB) -- models that require careful memory management
- M1 vs M4 -- hardware differences in [GPU](../glossary.md#gpu) generation can expose kernel bugs
- macOS version differences -- Metal API behavior changed between macOS 13, 14, and 15
- Specific quantization + model combinations that work on one chip but not another

**Why it matters:** A well-structured bug report with a minimal reproducer lets Apple's team fix the issue. Vague reports ("it doesn't work on my Mac") do nothing. Detailed reports with chip model, OS version, MLX version, and a 10-line reproducer are actionable.

**Where to start:**
- When you find a bug, file it at [github.com/ml-explore/mlx/issues](https://github.com/ml-explore/mlx/issues) (for core MLX bugs) or [github.com/ml-explore/mlx-examples/issues](https://github.com/ml-explore/mlx-examples/issues) (for model-specific bugs)
- Template: chip model + memory, macOS version, MLX version, minimal code that reproduces the issue, expected behavior, actual behavior (with full error message)

**Difficulty:** Accessible. Finding the bug is the work; reporting it well is just discipline.

---

## Tier 2: High Impact, Moderate Difficulty

These require either deeper MLX knowledge, architecture-specific expertise, or significant engineering effort. Python-only, but non-trivial.

---

### 4. Port Missing Model Architectures to mlx-examples

**What the gap is:** mlx-examples has reference implementations for major LLM families, Stable [Diffusion](../glossary.md#diffusion), and a handful of other models. Missing architecture families include:
- Video generation models (LTX-Video as a complete, clean port; Matrix-Game)
- Multi-modal generation (image+text -> image, video instruction following)
- Retrieval-augmented generation architectures
- Specialized vision transformers (DINOv2, SAM 2, depth estimation models)
- Speech synthesis beyond what mlx-audio covers (voice cloning, codec models)

**Why it matters:** A reference implementation in mlx-examples is the canonical destination for a port. It gets code review from Apple's team, it sets the pattern for subsequent community ports, and it is directly discoverable by the community.

**Where to start:**
- [github.com/ml-explore/mlx-examples/issues](https://github.com/ml-explore/mlx-examples/issues) -- filter by `model request` or `enhancement`
- Open a discussion before starting a large port: describe what you plan to port and ask for feedback on approach
- Follow the [Porting Guide](02-porting-guide.md) for the mechanics
- Look at similar existing implementations in mlx-examples as structural templates

**Difficulty:** Moderate. The work is Python, but requires understanding the source architecture, debugging weight conversion, and validating numerical agreement with the PyTorch reference.

---

### 5. ControlNet and IP-Adapter for MLX Image Generation

**What the gap is:** mflux and the mlx-examples Stable Diffusion implementation support base image generation. Neither supports ControlNet (conditioning on edge maps, depth maps, poses) or IP-Adapter (conditioning on reference images). These are standard features in the [CUDA](../glossary.md#cuda) ecosystem (via Diffusers) and commonly requested by users of Apple Silicon image generation.

**Why it matters:** ControlNet and IP-Adapter are used in almost every serious image generation workflow -- character consistency, pose-guided generation, style transfer. Without them, Apple Silicon image generation is significantly less capable than CUDA alternatives.

**Where to start:**
- [github.com/filipstrand/mflux/issues](https://github.com/filipstrand/mflux/issues) -- ControlNet feature request
- [github.com/ml-explore/mlx-examples/issues](https://github.com/ml-explore/mlx-examples/issues) -- Stable Diffusion enhancements
- Reference: HuggingFace Diffusers ControlNet implementation as the PyTorch source to port from
- The architecture is well-documented; the work is translating the PyTorch implementation and the ControlNet weight format into MLX

**Difficulty:** Moderate. The ControlNet architecture is straightforward (a parallel encoder that injects activations into the main UNet). [Weight](../glossary.md#weight) conversion and integration into the existing inference pipeline is the bulk of the work.

---

### 6. Video Generation: Complete LTX-Video Port

**What the gap is:** LTX-Video by Lightricks is one of the highest-quality open video generation models. Partial MLX ports exist, but a complete, clean implementation with:
- Full VAE encode/decode
- Full DiT (Diffusion [Transformer](../glossary.md#transformer)) inference
- Proper temporal attention
- Validated output quality matching the PyTorch reference
- Memory-efficient inference for 16 GB and 32 GB unified memory configs

...does not yet exist as a maintained, community-approved implementation.

**Why it matters:** Video generation is the frontier of Apple Silicon ML. LTX-Video running well on Apple Silicon would demonstrate that the hardware is viable for cutting-edge generative models, not just LLMs.

**Where to start:**
- [github.com/ml-explore/mlx-forge](https://github.com/ml-explore/mlx-forge) -- where new porting work on video models is coordinated
- [github.com/Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) -- the original PyTorch implementation
- Start with the VAE (smaller, self-contained, easier to validate); then the DiT; then end-to-end inference
- Temporal attention is the hard part -- see the custom op strategies in the [Porting Guide](02-porting-guide.md)

**Difficulty:** Moderate to hard. The architecture is well-understood, but temporal attention for video requires careful decomposition (or a Metal kernel), and memory management for long video sequences on Apple Silicon requires explicit engineering.

---

### 7. mlx-lm Server Improvements

**What the gap is:** mlx-lm includes a server mode (`mlx_lm.server`) that exposes an OpenAI-compatible API for LLM inference. The current implementation is single-request (no concurrent batching), lacks streaming improvements for very long generations, and has limited KV cache management for long contexts.

**Why it matters:** mlx-lm server is how many users run local LLMs on Apple Silicon as an API backend. Improvements here affect every application built on top of it (Continue, Cursor, Open WebUI, etc. configured to use local models).

**Where to start:**
- [github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) -- the mlx-lm source
- [github.com/ml-explore/mlx-examples/issues](https://github.com/ml-explore/mlx-examples/issues) -- filter by `mlx-lm` and `server`
- Reference: vLLM's API implementation for the feature set to target; llama.cpp server for Apple Silicon-specific lessons

**Difficulty:** Moderate. The changes are Python, but require understanding async request handling, KV cache management, and inference loop mechanics.

---

## Tier 3: High Impact, Hard

These require C++, Metal Shading Language, or are genuinely open research problems. The impact is high because solving them unlocks capabilities that are currently impossible in the MLX ecosystem.

---

### 8. Custom Metal Kernels for Video Generation

**What the gap is:** Video generation models use operations that do not have efficient MLX implementations:
- **Temporal attention:** 3D attention over (batch, time, height*width, head_dim) tensors. The naive MLX implementation creates large intermediate tensors. [Flash Attention](../glossary.md#flash-attention)-style tiling over the temporal dimension does not exist in MLX.
- **3D convolutions:** `nn.Conv3d` is not in MLX. Video models (including LTX-Video's VAE) use 3D convolutions for temporal feature extraction.
- **Causal video masking:** Attention masking that enforces temporal causality in video DiT models has no optimized implementation.

**Why it matters:** These kernels are on the critical path of every video generation forward pass. Without them, video generation on Apple Silicon is either slow (large intermediate tensors) or requires approximations (decomposed temporal attention) that hurt quality or speed.

**Where to start:**
- [github.com/ml-explore/mlx-forge](https://github.com/ml-explore/mlx-forge) -- this is where custom Metal kernels for MLX are being developed
- [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx) -- the existing Metal kernel implementations are in `mlx/backend/metal/kernels/`; study these before writing new ones
- Apple's Metal Shading Language specification: [developer.apple.com/metal/](https://developer.apple.com/metal/)
- Start by writing the [CPU](../glossary.md#cpu) fallback (pure MLX operations), validate correctness, then replace with a Metal kernel

**Difficulty:** Hard. Requires C++ for the MLX primitive, Metal Shading Language for the GPU kernel, and careful validation against a reference implementation.

---

### 9. Performance Profiling Infrastructure

**What the gap is:** MLX has no equivalent to PyTorch Profiler -- no tool that records per-operation timing, memory allocation per op, or produces a Chrome trace for visualization. The existing approach is manual timing wrappers, which are imprecise and hard to use.

A proper MLX profiler would:
- Wrap `mx.eval` to record operation timing
- Export to Chrome trace format (JSON) for visualization in any browser
- Report peak memory per operation
- Attribute time to named layers (like PyTorch's `record_function` context manager)

**Why it matters:** Without profiling, optimization is guesswork. A profiler built for MLX would accelerate every serious porting and optimization effort. It would also help Apple's team identify where the framework itself has bottlenecks.

**Where to start:**
- [github.com/ml-explore/mlx/issues](https://github.com/ml-explore/mlx/issues) -- search for "profiler" to find existing discussion
- MLX's C++ API exposes Metal command buffer timing; this is the hook for kernel-level timing
- A pure Python prototype (using `time.perf_counter` around `mx.eval` calls with operation logging) is a useful starting point even without Metal integration
- Chrome trace format is documented at [docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)

**Difficulty:** Hard. The Python wrapper layer is accessible; integrating with Metal command buffer timestamps requires C++.

---

### 10. Calibration-Based Quantization

**What the gap is:** MLX's quantization is round-to-nearest -- no calibration step, no reconstruction error minimization. CUDA's quantization tools ([GPTQ](../glossary.md#gptq), AWQ) are calibration-aware: they minimize the quantization error by adjusting scales based on a small calibration dataset. At 4-bit this makes a meaningful quality difference; at 3-bit or 2-bit the difference is large.

**Why it matters:** 3-bit and 2-bit quantization would allow 70B+ parameter models to run on 64 GB unified memory Macs. Currently, 4-bit is the practical floor. Calibration-aware quantization would improve quality at every bit width.

**Where to start:**
- GPTQ paper: [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323) -- the algorithm is well-specified
- AWQ paper: [arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- AutoGPTQ implementation in PyTorch as the reference: [github.com/PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- The quantization algorithm runs layer-by-layer (Cholesky decomposition of the Hessian); implementing this in MLX requires `mx.linalg` and careful memory management
- [github.com/ml-explore/mlx-examples/issues](https://github.com/ml-explore/mlx-examples/issues) -- search for "GPTQ" or "AWQ"

**Difficulty:** Hard. The math is involved (Hessian computation, Cholesky factorization), and the implementation is sensitive to numerical stability.

---

### 11. Distributed Inference Across Multiple Macs

**What the gap is:** MLX is single-device only. No multi-GPU, no multi-machine. The largest unified memory configuration (Mac Studio M2/M4 Ultra, 192 GB) cannot run the largest open models (Llama 3 405B at bfloat16 = 810 GB). Running those models requires distributing across multiple machines.

Distributed inference would require:
- [Tensor](../glossary.md#tensor) parallelism: split attention heads across devices
- Pipeline parallelism: split layers across devices
- Communication primitives: send/receive tensors over a network or Thunderbolt

**Why it matters:** 405B+ models are inaccessible on any single Apple Silicon machine. Distributed inference across 4-8 Mac Studios would make them accessible. This is also the only path to MLX training at scale.

**Where to start:**
- [github.com/ml-explore/mlx/discussions](https://github.com/ml-explore/mlx/discussions) -- search for "distributed"; there is ongoing discussion about the design
- This is a design problem as much as an implementation problem: what communication primitives does MLX need? What topology makes sense for Mac-to-Mac (Thunderbolt? Ethernet?)
- Reference: NCCL's collective operation API as the conceptual target; Petals (collaborative inference for large models) as an alternative architecture

**Difficulty:** Hard. This is a systems engineering problem requiring new MLX primitives, not just a Python implementation. The Apple team would need to be involved in the design.

---

## Quick Wins: Things You Can Do This Week

If you want to contribute but don't have a specific project in mind, these are the fastest paths to having an impact:

```
QUICK WIN CHECKLIST

  Day 1-2:
  [ ] Run mlx-lm with a model you use regularly. File any bugs you find.
  [ ] Check mlx-community on Hugging Face for a model you use that
      isn't there yet. Run mlx_lm.convert and upload it.

  Week 1:
  [ ] Pick a GitHub Discussion in mlx-examples that has a question
      without an answer. Write the answer (or write a notebook
      demonstrating the solution and link to it).
  [ ] Run the mlx-examples Stable Diffusion or LLM inference script
      and verify it works on your chip. If it doesn't, file a bug.

  Month 1:
  [ ] Port a model architecture that is missing from mlx-community.
      Write a conversion script, validate the output, upload the weights.
  [ ] Pick one "Moderate" opportunity above and open an issue to
      discuss approach before starting implementation.
```

---

## Tracking the Ecosystem's Needs

The best way to find what the community needs right now (not just what this page lists) is to watch:

- **GitHub Issues with the most reactions:** [github.com/ml-explore/mlx-examples/issues?sort=reactions](https://github.com/ml-explore/mlx-examples/issues?sort=reactions) -- community upvotes signal demand
- **Open PRs that are stalled:** sometimes a PR is most of the way done but the author ran out of time; picking up and finishing a stalled PR is high-leverage
- **mlx-community HuggingFace:** sort by recent activity to see what models are being requested
- **GitHub Discussions with many replies but no resolution:** these are the questions that don't yet have a good answer

---

## Sources

- ml-explore GitHub org issues: [github.com/ml-explore/mlx/issues](https://github.com/ml-explore/mlx/issues), [github.com/ml-explore/mlx-examples/issues](https://github.com/ml-explore/mlx-examples/issues)
- mlx-forge: [github.com/ml-explore/mlx-forge](https://github.com/ml-explore/mlx-forge)
- mflux issues: [github.com/filipstrand/mflux/issues](https://github.com/filipstrand/mflux/issues)
- mlx-vlm: [github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
- mlx-audio: [github.com/lucasnewman/mlx-audio](https://github.com/lucasnewman/mlx-audio)
- mlx-community on Hugging Face: [huggingface.co/mlx-community](https://huggingface.co/mlx-community)
- GPTQ paper: [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- AWQ paper: [arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- LTX-Video: [github.com/Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)

---

## See Also

- [Where to Contribute](01-where-to-contribute.md) -- repositories, communities, and how to engage; the map of where to submit your work
- [Porting Guide](02-porting-guide.md) -- the mechanics of CUDA-to-MLX conversion; what to do when you pick up a "Moderate" opportunity
- [CUDA vs MLX Overview](../02-ecosystem/01-cuda-vs-mlx-overview.md) -- the full gap analysis that this page distills into actionable opportunities
- [Video Generation](../02-ecosystem/05-video-generation.md) -- the context for the LTX-Video and temporal attention opportunities
- [Tooling](../02-ecosystem/08-tooling.md) -- the context for the profiling and quantization opportunities
