# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Foundations page 28, LoRA & Fine-Tuning in Depth: scale conventions (PEFT alpha/r vs mlx-lm scale 20 vs ltx-trainer), PEFT↔MLX adapter transpose, QLoRA, fusing into a 4-bit base (when re-quantization erases an adapter, measured), training memory incl. MLX's same-dtype AdamW state, gradient checkpointing (measured on Gemma 3 1B).
- Foundations page 27, Custom Metal Kernels: `mx.fast.metal_kernel` anatomy, when it beats `mx.compile`/built-ins (measured), grid semantics, `init_value`, atomics, non-contiguous inputs, template recompiles, math modes, gradients; grounded in mlx-arsenal's rasterizer.
- Foundations page 26, Apple Silicon Memory & Metal Limits: device limits from `mx.device_info()`, MLX's buffer cache, memory/cache/wired limits (behaviour tested on MLX 0.32.3), RSS vs physical footprint, `max_buffer_length`, command-buffer splitting (counted with smeltr) and the GPU watchdog, grounded in the LTX-2, Hunyuan3D and Matrix-Game ports.
- KV Cache & Inference: chunked prefill (`prefill_step_size`, measured: smaller chunks cut peak memory with no speed loss on an M2 Pro) and response prefill (`--prefill-response`, `continue_final_message`), tested on mlx-lm 0.32.0.
- Foundations page 25, Checkpoints & Weight Formats: formats and pickle safety (tested on torch 2.14.1), safetensors layout, lazy `mx.load`, sharding and a stale index found in `mlx-community/gemma-3-12b-it-4bit`, bf16 vs NumPy failures, MLX quantized layout, strict loading and `sanitize()`.
- Foundations page 24, Self-Supervised Learning & World Models: families of self-supervised objectives, V-JEPA 2 (context/target encoders, EMA, target LayerNorm, probes, action-conditioned predictor), Matrix-Game (action module, camera-aware memory), a representation-collapse demonstration in MLX 0.32.3, and the EMA-checkpoint loading trap.
- Foundations page 23, Conditioning & Guidance: five injection mechanisms (incl. inpainting channel counts from configs and LTX's denoise mask / appended keyframes), CFG / STG / modality guidance / rescale as implemented in the LTX-2 MLX port, sigma-binned schedules, and measured batched-vs-sequential CFG cost on MLX 0.32.2.
- Testing & Validation: measuring image similarity with PSNR and SSIM (reference points measured on MLX 0.32.2, data-range and seed traps); glossary entries PSNR and SSIM.
- Foundations page 22, Audio Representations: waveform and sample rate, STFT and mel spectrograms (Whisper recipe reproduced in MLX 0.32.2, with the error of each common deviation), vocoders, neural codecs and RVQ (demonstrated in MLX), codec frame rates from configs.
- Foundations page 21, VAEs in Depth: compression factors per model family, mean vs sample, three latent-normalization conventions (values from configs), fp16 upcasting, and measured memory for full, per-block-eval and tiled decoding.
- Foundations page 20, 3D Representations: meshes, point clouds, voxels, SDF vs occupancy, marching cubes, latent-set 3D generation (Hunyuan3D-2.1), hierarchical volume decoding, and 3D port failures, with measurements.
- Foundations page 19, Convolutions & Patchification: conv arithmetic, causal 3D convs, transposed convs, pixel shuffle, patch embedding, and a PyTorch → MLX weight-layout table verified on MLX 0.32.2 (incl. a grouped transposed-convolution bug in 0.31.1).
- Foundations page 18, Mixture of Experts: routing variants (Mixtral, Qwen MoE, DeepSeek-V3), shared experts, `gather_mm`/`SwitchGLU`, a measured MoE-vs-dense layer benchmark, and MoE port failures.
- Foundations page 17, Sampling & Decoding: greedy, temperature, top-k/top-p/min-p, penalties, seeds; three mlx-lm vs transformers mismatches (ignored `generation_config.json`, temperature/filter order, repetition-penalty window), checked in both sources.
- Foundations page 16, KV Cache & Inference Optimization: prefill vs decode, bandwidth ceiling, KV cache sizing (GQA, sliding window, `--kv-bits`), prompt caching, continuous batching, speculative decoding, with measurements on an M2 Pro (mlx-lm 0.31.3, Gemma 3 12B 4-bit).
- Foundations page 15, Lazy Evaluation: evaluation triggers, `mx.eval` placement, `mx.compile` rules, memory and timing measurements (MLX 0.31.1).
- Diffusion: section on sigma (σ), the noise level scheduler code iterates over, with both conventions (Euler/Karras and flow matching), `shift`, and porting mistakes.
- Embeddings & RAG: centroids and k-means, and how IVF indexes use them; Quantization: note on codebook (centroid-based) quantization.
- Glossary: Softmax, Sigma (σ), Centroid, k-means.

### Fixed

- mlx-forge was linked as `github.com/ml-explore/mlx-forge` (404) in 7 places and described as an ml-explore hub for porting and Metal kernels; it is a community conversion tool at `github.com/dgrauet/mlx-forge`. Porting Guide / Frameworks now recommend `mx.fast.metal_kernel` before a C++ extension (nanobind, not pybind11). Open Opportunities #8 and the CUDA-vs-MLX overview no longer claim MLX lacks Conv3d, causal masking or memory-efficient attention (fused SDPA measured at 0.2 GB for 16K tokens).
- Tooling: "no MLX-native kernel profiler exists" now mentions `mx.metal.start_capture` (Xcode Metal debugger) and smeltr; per-kernel hardware counters remain the gap.
- Audio: the Whisper diagram said the mel spectrogram has T/2 frames; it has 3000 frames per 30 s, halved to 1500 by the encoder's conv stem.
- Porting Guide: MLX has had `nn.Conv3d` (and transposed convolutions) for a long time; the guide said it didn't.
- LLMs, Serving, Frameworks, Open Opportunities: `mlx_lm.server` was described as single-request with experimental speculative decoding; it now has continuous batching, prompt caching and `--draft-model`.
- Attention: LLaMA-7B KV cache at 4K tokens is 2.15 GB, not 1.07 GB. Quantization: 70B example (no M2 Ultra MacBook Pro; KV cache ~2.7 GB with GQA).
- Several pages claimed MLX fuses kernels automatically through lazy evaluation; fusion requires `mx.compile` (measured 29.1 ms → 3.3 ms on a 40-op element-wise chain).

### Changed

- Pages 15-18 re-measured on MLX 0.32.3 / mlx-lm 0.32.0. Same conclusions; `--kv-bits` in `mlx_lm.server` is attributed to mlx-lm 0.32.0 (it was not in 0.31.3); KV cache size now distinguishes stored (462 MB) from allocated (470 MB) bytes.
- Pages 15-18 re-measured on MLX 0.32.2 (latest release) instead of 0.31.1. Conclusions unchanged; speculative decoding on the M2 Pro test setup now shows no gain at k=2 (was +3%).
