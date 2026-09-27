# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Foundations page 16, KV Cache & Inference Optimization: prefill vs decode, bandwidth ceiling, KV cache sizing (GQA, sliding window, `--kv-bits`), prompt caching, continuous batching, speculative decoding, with measurements on an M2 Pro (mlx-lm 0.31.3, Gemma 3 12B 4-bit).
- Foundations page 15, Lazy Evaluation: evaluation triggers, `mx.eval` placement, `mx.compile` rules, memory and timing measurements (MLX 0.31.1).
- Diffusion: section on sigma (σ), the noise level scheduler code iterates over, with both conventions (Euler/Karras and flow matching), `shift`, and porting mistakes.
- Embeddings & RAG: centroids and k-means, and how IVF indexes use them; Quantization: note on codebook (centroid-based) quantization.
- Glossary: Softmax, Sigma (σ), Centroid, k-means.

### Fixed

- LLMs, Serving, Frameworks, Open Opportunities: `mlx_lm.server` was described as single-request with experimental speculative decoding; it now has continuous batching, prompt caching and `--draft-model`.
- Attention: LLaMA-7B KV cache at 4K tokens is 2.15 GB, not 1.07 GB. Quantization: 70B example (no M2 Ultra MacBook Pro; KV cache ~2.7 GB with GQA).
- Several pages claimed MLX fuses kernels automatically through lazy evaluation; fusion requires `mx.compile` (measured 29.1 ms → 3.3 ms on a 40-op element-wise chain).
