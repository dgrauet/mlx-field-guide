# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Foundations page 15, Lazy Evaluation: evaluation triggers, `mx.eval` placement, `mx.compile` rules, memory and timing measurements (MLX 0.31.1).
- Diffusion: section on sigma (σ), the noise level scheduler code iterates over, with both conventions (Euler/Karras and flow matching), `shift`, and porting mistakes.
- Embeddings & RAG: centroids and k-means, and how IVF indexes use them; Quantization: note on codebook (centroid-based) quantization.
- Glossary: Softmax, Sigma (σ), Centroid, k-means.

### Fixed

- Several pages claimed MLX fuses kernels automatically through lazy evaluation; fusion requires `mx.compile` (measured 29.1 ms → 3.3 ms on a 40-op element-wise chain).
