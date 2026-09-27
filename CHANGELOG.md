# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Foundations page 15, Lazy Evaluation: evaluation triggers, `mx.eval` placement, `mx.compile` rules, memory and timing measurements (MLX 0.31.1).

### Fixed

- Several pages claimed MLX fuses kernels automatically through lazy evaluation; fusion requires `mx.compile` (measured 29.1 ms → 3.3 ms on a 40-op element-wise chain).
