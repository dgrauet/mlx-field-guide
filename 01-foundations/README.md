# Foundations

Core concepts of machine learning and generative AI, explained from scratch.

## Reading Order

These pages build on each other. If you're starting from zero, read them in order. If you're looking up a specific concept, each page lists its prerequisites in the "See Also" section.

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 1 | [Tensors](01-tensors.md) | The basic data structure behind all ML |
| 2 | [Neural Networks](02-neural-networks.md) | How layers of math learn patterns |
| 3 | [Training vs Inference](03-training-vs-inference.md) | The difference between learning and using what was learned |
| 4 | [GPU Computing](04-gpu-computing.md) | Why ML needs specialized hardware |
| 5 | [Transformers](05-transformers.md) | The architecture that powers modern AI |
| 6 | [Attention](06-attention.md) | The key mechanism inside transformers |
| 7 | [Latent Space](07-latent-space.md) | Why and how we compress representations |
| 8 | [Diffusion](08-diffusion.md) | How generative models create images and video |
| 9 | [Tokenization](09-tokenization.md) | How text gets converted into numbers |
| 10 | [Quantization](10-quantization.md) | Making models smaller without losing quality |
| 11 | [Positional Encoding](11-positional-encoding.md) | How models know token order; RoPE |
| 12 | [Numerical Stability](12-numerical-stability.md) | Why models produce NaN, inf, and black images |
| 13 | [Loss Functions](13-loss-functions.md) | What the loss was, and why it shapes the output |
| 14 | [Normalization](14-normalization.md) | LayerNorm vs RMSNorm vs GroupNorm, and port traps |
| 15 | [Lazy Evaluation](15-lazy-evaluation.md) | When MLX actually computes; mx.eval placement and mx.compile |
| 16 | [KV Cache & Inference Optimization](16-kv-cache-inference.md) | Why decode is bandwidth-bound, and which speed-ups pay off on a Mac |
| 17 | [Sampling & Decoding](17-sampling-decoding.md) | From logits to a token; default and order mismatches between libraries |
| 18 | [Mixture of Experts](18-mixture-of-experts.md) | Routers, active vs total parameters, MoE port traps |
| 19 | [Convolutions & Patchification](19-convolutions.md) | Conv layouts, causal 3D and transposed convs, patch embedding |
| 20 | [3D Representations](20-3d-representations.md) | Meshes, SDFs, marching cubes, latent 3D generation |
| 21 | [VAEs in Depth](21-vae.md) | Latent normalization per model family, fp16, tiling, VAE port traps |
| 22 | [Audio Representations](22-audio-representations.md) | Waveforms, mel spectrograms, codecs, and preprocessing parity |
| 23 | [Conditioning & Guidance](23-conditioning-guidance.md) | How conditions enter a diffusion model; CFG, STG, rescale, schedules |
| 24 | [Self-Supervised Learning & World Models](24-self-supervised-world-models.md) | JEPA, EMA target encoders, probes, action-conditioned world models |
| 25 | [Checkpoints & Weight Formats](25-checkpoints-weight-formats.md) | safetensors, sharding, bf16 vs NumPy, quantized layouts, strict loading |
| 26 | [Apple Silicon Memory & Metal Limits](26-memory-metal-limits.md) | GPU working set, MLX cache and limits, command buffers, the GPU watchdog |
| 27 | [Custom Metal Kernels](27-custom-metal-kernels.md) | Writing GPU kernels with mx.fast.metal_kernel, and their pitfalls |

## Prerequisites

None. This pillar assumes no prior ML knowledge.
