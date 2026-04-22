# The MLX Field Guide

A structured learning resource for understanding generative AI concepts, the MLX framework, and how Apple's ML ecosystem compares to the CUDA world.

## Who Is This For?

Developers who work with MLX (porting models, building apps) but want to deeply understand the concepts behind what they're building. No prior ML knowledge assumed -- we start from zero and build up.

## Learning Path

### Pillar 1: Foundations

Start here. These pages build on each other in order:

1. [Tensors](01-foundations/01-tensors.md) -- the fundamental data structure
2. [Neural Networks](01-foundations/02-neural-networks.md) -- layers, weights, and activations
3. [Training vs Inference](01-foundations/03-training-vs-inference.md) -- how models learn and how they run
4. [GPU Computing](01-foundations/04-gpu-computing.md) -- why GPUs matter and how they work
5. [Transformers](01-foundations/05-transformers.md) -- the architecture behind modern AI
6. [Attention](01-foundations/06-attention.md) -- how models focus on what matters
7. [Latent Space](01-foundations/07-latent-space.md) -- compressed representations
8. [Diffusion](01-foundations/08-diffusion.md) -- how images and videos are generated
9. [Tokenization](01-foundations/09-tokenization.md) -- how text enters a model
10. [Quantization](01-foundations/10-quantization.md) -- making models smaller and faster

### Pillar 2: Ecosystem

Once you understand the foundations, explore the landscape:

1. [CUDA vs MLX Overview](02-ecosystem/01-cuda-vs-mlx-overview.md) -- two philosophies compared
2. [Frameworks](02-ecosystem/02-frameworks.md) -- PyTorch vs JAX vs MLX
3. [LLMs](02-ecosystem/03-llms.md) -- large language model tooling
4. [Image Generation](02-ecosystem/04-image-generation.md) -- Stable Diffusion, Flux, and more
5. [Video Generation](02-ecosystem/05-video-generation.md) -- LTX, CogVideo, Wan
6. [Audio](02-ecosystem/06-audio.md) -- speech and music generation
7. [Training & Fine-tuning](02-ecosystem/07-training-finetuning.md) -- LoRA, datasets, full fine-tune
8. [Tooling](02-ecosystem/08-tooling.md) -- profilers, converters, quantizers
9. [Serving & Deployment](02-ecosystem/09-serving-deployment.md) -- from laptop to production API

### Pillar 3: Contributing

Ready to help close the gap?

1. [Where to Contribute](03-contributing/01-where-to-contribute.md) -- repos, communities, decision-making
2. [Porting Guide](03-contributing/02-porting-guide.md) -- CUDA to MLX patterns
3. [Open Opportunities](03-contributing/03-open-opportunities.md) -- concrete gaps to fill

## Quick Reference

Need to look up a term fast? See the [Glossary](glossary.md).

## Contributing

When editing content, install the pre-commit hook once per clone:

```bash
pre-commit install
```

It validates internal links, glossary anchors, and markdown link structure on every commit. To run the checks manually without committing: `python3 scripts/validate_docs.py`.
