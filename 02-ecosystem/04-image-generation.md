# Image Generation: CUDA vs MLX

> **Status:** 🟢 [CUDA](../glossary.md#cuda) (Mature -- canonical diffusers library, massive community, every major model) | 🟡 MLX (Emerging -- core models work, extension ecosystem thin)

## What This Domain Covers

Generating images from text prompts or other conditioning inputs. This encompasses the full pipeline of modern diffusion-based image generation: text-to-image, image-to-image, inpainting, and conditioned generation via ControlNet or IP-Adapter. It also includes the tooling layer -- the UIs, pipeline libraries, and model hubs that make these capabilities accessible.

Image generation is the domain where the CUDA/MLX gap is most visible in everyday tooling. On CUDA, Hugging Face diffusers is the single canonical library: every research model (Stable [Diffusion](../glossary.md#diffusion) 1.5, SDXL, Flux, DALL-E 3 distillations, PixArt-Alpha) ships with a diffusers pipeline, ComfyUI uses diffusers under the hood, Automatic1111 integrates with it, and the community has built thousands of extensions, ControlNets, LoRA packs, and sampling algorithm variants on top of it.

On MLX, coverage is narrower: Flux works via mflux, Stable Diffusion 1.5 and SDXL have implementations in mlx-examples, and ComfyUI has growing MLX node support. The core generation loop works. The extension ecosystem -- inpainting, ControlNet, IP-Adapter, arbitrary LoRA loading, img2img -- is where the gap opens.

---

## The CUDA Ecosystem

### Hugging Face Diffusers

Diffusers is the central organizing library for diffusion models. Released by Hugging Face, it provides:

- Standardized `Pipeline` classes for every major model
- Modular components (schedulers, VAEs, text encoders, U-Nets, transformers) that can be swapped independently
- A scheduler library with 20+ sampling algorithms ([DDPM](../glossary.md#ddpm), [DDIM](../glossary.md#ddim), DPM++, Euler, LMS, PNDM, etc.)
- Automatic mixed precision (FP16/BF16) and memory optimization utilities
- Integration with Hugging Face Hub for model loading

```
DIFFUSERS ARCHITECTURE

  StableDiffusionPipeline (example)
      |
      +--> Text encoder (CLIP ViT-L/14)
      |      encodes prompt -> text embeddings
      |
      +--> U-Net (backbone)
      |      conditioned on text embeddings + timestep
      |      progressively denoises latent
      |
      +--> VAE (Variational Autoencoder)
      |      decoder: latent [B, 4, H/8, W/8] -> image [B, 3, H, W]
      |      encoder: image -> latent (for img2img, inpainting)
      |
      +--> Scheduler (e.g., DPMSolverMultistepScheduler)
             manages noise schedule and denoising steps
             interchangeable: change sampler without changing model

  Every component is independently loadable, replaceable, and cacheable.
  Community has built custom VAEs, custom text encoders, custom U-Nets
  as drop-in replacements.
```

**Core pipeline usage:**

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Faster sampler: swap without reloading model
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

image = pipe(
    prompt="a photorealistic image of an astronaut riding a horse on Mars",
    negative_prompt="blurry, low quality, cartoon",
    num_inference_steps=20,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]
```

### Major Models

```
CUDA IMAGE GENERATION MODELS (2024)

  Stable Diffusion 1.5 (SD1.5):
    Release: 2022
    Architecture: U-Net + CLIP text encoder + VAE
    Latent: 512x512 native (64x64 latent)
    Ecosystem: Largest. 50,000+ fine-tunes and LoRAs on Hugging Face,
               Civitai. Every extension supports SD1.5 first.
    MLX status: mlx-examples has a reference implementation

  Stable Diffusion XL (SDXL):
    Release: 2023
    Architecture: Larger U-Net, dual CLIP encoders, higher res (1024x1024)
    Ecosystem: Strong. 10,000+ fine-tunes. Refiner model for detail pass.
    MLX status: mlx-examples SDXL port; less maintained than SD1.5 port

  Stable Diffusion 3 / 3.5 (SD3):
    Release: 2024
    Architecture: Diffusion Transformer (DiT), MMDiT; replaces U-Net
    Ecosystem: Growing. Requires Stability AI license for some weights.
    MLX status: Not yet ported to MLX

  Flux (Black Forest Labs):
    Release: 2024
    Architecture: Rectified flow transformer; FLUX.1-dev, FLUX.1-schnell
    Quality: State of the art for photorealism and prompt following (2024)
    Ecosystem: Rapidly growing. ControlNet, LoRA, IP-Adapter variants.
    MLX status: mflux (filipstrand/mflux) -- fully functional

  PixArt-Alpha / PixArt-Sigma:
    Architecture: DiT (Diffusion Transformer) + T5 text encoder
    Strength: High quality, efficient training
    MLX status: Not ported

  DALL-E 3 (OpenAI):
    API only. Not locally runnable.
    MLX status: Not applicable
```

### Extensions: ControlNet, IP-Adapter, LoRA

The CUDA ecosystem's real strength in image generation is the extension layer -- functionality that adds conditioning and control on top of base models.

**ControlNet** adds structural conditioning to generation. Instead of just a text prompt, you provide a control image (edge map, depth map, pose skeleton, normal map, etc.) that constrains the composition of the generated image.

```
CONTROLNET ARCHITECTURE

  Standard generation:
    text prompt --> text encoder --> U-Net --> denoised latent --> VAE --> image

  With ControlNet:
    text prompt   --> text encoder ----\
    control image --> edge detector --> ControlNet (copy of U-Net encoder)
                                            |
                                         control activations injected
                                         into U-Net at each resolution
                                            |
                                       U-Net --> denoised latent --> VAE --> image

  Result: generated image follows the structural composition
          of the control image while matching the text prompt style.

  CUDA: 30+ ControlNet models for SD1.5, SDXL, Flux (growing)
  MLX:  No ControlNet implementation
```

**IP-Adapter** conditions generation on a reference image's style or content, rather than just text. Useful for character consistency, style transfer, and product photography.

**LoRA (Low-Rank Adaptation) for diffusion models** adds fine-tuned style or subject matter to a base model with a small (5-200 MB) adapter file. Thousands of community LoRAs exist for SD1.5 and SDXL on Civitai and Hugging Face.

```
EXTENSION ECOSYSTEM COMPARISON

  Capability          CUDA (diffusers)        MLX
  ---------           ----------------        ---
  ControlNet          30+ variants, SD1.5/    Not implemented
                      SDXL/Flux
  IP-Adapter          Multi-model support     Not implemented
  LoRA loading        Any .safetensors LoRA   Limited; mlx-lm has LoRA
                      with automatic merging  for LLMs, not diffusion
  Inpainting          StableDiffusionInpaint  Not implemented
                      Pipeline built-in
  img2img             StableDiffusionImg2Img  Not implemented in mflux;
                      Pipeline built-in       mlx-examples has partial
  Textual inversion   Supported               Not implemented
  SDXL refiner        Two-stage pipeline      Not implemented
  Upscalers           Real-ESRGAN, LDSR,      Not implemented
                      tile-based              
```

### User Interfaces

CUDA's image generation UI ecosystem is mature and large:

**ComfyUI** is the dominant node-based interface for advanced users. Every pipeline step (load model, encode prompt, sample, decode) is a node. Complex workflows (ControlNet + LoRA + img2img + upscale) are built by connecting nodes. Has a large extension ecosystem. Backend: diffusers + PyTorch, primarily CUDA.

**Automatic1111 (AUTOMATIC1111/stable-diffusion-webui)** is the classic web UI for SD1.5 and SDXL. Script-based extensions, easy LoRA loading, built-in img2img and inpainting. Historically CUDA-focused; MPS support exists but is second-class.

**InvokeAI** is a polished web UI with canvas tools, node editor, and model management. CUDA and MPS support.

**Fooocus** is a simplified interface that hides most parameters and focuses on prompt quality. Easier to use than ComfyUI.

---

## The MLX Ecosystem

### mflux

mflux (filipstrand/mflux) is the primary MLX library for Flux model inference. It is a standalone implementation -- not a diffusers port -- built from scratch for Apple Silicon.

```
MFLUX CAPABILITIES (Q1 2025)

  Supported models:
    FLUX.1-dev
    FLUX.1-schnell
    FLUX.1-fill (inpainting variant -- experimental)

  Features:
    Text-to-image generation (primary use case)
    LoRA loading (.safetensors format from Hugging Face)
    Quantization: 4-bit and 8-bit
    Controlnet: growing support (community contributed)
    img2img: partial/experimental

  Performance (M3 Max, 36GB):
    FLUX.1-schnell (4 steps): ~60-90 seconds at 1024x1024
    FLUX.1-dev (20 steps):    ~5-8 minutes at 1024x1024
    vs NVIDIA RTX 4090:       ~3-5s (schnell), ~25-40s (dev)
```

**Basic mflux usage:**

```python
from mflux import Flux1, Config

# Load FLUX.1-schnell with 4-bit quantization
flux = Flux1.from_name(
    model_name="flux-schnell",
    quantize=4,
)

image = flux.generate_image(
    seed=42,
    prompt="a photorealistic image of a fox in autumn forest, golden hour",
    config=Config(
        num_inference_steps=4,      # schnell is designed for 1-4 steps
        height=1024,
        width=1024,
    )
)
image.save("output.png")
```

**With LoRA:**

```python
from mflux import Flux1, Config, StopImageGenerationException

flux = Flux1.from_name(
    model_name="flux-schnell",
    quantize=4,
    lora_paths=["path/to/my_lora.safetensors"],
    lora_scales=[0.8],
)
```

### Stable Diffusion in mlx-examples

The mlx-examples repository contains reference implementations of Stable Diffusion 1.5 and SDXL. These are working implementations, not just proof-of-concepts, but they are not as feature-complete as diffusers:

- No ControlNet support
- No inpainting pipeline
- No img2img pipeline
- No LoRA loading for diffusion
- Fewer sampling algorithms (typically just DDIM or DPM++)

They are primarily useful as reference code for understanding how to implement diffusion models in MLX, or as a starting point for custom implementations.

### ComfyUI with MLX Nodes

ComfyUI's architecture is extensible via custom nodes. MLX-compatible nodes allow ComfyUI workflows to use Apple Silicon acceleration for specific operations.

The `ComfyUI-LTXVideo-mlx` project (one of the developer's own contributions) is an example of this pattern: LTX-Video inference exposed as ComfyUI nodes backed by MLX. This architecture allows MLX-powered models to be used within the familiar ComfyUI node editor interface, accessing the broader ComfyUI ecosystem (image loading, saving, preview, compositing) while the compute-intensive steps run via MLX.

The pattern:
1. ComfyUI provides the workflow UI, node graph, and I/O utilities
2. Custom MLX nodes wrap mlx-based model implementations
3. Tensors are converted between PyTorch (ComfyUI's native type) and MLX arrays at the boundary

```python
# ComfyUI custom node backed by MLX (pattern)
import torch
import mlx.core as mx
import numpy as np

class MLXImageGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",),
                "steps": ("INT", {"default": 20}),
                "seed": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "MLX"

    def generate(self, prompt, steps, seed):
        # Run MLX model
        mlx_output = self.mlx_model.generate(prompt, steps=steps, seed=seed)

        # Convert MLX array -> numpy -> PyTorch tensor (ComfyUI format)
        np_image = np.array(mlx_output)           # MLX -> numpy
        torch_image = torch.from_numpy(np_image)  # numpy -> PyTorch
        return (torch_image,)                      # ComfyUI expects torch
```

This conversion boundary (MLX <-> PyTorch via numpy) is the standard pattern for MLX nodes in any PyTorch-native system.

### Memory Advantage for Large Image Models

Flux at FP16 is approximately 24 GB -- exactly the limit of an RTX 4090. In practice this means on CUDA you need either:
- An RTX 4090 (24 GB, barely fits, no headroom for ControlNet or high res)
- An A100/H100 (professional, expensive)
- [Quantization](../glossary.md#quantization) to FP8 or [INT8](../glossary.md#int8) with quality trade-offs

On Apple Silicon:

```
FLUX.1-DEV MEMORY REQUIREMENTS

  Precision      Size        RTX 4090 (24GB VRAM)     M3 Max (128GB)
  ---------      ----        --------------------     --------------
  BF16           24 GB       Just fits; tight          YES, comfortable
  4-bit quant    ~6 GB       YES, comfortably          YES, very fast load
  8-bit quant    ~12 GB      YES                       YES

  Key practical difference:
    On RTX 4090 at BF16: Flux fits but ControlNet or high-res upscaling
    may push over the 24GB limit.
    On M3 Max 128GB: 24GB is 19% of available memory. ControlNet,
    LoRA, and image buffers easily fit alongside the model.
```

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **Text-to-image (Flux)** | 🟢 diffusers FluxPipeline; full feature set; 3-5s on RTX 4090 | 🟡 mflux; text-to-image works well; slower (~60-90s schnell on M3 Max) | Medium for speed; small for functionality on this specific task |
| **Text-to-image (SD1.5)** | 🟢 Canonical; every feature; 3-5s on RTX 4090 | 🟡 mlx-examples implementation; basic generation works | Medium -- basic generation works; extensions missing |
| **Text-to-image (SDXL)** | 🟢 Full pipeline; refiner; SDXL-Turbo | 🟡 mlx-examples SDXL; less maintained | Medium -- works but not as polished |
| **Text-to-image (SD3/SD3.5)** | 🟢 Diffusers support; full pipeline | 🔴 Not ported | Large -- no MLX implementation |
| **Text-to-image (PixArt)** | 🟢 Diffusers support | 🔴 Not ported | Large -- no MLX implementation |
| **ControlNet** | 🟢 30+ models; SD1.5/SDXL/Flux variants; depth, edge, pose, normal | 🟡 Experimental in mflux (community PR); not stable | Large -- ControlNet is one of the most-used features in CUDA ecosystem; MLX has no stable equivalent |
| **IP-Adapter** | 🟢 Multi-model support; face adapter, style adapter | 🔴 Not implemented | Large -- no MLX equivalent |
| **Inpainting** | 🟢 StableDiffusionInpaintPipeline; mask-conditioned generation; every model | 🟡 FLUX.1-fill experimental in mflux | Large -- inpainting is a core workflow; MLX support is very early |
| **img2img** | 🟢 StableDiffusionImg2ImgPipeline; vary strength of conditioning | 🟡 Partial in mlx-examples; not in mflux stable | Large -- common workflow for iterative refinement |
| **LoRA loading (diffusion)** | 🟢 Any .safetensors LoRA; automatic merging; thousands of community LoRAs | 🟡 mflux has LoRA support; not all LoRA formats | Medium -- basic LoRA works in mflux; Civitai LoRAs in varied formats may require conversion |
| **Sampling algorithms** | 🟢 20+ schedulers in diffusers (Euler, DPM++, DDIM, LMS, PNDM, etc.) | 🟡 Fewer options; typically DDIM or DPM++ | Small for basic use; noticeable for advanced users who tune samplers |
| **ComfyUI integration** | 🟢 Native; all models supported; massive node library | 🟡 MLX custom nodes exist (including developer's own work); not all models covered | Medium -- ComfyUI works with MLX for supported models; the 10,000+ CUDA-native node extensions do not work |
| **Web UI (Automatic1111)** | 🟢 Full support; plugins; model management | 🔴 No MLX backend | Large -- Automatic1111 has MPS support via PyTorch, not MLX |
| **[Model](../glossary.md#model) diversity** | 🟢 Tens of thousands of fine-tunes, styles, characters | 🟡 Limited to what mflux/mlx-examples implement; growing LoRA support | Large -- the long tail of specialized models is CUDA-only |
| **Upscaling (ESRGAN, tile)** | 🟢 Real-ESRGAN, BSRGAN, tile-based upscaling in Automatic1111/ComfyUI | 🔴 No MLX upscaling library | Medium -- post-processing step; can run on [CPU](../glossary.md#cpu) or via separate tool |
| **Generation speed** | 🟢 RTX 4090: ~3-5s (20 steps, 512x512 SD1.5); ~25-40s (Flux dev 1024x1024) | 🟡 M3 Max: ~30s (SD1.5 20 steps); ~300s (Flux dev 1024x1024) | Large -- CUDA is 5-10x faster for equivalent tasks; speed gap narrows on M4 Ultra |
| **Textual inversion** | 🟢 Standard feature in diffusers and Automatic1111 | 🔴 Not implemented | Small -- textual inversion is less used since LoRA dominates |
| **SDXL refiner** | 🟢 Two-stage pipeline (base + refiner) | 🔴 Not implemented for MLX SDXL | Small -- mostly superseded by Flux for quality |

---

## Performance Deep Dive

Speed is the most noticeable gap in image generation. Here is a realistic comparison:

```
IMAGE GENERATION PERFORMANCE (text-to-image, approximate)

  MODEL: Stable Diffusion 1.5, 512x512, 20 steps, DPM++ 2M Karras

  Hardware              Time          Notes
  --------              ----          -----
  NVIDIA RTX 4090       2-4s          FP16; fastest consumer GPU
  NVIDIA RTX 3090       4-6s          FP16; still very fast
  NVIDIA RTX 4080       3-5s          FP16
  Apple M3 Max          25-35s        MLX FP16; 7-10x slower than RTX 4090
  Apple M2 Ultra        20-30s        MLX FP16; larger bandwidth helps

  MODEL: FLUX.1-schnell, 1024x1024, 4 steps

  Hardware              Time          Notes
  --------              ----          -----
  NVIDIA RTX 4090       3-6s          BF16; Flux just fits in 24GB
  Apple M3 Max (128GB)  60-100s       mflux 4-bit; 15-20x slower
  Apple M4 Max (128GB)  50-80s        mflux 4-bit; improved hardware

  PRACTICAL CONTEXT:
  For one-off image generation, 60-100s per image is usable.
  For batch generation, product photography pipelines, or iterative
  design work where dozens of images are needed, the speed gap
  is a real workflow constraint.
  
  The memory advantage inverts: Flux at BF16 is tight on RTX 4090
  (24GB) but comfortable on M3 Max (128GB). High-res generation
  (2048x2048) or multi-batch generation may OOM on RTX 4090
  where it runs fine on Apple Silicon.
```

---

## Contribution Opportunities

**Stable ControlNet for mflux.** ControlNet support is the single most-requested feature for MLX image generation. The community has started PRs in mflux. Completing the implementation -- handling multiple control types (Canny edge, depth, pose), proper conditioning injection into the Flux transformer blocks, and testing against the CUDA reference implementation -- would close the largest practical gap for power users.

**Diffusers MLX backend.** The highest-leverage architectural contribution: an official MLX backend for Hugging Face diffusers. This would mean that every new model added to diffusers (SD3, PixArt-Sigma, future models) automatically runs on Apple Silicon via MLX, without requiring a separate port. The precedent is the MPS backend -- but an MLX backend would be more complete and better performing. This is a large effort but would benefit the entire ecosystem.

**Inpainting pipeline for mflux.** FLUX.1-fill (the inpainting variant of Flux) exists. Completing a stable inpainting pipeline in mflux -- with proper mask conditioning, mask blur, padding, and strength control -- would add a core workflow that is currently missing.

**img2img pipeline.** Image-to-image generation (provide a starting image + prompt, generate a variation) is a fundamental diffusion workflow. Implementing it in mflux requires adding the VAE encoding step and noise addition to the current text-to-image pipeline. Conceptually straightforward; practically impactful.

**LoRA format compatibility.** CUDA-ecosystem LoRAs come in varied formats (.safetensors with different key schemas depending on the training tool). Improving mflux's LoRA loading to handle more of these format variants (Kohya, Hugging Face, Civitai) would allow the large existing library of community LoRAs to be used on Apple Silicon.

**ComfyUI MLX node library.** The ComfyUI-LTXVideo-mlx project demonstrates the pattern for MLX nodes in ComfyUI. Generalizing this into a shared library of MLX utilities for ComfyUI node authors (standard tensor conversion, model caching, memory management) would lower the barrier for future MLX ComfyUI nodes.

**Performance optimization for SD1.5.** The mlx-examples Stable Diffusion implementation is a reference, not an optimized production implementation. Profiling the bottlenecks (VAE encode/decode, attention in the [U-Net](../glossary.md#u-net), scheduler steps) and optimizing with custom [Metal](../glossary.md#metal) kernels or better fusion strategies could narrow the performance gap significantly.

---

## Sources

- Hugging Face diffusers: [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers). The canonical library for CUDA diffusion models; the benchmark against which MLX coverage is measured.
- mflux: [github.com/filipstrand/mflux](https://github.com/filipstrand/mflux). The primary MLX library for Flux model inference.
- mlx-examples (Stable Diffusion): [github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples). Apple's reference implementations for SD1.5 and SDXL.
- ComfyUI: [github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI). The node-based UI framework; extensible with MLX custom nodes.
- ControlNet paper (Zhang et al., 2023): "Adding Conditional Control to Text-to-Image Diffusion Models." The foundational paper for structural conditioning.
- IP-Adapter paper (Ye et al., 2023): "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models."
- Flux model (Black Forest Labs): [blackforestlabs.ai](https://blackforestlabs.ai). FLUX.1-dev and FLUX.1-schnell model cards and weights.
- Stability AI: [stability.ai](https://stability.ai). SD1.5, SDXL, SD3 model cards and documentation.
- LoRA for diffusion (Ryu, 2022): The LoRA technique applied to diffusion models; basis for community LoRA fine-tuning.

---

## See Also

- [Serving & Deployment](09-serving-deployment.md) -- serving image models (mflux-server and peers); different request shape from LLM serving
- [Diffusion](../01-foundations/08-diffusion.md) -- the mathematical basis of diffusion models; noise schedules, score matching, DDPM/DDIM; essential for understanding why sampling algorithm choice matters
- [Latent Space](../01-foundations/07-latent-space.md) -- explains VAEs and why diffusion happens in latent space rather than pixel space; the 8x spatial compression that makes U-Net computation tractable
- [Attention](../01-foundations/06-attention.md) -- cross-attention is how text conditioning works in SD1.5/SDXL; self-attention is the backbone of Flux's DiT architecture
- [Transformers](../01-foundations/05-transformers.md) -- Flux uses a Diffusion [Transformer](../glossary.md#transformer) (DiT) rather than a U-Net; understanding transformers helps with understanding the architecture shift
- [Quantization](../01-foundations/10-quantization.md) -- mflux uses 4-bit quantization to fit Flux in practical memory; explains the quality/memory trade-offs
- [Frameworks](02-frameworks.md) -- PyTorch/diffusers vs MLX; the pattern for porting diffusers pipelines to MLX
- [Video Generation](05-video-generation.md) -- the adjacent domain where image generation techniques extend to temporal sequences; where the MLX gap is widest
- [LLMs](03-llms.md) -- the domain where MLX is strongest; contrast with image generation to understand where MLX ecosystem investment has been prioritized
