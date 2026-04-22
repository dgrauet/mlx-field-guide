# Video Generation: CUDA vs MLX

> **Status:** 🟡 [CUDA](../glossary.md#cuda) (Emerging -- core models work, tooling maturing fast) | 🔴 MLX (Early -- pioneer territory; ltx-2-mlx is the only full-featured port)

## What This Domain Covers

Generating video from text prompts, reference images, or existing video clips. This encompasses the full pipeline of modern video diffusion models: text-to-video, image-to-video, video-to-video stylization, and multi-frame generation with temporal consistency.

Video generation is the newest frontier in generative AI. Where image generation (SD1.5, SDXL, Flux) had several years to mature on CUDA before anyone ported to MLX, video generation is still actively being built on CUDA -- and simultaneously, the first MLX ports are emerging in parallel. This means the domain is moving fast on both sides.

The CUDA ecosystem has multiple competing architectures, each with different trade-offs: DiT-based flow matching (LTX-Video, Wan), 3D VAE approaches (CogVideoX), and full transformer pipelines (HunyuanVideo). On Apple Silicon, this domain is pioneer territory: ltx-2-mlx (a full port of LTX-Video 2 to MLX) is the primary implementation, and ComfyUI-LTXVideo-mlx brings it into the ComfyUI workflow environment.

This page goes deeper than other ecosystem pages because video generation is the primary domain of the developer using this guide.

---

## The CUDA Ecosystem

### The Architecture Shift: Why Video Is Hard

Video generation inherits all the complexity of image generation and adds a temporal dimension that requires reasoning across frames. This is not simply running an image model multiple times:

```
IMAGE vs VIDEO: WHAT CHANGES

  Image model:
    Input:    text prompt
    Output:   [B, C, H, W]   -- batch, channels, height, width
    Latent:   [B, 4, H/8, W/8]
    Attention: spatial only (each token attends to all spatial positions)

  Video model:
    Input:    text prompt + (optional) reference image + (optional) reference video
    Output:   [B, C, T, H, W]   -- adds time dimension T (number of frames)
    Latent:   [B, 4, T/k, H/8, W/8]   -- temporal compression k (model-dependent)
    Attention: spatial AND temporal -- each token must attend across frames

  CHALLENGES:
    Memory:   A 121-frame video is ~121x a single image in raw data
              Temporal attention adds quadratic cost in T
    Compute:  Each denoising step processes the full spatiotemporal volume
    Consistency: Objects must be coherent across frames (no flickering)
    Motion:   Text must control not just appearance but movement
```

**Temporal attention** is the key architectural innovation. Where image transformers attend spatially (pixel to pixel within a frame), video transformers add temporal attention (same spatial position across different frames, or full 3D attention). Different models handle this differently: some use alternating spatial/temporal attention layers, others use full 3D attention in a patchified spatiotemporal latent.

### LTX-Video (Lightricks)

LTX-Video is a DiT-based ([Diffusion](../glossary.md#diffusion) [Transformer](../glossary.md#transformer)) video generation model using flow matching. It is notable for being the fastest high-quality open video model at release (Q4 2024), and for the active open-source ecosystem around it.

```
LTX-VIDEO ARCHITECTURE

  Text encoder:  T5-XXL (4096-dim, 512 token context)
                 Encodes the text prompt into conditioning signal

  VAE:           Temporal-spatial VAE
                 Temporal compression: 8x (8 frames -> 1 latent frame)
                 Spatial compression:  32x (higher than SD's 8x)
                 Output: [B, C, T/8, H/32, W/32]
                 Result: very compact latent space -> fast denoising

  Transformer:   DiT architecture; patchified spatiotemporal latent
                 Flow matching (not DDPM): straight-line paths in
                 latent space; fewer denoising steps needed
                 Conditioning: text via cross-attention and AdaLN

  Scheduler:     Flow matching scheduler
                 Can generate quality video in 25-40 steps
                 (DDPM-style models may need 50-100+ steps for video)

  VERSIONS:
    LTX-Video (v0.9):   Original release, 2B parameters
    LTX-Video 2 (v0.9.5): Improved quality, same architecture
    LTX-Video 2.1:      Higher resolution, longer duration
```

**Speed advantage:** LTX-Video was the first model that could generate 5-second, 768x512 video in under 30 seconds on an RTX 4090. This came from the aggressive VAE compression (32x spatial) reducing the latent size that the transformer processes.

**Modes:**
- Text-to-video: prompt -> video
- Image-to-video: reference image + prompt -> animated video

**MLX status:** ltx-2-mlx (the developer's own port) is the full MLX implementation of LTX-Video 2.

### CogVideoX (THUDM)

CogVideoX uses a 3D VAE approach that captures temporal correlations at the VAE level rather than relying entirely on the transformer to model time.

```
COGVIDEOX ARCHITECTURE

  3D VAE:
    Unlike standard 2D VAEs that process each frame independently,
    CogVideoX's 3D VAE processes the video as a spatiotemporal volume.
    
    Encoder: [B, C, T, H, W] -> [B, 4, T/4, H/8, W/8]
    Temporal compression: 4x (lighter than LTX-Video's 8x)
    Spatial compression: 8x (same as SD1.5)

    Key: the VAE itself encodes temporal correlations,
    so the latent is already "video-aware" before the transformer.

  Expert Transformer:
    Full 3D attention across the spatiotemporal latent
    Interleaved expert blocks for text and video conditioning

  VARIANTS:
    CogVideoX-2B:  2B parameters; faster; lower quality ceiling
    CogVideoX-5B:  5B parameters; higher quality; requires more VRAM

  MEMORY: CogVideoX-5B at FP16 requires ~24GB VRAM -- tight on RTX 4090
```

CogVideoX is fully integrated in Hugging Face diffusers, making it accessible to any developer already using diffusers for image generation.

### Wan (Alibaba/Wan Team)

Wan (2025) represents the current state of the art for open-source video generation quality. It scales the DiT approach to larger model sizes and longer training compute.

```
WAN ARCHITECTURE

  Text conditioning:  umt5-xxl (multilingual, 4096-dim)
  VAE:               3D causal VAE
                     Causal: each frame's latent depends on previous frames
                     This enforces temporal consistency structurally

  Transformer:       Wan Transformer (attention + feed-forward)
                     Full spatiotemporal attention
                     Cross-frame conditioning via temporal self-attention

  VARIANTS:
    Wan-T2V-1.3B:   Text-to-video, 1.3B params; good for fast iteration
    Wan-T2V-14B:    Text-to-video, 14B params; best quality
    Wan-I2V-14B:    Image-to-video, 14B params
    Wan-VACE-14B:   Video understanding + generation

  QUALITY CEILING: Among the best open-source video models as of 2025.
  COST: 14B model is heavy -- needs 40+ GB VRAM for FP16 inference.
        4-bit quantization brings it to ~7-8 GB.

  MLX status: No MLX port.
```

### HunyuanVideo (Tencent)

HunyuanVideo is Tencent's open-source video generation model, notable for very high resolution support and strong motion quality.

```
HUNYUANVIDEO HIGHLIGHTS

  Architecture:  Dual-stream transformer (text and video encoded separately
                 then merged, similar to Flux's dual-stream design)
  Parameters:    13B -- one of the largest open video models
  Resolution:    Up to 720p native; 1080p possible with tiling
  Duration:      Up to 10+ seconds
  Strengths:     Strong motion dynamics, high resolution

  REQUIREMENTS:  80GB VRAM recommended for FP16 full inference
                 24GB possible with quantization + aggressive optimization
                 
  Tools:         HunyuanVideo-I2V (image-to-video variant)
                 FastVideo (community optimization project)
                 ComfyUI nodes from Kijai and others

  MLX status: No MLX port.
```

### Mochi (Genmo)

Mochi is Genmo's open-source video model, focused on motion quality and natural-looking dynamics.

```
MOCHI HIGHLIGHTS

  Architecture:  AsymDiT (Asymmetric Diffusion Transformer)
                 Separate streams for video tokens and text tokens
                 Text tokens attend to video but not vice versa
  Parameters:    10B
  Strength:      Natural motion, particularly for human movement
  Weakness:      Slower generation; lower resolution than HunyuanVideo

  Hugging Face:  diffusers MochiPipeline (integrated)
  MLX status:    No MLX port.
```

### The CUDA Video Tooling Stack

```
CUDA VIDEO GENERATION TOOLING (2025)

  Hugging Face diffusers:
    Integrated pipelines for: LTX-Video, CogVideoX, Wan, HunyuanVideo,
    Mochi, AnimateDiff, ModelScopeT2V, SVD (Stable Video Diffusion)
    Standard VideoGenerationPipeline API
    Automatic mixed precision, memory-efficient attention
    
  ComfyUI video nodes:
    Kijai's nodes: HunyuanVideo, Wan, LTX-Video, CogVideoX, Mochi
    LightricksComfyUI: official LTX-Video nodes (CUDA)
    Frame interpolation nodes (RIFE, FILM)
    Video input/output (load, save, preview)
    
  Gradio demos and Spaces:
    Most video models have HF Spaces demos runnable without local GPU
    
  Video post-processing:
    RIFE: Real-Time Intermediate Flow Estimation (frame interpolation)
    FILM: Frame Interpolation for Large Motion
    Real-ESRGAN video: upscaling
    These are post-process tools, not generation -- run after the model

  NOTE: Even on CUDA, video generation is "emerging" -- the tooling
  is building out now (2024-2025). It does not have the 3-year head
  start that image generation had.
```

**AnimateDiff** deserves special mention: it is not a standalone video model but an adapter that adds temporal consistency to image diffusion models (SD1.5, SDXL). It inserts temporal attention layers into an existing [U-Net](../glossary.md#u-net), allowing the large SD1.5 LoRA and fine-tune ecosystem to generate animated sequences. This made video generation accessible before purpose-built video models existed.

```
ANIMATEDIFF (CUDA)

  Concept: Add temporal motion modules to SD1.5 or SDXL U-Net
  
  Benefits:
    - All SD1.5 LoRAs, styles, character models work for video
    - Huge existing ecosystem repurposed for video
    - Lower VRAM requirements than purpose-built video models

  Limitations:
    - Short clips only (16-32 frames typically)
    - Lower quality motion than purpose-built models
    - Inherited SD1.5 resolution limits (512x512 native)

  MLX status: No AnimateDiff port for MLX. Requires porting the
              temporal attention modules and the motion module
              weights into an MLX SD1.5 implementation.
```

---

## The MLX Ecosystem

### ltx-2-mlx

ltx-2-mlx is the developer's own port of LTX-Video 2 to Apple Silicon via MLX. This is the most complete open-source video generation implementation for MLX, and as of 2025, the primary MLX option for local video generation.

```
LTX-2-MLX ARCHITECTURE MAPPING

  Component          Original (PyTorch/CUDA)        MLX port
  ---------          ----------------------         --------
  Text encoder       T5-XXL (Hugging Face)          Loaded via transformers;
                                                     outputs converted to mx.array

  VAE encoder        CausalVideoVAE (PyTorch)        Ported to mlx.nn.Module
  VAE decoder        CausalVideoVAE (PyTorch)        Ported to mlx.nn.Module

  Transformer        LTX DiT (PyTorch)              Ported to mlx.nn.Module
                     ~2B parameters                  4-bit quantizable via
                                                     mlx.core.quantize

  Flow matching      scheduler in diffusers format   Reimplemented in MLX
  scheduler

  WEIGHT LOADING:
    Weights stored in safetensors format from Hugging Face
    Loaded via safetensors library, converted to mx.array
    Optional 4-bit quantization on load to reduce memory footprint
```

**Key implementation decisions:**

The VAE uses causal convolutions along the temporal dimension -- each frame's encoding depends on previous frames but not future ones. In PyTorch, this is straightforward with `nn.Conv3d`. In MLX, the equivalent is built from `mlx.nn.Conv3d` with appropriate padding management:

```python
# Causal temporal convolution pattern in MLX
import mlx.core as mx
import mlx.nn as nn

class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        # Temporal padding: pad only on the "past" side
        kt, kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*3
        self.temporal_pad = kt - 1   # causal: pad only the beginning
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(kt, kh, kw),
            stride=stride,
            padding=(0, kh//2, kw//2)  # no temporal padding in conv itself
        )

    def __call__(self, x):
        # x: [B, T, H, W, C]  (MLX uses channels-last)
        # Pad temporal dimension at the beginning only (causal)
        B, T, H, W, C = x.shape
        pad = mx.zeros([B, self.temporal_pad, H, W, C])
        x_padded = mx.concatenate([pad, x], axis=1)
        return self.conv(x_padded)
```

**The PyTorch-to-MLX conversion boundary** is where ltx-2-mlx handles the weight loading:

```python
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

def load_transformer_weights(model_path: str, model: nn.Module, quantize: int = None):
    """Load safetensors weights into MLX module, with optional quantization."""
    weights = {}
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            # Load as PyTorch tensor, convert to numpy, then to MLX array
            pt_tensor = f.get_tensor(key)
            weights[key] = mx.array(pt_tensor.numpy())

    # Remap key names if needed (PyTorch -> MLX naming differences)
    remapped = remap_keys(weights)
    model.load_weights(list(remapped.items()))

    if quantize:
        # Quantize linear layers to reduce memory
        nn.quantize(model, bits=quantize)

    # Force materialization of all parameters
    mx.synchronize()
    return model
```

**Generation pipeline:**

```python
import mlx.core as mx
from ltx_mlx import LTXVideoMLX, FlowMatchingScheduler

# Load model (4-bit quantization reduces ~8GB model to ~2GB)
model = LTXVideoMLX.from_pretrained(
    "Lightricks/LTX-Video",
    quantize=4
)

# Text conditioning
text_embeddings = model.encode_text(
    "a drone shot of a mountain lake at sunrise, smooth camera movement"
)
null_embeddings = model.encode_text("")  # for classifier-free guidance

# Generate latent video via flow matching scheduler
scheduler = FlowMatchingScheduler(num_steps=40)
latents = scheduler.generate(
    model.transformer,
    text_embeddings=text_embeddings,
    null_embeddings=null_embeddings,
    shape=(1, 8, 97, 24, 42),   # [batch, channels, T_latent, H_latent, W_latent]
    guidance_scale=3.0,
    seed=42
)

# Decode latents to pixel frames
frames = model.vae.decode(latents)   # [B, T, H, W, C] in MLX channels-last
mx.synchronize()  # wait for GPU computation to complete

# Convert to numpy/PIL for saving
import numpy as np
frames_np = np.array(frames[0])  # [T, H, W, C], float32 in [0, 1]
```

**Memory profile:**

```
LTX-2-MLX MEMORY USAGE (approximate)

  Precision      Model size    M3 Max (128GB)    M1/M2 (16GB)
  ---------      ----------    --------------    ------------
  BF16 full      ~8 GB         YES, comfortable  Tight; may OOM on long videos
  4-bit quant    ~2.5 GB       YES, very fast    YES (leaves room for VAE + buffers)

  VAE overhead:  ~1-2 GB additional during encode/decode
  Frame buffers: ~0.5 GB per 97 frames at 768x512

  PRACTICAL:
    M3 Max 128GB: Run BF16 full with room to spare
    M2 Pro 16GB:  Use 4-bit; short video only (49-97 frames)
    M1 16GB:      4-bit; expect slow; 49 frames max
```

### ComfyUI-LTXVideo-mlx

ComfyUI-LTXVideo-mlx is the developer's ComfyUI integration for ltx-2-mlx. It wraps the MLX implementation as ComfyUI custom nodes, enabling LTX-Video inference from within the familiar node graph UI.

```
COMFYUI-LTXVIDEO-MLX ARCHITECTURE

  ComfyUI (PyTorch native)
      |
      | Node graph: load image -> encode prompt -> sample -> decode -> save
      |
      +--> [LTXVideoMLXSampler node] -- custom MLX node
      |         |
      |         | Boundary: PyTorch tensors <-> MLX arrays
      |         | Conversion: torch.Tensor -> numpy -> mx.array (input)
      |         |             mx.array -> numpy -> torch.Tensor (output)
      |         |
      |         +--> ltx-2-mlx model (runs on Apple GPU via MLX)
      |
      +--> [standard ComfyUI nodes: SaveImage, LoadImage, CLIPTextEncode, etc.]
           These run in PyTorch; no MLX involvement
```

**The tensor conversion pattern** (discussed in Image Generation, applied here):

```python
import torch
import mlx.core as mx
import numpy as np

class LTXVideoMLXSamplerNode:
    """ComfyUI node: run LTX-Video via MLX, return video frames as torch tensor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 768, "min": 256, "max": 1280, "step": 32}),
                "height": ("INT", {"default": 512, "min": 256, "max": 720, "step": 32}),
                "num_frames": ("INT", {"default": 97, "min": 9, "max": 257, "step": 8}),
                "num_steps": ("INT", {"default": 40, "min": 10, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0}),
            },
            "optional": {
                "reference_image": ("IMAGE",),   # ComfyUI IMAGE: [B, H, W, C] torch, float32 0-1
            }
        }

    RETURN_TYPES = ("IMAGE",)   # VIDEO frames as ComfyUI IMAGE (batch of frames)
    FUNCTION = "sample"
    CATEGORY = "MLX/Video"

    def sample(self, prompt, negative_prompt, width, height, num_frames,
                num_steps, guidance_scale, seed, reference_image=None):
        # 1. Convert reference image from PyTorch to MLX (if provided)
        if reference_image is not None:
            ref_np = reference_image.numpy()          # torch -> numpy
            ref_mlx = mx.array(ref_np)                # numpy -> MLX

        # 2. Run MLX inference
        frames_mlx = self.model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width, height=height,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            reference_image=ref_mlx if reference_image is not None else None,
        )
        # frames_mlx: [T, H, W, C] mx.array, float32

        # 3. Wait for GPU computation to complete
        mx.synchronize()

        # 4. Convert back to PyTorch for ComfyUI
        frames_np = np.array(frames_mlx)              # MLX -> numpy
        frames_torch = torch.from_numpy(frames_np)    # numpy -> torch
        # ComfyUI expects [B, H, W, C]; B here = number of frames
        return (frames_torch,)
```

**What ComfyUI-LTXVideo-mlx enables:** Running LTX-Video inference from within ComfyUI on Apple Silicon, with access to ComfyUI's image loading, prompt management, preview, and workflow save/load capabilities. Users can build full video generation workflows: load a reference image -> encode text -> generate video -> save frames -> interpolate.

### Other MLX Video Efforts

Beyond ltx-2-mlx, the MLX video generation landscape is thin:

```
MLX VIDEO GENERATION STATUS (Q2 2025)

  ltx-2-mlx (developer's project):
    Model: LTX-Video 2
    Status: Functional, maintained, actively developed
    Capabilities: text-to-video, image-to-video
    URL: github.com/[developer]/ltx-2-mlx

  ComfyUI-LTXVideo-mlx (developer's project):
    Model: LTX-Video 2 (via ltx-2-mlx)
    Status: Functional ComfyUI integration
    Capabilities: workflow-based video generation

  Matrix-Game-mlx (developer's project, in development):
    Model: Matrix Game model (interactive world generation)
    Status: Work in progress
    Notes: Extends from video generation toward interactive environments

  CogVideoX MLX: Not ported
  Wan MLX:       Not ported
  HunyuanVideo MLX: Not ported
  Mochi MLX:     Not ported
  AnimateDiff MLX: Not ported
```

### Memory Advantage for Video

The unified memory architecture is especially valuable for video generation because video models are much more memory-hungry than image models:

```
VIDEO MODEL MEMORY: CUDA vs APPLE SILICON

  LTX-Video 2 (BF16, ~8GB model):
    RTX 4090 (24GB):  YES, fits. Frame buffers for long video (97+ frames)
                      may push total usage to 16-20GB; tight but workable.
    RTX 3080 (10GB):  NO. Model alone is 8GB; VAE + frame buffers push
                      total to 12-14GB -> OOM on 97-frame generation.
    M3 Max (128GB):   8GB model + 2GB VAE + 2GB frame buffers = ~12GB.
                      128GB is 90% available for other tasks.

  Wan-14B (FP16, ~28GB model):
    RTX 4090 (24GB):  NO. Model is 28GB alone.
    A100 (40GB):      YES, barely. Only professional/cloud hardware.
    H100 (80GB):      YES, comfortable.
    M3 Max (128GB):   4-bit quantized: ~7-8GB. YES, comfortable.
                      Full BF16: 28GB. YES, 22% of memory.
    M2 Ultra (192GB): Full BF16: YES with room for long video buffers.

  HunyuanVideo (FP16, ~26GB model):
    RTX 4090 (24GB):  NO at full precision; needs heavy quantization.
    M3 Max (128GB):   YES at BF16 with headroom.

  KEY INSIGHT:
    For large video models (14B+ parameters), Apple Silicon unified memory
    is the only consumer hardware that can run them at full or near-full
    precision. CUDA requires professional datacenter GPUs (A100, H100).
    This gap is more pronounced for video than for image generation.
```

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **LTX-Video text-to-video** | 🟢 diffusers LTXVideoPipeline; official nodes; full feature set | 🟢 ltx-2-mlx; functional text-to-video; developer-maintained | Small -- ltx-2-mlx covers the core use case; the CUDA side has more convenience tooling |
| **LTX-Video image-to-video** | 🟢 LTXImageToVideoPipeline in diffusers; conditioning on reference frame | 🟡 ltx-2-mlx has image-to-video support; experimental | Small-to-medium -- core functionality present in MLX port |
| **CogVideoX** | 🟢 CogVideoXPipeline in diffusers; 2B and 5B variants | 🔴 Not ported to MLX | Large -- no MLX implementation |
| **Wan 1.3B** | 🟢 diffusers WanPipeline; fast model for iteration | 🔴 Not ported to MLX | Large -- Wan 1.3B would be the most practical port given small model size |
| **Wan 14B** | 🟢 diffusers WanPipeline; best quality open model | 🔴 Not ported to MLX | Large -- but 14B at BF16 runs on Apple Silicon; port would unlock best quality locally |
| **HunyuanVideo** | 🟢 diffusers HunyuanVideoPipeline; 13B, high resolution | 🔴 Not ported to MLX | Large -- 13B would require aggressive quantization even on M3 Max; port feasible |
| **Mochi** | 🟢 diffusers MochiPipeline; strong motion quality | 🔴 Not ported to MLX | Medium -- 10B model; motion quality distinctive; no MLX port |
| **AnimateDiff** | 🟢 AnimateDiffPipeline; temporal modules for SD1.5; huge LoRA ecosystem | 🔴 No MLX SD1.5 AnimateDiff port | Large -- access to 50,000+ SD1.5 LoRAs for video generation is a significant capability gap |
| **ComfyUI video nodes** | 🟢 Kijai nodes cover HunyuanVideo, Wan, LTX-Video, CogVideoX, Mochi | 🟡 ComfyUI-LTXVideo-mlx covers LTX-Video only | Large -- MLX ComfyUI video support is narrow; other models have no ComfyUI nodes |
| **Generation speed (short video)** | 🟢 RTX 4090: LTX-Video 97-frame 768x512 in ~15-25s | 🟡 M3 Max: ~3-8 minutes for same; ~10-20x slower | Large -- generation speed is the primary practical gap |
| **Generation speed (long video)** | 🟢 RTX 4090: scales linearly; 10-second clips ~60-90s | 🟡 M3 Max: ~10-20 minutes; still usable for quality work | Large -- speed gap is real; M4 Ultra will reduce it somewhat |
| **Maximum resolution** | 🟢 HunyuanVideo: native 720p; tiling for 1080p | 🟡 ltx-2-mlx: 768x512 native; higher res untested | Medium -- for most use cases 768x512 is sufficient; high-res video needs other models |
| **Maximum duration / frame count** | 🟢 Wan: 81+ frames; HunyuanVideo: 10+ seconds | 🟡 ltx-2-mlx: 97 frames (LTX-Video limit); ~5-6 seconds at 24fps | Medium -- ltx-2-mlx matches LTX-Video's native limit; other models offer longer clips |
| **Temporal consistency** | 🟢 Purpose-built models (Wan, HunyuanVideo) have strong consistency | 🟡 LTX-Video has good but not best-in-class consistency | Medium -- LTX-Video is fast but Wan 14B has better quality; no MLX port of Wan |
| **[Quantization](../glossary.md#quantization)** | 🟢 [GPTQ](../glossary.md#gptq), AWQ, FP8 in diffusers; community quantized models | 🟡 4-bit MLX quantization in ltx-2-mlx; format-specific | Medium -- MLX quantization works; fewer community quantized video models |
| **Video-to-video / stylization** | 🟢 CogVideoX-I2V, Wan-I2V, img2vid strength control | 🔴 Not implemented in MLX | Large -- video-to-video is a common workflow; no MLX equivalent |
| **ControlNet for video** | 🟢 Experimental in CUDA (AnimateDiff + ControlNet); depth/pose control | 🔴 Not implemented in MLX | Large -- structural control of video generation has no MLX equivalent |
| **Video upscaling / frame interpolation** | 🟢 RIFE, FILM for frame interpolation; Real-ESRGAN for upscaling | 🔴 No MLX equivalents | Medium -- these are post-processing tools; [CPU](../glossary.md#cpu) fallback is possible |
| **[Model](../glossary.md#model) memory (large models)** | 🟡 Large models (HunyuanVideo 13B, Wan 14B) require A100/H100 at full precision | 🟢 Apple Silicon unified memory: Wan 14B at BF16 runs on M3 Max 128GB | Inverted -- MLX has the memory advantage for very large models that CUDA struggles with on consumer hardware |
| **Streaming / real-time generation** | 🔴 Not yet possible; video gen is inherently batch-only, multi-step | 🔴 Same -- neither ecosystem has this yet | None -- neither ecosystem has this; future research direction |

---

## Performance Deep Dive

Video generation performance depends heavily on video resolution and number of frames. Here are realistic benchmarks:

```
VIDEO GENERATION PERFORMANCE (approximate, 2025)

  MODEL: LTX-Video 2, 768x512, 97 frames (~4 seconds at 24fps), 40 steps

  Hardware              Time          Notes
  --------              ----          -----
  NVIDIA RTX 4090       15-25s        BF16; fast. The reference benchmark.
  NVIDIA RTX 3090       25-40s        BF16; older but still quick
  NVIDIA RTX 4080       20-30s        BF16
  Apple M4 Max (128GB)  3-5 min       MLX 4-bit; significant gap
  Apple M3 Max (128GB)  4-7 min       MLX 4-bit; slower hardware
  Apple M3 Pro (36GB)   8-15 min      MLX 4-bit; memory bandwidth limited

  MODEL: Wan-14B, 480x832, 81 frames (~4 seconds), 50 steps

  Hardware              Time          Notes
  --------              ----          -----
  NVIDIA A100 (80GB)    2-5 min       BF16; professional cloud hardware
  NVIDIA RTX 4090       OOM at BF16   Needs aggressive quantization
  NVIDIA RTX 4090       5-10 min      4-bit or FP8 quantized
  Apple M3 Max (128GB)  Cannot run (no MLX port)
  Apple M2 Ultra (192GB) Cannot run (no MLX port)

  NOTE ON Wan + Apple Silicon:
    If a Wan MLX port existed, M3 Max 128GB could run Wan-14B at BF16
    in approximately 20-40 minutes per video -- slow but unique:
    no consumer CUDA GPU can run Wan-14B at full precision.

  PRACTICAL TAKEAWAY:
    For LTX-Video specifically: CUDA is 10-20x faster. For a developer
    iterating on prompts and settings, this means 90-second waits
    (MLX) vs 5-10 second waits (CUDA). That's real friction.

    However, for the very large models (Wan 14B, HunyuanVideo 13B),
    Apple Silicon is the only consumer hardware that can run them
    at all -- if MLX ports existed.
```

---

## Contribution Opportunities

**Wan-1.3B MLX port.** The single most impactful contribution to the MLX video ecosystem. Wan-1.3B is the current state-of-the-art open video model at small scale. At 1.3B parameters, it is practical to run even on lower-memory Apple Silicon (M1 Pro 16GB with 4-bit quantization). The architecture -- a DiT with a causal 3D VAE and umt5-xxl text conditioning -- is similar to LTX-Video and CogVideoX, so ltx-2-mlx provides a useful reference for the port. This would immediately give MLX users access to the best-quality open video model currently available.

**Wan-14B MLX port.** The follow-on from the 1.3B port. Wan-14B at BF16 is approximately 28GB -- comfortable on M3 Max 128GB and M2 Ultra 192GB. On these machines, it would produce better quality video than is achievable on any consumer CUDA [GPU](../glossary.md#gpu) (which cannot fit 28GB at full precision). This is a genuine competitive advantage for Apple Silicon that only a MLX port can unlock.

**CogVideoX MLX port.** CogVideoX-2B is a practical target: 2B parameters, full diffusers integration (useful as reference), and a distinctive 3D VAE design that is architecturally interesting to port. The 3D VAE's temporal compression approach differs from LTX-Video's, so porting it would add architectural diversity to the MLX video ecosystem.

**AnimateDiff for MLX.** AnimateDiff temporal modules added to an MLX SD1.5 implementation would unlock the vast SD1.5 fine-tune and LoRA ecosystem for video generation. The temporal attention module architecture is documented and well-understood. The primary dependency is a solid MLX SD1.5 implementation (mlx-examples provides a starting point) plus the AnimateDiff motion module weights (open-source).

**ControlNet for video (via LTX-Video or future models).** Structural conditioning of video generation -- given a depth sequence, edge map sequence, or pose skeleton sequence, generate a video that follows the structure. This extends from image ControlNet into the temporal domain. For the LTX-Video architecture, ControlNet conditioning would inject into the DiT via additional cross-attention or residual connections, mirroring how image ControlNet injects into a U-Net.

**RIFE / frame interpolation in MLX.** RIFE (Real-Time Intermediate Flow Estimation) is the standard frame interpolation model: given two frames, generate the intermediate frame(s). Porting RIFE to MLX would provide the post-processing step that makes generated video smoother and extends effective frame rate. The model is small (~20MB) and the architecture (optical flow + synthesis network) is well-documented.

**Video generation benchmarking suite.** There is no standardized benchmark for MLX video generation quality or performance. Creating a benchmark that measures: (1) generation time per video, (2) memory usage, (3) qualitative metrics (FVD or similar) -- and tracking these across model versions and Apple Silicon hardware -- would be immediately useful to the community.

**Streaming latent display.** A usability improvement that does not require a new model: display partially-decoded video frames during the denoising loop, so the user can see the generation converge instead of waiting for a blank screen. This requires decoding the latent at each step (or every N steps) and pushing frames to a preview window. The architecture supports this without model changes.

---

## Sources

- LTX-Video (Lightricks): [github.com/Lightricks/LTXVideo](https://github.com/Lightricks/LTXVideo). Architecture reference, model weights, and official CUDA implementation.
- ltx-2-mlx (developer's project): The primary MLX implementation of LTX-Video 2; source reference for all MLX code in this page.
- ComfyUI-LTXVideo-mlx (developer's project): ComfyUI integration for ltx-2-mlx; demonstrates the MLX node pattern for video generation.
- CogVideo / CogVideoX (THUDM): [github.com/THUDM/CogVideo](https://github.com/THUDM/CogVideo). 3D VAE architecture reference.
- Wan (Alibaba): Wan model weights and technical report. State-of-the-art open video quality as of 2025.
- HunyuanVideo (Tencent): [github.com/Tencent/HunyuanVideo](https://github.com/Tencent/HunyuanVideo). Dual-stream transformer architecture.
- Mochi (Genmo): [github.com/genmoai/mochi](https://github.com/genmoai/mochi). AsymDiT architecture; natural motion quality.
- Hugging Face diffusers (video pipelines): [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers). The canonical CUDA video generation library; LTXVideoPipeline, CogVideoXPipeline, WanPipeline, HunyuanVideoPipeline.
- AnimateDiff (Guo et al., 2023): "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning." The foundational paper for temporal modules in image diffusion.
- RIFE (Huang et al., 2022): "Real-Time Intermediate Flow Estimation for Video Frame Interpolation." Reference for frame interpolation post-processing.
- [Matrix](../glossary.md#matrix)-Game-mlx (developer's project, in development): Extends video generation toward interactive world modeling on Apple Silicon.

---

## See Also

- [Diffusion](../01-foundations/08-diffusion.md) -- the mathematical foundation; noise schedules and denoising underlie all video models; flow matching (used in LTX-Video and Wan) is a variant of the diffusion framework
- [Latent Space](../01-foundations/07-latent-space.md) -- all video models work in latent space; explains the VAE compression that makes per-frame generation tractable; temporal VAEs compress both space and time
- [Transformers](../01-foundations/05-transformers.md) -- LTX-Video, CogVideoX, Wan, HunyuanVideo are all Diffusion Transformers (DiT); the transformer architecture is what replaced U-Nets for video generation
- [Attention](../01-foundations/06-attention.md) -- temporal attention across frames is the core mechanism that makes video models coherent; explains why full 3D attention is expensive and how alternating spatial/temporal attention trades quality for speed
- [Quantization](../01-foundations/10-quantization.md) -- 4-bit quantization is what makes large video models (Wan-14B, HunyuanVideo) run on Apple Silicon; explains the quality/memory trade-off
- [Image Generation](04-image-generation.md) -- the adjacent domain; video generation extends image techniques into the temporal dimension; mflux and mlx-examples are the image generation references
- [Frameworks](02-frameworks.md) -- PyTorch/diffusers vs MLX; the pattern for porting diffusers video pipelines to MLX follows the same structure as image pipeline ports
