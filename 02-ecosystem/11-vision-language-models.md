# Vision-Language Models: CUDA vs MLX

> **Status:** 🟢 [CUDA](../glossary.md#cuda) (Mature -- Transformers supports every VLM the day it ships) | 🟡 MLX (Growing -- mlx-vlm covers the major open VLMs; new architectures lag by days to weeks)

## What This Domain Covers

Models that take **images (or video frames) and text together** and produce text: "describe this photo," "what does this chart show," "read the handwriting in this scan," "is there a defect in this part." These are **vision-language models (VLMs)**, sometimes called multimodal LLMs. They are the bridge between the [image](04-image-generation.md) world and the [LLM](03-llms.md) world, and they are one of the fastest-moving areas of open-model research.

A VLM is, structurally, an LLM with an extra front door. The text path is an ordinary decoder LLM. The image path runs the picture through a **vision encoder** (usually a CLIP- or SigLIP-style [transformer](../01-foundations/05-transformers.md)) to produce a grid of image embeddings, and a small **projector** (a few [linear layers](../glossary.md#layer)) maps those embeddings into the LLM's token [embedding](../glossary.md#embedding) space. The LLM then attends over image "tokens" and text tokens together as one sequence.

```
VISION-LANGUAGE MODEL ARCHITECTURE

  Image ──> Vision Encoder ──> grid of patch embeddings
            (CLIP / SigLIP)         │
                                    v
                               Projector (MLP)   <-- maps vision dim -> LLM dim
                                    │
                                    v
  Text ──> Tokenizer ──> text embeds ──┐
                                        ├──> [img tokens | text tokens] ──> LLM decoder ──> text out
                              (one combined sequence)

  Key idea: images become "tokens" the LLM attends over alongside words.
```

This page deserves its own entry rather than a paragraph inside [LLMs](03-llms.md) because the porting story is genuinely different: a VLM port is *two* model ports (the encoder and the decoder) plus the projector and a multimodal preprocessing pipeline, and the failure modes are distinct.

---

## The CUDA Ecosystem

### The Models

Open VLMs have converged on the encoder + projector + decoder recipe, differing mainly in which encoder, which LLM backbone, and how images are tiled:

```
OPEN VISION-LANGUAGE MODELS (CUDA-native via Transformers)

  Model                 Backbone        Notes
  -----                 --------        -----
  LLaVA 1.5 / 1.6       Vicuna/Mistral  The reference open VLM; 1.6 adds tiling
  Qwen2-VL / 2.5-VL     Qwen2           Strong OCR, video, dynamic resolution
  Llama 3.2 Vision      Llama 3         11B and 90B; cross-attention design
  InternVL 2 / 2.5      InternLM/Qwen   High benchmark scores, many sizes
  Phi-3.5 Vision        Phi-3           Small, strong for its size
  PaliGemma 2           Gemma 2         SigLIP encoder; clean architecture
  Pixtral               Mistral Nemo    Native multi-image
  MiniCPM-V             Qwen/Llama      Efficient, mobile-oriented
  Molmo                 OLMo/Qwen       Open data + weights; pointing
```

### The Tooling

VLMs live inside the standard Hugging Face stack. **Transformers** provides `AutoModelForVision2Seq` / `AutoProcessor`, and a new VLM is usually loadable in Transformers on release day. **vLLM** and **SGLang** serve VLMs in production with the same paged-attention batching they use for text LLMs. **LLaVA's** original training repo and **lmms-eval** cover fine-tuning and benchmarking. The processor objects handle the fiddly multimodal preprocessing -- image resizing, patch tiling, inserting `<image>` placeholder tokens at the right positions -- which is where most of the per-model complexity hides.

---

## The MLX Ecosystem

### mlx-vlm

**mlx-vlm** (community, Prince Canuma) is the primary MLX library for vision-language models. It mirrors the mlx-lm API surface -- `load` / `generate` -- and bundles the per-architecture vision encoders, projectors, and processors:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

image = ["photo.jpg"]
prompt = apply_chat_template(processor, config, "What's in this image?", num_images=len(image))
# Note the argument order: prompt first, then the image(s).
output = generate(model, processor, prompt, image, max_tokens=300, verbose=True)
```

```
MLX-VLM SUPPORTED ARCHITECTURES (selection, mid-2025 -- track the repo; the list grows)

  LLaVA 1.5 / 1.6
  Qwen2-VL / Qwen2.5-VL
  Llama 3.2 Vision (11B, 90B)
  InternVL 2
  Phi-3 / Phi-3.5 Vision
  Idefics 2 / 3
  PaliGemma / PaliGemma 2
  Pixtral
  MiniCPM-V

  Capabilities: single + multi-image prompts, 4-bit/8-bit quantization,
  video (frame-sampled) on architectures that support it, mlx-vlm.server
  (OpenAI-compatible), and LoRA fine-tuning on supported models.
```

Converted weights live on the `mlx-community` Hugging Face org, typically appearing within days for popular releases. Conversion uses `mlx_vlm.convert` (analogous to `mlx_lm.convert`).

### The Memory Angle

VLMs benefit from [unified memory](../glossary.md#unified-memory) the same way LLMs do, and slightly more: the vision encoder, projector, and decoder all sit in one pool, and high-resolution image tiling can balloon the number of image tokens (Qwen2-VL's dynamic resolution can emit thousands of tokens for one large image). A 90B vision model that needs ~45 GB at 4-bit runs on a 64 GB+ Mac and on no consumer NVIDIA card.

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **New-model availability** | 🟢 Transformers, often day-one | 🟡 mlx-vlm + community port; days to weeks | Medium -- mainstream VLMs land fast; research models lag |
| **Single-image inference** | 🟢 Full support | 🟢 Works well across supported models | None for covered architectures |
| **Multi-image / interleaved** | 🟢 Native (Pixtral, Qwen2-VL) | 🟡 Supported on several models; varies by architecture | Small-medium -- check per model |
| **Video understanding** | 🟢 Frame sampling + native video models | 🟡 Frame-sampled on supporting architectures | Medium -- works but less polished |
| **High-throughput serving** | 🟢 vLLM/SGLang paged-attention batching | 🟡 mlx-vlm.server is single-user (no paged attention) | Large for multi-tenant; irrelevant for local use |
| **Fine-tuning** | 🟢 Full + LoRA, mature pipelines (lmms, LLaVA repo) | 🟡 LoRA on supported models; fewer recipes | Medium |
| **Preprocessing fidelity** | 🟢 Reference processors | 🟡 Reimplemented per model in mlx-vlm; occasional drift on new models | Small -- but the #1 source of wrong VLM output (see below) |
| **OCR / document tasks** | 🟢 Qwen2.5-VL, InternVL strong | 🟢 Same models, same quality when ported | None once ported |

---

## Why VLM Ports Break: The Preprocessing Trap

For a developer porting a VLM (or debugging mlx-vlm output), the single most important thing to know: **the model weights are rarely the problem -- the image preprocessing is.** A VLM produces fluent, confident, *wrong* descriptions when the image pipeline diverges even slightly from the reference, because the LLM half still generates perfectly grammatical text over garbage image tokens.

```
VLM PREPROCESSING FAILURE MODES (the usual suspects)

  Symptom                          Likely cause
  -------                          ------------
  Plausible but wrong description  Image normalization mean/std mismatch
  Ignores part of the image        Wrong tiling / patch count vs reference
  Off-by-one / scrambled content   <image> placeholder tokens at wrong positions
  Cyan / inverted colors fed in    RGB vs BGR, or [0,1] vs [0,255] scaling
  Works on small img, fails large  Dynamic-resolution tiling not replicated
  Right content, wrong layout      NCHW vs NHWC on the patch grid (see Latent Space)
```

The discipline is the same as any MLX port: verify the **vision encoder output** numerically against the reference (PyTorch) for a fixed image *before* trusting the generated text. Text-level "it looks reasonable" is not a parity check. See the [Porting Guide](../03-contributing/02-porting-guide.md) for the numerical-agreement methodology, and [Latent Space](../01-foundations/07-latent-space.md) for the channel-ordering issues that bite the encoder.

---

## What the Gap Actually Costs You

If you are running **a supported VLM locally** for single-image or document tasks: essentially no gap. mlx-vlm gives you Qwen2.5-VL or InternVL on a Mac at full quality, fully offline -- strong for private document/OCR pipelines.

If you need **the newest VLM the week it drops**: you may wait for the port, or do the port yourself.

If you are serving **a multi-tenant VLM API**: same story as text LLMs -- mlx-vlm.server is single-user; production batching is the open gap.

---

## Contributing Here

1. **Port new VLM architectures to mlx-vlm.** The highest-demand, highest-visibility VLM contribution. Budget for two encoders'-worth of care: the vision tower and the processor are where the bugs live, not the decoder.
2. **Preprocessing parity tests.** A harness that checks mlx-vlm's image pipeline against the reference processor for each model would catch the silent-wrong-output class before users hit it.
3. **Paged-attention serving for mlx-vlm.server** (shares the [LLMs](03-llms.md) serving gap).
4. **Video pipeline improvements** -- frame sampling and temporal handling are less mature than the image path.

See [Open Opportunities](../03-contributing/03-open-opportunities.md) for ranking against other gaps.

---

## Sources

- mlx-vlm: [github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
- Transformers (VLM support): [github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- LLaVA: [github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- Qwen2-VL: [github.com/QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
- InternVL: [github.com/OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)
- lmms-eval (VLM benchmarking): [github.com/EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- MLX Community on Hugging Face: [huggingface.co/mlx-community](https://huggingface.co/mlx-community)
- CLIP (Radford et al., 2021): the vision-encoder lineage underlying most VLMs
- SigLIP (Zhai et al., 2023): the encoder used by PaliGemma and others

---

## See Also

- [LLMs](03-llms.md) -- the decoder half of every VLM; mlx-lm is the substrate mlx-vlm builds on
- [Image Generation](04-image-generation.md) -- the inverse task (text -> image); shares CLIP/SigLIP encoder heritage
- [Embeddings & RAG](10-embeddings-rag.md) -- CLIP-style multimodal embeddings for image search
- [Latent Space](../01-foundations/07-latent-space.md) -- channel ordering and encoder representations; why preprocessing breaks ports
- [Attention](../01-foundations/06-attention.md) -- how image and text tokens attend to each other in one sequence
- [Porting Guide](../03-contributing/02-porting-guide.md) -- the numerical-parity discipline a VLM port demands
