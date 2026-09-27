# VAEs in Depth

> **One-liner:** The VAE is the part of an image or video generator that converts pixels to latents and back; the network itself is a plain stack of convolutions, but the **numbers around it** -- compression factors, the scale/shift or per-channel statistics applied to the latents, mean vs sample, fp16 safety, and tiling -- are different for every model family and are where VAE ports go wrong.

## The Intuition

[Latent Space](07-latent-space.md) explained why diffusion runs on a compressed latent instead of pixels. This page is about the component that does the compressing, the **variational autoencoder (VAE)**, from the angle of someone porting one.

```
            ENCODER                                    DECODER
  pixels  ---------->  mean, logvar  --> latent  ---------->  pixels
  (H, W, 3)           (H/8, W/8, 2C)    (H/8, W/8, C)         (H, W, 3)
                                          |
                          normalize  -----+-----  denormalize
                    (scale/shift or per-channel stats)
                                          |
                                  what the diffusion
                                   model actually sees
```

The encoder and decoder are convolutional networks (ResNet blocks, [GroupNorm](14-normalization.md), up/downsampling, sometimes a mid-block attention). Porting those is mostly the [convolution](19-convolutions.md) layout work. The rest of this page covers what surrounds them.

---

## How It Actually Works

### Compression: how much smaller the latent is

Each VAE has a spatial (and, for video, temporal) downsampling factor and a latent channel count. Values from the checkpoints' configs:

```
  model                 spatial   temporal   latent channels   pixels per latent value
  SD 1.5 / SDXL           8x         -             4                    48
  SD3 / Flux.1            8x         -            16                    12
  HunyuanVideo            8x         4x           16                    48
  Wan 2.1                 8x         4x           16                    48
  Wan 2.2 (TI2V-5B)      16x         4x           48                    64
  LTX-Video              32x         8x          128                   192

  pixels per latent value = (spatial² x temporal x 3 RGB channels) / latent channels
```

Two trends: more latent channels (so the latent holds more detail per position) and more aggressive spatial compression (so the diffusion transformer sees fewer tokens). LTX-Video's 32×32×8 compression is what makes its transformer fast.

Video VAEs compress time **causally** (see [Convolutions](19-convolutions.md#causal-3d-convolution-video-that-doesnt-look-ahead)): the first frame is encoded alone, then each group of 4 (or 8) frames. That's why frame counts must be `1 + 4k` for Wan and HunyuanVideo, and `1 + 8k` for LTX-Video.

### Mean, log-variance, and "sample vs mode"

A VAE encoder doesn't output a latent; it outputs a **distribution**: a mean and a log-variance per latent value (so `2 × C` channels, split in half). To get a latent you either take the mean (`.mode()`) or draw a sample (`mean + exp(logvar / 2) × noise`, i.e. `.sample()`).

Only the *encoder* is affected -- text-to-image generation never encodes, it starts from noise. But in image-to-image, inpainting, or image-conditioned video, the reference pipeline's choice matters. Diffusers' `retrieve_latents` defaults to `sample_mode="sample"` with the pipeline's generator. A port that takes the mean is deterministic and close, but won't match the reference exactly, and a port that samples with a different RNG won't either. For parity tests, force both sides to use the mean.

### Normalizing the latents: three conventions

Raw VAE latents don't have unit variance, but diffusion models are trained on latents that roughly do. Every pipeline normalizes latents after encoding and reverses it before decoding -- with a different formula per family:

```
  convention                  encode -> diffusion space          diffusion -> decode
  --------------------------  ---------------------------------  ------------------------------
  scale only                  z * scale                          z / scale
    SD 1.5 (0.18215), SDXL (0.13025), HunyuanVideo (0.476986)

  scale + shift               (z - shift) * scale                z / scale + shift
    SD3 (1.5305, 0.0609), Flux.1 (0.3611, 0.1159)

  per-channel mean / std      (z - mean[c]) / std[c]             z * std[c] + mean[c]
    Wan 2.1 (16 values each), Wan 2.2 (48), LTX-Video (128,       (LTX also multiplies by
    stored as buffers in the VAE weights)                          scaling_factor = 1.0)
```

Two traps hide in the reference code:

- **The values aren't always in `config.json`.** SD 1.5's VAE config has no `scaling_factor` (it's the class default, 0.18215). LTX-Video's per-channel statistics are tensors in the weights, not config entries. Wan's are config lists.
- **Variable names lie.** In diffusers' Wan pipeline, the variable called `latents_std` holds `1 / std`, and the code computes `latents / latents_std + latents_mean`. Copy the *math*, not the variable names.

### fp16 and the VAE decoder

The decoder's last blocks produce large activations. SDXL's original VAE overflows float16 and produces NaNs (black images). Its config says so (`force_upcast: true`), and diffusers then runs the VAE in float32 even when the rest of the pipeline is float16. A retrained drop-in, `madebyollin/sdxl-vae-fp16-fix`, avoids it. In MLX, run the VAE in float32 or bfloat16 when the reference upcasts (see [Numerical Stability](12-numerical-stability.md)).

### Memory, tiling, and evaluation points

The decoder is where image and video pipelines hit their memory peak: its last blocks run 128-channel feature maps at full output resolution. Measured with an SD-VAE-shaped decoder (random weights, float16, MLX 0.32.2, M2 Pro 32 GB):

```
  output         full decode      eval after each block      tiled (64-latent tiles, 16 overlap)
  512 x 512        2.52 GB               -                          -
  1024 x 1024      7.36 GB             5.73 GB                    2.55 GB   (1.7 s -> 2.4 s)
  2048 x 2048     28.58 GB            22.59 GB                      -
```

- **Peak memory grows with the pixel count** (4x per doubling) -- a 2048² decode nearly fills a 32 GB Mac.
- **Evaluating after each block** (see [Lazy Evaluation](15-lazy-evaluation.md)) trims ~20% by not letting the lazy graph keep intermediates alive.
- **Tiling** decodes overlapping latent tiles separately and blends them. The peak stops depending on the output size, at the cost of time and **exactness**: in the same test, the tiled output differed from the full decode by 0.022 on average and up to 0.50 near seams (output std 0.32). The decoder's receptive field is wider than the overlap, so each tile sees less context. Video VAEs tile in time too, with the same trade-off.

For a parity test, compare **untiled** decodes. Enable tiling only for real runs, with the same tile size and overlap as the reference if its outputs must match.

---

## Why It Matters for MLX

### Port failures

```
VAE PORT FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  Washed-out, grey, or oversaturated         latent normalization missing, applied
    images                                     twice, or wrong convention (scale-only
                                               vs scale+shift vs per-channel)
  Colors slightly off, structure fine        per-channel mean/std in the wrong
                                               order or inverted (std vs 1/std)
  Black image / NaNs from the decoder        fp16 overflow: reference upcasts the
                                               VAE (force_upcast)
  Visible grid of seams                      tiled decode with too little overlap,
                                               or no blending
  img2img differs from reference only        encoder: sample vs mean, or a
    slightly                                   different RNG for the sample
  Video: wrong number of output frames       frame count not 1 + 4k (or 1 + 8k),
                                               or causal padding wrong
  Output in the wrong range                  decoder outputs [-1, 1]; map with
                                               (x + 1) / 2, clip, then x 255
  Blotchy colors everywhere                  GroupNorm over NHWC with the channel
                                               grouping of NCHW (see Normalization)
```

### Porting checklist

1. **Find all the constants.** Scaling factor, shift factor, per-channel mean/std -- from the config, the class defaults, and the weights file. Print them from the reference pipeline to be sure.
2. **Test the VAE alone.** Encode and decode the same image in both frameworks with the mean (no sampling) and no tiling; compare latents and pixels before connecting it to the diffusion model.
3. **Match the reference's precision** for the decoder, and place `mx.eval` after each decoder block if memory is tight.
4. **Add tiling last**, as a memory option, and check the seams visually.

---

## Key Terms

| Term | Definition |
|------|------------|
| **VAE (variational autoencoder)** | Encoder-decoder pair that maps pixels to a compressed latent distribution and back; the pixel/latent bridge of latent diffusion. |
| **Compression factor** | How much smaller the latent is: spatial (8x, 16x, 32x) and, for video, temporal (4x, 8x). |
| **Latent channels** | Number of values per latent position (4, 16, 48, 128); more channels keep more detail. |
| **Mean / log-variance** | The encoder's output: a Gaussian per latent value, from which the latent is taken (mean) or sampled. |
| **Scaling factor / shift factor** | Constants that normalize latents for the diffusion model: `(z - shift) * scale`, reversed before decoding. |
| **Per-channel latent statistics** | A mean and std per latent channel (Wan, LTX-Video) used instead of a single scale. |
| **force_upcast** | Config flag meaning the VAE must run in float32 when the pipeline is float16 (fp16 overflow). |
| **Tiled decoding** | Decoding overlapping latent tiles separately and blending them, to cap memory; not bit-exact. |

---

## Sources

- Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes." [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- HaCohen, Y., et al. (2024). "LTX-Video: Realtime Video Latent Diffusion." [arxiv.org/abs/2501.00103](https://arxiv.org/abs/2501.00103)
- Wan Team (2025). "Wan: Open and Advanced Large-Scale Video Generative Models." [arxiv.org/abs/2503.20314](https://arxiv.org/abs/2503.20314)
- Kong, W., et al. (2024). "HunyuanVideo: A Systematic Framework For Large Video Generative Models." [arxiv.org/abs/2412.03603](https://arxiv.org/abs/2412.03603)
- VAE configs on the Hugging Face Hub (`vae/config.json` of SDXL, LTX-Video, Wan 2.1 / 2.2, HunyuanVideo) and ComfyUI's `comfy/latent_formats.py` (SD 1.5, SD3 and Flux constants): [github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py)
- Diffusers source: `pipelines/ltx/pipeline_ltx.py` (`_normalize_latents`), `pipelines/wan/pipeline_wan.py` (`1 / std`), `pipelines/flux/pipeline_flux.py`, `models/autoencoders/autoencoder_kl.py` (tiling), `retrieve_latents`: [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
- SDXL fp16-fix VAE: [huggingface.co/madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- Memory and tiling measurements: SD-VAE-shaped decoder with random weights, MLX 0.32.2, M2 Pro 32 GB.

---

## See Also

- [Latent Space](07-latent-space.md) -- prerequisite; why latents, and the VAE's place in the pipeline
- [Convolutions & Patchification](19-convolutions.md) -- the layers the VAE is made of, and causal 3D convolutions
- [Normalization](14-normalization.md) -- GroupNorm, used throughout VAE blocks
- [Numerical Stability](12-numerical-stability.md) -- fp16 overflow in the decoder
- [Lazy Evaluation](15-lazy-evaluation.md) -- evaluation points to lower the decoder's memory peak
- [Diffusion](08-diffusion.md) -- what runs on the normalized latents
