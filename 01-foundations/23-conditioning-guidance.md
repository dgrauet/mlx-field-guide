# Conditioning & Guidance

> **One-liner:** *Conditioning* is how extra inputs -- a text prompt, a first frame, keyframes, a mask, camera or keyboard actions -- get into a diffusion model; *guidance* is how the sampler amplifies their effect at each step by combining several predictions. Both live mostly outside the network, in pipeline code with small conventions (scale offsets, masks, per-token timesteps) that a port must copy exactly.

## The Intuition

A diffusion model predicts, at each step, how to move a noisy latent toward a clean one (see [Diffusion](08-diffusion.md)). **Conditioning** changes *what* it moves toward: the same noise becomes a cat or a car depending on the prompt, and a video can be forced to start from a given image or pass through given keyframes.

The model obeys its conditions only moderately on its own. **Guidance** runs it more than once per step -- with the condition, without it, or with a deliberately degraded version of itself -- and extrapolates *away from* the worse predictions:

```
  guided = cond + (scale - 1) * (cond - uncond)

      uncond ------- cond ---------------> guided
              the model's own          pushed further in the
              conditional shift        same direction (scale > 1)
```

The cost is one extra forward pass per guidance term per step, which is why guidance often dominates generation time.

---

## How It Actually Works

### Ways to inject a condition

```
  mechanism                     how                                    typical use
  ---------                     ---                                    -----------
  cross-attention               latent tokens attend to the           text prompts,
                                  condition's tokens                    image embeddings,
                                                                        actions (Matrix-Game)
  modulation (adaLN)            condition -> per-channel scale/shift   timestep, pooled
                                  applied after each norm                text, camera params
  channel concatenation         extra input channels next to the       inpainting, image-
                                  noisy latent                          to-video (Wan/CogVideoX
                                                                        Fun InP)
  latent replacement            some latent frames/tokens are          first-frame and
    + denoise mask                clean and never re-noised             keyframe conditioning
                                                                        (LTX)
  appended tokens               clean condition tokens added to the    keyframes at arbitrary
    + attention mask              sequence with their own positions     times, reference video
                                                                        (LTX)
```

**Channel concatenation** changes the model's input layer, which is visible in the config. The inpainting variants take the noisy latent plus a masked-video latent plus the mask: `in_channels` is **33** for `CogVideoX-Fun-V1.5-5b-InP` and **36** for `Wan2.1-Fun-1.3B-InP`, both with 16 output channels. Wan packs the mask across the 4 frames of each temporal latent, hence 4 mask channels instead of 1. Feeding such a model a 16-channel input fails at the first layer; feeding the right channels in the wrong order runs and produces nonsense.

**Latent replacement with a denoise mask** is how LTX conditions on a first frame. The conditioning frame's latent tokens are set to the clean encoded image, and a per-token **denoise mask** (1 = generate, 0 = keep, `1 - strength` in between) controls three things at every step (from the LTX-2 MLX port, `conditioning/types/latent_cond.py` and `utils/samplers.py`):

```
  1. noising      latent = noise * (mask * sigma) + clean * (1 - mask * sigma)
  2. timestep     per-token timestep = mask * sigma   -> clean tokens are told t = 0
  3. blending     x0 = x0_pred * mask + clean * (1 - mask)   after each prediction
```

All three are needed. Skip the per-token timestep and the model treats the clean frame as noisy; skip the blending and the conditioned frame drifts over the steps.

**Appended tokens** (LTX keyframes) leave the video latent untouched and append each keyframe's clean tokens to the sequence, with positions at the keyframe's time and an attention mask that keeps keyframes from attending to each other. The positions are part of the trick: the LTX port documents that a non-first keyframe's temporal position is computed differently from a regular frame's, because that is what the model was trained on.

### Guidance terms

LTX-2's guider (`components/guiders.py` in the MLX port, ported from `ltx-core`) combines up to four predictions per step:

```
  pred = cond
       + (cfg_scale - 1)      * (cond - uncond_text)       # CFG: prompt vs no prompt
       + stg_scale            * (cond - uncond_perturbed)  # STG: model vs model with
                                                           #      some blocks skipped
       + (modality_scale - 1) * (cond - uncond_modality)   # audio<->video coupling

  optional rescale:
       factor = std(cond) / std(pred)
       pred  *= rescale_scale * factor + (1 - rescale_scale)
```

- **CFG (classifier-free guidance)**: the unconditional prediction uses an empty or negative prompt. `cfg_scale = 1` disables it.
- **STG (spatiotemporal skip guidance)**: the "bad" prediction comes from the same model with selected transformer blocks skipped (`stg_blocks`), no extra model needed. Note the **different offset: `stg_scale = 0` disables it**, not 1.
- **Modality guidance**: for joint audio-video models, the prediction with the audio-to-video and video-to-audio cross-attention skipped in every block. `modality_scale = 1` disables it.
- **Guidance rescale**: large scales inflate the prediction's magnitude, which shows up as oversaturated, burnt images. The rescale pulls its standard deviation back toward the conditional prediction's.

Measured on synthetic predictions (MLX 0.32.2), the ratio `std(pred) / std(cond)`:

```
  cfg_scale   no rescale   rescale 0.7   rescale 1.0
     1.0         1.00         1.00          1.00
     3.0         1.41         1.12          1.00
     7.0         2.81         1.54          1.00
```

Other variants you'll meet: **APG** (adaptive projected guidance) removes the part of the guidance update parallel to the conditional prediction, which is what oversaturates. **CFG-Zero\*** adjusts CFG for flow-matching models. Both need the `projection_coef`-style dot products that the LTX guider also defines. **Guidance-distilled** models (LTX-2's distilled checkpoints, FLUX.1-dev) have guidance baked into the weights: they take a guidance value as an input, or none, and run **one** pass per step. Adding CFG on top double-applies it.

### Schedules: guidance that changes per step

Guidance doesn't have to be constant. LTX-2's `MultiModalGuiderFactory` selects parameters by **sigma bin**: different CFG, STG and rescale values for high-noise and low-noise steps. `skip_step` skips the extra passes on some steps entirely to save time. A port must reproduce the bin boundaries and the skip rule, not just the default scales.

---

## Why It Matters for MLX

### Batching the guidance passes doesn't pay on a Mac

Reference code often runs CFG as one batch of 2 (conditional and unconditional stacked) because it's faster on a datacenter GPU. On Apple Silicon at video sequence lengths, the forward pass is already compute-bound, so batching gains nothing. Measured with 4 DiT-like blocks (width 2048, bf16, MLX 0.32.2, M2 Pro):

```
  tokens   two sequential passes   one batch of 2
   1024          234 ms                249 ms
   4096        1,230 ms              1,265 ms
```

A batch of 2 also doubles activation memory. Running the passes **sequentially** is the better default for an MLX port: same speed, half the peak activations, and each extra guidance term (STG, modality) just adds one more pass. Make sure the per-pass inputs are identical to the reference's (same noisy latent, same timestep) and only the condition differs.

### Port failures

```
CONDITIONING & GUIDANCE PORT FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  Prompt ignored / washed out                CFG not applied, or cond and uncond
                                               swapped (guidance pushes toward the
                                               empty prompt)
  Oversaturated, burnt colors                guidance applied twice (distilled
                                               model + CFG), or rescale missing
  STG seems too strong or too weak           stg_scale offset: LTX uses
                                               stg_scale * (cond - perturbed), not
                                               (stg_scale - 1); or wrong stg_blocks
  First frame drifts away from the input     blending with the clean latent
    image                                      skipped after each step
  First frame blurry / re-noised             per-token timestep not zeroed for
                                               conditioned tokens (mask * sigma)
  Keyframe lands at the wrong time           keyframe positions computed like
                                               regular frames, or pixel vs latent
                                               frame index confused
  Inpainting model outputs noise             input channels in the wrong order
                                               (noisy / mask / masked video)
  Output matches reference except at some    per-sigma guidance schedule or
    steps                                      skip_step not reproduced
```

### Porting checklist

1. **Write down each guidance term with its offset** (`scale - 1` or `scale`) and when it is active, straight from the reference guider.
2. **Check whether the checkpoint is guidance-distilled** before adding CFG.
3. **Reproduce the three denoise-mask effects** (noising, per-token timestep, blending) for any latent-replacement conditioning.
4. **Test one sampler step in isolation**: same inputs on both sides, compare the cond, uncond and perturbed predictions separately, then the combined result.

---

## Key Terms

| Term | Definition |
|------|------------|
| **Conditioning** | Any extra input that steers generation: text, images, keyframes, masks, actions, camera parameters. |
| **CFG (classifier-free guidance)** | Combining conditional and unconditional predictions to amplify the condition: `cond + (s - 1)(cond - uncond)`. |
| **Negative prompt** | The prompt used for the "unconditional" pass; guidance pushes away from it. |
| **STG (spatiotemporal skip guidance)** | Guidance away from the same model's prediction with some blocks skipped. |
| **Guidance rescale** | Rescaling the guided prediction toward the conditional prediction's standard deviation to prevent oversaturation. |
| **APG (adaptive projected guidance)** | Guidance variant that drops the update component parallel to the conditional prediction. |
| **Guidance distillation** | Training a model to reproduce guided outputs in one pass; such models must not get CFG on top. |
| **Denoise mask** | Per-token mask saying which latent tokens are generated (1) or kept from a clean condition (0). |
| **Per-token timestep** | A separate timestep per token (`mask * sigma`), so clean conditioning tokens are processed as noise-free. |
| **Channel concatenation** | Conditioning by stacking extra channels (mask, masked video) next to the noisy latent at the model input. |
| **Keyframe conditioning** | Forcing a generated video to pass through given frames at given times. |

---

## Sources

- Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
- Lin, S., et al. (2023). "Common Diffusion Noise Schedules and Sample Steps are Flawed" (guidance rescale). [arxiv.org/abs/2305.08891](https://arxiv.org/abs/2305.08891)
- Hyung, J., et al. (2024). "Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling" (STG). [arxiv.org/abs/2411.18664](https://arxiv.org/abs/2411.18664)
- Sadat, S., et al. (2024). "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models" (APG). [arxiv.org/abs/2410.02416](https://arxiv.org/abs/2410.02416)
- Fan, W., et al. (2025). "CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models." [arxiv.org/abs/2503.18886](https://arxiv.org/abs/2503.18886)
- Yu, J., et al. (2025). "GameFactory: Creating New Games with Generative Interactive Videos" (the action module Matrix-Game builds on). [arxiv.org/abs/2501.08325](https://arxiv.org/abs/2501.08325)
- LTX-2 MLX port: `ltx_core_mlx/components/guiders.py` (multi-modal guider, rescale, sigma bins, skip_step), `conditioning/types/latent_cond.py` and `keyframe_cond.py` (denoise mask, appended keyframes), `ltx_pipelines_mlx/utils/samplers.py` (per-token timesteps): [github.com/dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)
- Inpainting channel counts: `transformer/config.json` of `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP` and `config.json` of `alibaba-pai/Wan2.1-Fun-1.3B-InP` on the Hugging Face Hub.
- Measurements on this page: MLX 0.32.2 on an M2 Pro, synthetic predictions and random-weight DiT-like blocks.

---

## See Also

- [Diffusion](08-diffusion.md) -- prerequisite; the sampling loop, sigmas and basic CFG
- [Attention](06-attention.md) -- cross-attention, the main injection path for text and actions
- [Positional Encoding](11-positional-encoding.md) -- why appended keyframe tokens need the right positions
- [VAEs in Depth](21-vae.md) -- encoding the conditioning image or video into latents
- [Testing & Validation](../03-contributing/05-testing-validation.md) -- comparing guided outputs with PSNR from identical initial noise
