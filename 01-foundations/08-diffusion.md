# Diffusion

> **One-liner:** [Diffusion](../glossary.md#diffusion) models generate images (or video) by learning to gradually remove noise from random static, step by step, until a clean result emerges.

## The Intuition

You know from the [Latent Space](../glossary.md#latent-space) page that the diffusion process runs inside a compact latent -- not on raw pixels. This page explains what that process actually is: how a model learns to generate images and video from nothing but noise.

### The static-to-signal analogy

Picture a clear photograph. Now imagine someone slowly adding television static to it -- a little at first, then more and more, until the original image is completely buried under noise. After enough steps, you have pure random static. No trace of the original remains.

Now run that backward. Start from pure static. Subtract a little noise. Subtract a little more. Keep going, step by step, and the image gradually emerges from the chaos.

That is diffusion. A diffusion model is trained to do that reversal. Given a noisy image at a specific point in the destruction process, predict what noise was added and remove it. Do this for enough steps, and you get a clean image.

```
THE CORE IDEA: DESTROY THEN REVERSE

  Clean image                                           Pure noise
  [photo of cat]  -->  -->  -->  -->  -->  -->  -->  [static]
       t=0         t=100    t=200    t=500    t=800    t=1000

  Forward process (training): gradually add Gaussian noise
  Nothing is learned here -- this is a fixed math formula

  ---- and now reverse it ----

  Pure noise                                           Clean image
  [static]        <--  <--  <--  <--  <--  <--  <--  [photo of cat]
     t=T         t=900    t=700    t=400    t=100      t=0

  Reverse process (generation): model removes noise step by step
  The model is trained to do this
```

### What the model actually learns

The model does not learn "what a cat looks like." It learns one simpler skill: *given a noisy image and the step number, predict what the noise is.* That is it.

At training time, you have real images. You add a known amount of noise (you chose how much -- you know the answer). You ask the model to predict the noise. The model gets it slightly wrong. You update the weights. Repeat millions of times.

At generation time, you feed the model random noise. It predicts (imperfectly) what the noise is. You subtract that prediction. The result is slightly less noisy. Feed that in. Predict again. Subtract again. After 20-50 steps, something coherent has emerged.

### The refinement loop

Each denoising step is not a dramatic leap. The model makes small, confident moves:

```
DENOISING STEP BY STEP

  Step 1  [very noisy]    Model: "I can barely tell, but I think there's
                                  something warm and orange in the center."
  Step 2  [slightly less  Model: "Yes, definitely an orange shape. Probably
           noisy]                 a face? Whiskers maybe?"
  Step 3  ...             Model: "Cat, definitely. Ears here, eyes there."
  Step 10 ...             Model: "Tabby cat, indoor, natural light, sitting."
  Step 50 [clean output]  Model: "Sharp fur detail, catchlights in eyes."

  At each step: less noise --> more confident about content --> tighter edits
```

This incremental refinement is what gives diffusion models their distinctive quality. They do not commit to a single interpretation at step 1 -- they explore, then narrow down.


## How It Actually Works

### The forward process: adding noise on a schedule

[Training](../glossary.md#training) begins with the forward process, which is pure math -- no learned parameters. You take a clean latent `x_0` and progressively corrupt it by adding Gaussian noise according to a noise schedule:

```
FORWARD PROCESS

  Clean latent:          x_0
                          |
  Add Gaussian noise:    x_1 = sqrt(1 - beta_1) * x_0 + sqrt(beta_1) * noise_1
                          |
  Add more noise:        x_2 = sqrt(1 - beta_2) * x_1 + sqrt(beta_2) * noise_2
                          |
                         ...
                          |
  After T steps:         x_T ~ N(0, I)   (approximately pure noise)

  beta values: small at first (gentle corruption), grow toward end
               typical range: 0.00085 to 0.012 over 1000 steps
```

A useful shortcut: you can compute `x_t` (the noisy latent at any timestep) directly from `x_0` without stepping through all intermediate states:

```
CLOSED-FORM NOISE ADDITION (used during training)

  x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * noise

  where:
    alpha_bar_t = product of (1 - beta_i) for i = 1 to t
    noise       = random N(0, I), same shape as x_0

  This means: at timestep t, we can sample a noisy latent in one shot.
  Training does not need to run through 1000 steps -- it jumps directly.
```

This shortcut is key to efficient training: pick a random timestep `t`, corrupt `x_0` directly to `x_t`, ask the model to predict the noise, compute the loss, update weights.

### The model: what goes in and what comes out

The denoising model -- a [U-Net](../glossary.md#u-net) or a DiT (Diffusion [Transformer](../glossary.md#transformer)) -- takes three inputs and produces one output:

```
DENOISING MODEL

  Inputs:
    noisy_latent    [B, C, H, W]     -- the partially-noised image latent
    timestep        [B]              -- integer from 0 to T, current noise level
    condition       [B, seq_len, D]  -- text embeddings, or None for uncond.

                    +-------------------+
  noisy_latent -->  |                   |
  timestep     -->  |   U-Net or DiT    |  --> predicted noise (or velocity)
  condition    -->  |                   |    same shape as noisy_latent
                    +-------------------+

  The timestep tells the model how much noise to expect.
  The condition (text) tells the model what content to aim for.
  The model outputs what it thinks the noise component is.
```

The **U-Net** architecture (used in Stable Diffusion) has a contracting path (encoder) and an expanding path (decoder) with skip connections that pass spatial detail across. The **DiT** (Diffusion Transformer, used in Flux, SD3, and LTX-Video) replaces the convolutional U-Net with a Vision Transformer operating on patches.

### The noise scheduler: controlling the trajectory

A noise scheduler defines the sequence of noise levels used for both training (forward process) and generation (reverse process). Different schedulers make different trade-offs between quality, speed, and faithfulness:

```
NOISE SCHEDULERS

  DDPM (Denoising Diffusion Probabilistic Models):
    - The original. Full 1000 steps required.
    - Stochastic (adds a small amount of randomness at each step).
    - Slow but theoretically principled.

  DDIM (Denoising Diffusion Implicit Models):
    - Deterministic. Can skip steps (e.g., use 50 out of 1000).
    - Same model, different sampling formula.
    - Faster at the cost of some diversity.

  Euler / DPM-Solver:
    - Treat the denoising ODE as a differential equation.
    - Higher-order numerical integration = fewer steps needed.
    - DPM-Solver can produce good results in 10-20 steps.

  Flow Matching (used by Flux, SD3, LTX-Video):
    - Learns straight-line paths from noise to data.
    - Instead of predicting noise, predicts velocity (direction of travel).
    - More uniform loss across all timesteps.
    - Typically 20-50 steps for high quality.

  Steps vs. quality trade-off:
    5 steps  --> fast, lower quality, may miss fine detail
   20 steps  --> good quality, most practical use cases
   50 steps  --> high quality, slower
  100+ steps --> diminishing returns for most schedulers
```

### Flow Matching: the scheduler LTX-Video uses

[Flow Matching](../glossary.md#flow-matching) deserves special attention because it is what LTX-Video, Flux, and SD3 use -- and it is conceptually different from [DDPM](../glossary.md#ddpm).

In DDPM, the forward process is a noisy diffusion that follows a curved path from data to noise. The model learns to reverse that curved path, step by step.

In Flow Matching, the forward process is a straight line -- literally a linear interpolation between data and noise. The model learns to predict the *velocity* (direction) of travel along that line:

```
DDPM vs. FLOW MATCHING

  DDPM path (noisy, curved):            Flow Matching path (straight):

       data                                  data
         \                                    |
          \    (curved, stochastic)           | (straight line)
           \                                  |
            \                                 |
           noise                            noise

  Model predicts: added noise           Model predicts: velocity vector
  (how noisy is this input?)            (which direction to step?)

  Flow matching advantage:
  - Paths don't cross, so each trajectory is unambiguous
  - More uniform training signal at all timesteps
  - Fewer steps needed for high-quality results
```

The model output is called a "velocity" -- the direction and magnitude to move in latent space to denoise. During sampling, you follow this vector field for the specified number of steps.

### Classifier-Free Guidance (CFG): amplifying the text signal

A text-conditioned model can generate images that loosely match a prompt. But "loosely match" is often not what you want -- you want the image to *strongly* match the prompt, even if that means some loss of diversity.

**Classifier-Free Guidance (CFG)** achieves this by running the model twice per step:

```
CLASSIFIER-FREE GUIDANCE

  At each denoising step, run the model twice:

  Run 1: model(noisy_latent, timestep, text_condition)
          --> noise_conditional       (guided by text)

  Run 2: model(noisy_latent, timestep, null_condition)
          --> noise_unconditional     (unguided)

  Final prediction:
    noise_cfg = noise_unconditional
              + cfg_scale * (noise_conditional - noise_unconditional)

  Intuition: noise_conditional - noise_unconditional
             = "the direction the text is pulling the output"

  Amplify that direction by cfg_scale.

  cfg_scale = 1.0  --> same as no guidance (ignore text)
  cfg_scale = 7.5  --> strong guidance (typical for images)
  cfg_scale = 15+  --> very strong, may over-saturate or distort

  Higher CFG = more faithful to prompt, less diverse/natural
```

CFG is why most diffusion model APIs have a `guidance_scale` or `cfg_scale` parameter. The cost: every step runs the model twice, doubling compute. Some techniques (PAG, CFG-distillation) reduce this, but standard CFG is the baseline.

### Conditioning: connecting text to image

The text prompt reaches the model through a text encoder that converts words into embedding vectors:

```
CONDITIONING PIPELINE

  "a tabby cat sitting in sunlight"
                |
         [Text Encoder]           (CLIP, T5, or both)
                |
         text embeddings          shape: [batch, seq_len, D]
                |
  fed into denoising model via cross-attention
  (each image patch attends to the most relevant text tokens)

  Common text encoders:
    CLIP    -- fast, 77-token limit, good for short prompts
    T5-XXL  -- slower, 512+ token limit, better for long detailed prompts
    Both    -- some models (SD3, Flux) use CLIP + T5 together
```

The text embeddings are computed once per prompt, then reused across all denoising steps. This is why changing the prompt but keeping the same random seed produces a related but different image -- the noise trajectory is the same, but the conditioning changes the direction of every denoising step.

### The sampling loop

Putting it all together: the full generation algorithm:

```
DIFFUSION SAMPLING LOOP

  Inputs:
    text_prompt        --> text_embeddings  (computed once)
    cfg_scale          = 7.5
    num_steps          = 50
    seed               = 42

  1. Sample random noise:
       x_T ~ N(0, I)   shape: [B, C, H, W] in latent space

  2. For t = T down to 0 (num_steps iterations):

     a. Predict noise (or velocity):
          eps_cond   = model(x_t, t, text_embeddings)
          eps_uncond = model(x_t, t, null_embeddings)

     b. Apply CFG:
          eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

     c. Denoise one step (scheduler-specific formula):
          x_{t-1} = scheduler.step(x_t, t, eps)

     d. x_t = x_{t-1}  (next iteration)

  3. Decode:
       image = vae_decoder(x_0)   --> pixel output

  Most of the compute is in step 2 (the loop).
  The loop runs num_steps times, each requiring 2 model forward passes.
  50 steps x 2 passes = 100 model forward passes per image.
```

This loop is the heart of any image or video generation pipeline. Everything else -- the VAE, the text encoder, the scheduler -- exists to support it.


## Why It Matters for MLX

### The sampling loop is where all the compute lives

When you profile a diffusion inference run, the breakdown is roughly:

```
COMPUTE BUDGET PER GENERATION

  Text encoding:       ~1%    (runs once)
  VAE encode:          ~2%    (runs once, image-to-image only)
  Sampling loop:       ~95%   (runs num_steps * 2 times)
  VAE decode:          ~2%    (runs once)

  The denoising model is called 100+ times per generation.
  Any performance improvement in the model multiplies by that factor.
```

This is why model optimization -- weight quantization, operator fusion, [Flash Attention](../glossary.md#flash-attention) -- matters far more for the denoising step than for any other component. A 2x speedup in the denoising model means ~2x faster generation end to end. A 2x speedup in the VAE decoder saves almost nothing in practice.

### Matching the scheduler exactly or outputs are wrong

The scheduler is not just a speed knob -- it determines the mathematical relationship between model outputs and the next noisy state. If you implement the scheduler formula incorrectly (wrong constant, wrong sign, wrong order of operations), the model will produce garbage:

```
COMMON SCHEDULER PORTING MISTAKES

  Problem                          Effect
  -------------------------------- ------------------------------------------
  Wrong alpha/beta values          Output is systematically darker/lighter,
                                   wrong saturation

  Wrong denoising formula          Images look like noise even at step 0
                                   (model never converged to clean)

  Wrong timestep ordering          Generation runs backwards (noise increases
  (ascending instead of            instead of decreasing) -- pure static output
  descending)

  Mismatched CFG implementation    Output ignores prompt completely (forgot
                                   to subtract unconditional prediction) or
                                   is over-saturated (wrong scale direction)

  Flow matching: predicting noise  Model outputs velocity but you treat it
  instead of velocity              as noise prediction -- wrong step formula
                                   produces incoherent images
```

When porting an LTX-Video or similar flow matching model, confirm that your model output is indeed a velocity, and that your scheduler step formula is the rectified flow variant -- not the DDPM or [DDIM](../glossary.md#ddim) formula, which are mathematically incompatible.

### The inference steps parameter is the main quality knob

Unlike most neural network parameters, inference steps are chosen at runtime, not baked into the model:

```
STEPS VS. QUALITY (approximate, model-dependent)

  LTX-Video (flow matching):
    10 steps  --> fast preview, visible artifacts
    20 steps  --> good for development and testing
    40 steps  --> high quality for production
    50 steps  --> marginal improvement over 40, 25% slower

  Stable Diffusion (DDIM):
    10 steps  --> fast, noticeably lower quality
    25 steps  --> good balance
    50 steps  --> high quality
   100+ steps --> diminishing returns

  When debugging a port:
  - Start with 1-2 steps to check that the model runs at all
  - Use 5-10 steps to check that denoising is happening (output should
    look less noisy than input)
  - Use full steps only once basic correctness is confirmed
```

### LTX-Video: DiT + Flow Matching

LTX-Video is built on a Diffusion Transformer (DiT) with flow matching. Understanding how this differs from the standard DDPM/U-Net pipeline is essential for working on it:

```
LTX-VIDEO INFERENCE PIPELINE

  Text prompt
      |
      v
  [T5 Text Encoder]       --> text_embeddings: [B, seq_len, D]
      |
  [Random noise latent]   <-- sampled N(0,1): [B, T/8, C, H/32, W/32]
      |                       video latent space (very compressed)
      |
      +------- SAMPLING LOOP (flow matching, ~20-50 steps) ----------+
      |                                                               |
      |  For each step t:                                             |
      |    velocity_cond   = DiT(latent, t, text_embeddings)         |
      |    velocity_uncond = DiT(latent, t, null_embeddings)         |
      |    velocity_cfg    = velocity_uncond                         |
      |                    + cfg_scale * (velocity_cond              |
      |                                 - velocity_uncond)           |
      |    latent = flow_step(latent, velocity_cfg, t)               |
      |                                                               |
      +---------------------------------------------------------------+
      |
      v
  [Clean latent]          -- shape: [B, T/8, C, H/32, W/32]
      |
      v
  [VAE Decoder]           --> pixel frames: [B, T, H, W, 3]
      |
      v
  Video output

  Key numbers for LTX-Video:
    DiT: 28 transformer blocks
    Typical: 20-50 sampling steps per generation
    Each step: 2 DiT forward passes (CFG)
    Total: 40-100 DiT forward passes per video
```

### What to check first when a port produces wrong output

When your MLX port of a diffusion model generates garbage, the failure points follow a predictable hierarchy:

```
DEBUGGING CHECKLIST (in order of likelihood)

  1. Scheduler formula wrong
     - Print x_t at step 0, 10, 20, 50 -- does noise decrease?
     - If x_T (initial noise) looks the same at step 50, scheduler is broken

  2. Model outputs the wrong thing (noise vs. velocity)
     - Flow matching models output velocity, not noise
     - Check the original model config: "prediction_type" field
     - "epsilon" = predicting noise (DDPM style)
     - "v_prediction" or "flow" = predicting velocity

  3. CFG applied incorrectly
     - Disable CFG (set cfg_scale=1.0, run only conditional pass)
     - Does output improve? If yes, CFG formula is the bug

  4. Timestep encoding wrong
     - The model embeds the timestep and uses it internally
     - Wrong timestep range (0-1 vs 0-1000) breaks the embedding
     - Check what range the original model expects

  5. Conditioning not reaching the model
     - Run with a very specific prompt and a null prompt
     - If both outputs look identical, cross-attention is broken
```

### Channel ordering reminder (applies here too)

The latent that flows through the sampling loop is a tensor. As noted in the Latent Space page, MLX is channel-last (NHWC / NTHWC) while PyTorch is channel-first (NCHW / NTCHW). The sampling loop operates on this tensor directly, so any transposition errors will produce systematic artifacts -- not random garbage, but wrongly-arranged spatial structure.

```python
# Expected latent shape through the LTX-Video sampling loop (MLX):
# [batch, frames//8, height//32, width//32, channels]  (NTHWC, channel-last)

# NOT:
# [batch, channels, frames//8, height//32, width//32]  (NTCHW, PyTorch)

# When initializing random noise at the start of generation:
latent = mx.random.normal(shape=(batch, t_compressed, h_compressed, w_compressed, channels))
#                                                                                 ^^^^^^^^^
#                                                      channels last in MLX

# When passing to the DiT: confirm your model expects channel-last input
# or transpose before the forward pass and transpose back after.
```


## Key Terms

| Term | Definition |
|------|------------|
| **Diffusion** | A class of generative models that learn to reverse a noise-addition process; generate data by iteratively denoising random noise |
| **[Denoising](../glossary.md#denoising)** | The core operation: given a noisy input and its noise level, predict and remove the noise; the skill a diffusion model is trained to perform |
| **[Forward Process](../glossary.md#forward-process)** | The training-time procedure of progressively adding Gaussian noise to clean data over T timesteps until it becomes pure noise; defined by a fixed mathematical formula, not learned |
| **[Reverse Process](../glossary.md#reverse-process)** | The generation-time procedure of iteratively applying the denoising model to remove noise step by step; what the trained model learns to do |
| **[Noise Schedule](../glossary.md#noise-schedule)** | The sequence of noise levels (betas or alphas) used across timesteps; defines how quickly the signal is corrupted in the forward process and how it is recovered in the reverse |
| **[Timestep](../glossary.md#timestep)** | An integer t from 0 (clean) to T (pure noise) indexing the current noise level; passed as input to the denoising model so it knows how noisy its input is |
| **U-Net** | A convolutional neural network architecture with encoder, decoder, and skip connections; the original backbone for diffusion models (used in Stable Diffusion) |
| **DiT (Diffusion Transformer)** | A transformer-based architecture for diffusion models; operates on flattened patches of the latent; used in Flux, SD3, LTX-Video |
| **CFG (Classifier-Free Guidance)** | A technique that runs the model with and without conditioning at each step, then amplifies the difference; makes output more faithful to the prompt at the cost of 2x compute |
| **DDPM** | Denoising Diffusion Probabilistic Models; the original formulation; stochastic, requires many steps (typically 1000) |
| **DDIM** | Denoising Diffusion Implicit Models; deterministic; uses the same trained model as DDPM but with a step-skipping formula; enables quality generation in 50-100 steps |
| **[Euler Scheduler](../glossary.md#euler-scheduler)** | A simple ODE solver for diffusion; treats denoising as numerical integration of an ODE; fast and widely used |
| **[DPM-Solver](../glossary.md#dpm-solver)** | A higher-order ODE solver for diffusion; fewer steps needed for comparable quality (10-20 steps); common in production pipelines |
| **Flow Matching** | A training and sampling approach that learns straight-line paths from noise to data; the model predicts velocity instead of noise; used in Flux, SD3, LTX-Video |
| **[Conditioning](../glossary.md#conditioning)** | External information (text, image, depth map) provided to the denoising model to guide what it generates; fed in via cross-attention |
| **[Sampling Steps](../glossary.md#sampling-steps)** | The number of denoising steps performed during generation; the primary trade-off lever between speed and output quality |
| **[Noise Prediction](../glossary.md#noise-prediction)** | The DDPM/DDIM model output mode: predict the noise that was added to the input (also called "epsilon prediction") |
| **[Velocity Prediction](../glossary.md#velocity-prediction)** | The flow matching model output mode: predict the direction of travel from noise to data; also called "v-prediction" in some frameworks |


## Sources

- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*. The foundational paper; introduces the forward/reverse process formulation and the denoising score matching objective. Available at [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Rombach, R., et al. (2022). "High-Resolution Image Synthesis with [Latent Diffusion](../glossary.md#latent-diffusion) Models." Introduces running diffusion in latent space (the architecture that powers Stable Diffusion); also covers DDIM sampling and conditioning via cross-attention. Available at [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- Peebles, W., & Xie, S. (2023). "Scalable Diffusion Models with Transformers." Introduces the DiT architecture; demonstrates that a pure transformer backbone outperforms U-Net on image generation benchmarks. Available at [arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
- Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling." Introduces the flow matching objective; explains straight-line (rectified) flows and velocity prediction. Available at [arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
- Weng, L. "What are Diffusion Models?" A thorough blog post covering the DDPM math, score matching, and the connection between different formulations. Available at [lilianweng.github.io/posts/2021-07-11-diffusion-models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- Hugging Face. "The Annotated Diffusion [Model](../glossary.md#model)." A line-by-line walkthrough of implementing DDPM with annotated code. Available at [huggingface.co/blog/annotated-diffusion](https://huggingface.co/blog/annotated-diffusion)


## See Also

- [Latent Space](07-latent-space.md) -- prerequisite; the diffusion process runs inside latent space; the VAE encoder and decoder bookend the sampling loop
- [Attention](06-attention.md) -- the cross-attention mechanism is how text conditioning reaches the denoising model at each step; understanding Q/K/V is necessary to understand why the text prompt controls generation
- [Transformers](05-transformers.md) -- DiT-based models (LTX-Video, Flux, SD3) are transformers applied to image/video patches; the transformer block structure is identical
