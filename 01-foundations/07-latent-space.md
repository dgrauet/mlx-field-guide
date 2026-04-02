# Latent Space

> **One-liner:** A latent space is a compressed representation where high-dimensional data (like images) is encoded into a smaller, meaningful form that captures its essential features.

## The Intuition

You know from the Transformers and Attention pages that the model operates on vectors of numbers -- token embeddings, hidden states, patch representations. All of that is computation in a learned, compressed space. But when we talk about *latent space* in image and video generation, we mean something more specific: the practice of compressing raw pixel data into a much smaller grid of numbers before doing any of the expensive computation.

### The photo compression analogy

A 512x512 RGB image is 786,432 numbers. You could describe that same image in a sentence: "a red car parked on a highway at sunset, lens flare in the upper left corner." That sentence is a few dozen words. Somewhere between the raw pixels and the caption is a compressed description -- the essential content of the image, stripped of redundancy.

A latent space encoder does something like this, but as a learned numeric compression:

```
COMPRESSION ANALOGY

  Raw image: 786,432 numbers (512 x 512 x 3 RGB channels)
      |
      |  "a red car at sunset"
      |  "soft lighting, warm tones"
      |  "highway background"
      |
      v
  Latent:    16,384 numbers (64 x 64 x 4 channels)
             -- 48x fewer numbers, same essential information
```

Like a ZIP file, but lossy: the latent keeps meaning, not every pixel.

### What "nearby means similar" looks like

In pixel space, two images can be identical in content but completely different in raw pixels -- a photo and its slightly rotated version would have completely different pixel values but nearly identical content.

In latent space, semantically similar things end up close together:

```
LATENT SPACE STRUCTURE

  "sunny beach" -------- "sunny shore"
        |                      |
        |                      |
  "cloudy beach" -------- "stormy coast"

  Semantically similar images are nearby in latent space.
  Moving along the "weather axis" smoothly morphs from sunny to stormy.
  Moving along the "location axis" moves from beach to ocean.
```

You can interpolate between two latents to get a smooth blend of two images -- something impossible to do meaningfully in raw pixel space.

### The encoder-decoder pipeline

Two components handle the translation between pixel space and latent space:

- The **encoder** squeezes raw pixel data down to a compact latent.
- The **decoder** expands that compact latent back to pixel data.

```
AUTOENCODER PIPELINE

  Pixel Space                    Latent Space              Pixel Space
  (high-dimensional)             (compressed)              (reconstructed)

  [512x512x3 image]              [64x64x4]                [512x512x3 image]
         |                          |                             |
         |                          |                             |
         v                          v                             v
  +-----------+   encode    +-------------+   decode   +-----------+
  |           | ----------> |   LATENT    | ---------> |           |
  |  Original |             |   VECTOR    |            | Reconstructed
  |   Image   |             |             |            |   Image   |
  +-----------+             +-------------+            +-----------+

  Information                Almost all the            Visually very
  rich, slow to              information, in a          close to original
  process directly           fraction of the space      (small loss)
```

The model trains by trying to reconstruct the original image from its latent. The reconstruction loss forces the encoder to keep what matters and discard redundancy.


## How It Actually Works

### Autoencoders: the basic mechanism

An **autoencoder** is a neural network with a bottleneck. The architecture forces information through a narrow channel, which compresses it:

```
AUTOENCODER ARCHITECTURE

   Input image: (512 x 512 x 3) = 786,432 values
         |
   +-----+------------------------------+
   |           ENCODER                 |
   |                                   |
   |  Conv  -->  Conv  -->  Conv  -->  |
   |  128ch      256ch      512ch      |
   |                                   |
   +-----+------------------------------+
         |
         v  BOTTLENECK (latent space)
   [64 x 64 x 4] = 16,384 values  <-- 48x smaller
         |
   +-----+------------------------------+
   |           DECODER                 |
   |                                   |
   | Upsample --> Upsample --> Conv --> |
   |  512ch         256ch      3ch     |
   |                                   |
   +-----+------------------------------+
         |
   Output image: (512 x 512 x 3)

   Training objective: output should match input.
   Loss = pixel-wise reconstruction error.
```

The encoder uses strided convolutions to shrink spatial dimensions while increasing channels. The decoder uses transposed convolutions (or upsampling + convolution) to expand back. The bottleneck forces the network to develop an efficient encoding.

### VAE: adding a learned distribution

A plain autoencoder maps each input to a single point in latent space. The problem: the space between points is arbitrary. You can pick a point midway between two latents and the decoder has no idea what to do with it -- it was never trained on that point, so it produces garbage.

A **Variational Autoencoder (VAE)** solves this by mapping each input to a *region* (a probability distribution) instead of a single point:

```
AUTOENCODER vs. VAE

  Standard Autoencoder:             VAE:

  image --> encoder --> [z]         image --> encoder --> [mean, log_var]
                                                              |
                                                    sample z = mean + noise * std
                                                              |
  [z] --> decoder --> image^        [z] --> decoder --> image^

  Each input maps to ONE point.     Each input maps to a DISTRIBUTION.
  Interpolation between points      Interpolation is smooth: nearby
  produces garbage.                 distributions overlap.
```

The encoder outputs two things: a mean vector and a log-variance vector. To get a latent, you sample from the resulting Gaussian distribution during training. The key training trick: a **KL divergence** term in the loss pushes all those learned distributions toward a standard normal distribution, ensuring the latent space is smooth and well-organized.

```
VAE LOSS

  Total loss = Reconstruction loss + KL divergence

  Reconstruction loss:  "does the output look like the input?"
  KL divergence:        "does the distribution look like N(0, 1)?"

  KL term keeps the space organized:
  - No isolated clusters
  - Smooth interpolation anywhere
  - Sampling random z from N(0,1) produces valid images
```

Because the space is smooth and organized, you can sample a random latent from the prior (a standard normal) and the decoder will produce a coherent image. This is the property that makes VAEs useful for generation, not just compression.

### Latent dimensions in practice

Real model configurations follow a predictable pattern:

```
COMPRESSION RATIOS (common configurations)

  Stable Diffusion (SDXL):
    Pixel space:  1024 x 1024 x 3 = 3,145,728 values
    Latent space:  128 x  128 x 4 =    65,536 values
    Compression: 48x in each spatial dimension (factor of 8), 4 latent channels

  Stable Diffusion (SD 1.5):
    Pixel space:  512 x 512 x 3 = 786,432 values
    Latent space:  64 x  64 x 4 =  16,384 values
    Compression: 48x (factor of 8 spatial, 4 channels)

  LTX-Video:
    Pixel space:  [B, F, H, W, 3]   (batch, frames, height, width, channels)
    Latent space: [B, F/8, C, H/32, W/32]
    Compression: 8x temporal, 32x spatial per axis, C latent channels
```

The 4 latent channels are not the same as RGB channels -- they are learned abstract features. They do not map to interpretable colors. You cannot directly visualize a latent.

### Latent diffusion: why this matters for generation

The key insight of Stable Diffusion (and LTX-Video) is to run the entire diffusion process in latent space instead of pixel space:

```
LATENT DIFFUSION MODEL PIPELINE

  Step 1: ENCODE (once, before diffusion)
  ----------------------------------------
  Real images --> VAE encoder --> latents
  Store latents for training.

  Step 2: DIFFUSE (during training: add noise; during inference: remove noise)
  --------------------------------------------------------------------------
  latent --> add noise --> noisy latent
  Transformer/UNet predicts the noise at each step.
  Operates entirely in latent space (small grids).

  Step 3: DECODE (once, after diffusion)
  ----------------------------------------
  Clean latent --> VAE decoder --> generated image


  COMPARISON: pixel space vs. latent space diffusion

  Method              Space size          Cost of diffusion step
  ------------------  ------------------  ----------------------
  Pixel diffusion     786,432 values      Very high
  Latent diffusion     16,384 values      ~48x lower

  Every denoising step in the diffusion process operates on the latent.
  With 50 diffusion steps, the savings compound dramatically.
```

The diffusion process (covered in the next page) runs for many steps. Running those steps on 16K numbers instead of 786K numbers is the practical reason video and image generation is feasible.

### What the VAE's latent channels encode

The 4 latent channels are learned from data, not hand-designed. But research into what they capture shows they tend to organize around:

- Low-frequency content (overall color, brightness)
- Mid-frequency detail (textures, edges)
- High-frequency detail (fine structure)
- Structural information (depth, shape)

This is not a rule -- it emerges from training. The key point: the latent is a learned, abstract representation that the decoder knows how to interpret. Nothing outside the VAE knows what the individual channels mean.


## Why It Matters for MLX

### Every image and video generation model has a VAE

When you port a model like LTX-Video, the VAE is a required component. The inference pipeline looks like:

```
LTX-VIDEO INFERENCE PIPELINE (MLX)

  Text prompt
       |
       v
  [Text Encoder]   --> text embeddings (fixed for this prompt)
       |
       v
  [Random latent]  <-- sampled from N(0,1), shape [B, T, C, H, W]
       |
       | (50 diffusion denoising steps)
       v
  [DiT Transformer]   <-- guided by text embeddings, timestep
  (runs in latent space)
       |
       v
  [Clean latent]
       |
       v
  [VAE Decoder]    --> pixel frames
       |
       v
  Video output
```

The VAE decoder is the last thing that runs. Its output quality determines whether the video looks right.

### VAE is often the easiest component to port

When porting from CUDA to MLX, the VAE encoder and decoder are typically simpler to port than the main transformer:

- Standard convolution layers, no exotic ops
- No attention masking complexity
- Weight shapes are predictable (kernel_h x kernel_w x in_ch x out_ch)
- Deterministic (no sampling during inference -- you run the encoder in "deterministic mode" using the mean, not a sample)

The weight loading follows a direct pattern: convolutional layers and normalization layers, stacked in the encoder and decoder halves.

### Diagnosing VAE problems

When your VAE decoder output is wrong, the failure mode tells you what is broken:

```
COMMON VAE DEBUGGING SCENARIOS

  Symptom                            Likely cause
  ---------------------------------  ---------------------------
  Blurry output, correct colors      Decoder weight mismatch (shape OK,
                                     values wrong -- wrong checkpoint?)

  Correct sharpness, wrong colors    Channel ordering issue (RGB vs BGR)
                                     or normalization range wrong

  Tiled/block artifacts              Channel dimension transposition wrong
                                     (MLX is channel-last, PyTorch is
                                     channel-first)

  Complete garbage / noise           Wrong weights loaded entirely,
                                     or major dtype precision issue

  Output is exactly input            Decoder accidentally wired to
                                     skip the transformer -- latent
                                     is the input image's own encoding
```

The most common MLX-specific issue: **channel ordering**. PyTorch convolutions are `[batch, channels, height, width]` (NCHW). MLX convolutions are `[batch, height, width, channels]` (NHWC). When loading convolutional weights from a PyTorch checkpoint, the weight tensors need to be transposed accordingly.

```python
# PyTorch conv weight shape: [out_ch, in_ch, kernel_h, kernel_w]
# MLX conv weight shape:     [out_ch, kernel_h, kernel_w, in_ch]

# Transposing when loading:
pytorch_weight = checkpoint["vae.decoder.conv_in.weight"]  # [out, in, kH, kW]
mlx_weight = mx.array(pytorch_weight).transpose(0, 2, 3, 1)  # [out, kH, kW, in]
```

This transposition must happen for every convolutional layer in the VAE.

### LTX-Video's video latent space

LTX-Video extends the 2D image latent space to 3D video:

```
LTX-VIDEO LATENT SHAPE

  Pixel space:  [batch, frames,    height, width, 3]
                          |            |       |
                          | /8         | /32   | /32
                          v            v       v
  Latent space: [batch, frames/8, C, height/32, width/32]

  Example: generating 9 frames at 768x512
    Pixel:  [1, 9,   768, 512, 3]   = 10,616,832 values
    Latent: [1, 2, C,  24,  16]     = 768*C values

  At C=128: latent has 98,304 values -- 108x smaller than pixel space.

  The temporal compression (factor of 8 frames) is handled by
  3D convolutions in the VAE encoder/decoder.
```

The diffusion transformer operates on this small 3D latent. The 28 transformer blocks process `[frames/8 * height/32 * width/32]` patches per forward pass -- not millions of pixels. That is the architectural reason video generation is computationally tractable.

When debugging latent shapes in the LTX-Video MLX port, the key is tracking where the compression happens:

```python
# Expected shape flow through LTX-Video components:

# Input video frames (after preprocessing):
# video: [1, 9, 3, 768, 512]  (batch, frames, channels, height, width)

# After VAE encode:
# latent: [1, 2, 128, 24, 16]  (batch, f/8, latent_ch, h/32, w/32)

# After adding noise and running diffusion steps:
# denoised_latent: [1, 2, 128, 24, 16]  (same shape throughout diffusion)

# After VAE decode:
# output: [1, 9, 3, 768, 512]  (back to pixel space)
```

Shape mismatches at the encode or decode boundary are one of the first things to check when integrating the VAE into the larger pipeline.


## Key Terms

| Term | Definition |
|------|------------|
| **Latent Space** | The compressed, lower-dimensional space in which an encoder represents input data; semantically meaningful: nearby points represent similar inputs |
| **Latent** | A specific point or vector in latent space; the encoded representation of a single input (image, video, etc.) |
| **Encoder** | The neural network component that maps high-dimensional input data (pixels) to a compact latent representation |
| **Decoder** | The neural network component that maps a latent representation back to high-dimensional output (pixels); the inverse of the encoder |
| **Autoencoder** | A neural network with a bottleneck architecture that learns to compress input into a latent and reconstruct it; trained by minimizing reconstruction loss |
| **VAE (Variational Autoencoder)** | An autoencoder where the encoder outputs a probability distribution (mean + variance) over latent space rather than a single point; trained with both reconstruction loss and KL divergence; produces smooth, interpolatable latent spaces |
| **Reconstruction** | The decoded output of an autoencoder; a measure of how well the latent preserved the original input's information |
| **KL Divergence** | In VAEs, a regularization term that pushes the learned latent distributions toward a standard normal; what makes the latent space organized and sample-able |
| **Latent Dimensions** | The shape of the latent tensor; for image models typically (H/8 x W/8 x 4) or similar; for video models adds a temporal dimension |
| **Latent Diffusion** | The technique of running the diffusion (noise addition and removal) process in latent space rather than pixel space; enables image/video generation at practical computational cost |
| **Pixel Space** | The raw representation of an image as pixel values; high-dimensional and redundant; contrasted with latent space where information is compressed |


## Sources

- Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes." The foundational VAE paper; introduces the reparameterization trick and the ELBO objective that combines reconstruction loss with KL divergence. Available at [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." Introduces the latent diffusion approach that powers Stable Diffusion; Section 3 explains the VQ-regularized and KL-regularized autoencoders used to build the latent space. Available at [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- Weng, L. "From Autoencoder to Beta-VAE." A detailed walkthrough of the autoencoder family from vanilla AE to VAE to beta-VAE, with clear explanations of the KL divergence term and the latent space structure it produces. Available at [lilianweng.github.io/posts/2018-08-12-vae](https://lilianweng.github.io/posts/2018-08-12-vae/)


## See Also

- [Neural Networks](02-neural-networks.md) -- the encoder and decoder are neural networks built from the layer types (Conv, Linear, LayerNorm) described there; understanding forward passes and weight loading applies directly to the VAE
- [Diffusion](08-diffusion.md) -- the process that runs inside latent space; the next concept after understanding what latent space is
- [Attention](06-attention.md) -- attention operates in a learned latent space; transformer hidden states are themselves latent representations, enriched token by token
