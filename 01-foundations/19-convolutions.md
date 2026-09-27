# Convolutions & Patchification

> **One-liner:** A [convolution](../glossary.md#convolution) slides one small set of weights over an image, a video or a signal; it is the building block of every VAE, U-Net and patch embedding in image and video models -- and porting it to MLX means getting four things exactly right: the channels-last layout, the weight transpose (which differs for transposed convolutions), padding, and groups.

## The Intuition

A linear layer connects every input to every output. For a 512×512 image that is absurd: each output would need a weight for each of 786,432 input values, and a cat in the top-left corner would have to be learned separately from the same cat in the bottom-right.

A convolution does the opposite. It learns one small **kernel** -- say 3×3 pixels × all input channels -- and slides it across the whole image, computing the same weighted sum at every position. The same weights detect the same pattern (an edge, a texture) wherever it appears, and the layer's size no longer depends on the image size.

```
2D CONVOLUTION, 3x3 kernel, one output channel

  input (H x W x C_in)              kernel (3 x 3 x C_in)        output (H' x W')

  . . . . . . .
  . [# # #] . .    weighted sum      w w w                        . . . . .
  . [# # #] . .  ------------------> w w w  --> one number -->    . o . . .
  . [# # #] . .                      w w w                        . . . . .
  . . . . . . .    then slide by `stride` and repeat

  C_out output channels = C_out different kernels
```

Image and video models use convolutions in three places this page covers: **VAEs and U-Nets** (stacks of 2D or 3D convolutions that shrink and grow feature maps), **upsampling** (transposed convolutions, pixel shuffle), and **patch embedding** (the first layer of a DiT or ViT, which cuts the input into patches -- it's a convolution in disguise).

---

## How It Actually Works

### The four knobs: kernel, stride, padding, dilation (and groups)

```
  kernel_size   size of the window: 3 -> 3x3 (2D), 3x3x3 (3D)
  stride        step between windows: 2 halves each spatial dimension
                  (the standard way encoders downsample)
  padding       zeros added around the input so windows fit at the edges;
                  padding = kernel//2 with stride 1 keeps the size unchanged
  dilation      gaps inside the window: dilation 2 makes a 3x3 kernel
                  cover 5x5 (wider view, same number of weights)
  groups        split channels into G independent groups, each with its own
                  kernels; groups = C_in is a "depthwise" convolution

  output size = floor((in + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
```

### 1D, 2D, 3D

Same operation, different number of sliding dimensions:

```
  Conv1d   slides over time            audio, sequences      (N, L, C)
  Conv2d   slides over height, width   images, latents       (N, H, W, C)
  Conv3d   slides over time, H, W      video VAEs            (N, T, H, W, C)
                                         (LTX, Wan, Hunyuan)
```

Shapes above are MLX's. MLX 0.32 has all of them in `mlx.nn` (`Conv1d`, `Conv2d`, `Conv3d`) and in `mlx.core` (`conv1d`, `conv2d`, `conv3d`, `conv_general`).

### Causal 3D convolution: video that doesn't look ahead

Video VAEs usually make their 3D convolutions **causal in time**: frame *t* may only see frames ≤ *t*. Instead of padding the time axis symmetrically, they pad `kernel_t - 1` frames on the *past* side only -- typically by repeating the first frame -- and use no time padding in the convolution itself. The output has as many frames as the input, and the first frame can be encoded on its own (which is why these VAEs map 1 + 8k frames to 1 + k latent frames).

```python
# Causal Conv3d, kernel 3 in time (MLX, channels-last: N, T, H, W, C)
conv = nn.Conv3d(c_in, c_out, kernel_size=3, padding=(0, 1, 1))   # no time padding
x = mx.concatenate([x[:, :1]] * 2 + [x], axis=1)                   # repeat frame 0 twice
y = conv(x)                                                         # T frames in, T frames out
```

Checked against the same construction in PyTorch: max difference 6e-7. Exactly *how* the past is padded (first-frame replication, zeros, or a cache of previous chunk frames) varies by model and must be copied from the reference.

### Transposed convolution: learned upsampling

A **transposed convolution** (`ConvTranspose2d`, sometimes wrongly called "deconvolution") runs the convolution's geometry backwards: with stride 2 it *doubles* the spatial size. Decoders use it, or -- more often in modern VAEs, to avoid checkerboard artifacts -- a plain upsample (`nn.Upsample`, nearest) followed by a normal convolution.

```
  output size = (in - 1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

  in 16, kernel 4, stride 2, padding 1  ->  32
```

### Pixel shuffle: upsampling by rearranging channels

`PixelShuffle(r)` turns a `(H, W, C·r²)` feature map into `(H·r, W·r, C)` by moving groups of channels into spatial positions. No weights, just a reshape and transpose -- but the order in which the `C·r²` channels are split matters. MLX has no `PixelShuffle` layer; in channels-last it is:

```python
def pixel_shuffle(x, r):                       # x: (B, H, W, C*r*r), PyTorch channel order
    B, H, W, C = x.shape
    x = x.reshape(B, H, W, C // (r * r), r, r)
    x = x.transpose(0, 1, 4, 2, 5, 3)          # -> B, H, r, W, r, C
    return x.reshape(B, H * r, W * r, C // (r * r))
```

This matches `torch.nn.PixelShuffle` exactly (difference 0). Splitting the channels as `(r, r, C)` instead of `(C, r, r)` also runs and gives the right shape -- and a max error of 3.6 on random data.

### Patch embedding: a convolution with kernel = stride

A DiT or ViT turns an image (or latent) into a sequence of tokens by cutting it into p×p patches and projecting each patch to the model width. That is exactly a convolution with `kernel_size = stride = p`: the windows don't overlap, and each one produces one token.

```
  latent (1, 64, 64, 16)  --Conv2d(16, 1152, kernel=2, stride=2)-->  (1, 32, 32, 1152)
                          --reshape-->  (1, 1024, 1152)  = 1,024 tokens

  identical to: cut into 2x2x16 patches, flatten each to 64 values, Linear(64, 1152)
```

Both forms appear in checkpoints: some models store the patch embedding as a conv weight `(D, C, p, p)`, others as a linear weight `(D, p·p·C)` applied after a "patchify" reshape. They are interchangeable -- verified equal to 3e-7 -- *if* the flattening order of the patch (which of p, p, C varies fastest) matches the one the weights were trained with. Video DiTs do the same in 3D with `(p_t, p_h, p_w)` patches.

---

## Why It Matters for MLX

### Layout: MLX is channels-last, for activations *and* weights

PyTorch convolutions are channels-first; MLX's are channels-last. Both the input and the weight must be permuted, and the weight permutation depends on the layer type:

```
                   PyTorch weight          MLX weight            permutation
  Conv1d           (out, in, K)            (out, K, in)          (0, 2, 1)
  Conv2d           (out, in, kH, kW)       (out, kH, kW, in)     (0, 2, 3, 1)
  Conv3d           (out, in, kT, kH, kW)   (out, kT, kH, kW, in) (0, 2, 3, 4, 1)
  ConvTranspose2d  (in, out, kH, kW)  <--  (out, kH, kW, in)     (1, 2, 3, 0)
  ConvTranspose3d  (in, out, kT, kH, kW)   (out, kT, kH, kW, in) (1, 2, 3, 4, 0)

  With groups, PyTorch's "in" is in/groups; MLX's last dim is too.
```

Note the transposed-convolution row: **PyTorch stores `(in, out, ...)`**, so the Conv2d permutation is wrong for it. When `in ≠ out`, MLX raises a channel-mismatch error. When `in == out` -- common in decoders -- the wrong permutation runs silently and produces the right shape with the wrong values (measured max error 1.7).

All rows were checked against PyTorch on MLX 0.32.2 (max difference ≤ 6e-7).

**Transpose once, at the model boundary.** Convert the input to NHWC (or NTHWC) at the start, keep everything channels-last inside, and convert back at the end for comparison. Transposing around every convolution is not much slower (measured 4.0 ms vs 3.7 ms for 8 convolutions), but every extra transpose is a place to get the axis order wrong.

### Padding modes, groups, and what MLX 0.32 still lacks

PyTorch's `padding_mode="reflect"` / `"replicate"` has no equivalent argument on MLX's conv layers: pad with `mx.pad` first, then convolve with `padding=0`. Since MLX 0.32, `mx.pad` supports `"reflect"` and `"symmetric"` in addition to `"constant"` and `"edge"` (earlier versions had only the last two):

```python
# PyTorch: nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect")
x = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], mode="reflect")   # N, H, W, C
y = nn.Conv2d(c, c, 3, padding=0)(x)                              # matches torch, 6e-7
```

Match the mode exactly: `"edge"` (= PyTorch `replicate`) instead of `"reflect"` gave a max error of 1.9 on random data.

```
  PyTorch feature                   MLX 0.32.2
  ---------------                   ----------
  padding_mode reflect/replicate/   mx.pad(mode="reflect" / "edge" / "symmetric"),
    circular                          then padding=0; circular: build it with
                                      mx.concatenate
  groups in Conv1d / Conv2d         nn.Conv1d / nn.Conv2d(groups=G)
  groups in Conv3d                  not supported ("Can only handle groups != 1
                                      in 1D or 2D convolutions"): split channels
                                      into groups, one conv3d per group,
                                      concatenate (verified vs PyTorch)
  groups in ConvTranspose           not on nn.ConvTranspose*; use
                                      mx.conv_transpose2d(..., groups=G)
                                      directly (weight layout below)
  nn.PixelShuffle                   the reshape above
```

For a **grouped transposed convolution**, PyTorch's weight is `(in, out/G, kH, kW)`. MLX expects `(out, kH, kW, in/G)` with the output channels ordered group by group:

```python
w = torch_w.numpy().reshape(G, cin // G, cout // G, kH, kW)   # split "in" into groups
w = w.transpose(0, 2, 3, 4, 1).reshape(cout, kH, kW, cin // G)
y = mx.conv_transpose2d(x, mx.array(w), stride=2, padding=1, groups=G)
```

This matches PyTorch (1e-7) on MLX 0.32.2. **On MLX 0.31.1 the same call ran and returned wrong values** (max error 2.1) -- if you are pinned to an older MLX, split into per-group `conv_transpose2d` calls instead, which is correct on both.

### Port failures

```
CONVOLUTION PORT FAILURES

  Symptom                                     Likely cause
  -------                                     ------------
  Shape error "[conv] Expect the input        input still NCHW, or weight not
    channels ... to match"                      permuted
  Right shapes, garbage output                weight reshaped instead of permuted
                                                (measured error 3.0), or wrong
                                                permutation
  Decoder output wrong, encoder fine          ConvTranspose weight permuted like a
                                                Conv2d: (in, out, ...) not (out, in, ...)
  Colored/blotchy borders only                padding mode: reflect or replicate
                                                in the reference, zeros in the port
  Grouped decoder upsampling wrong on         grouped mx.conv_transpose2d on
    MLX < 0.32                                  MLX 0.31: upgrade, or one call
                                                per group
  Output one pixel too small after upsample   output_padding missing on ConvTranspose
  Video: first frames wrong, later fine       causal time padding done
                                                symmetrically, or with zeros instead
                                                of repeated frames
  Checkerboard / tiled pattern                pixel-shuffle or patchify channel
                                                order wrong
  DiT tokens scrambled, shapes correct        patch flattening order (p, p, C) vs
                                                (C, p, p) differs from the checkpoint
```

---

## Key Terms

| Term | Definition |
|------|------------|
| **Convolution** | A layer that slides one small kernel of weights over the input and computes the same weighted sum at every position. |
| **Kernel** | The small learned weight window of a convolution (e.g. 3×3×C_in per output channel). |
| **Stride** | Step between kernel positions; stride 2 halves each spatial dimension. |
| **Padding** | Values added around the input so the kernel fits at the edges; zeros, reflected or replicated values depending on the model. |
| **Dilation** | Gaps inside the kernel window, widening its view without adding weights. |
| **Groups / depthwise convolution** | Channels split into independent groups; with groups = channels, each channel has its own kernel. |
| **Causal convolution** | A convolution padded only on the past side along time, so no output depends on future frames. |
| **Transposed convolution** | Learned upsampling that runs a convolution's geometry in reverse; PyTorch stores its weight as (in, out, ...). |
| **Pixel shuffle** | Upsampling by rearranging groups of channels into spatial positions; no weights. |
| **Patch embedding** | The first layer of a DiT/ViT: a convolution with kernel = stride = patch size, turning an image or latent into tokens. |
| **NCHW / NHWC** | Channels-first (PyTorch) vs channels-last (MLX) memory layout for images; NCTHW / NTHWC for video. |

---

## Sources

- MLX API reference, convolution layers and functions (`nn.Conv1d/2d/3d`, `nn.ConvTranspose1d/2d/3d`, `mx.conv3d`, `mx.conv_transpose2d`, `mx.pad`): [ml-explore.github.io/mlx/build/html/python/nn/layers.html](https://ml-explore.github.io/mlx/build/html/python/nn/layers.html)
- PyTorch `Conv2d`, `ConvTranspose2d`, `PixelShuffle` documentation (weight layouts, output size formulas): [pytorch.org/docs/stable/nn.html#convolution-layers](https://pytorch.org/docs/stable/nn.html#convolution-layers)
- Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." [arxiv.org/abs/1603.07285](https://arxiv.org/abs/1603.07285)
- Odena, A., Dumoulin, V., & Olah, C. (2016). "Deconvolution and Checkerboard Artifacts." [distill.pub/2016/deconv-checkerboard](https://distill.pub/2016/deconv-checkerboard/)
- Shi, W., et al. (2016). "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" (pixel shuffle). [arxiv.org/abs/1609.05158](https://arxiv.org/abs/1609.05158)
- Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words" (ViT patch embedding). [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- Every MLX-vs-PyTorch equivalence and limitation on this page: MLX 0.32.2 against PyTorch on CPU, random weights and inputs (the grouped transposed-convolution bug: reproduced on 0.31.1, fixed in 0.32.2).

---

## See Also

- [Neural Networks](02-neural-networks.md) -- prerequisite; first introduction of Conv2d
- [Latent Space](07-latent-space.md) -- the VAE encoders and decoders built from these layers
- [Normalization](14-normalization.md) -- GroupNorm, which sits between convolutions in VAEs and has the same layout trap
- [Transformers](05-transformers.md) -- DiTs, whose first layer is a patch embedding
- [Porting Guide](../03-contributing/02-porting-guide.md) -- weight conversion in general
- [Video Generation](../02-ecosystem/05-video-generation.md) -- the video models whose VAEs use causal 3D convolutions
