# Checkpoints & Weight Formats

> **One-liner:** A checkpoint is just named tensors on disk plus a config, but the details -- pickle vs safetensors, sharding and its index, the dtype the weights are stored in, quantized layouts, which keys a loader expects -- decide whether a port loads safely, fully, and correctly. Most "the weights don't load" problems, and some "the weights load but the output is wrong" ones, are format problems.

## The Intuition

A trained model is a dictionary: `{"layers.0.attn.q_proj.weight": tensor, ...}`. Shipping it means writing that dictionary to disk with enough information to rebuild every tensor -- name, dtype, shape, bytes -- plus a `config.json` describing the architecture that the names refer to.

```
  a Hugging Face model repo

  config.json                         architecture: sizes, layer counts, flags
  generation_config.json              default generation settings (see Sampling)
  tokenizer.json, ...                 tokenizer files
  model.safetensors                   the weights (one file)
    or
  model-00001-of-00002.safetensors    the weights (sharded)
  model-00002-of-00002.safetensors
  model.safetensors.index.json        which tensor lives in which shard

  diffusers pipelines: one subfolder per component, each with its own
  config + weights:  transformer/  vae/  text_encoder/  scheduler/ ...
```

---

## How It Actually Works

### Formats

```
  format            what it is                               safe to load?   used by
  ------            ----------                               -------------   -------
  .bin / .pt / .pth PyTorch pickle: Python objects that       only with        older HF repos,
                      can run code when unpickled               weights_only    research checkpoints
  .safetensors      JSON header + raw tensor bytes            yes (no code)    HF standard, MLX
  .gguf             llama.cpp's format, many quant types      yes              llama.cpp, Ollama
  .npz              NumPy zip of arrays                       yes              mx.savez
```

**Pickle runs code.** A `.pt` file is a pickled Python object, and unpickling can execute arbitrary code. Recent PyTorch defaults to `torch.load(..., weights_only=True)`, which refuses anything but tensors and plain containers. Tested with torch 2.14.1 on a file carrying a code payload: the default load raised `UnpicklingError: Weights only load failed`, while `weights_only=False` ran the payload. Only pass `weights_only=False` for files you trust, and convert research checkpoints to safetensors once.

### Inside a safetensors file

```
  [ 8 bytes: header length N, little-endian u64 ]
  [ N bytes: JSON header                          ]
      { "__metadata__": {"format": "mlx"},
        "layer.weight": {"dtype": "BF16", "shape": [262208, 60],
                         "data_offsets": [754116352, 785581312]}, ... }
  [ raw tensor bytes, back to back                ]
```

Those values are from the first shard of `mlx-community/gemma-3-12b-it-4bit` (5.37 GB, 171 KB header, 1,293 tensors, dtypes `BF16` and `U32`). Because the header gives every tensor's exact byte range, a loader can memory-map the file and read only what it needs, with no parsing of the data itself.

### MLX loads lazily

`mx.load` returns arrays that are not read yet (see [Lazy Evaluation](15-lazy-evaluation.md)). Measured on that 5.37 GB shard, MLX 0.32.3:

```
  mx.load(...)           0.00 s    +0.00 GB active memory   1,293 arrays, nothing read
  mx.eval(all arrays)    0.44 s    +5.37 GB                 (file already in the OS page cache)
```

That's why loading a model is near-instant until the first forward pass, and why `model.load_weights(...)` followed by `mx.eval(model.parameters())` is the honest way to time or memory-check a load.

`mx.load` picks the format from the **file extension**. Files in the Hugging Face cache are symlinks (`snapshots/<rev>/model.safetensors`) to extension-less blobs (`blobs/<hash>`). Resolving the symlink and passing the blob path fails with `ValueError: [load] Unknown file format`. Load through the snapshot path, or pass `format="safetensors"`.

### Sharding and the index

Large models are split into shards of a few GB, and `model.safetensors.index.json` maps each tensor name to its shard. The index is **metadata that can go stale**. The `mlx-community/gemma-3-12b-it-4bit` repo ships two shards (`-of-00002`) but an index listing five (`-of-00005`, 24.4 GB total: the original bf16 checkpoint's index, copied during conversion). `mlx-lm` loads by globbing `model*.safetensors`, so it works. A loader that trusts `weight_map` fails on files that don't exist. When in doubt, read the shards' own headers.

### Dtypes on disk, and bfloat16 vs NumPy

Most checkpoints are stored in **bfloat16**. NumPy has no bfloat16, and every NumPy-based path fails on it (MLX 0.32.3, torch 2.14.1, safetensors 0.8.0):

```
  np.array(mlx_bf16_array)                 ValueError: 'bfloat16' is not a valid PEP 3118
                                             buffer format string
  torch_bf16_tensor.numpy()                TypeError: Got unsupported ScalarType BFloat16
  safetensors.numpy.load_file(bf16 file)   TypeError: data type 'bfloat16' not understood
  mx.load(bf16 file)                       ok -> mlx.core.bfloat16
```

In conversion scripts, either load directly with `mx.load` / `safetensors.torch`, or cast to float32 *before* crossing into NumPy (`t.float().numpy()`, `x.astype(mx.float32)`), and cast back to bfloat16 when saving. Never silently store a bf16 model as float32 (2x the size) or a float32 model as float16 (overflow risk: see [Numerical Stability](12-numerical-stability.md)).

### Quantized checkpoints in MLX

An MLX-quantized linear layer is stored as three tensors (see [Quantization](10-quantization.md)):

```
  QuantizedLinear(4096 -> 4096, bits=4, group_size=64)
    weight   (4096, 512)  uint32     eight 4-bit values packed per uint32
    scales   (4096, 64)   one per group of 64
    biases   (4096, 64)   one per group of 64

  bits per weight = 4 + 2 x 16 / 64 = 4.5 with bf16 scales/biases
                  (5.0 if they are float32, as in a float32 model)
```

The checkpoint's `config.json` carries `"quantization": {"group_size": 64, "bits": 4}`, plus per-layer overrides when some layers use other settings (an MoE router at 8 bits, for instance: see [Mixture of Experts](18-mixture-of-experts.md)). The loader must quantize the model's layers with the same settings *before* loading the weights, or the shapes won't match.

### Loading strictly

`Module.load_weights` is strict by default, and its errors are informative (MLX 0.32.3):

```
  missing key        ValueError: Missing 1 parameters: bias.
  unexpected key     ValueError: Received 1 parameters not in model: extra.
  wrong shape        ValueError: Expected shape (4, 4) but received shape (4, 5)
                       for parameter weight
  strict=False       loads silently, leaving missing parameters at their
                       random initialization
```

Keep `strict=True`. A port that "loads fine" only with `strict=False` is usually missing weights, and the model runs on random values for those layers. Handle known differences explicitly in a `sanitize()` step instead:

- **renamed keys** (`model.layers.N` → `layers.N`, prefixes like `module.` or `language_model.`);
- **transposed layouts** (convolutions: see [Convolutions](19-convolutions.md));
- **fused or split tensors** (QKV, stacked MoE experts);
- **keys to drop**: buffers recomputed at load time (RoPE `inv_freq`), optimizer state, training-only heads;
- **tied weights**: an `lm_head` that shares the embedding matrix is often absent from the file, and the model must reuse `embed_tokens`;
- **multiple copies**: online vs EMA encoders (see [Self-Supervised Learning](24-self-supervised-world-models.md)), teacher vs student.

---

## Why It Matters for MLX

### Port failures

```
CHECKPOINT PORT FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  "[load] Unknown file format"               path without .safetensors extension
                                               (resolved HF blob): use the snapshot
                                               path or format="safetensors"
  FileNotFoundError on a shard listed in     stale model.safetensors.index.json:
    the index                                  glob the shards instead
  "Missing N parameters"                     key renaming incomplete, tied weight
                                               not reused, or wrong component loaded
  "Received N parameters not in model"       buffers / optimizer state / extra
                                               heads not dropped in sanitize()
  Shape mismatch on quantized layers         model not quantized (or quantized with
                                               other bits / group size) before loading
  Model loads with strict=False, output      missing weights left at random init
    is garbage
  bf16 conversion script crashes in NumPy    NumPy has no bfloat16: cast to float32
                                               first, or stay in MLX / torch
  Output slightly off after conversion       weights stored at lower precision than
                                               the original (fp32 -> fp16), or
                                               float16 overflow in large values
  Security warning / UnpicklingError         pickle checkpoint: weights_only=True
                                               is refusing non-tensor content
```

### Conversion checklist

1. **Prefer safetensors sources**; convert pickles once, in a trusted environment.
2. **List the source keys and shapes** from the headers, and the target model's expected keys and shapes, before writing any mapping.
3. **Write `sanitize()` explicitly**: renames, transposes, splits/merges, drops, ties.
4. **Load with `strict=True`**, then `mx.eval(model.parameters())`.
5. **Save with the original dtype** (`mx.save_safetensors` with `metadata={"format": "mlx"}`), shard large models, and write an index that matches the shards.
6. **Verify numerically**: one layer's output before and after conversion (see [Testing & Validation](../03-contributing/05-testing-validation.md)).

---

## Key Terms

| Term | Definition |
|------|------------|
| **Checkpoint** | A model's weights saved to disk (often with optimizer state), plus the config needed to rebuild the architecture. |
| **safetensors** | Weight format with a JSON header (name, dtype, shape, byte offsets) and raw tensor bytes; cannot execute code; memory-mappable. |
| **Pickle checkpoint (.bin / .pt)** | PyTorch's Python-pickle format; can run arbitrary code when loaded unless `weights_only=True`. |
| **Sharding** | Splitting a checkpoint into several files; `model.safetensors.index.json` maps tensor names to shards. |
| **Weight map** | The name → shard mapping in a sharded checkpoint's index; can be stale. |
| **Tied weights** | Two layers sharing one tensor (typically the input embedding and the output head); often stored once. |
| **sanitize()** | The conversion step that renames, transposes, splits, drops or ties source tensors to match the MLX model. |
| **Strict loading** | Refusing to load if any key is missing, unexpected, or mis-shaped (`load_weights(strict=True)`, MLX's default). |
| **GGUF** | llama.cpp's single-file format with metadata and many quantization types; readable by `mx.load`. |

---

## Sources

- safetensors documentation and format specification: [huggingface.co/docs/safetensors](https://huggingface.co/docs/safetensors/index), [github.com/huggingface/safetensors](https://github.com/huggingface/safetensors)
- MLX `mlx.core.load` (formats, lazy loading): [ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html)
- PyTorch `torch.load` and `weights_only`: [pytorch.org/docs/stable/generated/torch.load.html](https://pytorch.org/docs/stable/generated/torch.load.html)
- Hugging Face cache layout (snapshots and blobs): [huggingface.co/docs/huggingface_hub/guides/manage-cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- GGUF specification: [github.com/ggml-org/ggml/blob/master/docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- `mlx-lm` v0.32.0 `utils.py` (loading by glob, writing the index): [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- Measurements on this page: MLX 0.32.3, torch 2.14.1, safetensors 0.8.0; header, laziness and stale-index observations on `mlx-community/gemma-3-12b-it-4bit`.

---

## See Also

- [Quantization](10-quantization.md) -- what the packed weights, scales and biases mean
- [Lazy Evaluation](15-lazy-evaluation.md) -- why `mx.load` returns instantly
- [Convolutions & Patchification](19-convolutions.md) -- the weight transposes a conversion must apply
- [Self-Supervised Learning & World Models](24-self-supervised-world-models.md) -- checkpoints with several encoder copies
- [Porting Guide](../03-contributing/02-porting-guide.md) -- step 4, converting the weights
