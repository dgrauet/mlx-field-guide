# Quantization

> **One-liner:** [Quantization](../glossary.md#quantization) reduces a model's memory footprint and speeds up inference by representing weights with fewer bits (e.g., 4-bit instead of 16-bit), with minimal quality loss.

## The Intuition

Consider a single model weight: `0.0234375`. In memory, that number is stored as a float32 -- 32 bits, 4 bytes. The model has billions of weights. A 7-billion-parameter model stores 7,000,000,000 such numbers.

The question quantization asks: do we actually need that precision?

Think about what a weight represents -- a learned tendency, a direction of influence, a learned feature detector. The difference between `0.0234375` and `0.023` is real, but does it meaningfully change what the model does? For many weights, the answer is no. The exact value matters far less than the approximate region it occupies.

**Quantization trades precision for efficiency.** Instead of storing `0.0234375` as a 32-bit float, you round it to the nearest value representable in fewer bits. Fewer bits per weight means:

- The model weighs less in memory
- More of the model fits in fast cache rather than slow DRAM
- Moving weights from memory to compute units is faster
- For integer formats, the arithmetic itself can be faster

```
THE PRECISION TRADEOFF

  Weight value:  0.0234375

  float32 (32 bits):  0.0234375          -- exact, 4 bytes
  float16 (16 bits):  0.02344            -- very close, 2 bytes
  bfloat16 (16 bits): 0.02344            -- very close, 2 bytes
  int8    ( 8 bits):  0.023              -- approximate, 1 byte
  int4    ( 4 bits):  0.02               -- rough, 0.5 bytes

  Memory for 7B weights:
  float32:  7B * 4 bytes = 28 GB   -- won't fit on most Macs
  float16:  7B * 2 bytes = 14 GB   -- needs 24 GB+ Mac
  int8:     7B * 1 byte  =  7 GB   -- fits M2 Pro (16 GB)
  int4:     7B * 0.5 byte=  3.5 GB -- fits any modern Mac
```

The best analogy is JPEG compression for images. A JPEG at 80% quality discards fine detail that most viewers cannot perceive, and the file is 10x smaller. The image still looks like the image. Push too hard -- JPEG at 5% -- and artifacts appear, details smear, quality degrades visibly. Quantization follows the same curve: moderate compression is nearly lossless; extreme compression hurts.

The practical boundary is around 4-bit quantization for most modern LLMs. Below that, quality drops noticeably. At 4-bit, the tradeoff is generally acceptable for inference: models produce coherent, high-quality output while using a fraction of the memory.


## How It Actually Works

### Number formats: from float32 to int4

Every number in a neural network has a **dtype** -- a data type that determines how many bits represent it and what range of values is possible.

```
NUMBER FORMAT COMPARISON

  Format    Bits  Bytes  Range (approx)        Precision     Use case
  --------- ----- ------ --------------------- ------------- -------------------------
  float32   32    4      ±3.4 × 10^38          ~7 sig. digits Training, reference
  float16   16    2      ±65,504               ~3 sig. digits Inference, training
  bfloat16  16    2      ±3.4 × 10^38          ~2 sig. digits Training (wider range)
  int8       8    1      -128 to 127           256 levels     Quantized inference
  int4       4    0.5    -8 to 7               16 levels      Aggressive quantization

  Two of these deserve extra explanation:

  float16:   Same 16-bit width as bfloat16, but allocates more bits to the
             fractional part. Higher precision for numbers near zero. Lower
             maximum value. Used heavily in CUDA inference.

  bfloat16:  "Brain float 16." Same exponent range as float32, truncated
             mantissa. Handles very large and very small values without
             overflow. Preferred for training because gradients can be huge.
             MLX defaults to bfloat16 for most operations.
```

The jump from float16 to int8 is a fundamental shift in representation. Float formats encode numbers on a logarithmic scale -- they can represent values as large as 65,000 or as small as 0.000001, with varying precision throughout. Integer formats divide a fixed range into uniform steps. A weight stored as int8 maps to one of exactly 256 possible values; int4 maps to one of 16.

### The quantization procedure

Quantizing a weight tensor requires two steps: choosing the mapping from float to integer, then applying it.

**Step 1: Compute the scale factor.**

Find the range of the weights you want to quantize -- the minimum and maximum values in that tensor. Compute a scale that maps this float range onto the integer range:

```
SIMPLE SYMMETRIC QUANTIZATION (int8 example)

  Weight tensor:   [-1.2, 0.45, -0.03, 0.89, -0.67, 1.1]
  Max abs value:   1.2

  int8 range: -127 to 127
  Scale = 1.2 / 127 = 0.009449...

  Quantize each weight:
    -1.2  / 0.009449 = -127  --> stored as -127
     0.45 / 0.009449 =   47  --> stored as   47
    -0.03 / 0.009449 =   -3  --> stored as   -3
     0.89 / 0.009449 =   94  --> stored as   94
    -0.67 / 0.009449 =  -70  --> stored as  -71  (rounding)
     1.1  / 0.009449 =  116  --> stored as  116

  Memory: 6 floats * 4 bytes = 24 bytes (original)
          6 int8   * 1 byte  =  6 bytes (quantized)
          1 float  * 4 bytes =  4 bytes (scale factor)
  Total quantized: 10 bytes -- 58% savings, one scale to store

  Dequantize to recover approximate float:
    -127 * 0.009449 = -1.200   (exact match)
      47 * 0.009449 =  0.444   (was 0.45, error: 0.006)
      -3 * 0.009449 = -0.028   (was -0.03, error: 0.002)
```

**Step 2: Store the integer values and the scale.**

At inference time, **dequantization** reconstructs the float values by multiplying the stored integer by the stored scale. This happens on the fly -- the weights live in memory as compact integers, and the scale multiplication happens in the compute unit just before the weight is used.

### Group quantization: the key accuracy improvement

Whole-tensor quantization (one scale for the entire weight matrix) works, but struggles when a tensor contains both very large and very small values -- the scale must accommodate the large values, which means the small values get quantized with very little precision.

The solution is **group quantization**: instead of one scale for the whole tensor, compute a separate scale for every small group of consecutive weights (typically 32 or 64 weights per group).

```
GROUP QUANTIZATION (group_size = 4, simplified)

  Weight tensor: [0.9, 0.85, 0.91, 0.87,  |  0.001, 0.003, -0.002, 0.002]
                  --------- Group 1 --------     -------- Group 2 ----------

  With whole-tensor scale (max = 0.91):
    Group 2 weights are tiny relative to the scale.
    0.001 / scale ≈ 0   --> quantizes to 0   (error: 100%)
    0.003 / scale ≈ 0   --> quantizes to 0   (error: 100%)

  With per-group scales:
    Group 1: max = 0.91,   scale_1 = 0.91 / 127 = 0.00717
    Group 2: max = 0.003,  scale_2 = 0.003 / 127 = 0.0000236

    Group 2 now uses its own scale -- all four values are resolved
    with full int8 precision within their own range.

  Cost: one scale per group instead of one scale per tensor.
  Typical cost: 64 int4 weights + 1 float16 scale = 32 + 16 = 48 bits
  Instead of: 64 float16 weights = 1024 bits
  Compression ratio still ~20x despite per-group scales.
```

MLX's `nn.QuantizedLinear` always uses group quantization. The default group size is 64, which balances accuracy against memory overhead well.

### PTQ vs. QAT: two approaches to quantizing a model

**Post-[Training](../glossary.md#training) Quantization (PTQ)** quantizes a model after training is complete. You take a trained float16 model, compute scales from its weights (sometimes passing a small calibration dataset through), and write out the quantized result. No retraining required. Fast and simple.

```
PTQ WORKFLOW

  float32 model weights
          |
  [analyze weight ranges]
  [optional: calibration pass on small dataset]
          |
  [compute scale factors]
  [round weights to integer grid]
          |
  quantized model weights + scale factors
  (stored on disk, loaded for inference)
```

PTQ is by far the most common approach in practice. It is what `mlx_lm.convert`, [GPTQ](../glossary.md#gptq), and AWQ all do (with varying sophistication in how they compute the optimal quantization).

**Quantization-Aware Training (QAT)** simulates quantization during training itself. The model is trained with fake quantization nodes inserted in the computation graph -- these round activations and weights to the quantized grid on the forward pass, then pass gradients through on the backward pass as if quantization never happened (the "straight-through estimator"). The model learns to be robust to the rounding that quantization introduces.

```
QAT WORKFLOW

  float32 model weights (start from pretrained or scratch)
          |
  [training loop]
    forward:  quantize weights/activations (fake quantize)
              compute loss with quantized values
    backward: pass gradients through as if no quantization
              update float32 weights
  [iterate until converged]
          |
  model that knows it will be quantized at inference time
          |
  [quantize for deployment]
```

QAT produces better accuracy at the same bit width, especially at 4-bit and below -- the model has been trained to compensate for precision loss. The cost is that you need to run training, which requires significant compute. PTQ is used when you are working with a model you cannot retrain (the typical case when porting someone else's model).

### GPTQ, AWQ, and GGUF: what the format names mean

When you download a quantized model from Hugging Face or llama.cpp, it has a format tag: `Q4_K_M`, `GPTQ-4bit`, `AWQ`, etc. Here is what each means:

**GPTQ** (Frantar et al., 2022): A PTQ algorithm that uses second-order information (approximate Hessians) to find the quantization that minimizes output error, rather than just rounding each weight independently. The key insight: some weights matter more than others for the output; GPTQ compensates errors in important weights by adjusting less important ones. Produces better accuracy than naive PTQ at 4-bit. Runs layer by layer, using a calibration dataset. Common in Hugging Face models tagged `GPTQ`.

**AWQ** (Lin et al., 2023): Activation-Weighted Quantization. The insight: not all weights in a tensor are equally important for accuracy. Weights connected to large activations matter more (they have bigger impact on the output). AWQ identifies these salient weights using activation statistics from a calibration dataset, then scales the weight distribution before quantization to protect the important values. Produces strong 4-bit accuracy. Common in Hugging Face models tagged `AWQ`.

**[GGUF](../glossary.md#gguf)**: A file format from llama.cpp (not a quantization algorithm itself). Packages model weights, tokenizer, and metadata into a single portable file. The quantization scheme is embedded in the filename: `Q4_K_M` means 4-bit, K-quant family, medium variant. K-quants are a family of mixed-precision schemes where some layers are stored at higher precision than others. GGUF models run on [CPU](../glossary.md#cpu) via llama.cpp or on Apple Silicon via llama.cpp's [Metal](../glossary.md#metal) backend and the `mlx-lm` loader.

```
QUANTIZATION FORMAT CHEAT SHEET

  Format  Algorithm          Common tags          Loader
  ------- ------------------ -------------------- -----------------------
  GPTQ    Second-order PTQ   model-GPTQ-4bit      HuggingFace transformers
  AWQ     Activation-aware   model-AWQ            HuggingFace + autoawq
  GGUF    llama.cpp format   model-Q4_K_M.gguf    llama.cpp, mlx-lm
  MLX     MLX native         mlx-community/model  mlx-lm (native)
  BitsAndBytes  On-the-fly   load_in_4bit=True    HuggingFace (CUDA only)
```

### The memory math in concrete numbers

This is the calculation that determines whether a model fits on your machine:

```
MEMORY CALCULATION

  Formula:
    memory_GB = (parameters * bits_per_weight) / (8 * 1024^3)

  7B model (e.g., LLaMA-3-8B):
    float32:   7e9 * 32 / (8 * 1024^3) = 26.1 GB
    bfloat16:  7e9 * 16 / (8 * 1024^3) = 13.0 GB
    int8:      7e9 *  8 / (8 * 1024^3) =  6.5 GB
    4-bit:     7e9 *  4 / (8 * 1024^3) =  3.3 GB

  13B model (e.g., LLaMA-3-13B):
    bfloat16:  13e9 * 16 / (8 * 1024^3) = 24.2 GB
    4-bit:     13e9 *  4 / (8 * 1024^3) =  6.1 GB

  70B model (e.g., LLaMA-3-70B):
    bfloat16:  70e9 * 16 / (8 * 1024^3) = 130 GB   -- no Mac has this
    4-bit:     70e9 *  4 / (8 * 1024^3) =  32.5 GB  -- fits M2 Ultra (192 GB)

  Note: add ~20% for KV cache and activations during inference.
  The weights are the dominant cost; activations are a smaller overhead.

  Mac memory guide (approximate):
    16 GB:  up to 7B model at 4-bit (3.3 GB weights + overhead)
    24 GB:  up to 13B at 4-bit or 7B at bfloat16
    32 GB:  up to 20B at 4-bit or 13B at bfloat16
    64 GB:  up to 40B at 4-bit or 30B at bfloat16
    96 GB:  up to 70B at 4-bit (tight)
   128 GB:  70B at 4-bit with room for context
   192 GB:  70B at bfloat16 or 405B at 4-bit
```

These numbers are why quantization is not optional when running large models on a laptop or even a high-end Mac. A 70B LLM in bfloat16 requires 130 GB of memory -- a number no current Mac can reach. At 4-bit, it requires 32.5 GB, which fits comfortably on a 64 GB Mac. Quantization is the technology that makes these models runnable at all outside a data center.


## Why It Matters for MLX

### What mlx-lm does for you

The `mlx-lm` package handles quantized model loading transparently. When you download a pre-quantized model from the `mlx-community` Hugging Face organization and load it with `mlx_lm.load`, the quantized weights are loaded directly into `nn.QuantizedLinear` layers -- no manual work required:

```python
from mlx_lm import load, generate

# mlx-community distributes pre-quantized models
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

response = generate(
    model,
    tokenizer,
    prompt="Explain unified memory in one paragraph.",
    max_tokens=200,
)
print(response)
```

To quantize a model yourself -- say, you have downloaded a float16 model and want a 4-bit version:

```python
# Convert a Hugging Face model to MLX 4-bit quantized format
# Run from the command line:
# python -m mlx_lm.convert --hf-path meta-llama/Llama-3.2-3B-Instruct \
#                          --mlx-path ./llama-3-3b-4bit \
#                          -q

# Or in Python:
from mlx_lm import convert
convert(
    hf_path="meta-llama/Llama-3.2-3B-Instruct",
    mlx_path="./llama-3-3b-4bit",
    quantize=True,
    q_bits=4,
    q_group_size=64,
)
```

### nn.QuantizedLinear: the building block

MLX implements quantized inference through `mlx.nn.QuantizedLinear`. A standard `nn.Linear` layer stores weights as float16 or bfloat16; a `QuantizedLinear` stores them as int4 (packed) plus per-group float16 scale factors. During the forward pass, the layer dequantizes the weights on the fly and performs the matrix multiplication.

```
QUANTIZEDLINEAR INTERNALS

  Standard nn.Linear:
    weight: shape [out_features, in_features], dtype bfloat16
    Memory: out_features * in_features * 2 bytes

  nn.QuantizedLinear (4-bit, group_size=64):
    weight: shape [out_features, in_features // 2], dtype uint8
            (two 4-bit values packed per byte)
    scales: shape [out_features, in_features // 64], dtype float16
            (one scale per group of 64 input weights)
    biases: shape [out_features], dtype float16  (optional)

  Forward pass:
    [packed int4 weights]
          |
    [dequantize: unpack + multiply by scale]
          |
    [bfloat16 weight matrix, reconstructed]
          |
    [standard matmul: input @ weight.T + bias]
          |
    [output]
```

The dequantization is fast -- a multiply-and-unpack that happens in the [GPU](../glossary.md#gpu) compute units, not a separate memory operation. The key benefit is that the weights travel from unified memory to the GPU as 4-bit integers (half the bytes of float16), which reduces memory bandwidth pressure. Since LLM inference is bandwidth-bound (the weights are larger than the activations, and new weights must be loaded for every token), reducing weight size directly reduces latency.

### Scheme matching: the critical constraint

When you port a quantized model to MLX, you must match the quantization scheme exactly. The model's saved integer weights are only meaningful relative to the scale factors and the scheme used to produce them. Using the wrong scheme produces garbage output with no error message.

The parameters that must match:

```
QUANTIZATION SCHEME PARAMETERS

  Parameter      Typical values          What happens if wrong
  -------------- ----------------------- ---------------------
  bits           4, 8                    Completely wrong weights
  group_size     32, 64, 128             Wrong scale mapping -> garbage
  symmetric      True / False            Sign flipped or offset by constant
  dtype (scales) float16 / bfloat16      Precision of scale factors

  Check these in the model's config.json or quantization_config.json:

  {
    "bits": 4,
    "group_size": 64,
    "quantize_embeddings": false,
    "linear_class_name": "QuantizedLinear"
  }

  When loading with mlx_lm.load(), the config is read automatically.
  When porting a GPTQ or AWQ model manually, you must read the config
  and instantiate nn.QuantizedLinear with matching parameters.
```

A common porting bug: loading a GPTQ model (group_size=128) with MLX's default (group_size=64). The weights load without error but produce incoherent output. Always inspect the quantization config before writing loading code.

### Mixed precision: not all layers are equal

Not every layer in a model is quantized to the same bit width. Certain layers are more sensitive to quantization error:

- **[Embedding](../glossary.md#embedding) layers**: the first and last layers of an LLM. They map token IDs to vectors. Quantizing them aggressively loses vocabulary detail. Many models keep embeddings in float16.
- **LM head**: the final projection from hidden state to vocabulary logits. Closely related to embeddings; often kept at float16.
- **Small layers**: normalization layers (RMSNorm, LayerNorm) have very few parameters. Quantizing them saves almost no memory but can hurt accuracy significantly. Usually kept at float16.

```
TYPICAL MIXED-PRECISION LAYOUT (7B LLM)

  Layer                    Parameters     Quantized?   Dtype
  ------------------------ -------------- ------------ ---------
  Token embedding          vocab * d_model  Usually no  float16
  Attention QKV projections large          Yes          int4
  Attention output proj.   large          Yes          int4
  FFN gate / up / down     very large     Yes          int4
  RMSNorm (all)            d_model each   No           float16
  LM head                  vocab * d_model Usually no  float16

  The bulk of parameters are in the linear layers (QKV, FFN).
  These are what 4-bit quantization targets.
  The un-quantized layers are small enough not to matter for memory.
```

When inspecting an MLX model:

```python
import mlx.nn as nn
from mlx_lm import load

model, _ = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Count quantized vs. unquantized linear layers
quantized = []
unquantized = []

for name, module in model.named_modules():
    if isinstance(module, nn.QuantizedLinear):
        quantized.append(name)
    elif isinstance(module, nn.Linear):
        unquantized.append(name)

print(f"Quantized layers: {len(quantized)}")
print(f"Unquantized layers: {len(unquantized)}")
```

### The 70B on MacBook Pro example

A concrete end-to-end example of why this matters:

```
70B LLM ON 96 GB MACBOOK PRO M2 ULTRA

  Model: LLaMA-3-70B (70 billion parameters)

  Without quantization (bfloat16):
    Weights: 70e9 * 2 bytes = 140 GB
    Result: does not fit. Cannot run.

  With 4-bit quantization:
    Weights: 70e9 * 0.5 bytes = 35 GB
    KV cache (8K context): ~4 GB
    Activations: ~1 GB
    Total: ~40 GB

    96 GB Mac: 40 GB used, 56 GB free. Runs.

  Performance:
    Tokens per second: ~4-8 tok/s (bandwidth-bound at 70B)
    Quality: noticeably good; 4-bit is well within acceptable range
    Use case: capable of real work, not just demos

  The same model on a 192 GB Mac:
    Even at 4-bit: 40 GB weights, can run much longer contexts
    At 8-bit: 70 GB weights, still fits, higher quality
```

This is not a hypothetical. The `mlx-community` organization maintains 4-bit quantized versions of 70B models that run on Apple Silicon with `mlx-lm`. The quantization is what makes this possible -- full stop.


## Key Terms

| Term | Definition |
|------|------------|
| **Quantization** | The process of representing model weights (and sometimes activations) with fewer bits than their original training format, trading precision for reduced memory and faster inference. |
| **FP32 (float32)** | 32-bit floating point. The standard precision for training. 4 bytes per value. Full range and precision, but expensive in memory. |
| **FP16 (float16)** | 16-bit floating point. 2 bytes per value. Reduced range (max ~65,000); higher precision near zero than bfloat16. Common in [CUDA](../glossary.md#cuda) inference. |
| **BF16 (bfloat16)** | "Brain float 16." 16-bit floating point with the same exponent range as float32 but reduced mantissa precision. 2 bytes per value. MLX's default dtype; preferred for training because it handles large gradients without overflow. |
| **[INT8](../glossary.md#int8)** | 8-bit integer. 1 byte per value, 256 possible levels. Requires a scale factor to map the integer range to the float range. Common for inference quantization. |
| **[INT4](../glossary.md#int4)** | 4-bit integer. 0.5 bytes per value, 16 possible levels. Standard for aggressive LLM quantization; 4-bit is the practical lower limit before quality degrades significantly. |
| **PTQ (Post-Training Quantization)** | Quantizing a model after training is complete. No retraining required. The dominant approach for model distribution and inference optimization. |
| **QAT (Quantization-Aware Training)** | Simulating quantization during training so the model learns to compensate for precision loss. Produces better accuracy than PTQ at the same bit width, but requires a training run. |
| **GPTQ** | A PTQ algorithm that uses second-order (Hessian) information to minimize output error when quantizing, layer by layer. Better accuracy than naive rounding at 4-bit. |
| **AWQ (Activation-Weighted Quantization)** | A PTQ algorithm that identifies salient weights using activation statistics and scales the weight distribution before quantization to protect them. Strong 4-bit accuracy. |
| **GGUF** | A file format from llama.cpp that packages model weights, tokenizer, and metadata into a single portable file. Not a quantization algorithm; embeds the quantization scheme in the filename (e.g., Q4_K_M). |
| **[Group Size](../glossary.md#group-size)** | The number of consecutive weights that share a single scale factor in group quantization. Smaller group size = more scales = better accuracy = slightly more memory overhead. Common values: 32, 64, 128. |
| **[Scale Factor](../glossary.md#scale-factor)** | The float value stored alongside quantized weights that maps the integer range back to the approximate float range. One scale per group in group quantization. |
| **[Dequantization](../glossary.md#dequantization)** | The operation of converting quantized integer weights back to approximate float values at inference time, by multiplying the stored integer by the stored scale factor. |
| **[Mixed Precision](../glossary.md#mixed-precision)** | Using different bit widths for different layers in the same model. Typically: attention and FFN layers at 4-bit; embedding, normalization, and output layers at float16. |


## Sources

- Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*. Introduces the second-order PTQ algorithm behind GPTQ. Available at [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- Lin, J., Tang, J., Tang, H., Yang, S., Chen, W., Wang, W., & Han, S. (2023). "AWQ: Activation-aware [Weight](../glossary.md#weight) Quantization for LLM Compression and Acceleration." *MLSys 2024*. Introduces AWQ's salient weight identification technique. Available at [arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- MLX Documentation. "Quantization" and `mlx.nn.QuantizedLinear` reference. Covers MLX's group quantization implementation, `nn.QuantizedLinear`, and `mlx_lm.convert`. Available at [ml-explore.github.io/mlx/build/html/python/nn.html](https://ml-explore.github.io/mlx/build/html/python/nn.html)
- Gerganov, G. et al. llama.cpp and the GGUF format. The reference implementation for GGUF quantization, including the K-quant family (Q4_K_M, Q5_K_S, etc.). Available at [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)


## See Also

- [Tensors](01-tensors.md) -- prerequisite; dtypes (float32, float16, bfloat16) are tensor properties; quantization changes the dtype of stored weights
- [Training vs. Inference](03-training-vs-inference.md) -- quantization is primarily an inference optimization; PTQ runs after training; QAT bridges both phases
- [GPU Computing](04-gpu-computing.md) -- memory bandwidth is the bottleneck that quantization addresses; unified memory on Apple Silicon makes quantized weights available to both CPU and GPU without copying
