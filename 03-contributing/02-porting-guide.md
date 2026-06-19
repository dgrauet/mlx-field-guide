# Porting Guide: CUDA to MLX

> **The practical guide for translating a PyTorch/[CUDA](../glossary.md#cuda) model to MLX.** This page is organized as a step-by-step process with real code. If you are porting LTX-Video, [Matrix](../glossary.md#matrix)-Game, or any similar CUDA-based generative model, start here.

## The Process at a Glance

```
PORTING PROCESS

  1. Get the PyTorch model running locally
     -> Understand what it does; generate reference outputs

  2. Understand the architecture
     -> Map out layers, data flow, custom ops

  3. Translate layer by layer to MLX
     -> Standard layers: mechanical substitution
     -> Custom CUDA ops: one of three strategies (see below)

  4. Convert the weights
     -> Load safetensors, remap keys, transpose where needed

  5. Verify numerical agreement
     -> Compare layer outputs between PyTorch and MLX
     -> np.allclose with tolerance

  6. Optimize
     -> mx.eval placement, mx.compile, memory management
```

Working through these steps in order matters. Trying to convert weights before you have a working model architecture is a common source of confusion -- you cannot tell whether a mismatch is a weight mapping problem or an architecture problem.

---

## Step 1: Get the PyTorch Model Running

Before writing any MLX code, run the original model and capture reference outputs. These are your ground truth for verification.

```python
import torch
import numpy as np

# Load the PyTorch model (example: a simple transformer block)
model = YourModel.from_pretrained("path/to/weights")
model.eval()

# Generate a fixed random input for reproducibility
torch.manual_seed(42)
x = torch.randn(1, 64, 768)  # batch=1, seq=64, d_model=768

# Run forward pass and capture intermediate outputs
with torch.no_grad():
    # Hook to capture intermediate tensors
    intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            intermediates[name] = output.detach().cpu().numpy()
        return hook

    # Register hooks on key layers
    model.attention.register_forward_hook(make_hook("attention_out"))
    model.mlp.register_forward_hook(make_hook("mlp_out"))

    out = model(x)
    intermediates["final_out"] = out.detach().cpu().numpy()

# Save reference outputs for later comparison
np.save("reference_outputs.npy", intermediates)
print("Reference output shape:", out.shape)
print("Reference output sample:", out[0, 0, :5].numpy())
```

**Why this matters:** MLX and PyTorch may produce slightly different outputs for identical weights due to floating-point operation order, different kernel implementations, and precision differences. Capturing reference outputs before you start tells you what "correct" looks like and gives you an unambiguous comparison target.

---

## Step 2: Understand the Architecture

Read the source model's code before writing MLX code. Look for:

1. **[Layer](../glossary.md#layer) types used.** Are there standard layers (Linear, LayerNorm, MultiheadAttention) or custom CUDA extensions?

2. **[Tensor](../glossary.md#tensor) layouts.** Does the model use NCHW (channels-first) or NHWC (channels-last) for convolutions? Does it reshape between them? What are the actual tensor shapes at each stage?

3. **Custom operations.** Look for:
   - `torch.ops.custom.*` calls
   - Imports from `.so` files (compiled CUDA extensions)
   - Functions decorated with `@torch.jit.script`
   - `flash_attn` imports ([Flash Attention](../glossary.md#flash-attention) custom kernels)
   - `rotary_emb` or positional embedding custom kernels

4. **Data types.** Does the model require bfloat16 explicitly? Are there mixed-precision sections?

```python
# Quick audit: find custom ops in a model
import inspect

def find_custom_ops(model):
    """Walk a model's source files looking for non-standard torch ops."""
    custom_ops = []
    for name, module in model.named_modules():
        src = inspect.getsource(type(module))
        for line in src.splitlines():
            if any(marker in line for marker in [
                'torch.ops.',
                '.so',
                'flash_attn',
                'triton',
                'cuda_extension',
                'torch.utils.cpp_extension'
            ]):
                custom_ops.append((name, line.strip()))
    return custom_ops

custom_ops = find_custom_ops(model)
for name, line in custom_ops:
    print(f"[{name}] {line}")
```

Custom ops are the hard part of a port. Standard layers translate mechanically. Custom CUDA kernels require a decision: decompose, rewrite in [Metal](../glossary.md#metal), or accept a performance cost. This decision should happen at Step 2, not midway through the port.

---

## Step 3: Translate Layer by Layer

### Layer Mapping Reference

Most standard PyTorch layers have direct MLX equivalents:

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.nn.Linear(in, out, bias=True)` | `mlx.nn.Linear(in, out, bias=True)` | Direct equivalent |
| `torch.nn.Embedding(vocab, dim)` | `mlx.nn.Embedding(vocab, dim)` | Direct equivalent |
| `torch.nn.LayerNorm(dim, eps)` | `mlx.nn.LayerNorm(dim, eps)` | Direct equivalent |
| `torch.nn.RMSNorm(dim, eps)` | `mlx.nn.RMSNorm(dim, eps)` | Available in MLX |
| `torch.nn.Conv1d(in, out, k)` | `mlx.nn.Conv1d(in, out, k)` | See layout note below |
| `torch.nn.Conv2d(in, out, k)` | `mlx.nn.Conv2d(in, out, k)` | See layout note below |
| `torch.nn.Conv3d(in, out, k)` | Not yet in mlx.nn | Custom or decompose |
| `torch.nn.MultiheadAttention` | `mlx.nn.MultiHeadAttention` | Similar API |
| `torch.nn.GELU()` | `mlx.nn.GELU()` | Direct equivalent |
| `torch.nn.SiLU()` | `mlx.nn.SiLU()` | Direct equivalent |
| `torch.nn.Dropout(p)` | `mlx.nn.Dropout(p)` | Only active in training |
| `torch.nn.BatchNorm2d(dim)` | `mlx.nn.BatchNorm(dim)` | Different API; check dims |
| `torch.nn.GroupNorm(groups, dim)` | `mlx.nn.GroupNorm(groups, dim)` | Direct equivalent |
| `F.softmax(x, dim=-1)` | `mx.softmax(x, axis=-1)` | `axis` not `dim` |
| `F.relu(x)` | `mx.maximum(x, 0)` or `mlx.nn.relu(x)` | Both work |
| `torch.sigmoid(x)` | `mx.sigmoid(x)` | Direct equivalent |
| `torch.tanh(x)` | `mx.tanh(x)` | Direct equivalent |
| `x.view(shape)` | `x.reshape(shape)` | MLX uses `reshape` |
| `x.permute(dims)` | `x.transpose(dims)` | MLX uses `transpose` |
| `torch.cat([a, b], dim=1)` | `mx.concatenate([a, b], axis=1)` | `axis` not `dim` |
| `torch.stack([a, b], dim=0)` | `mx.stack([a, b], axis=0)` | `axis` not `dim` |
| `torch.einsum(eq, a, b)` | `mx.einsum(eq, a, b)` | Direct equivalent |
| `x.float()` | `x.astype(mx.float32)` | Explicit dtype |
| `x.half()` | `x.astype(mx.float16)` | Explicit dtype |
| `x.bfloat16()` | `x.astype(mx.bfloat16)` | 16-bit, ~float32 range |

### Building the MLX Module

The mechanical translation of a module looks like this:

```python
# PYTORCH original
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model, eps=eps)
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Pre-norm attention (residual)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm FFN (residual)
        normed = self.norm2(x)
        ffn_out = self.fc2(F.gelu(self.fc1(normed)))
        x = x + ffn_out
        return x
```

```python
# MLX translation
import mlx.core as mx
import mlx.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        self.attn = nn.MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=eps)
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def __call__(self, x):
        # __call__ instead of forward
        normed = self.norm1(x)
        # MLX attention takes (queries, keys, values) positionally
        attn_out = self.attn(normed, normed, normed)
        x = x + attn_out

        normed = self.norm2(x)
        ffn_out = self.fc2(nn.gelu(self.fc1(normed)))
        x = x + ffn_out
        return x
```

Key differences in the translation:
- `forward` becomes `__call__`
- `nn.MultiheadAttention` -> `nn.MultiHeadAttention` (note capitalization)
- MLX attention drops the second return value (attention weights) by default
- No `F.` prefix -- activation functions are in `nn.` or `mx.`
- No device management (`.cuda()`, `.to(device)`)

### The Convolution Layout Problem

**This is the most common gotcha in video and image model porting.**

PyTorch uses channels-first (NCHW) by default for 2D convolutions:
- N = batch, C = channels, H = height, W = width
- A 4-channel 8x8 image in PyTorch: shape `(1, 4, 8, 8)`

MLX uses channels-last (NHWC):
- N = batch, H = height, W = width, C = channels
- The same image in MLX: shape `(1, 8, 8, 4)`

When porting a model with convolutions, you must transpose:

```python
import torch
import mlx.core as mx
import numpy as np

# PyTorch tensor (NCHW)
x_torch = torch.randn(1, 3, 64, 64)  # (N, C, H, W)

# Convert to MLX (NHWC)
x_np = x_torch.permute(0, 2, 3, 1).numpy()  # -> (N, H, W, C)
x_mlx = mx.array(x_np)

# Similarly, Conv2d weights:
# PyTorch: (out_channels, in_channels, kH, kW)
# MLX:     (out_channels, kH, kW, in_channels)

def convert_conv2d_weight(torch_weight: torch.Tensor) -> mx.array:
    """Convert PyTorch Conv2d weight (OIHW) to MLX (OHWI)."""
    # (out, in, kH, kW) -> (out, kH, kW, in)
    w = torch_weight.permute(0, 2, 3, 1).numpy()
    return mx.array(w)

# After the MLX model runs, convert output back to NCHW for comparison:
out_mlx = model(x_mlx)             # output is (N, H', W', C')
mx.eval(out_mlx)
out_np = np.array(out_mlx)
out_torch_layout = np.transpose(out_np, (0, 3, 1, 2))  # (N, C, H', W')
```

For 3D convolutions (video models like LTX-Video and Matrix-Game), the layout difference extends to the temporal dimension:
- PyTorch 3D: NCTHW (batch, channels, time, height, width)
- MLX does not have `nn.Conv3d` -- you need to either implement it manually or decompose into 2D convolutions over slices

### Handling Missing Operations

When MLX does not have an operation, you have three options in order of effort:

**Option 1: Decompose into MLX primitives (try this first)**

Most custom CUDA kernels are performance optimizations of composable operations. Decomposing them into MLX primitives lets MLX's lazy evaluation and kernel fusion recover most of the performance:

```python
# Custom CUDA kernel: fused rotary embedding
# PyTorch (calling a custom CUDA extension):
q, k = apply_rotary_emb_cuda(q, k, cos, sin, position_ids)

# MLX: implement from primitives
# MLX will fuse these automatically
def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """
    x: (batch, seq, n_heads, head_dim)
    cos, sin: (seq, head_dim)
    """
    # Split into two halves along head_dim
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    # Rotate: [-x2, x1] * [sin, sin] + [x1, x2] * [cos, cos]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos[None, :, None, :] + rotated * sin[None, :, None, :]
```

```python
# Custom CUDA kernel: fused add + RMSNorm
# PyTorch:
out = fused_add_norm_cuda(x, residual, weight, eps=1e-5)

# MLX: decompose; kernel fusion handles performance
def fused_add_rms_norm(x: mx.array, residual: mx.array,
                        weight: mx.array, eps: float = 1e-5) -> mx.array:
    x = x + residual
    variance = mx.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x * mx.rsqrt(variance + eps)
    return weight * x_norm
```

**Option 2: Write a Metal kernel**

For operations that cannot be efficiently expressed as composed primitives -- operations with complex memory access patterns, reduction operations with specific tiling requirements, or custom attention variants -- you can write a Metal Shading Language kernel and register it with MLX.

This requires C++ and MSL. The mechanism is MLX's custom primitive system:

```cpp
// In a .cpp file compiled as a Python extension:
// (simplified example -- real Metal kernel writing has more structure)

#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>

// 1. Define the Metal kernel in a .metal file
// kernel void my_op(device float* A [[buffer(0)]],
//                   device float* B [[buffer(1)]],
//                   device float* C [[buffer(2)]],
//                   uint index [[thread_position_in_grid]]) {
//     C[index] = custom_function(A[index], B[index]);
// }

// 2. Define the MLX primitive
class MyCustomOp : public mlx::core::Primitive {
public:
    void eval_cpu(const std::vector<mlx::core::array>& inputs,
                  mlx::core::array& out) override {
        // CPU fallback (required)
    }
    void eval_gpu(const std::vector<mlx::core::array>& inputs,
                  mlx::core::array& out) override {
        // Metal kernel dispatch
    }
};

// 3. Expose to Python via pybind11
// Then call from Python:
// import your_extension
// out = your_extension.my_op(a, b)
```

See `mlx-forge` ([github.com/ml-explore/mlx-forge](https://github.com/ml-explore/mlx-forge)) for working examples of this pattern.

**Option 3: Accept the Python fallback**

For operations that are rare in the forward pass or not on the critical path, a pure Python implementation may be fast enough. Profile before investing in Metal kernel work:

```python
import mlx.core as mx
import time

def benchmark(fn, *args, n=20, warmup=5):
    # warmup: allow Metal to compile shaders and warm caches
    for _ in range(warmup):
        # mx.eval forces the lazy graph to execute on the GPU
        mx.eval(fn(*args))
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args)
        # Force GPU execution before stopping the clock
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000
    print(f"  avg: {avg_ms:.2f} ms")
    return avg_ms

x = mx.random.normal((16, 128, 512))
# Compare: does the slow Python version matter in context?
print("Custom op:")
benchmark(my_custom_op_python, x)
print("In full forward pass:")
benchmark(full_model, x)
```

If the custom op is 5ms but the full forward pass is 2000ms, optimizing the custom op is not the priority.

---

## Step 4: Convert the Weights

[Weight](../glossary.md#weight) conversion is the part that breaks most ports. The architecture and the weights must agree on:
- Key naming (the dictionary key for each weight tensor)
- Tensor shape (including transpositions)
- Data type

### Loading PyTorch Weights

```python
from safetensors import safe_open
import numpy as np
import mlx.core as mx

def load_safetensors(path: str) -> dict[str, mx.array]:
    """Load a safetensors file into MLX arrays."""
    arrays = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            arrays[key] = mx.array(f.get_tensor(key))
    return arrays

# Or load all keys at once:
from safetensors.numpy import load_file

def load_pytorch_weights(path: str) -> dict[str, mx.array]:
    weights = load_file(path)  # returns dict[str, np.ndarray]
    return {k: mx.array(v) for k, v in weights.items()}
```

### Remapping Weight Keys

PyTorch model weight keys often don't match MLX model attribute paths. You need to remap:

```python
import mlx.core as mx
from safetensors import safe_open

def remap_keys(weights: dict, key_map: dict) -> dict:
    """Apply a key renaming map to a weight dictionary."""
    return {key_map.get(k, k): v for k, v in weights.items()}

# Example: LLaMA-style key remapping
# PyTorch key              -> MLX key
key_map = {
    "model.embed_tokens.weight":                  "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight":     "model.layers.0.attention.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight":     "model.layers.0.attention.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight":     "model.layers.0.attention.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight":     "model.layers.0.attention.out_proj.weight",
    "model.layers.0.mlp.gate_proj.weight":        "model.layers.0.feed_forward.w1.weight",
    "model.layers.0.mlp.down_proj.weight":        "model.layers.0.feed_forward.w2.weight",
    "model.layers.0.mlp.up_proj.weight":          "model.layers.0.feed_forward.w3.weight",
    "model.layers.0.input_layernorm.weight":      "model.layers.0.attention_norm.weight",
    "model.layers.0.post_attention_layernorm.weight": "model.layers.0.ffn_norm.weight",
}

# For models with many layers, generate the map programmatically:
def build_key_map(n_layers: int) -> dict:
    key_map = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "lm_head.weight": "lm_head.weight",
        "model.norm.weight": "model.norm.weight",
    }
    for i in range(n_layers):
        pt = f"model.layers.{i}"
        mlx_prefix = f"model.layers.{i}"
        key_map.update({
            f"{pt}.self_attn.q_proj.weight": f"{mlx_prefix}.attention.q_proj.weight",
            f"{pt}.self_attn.k_proj.weight": f"{mlx_prefix}.attention.k_proj.weight",
            f"{pt}.self_attn.v_proj.weight": f"{mlx_prefix}.attention.v_proj.weight",
            f"{pt}.self_attn.o_proj.weight": f"{mlx_prefix}.attention.out_proj.weight",
            f"{pt}.mlp.gate_proj.weight":    f"{mlx_prefix}.feed_forward.w1.weight",
            f"{pt}.mlp.down_proj.weight":    f"{mlx_prefix}.feed_forward.w2.weight",
            f"{pt}.mlp.up_proj.weight":      f"{mlx_prefix}.feed_forward.w3.weight",
            f"{pt}.input_layernorm.weight":  f"{mlx_prefix}.attention_norm.weight",
            f"{pt}.post_attention_layernorm.weight": f"{mlx_prefix}.ffn_norm.weight",
        })
    return key_map
```

### Transposing Weights

`nn.Linear` in PyTorch stores weights as `(out_features, in_features)`. MLX's `nn.Linear` stores them in the same shape. **However**, some models split QKV into a single tensor, or store weights in transpose form for custom matmul kernels. When in doubt:

```python
# Debug weight shape mismatches:
# Load PyTorch weights and MLX model, then compare:

pt_weights = load_safetensors("model.safetensors")
mlx_model = MyModel(config)

# Get the MLX model's expected shapes:
mlx_weights = dict(mlx_model.parameters())

for key in mlx_weights:
    pt_key = key_map.get(key, key)
    if pt_key in pt_weights:
        pt_shape = pt_weights[pt_key].shape
        mlx_shape = mlx_weights[key].shape
        if pt_shape != mlx_shape:
            print(f"SHAPE MISMATCH: {key}")
            print(f"  PyTorch: {pt_shape}")
            print(f"  MLX:     {mlx_shape}")
            # Common fix: transpose
            if pt_shape == mlx_shape[::-1]:
                print(f"  -> Needs transpose")
```

### Loading Weights Into the Model

```python
import mlx.core as mx
import mlx.nn as nn

def load_model(model: nn.Module, weights_path: str, key_map: dict) -> nn.Module:
    """Load and remap weights from safetensors into an MLX model."""
    # Load raw weights
    raw_weights = load_pytorch_weights(weights_path)

    # Remap keys
    remapped = {}
    for k, v in raw_weights.items():
        new_key = key_map.get(k)
        if new_key is not None:
            remapped[new_key] = v
        else:
            # Log unmapped keys; they may be unused (e.g., position_ids buffer)
            print(f"Unmapped key: {k}")

    # Update model parameters
    model.load_weights(list(remapped.items()))

    # Force all parameters to materialize on the GPU
    mx.eval(model.parameters())

    return model
```

### Handling Fused QKV Weights

Some models store Q, K, V as a single concatenated tensor. Others split them. You need to match what the MLX model expects:

```python
def split_fused_qkv(fused_weight: mx.array, n_heads: int, head_dim: int) -> tuple:
    """Split a (3*n_heads*head_dim, d_model) QKV weight into Q, K, V."""
    d = n_heads * head_dim
    q = fused_weight[:d]   # (d_model, d_model)
    k = fused_weight[d:2*d]
    v = fused_weight[2*d:]
    return q, k, v

def fuse_qkv(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Fuse separate Q, K, V weights into a single tensor."""
    return mx.concatenate([q, k, v], axis=0)
```

---

## Step 5: Verify Numerical Agreement

Never assume the port is correct. Verify by comparing outputs at each stage.

### Layer-by-Layer Comparison

```python
import torch
import mlx.core as mx
import numpy as np

def compare_tensors(name: str,
                    pt_tensor: torch.Tensor,
                    mlx_tensor: mx.array,
                    atol: float = 1e-3,
                    rtol: float = 1e-3) -> bool:
    """Compare a PyTorch tensor and MLX array. Print diff stats."""
    pt_np = pt_tensor.detach().cpu().float().numpy()
    # Force GPU execution before converting to numpy
    mx.eval(mlx_tensor)
    mx_np = np.array(mlx_tensor.astype(mx.float32))

    abs_diff = np.abs(pt_np - mx_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    close = np.allclose(pt_np, mx_np, atol=atol, rtol=rtol)

    status = "PASS" if close else "FAIL"
    print(f"[{status}] {name}")
    print(f"       max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    if not close:
        # Find where the biggest differences are
        worst_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        print(f"       worst at {worst_idx}: pt={pt_np[worst_idx]:.6f} mlx={mx_np[worst_idx]:.6f}")
    return close

# Example: compare a forward pass
x_np = np.random.RandomState(42).randn(1, 64, 768).astype(np.float32)
x_pt = torch.tensor(x_np)
x_mlx = mx.array(x_np)

with torch.no_grad():
    out_pt = pytorch_model(x_pt)

out_mlx = mlx_model(x_mlx)
# Force GPU execution before extracting numpy data
mx.eval(out_mlx)

compare_tensors("full_output", out_pt, out_mlx)
```

### Acceptable Tolerance Ranges

Not all differences indicate a bug. Floating-point arithmetic is not associative -- different operation orderings produce different results.

```
TYPICAL TOLERANCE RANGES

  bfloat16 vs bfloat16 (same algorithm, different implementation):
    max_diff < 1e-2  -> usually fine
    max_diff > 1e-1  -> investigate

  float32 vs float32 (same algorithm):
    max_diff < 1e-5  -> fine
    max_diff > 1e-3  -> investigate

  float16 vs bfloat16 (different precision):
    max_diff < 1e-2  -> acceptable (inherent precision difference)

  After quantization (4-bit):
    max_diff < 0.1 per element -> generally acceptable
    Qualitative output comparison is the real test
```

### Debugging Numerical Mismatches

If `compare_tensors` fails, find the first layer that disagrees:

```python
def trace_forward(pt_model, mlx_model, x_np: np.ndarray) -> None:
    """Run both models and compare outputs at each module."""
    x_pt = torch.tensor(x_np)
    x_mlx = mx.array(x_np)

    pt_intermediates = {}
    mlx_intermediates = {}

    # PyTorch: hook every layer
    def make_pt_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                pt_intermediates[name] = out.detach().cpu().numpy()
        return hook

    for name, module in pt_model.named_modules():
        module.register_forward_hook(make_pt_hook(name))

    with torch.no_grad():
        pt_model(x_pt)

    # MLX has no forward hooks. Instrument your model to record activations:
    # pass a dict into __call__ and have each sub-module write into it, e.g.
    #   def __call__(self, x, trace=None):
    #       x = self.attn(x)
    #       if trace is not None: trace["attn"] = x
    #       ...
    # This is a sketch -- you add the `trace` plumbing to your own model.
    mlx_model(x_mlx, trace=mlx_intermediates)
    for k, v in mlx_intermediates.items():
        mx.eval(v)

    # Compare shared layer names
    for name in sorted(pt_intermediates.keys()):
        if name in mlx_intermediates:
            compare_tensors(name,
                torch.tensor(pt_intermediates[name]),
                mlx_intermediates[name])
```

Common causes of mismatches and their fixes:
- **Weight not transposed:** the shape matches accidentally but values are wrong -- check key mapping and any needed `mx.transpose` calls
- **Wrong axis:** PyTorch `dim` vs MLX `axis` parameter; double-check calls to `softmax`, `concat`, `mean`, etc.
- **NCHW vs NHWC:** the output shape is transposed compared to the input -- add the layout conversion
- **[Bias](../glossary.md#bias) vs no bias:** `nn.Linear(bias=False)` vs `nn.Linear(bias=True)` -- the weight shape is the same but values differ
- **[Normalization](../glossary.md#normalization) stats:** BatchNorm has running statistics (`running_mean`, `running_var`) in addition to learned parameters; these must be loaded too

---

## Step 6: Performance Optimization

A correct port is the goal. A fast port is the next step.

### mx.eval Placement

MLX builds a computation graph lazily and only executes it when `mx.eval` is called. Every `mx.eval` call is a synchronization point -- the [CPU](../glossary.md#cpu) waits for the [GPU](../glossary.md#gpu) to finish. The rule: **fewer `mx.eval` calls = more work batched = better GPU utilization.**

```python
# BAD: too many synchronization points
for layer in model.layers:
    x = layer(x)
    # Forcing execution after each layer serializes CPU scheduling with GPU work
    mx.eval(x)

# GOOD: let the full forward pass accumulate, then execute at once
for layer in model.layers:
    x = layer(x)
# One synchronization point; GPU runs the full graph continuously
mx.eval(x)
```

The exception: memory-limited workloads. If a model has very large intermediate tensors, forcing execution between stages allows earlier tensors to be freed:

```python
# Memory-constrained forward pass (video models with large temporal dims)
# Stage 1: process temporal attention
x = temporal_attention(x)
mx.eval(x)  # free the attention matrices before spatial layers

# Stage 2: process spatial layers
x = spatial_layers(x)
mx.eval(x)
```

### mx.compile

`mx.compile` traces a function and compiles it into a more efficient execution plan. It is most effective on functions called many times with the same shapes:

```python
import mlx.core as mx

# Compile the forward pass for repeated inference. The function captures
# the model; mx.compile traces the call and reuses the compiled plan.
@mx.compile
def compiled_forward(x):
    return model(x)

# First call: compilation overhead (~0.1-1s)
out = compiled_forward(x)
mx.eval(out)

# Subsequent calls: uses compiled plan (faster)
for i in range(1000):
    out = compiled_forward(x)
    mx.eval(out)
```

`mx.compile` works best on:
- Fixed input shapes (recompiles if shapes change)
- Pure functions (no side effects, no Python control flow that depends on values)
- Functions called repeatedly (the compilation overhead must amortize)

### Memory Management

MLX's lazy evaluation means intermediate tensors accumulate until execution is forced. For large models:

```python
import mlx.core as mx

# Monitor memory during development
mx.reset_peak_memory()

x = run_large_forward_pass(model, input)
mx.eval(x)

peak_mb = mx.get_peak_memory() / 1e6
print(f"Peak memory: {peak_mb:.0f} MB")

# Strategy: reduce intermediate tensor size
# Instead of computing full attention for all layers then running FFN:
# Run layer by layer with forced execution between stages if memory-constrained.
```

### dtype Considerations

MLX *arrays* default to `float32`, but for inference you usually cast the model to `bfloat16`: it halves memory and is accurate enough for most generation. Note that `nn.Module` exposes `set_dtype` (which mutates the module in place and returns `None`) -- there is no `Module.astype`:

```python
import mlx.core as mx
import mlx.nn as nn

model = MyModel(config)               # parameters are float32 by default

# Reference forward in float32 first, BEFORE casting the model in place
x_f32 = mx.random.normal((1, 64, 768))
out_f32 = model(x_f32)
mx.eval(out_f32)

# Cast all parameters to bfloat16 (in place), then run the bf16 forward
model.set_dtype(mx.bfloat16)
x_bf16 = x_f32.astype(mx.bfloat16)
out_bf16 = model(x_bf16)
mx.eval(out_bf16)

# Verify the bfloat16 forward matches float32 to tolerance
max_diff = mx.abs(out_f32 - out_bf16.astype(mx.float32)).max().item()
print(f"float32 vs bfloat16 max diff: {max_diff:.5f}")
```

---

## Common Port Failure Modes

The generic symptoms table (wrong shape, all zeros, slow) covers the obvious. The failures below are the ones that waste days. Each is from real port experience on diffusion, LLM, and 3D-generation models.

### 1. QKV interleaving mismatch in attention

**Symptom.** Layer parity on every layer *except* attention. Attention output diverges by orders of magnitude even though the weights loaded cleanly and the shapes match.

**Cause.** PyTorch models pack the combined QKV projection in different layouts across codebases. Three patterns are common:

```
Layout A (interleaved per head):  [q0, k0, v0, q1, k1, v1, ..., qH, kH, vH]
Layout B (grouped):               [q0, q1, ..., qH, k0, k1, ..., kH, v0, v1, ..., vH]
Layout C (HuggingFace GPT-NeoX):  [q_head0_dim0, k_head0_dim0, v_head0_dim0, q_head0_dim1, ...]
```

If your MLX port assumes one layout and the weights are in another, the Q, K, and V tensors you compute with look numerically plausible (no NaNs, reasonable magnitudes) but are actually scrambled across heads.

**Diagnosis.** Read the reference PyTorch source for the *exact reshape pattern* applied to the `qkv` weight. Look for lines like:

```python
# Layout A (interleaved per head)
qkv = qkv.view(batch, seq, n_heads, 3, head_dim).transpose(...)
# vs Layout B (grouped)
qkv = qkv.view(batch, seq, 3, n_heads, head_dim).transpose(...)
```

The dimension order in `.view(...)` tells you the layout. Do not assume; read the code.

**Fix.** Match the source layout exactly in your MLX implementation. If you need the other layout for a downstream op, do the re-interleaving *after* projection, not by changing how you slice the weight.

**See also.** [Attention](../01-foundations/06-attention.md) -- the Q/K/V projection math.

### 2. Silent default divergence

**Symptom.** Port looks correct, no errors, outputs look plausible -- but they don't match the reference. No layer shows a large divergence; everything is slightly off.

**Cause.** Your module's constructor has a default argument that differs from the source model's config. Classic examples:
- `eps=1e-5` in your `LayerNorm` when the source uses `eps=1e-6`.
- `bias=True` in your `Linear` when the source model has `bias=False`.
- `rope_theta=10000` when the source uses `rope_theta=500000` (LLaMA 3).
- `activation="gelu"` when the source uses `"silu"`.

These defaults are usually invisible in a diff because the config file uses the source model's real values, and your port just... doesn't read them.

**Diagnosis.** Print the effective config of your ported module and cross-check every field against the source model's `config.json`:

```python
print("Port config:", {k: getattr(my_model.config, k) for k in vars(my_model.config)})
print("Source config:", json.load(open("source_config.json")))
# Diff them manually. Every discrepancy is suspicious.
```

**Fix.** Never rely on constructor defaults for a port. Always thread every config field through from the source `config.json` to the MLX module. Treat "the default happens to match" as coincidence, not correctness.

**Why this is worth a dedicated section.** On two separate ports (Hunyuan3D-2.1 and ERNIE-Image), this one class of bug cost more debugging time than any other single issue. It is also the easiest to miss on review -- the port "looks right."

### 3. The `attention_head_dim` naming trap in `UNet2DConditionModel`

**Symptom.** Port of a Stable Diffusion 1.x or 2.x UNet produces near-identical weight shapes but wrong attention output — textures come out neutral, cyan, or checkerboard after full generation. Weights load cleanly. Layer-level parity in isolation looks correct; only full-pipeline generation reveals the bug.

**Cause.** In `diffusers.UNet2DConditionModel`, the config field `attention_head_dim` is **misnamed**: it is actually the number of heads per block, not the per-head dimension. The real per-head dimension is `channels // attention_head_dim`.

Concrete example from SD 2.1's paint UNet:

```python
# UNet2DConditionModel config
{
    "block_out_channels": [320, 640, 1280, 1280],
    "attention_head_dim": [5, 10, 20, 20]  # THESE ARE HEAD COUNTS, NOT PER-HEAD DIMS
}

# Actual geometry, per block:
# Block 0: 5 heads  of dim 64   (channels 320 // heads 5   = 64)
# Block 1: 10 heads of dim 64   (channels 640 // heads 10  = 64)
# Block 2: 20 heads of dim 64   (channels 1280 // heads 20 = 64)
# Block 3: 20 heads of dim 64
```

The reason this loads silently: the Q/K/V/O Linear weights have shape `(channels, channels)` regardless of how you partition them into heads. Only the reshape-for-attention step `x.reshape(B, L, heads, head_dim)` exposes the error, and only after the softmax scaling produces a different distribution.

**Diagnosis.** When porting any `UNet2DConditionModel`, treat `attention_head_dim` as `num_heads`. Do not divide. Verify by comparing your MLX module's effective head count against the reference:

```python
from diffusers import UNet2DConditionModel
ref = UNet2DConditionModel.from_config(cfg)
print("heads per block:", [b.attentions[0].transformer_blocks[0].attn1.heads for b in ref.down_blocks if hasattr(b, "attentions")])
# Compare against your port's configured head count per block.
```

**Fix.** In your MLX port, set `num_heads = config.attention_head_dim[i]` directly for each block, then derive `head_dim = block_out_channels[i] // num_heads`. In the Hunyuan3D-2.1 paint UNet port, fixing this inversion reduced a 0.21 max_abs per-UNet-pass error to 1e-5 against PyTorch.

**Scope note.** This trap is specific to `UNet2DConditionModel` (SD 1.x / 2.x UNets, plus derivatives like the Hunyuan3D paint UNet). DiT-based models that use `diffusers.models.transformers.Transformer2DModel` — Stable Diffusion 3, Flux, many ControlNets-on-DiT — follow the natural naming convention: `inner_dim = num_attention_heads * attention_head_dim`, where `attention_head_dim` really is the per-head dimension. When porting from a DiT, do not apply this fix — you would invert a correct config.

**See also.** [Attention](../01-foundations/06-attention.md).

### 4. Layer parity does not imply end-to-end correctness

**Symptom.** Every layer matches PyTorch to ~3e-6 in isolation. You declare the port done. First real generation produces a checkerboard pattern, cyan tint, gray slush, or garbage tokens.

**Cause.** Small per-layer errors compound across the denoising chain (or generation loop). 3e-6 per layer, amplified over 50 diffusion steps and thousands of operations, becomes visible artifacts. Also: some bugs (wrong normalization scale, off-by-one positional encoding) do not show up in a single-step forward comparison but dominate after iteration.

**Diagnosis.** Run a three-test noise-chain diagnostic before claiming a port is done:

1. **Single-step parity.** Each layer matches PyTorch within 1e-5 (fp32) or 1e-2 (bf16). Table stakes.
2. **Full-forward parity.** One full forward pass from noise to denoised latent matches PyTorch end-to-end within 1e-4 (fp32).
3. **Multi-step parity.** Run 10 denoising steps; intermediate latents at steps 1, 5, 10 all match PyTorch within 1e-3. If step 1 matches but step 5 diverges, there is an iterative-error bug.

Only after all three pass, run a full image/token generation and inspect.

**Fix.** Varies by bug. Common culprits: missing forced-execution placement (lazy-eval accumulates precision issues), wrong scheduler sigma schedule, positional encoding drift across steps.

**See also.** [Diffusion](../01-foundations/08-diffusion.md) -- why multi-step error matters for generative models specifically.

### 5. Tekken-family tokenizer BOS not auto-prepended

**Symptom.** Text encoder output diverges from HuggingFace `transformers` by ~100x starting at layer 2. Layer 1 matches; layer 2 onward drifts heavily.

**Cause.** Pixtral / Mistral-family / Tekken-V3 tokenizers do not automatically prepend the `<s>` BOS token when called with `add_special_tokens=True`. HuggingFace `transformers` handles this internally via a different code path; tokenizers-only callers (including ports that call the raw tokenizer) do not.

If your port tokenizes the prompt without manually prepending BOS, token position 0 is wrong, every token's positional encoding shifts by one, and the embedding layer produces a subtly different output that amplifies through the transformer.

**Diagnosis.** Compare the tokenized prompt IDs byte-for-byte between your port and HuggingFace `transformers`:

```python
from transformers import AutoTokenizer
hf_ids = AutoTokenizer.from_pretrained("mistralai/...").encode("hello")
port_ids = my_tokenizer.encode("hello")
assert hf_ids == port_ids, f"HF: {hf_ids}, port: {port_ids}"
```

If token 0 differs, BOS is the bug.

**Fix.** In your port's `_tokenize` / equivalent, explicitly prepend the BOS token ID:

```python
def _tokenize(self, text: str) -> list[int]:
    ids = self._raw_tokenize(text, add_special_tokens=True)
    if self.needs_manual_bos:  # Tekken-family
        ids = [self.bos_token_id] + ids
    return ids
```

**See also.** [Tokenization](../01-foundations/09-tokenization.md).

### 6. pip editable install shadowed by site-packages

**Symptom.** You edit a file in a locally-checked-out library (installed via `pip install -e .`). Print statements don't appear. Fixes don't take effect. `__file__` of the imported module correctly points at your local checkout, yet the behavior is the old behavior.

**Cause.** A stale non-editable install of the same package is present in site-packages and takes precedence over the editable install, even though `__file__` reports the local path. This happens when a package was installed two ways in the same environment (typically: `pip install <pkg>` followed later by `pip install -e .` without uninstalling).

**Diagnosis.** First command to run when a fix does not appear to take effect:

```bash
pip show <package-name>
```

Look at the `Location:` field. If it points to `site-packages/` instead of your local checkout, the editable install is being shadowed.

**Fix.**

```bash
pip uninstall <package-name> -y
pip install -e /path/to/local/checkout
```

Re-run `pip show` to confirm the `Location:` now points at your local path.

**Why this is in a porting guide.** Most MLX porting work involves modifying a local fork of an upstream library. This failure mode is environmental, not code, but it masquerades as a code bug and wastes more time than any other single non-code issue.

---

## Quick-Reference Symptom Table

The sections above cover the deep failures. For fast lookup of simpler issues:

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ValueError: wrong shape` on weight load | Key mapping correct but tensor needs transposing | Add `.T` or `mx.transpose` to the weight before loading |
| Output is all zeros or NaN from layer 1 | Weight not loaded; model using random init | Check that `load_weights` was called and returned no errors |
| Output diverges after N layers | Accumulating FP error; mismatched norm eps | Compare layer-by-layer; check LayerNorm eps values |
| Correct results in float32, NaN in bfloat16 | Operation numerically unstable at lower precision | Add explicit float32 cast around the unstable op |
| Model runs slower than expected | Too many forced execution calls serializing GPU work | Consolidate them; see forced-execution placement section |
| Memory grows unbounded during inference | Intermediate arrays not freed; too-large graph | Force execution at stage boundaries; check Python list accumulation |
| Conv2d produces wrong results | NCHW/NHWC layout mismatch | Transpose inputs before conv and outputs after |
| Attention scores overflow | QK^T values too large before softmax | Check scaling by `1/sqrt(head_dim)` is present |
| Custom CUDA extension import fails | Expected: on MLX you cannot import CUDA extensions | Replace with MLX primitives or Metal kernel |

---

## Sources

- MLX documentation -- core API: [ml-explore.github.io/mlx/](https://ml-explore.github.io/mlx/)
- MLX documentation -- Python API reference: [ml-explore.github.io/mlx/build/html/python/index.html](https://ml-explore.github.io/mlx/build/html/python/index.html)
- mlx-examples (reference implementations): [github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)
- mlx-forge (porting tools and Metal kernel examples): [github.com/ml-explore/mlx-forge](https://github.com/ml-explore/mlx-forge)
- HuggingFace safetensors documentation: [huggingface.co/docs/safetensors](https://huggingface.co/docs/safetensors)
- Apple Metal Shading Language specification: [developer.apple.com/metal/Metal-Shading-Language-Specification.pdf](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

---

## See Also

- [Where to Contribute](01-where-to-contribute.md) -- repositories and communities; where to submit your port when it is working
- [Development Environment](04-dev-environment.md) -- the two-stack workspace setup this guide assumes you already have running
- [Testing & Validation](05-testing-validation.md) -- the next step after porting: proving the result is numerically correct and locking it in with regression tests
- [Open Opportunities](03-open-opportunities.md) -- specific models and gaps ranked by impact; what to port if you don't already have a target
- [Frameworks](../02-ecosystem/02-frameworks.md) -- PyTorch vs MLX API comparison; the source material for the layer mapping table above
- [Tooling](../02-ecosystem/08-tooling.md) -- profiling and debugging tools for MLX; what to use when the port is slow or producing unexpected outputs
- [GPU Computing](../01-foundations/04-gpu-computing.md) -- the hardware model behind lazy evaluation, kernel fusion, and why execution-point placement matters
