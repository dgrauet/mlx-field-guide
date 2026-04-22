# Tensors

> **One-liner:** Tensors are multi-dimensional arrays of numbers -- the universal container for all data in machine learning.

## The Intuition

Before you can understand any ML framework, you need to understand the one data structure that underlies everything: the tensor. The good news is that you already understand tensors. You've been working with them for years under different names.

Start with the simplest case and build up:

**A scalar** is a single number. Nothing more.

```
22.5
```

That's the temperature outside. One number, zero dimensions.

**A vector** is a list of numbers arranged in a line.

```
[ 255, 128, 0 ]
```

That's an RGB color -- red, green, blue. Three numbers, one dimension. A vector has a length (here: 3).

**A matrix** is a grid of numbers -- rows and columns.

```
 120  95  80
  45  60  72
 200 185 170
```

That's a tiny 3x3 grayscale image where each number is a pixel brightness (0=black, 255=white). Two dimensions: rows and columns. A matrix has a shape like (3, 3).

**A tensor** is the generalization of all of the above to any number of dimensions.

```
number  ->  list  ->  spreadsheet  ->  stack of spreadsheets  ->  ...
scalar     vector       matrix              3D tensor
```

Visually:

```
Scalar (0D)     Vector (1D)     Matrix (2D)       3D Tensor
                                ┌───────────┐    ┌───────────┐
                                │ 1  2  3  │   /│ 1  2  3  │
   42           [1, 2, 3]       │ 4  5  6  │  / │ 4  5  6  │
                                │ 7  8  9  │ /  │ 7  8  9  │
                                └───────────┘   └───────────┘
                                               /│ ...        │
                                              / └────────────┘
```

A color image is a 3D tensor: height x width x 3 (the 3 channels are red, green, blue).

A batch of color images (how a model actually processes images) is a 4D tensor: batch_size x height x width x 3. Stack 32 images together to feed them through a network at once, and you have a (32, 256, 256, 3) tensor.

A color video goes one dimension further: frames x height x width x 3. That's a 4D tensor where the first axis is time.

There is no conceptual upper limit. Models for video generation routinely work with 5D tensors. The pattern is always the same -- add another dimension to represent another "axis" of variation in your data.


## How It Actually Works

### Shape

Every tensor has a **shape**: an ordered tuple of integers describing how many elements exist along each dimension.

```
Scalar:           shape = ()            -- 0 dimensions, 1 element
Vector of 3:      shape = (3,)          -- 1 dimension, 3 elements
3x4 matrix:       shape = (3, 4)        -- 2 dimensions, 12 elements
RGB image:        shape = (256, 256, 3) -- 3 dimensions, 196,608 elements
Batch of images:  shape = (32, 3, 256, 256) -- 4 dimensions, 6,291,456 elements
```

The total number of elements is the product of all the shape values. A (32, 3, 256, 256) tensor holds 32 * 3 * 256 * 256 = 6,291,456 numbers.

[Shape](../glossary.md#shape) is the single most important property to keep track of when working with ML code. When something goes wrong, checking shapes is almost always the first debugging step.

Here is a labeled 3D tensor to make the axes concrete:

```
         axis 2 (width) -->
        ┌─────────────────────┐
axis 1  │  [0,0]  [0,1]  [0,2] │
(height)│  [1,0]  [1,1]  [1,2] │   <-- this is one 2D slice (one "channel")
  |     └─────────────────────┘
  v          /                      another slice behind it
        ┌───/─────────────────┐
        │  [0,0]  [0,1]  [0,2] │
        │  [1,0]  [1,1]  [1,2] │   axis 0 (depth/channel) goes "into the screen"
        └─────────────────────┘
```

Each axis has a name and a meaning that depends on context. There is nothing intrinsically special about axis 0 versus axis 1 -- the meaning is assigned by the code that creates and uses the tensor.

### Dtype

Every tensor stores numbers in a specific numeric format, called the **dtype** (data type). The dtype determines:

- How much memory each element occupies
- What range of values can be represented
- How much precision is available

Common dtypes in ML:

| [Dtype](../glossary.md#dtype)    | Bits | Bytes | Range (approx)             | Use case                           |
|----------|------|-------|----------------------------|------------------------------------|
| float32  | 32   | 4     | ±3.4 × 10^38, ~7 sig figs  | Default for training and inference |
| float16  | 16   | 2     | ±65,504, ~3 sig figs        | Faster inference, less memory      |
| bfloat16 | 16   | 2     | ±3.4 × 10^38, ~2 sig figs  | [Training](../glossary.md#training) on modern hardware        |
| int8     | 8    | 1     | -128 to 127                | Quantized models (see [Quantization](../glossary.md#quantization))|
| int32    | 32   | 4     | ±2.1 × 10^9                | Indices, token IDs                 |
| bool     | 1    | 1*    | True / False               | Masks                              |

*booleans are usually stored as bytes in practice

The tradeoff is always precision vs. memory. A float32 model uses twice the memory of a float16 model. Quantized models (int8 or lower) can be 4-8x smaller than their float32 counterparts, at some cost to output quality.

When porting a [CUDA](../glossary.md#cuda) model to MLX, dtype mismatches are a common source of silent bugs: the computation runs but produces slightly wrong numbers because precision was quietly lost somewhere.

### Operations

Tensors support a rich vocabulary of operations. The most important categories:

**Element-wise operations** apply a function to each element independently. When two tensors have the same shape, you can add, subtract, multiply, or divide them element by element:

```
A = [[1, 2],    B = [[5, 6],    A + B = [[6,  8],
     [3, 4]]         [7, 8]]             [10, 12]]
```

**Reductions** collapse one or more dimensions by combining elements:

```
A = [[1, 2, 3],    sum(axis=0) = [5, 7, 9]   -- sum each column
     [4, 5, 6]]    sum(axis=1) = [6, 15]      -- sum each row
                   sum()       = 21            -- sum everything
```

**[Matrix](../glossary.md#matrix) multiplication** (`matmul` or `@`) is the operation at the heart of every neural network layer. It combines two matrices according to the rules of linear algebra. Shape rule: `(m, k) @ (k, n) -> (m, n)`. The inner dimensions must match.

**Reshaping** reinterprets the same data under a different shape, without changing the underlying values:

```
A = [1, 2, 3, 4, 5, 6]    shape (6,)
A.reshape(2, 3) = [[1, 2, 3],    shape (2, 3)
                   [4, 5, 6]]
A.reshape(3, 2) = [[1, 2],       shape (3, 2)
                   [3, 4],
                   [5, 6]]
```

**Transposing** swaps axes:

```
A = [[1, 2, 3],    A.T = [[1, 4],
     [4, 5, 6]]           [2, 5],
                           [3, 6]]
```

### Broadcasting

**[Broadcasting](../glossary.md#broadcasting)** is a rule for automatically expanding tensors with mismatched shapes so that element-wise operations work. Instead of raising an error when shapes don't match, the framework "broadcasts" the smaller tensor to match the larger one.

```
A = [[1, 2, 3],    B = [10, 20, 30]   -- B has shape (3,), A has shape (2, 3)
     [4, 5, 6]]

A + B = [[11, 22, 33],   -- B is "broadcast" across each row of A
         [14, 25, 36]]
```

Broadcasting follows rules: dimensions are aligned from the right, and a dimension of size 1 (or a missing dimension) can be stretched to match the other tensor. No data is actually copied -- the framework just pretends the data repeats.

Broadcasting is powerful but easy to misuse. A shape mismatch that happens to be "compatible" under broadcasting rules will silently produce wrong results instead of an error.

### Memory Layout

Tensors are stored as flat, contiguous blocks of memory. A 2x3 matrix stored in **row-major** order (also called C order, used by NumPy and MLX):

```
Memory:  [ 1, 2, 3, 4, 5, 6 ]
                  ^
         [1, 2, 3]  [4, 5, 6]
         row 0      row 1
```

The elements of each row are adjacent in memory. Moving to the next row means jumping by (number of columns) elements.

**Strides** are how the framework tracks this layout. The stride for each axis tells you how many elements to skip in raw memory to advance by one step along that axis. For a (2, 3) float32 tensor in row-major order:

```
strides = (3, 1)   -- advance 3 elements to move one row, 1 element to move one column
```

Strides matter because some operations (like transpose) don't move data in memory -- they just change the strides. A tensor is **contiguous** when its strides match what you'd expect from its shape stored in the standard order. Some operations require contiguous tensors, which is why you sometimes need to call `.contiguous()` or equivalent.


## Why It Matters for MLX

### mx.array is your tensor

In MLX, the tensor type is `mx.array`. It plays the same role as `torch.Tensor` in PyTorch and `np.ndarray` in NumPy. If you're reading ported code or writing new MLX code, `mx.array` is everywhere.

```python
import mlx.core as mx

# Create a tensor from a list
x = mx.array([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]])

print(x.shape)   # (2, 3)
print(x.dtype)   # mlx.core.float32
```

### Shape mismatches are the #1 source of porting bugs

When you port a model from PyTorch to MLX, the logic is usually the same -- you're just translating API calls. But the shapes have to match exactly, and different frameworks have different conventions.

A common pitfall: PyTorch uses `(batch, channels, height, width)` (NCHW) by default, while some MLX operations expect `(batch, height, width, channels)` (NHWC). Get this wrong and you'll get either an error (if you're lucky) or silently scrambled outputs (if you're not).

A (32, 3, 256, 256) tensor and a (32, 256, 256, 3) tensor contain the exact same number of elements, so reshape won't catch the mistake. You have to know which axis is which.

Print shapes aggressively while porting:

```python
def forward(self, x):
    print(f"input: {x.shape}")
    x = self.conv(x)
    print(f"after conv: {x.shape}")
    return x
```

### Lazy evaluation: operations don't run until you force them

MLX uses **lazy evaluation**. When you write:

```python
y = x * 2 + 1
```

MLX does not immediately compute `y`. It builds a computation graph describing what needs to happen, but defers the actual work. This allows MLX to optimize the computation before executing it.

To force the computation to run, call `mx.eval()`:

```python
mx.eval(y)   # force execution of the pending computation
print(y)     # safe to read the result
```

This is different from PyTorch (which is eager by default) and more like JAX. If you're debugging and your print statement shows an unevaluated array, that's why.

### Real example: LTX video tensor shapes

In the LTX-Video pipeline, video frames are packed into a tensor before being passed to the model. A typical input shape looks like:

```
(batch, frames, channels, height, width)
 = (1, 25, 3, 512, 768)
```

Each dimension carries meaning:

```
batch    = 1      -- one video at a time
frames   = 25     -- 25 frames of video (~1 second at 25fps)
channels = 3      -- RGB
height   = 512    -- 512 pixels tall
width    = 768    -- 768 pixels wide
```

Total elements: 1 * 25 * 3 * 512 * 768 = 29,491,200 float32 values -- roughly 113 MB of memory for a single input tensor.

When this tensor passes through convolutional layers, gets downsampled by the VAE encoder, and gets rearranged for the attention layers, its shape changes at every step. Following those transformations is the core skill of reading and porting video model code.


## Key Terms

| Term | Definition |
|------|------------|
| **[Tensor](../glossary.md#tensor)** | A multi-dimensional array of numbers; the generalization of scalars, vectors, and matrices to any number of dimensions |
| **[Scalar](../glossary.md#scalar)** | A single number; a 0-dimensional tensor |
| **[Vector](../glossary.md#vector)** | A 1-dimensional tensor; a list of numbers |
| **Matrix** | A 2-dimensional tensor; a grid of numbers arranged in rows and columns |
| **Shape** | An ordered tuple of integers describing the size of each dimension; e.g., `(32, 3, 256, 256)` |
| **Dtype** | The numeric format used to store each element; determines precision and memory cost (e.g., float32, float16, int8) |
| **Broadcasting** | A set of rules for automatically expanding tensors with compatible but mismatched shapes during element-wise operations |
| **[Axis / Dimension](../glossary.md#axis-dimension)** | A single "direction" in a tensor; a tensor's rank is the number of axes it has |
| **[Contiguous](../glossary.md#contiguous)** | A tensor whose elements are stored in a single unbroken block of memory in the expected order; required by some operations |
| **[Stride](../glossary.md#stride)** | The number of elements to skip in memory to move one step along a given axis; how the framework maps multi-dimensional indices to flat memory addresses |


## Sources

- [MLX Documentation -- mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html) -- the authoritative reference for `mx.array`, its properties, and its methods
- [MLX Quickstart](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html) -- short introduction to MLX array creation and operations
- [NumPy: Understanding array shapes and strides](https://numpy.org/doc/stable/reference/arrays.ndarray.html) -- MLX follows NumPy conventions closely; the NumPy docs are a useful complement
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. **Chapter 2: Linear Algebra** -- rigorous treatment of scalars, vectors, matrices, and tensors as mathematical objects, and the operations defined on them. Available free at [deeplearningbook.org](https://www.deeplearningbook.org/contents/linear_algebra.html)


## See Also

- [Neural Networks](02-neural-networks.md) -- tensors flow through layers; weights are tensors; outputs are tensors. Once you know what a tensor is, you can understand what a neural network actually does with them.
- [GPU Computing](04-gpu-computing.md) -- why GPUs are so effective at tensor operations, and what that means for the frameworks built on top of them.
