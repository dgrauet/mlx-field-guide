# Neural Networks

> **One-liner:** A neural network is a chain of simple math operations that learns to transform inputs into useful outputs by adjusting its internal numbers.

## The Intuition

You've read the Tensors page, so you know that ML models work entirely with tensors -- multi-dimensional arrays of numbers. The question is: what do you do with those numbers to make a model that can recognize an image, generate text, or synthesize video?

The answer is a neural network: a sequence of simple mathematical transformations, stacked one after another.

### The assembly line analogy

Picture a factory that turns raw metal into finished car parts.

```
Raw material --> [Station 1] --> [Station 2] --> [Station 3] --> Finished part
                 (shape it)    (smooth it)    (coat it)
```

Each station takes the material in, applies one specific transformation, and passes the result to the next station. The stations happen in a fixed order. The output quality depends entirely on how each station is configured.

A neural network is that factory. The "raw material" is your input tensor (say, pixel values from an image). Each "station" is a **layer** -- a mathematical transformation. The "finished part" is the output tensor (say, a probability distribution over categories like "cat", "dog", "car").

Now add one more detail: each station has a set of **knobs** called weights. The knobs control exactly how that station transforms the material. The key insight of machine learning is that you can automatically find the right knob settings by repeatedly showing the factory examples of inputs and correct outputs, then adjusting the knobs based on how wrong the output was.

That adjustment process is **training**.

### The "neural" part

The name comes from a loose analogy to biological neurons. A neuron in your brain receives signals from many other neurons, combines them, and if the combined signal is strong enough, fires a signal to the next neurons.

An artificial neuron does something similar in math:

1. Receive a list of numbers (inputs)
2. Multiply each by a weight (how much to "trust" each input)
3. Sum everything up
4. Apply a function that determines whether to "fire" (activation function)

The "neural" analogy is weak and should not be over-read. No one believes large language models work like brains. The term stuck because it sounded good in the 1950s. What matters is the math, and the math is straightforward: matrix multiplications followed by simple non-linear functions, chained together many times.


## How It Actually Works

### A single layer

The core operation in a neural network layer is:

```
output = activation_function(inputs @ weights + bias)
```

Breaking that down with shapes:

```
inputs:   tensor of shape (batch_size, in_features)
weights:  tensor of shape (in_features, out_features)
bias:     tensor of shape (out_features,)

inputs @ weights  -->  shape (batch_size, out_features)   [matrix multiply]
+ bias            -->  shape (batch_size, out_features)   [broadcast add]
activation(...)   -->  shape (batch_size, out_features)   [element-wise]
```

The weights and bias are the "knobs" -- the numbers that get learned during training. The activation function is fixed.

A concrete example with small numbers:

```
inputs  = [0.5, 1.2, -0.3]        shape (3,)   -- e.g., 3 input features

weights = [[ 0.8,  0.1],          shape (3, 2) -- maps 3 inputs to 2 outputs
           [-0.4,  0.6],
           [ 0.2, -0.9]]

bias    = [0.1, -0.2]              shape (2,)

step 1: inputs @ weights
        = [0.5*0.8 + 1.2*(-0.4) + (-0.3)*0.2,   0.5*0.1 + 1.2*0.6 + (-0.3)*(-0.9)]
        = [0.4 - 0.48 - 0.06,                    0.05 + 0.72 + 0.27]
        = [-0.14,                                 1.04]

step 2: + bias
        = [-0.14 + 0.1, 1.04 + (-0.2)]
        = [-0.04, 0.84]

step 3: ReLU activation
        = [0.0, 0.84]   -- ReLU turns -0.04 into 0.0, leaves 0.84 alone
```

Three input numbers went in, two output numbers came out. This is a **Linear layer** (also called Dense or Fully Connected), the simplest and most fundamental layer type.

### Activation functions

Without the activation function, something important breaks. Here is why.

If you stack two Linear layers with no activation:

```
output = (inputs @ W1 + b1) @ W2 + b2
       = inputs @ W1 @ W2 + b1 @ W2 + b2
       = inputs @ W_combined + b_combined
```

Two layers with no activation collapse into one layer. You could add 100 Linear layers with no activations and the whole stack would behave identically to a single Linear layer. No matter how many layers you add, the model can only represent linear functions -- straight lines in any number of dimensions.

Real-world data is not linear. The activation function breaks the linearity at each layer, allowing the network to learn curved, complex patterns.

The most common activation functions you will see:

**ReLU (Rectified Linear Unit)** -- the simplest:

```
ReLU(x) = max(0, x)

  output
    |          /
    |         /
    |        /
  0 |_______/___
              input
```

Negative values become zero. Positive values pass through unchanged. That's it. Brutally simple, widely effective.

**GELU (Gaussian Error Linear Unit)** -- smoother version of ReLU:

```
GELU(x) ≈ x * sigmoid(1.702 * x)

  output
    |            /
    |           /
    |          /
  0 |_______.-/___
         small bump near -0.1
              input
```

Unlike ReLU, GELU doesn't have a hard corner at zero -- it has a smooth curve. Transformers (GPT, BERT, and many video models) use GELU in their feed-forward layers.

**SiLU / Swish** -- another smooth activation:

```
SiLU(x) = x * sigmoid(x)
```

Visually similar to GELU. Used in many modern architectures including the LTX-Video model's feed-forward blocks.

**Softmax** -- converts a vector of arbitrary numbers into a probability distribution:

```
softmax([1.2, 0.5, -0.8]) = [0.64, 0.31, 0.05]   -- sums to 1.0
```

Used in the final layer of classifiers and inside attention mechanisms.

### Common layer types

Linear layers are just the start. Real models use a variety of layer types, each designed for a specific kind of transformation.

**Linear (also called Dense or Fully Connected):**

Every input is connected to every output via the weight matrix. [Shape](../glossary.md#shape) transform: `(batch, in_features) -> (batch, out_features)`.

```python
# PyTorch
nn.Linear(768, 3072)

# MLX (same concept, same name)
nn.Linear(768, 3072)
```

**Conv2d (2D [Convolution](../glossary.md#convolution)):**

A sliding window that scans across a 2D spatial input (like an image). Instead of connecting every input to every output, a small kernel (e.g., 3x3) slides across the input, applying the same weights at every position. This makes Convolutions efficient for images and naturally captures local patterns (edges, textures).

```
Input image (5x5):          3x3 kernel:
┌─────────────────┐         ┌─────────┐
│  1  2  3  4  5  │         │ 1  0 -1 │
│  6  7  8  9 10  │         │ 1  0 -1 │
│ 11 12 13 14 15  │    *    │ 1  0 -1 │
│ 16 17 18 19 20  │         └─────────┘
│ 21 22 23 24 25  │
└─────────────────┘
        |
        v
   Output (3x3): each value is a sum of kernel * patch
```

The kernel slides across every valid position, computing a dot product at each step.

**LayerNorm ([Layer](../glossary.md#layer) [Normalization](../glossary.md#normalization)):**

Rescales the values within a layer to have approximately zero mean and unit variance. Without it, values can drift wildly as they pass through many layers -- either exploding toward infinity or collapsing toward zero. LayerNorm stabilizes training and inference.

```
Given x = [10.0, 200.0, -5.0, 0.3]:

mean = (10.0 + 200.0 + (-5.0) + 0.3) / 4 = 51.33
std  = standard deviation ≈ 84.7

normalized = [(x - mean) / std for x in values]
           ≈ [-0.49, 1.76, -0.66, -0.61]
```

**RMSNorm (Root Mean Square Normalization):**

A simpler variant of LayerNorm that only normalizes by the root-mean-square magnitude without subtracting the mean. Used in LLaMA and many recent transformer architectures. Slightly cheaper to compute than LayerNorm.

**[Embedding](../glossary.md#embedding):**

A lookup table that converts integer indices into dense vectors. Used everywhere: to turn token IDs into vectors in language models, to turn pixel positions into position vectors, to encode any categorical input.

```
Vocabulary of 50,000 tokens, embedding dimension 768:

embedding table shape = (50000, 768)

token_id = 5423
lookup:  embedding_table[5423]  --> a vector of shape (768,)
```

The embedding table is itself a learned weight tensor -- during training, the model learns which vector to associate with each token.

### Stacking layers: a complete network

Put the pieces together. A simple network for classifying images into 10 categories:

```
Input
  |
  |  shape: (batch, 784)   [e.g., 28x28 pixel image flattened to a vector]
  v
[Linear(784 -> 256)]
  |
  v
[ReLU]
  |
  |  shape: (batch, 256)
  v
[Linear(256 -> 128)]
  |
  v
[ReLU]
  |
  |  shape: (batch, 128)
  v
[Linear(128 -> 10)]
  |
  v
[Softmax]
  |
  |  shape: (batch, 10)   [probability for each of 10 classes]
  v
Output
```

The input tensor flows from top to bottom, each layer transforming its shape and values. This top-to-bottom pass through all layers is called the **forward pass**.

### Parameters vs. hyperparameters

These two terms are often confused.

**Parameters** are the numbers inside the model that get learned during training:
- The weights in every Linear layer
- The biases in every Linear layer
- The embedding table values
- The learned scale/shift values in LayerNorm

Parameters start as random numbers and get gradually adjusted by training to produce useful outputs. When someone says "a 7 billion parameter model", they mean the total count of all these learned numbers across the entire network.

**Hyperparameters** are choices made by the developer before training begins:
- Learning rate (how big of a step to take when adjusting weights)
- Number of layers
- Width of each layer (how many features)
- Which activation function to use
- [Batch](../glossary.md#batch) size

Hyperparameters are not learned from data -- they are design decisions baked into the model architecture. Choosing good hyperparameters is part of the art of ML research.

### What "model" means

The word **model** has a precise meaning: a specific architecture (arrangement of layers and their hyperparameters) paired with a specific set of trained parameter values.

The architecture defines the shape. The parameters fill it with learned knowledge.

This is why porting a model involves two things:
1. Recreating the architecture in the target framework (rewrite the layer definitions)
2. Loading the trained parameters into those layers (load the weight files)

If you only do step 1, you have the right structure but random weights -- the model will produce garbage outputs. You need both.


## Why It Matters for MLX

### mlx.nn contains all the layer types

MLX's neural network module lives at `mlx.nn`. It provides implementations of all the common layer types:

```python
import mlx.nn as nn

# Linear layer: maps 768 inputs to 3072 outputs
layer = nn.Linear(768, 3072)

# Convolution: 64 output channels, 3x3 kernel
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

# Layer normalization
norm = nn.LayerNorm(768)

# RMS normalization (LLaMA-style)
rms_norm = nn.RMSNorm(768)

# Embedding table: 50257 tokens, 768 dimensions
embed = nn.Embedding(50257, 768)
```

### PyTorch to MLX: same concepts, similar names

When porting [CUDA](../glossary.md#cuda) models, most layer names translate directly. The PyTorch mental model carries over:

| PyTorch              | MLX                  | Notes                                    |
|----------------------|----------------------|------------------------------------------|
| `nn.Linear`          | `nn.Linear`          | Identical concept, very similar API      |
| `nn.Conv2d`          | `nn.Conv2d`          | Same concept; check channel order (NCHW vs NHWC) |
| `nn.LayerNorm`       | `nn.LayerNorm`       | Same                                     |
| `nn.RMSNorm`         | `nn.RMSNorm`         | Same                                     |
| `nn.Embedding`       | `nn.Embedding`       | Same                                     |
| `nn.Linear(bias=False)` | `nn.Linear(bias=False)` | Same                               |
| `F.relu`             | `nn.relu` or `mx.maximum(x, 0)` | Functional form               |
| `F.gelu`             | `nn.gelu`            | Same                                     |
| `F.silu`             | `nn.silu`            | Same                                     |

The main differences are in how models are structured (how layers are registered), how parameters are accessed, and some framework-specific conventions. But the underlying math is the same.

### Weight files contain the trained parameters

When you download a pretrained model (a `.safetensors` or `.npz` file), you are downloading the trained parameter values -- the weights and biases for every layer in the network.

The file contains a dictionary mapping parameter names to tensors:

```
"transformer_blocks.0.attn.to_q.weight"   -> tensor of shape (768, 768)
"transformer_blocks.0.attn.to_k.weight"   -> tensor of shape (768, 768)
"transformer_blocks.0.attn.to_v.weight"   -> tensor of shape (768, 768)
"transformer_blocks.0.ff.net.0.weight"    -> tensor of shape (3072, 768)
...  (thousands more)
```

Porting a model means:
1. Defining the same layer structure in MLX (so the code knows what shape to expect)
2. Loading the weight tensors from the file
3. Placing each weight tensor into the correct layer in the MLX model

If the layer structure doesn't match the weight shapes, you get a shape mismatch error. If the layer names don't match, you have to map them manually.

### Navigating the layer hierarchy in LTX-Video

The LTX-Video model is a video generation transformer with thousands of layers. When you see code like this:

```python
model.transformer_blocks[0].attn.to_q
```

You're navigating a tree of nested layers:

```
model
 └── transformer_blocks          [list of 28 transformer blocks]
      └── [0]                    [first transformer block]
           └── attn              [self-attention sub-module]
                └── to_q         [Linear layer projecting queries]
                     └── weight  [tensor of shape (768, 768)]
```

`to_q.weight` is the actual tensor -- a (768, 768) matrix of learned numbers. It sits at the bottom of a hierarchy of named containers. The hierarchy exists only to organize the layers; the math at the bottom is always the same linear algebra.

Understanding this hierarchy is essential when you're debugging shape mismatches or loading weights from a file: you need to know which named parameter in the file corresponds to which layer object in the code.


## Key Terms

| Term | Definition |
|------|------------|
| **Layer** | A single mathematical transformation within a neural network; takes a tensor in, produces a tensor out |
| **[Weight](../glossary.md#weight)** | A learned parameter tensor inside a layer; the "knobs" that get adjusted during training |
| **[Bias](../glossary.md#bias)** | A learned offset vector added after the weight multiplication; allows the layer to shift its output |
| **[Activation Function](../glossary.md#activation-function)** | A non-linear function applied after each layer; without it, stacking layers provides no benefit over a single layer (e.g., ReLU, GELU, SiLU) |
| **[Parameter](../glossary.md#parameter)** | Any number inside the model that is learned from data during training; includes weights, biases, embedding vectors, and normalization scales |
| **[Hyperparameter](../glossary.md#hyperparameter)** | A design choice made before training that is not learned from data; examples include learning rate, number of layers, and layer width |
| **[Model](../glossary.md#model)** | A specific layer architecture combined with a specific set of trained parameter values |
| **[Forward Pass](../glossary.md#forward-pass)** | The process of passing an input tensor through every layer in sequence to produce an output; also called "inference" when done on a trained model |
| **[Linear Layer](../glossary.md#linear-layer)** | A layer that performs `output = input @ weight + bias`; the most fundamental layer type (also called Dense or Fully Connected) |
| **Convolution** | A layer that applies a sliding kernel across a spatial input; efficient for images because the same weights are reused at every position |
| **Normalization** | A layer that rescales activations to have controlled magnitude; prevents values from exploding or vanishing as they pass through many layers (e.g., LayerNorm, RMSNorm) |
| **Embedding** | A layer that converts integer indices into dense learned vectors via a lookup table; used to represent tokens, positions, or any categorical input |


## Sources

- Nielsen, M. (2015). *Neural Networks and Deep Learning*. Free online at [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com). Chapters 1 and 2 cover the core concepts with clear visual explanations -- the best free introduction available.
- [MLX nn module documentation](https://ml-explore.github.io/mlx/build/html/python/nn/index.html) -- the authoritative reference for all layer types available in `mlx.nn`, their parameters, and their behavior.
- 3Blue1Brown. *Neural Networks* video series. Available at [3b1b.co/neural-networks](https://www.3blue1brown.com/topics/neural-networks). Particularly chapters 1 ("But what is a neural network?") and 2 ("[Gradient](../glossary.md#gradient) descent, how neural networks learn") -- excellent visual intuition for how layers, weights, and training work together.


## See Also

- [Tensors](01-tensors.md) -- prerequisite; neural networks are built entirely from tensor operations. If the matrix multiply in this page felt unfamiliar, review the Operations section of the Tensors page first.
- [Training vs. Inference](03-training-vs-inference.md) -- this page explains what a trained model is; the next page explains how training actually happens and what changes between training and running the model.
- [Transformers](05-transformers.md) -- the specific architecture used by LTX-Video, all language models, and most state-of-the-art models. Transformers are a particular arrangement of the layer types described here, plus the attention mechanism.
