# Transformers

> **One-liner:** The transformer is an architecture that processes all parts of its input simultaneously using attention, making it the foundation of virtually all modern AI models.

## The Intuition

You know from the Neural Networks page that a model is a sequence of layers, and from the GPU Computing page that those layers run in parallel on specialized hardware. The question now is: what specific arrangement of layers makes a model capable of understanding language, generating images, or synthesizing video?

The answer -- for the vast majority of modern AI -- is the transformer.

### The reading-with-bad-memory problem

Before transformers, the dominant approach for sequence processing was the **Recurrent Neural Network (RNN)**. An RNN reads its input one element at a time, left to right, and carries a "memory" vector that gets updated at each step.

```
RNN reading a sentence:

  "The cat sat on the mat"

  "The" --> [RNN] --> memory_1
                         |
  "cat" --> [RNN] --> memory_2
                         |
  "sat" --> [RNN] --> memory_3
                         |
  "on"  --> [RNN] --> memory_4
                         |
  "the" --> [RNN] --> memory_5
                         |
  "mat" --> [RNN] --> memory_6 --> output
```

This design has a fundamental problem: by the time the RNN finishes reading the sentence, the memory vector has been updated six times. The earliest words have been compressed, overwritten, and blurred. If you ask the model to relate "mat" back to "The cat", that connection has to survive six updates -- and in practice, for long sequences, it mostly doesn't.

RNNs have bad memory because they process sequentially, one step at a time.

### The "see the whole page" insight

The 2017 paper "Attention Is All You Need" (Vaswani et al.) proposed discarding the sequential reading model entirely.

Instead: give the model the entire input at once, and let every element look directly at every other element to figure out what's related to what.

```
Transformer reading the same sentence:

  "The" "cat" "sat" "on" "the" "mat"
    |     |     |     |     |     |
    +-----+-----+-----+-----+-----+
                  |
           [Attention: every word can
            look directly at every other word]
                  |
    +-----+-----+-----+-----+-----+
    |     |     |     |     |     |
  "The" "cat" "sat" "on" "the" "mat"
  (enriched with context from all other words)
```

When processing "sat", the transformer can directly inspect "cat" and "mat" without the information having to survive a chain of updates. When processing "mat", it can directly attend back to "The cat" at the beginning. No information bottleneck, no blurring.

This is the core insight of the transformer: **transform a sequence of inputs into a sequence of outputs by letting every element attend to every other element.** The attention mechanism is what makes this possible -- and it is covered in depth on the next page.

The consequences were enormous. Because transformers can process the whole input in parallel, they map perfectly to GPU hardware. Because they maintain direct connections between any two elements regardless of distance, they learn long-range relationships that RNNs struggled with. That combination -- parallel and long-range -- is why transformers displaced every prior architecture.


## How It Actually Works

### The original architecture: encoder + decoder

The 2017 paper introduced a specific architecture for sequence-to-sequence tasks (translating French to English, for example). It had two halves:

```
                    ORIGINAL TRANSFORMER
                    (Vaswani et al., 2017)

Input sequence                Output sequence
(French text)                (English text)

     |                              ^
     v                              |
+----------+                 +----------+
|          |                 |          |
| ENCODER  |   context  ---> | DECODER  |
|          |   vectors       |          |
|  N blocks|                 |  N blocks|
|          |                 |          |
+----------+                 +----------+

Encoder: reads the full input and produces
         a rich vector representation of it.

Decoder: generates the output token by token,
         attending both to what it has generated
         so far and to the encoder's context.
```

The encoder's job: understand the input sequence. The decoder's job: generate the output sequence, using the encoder's understanding as a guide.

### Modern variants: three families

In the years after 2017, researchers discovered that you could use just one half -- or modify the attention pattern -- to get a model tuned for a specific task:

**Encoder-only (e.g., BERT):**

The full encoder stack, no decoder. You feed in a sequence and get back a rich contextual representation of it. Good for tasks that require understanding: classification, named-entity recognition, semantic search. Not used for text generation.

**Decoder-only (e.g., GPT, LLaMA, Mistral, Gemma):**

Just the decoder stack, modified so each position can only attend to positions before it (no "cheating" by looking at future tokens). Used for text generation: predicts the next token, over and over. This is what modern language models are.

**Encoder-decoder (e.g., T5, Whisper):**

The full original architecture. Used when the task is explicitly about mapping one sequence to another: translation, summarization, speech-to-text.

For the models you work with (LLaMA for text, LTX-Video for video), the relevant family is decoder-only for text models, and a modified architecture called DiT (Diffusion Transformer) for image/video -- described below.

### A transformer block: the repeating unit

The key design principle of all transformer variants is **stacking identical blocks**. One transformer block is:

```
SINGLE TRANSFORMER BLOCK

           Input from previous block
                    |
                    v
          +-----------------+
          |   Layer Norm    |   <-- normalize before attention
          +-----------------+
                    |
                    v
          +-----------------+
          |  Self-Attention |   <-- every position attends to all others
          +-----------------+
                    |
                    +<--- residual (add original input back)
                    |
                    v
          +-----------------+
          |   Layer Norm    |   <-- normalize before feed-forward
          +-----------------+
                    |
                    v
          +-----------------+
          | Feed-Forward    |   <-- position-wise MLP
          | Network (FFN)   |       (same weights applied to each position)
          +-----------------+
                    |
                    +<--- residual (add pre-FFN input back)
                    |
                    v
          Output to next block
```

Breaking down each component:

**Layer Norm** normalizes the activations to have controlled magnitude. Without it, values can drift to extreme scales as they pass through many blocks. Placed *before* each sub-layer (this is called "pre-norm" and is the standard in modern architectures like LLaMA, though the original paper placed it after).

**Self-Attention** is the core mechanism. Each position in the sequence produces three vectors -- a Query, a Key, and a Value -- and the output for each position is a weighted sum of all the Values, where the weights come from how much each Query matches each Key. The full mechanism is covered in detail on the Attention page.

**Residual connection** (also called skip connection) adds the block's input directly to its output:

```
block_output = sub_layer(input) + input
               ^^^^^^^^^^^^^^^^^   ^^^^^
               what the block      what
               computed            came in
```

This is not optional decoration -- it is essential for training deep networks. Without residual connections, gradients vanish as they flow back through many layers, and the network cannot learn. With them, gradients have a direct path through the addition, all the way back to the earliest blocks.

**Feed-Forward Network (FFN)** is a two-layer MLP applied identically to each position in the sequence:

```
FFN(x) = Linear_2( activation( Linear_1(x) ) )

          x: shape (seq_len, d_model)
          Linear_1 expands:  d_model --> d_ffn   (typically 4x larger)
          activation:        GELU or SiLU
          Linear_2 contracts: d_ffn --> d_model
```

The FFN is where the model does most of its "memorization" -- where knowledge from training is stored in the weights. The expansion-contraction pattern (sometimes called a "bottleneck inverted") is standard across all transformer architectures.

### How blocks stack: the full model

A transformer model is N identical blocks applied in sequence:

```
STACKED TRANSFORMER BLOCKS (decoder-only, e.g., LLaMA-7B: 32 blocks)

 Input tokens
      |
      v
[Embedding Layer]   <-- token IDs -> dense vectors
      |
      v
[Positional Encoding]   <-- inject word order information
      |
      v
+-----------+
| Block 0   |
+-----------+
      |
      v
+-----------+
| Block 1   |
+-----------+
      |
      v
    ...
      |
      v
+-----------+
| Block 31  |
+-----------+
      |
      v
[Final LayerNorm]
      |
      v
[Output head]   <-- projects to vocabulary size, produces token probabilities
      |
      v
 Output tokens
```

Every block has the same structure and the same dimensions. The difference between a 7B and a 70B model is not a different architecture -- it is more blocks, wider dimensions (larger `d_model`), and more attention heads. GPT-3 has 96 blocks. LLaMA-70B has 80 blocks. Each block sees the output of the previous one and adds its own refinement.

A useful way to think about it: the first few blocks typically extract low-level patterns (syntax, word relationships near each other). Later blocks handle higher-level reasoning (semantics, long-range coherence). This is an emergent property from training, not an explicit design -- the blocks are all identical at initialization.

### Positional encoding: solving the "no sense of order" problem

Because the transformer sees all positions simultaneously, it has no built-in sense of order. The token at position 5 and the token at position 50 would look identical to the attention mechanism without some additional signal.

Positional encoding injects position information into each token's representation before the blocks begin:

```
token vector + position vector = position-aware token vector
```

The original paper used **sinusoidal positional encoding**: a fixed mathematical formula based on sine and cosine waves at different frequencies. Different dimensions of the position vector oscillate at different rates, giving each position a unique fingerprint.

Modern architectures use **Rotary Position Embedding (RoPE)**, used in LLaMA, Mistral, and most current models. Instead of adding position vectors to the tokens, RoPE rotates the Query and Key vectors inside the attention mechanism by an angle proportional to position. The rotation encodes not absolute position but *relative* position -- how far apart two tokens are. This generalizes better to sequences longer than those seen in training.

```
Sinusoidal (original):
  token_embedding + position_embedding --> positional_token

RoPE (modern):
  Q rotated by angle(pos_i) @ K rotated by angle(pos_j)
  --> encodes relative distance (pos_j - pos_i) in the attention score
```

You will see RoPE everywhere in transformer model code: it appears in the attention sub-layer, applied to Q and K tensors before the dot product.


## Why It Matters for MLX

### Nearly every model you port is a transformer

Language models (LLaMA, Mistral, Gemma): transformer decoder stacks. Image models (DiT-based): transformer blocks operating on image patches. Video models (LTX-Video): transformer blocks operating on video patches. Speech models (Whisper): encoder-decoder transformers. Multi-modal models: transformers with specialized embedding layers for different modalities.

When you port a model to MLX, you are almost certainly porting a transformer. Understanding the block structure means you can read any new model's architecture description and immediately know what to expect in the code.

### Code structure mirrors the architecture exactly

The layer hierarchy you navigate when loading weights is a direct reflection of the transformer block diagram. In LTX-Video's code:

```python
model.transformer_blocks[i].attn       # self-attention sub-layer
model.transformer_blocks[i].attn.to_q  # query projection (Linear layer)
model.transformer_blocks[i].attn.to_k  # key projection (Linear layer)
model.transformer_blocks[i].attn.to_v  # value projection (Linear layer)
model.transformer_blocks[i].ff         # feed-forward network sub-layer
model.transformer_blocks[i].ff.net[0]  # first linear in the FFN
model.transformer_blocks[i].norm1      # layer norm before attention
model.transformer_blocks[i].norm2      # layer norm before FFN
```

When a weight file contains a key like:

```
"transformer_blocks.3.attn.to_q.weight"
```

You now know exactly what it is: the weight matrix of the Query projection Linear layer in the 4th transformer block's self-attention sub-module. You can trace the path in your head without looking at the code.

When you hit a shape mismatch error while loading weights, the error name tells you which block and sub-layer is wrong. You can navigate there directly.

### Debugging layer by layer

Because the model is N identical blocks, a useful debugging technique is to inspect intermediate outputs block by block:

```python
import mlx.core as mx

# Run the model and capture block outputs
x = embeddings  # shape: (batch, seq_len, d_model)
for i, block in enumerate(model.transformer_blocks):
    x = block(x)
    print(f"Block {i}: mean={mx.mean(x):.4f}, std={mx.std(x):.4f}")
    # Look for: values exploding, values collapsing toward 0,
    #           NaN appearing at a specific block
```

If outputs go wrong at block 7 but are fine at block 6, the problem is in block 7 (or in the weight loading for block 7). The block structure gives you a natural bisect point for debugging.

### LTX-Video: a DiT (Diffusion Transformer)

LTX-Video is built on a **Diffusion Transformer (DiT)** -- a transformer where the input is not word tokens but video patches.

Here is how it maps to the transformer concepts on this page:

```
Standard Transformer              LTX-Video DiT
---------------------------------  ----------------------------------
Input: word tokens                 Input: video frame patches
       (each token = a word)              (each patch = a spatial region
                                           across time)

Embedding: token ID -> vector      Embedding: pixel values -> vector
                                   (via patch embedding Conv layer)

Positional encoding: word order    Positional encoding: spatial (x, y)
                                   + temporal (t) positions

Transformer blocks: N identical    Transformer blocks: 28 identical
attention + FFN layers             attention + FFN layers
                                   with added conditioning on:
                                   - noise timestep
                                   - text prompt embeddings

Output: token probabilities        Output: denoised video patches
```

The transformer block structure is preserved almost exactly. The main differences are:
1. The input patches replace word tokens as the sequence elements.
2. Extra conditioning signals (timestep, text) are injected into each block, typically via adaptive layer norm or cross-attention.
3. The output is interpreted as denoised pixels rather than token probabilities.

This is why understanding transformers is prerequisite to understanding diffusion models. The diffusion process (adding and removing noise) is the outer loop; the transformer is the neural network that does the denoising at each step.


## Key Terms

| Term | Definition |
|------|------------|
| **Transformer** | A neural network architecture that processes all input positions simultaneously using attention, allowing every element to directly interact with every other element |
| **Encoder** | The transformer component that reads an input sequence and produces a contextual representation of it; used in BERT and the first half of translation models |
| **Decoder** | The transformer component that generates an output sequence, attending to its own previous outputs and (optionally) to an encoder's output; used in GPT-style language models |
| **Transformer Block** | The repeating unit of a transformer: LayerNorm -> Self-Attention -> Residual + LayerNorm -> FFN -> Residual. N of these are stacked to form the full model |
| **Residual Connection** | A connection that adds a layer's input to its output (`output = layer(x) + x`); essential for training deep networks by providing direct paths for gradient flow |
| **Feed-Forward Network (FFN/MLP)** | The two-layer MLP inside each transformer block, applied identically to each position; expands dimension by ~4x, applies activation, then contracts back |
| **Positional Encoding** | A mechanism to inject word/token order information into a transformer that otherwise sees all positions simultaneously; common forms are sinusoidal (original) and RoPE (modern) |
| **RoPE** | Rotary Position Embedding; the positional encoding used in LLaMA and most modern transformers; encodes relative position by rotating Query and Key vectors inside attention |
| **Layer Stacking** | The practice of repeating identical transformer blocks in sequence; depth (number of blocks) is a primary driver of model capability |
| **DiT** | Diffusion Transformer; a transformer architecture applied to image or video patches instead of word tokens, used in image and video generation models including LTX-Video |


## Sources

- Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems 30*. The original transformer paper -- short and worth reading. Available at [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Alammar, J. "The Illustrated Transformer." A visual walkthrough of the original architecture with clear diagrams of the attention mechanism and encoder-decoder structure. Available at [jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)
- Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." The LLaMA architecture paper, which describes the pre-norm, RoPE, SwiGLU FFN, and other modern transformer conventions. Available at [arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)
- Peebles, W., & Xie, S. (2023). "Scalable Diffusion Models with Transformers." The DiT paper, which introduces replacing U-Net backbones with transformer blocks for diffusion models. Available at [arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)


## See Also

- [Neural Networks](02-neural-networks.md) -- prerequisite; transformers are built from the layer types (Linear, LayerNorm, Embedding, activations) described there. The weight hierarchy, forward pass, and parameter loading concepts apply directly.
- [Attention](06-attention.md) -- the mechanism at the heart of every transformer block; this page described *what* self-attention does (every element attends to every other), the next page explains *how* it computes that.
- [Diffusion](08-diffusion.md) -- the outer process that wraps the transformer in LTX-Video; understanding the transformer block is prerequisite to understanding how DiT fits into the diffusion pipeline.
