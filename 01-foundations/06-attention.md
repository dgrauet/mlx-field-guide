# Attention

> **One-liner:** [Attention](../glossary.md#attention) is a mechanism that lets each element in a sequence dynamically decide how much to focus on every other element.

## The Intuition

You know from the Transformers page that a transformer gives every token a direct line of sight to every other token. Attention is the mechanism that implements that line of sight -- the specific computation that decides which tokens to focus on and by how much.

### The reference problem

Read this sentence:

> "The cat sat on the mat because **it** was tired."

What does "it" refer to? You don't have to think about it -- you immediately know it's "the cat". But how? You scanned the sentence, found "the cat" as the only nearby noun that could be tired, and assigned "it" the meaning of "cat".

That is attention. For each word ("it"), you looked at all the other words, decided which were relevant ("cat" scores high, "mat" scores low, "sat" scores medium-low), and used that relevance pattern to interpret the current word.

```
Processing the word "it":

  "The"   "cat"   "sat"   "on"   "the"   "mat"   "because"   "it"
    |        |       |      |       |       |          |        |
    v        v       v      v       v       v          v        v
  0.02     0.65    0.10   0.01    0.02    0.03       0.04     0.13
                                                              (self)
   ^                ^
   low              HIGH -- "it" strongly attends to "cat"
   relevance        relevance

Output for "it" = weighted combination of all words' information,
                  with "cat" contributing most heavily
```

Three things make this different from a lookup table or hardcoded rule:

1. **The scores are computed, not stored.** There is no database of "it -> cat". The model computes a relevance score between every pair of positions on every forward pass.
2. **The scores depend on content, not just position.** "it" attends to "cat" not because of where they sit in the sentence, but because of what their vectors represent.
3. **The scores are learned, not designed.** The weight matrices that produce those relevance scores were learned from data. No human wrote rules for pronoun resolution.

### Attention as a spotlight

A useful mental model: for each word, the model shines a spotlight across the entire sequence. The spotlight is not uniform -- it is brighter on some words than others. The brightness is determined by relevance. The output for that word is the sum of all other words' information, each weighted by how much light hits it.

```
Processing each word in "The cat sat":

  Word being processed: "sat"
  Spotlight intensities:

  "The"  "cat"  "sat"
    |      |      |
    v      v      v
   [dim] [bright] [medium]
    0.1    0.7     0.2

  Output for "sat" =
    0.1 * (info from "The") +
    0.7 * (info from "cat") +
    0.2 * (info from "sat")
```

The spotlight analogy captures two important properties: every element always contributes *something* (no word is completely ignored -- the minimum weight is just very small), and the pattern is dynamic (the same word "it" would attend to different targets in different sentences).


## How It Actually Works

### Query, Key, Value: the retrieval framework

The mechanism is built around three vectors computed from each input token. The names come from database retrieval, and the analogy is useful:

```
Database retrieval:                 Attention:
  query  = "what I'm looking for"    Q = "what am I looking for?"
  key    = "what this entry is"      K = "what do I contain?"
  value  = "the actual content"      V = "what information do I provide?"
```

For each token in the sequence, three separate linear projections produce Q, K, and V:

```
INPUT TOKEN PROJECTIONS

    token embedding
    (shape: d_model)
          |
    +-----+-----+-----+
    |     |     |     |
    v     v     v     |
  W_Q   W_K   W_V     |   (three separate weight matrices,
    |     |     |     |    all learned during training)
    v     v     v     |
    Q     K     V     |
  (d_k) (d_k) (d_v)  |

  For each of the N tokens in the sequence, we produce N Q's, N K's, N V's.

  Stacked into matrices:
    Q: shape (seq_len, d_k)    -- one row per token
    K: shape (seq_len, d_k)    -- one row per token
    V: shape (seq_len, d_v)    -- one row per token
```

The intuition for each:

- **Query (Q):** "What am I looking for?" [Token](../glossary.md#token) at position i uses its Q vector to "ask a question" -- it encodes what kind of information it needs from the rest of the sequence.
- **Key (K):** "What do I contain?" Token at position j uses its K vector to "advertise" -- it encodes what kind of information it holds.
- **Value (V):** "What do I actually provide?" Token at position j's V vector holds the actual information to be passed along if it's selected.

### Computing attention scores

The attention score between two tokens measures how relevant token j is to token i. It is computed as the dot product of token i's query with token j's key:

```
score(i, j) = Q[i] · K[j]

            = Q[i, 0]*K[j, 0] + Q[i, 1]*K[j, 1] + ... + Q[i, d_k-1]*K[j, d_k-1]
```

The dot product is large when Q[i] and K[j] point in similar directions in d_k-dimensional space. The model learns to make Q vectors and K vectors align when the corresponding tokens are relevant to each other.

Computing all pairs at once is a single matrix multiplication:

```
ATTENTION SCORE MATRIX

      Q                  K^T                  Scores
  (seq x d_k)        (d_k x seq)           (seq x seq)

 token 0: [...]    token 0 [...]    [ s(0,0)  s(0,1)  s(0,2) ]
 token 1: [...]  x token 1 [...]  = [ s(1,0)  s(1,1)  s(1,2) ]
 token 2: [...]    token 2 [...]    [ s(2,0)  s(2,1)  s(2,2) ]

Scores[i, j] = "how much should token i attend to token j?"

For a sequence of 6 tokens: a 6x6 matrix of scores.
For a sequence of 1024 tokens: a 1024x1024 matrix.
For a sequence of 4096 tokens: a 4096x4096 matrix = 67M values.
         ^^^-- this is why attention is expensive at long context
```

### Scaling: preventing dot product saturation

There is a problem with raw dot products: as `d_k` grows, the dot products grow in magnitude proportionally. When scores are very large, the softmax that follows saturates -- it produces outputs very close to 0 or 1, giving near-zero gradients and making training slow.

The fix is to scale by the square root of d_k before softmax:

```
scaled_score(i, j) = Q[i] · K[j]  /  sqrt(d_k)
```

Why `sqrt(d_k)`? If Q and K are vectors of independent random values with mean 0 and variance 1, their dot product has variance d_k. Dividing by `sqrt(d_k)` brings the variance back to 1, keeping scores in a reasonable range regardless of the model's hidden dimension size.

### Softmax: scores to probabilities

The scaled scores for token i across all other tokens are passed through softmax to become a probability distribution -- they sum to 1, all non-negative:

```
SOFTMAX OVER ATTENTION SCORES

  Raw scores for token "it" attending to all positions:
  [-0.3,  2.1,  0.4, -1.2,  0.1,  0.2,  0.5,  0.8]
  "The" "cat" "sat" "on" "the" "mat" "because" "it"

  After softmax:
  [0.02,  0.65, 0.10, 0.01, 0.02, 0.03,  0.07, 0.10]

  These are the attention weights: they sum to 1.0.
  "cat" dominates with 0.65 -- the model has learned that
  "it" should draw heavily from the cat's representation.
```

The softmax ensures that attending more strongly to one token means attending less to others. It also makes the scores differentiable, enabling gradient-based training.

### Output: weighted sum of Values

The final output for each token is the weighted sum of all Value vectors, where the weights are the softmax attention probabilities:

```
OUTPUT COMPUTATION

  Attention weights for token "it":
  w = [0.02, 0.65, 0.10, 0.01, 0.02, 0.03, 0.07, 0.10]

  Value vectors (one per token in the sequence):
  V[0] = value vector for "The"
  V[1] = value vector for "cat"     <- contributes most
  V[2] = value vector for "sat"
  ...

  Output for "it":
  out["it"] = 0.02*V["The"] + 0.65*V["cat"] + 0.10*V["sat"] + ...

  In matrix form: output = softmax_weights @ V
                  shape:  (seq_len, d_v)   <-- same shape as input
```

### The complete formula

Putting it all together:

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

                     ┌────────────────────────────────────┐
                     │                                    │
           ┌─────────┴──────────┐            ┌───────────┴───────────┐
           │  Attention weights │            │     Value lookup       │
           │  shape: (seq, seq) │            │     shape: (seq, d_v)  │
           └────────────────────┘            └───────────────────────┘

  Step by step for a sequence of length N with d_k = 64:

  Q:          (N, 64)
  K^T:        (64, N)
  Q @ K^T:    (N, N)       <- raw attention scores
  / sqrt(64): (N, N)       <- scaled scores
  softmax:    (N, N)       <- attention weights (each row sums to 1)
  @ V:        (N, d_v)     <- output (enriched token representations)
```

This is **[Scaled Dot-Product Attention](../glossary.md#scaled-dot-product-attention)**, the operation at the core of every transformer.

### Multi-head attention: running it in parallel

A single attention operation might learn "this verb attends to its subject". But attention needs to learn many different types of relationships simultaneously -- syntax, coreference, positional proximity, semantic similarity.

Multi-head attention runs N independent attention operations in parallel, each with its own set of Q, K, V projection matrices:

```
MULTI-HEAD ATTENTION

                  Input (seq_len, d_model)
                         |
         +---------------+---------------+
         |               |               |
    Head 0           Head 1         Head H-1
    (W_Q0,W_K0,W_V0) (W_Q1,W_K1,W_V1) (...)
         |               |               |
  Attention(Q0,K0,V0) Attention(Q1,K1,V1) ...
  output: (seq, d_head) output: (seq, d_head)
         |               |               |
         +-------+-------+-------+-------+
                         |
                    Concatenate
                  (seq_len, H * d_head)
                         |
                   Output projection W_O
                  (seq_len, d_model)
                         |
                       Output

  Typical dimensions (LLaMA-7B):
    d_model = 4096
    H = 32 heads
    d_head = d_model / H = 128  (each head's Q/K/V dimension)
```

Each head learns different relationships. In practice, different heads specialize:
- Some heads learn local relationships (nearby tokens)
- Some heads learn long-range dependencies
- Some heads track syntactic structure (subject-verb relationships)
- Some heads track coreference ("it" -> antecedent)

No one programs these specializations -- they emerge from training. The "multi-head" design gives the model capacity to represent many types of relationships simultaneously without any of them interfering.

### Self-attention vs. cross-attention

The mechanism described above is **self-attention**: Q, K, and V all come from the same sequence. Each token attends to all tokens in the same sequence, including itself.

**Cross-attention** uses Q from one sequence and K, V from a different sequence. This is how one modality guides another:

```
SELF-ATTENTION (e.g., decoder processing video)

  video patches: [p0, p1, p2, p3, p4, p5]
                  |    |    |    |    |    |
                 Q,K,V Q,K,V ...              <- all from same sequence

  Each video patch attends to all other video patches.
  Use case: video patches understanding their spatial/temporal context.


CROSS-ATTENTION (e.g., text guiding video in LTX-Video)

  Query source:     video patches  [p0, p1, p2, ...]
                         |
                      Q = W_Q * video_patch_embeddings

  Key/Value source: text tokens   [t0, t1, t2, ...]
                         |
                      K = W_K * text_embeddings
                      V = W_V * text_embeddings

  Attention score: how much does each video patch "need"
                   each text token's information?

  Output: each video patch is enriched with information
          from the most relevant parts of the text prompt.
```

In LTX-Video, cross-attention is the mechanism that connects a text prompt ("a cat running across a field") to the video generation. The video patches ask questions (Q), the text tokens answer (K and V). The text prompt shapes every video frame by modulating what each spatial-temporal patch attends to.

### KV-cache: making inference efficient

During text generation, the model generates one token at a time. At each step, it runs attention over all tokens so far. Without optimization, this means recomputing K and V for every token that was already processed at every generation step -- redundant work that grows quadratically.

The KV-cache stores previously computed K and V tensors so they don't need to be recomputed:

```
GENERATION WITHOUT KV-CACHE (wasteful)

  Step 1: generate token 1
    Input: [tok_0]
    Compute K, V for: tok_0
    Attend, produce tok_1

  Step 2: generate token 2
    Input: [tok_0, tok_1]
    Compute K, V for: tok_0 (AGAIN), tok_1
    Attend, produce tok_2

  Step 3: generate token 3
    Input: [tok_0, tok_1, tok_2]
    Compute K, V for: tok_0 (AGAIN), tok_1 (AGAIN), tok_2
    Attend, produce tok_3

  Step N: recomputes K,V for all N-1 previous tokens. O(N^2) total.


GENERATION WITH KV-CACHE (efficient)

  Step 1: generate token 1
    Input: [tok_0]
    Compute K, V for: tok_0 --> cache: {K: [K_0], V: [V_0]}
    Attend, produce tok_1

  Step 2: generate token 2
    Input: just tok_1 (new token only)
    Compute K, V for: tok_1 --> cache: {K: [K_0, K_1], V: [V_0, V_1]}
    Attend tok_1's Q against full cached K, V
    Produce tok_3

  Step N: only compute K,V for the ONE new token. O(N) total.
```

In PyTorch code, you'll see KV-cache passed as `past_key_values`. In MLX ports, you may see it as `cache` or `kv_cache`. The pattern is the same: a list of (K, V) pairs, one per transformer block, that grows by one slot each generation step.

The memory cost: at 4096 tokens with LLaMA-7B, the KV-cache holds 32 blocks × 2 (K and V) × 32 heads × 128 head_dim × 4096 tokens × 2 bytes (float16) ≈ 1.07 GB. At long contexts or large batch sizes, KV-cache memory dominates inference memory usage.

### Grouped-Query Attention (GQA)

Full multi-head attention has one K and V head per Q head. At 32 heads, that means 32 separate K matrices and 32 separate V matrices stored in the KV-cache.

**Grouped-Query Attention (GQA)** reduces this by sharing K and V across groups of Q heads:

```
MULTI-HEAD ATTENTION (MHA)       GROUPED-QUERY ATTENTION (GQA)
(H Q heads, H K/V heads)         (H Q heads, G K/V groups)

  Q0, Q1, Q2, Q3                   Q0, Q1, Q2, Q3
   |   |   |   |                    |    |   |    |
  K0  K1  K2  K3                   K0       K1
  V0  V1  V2  V3                   V0       V1
                                    |    |   |    |
                                   Q0,Q1 attend  Q2,Q3 attend
                                   to K0/V0      to K1/V1

GQA with G=1: every Q head shares one K/V head.
             (called Multi-Query Attention, MQA)

LLaMA-2-70B: 64 Q heads, 8 K/V groups (G=8).
             KV-cache is 8x smaller than standard MHA.
```

GQA reduces KV-cache memory with minimal quality degradation. Most modern large models use it -- look for `num_key_value_heads` (distinct from `num_attention_heads`) in model configs.


## Why It Matters for MLX

### Attention is the compute bottleneck

At inference time, the attention score matrix computation (`Q @ K^T`) is O(seq_len²) in both time and memory. For a sequence of 1024 tokens, that's 1M values per head. For 4096 tokens: 16M values per head, 32 heads: 512M values per attention layer. With 32 blocks, a single forward pass touches 16B attention values.

This is why attention is where optimization work is concentrated. The model's FFN layers dominate parameter count; attention dominates runtime at long context.

### Flash Attention: the key algorithm

Standard attention computes and writes the full attention matrix to [GPU](../glossary.md#gpu) memory before multiplying by V. For long sequences, this intermediate matrix doesn't fit in fast on-chip memory (SRAM), so it gets written to [VRAM](../glossary.md#vram) and read back -- a memory bandwidth bottleneck.

**[Flash Attention](../glossary.md#flash-attention)** (Dao et al., 2022) rewrites the attention computation to avoid materializing the full attention matrix. It tiles the inputs into blocks that fit in SRAM, computes attention block by block, and uses a numerically stable online softmax algorithm that never needs to see all scores at once.

```
STANDARD ATTENTION (memory-bound at long context)

  Q, K, V in VRAM
      |
  Compute Q @ K^T  -->  write full scores matrix to VRAM (bottleneck)
      |                 (seq_len x seq_len = huge)
  Read scores back
      |
  Softmax
      |
  Write softmax weights to VRAM (bottleneck again)
      |
  Read weights back, compute weights @ V
      |
  Output in VRAM


FLASH ATTENTION (SRAM-resident, IO-efficient)

  Q, K, V in VRAM
      |
  Load tiles of Q, K, V into SRAM (fits in fast memory)
      |
  Compute partial attention in tiles, accumulate result
  (never writes full seq x seq matrix to VRAM)
      |
  Output in VRAM

  Result: same mathematical output, significantly lower
          memory bandwidth usage, faster on long sequences.
```

In [CUDA](../glossary.md#cuda)-land, PyTorch uses Flash Attention automatically for `F.scaled_dot_product_attention` when the input shapes support it.

In MLX, the equivalent is `mx.fast.scaled_dot_product_attention`:

```python
import mlx.core as mx

# Standard attention (manual):
scores = (queries @ keys.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
if mask is not None:
    scores = scores + mask
weights = mx.softmax(scores, axis=-1)
output = weights @ values

# Flash Attention in MLX (preferred for performance):
output = mx.fast.scaled_dot_product_attention(
    queries, keys, values,
    scale=1.0 / math.sqrt(head_dim),
    mask=mask
)
# Same result, memory-efficient tiled implementation.
```

When porting a CUDA model's attention block to MLX, reach for `mx.fast.scaled_dot_product_attention` rather than manually implementing the four-step computation. It handles the memory tiling internally and is meaningfully faster on long contexts.

### KV-cache management in MLX ports

When porting a model that supports incremental generation, KV-cache handling is one of the trickier translation tasks. The interface varies between frameworks:

```python
# PyTorch (HuggingFace style) -- past_key_values is a tuple of tuples
outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
new_past = outputs.past_key_values

# MLX (common pattern in mlx-lm and ported models) -- cache is a list
# Each attention layer receives its own cache object
cache = [KVCache() for _ in range(model.num_layers)]
logits = model(input_ids, cache=cache)
# cache is mutated in place by each attention layer
```

When porting, you need to:
1. Identify how the CUDA model stores and passes KV-cache (`past_key_values`, `cache`, etc.)
2. Design an equivalent MLX structure
3. Ensure the cache grows correctly at each step (append new K, V; don't overwrite)
4. Handle cache invalidation when starting a new sequence

In LTX-Video, the model uses cross-attention (text → video) whose K and V come from the text encoder. These are constant across all denoising steps -- they can be computed once and reused, a form of static caching.

### LTX-Video: cross-attention in practice

LTX-Video has two types of attention in each transformer block:

1. **Self-attention** among video patches: each video patch attends to all others, building spatial and temporal context.
2. **Cross-attention** from video patches to text embeddings: each video patch attends to the text tokens, pulling in the semantic content of the text prompt.

The text embeddings (from a text encoder like T5) are the K and V source in cross-attention. They are fixed for a given prompt. The video patches are the Q source -- they "read" the text to figure out what content to generate.

This is why changing the text prompt changes the video so dramatically: the cross-attention weights shift entirely, redirecting each video patch to attend to different parts of the semantic description.

In the MLX port, you'd find this in the transformer block as two distinct attention sub-layers:

```python
# LTX-Video transformer block structure (simplified)
class TransformerBlock:
    def __call__(self, hidden_states, encoder_hidden_states, ...):
        # 1. Self-attention: video patches attend to each other
        attn_out = self.attn1(hidden_states, hidden_states, hidden_states)
        hidden_states = hidden_states + attn_out

        # 2. Cross-attention: video patches attend to text
        attn_out = self.attn2(
            query=hidden_states,          # Q from video
            key=encoder_hidden_states,    # K from text encoder
            value=encoder_hidden_states   # V from text encoder
        )
        hidden_states = hidden_states + attn_out

        # 3. Feed-forward
        hidden_states = hidden_states + self.ff(hidden_states)
        return hidden_states
```


## Key Terms

| Term | Definition |
|------|------------|
| **Attention** | A mechanism where each element in a sequence computes a weighted sum over all other elements, where the weights are learned based on content relevance |
| **[Self-Attention](../glossary.md#self-attention)** | Attention where Q, K, and V all come from the same sequence; each token attends to all tokens in the same sequence, including itself |
| **[Cross-Attention](../glossary.md#cross-attention)** | Attention where Q comes from one sequence and K, V come from a different sequence; used to let one modality guide another (e.g., text guiding video) |
| **Query (Q)** | The "what am I looking for?" vector; each token's query is matched against all other tokens' keys to produce attention scores |
| **Key (K)** | The "what do I contain?" vector; each token's key is matched against all queries, determining how much other tokens attend to it |
| **Value (V)** | The "what information do I provide?" vector; the actual content that is aggregated using the attention weights |
| **[Attention Score](../glossary.md#attention-score)** | The dot product of a query with a key, measuring relevance between two tokens; divided by sqrt(d_k) before softmax |
| **Scaled Dot-Product Attention** | The complete single-head attention operation: softmax(QK^T / sqrt(d_k)) V; the building block for multi-head attention |
| **[Multi-Head Attention](../glossary.md#multi-head-attention)** | Running H independent attention operations in parallel, each with different projections; the heads' outputs are concatenated and projected back to d_model |
| **[Attention Head](../glossary.md#attention-head)** | One of H parallel attention operations in multi-head attention, each with its own W_Q, W_K, W_V projection matrices |
| **[KV-Cache](../glossary.md#kv-cache)** | Stored K and V tensors from previous generation steps, enabling efficient autoregressive generation by avoiding recomputation |
| **Flash Attention** | An IO-efficient attention algorithm (Dao et al., 2022) that tiles the attention computation to avoid writing the full seq×seq matrix to GPU memory; in MLX: `mx.fast.scaled_dot_product_attention` |
| **Grouped-Query Attention (GQA)** | Multi-head attention variant where K and V heads are shared across groups of Q heads; reduces KV-cache memory with minimal quality loss |


## Sources

- Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems 30*. Introduces the transformer and defines scaled dot-product attention and multi-head attention. Available at [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." Introduces the tiled attention algorithm that avoids materializing the full attention matrix. Available at [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- MLX Documentation. `mx.fast.scaled_dot_product_attention`. Available at [ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
- Alammar, J. "The Illustrated [Transformer](../glossary.md#transformer)." Visual walkthrough of attention with clear diagrams of Q/K/V projections and the attention matrix. Available at [jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)


## See Also

- [Transformers](05-transformers.md) -- prerequisite; explains why attention exists (the RNN bottleneck, the "see the whole page" insight) and how attention fits inside the transformer block
- [GPU Computing](04-gpu-computing.md) -- explains why attention is expensive (the seq×seq matrix is memory-bandwidth-bound) and what Flash Attention is actually doing at the hardware level
- [Latent Space](07-latent-space.md) -- the next concept; attention operates in latent space, transforming token embeddings into contextually enriched representations
