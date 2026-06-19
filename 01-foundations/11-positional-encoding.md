# Positional Encoding

> **One-liner:** [Attention](../glossary.md#attention) is blind to order, so positional encoding injects "where is this token" information into the model -- and the modern answer, [RoPE](../glossary.md#rope), does it by *rotating* each query and key by an angle proportional to its position.

## The Intuition

Attention computes a weighted sum over all tokens. Look closely at that computation and you'll notice something: it treats the input as a **set**, not a sequence. Shuffle the tokens and the attention math produces the same outputs in shuffled order -- nothing in the dot-product of a query and a key knows that one word came before another.

But order is meaning. "The dog bit the man" and "the man bit the dog" use identical tokens. A model that cannot tell positions apart cannot tell these sentences apart. **Positional encoding** is how we hand the model the information that attention throws away: the position of each token in the sequence.

The naive fix -- "just add a position number to each token" -- fails. Raw integers grow without bound, dominate the embedding values, and don't generalize past the lengths seen in training. The history of positional encoding is the history of doing this *well*.

---

## Why Attention Needs It: A Closer Look

Recall the core attention operation: a token's query vector `q` is compared with every key vector `k` via a dot product `q · k`. That score depends only on the *contents* of the two vectors, not on where they sit. Two identical tokens at positions 2 and 50 produce identical keys, hence identical scores. The model literally cannot distinguish them.

```
THE PROBLEM

  tokens:    "the"   "cat"   "sat"
  positions:   0       1       2

  Without position info, attention sees:
    { "the", "cat", "sat" }   <- a SET, order lost

  "cat sat the" would produce the same internal representation.
  Language is not a set. We must put the order back in.
```

So somewhere before (or inside) attention, position must be encoded into the vectors. There are three generations of how.

---

## How It Actually Works

### Generation 1: Sinusoidal (the original transformer)

The 2017 transformer added a fixed pattern of sines and cosines to each token [embedding](../glossary.md#embedding). Each dimension oscillates at a different frequency, so every position gets a unique multi-frequency "fingerprint":

```
SINUSOIDAL ENCODING (added to token embeddings)

  position 0:  [ sin(0/f0), cos(0/f0), sin(0/f1), cos(0/f1), ... ]
  position 1:  [ sin(1/f0), cos(1/f0), sin(1/f1), cos(1/f1), ... ]
  position 2:  [ sin(2/f0), ... ]

  - Low-frequency dims change slowly  -> encode coarse position
  - High-frequency dims change fast    -> encode fine position
  - Fixed (not learned): generalizes to unseen lengths in principle
```

It works, but the position signal is *added* to content and the two interfere; relative position (what attention actually cares about) is only implicit.

### Generation 2: Learned Absolute

BERT and GPT-2 replaced the fixed pattern with a **learned** position embedding table -- one trainable vector per position, added to the token embedding. Simple and effective, but with a hard ceiling: a table of size 512 *cannot* represent position 513. The model has no notion of positions it never trained on.

### Generation 3: RoPE (Rotary Position Embedding)

This is what LLaMA, Qwen, Mistral, and essentially every modern open *decoder* LLM use (encoder-only and some other architectures still use learned absolute, relative bias, or ALiBi). The insight is elegant: instead of *adding* a position signal, **rotate** the query and key vectors by an angle proportional to their position.

```
ROTARY POSITION EMBEDDING (RoPE)

  Treat each pair of dimensions i as a 2D point. Rotate it by angle
  m · θ_i, where m = the token's position and θ_i = base^(-2i/d) is
  that pair's own frequency.

  query at position m:   q  ->  R(m·θ) · q
  key   at position n:   k  ->  R(n·θ) · k

  Then the attention score q' · k' depends on the RELATIVE
  rotation (m - n)·θ -- exactly the relative position.

  Same query vector, rotated more the later its position:

      position 0:   q  (no rotation)
      position 1:   q  rotated by 1·θ
      position 3:   q  rotated by 3·θ
```

Two properties fall out of this for free, and they are why RoPE won:

1. **It encodes *relative* position.** Because rotating `q` by `m·θ` and `k` by `n·θ` makes their dot product a function of `m − n`, the model naturally sees "how far apart" two tokens are -- which is what language structure depends on.
2. **It extrapolates.** Rotation is defined for any angle, so there's no position table to run off the end of. Long-context tricks stretch a model trained on 4k tokens to 32k+ without retraining from scratch, via two distinct mechanisms: **Position Interpolation** scales the *positions* down (feed `m/k` instead of `m`), while **NTK scaling** and **YaRN** adjust the rotation *frequencies* themselves. MLX exposes these as separate knobs (`scale` for the former, `freqs` for the latter) -- they are not the same lever.

RoPE is applied **inside attention**, to `q` and `k` only (not to values), at every layer -- not once at the input like the older schemes.

---

## Why It Matters for MLX

Positional encoding is one of the **highest-frequency sources of silent port bugs**, because RoPE has many small implementation details that each produce running-but-wrong output:

```
ROPE PORTING PITFALLS (running code, wrong numbers)

  Symptom                          Likely cause
  -------                          ------------
  Coherent for short prompts,      Wrong base frequency (theta), or scaling
    degrades as context grows        factor for long-context not applied
  Garbage from token 1             Dimension pairing wrong: interleaved
                                     (x0,x1),(x2,x3) vs split-half (x0,x_{d/2})
  Off by a constant rotation       Position index off-by-one / wrong start
  Works at fp32, breaks at fp16    Computing angles in low precision
```

The **dimension-pairing convention** is the classic trap. Implementations differ in whether they rotate *interleaved* pairs `(x₀,x₁), (x₂,x₃), …` or *split-half* pairs `(x₀, x_{d/2}), (x₁, x_{d/2+1}), …`. Pick the wrong one and the weights load fine, shapes match, and the output is nonsense. The reference and your MLX port must agree exactly.

MLX provides this as a fast, fused primitive: **`mlx.fast.rope(x, dims, *, traditional, base, scale, offset, freqs=None)`**. Use it rather than hand-rolling the rotation -- it is correct, fast, and exposes precisely the knobs that matter for matching a reference:

```
mlx.fast.rope -- the knobs porters actually get wrong

  traditional   layout convention, the #1 trap:
                  True  = interleaved pairs (original Meta/RoFormer, GPT-NeoX)
                  False = split-half pairs (HuggingFace LLaMA's rotate_half)
                Match whichever your reference uses, or output is garbage.
  offset        starting position index -- this is the KV-cache / generation
                position. Getting it wrong is the "off-by-one start" pitfall.
  base          the θ base frequency (commonly 10000); set per architecture.
  scale         scales positions (m -> m/scale) -- this is Position
                Interpolation for context extension.
  freqs         custom per-pair frequencies (how NTK/YaRN scaling is wired).
                Mutually exclusive with `base` -- pass exactly one.
  dims          how many dims to rotate; may be < head_dim (partial RoPE),
                the rest pass through unrotated.
```

mlx-lm's model definitions are the canonical examples of wiring this up per architecture. The two parameters newcomers overlook -- `offset` and the `traditional` mapping -- are exactly the two the pitfall table above warns about.

The verification discipline: confirm RoPE numerically against the reference for fixed `q`, `k`, and positions *before* trusting any generated text. A model with wrong RoPE still produces fluent output -- it's just wrong.

---

## Key Terms

| Term | Definition |
|------|------------|
| **Positional Encoding** | Any scheme that injects token-position information into a model, compensating for attention's order-blindness. |
| **Sinusoidal Encoding** | The original transformer's fixed sine/cosine position pattern added to embeddings. |
| **Learned Absolute** | A trainable per-position embedding table (BERT, GPT-2); cannot exceed its trained length. |
| **RoPE** | Rotary Position Embedding; rotates query/key vectors by a position-dependent angle, encoding relative position and extrapolating to longer contexts. |
| **Relative Position** | The distance `m − n` between two tokens; what attention structurally depends on, and what RoPE encodes directly. |
| **NTK scaling / YaRN** | Methods that rescale RoPE's frequencies to extend a model's usable context length beyond its training length. |
| **Dimension pairing** | The convention (interleaved vs split-half) for which vector dimensions RoPE rotates together; a common port bug. |

---

## Sources

- Vaswani, A., et al. (2017). "Attention Is All You Need." Introduces sinusoidal positional encoding. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." The RoPE paper. [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- Chen, S., et al. (2023). "Extending Context Window of Large Language Models via Positional Interpolation." [arxiv.org/abs/2306.15595](https://arxiv.org/abs/2306.15595)
- Peng, B., et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." [arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)
- MLX `mlx.fast.rope` documentation: [ml-explore.github.io/mlx/build/html/python/fast.html](https://ml-explore.github.io/mlx/build/html/python/fast.html)

---

## See Also

- [Attention](06-attention.md) -- prerequisite; positional encoding exists to fix attention's order-blindness, and RoPE lives inside the attention computation
- [Transformers](05-transformers.md) -- where positional encoding sits in the overall architecture; the LLaMA conventions (pre-norm, RoPE, SwiGLU) introduced there
- [Tensors](01-tensors.md) -- the rotation operates on dimension pairs of the query/key tensors; shape and layout intuition applies
- [LLMs](../02-ecosystem/03-llms.md) -- long-context behavior (32k, 128k) is governed by RoPE scaling; the ecosystem context for why this matters
- [Porting Guide](../03-contributing/02-porting-guide.md) -- RoPE is a recurring entry in the Common Port Failure Modes; the dimension-pairing trap in practice
