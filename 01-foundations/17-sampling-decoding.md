# Sampling & Decoding

> **One-liner:** An LLM doesn't output a word, it outputs a score for every token in its vocabulary; **sampling** is the small, separate piece of code that turns those scores into one chosen token -- and because each library sets it up differently (defaults, order of filters, which settings it reads from the model), the same model can behave very differently in PyTorch and in MLX.

## The Intuition

At every step, the model's last layer produces one raw score per vocabulary entry -- the [logits](../glossary.md#logits) (262,208 of them for Gemma 3 12B). [Softmax](../glossary.md#softmax) turns them into probabilities. Then *something* has to pick one token, append it to the text, and run the model again.

That "something" is not part of the model. It has no weights and isn't in the checkpoint. It is a few lines of code in the inference library, configured by a handful of numbers: temperature, top-k, top-p, min-p, repetition penalty. Change them and the same weights go from repetitive and robotic to creative to incoherent.

```
ONE GENERATION STEP

  model  ->  logits (one score per token)
                |
                |  1. penalties      push down tokens already used
                |  2. temperature    sharpen or flatten the distribution
                |  3. filters        drop unlikely tokens (top-k / top-p / min-p)
                v
         probabilities over the remaining tokens
                |
                |  4. pick           argmax (greedy) or random draw (sampling)
                v
         one token  ->  appended, fed back to the model
```

The step order shown is conceptual. Libraries don't all use it, and as the rest of this page shows, the order changes the result.

---

## How It Actually Works

### Greedy: always take the top token

The simplest rule: pick the highest-scoring token (`argmax`). It is **deterministic**: same prompt, same output. That makes it the right choice for testing a port (see [Testing & Validation](../03-contributing/05-testing-validation.md)). But it is often a poor choice for real use. Greedy text tends to loop ("I think that I think that I think...") and to be bland, because the single most likely continuation is rarely the most useful one. Reasoning models are especially sensitive: Qwen3's model card, for example, advises against greedy decoding in thinking mode because it leads to endless repetition.

In `mlx-lm`, **`temp = 0` means greedy**, and it is the default.

### Temperature: sharpen or flatten

Temperature divides every logit before softmax: `softmax(logits / T)`.

```
Same 10 logits [4.0, 3.5, 3.0, 2.5, ...], top-3 probabilities:

  T = 0.6   [0.566, 0.246, 0.107]   sharper: the favorite dominates
  T = 1.0   [0.396, 0.240, 0.146]   the model's own distribution
  T = 1.5   [0.294, 0.211, 0.151]   flatter: long-shot tokens get a chance

  T -> 0 : becomes greedy        T -> ∞ : becomes uniform (random tokens)
```

Low temperature (0.2-0.7) for factual answers and code, around 1.0 for creative writing. Above ~1.3 the output usually degrades into nonsense, because the thousands of garbage tokens in the tail collectively get real probability mass.

### Truncation filters: cut the tail

Temperature alone can't stop a one-in-a-million token from being drawn once in a while, and over hundreds of tokens that happens. The filters remove the tail entirely before drawing:

```
  top-k   keep only the k highest-probability tokens (e.g. k = 20, 64)
          fixed count, whether the model is sure or not

  top-p   "nucleus": keep the smallest set of top tokens whose
          probabilities add up to p (e.g. 0.9, 0.95)
          adaptive: 1-2 tokens when the model is sure, hundreds when unsure

  min-p   keep tokens whose probability is at least min_p × (top token's
          probability), e.g. 0.05 or 0.1
          adaptive, and robust at high temperature
```

Filters stack: every enabled filter removes tokens, and the draw happens among the survivors.

### Penalties: discourage repetition

A **repetition penalty** (e.g. 1.05-1.2) makes already-used tokens less likely. The common implementation (from the CTRL paper, used by both `transformers` and `mlx-lm`) divides a positive logit by the penalty, or multiplies a negative one by it. Presence and frequency penalties (OpenAI-style) subtract a fixed amount, or an amount per occurrence. Penalties fight loops, but too strong and the model avoids words it legitimately needs ("the", a variable name in code).

### Seeds and reproducibility

Sampling draws random numbers. Fix the seed (`mx.random.seed(...)`, or `--seed` on the `mlx-lm` CLI) and the draws repeat -- on the same machine, same library version, same model. Across frameworks, a seed reproduces nothing: PyTorch and MLX use different random generators, so identical settings produce different samples. **Compare frameworks with greedy decoding, never with sampling.**

---

## Why It Matters for MLX

### Mismatch 1: `mlx-lm` ignores the model's recommended settings

Hugging Face checkpoints ship a `generation_config.json` holding the author's recommended sampling settings, which `transformers`' `generate()` applies by default:

```
  Qwen/Qwen2.5-7B-Instruct   do_sample, temperature 0.7, top_p 0.8, top_k 20,
                               repetition_penalty 1.05
  Qwen/Qwen3-8B              do_sample, temperature 0.6, top_p 0.95, top_k 20
```

`mlx-lm` reads that file **only for `eos_token_id`**. Its CLI, Python API and server all default to `temp = 0.0` (greedy), `top_p = 1.0`, `top_k = 0`, `min_p = 0.0`. So "the same model" in `transformers` samples with the author's settings, and in `mlx-lm` decodes greedily unless you pass them yourself:

```bash
mlx_lm.generate --model mlx-community/Qwen3-8B-4bit \
  --temp 0.6 --top-p 0.95 --top-k 20 --prompt "..."
```

In Python, build a sampler with `mlx_lm.sample_utils.make_sampler(temp=0.6, top_p=0.95, top_k=20)`, and penalties with `make_logits_processors(repetition_penalty=1.05)`. Note also that `mlx-community` conversions don't always carry the original sampling fields over (the `gemma-3-12b-it-4bit` conversion's `generation_config.json` has none), so check the original repository's file or model card.

### Mismatch 2: the order of temperature and filters

`transformers` applies **temperature first**, then top-k, top-p and min-p on the tempered distribution. `mlx-lm` applies top-p, min-p and top-k on the **untempered** log-probabilities, and temperature only at the final draw. `llama.cpp`'s default chain also puts temperature last.

At `T = 1` the two orders are identical. Anywhere else, the same `top_p` keeps a different set of candidates. Measured with `mlx-lm`'s own `apply_top_p` / `apply_min_p` on the 10-logit example above:

```
                      tokens kept by top_p = 0.9     tokens kept by min_p = 0.1
                      mlx-lm order    HF order        mlx-lm order    HF order
  T = 0.6                  5              3               5              3
  T = 1.0                  5              5               5              5
  T = 1.5                  5              7               5              7
```

Neither order is a bug; they are different definitions. But a `temperature 0.6, top_p 0.95` recipe tuned in `transformers` samples from a wider pool in `mlx-lm` (at T < 1) and a narrower one at T > 1. If you port a model *and* its recommended settings and outputs "feel" different, this is a likely cause.

### Mismatch 3: repetition-penalty window

`transformers` applies the repetition penalty to every token in the context (prompt + generated so far). `mlx-lm`'s `make_repetition_penalty` looks only at the **last 20 tokens** by default (`repetition_context_size`). Same penalty value, different strength.

### Symptoms and causes

```
SAMPLING-RELATED "PORT BUGS" (the model is fine, the decoding differs)

  Symptom                                    Likely cause
  -------                                    ------------
  MLX output loops / repeats, reference      mlx-lm defaults to greedy; the
    doesn't                                    reference samples with the
                                               generation_config.json settings
  Reasoning model never finishes thinking    greedy decoding on a model whose
                                               card requires sampling
  Output "feels" different with the same     temperature/filter order differs
    temp and top_p                             (HF: temp first; mlx-lm: last)
  Same seed, different text vs PyTorch       expected: different RNGs; compare
                                               greedily instead
  Greedy output matches for N tokens,        tiny numerical differences
    then diverges                              (bf16 vs fp32, fused kernels)
                                               flipping a near-tie in argmax;
                                               compare logits, not text
  Repetition penalty seems weaker in MLX     20-token window vs full context
```

### Porting rules

1. **Validate with greedy, deploy with sampling.** Compare logits (or greedy tokens) against the reference to prove the port. Then configure sampling explicitly for real use.
2. **Copy the sampling settings explicitly.** Read the original `generation_config.json` or model card, and pass every value to `mlx-lm`. Never rely on defaults matching.
3. **When greedy outputs diverge after many tokens, check the logits at the divergence point.** If the top two logits are within ~1e-2, it is a numerical near-tie, not a bug (see [Numerical Stability](12-numerical-stability.md)).

---

## Key Terms

| Term | Definition |
|------|------------|
| **Sampling** | Choosing the next token from the model's probability distribution, as opposed to always taking the most likely one. |
| **Greedy decoding** | Always picking the highest-probability token (argmax); deterministic; `temp = 0` in mlx-lm. |
| **Temperature** | Divides the logits before softmax; below 1 sharpens the distribution, above 1 flattens it. |
| **Top-k** | Keep only the k most likely tokens before drawing. |
| **Top-p (nucleus sampling)** | Keep the smallest set of most likely tokens whose probabilities sum to p. |
| **Min-p** | Keep tokens whose probability is at least min_p times the top token's probability. |
| **Repetition penalty** | Lowers the logits of tokens already in the context to discourage loops; mlx-lm looks at the last 20 tokens by default. |
| **generation_config.json** | File in a Hugging Face checkpoint with the author's default generation settings; `transformers` applies them, `mlx-lm` reads only the EOS token from it. |
| **Seed** | Initial state of the random generator; makes sampling repeatable within one framework, never across frameworks. |

---

## Sources

- `mlx-lm` source: `mlx_lm/sample_utils.py` (`make_sampler`, `apply_top_p`, `apply_min_p`, `make_repetition_penalty`), `mlx_lm/generate.py` (CLI defaults), `mlx_lm/utils.py` (`load_config` reads only `eos_token_id` from `generation_config.json`): [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- Hugging Face `transformers` source: `generation/utils.py` (processor order: repetition penalty, then temperature, top-k, top-p, min-p) and `generation/logits_process.py`: [github.com/huggingface/transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/generation)
- `llama.cpp` default sampler chain (`common/common.h`, temperature last): [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- Holtzman, A., et al. (2020). "The Curious Case of Neural Text Degeneration" (nucleus / top-p sampling, why greedy loops). [arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751)
- Keskar, N. S., et al. (2019). "CTRL: A Conditional Transformer Language Model for Controllable Generation" (repetition penalty). [arxiv.org/abs/1909.05858](https://arxiv.org/abs/1909.05858)
- Nguyen, M., et al. (2024). "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs." [arxiv.org/abs/2407.01082](https://arxiv.org/abs/2407.01082)
- Qwen3-8B model card and `generation_config.json` (recommended settings, no greedy in thinking mode): [huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- Filter-order measurements on this page: `mlx-lm` 0.31.3 with MLX 0.32.2, using its own `apply_top_p` / `apply_min_p` on a toy distribution.

---

## See Also

- [Loss Functions](13-loss-functions.md) -- why the model outputs logits and not probabilities
- [Attention](06-attention.md) -- the other place softmax appears
- [KV Cache & Inference Optimization](16-kv-cache-inference.md) -- the generation loop that sampling sits inside
- [Agents & Tool Use](../02-ecosystem/12-agents-tool-use.md) -- constrained generation: masking logits so only valid tokens can be sampled
- [Testing & Validation](../03-contributing/05-testing-validation.md) -- greedy decoding as the reference comparison for a port
