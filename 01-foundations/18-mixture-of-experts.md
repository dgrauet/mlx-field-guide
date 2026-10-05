# Mixture of Experts

> **One-liner:** A Mixture-of-Experts (MoE) model replaces each feed-forward block with many smaller "expert" blocks and a **router** that sends each token to only a few of them -- so it has the knowledge (and memory footprint) of a big model but the per-token cost of a small one, which is a particularly good trade on a Mac with lots of unified memory.

## The Intuition

In a [transformer](05-transformers.md), most parameters live in the [feed-forward network](../glossary.md#feed-forward-network-ffn-mlp) of each block, and in a dense model every token goes through all of them. An MoE model splits that feed-forward block into, say, 128 independent experts and adds a tiny router in front. For each token, the router picks the 8 experts it considers most relevant; the other 120 are skipped for that token.

```
DENSE BLOCK                              MoE BLOCK

  token --> [ one big FFN ] --> out        token --> router: "experts 3, 17, 42 ... (top-8)"
                                                        |
            every weight used                           v
            for every token                  [e0] [e1] ... [e3] ... [e17] ... [e127]
                                                          \      |      /
                                                  weighted sum of the 8 chosen
                                                                 |
                                                                out

                                             all 128 experts stored,
                                             only 8 computed per token
```

The model's name usually tells you both sizes. **Qwen3-30B-A3B** has 30.5B parameters in total and 3.3B *activated* per token. That is the key to reading every MoE model card: **total parameters decide the memory you need; active parameters decide the speed.**

---

## How It Actually Works

### The router

The router (called `gate` in most checkpoints) is a single linear layer that produces one score per expert. The top-k scores pick the experts, and the scores become the weights of the final sum. The details differ between model families, and this is where ports go wrong:

```
ROUTING VARIANTS (all from the mlx-lm implementations)

  Mixtral            top-k on raw scores, then softmax over the k chosen
                       -> the k weights sum to 1

  Qwen MoE           softmax over ALL experts, then top-k;
                       renormalize only if config.norm_topk_prob is true
                       Qwen3-30B-A3B:  norm_topk_prob = true
                       Qwen1.5-MoE:    norm_topk_prob = false
                                       (weights sum to < 1 -- by design)

  DeepSeek-V3        SIGMOID scores (not softmax); a learned bias is added
                       for choosing experts but NOT for weighting them;
                       experts chosen within the best groups (n_group,
                       topk_group); weights renormalized, then multiplied
                       by routed_scaling_factor
```

The `norm_topk_prob` flag is a good example of a silent difference. In random tests with 128 experts and top-8, the chosen experts' softmax weights summed to only **0.55-0.68**. Skip the renormalization when the config asks for it (or add it when it doesn't) and every MoE layer's output is scaled wrong -- no crash, just a degraded model.

### Shared experts

Some families (DeepSeek, Qwen1.5/2-MoE, Llama 4) add one or more **shared experts**: an ordinary feed-forward block that every token goes through, added to the routed experts' output. It holds knowledge every token needs, so the routed experts can specialize. In checkpoints it appears as `shared_expert(s)` with its own size (e.g. `shared_expert_intermediate_size`), sometimes with its own sigmoid gate.

### Computing it efficiently: gather matmuls

A naive implementation loops over experts in Python and runs a matmul per expert. MLX has a primitive made for MoE: `mx.gather_mm` (and `mx.gather_qmm` for quantized weights) multiplies each token by *the weight matrix selected by an index*, from a single stacked tensor of shape `[num_experts, out, in]`. `mlx-lm` wraps it in `SwitchLinear` / `SwitchGLU`, and when many tokens are routed at once (≥ 64 routing decisions) it sorts them by expert first so memory access stays sequential.

This is why `mlx-lm` conversions **stack** the experts. A Hugging Face checkpoint stores `mlp.experts.0.up_proj.weight`, `mlp.experts.1.up_proj.weight`, ... as separate tensors; each model's `sanitize()` stacks them into `mlp.switch_mlp.up_proj.weight` with a leading expert dimension.

### Measured: big-model memory, small-model speed

One MoE layer with Qwen3-30B-A3B's dimensions (hidden 2048, 128 experts of 768, top-8), 4-bit, compared with two dense layers -- one with the same *active* size, one with the same *total* size (MLX 0.32.3, M2 Pro):

```
                                     weights   1 token    512 tokens
  MoE, 128 experts, top-8             379 MB   0.56 ms     19.4 ms
  dense, same ACTIVE size (8×768)      24 MB   0.82 ms     11.2 ms
  dense, same TOTAL size (128×768)    377 MB   2.26 ms    202.7 ms
```

- **Memory** is that of the total size: all 128 experts must be resident, because the next token may pick any of them.
- **Decode (1 token)** costs about the same as the small dense layer (both are sub-millisecond and dominated by fixed overheads; the order flips between runs) and 4-5x less than the large one: only 8 experts' weights are read.
- **Prefill (512 tokens)** costs about 1.7x the small dense layer -- different tokens pick different experts, so almost every expert gets read, plus routing overhead -- but still ~10x less than the equally large dense layer.

At the whole-model level, the [decode ceiling](16-kv-cache-inference.md) formula (bandwidth ÷ bytes read per token) uses the *active* bytes. `mlx-community/Qwen3-30B-A3B-4bit` is 17.2 GB on disk; reading ~3.3/30.5 of it per token is ~1.9 GB, so the ceiling on a 200 GB/s M2 Pro is ~100 tok/s, versus ~12 tok/s for a dense 30B model of the same file size.

---

## Why It Matters for MLX

### MoE suits Apple Silicon

On a CUDA GPU, MoE's weak point is memory: a 24 GB card can't hold a 17 GB model plus its KV cache comfortably, and splitting experts across GPUs adds communication. A Mac with 32-192 GB of unified memory holds all the experts in one pool, and decode only pays for the active ones. This is why MoE models are among the best-performing models to run locally with `mlx-lm` (`mlx_lm/models/` has 20+ MoE architectures: Mixtral, Qwen2/3-MoE, DeepSeek V2/V3, gpt-oss, Llama 4, OLMoE, GLM-4 MoE, ...).

Caveats:

- **Memory is sized by total parameters.** A "3B-active" model still needs its full 17 GB (4-bit) resident. Check the total, not the active count, against your RAM.
- **Batching should help less than for dense models** (reasoning, not measured here): different requests route to different experts, so a batch reads more distinct weights than one token does (see [KV Cache & Inference Optimization](16-kv-cache-inference.md) for batching on dense models).
- **Keep the router precise.** The `gate` layer is tiny and its precision decides which experts run. In `mlx-lm`, most MoE models define a `quant_predicate` for this; Qwen3-MoE's, for example, quantizes `mlp.gate` at 8 bits while the experts go to 4. Do the same in a new port.

### Port failures

```
MoE PORT FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  Coherent but noticeably dumber than the    routing weights wrong: missing or
    reference                                  extra top-k renormalization
                                               (norm_topk_prob), softmax vs
                                               sigmoid, missing routed_scaling_factor
  Output degraded, config mentions shared    shared expert (or its gate) not
    experts                                    loaded or not added
  Garbage from the first token               experts stacked in the wrong order
                                               (sort expert indices numerically:
                                               "experts.10" sorts before "experts.2"
                                               as a string)
  Very slow prefill                          Python loop over experts instead of
                                               gather_mm / SwitchGLU
  Weight-loading error on "experts.N"        checkpoint not sanitized: per-expert
                                               tensors not stacked into switch_mlp
  Only some layers look wrong                model mixes dense and MoE layers
                                               (decoder_sparse_step, mlp_only_layers,
                                               first_k_dense_replace) -- the dense
                                               ones use a normal MLP
```

Porting rules:

1. **Start from the closest `mlx-lm` MoE model.** The routing, `sanitize()` stacking, and `SwitchGLU` usage are already correct there; the per-family differences are a few lines in the gate.
2. **Read every routing field in the config** -- `num_experts`, `num_experts_per_tok`, `norm_topk_prob`, `scoring_func`, `routed_scaling_factor`, `n_group` / `topk_group`, shared-expert sizes, and which layers are sparse -- and match each one.
3. **Test the router separately.** Compare the selected expert indices and weights for a fixed input against the reference before comparing the layer output. A wrong index shows up there immediately; in the final output it's hidden in the noise.

---

## Key Terms

| Term | Definition |
|------|------------|
| **Mixture of Experts (MoE)** | Architecture where each feed-forward block is split into many experts and a router sends each token to only a few. |
| **Expert** | One of the parallel feed-forward blocks in an MoE layer; typically a small SwiGLU MLP. |
| **Router (gate)** | The linear layer that scores experts for each token; the top-k scores pick the experts and weight their outputs. |
| **Top-k routing** | Sending each token to its k highest-scoring experts (`num_experts_per_tok`). |
| **Active parameters** | Parameters actually used for one token (shared layers + the k chosen experts); decide speed. |
| **Total parameters** | All parameters including every expert; decide memory. |
| **Shared expert** | A feed-forward block that every token goes through in addition to its routed experts. |
| **`norm_topk_prob`** | Config flag saying whether the k chosen routing weights are renormalized to sum to 1. |
| **`gather_mm` / `gather_qmm`** | MLX primitives that multiply each input by the weight matrix selected by an index from a stacked tensor; the building block of MoE layers in MLX. |

---

## Sources

- Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." [arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)
- Fedus, W., Zoph, B., & Shazeer, N. (2021). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." [arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)
- Jiang, A. Q., et al. (2024). "Mixtral of Experts." [arxiv.org/abs/2401.04088](https://arxiv.org/abs/2401.04088)
- DeepSeek-AI (2024). "DeepSeek-V3 Technical Report" (sigmoid routing, auxiliary-loss-free bias, shared experts). [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- `mlx-lm` source: `mlx_lm/models/switch_layers.py` (`SwitchLinear`, `SwitchGLU`, expert sorting), `mixtral.py`, `qwen3_moe.py`, `deepseek_v3.py` (routing variants and `sanitize()` stacking): [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- Qwen3-30B-A3B and Qwen1.5-MoE-A2.7B `config.json` (routing fields): [huggingface.co/Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- Layer timings on this page: MLX 0.32.3 / `mlx-lm` 0.32.0 on an M2 Pro, one synthetic 4-bit layer with random weights; whole-model figures are derived from the bandwidth formula, not measured.

---

## See Also

- [Transformers](05-transformers.md) -- the feed-forward block that MoE replaces
- [Neural Networks](02-neural-networks.md) -- linear layers and activations inside each expert
- [KV Cache & Inference Optimization](16-kv-cache-inference.md) -- the bandwidth ceiling that makes active parameters the speed metric
- [Quantization](10-quantization.md) -- quantized experts (`gather_qmm`) and keeping the router precise
- [Porting Guide](../03-contributing/02-porting-guide.md) -- weight conversion and `sanitize()` in general
