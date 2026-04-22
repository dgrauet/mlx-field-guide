# Training / Fine-tuning: CUDA vs MLX

> **Status:** 🟢 [CUDA](../glossary.md#cuda) (Mature -- full training stack, distributed, every scale) | 🟡 MLX (Limited -- LoRA/QLoRA for LLMs; no distributed, no full pretraining at scale)

## What This Domain Covers

[Training](../glossary.md#training) a model from random initialization, and fine-tuning a pre-trained model to a new task or dataset. This includes the full pipeline: data loading and preprocessing, training loop infrastructure, optimizer algorithms, gradient management, experiment tracking, and checkpoint handling. It covers both parameter-efficient methods (LoRA, QLoRA) and full-parameter training.

Training is where the gap between CUDA and MLX is largest. MLX's inference story is strong. MLX's training story -- specifically anything beyond fine-tuning a single LLM on a single Mac -- is limited by the absence of distributed training, mature gradient checkpointing, mixed-precision infrastructure, and the broad ecosystem of tools that CUDA has accumulated over a decade of large-scale training.

This page answers: what does each ecosystem provide for training? Where does the gap matter? What is possible today versus what requires waiting for the ecosystem to mature?

---

## The CUDA Ecosystem

### Full Training: Scale from One GPU to Thousands

CUDA's training ecosystem handles every scale point, from a single RTX 4090 to thousands of H100s in a supercluster.

**PyTorch** provides the foundation: autograd, optimizers, `DataLoader`, gradient clipping, and `torch.compile` for kernel fusion. Every training framework in the CUDA ecosystem is built on top of PyTorch.

**FSDP (FullyShardedDataParallel)** is PyTorch's built-in distributed training strategy. It shards model parameters, gradients, and optimizer states across GPUs, enabling training of models that do not fit in a single [GPU](../glossary.md#gpu)'s [VRAM](../glossary.md#vram). Each GPU holds a shard; during the forward pass, shards are all-gathered on demand, used, and immediately discarded.

```
FSDP MEMORY MODEL

  Without FSDP (DDP):
    Each GPU holds: full model params + full gradients + full optimizer states
    Memory per GPU = params + grads + opt_states  (3-4x model size)
    8x RTX 4090 (8 * 24GB = 192GB) -> max ~48GB model in FP16

  With FSDP (ZeRO-3 equivalent):
    Each GPU holds: 1/N of params + 1/N of grads + 1/N of optimizer states
    Memory per GPU = (params + grads + opt_states) / N
    8x RTX 4090 -> effectively pools 192GB -> much larger models feasible
    Communication overhead: all-gather before each layer's forward/backward

  FSDP is the PyTorch-native distributed training strategy since PyTorch 1.12.
  Replaced DistributedDataParallel (DDP) for large models.
```

**DeepSpeed** (Microsoft) is the dominant framework for large-scale distributed training. Its ZeRO optimizer (Zero Redundancy [Optimizer](../glossary.md#optimizer)) has three stages:

```
DEEPSPEED ZERO STAGES

  ZeRO-1:  Partition optimizer states across GPUs.
           Gradient and parameter replication still full.
           Memory: ~1/N of optimizer states per GPU.

  ZeRO-2:  Partition optimizer states + gradients.
           Parameter replication still full.
           Memory: ~1/N of optimizer states + gradients per GPU.

  ZeRO-3:  Partition optimizer states + gradients + parameters.
           Full model sharded across all GPUs.
           Memory: ~1/N of everything per GPU.
           Communication: gather params before each layer (like FSDP).

  ZeRO-Infinity:  ZeRO-3 + offload to CPU RAM and NVMe SSD.
                  Enables 1T+ parameter models on commodity hardware.
                  Very slow (NVMe bandwidth), but possible.

  In practice: ZeRO-2 is the most common choice.
               Low communication overhead. Most memory savings.
```

**Megatron-LM** (NVIDIA) implements tensor parallelism and pipeline parallelism -- strategies that split the model differently than ZeRO:

```
PARALLELISM STRATEGIES

  Data Parallelism (DDP, FSDP):
    Each GPU has the full model (or shard of it).
    Different data batches on each GPU.
    Gradients averaged across GPUs.
    Good when model fits in GPU memory.

  Tensor Parallelism (Megatron-LM):
    Each linear layer split across GPUs column-wise or row-wise.
    All GPUs cooperate on every forward pass (high bandwidth needed).
    Good for very large individual layers (wide feedforward blocks).

  Pipeline Parallelism (Megatron-LM, GPipe):
    Model layers split in stages across GPUs.
    GPU 1 runs layers 0-12, GPU 2 runs layers 13-24, etc.
    Micro-batches pipelined to keep all stages busy.
    Communication at stage boundaries only.

  Expert Parallelism (for MoE models):
    Each GPU holds different experts.
    Routing decides which GPU processes each token.

  Hybrid: Megatron-LM + DeepSpeed combine all four.
          Llama 3 405B was trained with tensor + pipeline + data parallel.
```

### Fine-tuning: Parameter-Efficient Methods

**PEFT (Hugging Face)** is the canonical fine-tuning library. It implements LoRA, QLoRA, IA3, prompt tuning, prefix tuning, and other adapter methods. All work by adding a small number of trainable parameters on top of a frozen (or quantized) base model.

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=64,                     # rank: controls adapter size
    lora_alpha=16,            # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 3,407,872 || all params: 3,215,310,848 || trainable%: 0.1060
```

**QLoRA** combines 4-bit NF4 quantization of the base model with LoRA adapters trained in 16-bit. This halves the memory needed for the base model, enabling fine-tuning of much larger models on single GPUs:

```
QLORA MEMORY COMPARISON (Llama 3 70B)

  Approach               GPU Memory Required
  --------               ------------------
  Full fine-tuning FP16  ~280 GB  (4x A100 80GB minimum)
  LoRA FP16              ~160 GB  (2x A100 80GB)
  QLoRA 4-bit + LoRA     ~45 GB   (1x A100 80GB -- fits!)
                                  (1x RTX 4090 24GB -- does NOT fit)

  For 7B model:
  Full fine-tuning FP16  ~28 GB   (1x A100 40GB)
  QLoRA 4-bit + LoRA     ~6 GB    (1x RTX 3080 10GB -- fits!)
```

**Axolotl** wraps PEFT and Transformers with a complete training pipeline: YAML configuration, dataset preprocessing, multi-format data support (ShareGPT, Alpaca, completion), evaluation integration, and checkpoint management. The standard tool for instruction tuning and domain fine-tuning.

**Unsloth** achieves 2x faster LoRA training than vanilla PEFT by writing Triton kernels that fuse the LoRA backward pass operations. It also implements 4-bit quantization-aware LoRA without the BitsAndBytes library dependency.

```
FINE-TUNING TOOL COMPARISON (CUDA)

  Tool          Purpose                       Notes
  ----          -------                       -----
  PEFT          LoRA/QLoRA/adapter library    Canonical; used by everything else
  Axolotl       Full training pipeline        Data formatting, configs, eval
  Unsloth       Speed-optimized LoRA          2x faster, same accuracy
  LLaMA-Factory Multi-method fine-tuning UI   Browser interface available
  TRL (HF)      RLHF/PPO/DPO training         Alignment training; builds on PEFT
  Torchtune     PyTorch-native fine-tuning     Newer; simpler than Axolotl
```

### Dataset Tools

**Hugging Face Datasets** is the standard library for ML datasets. Streaming from disk or cloud without loading into RAM, automatic caching, built-in preprocessing transforms, and thousands of pre-loaded datasets on the Hub. Both [CPU](../glossary.md#cpu) preprocessing and GPU-side augmentations are supported.

**WebDataset** handles web-scale image/video/text datasets stored as tar shards. Supports streaming from cloud storage (S3, GCS) with multi-worker prefetching. Used for training image generation models at scale.

**TFDS (TensorFlow Datasets)** is cross-framework compatible and provides another large library of preprocessed datasets.

### Experiment Tracking

A mature training run needs to log losses, gradients, learning rate curves, and evaluation metrics in real time, and compare experiments.

```
CUDA EXPERIMENT TRACKING ECOSYSTEM

  Weights & Biases (W&B):
    - Automatic metric logging, artifact versioning
    - Hyperparameter sweep integration (Bayesian optimization)
    - Team collaboration, experiment comparison dashboards
    - Industry standard for ML research teams

  MLflow:
    - Open-source, self-hostable alternative to W&B
    - Metric logging, model registry, experiment comparison
    - Strong enterprise adoption

  TensorBoard:
    - Google's original experiment tracker; ships with TensorFlow
    - PyTorch natively supports TensorBoard via SummaryWriter
    - Scalar plots, histograms, embedding visualizers

  Comet ML, Neptune AI:
    - Additional W&B alternatives with different pricing models
```

---

## The MLX Ecosystem

### Core Training Primitives

MLX provides the foundational primitives needed for training: functional autograd, optimizers, and a module system. These work correctly and are sufficient for small-scale training from scratch on Apple Silicon.

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))

model = MLP(784, 256, 10)
optimizer = optim.Adam(learning_rate=1e-3)

def loss_fn(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y).mean()

loss_and_grad = nn.value_and_grad(model, loss_fn)

# Training step
x_batch = mx.random.normal((32, 784))
y_batch = mx.zeros((32,), dtype=mx.int32)

loss, grads = loss_and_grad(model, x_batch, y_batch)
optimizer.update(model, grads)
# Force computation: mx.eval materializes the lazy graph
mx.eval(model.parameters(), optimizer.state)
```

The pattern -- `nn.value_and_grad` to get both the loss value and parameter gradients, then `optimizer.update` to apply them, then `mx.eval` to force computation -- is the core training loop for all MLX training.

### Available Optimizers

MLX ships with a standard set of optimizers in `mlx.optimizers`:

```
MLX OPTIMIZERS (mlx.optimizers)

  SGD               -- stochastic gradient descent with optional momentum
  Adam              -- adaptive moment estimation (Kingma & Ba, 2014)
  AdamW             -- Adam with decoupled weight decay
  Adagrad           -- adaptive learning rates per parameter
  RMSprop           -- root mean square propagation
  Adamax            -- Adam variant with L-infinity norm
  Lion              -- EvoLved Sign Momentum (newer, memory efficient)
  Adafactor         -- memory-efficient adaptive optimizer
  Adan              -- Adaptive Nesterov Momentum Algorithm

  Learning rate schedules:
    cosine_decay
    linear_schedule
    join_schedules    -- chain multiple schedules
    warmup            -- linear warmup wrapper
```

This covers the optimizers needed for most fine-tuning and small-scale training.

### mlx-lm: LoRA and QLoRA

mlx-lm is the most complete MLX training implementation. It covers parameter-efficient fine-tuning of language models with LoRA and QLoRA.

```python
# LoRA fine-tuning via command line (most common workflow)
#
# mlx_lm.lora \
#   --model mlx-community/Llama-3.2-3B-Instruct-4bit \
#   --train \
#   --data ./data \
#   --batch-size 4 \
#   --num-layers 16 \
#   --iters 1000 \
#   --learning-rate 1e-4 \
#   --adapter-path ./adapters

# Data format: JSONL with "text" field (completion) or
# {"messages": [...]} (chat format)
```

**What mlx-lm LoRA covers:**
- LoRA adapters on attention projection layers (q, k, v, o projections)
- QLoRA: 4-bit quantized base model with FP16/BF16 LoRA adapters
- DoRA: [Weight](../glossary.md#weight) decomposition LoRA (experimental)
- [Gradient](../glossary.md#gradient) accumulation via the `--grad-checkpoint` flag (limited)
- Adapter merging: merge adapters back into base model weights

**What mlx-lm LoRA does not cover:**
- Multi-GPU distributed training
- Mixed-precision training (BF16 training is the default; FP8 does not exist)
- Full parameter fine-tuning at scale
- DPO/PPO/RLHF training
- Custom data collation beyond built-in formats
- Integration with W&B, MLflow, or TensorBoard out of the box

### Gradient Checkpointing in MLX

Gradient checkpointing trades compute for memory: instead of storing all intermediate activations for the backward pass, they are recomputed during backprop. This allows training deeper models or larger batches at the cost of ~30% more compute.

MLX has basic support via `mlx.nn.utils.checkpoint`, but it is less mature than PyTorch's `torch.utils.checkpoint`:

```python
import mlx.nn as nn

# Basic checkpointing in MLX (experimental)
# Recomputes the function during backward pass instead of caching output
checkpointed_layer = nn.checkpoint(transformer_layer)
```

PyTorch's equivalent is more featureful: activation checkpointing policies, selective checkpointing of specific layers, and tight integration with FSDP.

### Training Examples in mlx-examples

The official mlx-examples repository includes training scripts beyond mlx-lm:

```
MLX TRAINING EXAMPLES (mlx-examples repository)

  lora/           -- LoRA fine-tuning (now part of mlx-lm)
  mnist/          -- MNIST classification from scratch
  cifar/          -- CIFAR-10 with CNN
  bert/           -- BERT pretraining (small scale)
  transformer_lm/ -- GPT-style language model from scratch
  stable_diffusion/ -- Stable Diffusion (inference; no training)

  Note: examples are for illustration, not production pipelines.
  For actual LLM fine-tuning, use mlx-lm directly.
```

### Experiment Tracking: What Exists

MLX has minimal built-in experiment tracking. The common approach is manual:

```python
import mlx.core as mx
import json

# Typical MLX training logging (manual)
log = []
for step, (x, y) in enumerate(data_loader):
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    if step % 100 == 0:
        log.append({"step": step, "loss": loss.item()})
        print(f"Step {step}: loss={loss.item():.4f}")

# Save log to JSON
with open("train_log.json", "w") as f:
    json.dump(log, f)
```

W&B and MLflow are Python libraries and technically work with MLX, but there is no built-in integration -- you add `wandb.log({"loss": loss.item()})` manually. There is no automatic gradient histogram logging or model graph visualization.

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **Full pretraining** | 🟢 PyTorch + FSDP/DeepSpeed; multi-node, trillion-parameter scale | 🟡 Possible for small models on single Mac; no distributed; no gradient checkpointing at scale | Large -- MLX cannot pretrain anything beyond toy models in reasonable time |
| **LoRA fine-tuning (LLMs)** | 🟢 PEFT: full-featured, all architectures, optimized kernels via Unsloth | 🟡 mlx-lm: works well for supported architectures; fewer optimization hooks | Small for basic use; medium for edge cases (very long context, custom architectures) |
| **QLoRA fine-tuning** | 🟢 PEFT + BitsAndBytes NF4; stable, well-documented | 🟡 mlx-lm QLoRA: works; fewer configuration options | Small -- both work; CUDA ecosystem has more tooling around it |
| **Full-parameter fine-tuning** | 🟢 Standard with Adam + gradient checkpointing; 7B on single A100 40GB | 🟡 Works for small models; impractical beyond ~1-3B parameters on Mac | Medium -- Mac Studio 192GB can hold large models, but no gradient checkpointing, no FSDP |
| **Distributed training** | 🟢 FSDP, DeepSpeed ZeRO, NCCL; linear scaling to thousands of GPUs | 🔴 Not supported -- single-device only | Large -- fundamental architecture gap; no path to multi-device MLX training |
| **Mixed precision training** | 🟢 BF16/FP16 training with FP32 optimizer states; FP8 on H100; automatic loss scaling | 🟡 BF16 training available; no FP8; no automatic loss scaler | Medium -- BF16 covers most needs; FP8 for maximum throughput is missing |
| **Gradient accumulation** | 🟢 Standard: accumulate over N micro-batches before optimizer step | 🟡 Manual accumulation possible; not a first-class API | Small -- implementable manually; not ergonomic |
| **Gradient checkpointing** | 🟢 `torch.utils.checkpoint`; selective policies; FSDP integration | 🟡 `mlx.nn.utils.checkpoint` exists; limited, experimental | Medium -- important for training larger models on constrained memory |
| **DPO / RLHF / alignment training** | 🟢 TRL (Hugging Face): PPO, DPO, GRPO, reward modeling; widely used | 🔴 No equivalent; would require building from scratch in MLX | Large for alignment work; irrelevant for standard fine-tuning |
| **Optimizer variety** | 🟢 All standard optimizers + LAMB, Sophia, Muon; optimizer state sharding | 🟡 Standard optimizers available; no LAMB or experimental optimizers | Small for most use cases |
| **Experiment tracking** | 🟢 W&B, MLflow, TensorBoard; native integrations in Transformers, Axolotl | 🟡 Manual integration only; no official W&B/MLflow hooks | Medium -- works with manual logging; not ergonomic |
| **[Hyperparameter](../glossary.md#hyperparameter) search** | 🟢 W&B Sweeps, Optuna, Ray Tune; distributed search across GPUs | 🟡 Python libraries work; no MLX-aware distributed search | Small -- single-machine search works; distributed HPO not possible |
| **Dataset infrastructure** | 🟢 Hugging Face Datasets: streaming, caching, maps, multi-worker; WebDataset for sharded data | 🟡 No native MLX data loader; use PyTorch DataLoader + numpy conversion or manual | Medium -- workaround works; not ergonomic; multi-worker prefetch awkward |
| **Training pipeline tools** | 🟢 Axolotl, LLaMA-Factory, Torchtune: YAML configs, dataset formatting, eval hooks | 🟡 mlx-lm covers LLM fine-tuning; nothing comparable for other domains | Large for non-LLM training (vision, audio, custom architectures) |
| **[Checkpoint](../glossary.md#checkpoint) management** | 🟢 Transformers/Axolotl: automatic checkpointing, resumable training, safetensors | 🟡 mlx-lm saves adapters; full checkpoint management is manual | Medium -- saving works; resumable training with optimizer state is not ergonomic |
| **[Model](../glossary.md#model) parallelism** | 🟢 Megatron-LM tensor + pipeline parallelism; training models too large for any single GPU | 🔴 Not supported | Large for frontier models; irrelevant for fine-tuning on consumer hardware |
| **Multi-node training** | 🟢 Standard: PyTorch distributed, NCCL, InfiniBand; scales linearly | 🔴 Not supported | Large for large-scale pretraining; irrelevant for single-Mac use |
| **Training throughput (tokens/sec)** | 🟢 H100: ~15,000 tok/s for 7B training; RTX 4090: ~2,000 tok/s | 🟡 M3 Max: ~200-400 tok/s for 7B training; M2 Ultra: ~400-600 tok/s | Large raw throughput gap; Apple Silicon is ~5-10x slower for training vs RTX 4090 |

---

## Contribution Opportunities

**Distributed training via Apple Thunderbolt/networking.** The most impactful missing feature. MLX's single-device constraint means a Mac Studio (192 GB) is the ceiling. Implementing data-parallel training across multiple Macs via NCCL-style collective communication (all-reduce, broadcast) would let users scale beyond a single device. Apple's [Metal](../glossary.md#metal) does not have a native collective communication library; this would require building one.

**PyTorch DataLoader bridge with zero-copy.** Currently, the cleanest way to load data for MLX training is: PyTorch DataLoader (for multi-worker prefetch, augmentation) -> numpy arrays -> `mx.array(batch)`. The numpy conversion copies data. A first-class bridge that uses shared memory or avoids copies would make data loading faster and more ergonomic.

**W&B / MLflow native integration for mlx-lm.** Adding `--report-to wandb` support to mlx-lm's training CLI (mirroring Transformers' `--report_to` argument) would make experiment tracking a one-line addition rather than manual `wandb.log()` calls.

**DPO training in mlx-lm.** Direct Preference Optimization is the current standard for instruction following and alignment. Hugging Face's TRL library makes DPO straightforward for CUDA. An MLX-native DPO implementation in mlx-lm would let Mac users align fine-tuned models without a CUDA machine.

**Axolotl MLX backend.** Axolotl abstracts training configuration into YAML files and handles data formatting, multi-dataset mixing, evaluation, and checkpointing. Adding an MLX backend to Axolotl (analogous to how it supports PyTorch + PEFT) would give MLX users the same ergonomic fine-tuning pipeline without reinventing it.

**Gradient checkpointing improvements.** Stabilizing `mlx.nn.utils.checkpoint`, adding selective checkpointing policies, and documenting it properly would enable training larger models on smaller-memory Macs.

---

## Sources

- Hugging Face PEFT: [github.com/huggingface/peft](https://github.com/huggingface/peft). LoRA, QLoRA, and adapter methods for PyTorch.
- DeepSpeed: [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed). ZeRO optimizer documentation and usage.
- Axolotl: [github.com/axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl). LLM fine-tuning pipeline.
- Unsloth: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth). Triton-optimized LoRA training.
- TRL (Transformers Reinforcement Learning): [github.com/huggingface/trl](https://github.com/huggingface/trl). DPO, PPO, and GRPO training.
- LoRA paper (Hu et al., 2021): [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685). Original LoRA proposal.
- QLoRA paper (Dettmers et al., 2023): NF4 quantization and quantized fine-tuning.
- mlx-lm fine-tuning documentation: [github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm). LoRA and QLoRA via mlx-lm.
- MLX optimizers documentation: [ml-explore.github.io/mlx/build/html/python/optimizers.html](https://ml-explore.github.io/mlx/build/html/python/optimizers.html).

---

## See Also

- [Training vs. Inference](../01-foundations/03-training-vs-inference.md) -- the memory and compute profiles differ fundamentally; why the CUDA gap is larger for training than inference
- [Quantization](../01-foundations/10-quantization.md) -- essential context for QLoRA; NF4, group quantization, and quantization error
- [Frameworks](02-frameworks.md) -- PyTorch vs MLX API; the autograd model, optimizer API, and module system that training is built on
- [LLMs](03-llms.md) -- the domain where MLX training (mlx-lm LoRA) is most mature; inference and fine-tuning together
