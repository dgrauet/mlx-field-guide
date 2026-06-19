# Loss Functions

> **One-liner:** A [loss function](../glossary.md#loss-function) collapses "how wrong is this output?" into a single number that [training](../glossary.md#training) minimizes -- and even though you never compute it during inference, *which* loss a model was trained with explains why it outputs what it outputs.

## The Intuition

[Training vs. Inference](03-training-vs-inference.md) introduced the loss as the number training drives toward zero. This page goes deeper, because the loss is not one formula -- it's a *family*, and the choice of family is a design decision baked into every model you port.

Here is the part that matters for an inference-only porter: **the loss function shapes the model's output format.** A model doesn't emit probabilities because someone thought that was nice; it emits them because it was trained with a loss that expects them. A diffusion model predicts *noise* rather than a clean image because its loss was defined on noise. When you port a model and its output "looks wrong," the loss the original was trained with is often the missing context that tells you what the output is *supposed* to be.

So while you will "never call a loss function" in an inference port, understanding losses is how you read the architecture correctly.

```
WHERE THE LOSS LIVES (and why a porter still cares)

  training:   input -> model -> output -> LOSS(output, target) -> gradients
                                            |
                                            +-- you strip this out for inference
                                            +-- BUT it dictated the output's shape,
                                                range, and meaning, which you keep
```

---

## How It Actually Works

A loss takes the model's prediction and the ground-truth target and returns a scalar. The three families below cover essentially everything you will meet in generative AI.

### Mean Squared Error (regression, and diffusion)

MSE measures the average squared difference between prediction and target:

```
MSE = mean( (prediction - target)^2 )

  - Penalizes large errors quadratically (being very wrong hurts a lot)
  - Output space is continuous values, not categories
  - The gradient is simply proportional to (prediction - target)
```

This is the loss behind **[diffusion](08-diffusion.md) models**. The denoising network is trained to predict the noise (or the velocity) added to an image, and "predict a continuous tensor and match it" is exactly an MSE problem. This is *why* a diffusion model's raw output is noise-shaped, not image-shaped: the loss asked it to predict noise. A porter who expects the model to emit a picture directly has misread what the loss defined.

### Cross-Entropy (classification, and every LLM)

When the target is a *category* -- which of 50,000 vocabulary [tokens](../glossary.md#token) comes next -- the loss is **cross-entropy**:

```
CROSS-ENTROPY (for one prediction)

  CE = - log( p_correct )

  where p_correct is the probability the model assigned to the right answer.

  - Predict the right token with p=1.0  -> loss = -log(1)   = 0     (perfect)
  - Predict it with p=0.5               -> loss = -log(0.5) = 0.69
  - Predict it with p=0.01              -> loss = -log(0.01)= 4.6   (badly wrong)

  Confidently wrong is punished far more than uncertainly wrong.
```

Cross-entropy is the **pre-training objective behind every [LLM](../02-ecosystem/03-llms.md)** (alignment stages like RLHF/DPO add other objectives on top, but the next-token head stays the same). The model produces a score for each vocabulary token, softmax turns those scores into probabilities, and cross-entropy rewards putting probability mass on the actual next token.

This explains a fact that trips up porters: **LLMs output raw scores (logits), not probabilities.** The model's final layer emits one unnormalized number per token. Softmax-and-cross-entropy is applied *on top* during training, and at inference you apply softmax (or just argmax) yourself. If you expected the model to hand you probabilities directly, the loss is the reason it doesn't. (For the numerical care this combination needs, see [Numerical Stability](12-numerical-stability.md) -- "log of softmax" is computed as a single fused, stable operation precisely because the naive version overflows.)

### KL Divergence (VAEs, and distribution matching)

**KL divergence** measures how far one probability distribution is from another. It is the second half of a **[VAE](../glossary.md#vae-variational-autoencoder)'s** loss: reconstruction error (an MSE-like term) keeps the decoded output faithful, while a KL term pulls the [latent](../glossary.md#latent-space) distribution toward a smooth standard normal. That KL term is *why* a VAE's latent space is smooth and interpolatable -- a property [latent diffusion](07-latent-space.md) depends on. Remove it conceptually and you have a plain autoencoder with a lumpy latent.

### A map of which loss goes with which model

```
LOSS  ->  MODEL FAMILY  ->  WHAT THE OUTPUT IS

  MSE            diffusion U-Net / DiT      predicted noise or velocity
  Cross-entropy  LLM, classifier            per-token logits over a vocabulary
  MSE + KL       VAE                         a smooth, sampleable latent
  Cross-entropy  VLM (text head)             per-token logits, conditioned on image
```

---

## Why It Matters for MLX

You are almost always doing inference-only ports, so you **strip the loss out** -- but it leaves three fingerprints you must read correctly:

1. **Output format.** Logits vs probabilities (cross-entropy), noise vs image (MSE-on-noise), distribution vs point (VAE). Mislabel any of these and the post-processing is wrong even when the model is correct. The most common symptom: applying softmax twice, or not at all, because you misjudged whether the model's head already normalizes (it doesn't -- cross-entropy training means raw logits come out).

2. **Prediction target in diffusion.** A scheduler must know whether the network predicts noise (`epsilon`), the clean sample, or velocity (`v`) -- and that choice is a *loss-target* decision from training. Match the sampler to the training target or the denoising loop diverges (see [Diffusion](08-diffusion.md)). This is a frequent silent port bug: the weights are fine, but the sampler assumes the wrong prediction target.

3. **Fine-tuning, if you do touch training.** When you *do* run training -- LoRA fine-tuning via mlx-lm -- the loss reappears. MLX provides standard losses in **`mlx.nn.losses`** (e.g. `cross_entropy`, `mse_loss`), and mlx-lm's LoRA trainer uses token-level cross-entropy. For the inference-only majority of ports, these never get called.

The practical rule: when an output looks wrong, ask "what loss was this trained with, and what does that loss imply the output *is*?" before assuming the weights or the math are broken.

---

## Key Terms

| Term | Definition |
|------|------------|
| **Loss Function** | A formula mapping (prediction, target) to a single scalar that training minimizes. |
| **MSE (Mean Squared Error)** | Average squared difference between prediction and target; the loss for continuous outputs, including diffusion noise prediction. |
| **Cross-Entropy** | The loss for categorical targets; `-log(p_correct)`; used to train every LLM and classifier over a vocabulary/label set. |
| **KL Divergence** | A measure of how far one distribution is from another; the regularizing half of a VAE's loss that makes its latent space smooth. |
| **Logits** | The raw, unnormalized scores a model emits before softmax; LLMs output these because cross-entropy training applies softmax on top. |
| **Prediction target** | In diffusion, what the network is trained to output (noise / sample / velocity); a loss-definition choice the sampler must match. |

---

## Sources

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Ch. 5 & 6 (loss functions, maximum likelihood). [deeplearningbook.org](https://www.deeplearningbook.org/)
- Bishop, C. (2006). *Pattern Recognition and Machine Learning* -- cross-entropy, MSE, and their probabilistic origins.
- Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes." The VAE / KL-divergence loss. [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." The MSE-on-noise training objective. [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- MLX `mlx.nn.losses`: [ml-explore.github.io/mlx/build/html/python/nn/losses.html](https://ml-explore.github.io/mlx/build/html/python/nn/losses.html)

---

## See Also

- [Training vs. Inference](03-training-vs-inference.md) -- prerequisite; the loss is the heart of the training loop you strip out for inference
- [Diffusion](08-diffusion.md) -- the MSE-on-noise objective and why the prediction target (epsilon/sample/velocity) must match the sampler
- [Latent Space](07-latent-space.md) -- the VAE's reconstruction + KL loss is what makes the latent smooth enough for latent diffusion
- [Numerical Stability](12-numerical-stability.md) -- why softmax-plus-cross-entropy is computed as one fused, max-subtracted operation
- [Tokenization](09-tokenization.md) -- cross-entropy operates over the token vocabulary; vocabulary size sets the size of the logit vector
