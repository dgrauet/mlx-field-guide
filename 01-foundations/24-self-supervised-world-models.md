# Self-Supervised Learning & World Models

> **One-liner:** Self-supervised models learn from raw images and video with no labels, by predicting one part of the data from another; when they learn to predict *what happens next* -- especially *given an action* -- they become **world models**. V-JEPA 2 predicts in an abstract feature space; Matrix-Game generates the next pixels with a video diffusion model. Porting either means getting the training-time machinery right too: which encoder copy to load, how features are normalized, how actions and memory are fed in.

## The Intuition

Labels are expensive; video is everywhere. **Self-supervised learning** turns unlabeled data into a training signal by hiding part of it and asking the model to predict the hidden part. What differs between methods is *what* the model predicts:

```
WHAT IS PREDICTED                         EXAMPLE              USED FOR

  the hidden pixels                       MAE, VideoMAE        pretraining encoders
  which pairs belong together             CLIP, SigLIP         image-text encoders (VLMs)
    (contrastive)
  the teacher's view of the same image    DINOv2               general image features
    (self-distillation)                                          (Hunyuan3D conditioning)
  the hidden part's FEATURES, not pixels  I-JEPA, V-JEPA 2     video understanding,
    (joint-embedding predictive)                                 prediction, planning
  the next frames, given an action        Matrix-Game,         interactive "world"
    (generative world model)                GameFactory          simulation
```

Predicting pixels forces a model to spend capacity on details nobody needs (every leaf's texture). Predicting **features** lets it ignore what is unpredictable and model what matters: objects, motion, cause and effect. That is the JEPA bet.

---

## How It Actually Works

### V-JEPA 2: predicting features of masked video

V-JEPA 2 (Meta, 2025) cuts a clip into **tubelets** -- 2 frames × 16 × 16 pixels -- and embeds each one as a token with a ViT encoder using 3D RoPE ([Positional Encoding](11-positional-encoding.md)). The patch embedding is a Conv3d with kernel = stride = tubelet ([Convolutions](19-convolutions.md#patch-embedding-a-convolution-with-kernel--stride)). Training uses three networks:

```
  video clip --> tubelets
     |
     |  mask large spatio-temporal blocks (multi-block masking)
     v
  CONTEXT ENCODER        sees only the visible tokens            (trained by gradient)
     |
  PREDICTOR              + learnable mask tokens at the hidden    (trained by gradient)
     |                     positions -> predicted features
     v
  L1 loss  <-->  TARGET ENCODER sees the full clip,              (NOT trained by gradient:
                 features of the hidden positions,                 an exponential moving
                 LayerNorm'd, gradient stopped                     average of the context
                                                                   encoder's weights)
```

From the vjepa2-mlx port's training code: the loss is `mean(|z_pred - h|)` over predicted tokens, the targets are `layer_norm(target_encoder(clip))` with eps 1e-5, and after each step the target weights move toward the online ones with momentum 0.996 → 0.9999.

### Why the target encoder is a slow copy

If the model could minimize the loss by changing *both* the prediction and the target, the easiest solution would be to output the same vector for everything. The loss would hit zero and the features would be worthless. This is **representation collapse**. A toy JEPA in MLX 0.32.3 (context half of a vector predicts the other half's features, 1,500 AdamW steps) shows it:

```
  target setup                                 final loss   embedding std across samples
  no stop-gradient, same encoder                 0.0007          0.045   <- collapsed
  stop-gradient, same encoder                    0.0762          0.673
  EMA target + stop-gradient + LayerNorm         0.0644          1.183   <- V-JEPA recipe
```

The collapsed model has the **lowest** loss. That's the trap: a JEPA loss going to zero is a bug signal, not a success. Stopping the gradient through the target, making the target a slowly-updated average (EMA), and normalizing the targets keep the features informative.

### Using the features: probes and predictors

A self-supervised encoder isn't a classifier. To use it, you freeze it and train a small **probe** on top. V-JEPA 2's is an **attentive probe**: learned query tokens cross-attend to the encoder's tokens, then a linear head (174 classes for Something-Something v2, 48 for Diving-48). The vjepa2-mlx port matches the reference logits to 3.3e-5 on SSv2.

The **predictor** is the world-model part. It takes features of what was seen and predicts features of what wasn't. V-JEPA 2-AC adds actions to the predictor, so it predicts the features after a robot action. Planning then means searching for the action sequence whose predicted features are closest to a goal image's features, entirely in feature space, without generating a single pixel.

### Generative world models: Matrix-Game

Matrix-Game takes the other road: it generates the actual video, frame by frame, conditioned on keyboard and mouse input. Matrix-Game 3.0 is a Wan-style video diffusion transformer with:

- an **action module** that injects mouse and keyboard signals into the DiT's hidden states through cross-attention (from GameFactory);
- **autoregressive clips**: each new clip (40 frames after the first) continues from the previous ones;
- **camera-aware memory**: previous latent frames are selected by field-of-view overlap with the current camera pose, encoded with Plücker ray embeddings, and fed back as clean context. In the MLX port they get timestep 0 and neutral actions, the same "clean conditioning token" pattern as the denoise mask in [Conditioning & Guidance](23-conditioning-guidance.md);
- **few-step distillation** (DMD) so each clip needs ~3 denoising steps.

The reference reaches 40 fps on datacenter GPUs. The MLX port's README measures ~3 minutes per 2-second clip at 480p on an M2 Pro 32 GB (~7 min at 720p). Attention over ~13,000 patches per step, and a ~5x memory-bandwidth gap with an H100 even on the fastest Mac, put real time out of reach. Offline generation and experimentation are what a Mac is good for here.

---

## Why It Matters for MLX

### Load the right copy of the encoder

Self-supervised checkpoints often contain **several copies** of the encoder, and the right one is not called `encoder`:

```
  V-JEPA 2.1 checkpoint            V-JEPA 2.0 checkpoint
    ema_encoder   <- use this         target_encoder  <- use this
    encoder                           encoder
    predictor                         predictor
    opt, scaler, epoch, ...           opt, scaler, epoch, ...
```

The EMA/target copy is the one evaluated in the papers (from the mlx-forge V-JEPA 2 conversion notes). Loading `encoder` gives a model that runs, produces plausible features, and scores lower on every benchmark. The same pattern -- a `teacher`/`student` or `ema`/`online` pair -- appears in DINO-family checkpoints.

### Port failures

```
SELF-SUPERVISED & WORLD-MODEL PORT FAILURES

  Symptom                                    Likely cause
  -------------------------------------      ------------
  Probe accuracy a few points below the      online `encoder` loaded instead of
    paper                                      `ema_encoder` / `target_encoder`
  Features fine, probe logits off            input preprocessing (resize, crop,
                                               ImageNet mean/std, frame sampling
                                               stride) differs from the eval transform
  Training loss drops to ~0 quickly          representation collapse: gradient
                                               flowing into the target, EMA missing,
                                               or target LayerNorm missing
  Predictor output wrong, encoder exact      mask tokens / positions of hidden
                                               tokens built differently (the
                                               predictor needs the hidden positions'
                                               RoPE, not the visible ones')
  World model ignores the controls           actions misaligned with latent
                                               frames (temporal compression: several
                                               video frames per latent frame)
  Long rollouts drift or lose the scene      memory frames not re-injected, or
                                               given a non-zero timestep
  OOM on long generations                    keeping every past latent instead
                                               of the selected memory frames
```

### What a Mac is good for

- **Feature extraction and probing** with V-JEPA 2 or DINOv2: the encoder runs once per clip, memory is moderate, and unified memory lets a ViT-L and a long video share RAM.
- **Training small JEPAs** and probes: the vjepa2-mlx port includes the full training stack (masking, EMA, distributed over several Macs via `mx.distributed`).
- **World models offline**, not interactively, at current hardware speeds.

---

## Key Terms

| Term | Definition |
|------|------------|
| **Self-supervised learning** | Training on unlabeled data by predicting hidden parts of it from visible parts. |
| **JEPA (joint-embedding predictive architecture)** | Self-supervised method that predicts the *features* of masked regions, not their pixels. |
| **Context / target encoder** | In a JEPA, the encoder that sees visible tokens (trained) and the one that produces targets from the full input (EMA copy, no gradient). |
| **EMA (exponential moving average)** | Slowly updated copy of a network's weights (`target = m * target + (1 - m) * online`); the version usually released. |
| **Representation collapse** | A self-supervised failure where the encoder outputs near-identical features for every input; the loss looks excellent. |
| **Tubelet** | A video patch spanning several frames (2 × 16 × 16 in V-JEPA 2), embedded as one token. |
| **Attentive probe** | Small classifier with learned queries that cross-attend to frozen encoder features. |
| **World model** | A model that predicts how an environment evolves, often conditioned on actions; in feature space (V-JEPA 2-AC) or pixel space (Matrix-Game). |
| **Action conditioning** | Feeding control inputs (keyboard, mouse, robot commands) to a predictor or generator. |
| **Plücker embedding** | Encoding of a camera's rays (origin and direction per pixel) used to tell a model the camera pose. |

---

## Sources

- Assran, M., et al. (2025). "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." [arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985)
- Bardes, A., et al. (2024). "Revisiting Feature Prediction for Learning Visual Representations from Video" (V-JEPA). [arxiv.org/abs/2404.08471](https://arxiv.org/abs/2404.08471)
- Assran, M., et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA). [arxiv.org/abs/2301.08243](https://arxiv.org/abs/2301.08243)
- He, K., et al. (2021). "Masked Autoencoders Are Scalable Vision Learners" (MAE). [arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)
- Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." [arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)
- Matrix-Game 2.0: "An open-source, real-time, and streaming interactive world model." [arxiv.org/abs/2508.13009](https://arxiv.org/abs/2508.13009); Matrix-Game 3.0 (long-horizon memory) in [github.com/SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game)
- Yu, J., et al. (2025). "GameFactory: Creating New Games with Generative Interactive Videos" (action module). [arxiv.org/abs/2501.08325](https://arxiv.org/abs/2501.08325)
- vjepa2-mlx port (training loss, EMA, target LayerNorm, probes, parity table): [github.com/dgrauet/vjepa2-mlx](https://github.com/dgrauet/vjepa2-mlx); mlx-forge V-JEPA 2 conversion notes (checkpoint containers): [github.com/dgrauet/mlx-forge](https://github.com/dgrauet/mlx-forge)
- Matrix-Game-mlx port (memory selection, timings on Apple Silicon): [github.com/dgrauet/Matrix-Game-mlx](https://github.com/dgrauet/Matrix-Game-mlx)
- Collapse demonstration on this page: toy MLP JEPA in MLX 0.32.3, random structured data.

---

## See Also

- [Training vs Inference](03-training-vs-inference.md) -- the training loop the EMA update sits in
- [Attention](06-attention.md) -- the cross-attention used by probes and action modules
- [Positional Encoding](11-positional-encoding.md) -- 3D RoPE over time, height and width
- [Convolutions & Patchification](19-convolutions.md) -- tubelet embedding as a Conv3d
- [Conditioning & Guidance](23-conditioning-guidance.md) -- clean memory frames and action injection in generative world models
- [Vision-Language Models](../02-ecosystem/11-vision-language-models.md) -- contrastive encoders (CLIP, SigLIP) in practice
