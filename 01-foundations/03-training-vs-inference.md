# Training vs Inference

> **One-liner:** Training teaches a model by adjusting its weights based on examples; inference uses the trained model to make predictions.

## The Intuition

You know from the Neural Networks page that a model is a layer architecture filled with learned parameter values. But how do those parameters get their values in the first place -- and what exactly happens when you run a model after that?

The answer splits into two completely different modes of operation.

### Studying vs. taking the exam

**Training is studying for an exam.**

You read through many examples (the training data). After each one, you check your answer against the correct answer (the loss function). If you were wrong, you figure out which parts of your understanding need adjusting (backpropagation) and make a small correction (gradient descent). You repeat this thousands of times until your answers are consistently good.

```
              +--------------------------------------------------+
              |               TRAINING LOOP                     |
              |                                                  |
              |   data --> [model] --> output                    |
              |                          |                       |
              |                       [loss]  <-- correct answer |
              |                          |                       |
              |                    [gradients]                   |
              |                          |                       |
              |                  [update weights]                |
              |                          |                       |
              |              repeat millions of times            |
              +--------------------------------------------------+
```

**Inference is taking the exam.**

You sit down, read the question, and produce your best answer using everything you studied. The exam doesn't change your understanding -- you don't update anything. You just apply what you already know.

```
              +--------------------------------------------------+
              |                  INFERENCE                       |
              |                                                  |
              |   input --> [model] --> output                   |
              |                                                  |
              |   (weights stay fixed, no gradients,            |
              |    no loss function, no weight updates)          |
              +--------------------------------------------------+
```

### Where you actually live as an MLX porter

Here is the critical context for understanding this field guide: **almost all MLX porting work is inference-only.**

LTX-Video was trained on clusters of NVIDIA H100 GPUs -- hardware that costs hundreds of thousands of dollars and runs for weeks. Matrix-Game-mlx was trained the same way. When you port these models to MLX, you are not re-doing that training. You are taking the result of that training (a `.safetensors` weight file) and building a system to run inference on Apple Silicon.

The training already happened. It happened somewhere else, on different hardware, using different code. Your job is to run the forward pass efficiently on an M-series chip.

This means most of the training-specific concepts in this page -- loss functions, gradients, optimizers -- are background knowledge for understanding ML code, not things you will directly implement. You will encounter them in the original PyTorch code you're porting from, and you need to recognize what they do so you can correctly strip them out or ignore them in the inference path.


## How It Actually Works

### The forward pass

The forward pass is the computation you already saw in the Neural Networks page: input flows through every layer in sequence, each layer applies its transformation, and you get an output at the end.

```
input tensor
      |
      v
 [Layer 1: Linear + ReLU]
      |
      v
 [Layer 2: Linear + ReLU]
      |
      v
 [Layer 3: Linear + Softmax]
      |
      v
output tensor (predictions)
```

This is all that happens during inference. The weights are frozen. No state is updated.

### The loss function

During training, the output of the forward pass is immediately fed into a **loss function** -- a formula that measures how wrong the output is.

The loss function produces a single number: the **loss** (also called the **cost**). Lower is better. The goal of training is to drive this number toward zero.

Two concrete examples:

**Mean Squared Error (MSE):** used for regression problems where the output is a continuous number.

```
predicted = [2.3, 0.8, 1.5]
actual    = [2.0, 1.0, 1.0]

squared errors = [(2.3-2.0)^2, (0.8-1.0)^2, (1.5-1.0)^2]
               = [0.09,        0.04,         0.25]

MSE = mean([0.09, 0.04, 0.25]) = 0.127
```

**Cross-Entropy Loss:** used for classification problems where the output is a probability distribution over categories.

```
predicted probabilities = [0.7, 0.2, 0.1]   (cat, dog, car)
actual label            = "cat"   (index 0, one-hot: [1, 0, 0])

cross-entropy = -log(0.7) = 0.357   (negative log of the predicted probability for the correct class)
```

If the model had predicted [0.95, 0.03, 0.02] for the correct class, the loss would be -log(0.95) = 0.051 -- much lower. If it had predicted [0.1, 0.8, 0.1], the loss would be -log(0.1) = 2.303 -- much higher. The loss rewards confident correct predictions and punishes confident wrong ones.

During inference, there is no loss function. You don't have "correct answers" to compare against -- you're trying to produce those answers.

### Backpropagation

Once you have the loss value, training needs to figure out how to adjust every weight in the network to reduce the loss. This is the job of **backpropagation** (often shortened to "backprop").

Backpropagation works by computing **gradients**: for each weight, "if I increase this weight by a tiny amount, how much does the loss change?"

```
gradient of weight W = dLoss / dW
```

If the gradient is positive, increasing W makes the loss go up -- so you should decrease W.
If the gradient is negative, increasing W makes the loss go down -- so you should increase W.
If the gradient is zero, W has no effect on the loss at this moment.

Backpropagation does this for every single weight in the network simultaneously, using the chain rule of calculus to propagate gradients backward through each layer:

```
output layer gradients
        |
        v  (chain rule applied backward through layer 3)
layer 3 weight gradients
        |
        v  (chain rule applied backward through layer 2)
layer 2 weight gradients
        |
        v  (chain rule applied backward through layer 1)
layer 1 weight gradients
```

This is where the "backward pass" name comes from: you first run data forward through the network (forward pass), then run gradient information backward through the network (backward pass).

During inference, there is no backward pass. Gradients are never computed. This is why inference is faster and uses less memory than training.

### Gradient descent and optimizers

Once you have the gradients, **gradient descent** adjusts each weight in the direction that reduces the loss:

```
new_weight = old_weight - learning_rate * gradient
```

The **learning rate** controls how large a step to take. Too large and the weights bounce around and never settle. Too small and training takes forever.

```
learning_rate = 0.01
weight = 0.5
gradient = 0.3

new_weight = 0.5 - 0.01 * 0.3 = 0.497
```

A tiny adjustment. Repeat millions of times and the weights gradually find values that minimize the loss.

**Optimizers** are smarter versions of plain gradient descent. Instead of taking a fixed-size step, they track history and adapt their behavior:

| Optimizer | What it does |
|-----------|-------------|
| **SGD** (Stochastic Gradient Descent) | Plain gradient descent with optional momentum to smooth updates |
| **Adam** | Tracks both the mean and variance of recent gradients; adapts the step size per weight; most commonly used |
| **AdamW** | Adam with weight decay -- a regularization term that prevents weights from growing too large; standard for transformer training |

You will see these in original PyTorch training code. In an inference pipeline, they don't appear -- there's nothing to optimize.

### Epochs, batches, and iterations

Training doesn't process the entire dataset at once. It works in chunks:

```
Dataset: 1,000,000 training examples

Batch size: 32    (process 32 examples at once)
                  --> 31,250 iterations to see the full dataset once

Epoch: one complete pass through the entire dataset
       After 10 epochs: the model has seen each example 10 times
```

A **batch** is one chunk of examples processed together. Running a batch through the network is a single forward pass and backward pass -- one "step" of training.

An **iteration** (or **step**) is one forward+backward pass on one batch.

An **epoch** is one full pass through the entire training dataset. Training typically runs for many epochs -- sometimes dozens, sometimes hundreds.

During inference, there are no epochs or training iterations. You process one input (or one batch of inputs) and stop.

### The complete training loop vs. inference

A training loop in pseudocode:

```python
for epoch in range(num_epochs):
    for batch in dataloader:          # iterate over batches
        inputs, labels = batch

        # Forward pass
        predictions = model(inputs)

        # Compute loss
        loss = loss_function(predictions, labels)

        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()         # reset gradients for next iteration

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

An inference pipeline in PyTorch looks like this:

```python
model.eval()                          # disable training-specific behavior (dropout, etc.)

with torch.no_grad():                 # skip gradient computation entirely
    predictions = model(inputs)
```

That's all. No loss, no backward pass, no optimizer step. The `torch.no_grad()` context manager tells PyTorch not to build the computation graph needed for backpropagation -- which saves significant memory and compute time.

### Training vs. Inference: comparison

| | Training | Inference |
|---|---|---|
| **What runs** | Forward pass + backward pass | Forward pass only |
| **Weight updates** | Yes, after every batch | No |
| **Gradients computed** | Yes | No |
| **Loss function needed** | Yes | No |
| **Labeled data needed** | Yes | No |
| **Memory use** | High (stores gradients + activations for backprop) | Lower (weights + activations only) |
| **Speed focus** | Throughput (samples/second over days) | Latency (time to produce one output) |
| **Hardware** | Multi-GPU clusters; NVIDIA dominates | Single GPU or CPU; Apple Silicon viable |
| **Duration** | Hours to weeks | Milliseconds to seconds |
| **Typical hardware cost** | Very high | Much lower |


## Why It Matters for MLX

### You are working in inference mode

When you load a pretrained model in MLX and run it, you are doing pure inference. The weights you load from a `.safetensors` file are the result of training that already happened. Your code never calls a loss function, never runs backpropagation, never updates those weights.

This simplifies things considerably. You don't need to understand how the LTX-Video model was trained to run it. You need to understand the forward pass: what shape input it expects, what transformations it applies in sequence, and what shape output it produces.

If your forward pass produces correct shapes but wrong values, the likely culprits are:

1. **Incorrect weight loading** -- a weight from the file was placed into the wrong layer, or the keys don't match the architecture
2. **Numerical precision differences** -- the original model used float16 and you loaded float32, or vice versa; small but accumulating precision differences between CUDA and Metal
3. **Missing or wrong data preprocessing** -- the input wasn't normalized the way the model expects
4. **Transposed weights** -- some weight files store weights in a different orientation than the framework expects; loading without transposing silently produces wrong outputs

None of these require understanding backpropagation. They require understanding the forward pass and weight loading.

### `model.eval()` and its MLX equivalent

In PyTorch, calling `model.eval()` before running inference does two important things:

1. **Disables Dropout** -- Dropout randomly zeroes out activations during training (a regularization technique). During inference, you want all activations to pass through.
2. **Disables BatchNorm updates** -- Batch Normalization tracks running statistics during training. During inference, it uses the saved statistics instead.

Forgetting `model.eval()` in PyTorch is a common bug: the model will produce different outputs every time you run it (due to random dropout) and may produce slightly wrong outputs (due to batch norm behavior).

MLX handles this differently. MLX modules don't have a single toggle method that switches all training-specific behavior. Instead:

- **Dropout**: `nn.Dropout` in MLX has a `training` parameter. When building an inference-only pipeline, you typically omit dropout layers entirely, or pass `training=False` explicitly.
- **BatchNorm**: If the original model uses batch normalization (less common in modern transformers), you handle it by loading the saved running mean/variance from the weight file.

In practice, the models you're porting (LTX-Video, Matrix-Game-mlx) are transformer-based. Transformers use LayerNorm rather than BatchNorm, and use dropout only during training. When you're building an inference pipeline, these layers either don't appear or have no training-specific behavior to disable.

### LTX-Video is pure inference on Apple Silicon

The LTX-Video model was trained on NVIDIA H100 GPUs. The training job consumed weeks of compute time and updated model weights billions of times. The result of that training was saved to weight files.

When you run LTX-Video on an M-series Mac, here is what actually happens:

```
1. Load weight file (.safetensors)
   --> copies trained parameters into MLX arrays

2. Encode the text prompt
   --> forward pass through CLIP or T5 text encoder
   --> produces a conditioning tensor

3. Run the diffusion denoising loop
   --> ~50 iterations, each a forward pass through the video transformer
   --> no gradients, no weight updates between iterations
   --> the "loop" is iterative refinement of the output tensor, not training

4. Decode the latent tensor
   --> forward pass through the VAE decoder
   --> produces pixel-space video frames
```

Steps 2, 3, and 4 are all forward passes. The "50 iterations" of the diffusion loop might look like training iterations, but they are not -- no weights are being updated. The model is fixed; only the output tensor changes as noise is progressively removed.

This is the key insight: even iterative inference algorithms (diffusion, autoregressive decoding) are pure inference. The model's weights never change after loading.

### Training on MLX: when it does come up

MLX supports training. The `mlx.optimizers` module contains Adam, SGD, AdamW, and others. You can compute gradients with `mx.grad()` or `mx.value_and_grad()`. MLX uses lazy evaluation to build efficient computation graphs for training workloads.

But training large models on Apple Silicon is currently less common for a few reasons:

- **Memory limits**: Even a Mac with 128 GB unified memory is small compared to a training cluster. The LTX-Video model alone has billions of parameters -- training it from scratch would require far more memory than any current Mac has.
- **Tooling maturity**: Training pipelines built on PyTorch have years of optimizations, monitoring tools, and established practices. MLX training tooling is younger.
- **The practical split**: Training happens on cloud infrastructure with NVIDIA hardware. Inference (and deployment) is where MLX shines -- high-quality Apple Silicon inference with no cloud costs.

If you do encounter training in an MLX context, it will most likely be fine-tuning: loading a pretrained model and running a small number of training steps on a specific dataset to specialize its behavior. The mechanics are the same (forward pass, loss, backward pass, optimizer step), just at a much smaller scale.


## Key Terms

| Term | Definition |
|------|------------|
| **Training** | The process of adjusting a model's weights based on labeled examples, using a loss function and gradient descent, to minimize prediction error |
| **Inference** | Running a trained model on new inputs to produce predictions; weights stay fixed, no gradients are computed |
| **Forward Pass** | The computation of passing an input through all layers in sequence to produce an output; happens in both training and inference |
| **Backward Pass (Backpropagation)** | The computation of gradients by applying the chain rule backward through each layer; happens only during training |
| **Loss Function** | A formula that measures how wrong the model's output is compared to the correct answer; produces a single scalar value (the loss) that training tries to minimize |
| **Gradient** | For a given weight, the gradient is the rate of change of the loss with respect to that weight -- how much the loss would change if the weight changed by a tiny amount |
| **Gradient Descent** | The optimization algorithm that updates each weight in the direction that reduces the loss, scaled by the learning rate |
| **Learning Rate** | A hyperparameter that controls how large a step to take when updating weights; too large causes instability, too small causes slow training |
| **Optimizer** | An algorithm for computing weight updates from gradients; common optimizers (Adam, AdamW, SGD) use techniques like momentum and adaptive step sizes to converge faster |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | A subset of the training data processed together in a single forward+backward pass; batching is more computationally efficient than processing one example at a time |
| **Overfitting** | When a model learns the training data so precisely that it performs well on training examples but poorly on new, unseen examples; a sign that the model has memorized rather than generalized |
| **Checkpoint** | A saved snapshot of a model's weights at a particular point during training; loading a checkpoint is how pretrained models are distributed and how inference pipelines begin |


## Sources

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. **Chapters 6-8** cover feedforward networks, training mechanics, and regularization in full mathematical detail. Available free at [deeplearningbook.org](https://www.deeplearningbook.org)
- Karpathy, A. *Neural Networks: Zero to Hero* lecture series. "The spelled-out intro to neural networks and backpropagation: building micrograd" is the clearest available walkthrough of how backpropagation actually works in code. Available on YouTube.
- [MLX Examples repository](https://github.com/ml-explore/mlx-examples) -- concrete inference pipelines in MLX for language models, image generation, and other tasks; the best reference for how MLX inference code is actually structured


## See Also

- [Neural Networks](02-neural-networks.md) -- prerequisite; this page assumes you understand layers, weights, and the forward pass. If the forward pass section above felt unclear, review the Neural Networks page first.
- [GPU Computing](04-gpu-computing.md) -- why training and inference have such different hardware requirements, and what makes Apple Silicon efficient for inference workloads.
- [Quantization](10-quantization.md) -- a technique applied to pretrained models before inference that reduces weight precision (float16 -> int4) to save memory and increase speed; one of the most impactful techniques for running large models on Apple Silicon.
