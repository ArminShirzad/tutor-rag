# Optimizers and the Training Loop

## Gradient descent

Gradient descent updates each parameter in the direction that reduces the loss:

    w = w - learning_rate * gradient

The gradient points uphill, so we step against it. The **learning rate** controls the
step size and is the single most important hyperparameter in deep learning.

- Too high: the loss oscillates or diverges to NaN.
- Too low: training is correct but impractically slow, and it may stall in a plateau.

A reasonable starting point is 1e-3 for Adam and 1e-2 for SGD with momentum.

## Variants

**Batch gradient descent** computes the gradient over the entire dataset before each
update. Accurate but slow and memory-hungry.

**Stochastic gradient descent (SGD)** updates after every single example. Fast and
noisy; the noise can actually help escape sharp local minima.

**Mini-batch gradient descent** updates after a small batch, typically 32 to 256
examples. This is what everyone actually uses -- it balances gradient quality with
hardware efficiency, because GPUs are built for batched matrix operations.

## Momentum

Plain SGD oscillates across the walls of steep, narrow valleys. **Momentum**
accumulates an exponentially decaying average of past gradients and moves along that
instead:

    v = beta * v + gradient
    w = w - learning_rate * v

with beta typically 0.9. The physical analogy is a heavy ball rolling downhill: it
builds speed in consistent directions and damps out oscillation in inconsistent ones.

## Adam

**Adam** (Adaptive Moment Estimation) is the default optimizer for most deep learning
work. It maintains two running averages per parameter:

- the first moment, an estimate of the mean gradient (momentum)
- the second moment, an estimate of the uncentred variance of the gradient

It divides the update by the square root of the second moment, which gives every
parameter its own effective learning rate. Parameters with consistently large
gradients take smaller steps; rarely-updated parameters take larger ones. This makes
Adam robust to badly scaled features and to sparse gradients.

Default hyperparameters that rarely need changing: `beta1 = 0.9`, `beta2 = 0.999`,
`epsilon = 1e-8`.

**AdamW** decouples weight decay from the gradient update. In plain Adam the L2
penalty gets scaled by the adaptive term, which weakens it unpredictably. AdamW
applies the decay directly to the weights instead. It is the correct choice for
training transformers and is the standard in modern NLP.

## Learning rate schedules

A fixed learning rate is rarely optimal. Common schedules:

- **Step decay**: multiply by 0.1 every N epochs.
- **Cosine annealing**: smoothly decrease following a cosine curve. Widely used.
- **Warmup**: start very small and increase linearly for the first few hundred steps.
  Essential for transformers, where early large updates destabilise attention layers.
- **ReduceLROnPlateau**: cut the learning rate when validation loss stops improving.

## Vanishing and exploding gradients

In deep networks the chain rule multiplies many gradients together. If they are
consistently below 1, the product shrinks toward zero and early layers stop learning
-- the **vanishing gradient** problem. If consistently above 1, the product explodes
into NaN.

Mitigations: ReLU-family activations (gradient of 1 for positive inputs), residual
connections (a gradient highway that skips layers), careful initialisation (He
initialisation for ReLU, Xavier/Glorot for tanh), batch or layer normalisation, and
**gradient clipping** -- rescaling the gradient vector when its norm exceeds a
threshold, typically 1.0.

## Normalization layers

**Batch normalization** normalises each feature across the batch dimension. It speeds
up training and has a mild regularizing effect, but it depends on batch statistics,
so it behaves badly with very small batches and needs separate train/eval behaviour.

**Layer normalization** normalises across the feature dimension within each single
example. It is independent of batch size, which is why transformers use it
exclusively.

## A practical debugging checklist

When a model will not train:

1. Overfit a single batch deliberately. If the model cannot reach near-zero loss on
   10 examples, there is a bug, not a tuning problem.
2. Check the learning rate first; it is wrong more often than anything else.
3. Verify the input data: shapes, normalisation, and that labels line up with inputs.
4. Confirm `model.train()` and `model.eval()` are called in the right places.
5. Check that the optimizer's gradients are actually being zeroed each step
   (`optimizer.zero_grad()`); otherwise gradients accumulate across batches.
6. Inspect gradient norms. All zero means a disconnected graph; NaN means explosion.
