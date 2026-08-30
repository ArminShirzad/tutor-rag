# Neural Networks: Foundations

## What a neural network actually is

A neural network is a function approximator. It takes an input vector, passes it
through a sequence of layers, and produces an output. Each layer applies a linear
transformation followed by a non-linear activation:

    h = activation(W @ x + b)

W is the weight matrix, b is the bias vector, and x is the input. Learning means
finding values of W and b that make the network's outputs match the training data.

Without the non-linear activation, stacking layers is pointless: the composition of
two linear functions is still a linear function, so a 50-layer linear network has
exactly the same expressive power as a single layer. The non-linearity is what makes
depth useful.

## Activation functions

**ReLU** (Rectified Linear Unit) is defined as `f(x) = max(0, x)`. It is the default
choice for hidden layers in most modern architectures. It is cheap to compute, and
its gradient is exactly 1 for positive inputs, which avoids the vanishing gradient
problem that plagued sigmoid networks.

ReLU's weakness is the "dying ReLU" problem: a neuron whose pre-activation is always
negative outputs zero forever and receives zero gradient, so it never recovers.
**Leaky ReLU** (`f(x) = max(0.01x, x)`) and **GELU** address this by allowing a small
negative slope.

**Sigmoid** squashes any input into (0, 1) and is used for binary classification
outputs. **Softmax** generalises this to multi-class outputs, producing a probability
distribution that sums to 1.

**Tanh** maps to (-1, 1) and is zero-centred, which historically made it preferable
to sigmoid for hidden layers.

## Forward pass and backward pass

The **forward pass** computes predictions layer by layer from input to output. The
**backward pass** (backpropagation) computes the gradient of the loss with respect to
every parameter, working from the output backwards using the chain rule.

Backpropagation is not a learning algorithm. It is only an efficient way to compute
gradients. The learning algorithm is the optimiser, which decides what to do with
those gradients.

## Loss functions

The loss function measures how wrong the prediction is. Training minimises it.

- **Mean Squared Error (MSE)** for regression: `mean((y_pred - y_true)^2)`. It
  punishes large errors quadratically, so it is sensitive to outliers.
- **Cross-Entropy Loss** for classification: `-sum(y_true * log(y_pred))`. It
  punishes confident wrong answers very heavily, which is exactly what you want from
  a classifier.
- **Mean Absolute Error (MAE)** for regression when outliers should not dominate.

A common bug is pairing a softmax output layer with a loss function that applies
softmax again internally. In PyTorch, `CrossEntropyLoss` expects raw logits, not
softmax probabilities. Applying softmax twice flattens the gradients and training
stalls.

## Batch size and epochs

An **epoch** is one full pass over the training set. A **batch** is the subset of
examples processed before one parameter update. **Batch size** trades off gradient
quality against speed and memory: larger batches give smoother, more reliable
gradient estimates but each update costs more and generalisation can suffer.

Typical batch sizes are 32 to 256. Very large batches often need a higher learning
rate and a warmup schedule to train stably.
