# Overfitting and Regularization

## The core problem

A model that memorises the training set is useless. What matters is performance on
data it has never seen. **Overfitting** is when training loss keeps falling while
validation loss starts rising: the model is learning noise specific to the training
examples rather than the underlying pattern.

**Underfitting** is the opposite failure: the model is too simple to capture the
pattern, so both training and validation loss stay high.

The diagnostic is always the same. Plot training and validation loss on the same
axes:

- Both high and flat: underfitting. Increase capacity or train longer.
- Training low, validation high and rising: overfitting. Apply regularization.
- Both low and close: healthy.

## The bias-variance tradeoff

**Bias** is error from wrong assumptions -- the model is systematically off. High
bias means underfitting. **Variance** is sensitivity to the particular training
sample -- retrain on a slightly different dataset and you get a very different model.
High variance means overfitting.

Increasing model capacity lowers bias and raises variance. Regularization does the
reverse. The goal is the point where total error is minimised, not zero of either.

## Dropout

**Dropout** randomly sets a fraction p of activations to zero during each training
step. A typical value is p = 0.5 for fully connected layers and p = 0.1 to 0.3 for
convolutional or transformer layers.

Why it works: a neuron cannot rely on any specific other neuron being present, so the
network cannot build fragile co-adapted chains of neurons. It is forced to learn
redundant, robust representations. It also approximates training an ensemble of
exponentially many sub-networks and averaging them.

Critically, dropout is **only active during training**. At inference time all neurons
are used and activations are scaled to compensate. Forgetting to call `model.eval()`
in PyTorch leaves dropout on during evaluation and produces mysteriously poor and
non-deterministic validation scores. This is one of the most common bugs in practice.

## L1 and L2 regularization

Both add a penalty term to the loss that discourages large weights.

**L2 regularization** (also called weight decay, or ridge) adds `lambda * sum(w^2)`.
It shrinks all weights smoothly toward zero but rarely to exactly zero. It is the
default choice.

**L1 regularization** (lasso) adds `lambda * sum(|w|)`. Its gradient is constant, so
it drives weights to exactly zero, producing a sparse model. This makes it useful for
feature selection: the features whose weights survive are the ones that matter.

**Elastic Net** combines both.

The hyperparameter lambda controls strength. Too small and there is no effect; too
large and the model underfits.

## Early stopping

Monitor validation loss each epoch and stop when it stops improving for a set number
of epochs (the "patience"). Keep the checkpoint from the best epoch, not the last
one. This is the cheapest regularization technique available and should almost always
be used.

## Data augmentation

Artificially expand the training set with label-preserving transformations. For
images: random crops, flips, rotations, colour jitter. For text: synonym replacement,
back-translation. More effective training data is usually a better investment than a
more sophisticated model.

## Cross-validation

**k-fold cross-validation** splits the data into k parts, trains k times, each time
holding out a different fold for validation, and averages the results. It gives a far
more reliable estimate than a single train/validation split, at k times the compute
cost. Use it when data is scarce; a single held-out split is fine when data is
plentiful.

Never let information from the validation or test set leak into training. Fit scalers
and encoders on the training fold only, then apply them to the validation fold.
Fitting a `StandardScaler` on the full dataset before splitting is a subtle and very
common form of data leakage.
