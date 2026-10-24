"""Mini-batch gradient training with validation checkpointing.

The three routines share one protocol. Each epoch walks the training data in
shuffled mini-batches, then evaluates once on validation and keeps the weights
if that score is the best so far. The returned weights are the checkpoint, not
the final epoch, so a run that overfits late still returns its best model.

Two details that are easy to get wrong and are handled explicitly here:

* the gradient is divided by the **actual** length of each batch, so a short
  final batch does not receive an inflated step;
* the epoch training loss is recomputed over the whole training set after the
  epoch, rather than averaging the per-batch losses seen during it. Those
  per-batch values were each computed under different weights, so averaging
  them describes no single model.
"""
from dataclasses import dataclass

import numpy as np

from .core import add_intercept, sigmoid, softmax


@dataclass
class TrainingHistory:
    """Per-epoch curves plus the checkpoint that was selected."""
    training_loss: list
    validation_score: list
    best_epoch: int
    best_score: float


def fit_linear_sgd(X, y, X_val, y_val, learning_rate=.01, epochs=100, batch_size=32,
                   l2=0., seed=42, output_scale=1.):
    """Linear regression by mini-batch SGD, checkpointed on validation MAE.

    `output_scale` multiplies the validation error so the reported figure can
    be in the original target units even when the model is trained on
    standardised targets.
    """
    if batch_size < 1 or epochs < 1:
        raise ValueError('batch size and epochs must be positive')

    A = add_intercept(X)
    V = add_intercept(X_val)
    t = np.asarray(y, dtype=float).ravel()
    v = np.asarray(y_val, dtype=float).ravel()
    if len(A) != len(t):
        raise ValueError('features and targets have different sample counts')

    rng = np.random.default_rng(seed)
    w = np.zeros(A.shape[1])
    best_w, best_score, best_epoch = w.copy(), float('inf'), 0
    losses, scores = [], []

    for epoch in range(1, epochs + 1):
        batches = np.array_split(rng.permutation(len(t)),
                                 max(1, int(np.ceil(len(t) / batch_size))))
        for batch in batches:
            residual = A[batch] @ w - t[batch]
            gradient = A[batch].T @ residual / len(batch)
            penalty = w.copy()
            penalty[0] = 0                      # never shrink the intercept
            w -= learning_rate * (gradient + l2 * penalty)

        losses.append(float(np.mean((A @ w - t) ** 2) / 2))
        score = float(np.mean(np.abs(V @ w - v)) * output_scale)
        scores.append(score)
        if score < best_score:
            best_score, best_epoch, best_w = score, epoch, w.copy()

    return best_w, TrainingHistory(losses, scores, best_epoch, best_score)


def fit_logistic(X, y, learning_rate=.1, epochs=1000, l2=0.):
    """Logistic regression by full-batch gradient descent on cross-entropy.

    The gradient of the log-loss with respect to the weights is
    `X' (sigmoid(Xw) - y) / n`, which needs no explicit loss evaluation.
    Labels must be 0/1; anything else is rejected rather than silently
    coerced, because a {-1, +1} encoding would train towards the wrong target.
    """
    A = add_intercept(X)
    t = np.asarray(y, dtype=float).ravel()
    if not set(np.unique(t)).issubset({0., 1.}):
        raise ValueError('binary labels must be encoded as 0 and 1')

    w = np.zeros(A.shape[1])
    for _ in range(epochs):
        gradient = A.T @ (sigmoid(A @ w) - t) / len(t)
        penalty = w.copy()
        penalty[0] = 0
        w -= learning_rate * (gradient + l2 * penalty)
    return w


def fit_softmax_sgd(X, y, X_val, y_val, learning_rate=.1, epochs=50, batch_size=100,
                    l2=0., seed=42):
    """Multinomial softmax regression by mini-batch SGD, checkpointed on accuracy.

    The gradient of the multinomial cross-entropy is `X' (P - Y) / n`, where
    `P` holds the predicted class probabilities and `Y` is the one-hot target.
    Subtracting 1 from the true-class column of `P` builds `P - Y` in place
    without materialising the one-hot matrix.

    Note that the optimised objective (cross-entropy) and the selection metric
    (accuracy) are deliberately different: accuracy is what the task is judged
    on, but it is piecewise constant and so useless as a gradient signal.
    """
    A = add_intercept(X)
    V = add_intercept(X_val)
    t = np.asarray(y, dtype=int)
    v = np.asarray(y_val, dtype=int)
    n_classes = int(max(t.max(), v.max()) + 1)

    rng = np.random.default_rng(seed)
    W = np.zeros((A.shape[1], n_classes))
    best_W, best_score, best_epoch = W.copy(), -1., 0
    losses, scores = [], []

    for epoch in range(1, epochs + 1):
        batches = np.array_split(rng.permutation(len(t)),
                                 max(1, int(np.ceil(len(t) / batch_size))))
        for ix in batches:
            P = softmax(A[ix] @ W)
            P[np.arange(len(ix)), t[ix]] -= 1
            penalty = W.copy()
            penalty[0] = 0
            W -= learning_rate * (A[ix].T @ P / len(ix) + l2 * penalty)

        logits = A @ W
        # log-sum-exp over classes, computed without exponentiating raw scores
        losses.append(float(np.mean(np.logaddexp.reduce(logits, axis=1)
                                    - logits[np.arange(len(t)), t])))
        score = float(np.mean(np.argmax(V @ W, axis=1) == v))
        scores.append(score)
        if score > best_score:
            best_score, best_epoch, best_W = score, epoch, W.copy()

    return best_W, TrainingHistory(losses, scores, best_epoch, best_score)
