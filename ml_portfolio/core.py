"""Numerical building blocks: closed-form fits, link functions, splitting, scaling.

Everything here is deliberately small and dependency-free so each piece can be
tested in isolation and reused across the five studies. The one rule the whole
package follows is that any transform learned from data is fitted on the
training split alone.
"""
import numpy as np


def add_intercept(X):
    """Prepend a column of ones so the bias is learned like any other weight."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X must be 2D')
    return np.column_stack((np.ones(len(X)), X))


def ordinary_least_squares(X, y):
    """Least-squares fit via `lstsq`, which handles rank deficiency gracefully.

    Solving the normal equations directly would square the condition number;
    `lstsq` uses an SVD and returns the minimum-norm solution when the design
    is rank deficient instead of failing or returning something arbitrary.
    """
    target = np.asarray(y, dtype=float).ravel()
    return np.linalg.lstsq(add_intercept(X), target, rcond=None)[0]


def ridge_least_squares(X, y, regularization=0.0):
    """Ridge regression, with the intercept left unpenalised.

    Implemented by stacking `sqrt(lambda) * I` beneath the design matrix
    rather than forming `X'X + lambda I`, which keeps the same numerical
    advantage as the OLS routine. The first diagonal entry of the penalty
    block is zeroed so shrinkage never pulls the intercept towards zero:
    penalising it would make the fit depend on where the target happens to
    be centred.

    At `regularization=0` this reduces exactly to ordinary least squares, so
    a search that selects zero penalty is reporting that ridge did not help,
    not that ridge helped with a small penalty.
    """
    if regularization < 0:
        raise ValueError('regularization must be nonnegative')
    A = add_intercept(X)
    target = np.asarray(y, dtype=float).ravel()
    penalty = np.eye(A.shape[1])
    penalty[0, 0] = 0
    stacked = np.vstack((A, np.sqrt(regularization) * penalty))
    padded = np.concatenate((target, np.zeros(A.shape[1])))
    return np.linalg.lstsq(stacked, padded, rcond=None)[0]


def sigmoid(z):
    """Logistic function, written so large negative inputs cannot overflow.

    `1 / (1 + exp(-z))` overflows for very negative `z`. `exp(-logaddexp(0, -z))`
    is algebraically identical and stable across the whole real line.
    """
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0., -z))


def softmax(z, axis=-1):
    """Softmax with the max subtracted first, which cannot overflow.

    Subtracting the row maximum leaves the result unchanged mathematically
    because the shift cancels between numerator and denominator, but it
    bounds every exponent at zero.
    """
    z = np.asarray(z, dtype=float)
    shifted = z - z.max(axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def binary_log_loss(logits, labels):
    """Mean binary cross-entropy computed from logits, not from probabilities.

    Going through probabilities first would require a log of a number that can
    underflow to zero. `logaddexp(0, z) - t*z` is the same quantity and stays
    finite.
    """
    t = np.asarray(labels, dtype=float)
    return float(np.mean(np.logaddexp(0., logits) - t * logits))


def stratified_split(X, y, seed=42, fractions=(.7, .15, .15)):
    """Split into train/validation/test keeping each class proportionally present.

    A plain random split can leave a rare class absent from validation or test
    entirely, which silently makes the metric for that class meaningless. This
    splits within each class and then concatenates, so every fold contains
    every class. Each class needs at least three examples for that to be
    possible, and the function says so rather than producing an empty fold.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError('three positive split fractions must sum to one')
    if len(X) != len(y) or len(y) == 0:
        raise ValueError('X and y must be nonempty and aligned')

    rng = np.random.default_rng(seed)
    buckets = [[], [], []]
    for label in np.unique(y):
        index = np.flatnonzero(y == label)
        rng.shuffle(index)
        if len(index) < 3:
            raise ValueError('each class needs at least three observations to be split three ways')
        cut1 = max(1, int(len(index) * fractions[0]))
        cut2 = min(len(index) - 1, max(cut1 + 1, int(len(index) * (fractions[0] + fractions[1]))))
        for bucket, part in zip(buckets, (index[:cut1], index[cut1:cut2], index[cut2:])):
            bucket.extend(part)
    return tuple((X[np.array(b, dtype=int)], y[np.array(b, dtype=int)]) for b in buckets)


class Standardizer:
    """Zero-mean, unit-variance scaling with the statistics fitted once.

    Separating `fit` from `transform` is the whole point: the mean and scale
    come from the training split and are then applied unchanged to validation
    and test. A constant column has zero spread, so its scale is set to one
    rather than dividing by zero.
    """

    def fit(self, X):
        arr = np.asarray(X, dtype=float)
        self.mean_ = arr.mean(axis=0)
        scale = arr.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
