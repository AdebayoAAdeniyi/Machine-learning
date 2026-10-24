"""Housing and MNIST experiments, runnable end to end from the command line.

Both experiments follow the same protocol: fit on a training split, select the
checkpoint and any hyperparameter on a validation split, and score the chosen
model once on a test split that nothing else has touched.  Every run records
its full configuration and per-epoch curves to JSON so a result can be traced
back to the settings that produced it.

Note on `housing.pkl`: it is a Python pickle, which can execute arbitrary code
on load.  Only point this at a file whose provenance you trust.
"""
from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Housing regression
# ---------------------------------------------------------------------------

def read_housing(path: str | Path):
    """Load the 506 x 13 housing design matrix and its target vector."""
    with open(path, 'rb') as file:
        X, y = pickle.load(file)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError('Housing features and labels must be aligned 2D/1D arrays')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Housing data contains non-finite values')
    return X, y


def housing_experiment(path, *, quadratic=False, quadratic_scheme='standardized',
                       lr=.001, batch_size=10, epochs=100, decay=0., seed=314,
                       legacy_preprocessing=False, penalize_intercept=None):
    """Mini-batch gradient descent on the housing data, MAE in original units.

    Parameters that change the experiment rather than tune it
    ---------------------------------------------------------
    `legacy_preprocessing`
        Standardise using all 506 rows rather than the 300 training rows.
        Fitting the scaler on everything lets the held-out rows influence the
        transform, which is a mild form of leakage.  Training-only scaling is
        the default; the flag exists so the two protocols can be compared
        directly on the same data.

    `quadratic_scheme`
        How the squared features are built, which matters more than it looks.

        `'standardized'`  square the already-standardised columns and append
                          them unscaled, giving the design `[1, z, z**2]`.
        `'raw'`           square the raw columns first, then standardise
                          everything, giving `[1, std(x), std(x**2)]`.

        The two span the *same* column space, because `z**2` is a linear
        combination of `x**2`, `x` and `1`.  An exact least-squares solve
        would therefore return identical predictions.  Gradient descent run
        for a fixed 100 epochs at a fixed learning rate does not: the two
        parameterisations are conditioned differently, so the optimiser gets
        to a different place in the same budget.  On this dataset that is
        worth about 0.3 MAE, which is larger than the effect of the
        regularisation being tuned.  See docs/METHOD_NOTES.md.

    `penalize_intercept`
        Whether the L2 penalty applies to the bias term.  Defaults to True
        under `quadratic_scheme='standardized'` so that configuration matches
        the protocol the reported figures came from, and False otherwise,
        which is the textbook choice.  The difference here is about 0.0004
        MAE; it is exposed only so neither behaviour is implicit.
    """
    if epochs < 1 or batch_size < 1 or lr <= 0 or decay < 0:
        raise ValueError('epochs, batch_size and learning rate must be positive; decay nonnegative')
    if quadratic_scheme not in ('standardized', 'raw'):
        raise ValueError("quadratic_scheme must be 'standardized' or 'raw'")
    if penalize_intercept is None:
        penalize_intercept = quadratic and quadratic_scheme == 'standardized'

    X, y = read_housing(path)

    if quadratic and quadratic_scheme == 'raw':
        X = np.column_stack((X, X ** 2))

    # The split is a seeded permutation of the row order, held fixed across
    # every configuration so that comparisons are on identical folds.
    indices = np.random.RandomState(seed).permutation(len(y))
    Xtr, Xva, Xte = X[indices[:300]], X[indices[300:400]], X[indices[400:]]
    ytr, yva, yte = y[indices[:300]], y[indices[300:400]], y[indices[400:]]

    scale_X = X if legacy_preprocessing else Xtr
    scale_y = y if legacy_preprocessing else ytr
    x_mean, x_std = scale_X.mean(axis=0), scale_X.std(axis=0)
    x_std[x_std == 0] = 1
    y_mean, y_std = scale_y.mean(), scale_y.std()
    if y_std == 0:
        raise ValueError('Targets are constant; MAE in original units is undefined')

    def design(block):
        z = (block - x_mean) / x_std
        if quadratic and quadratic_scheme == 'standardized':
            z = np.column_stack((z, z ** 2))
        return np.column_stack((np.ones(len(block)), z))

    Atr, Ava, Ate = design(Xtr), design(Xva), design(Xte)
    target = ((ytr - y_mean) / y_std).reshape(-1, 1)

    w = np.zeros((Atr.shape[1], 1))
    best_w, best_val, best_epoch = w.copy(), float('inf'), 0
    train_losses, val_curve = [], []

    for epoch in range(1, epochs + 1):
        # Deterministic batch order: the split is already a random permutation,
        # so reshuffling every epoch would add variance without adding
        # information, and it would make the curve irreproducible across runs.
        for start in range(0, len(Atr), batch_size):
            xb = Atr[start:start + batch_size]
            yb = target[start:start + batch_size]
            penalty = w.copy()
            if not penalize_intercept:
                penalty[0] = 0
            # Divide by the actual batch length so a short final batch is not
            # given an inflated gradient.
            w -= lr * (2 * xb.T @ (xb @ w - yb) / len(xb) + 2 * decay * penalty)

        train_losses.append(float(np.mean((Atr @ w - target) ** 2)))
        mae = float(np.mean(np.abs((Ava @ w).reshape(-1) * y_std + y_mean - yva)))
        val_curve.append(mae)
        if mae < best_val:
            best_val, best_epoch, best_w = mae, epoch, w.copy()

    test_mae = float(np.mean(np.abs((Ate @ best_w).reshape(-1) * y_std + y_mean - yte)))
    return {
        'seed': seed,
        'preprocessing': 'all 506 rows (legacy)' if legacy_preprocessing else 'training rows only',
        'quadratic': quadratic,
        'quadratic_scheme': quadratic_scheme if quadratic else None,
        'penalize_intercept': bool(penalize_intercept),
        'alpha': lr, 'batch_size': batch_size, 'epochs': epochs, 'decay': decay,
        'best_epoch': best_epoch,
        'validation_mae': best_val,
        'test_mae': test_mae,
        'training_loss': train_losses,
        'validation_curve': val_curve,
    }


def housing_decay_search(path, candidates=(3, 1, .3, .1, .03, .01), **kwargs):
    """Sweep the L2 penalty, choosing the value with the lowest validation MAE.

    Ties and the final choice are decided on validation only; the test score
    of the selected configuration is reported but never used to select.
    """
    trials = [housing_experiment(path, decay=float(d), **kwargs) for d in candidates]
    best = min(trials, key=lambda r: r['validation_mae'])
    return {
        'candidates': [
            {'decay': t['decay'], 'validation_mae': t['validation_mae'],
             'test_mae': t['test_mae'], 'best_epoch': t['best_epoch']}
            for t in trials
        ],
        'selected_decay': best['decay'],
        'validation_mae': best['validation_mae'],
        'test_mae': best['test_mae'],
        'best_epoch': best['best_epoch'],
        'settings': {k: best[k] for k in
                     ('seed', 'preprocessing', 'quadratic', 'quadratic_scheme',
                      'penalize_intercept', 'alpha', 'batch_size', 'epochs')},
    }


# ---------------------------------------------------------------------------
# MNIST softmax regression
# ---------------------------------------------------------------------------

def _read_idx(path, images):
    """Parse one IDX file, validating the magic number and declared shape."""
    with open(path, 'rb') as handle:
        magic, count = struct.unpack('>II', handle.read(8))
        expected = 2051 if images else 2049
        if magic != expected:
            raise ValueError(f'Unexpected IDX magic number in {path}: {magic}')
        if images:
            rows, cols = struct.unpack('>II', handle.read(8))
            data = np.frombuffer(handle.read(), dtype=np.uint8)
            if data.size != count * rows * cols:
                raise ValueError(f'{path}: declared {count} images but found {data.size} bytes')
            return data.reshape(count, rows * cols)
        data = np.frombuffer(handle.read(), dtype=np.uint8)
        if len(data) != count:
            raise ValueError(f'{path}: declared {count} labels but found {len(data)}')
        return data


def read_mnist(folder):
    """Load the four uncompressed IDX files into arrays."""
    folder = Path(folder)
    train_X = _read_idx(folder / 'train-images-idx3-ubyte', True)
    train_y = _read_idx(folder / 'train-labels-idx1-ubyte', False)
    test_X = _read_idx(folder / 't10k-images-idx3-ubyte', True)
    test_y = _read_idx(folder / 't10k-labels-idx1-ubyte', False)
    if len(train_X) != len(train_y) or len(test_X) != len(test_y):
        raise ValueError('MNIST image and label counts disagree')
    return train_X, train_y, test_X, test_y


def mnist_experiment(folder, *, lr=.1, batch_size=50, epochs=60, decay=0., seed=314,
                     verbose=True):
    """Multinomial softmax regression on MNIST by mini-batch SGD.

    The 60,000 labelled training images are split 50,000 / 10,000 into model
    fitting and validation.  The separate 10,000-image test set is scored once,
    with the weights from the best validation epoch.
    """
    if epochs < 1 or batch_size < 1 or lr <= 0 or decay < 0:
        raise ValueError('epochs, batch_size and learning rate must be positive; decay nonnegative')

    train_X, train_y, test_X, test_y = read_mnist(folder)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_y))
    fit_ix, val_ix = order[:50000], order[50000:]

    x = train_X[fit_ix].astype('float32') / 255
    x_val = train_X[val_ix].astype('float32') / 255
    x_test = test_X.astype('float32') / 255
    y = train_y[fit_ix].astype('int64')
    y_val = train_y[val_ix].astype('int64')

    # Bias is a separate vector so that weight decay never touches it.
    w = np.zeros((x.shape[1], 10), dtype=np.float32)
    b = np.zeros(10, dtype=np.float32)
    best_w, best_b, best_accuracy, best_epoch = w.copy(), b.copy(), -1., 0
    losses, validations = [], []

    def predict_in_chunks(data, weights, bias, size=4096):
        return np.concatenate([np.argmax(data[i:i + size] @ weights + bias, axis=1)
                               for i in range(0, len(data), size)])

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(y))
        for begin in range(0, len(indices), batch_size):
            ix = indices[begin:begin + batch_size]
            scores = x[ix] @ w + b
            scores -= scores.max(axis=1, keepdims=True)   # max-shift: no overflow
            probs = np.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(ix)), y[ix]] -= 1
            w -= lr * (x[ix].T @ probs / len(ix) + decay * w)
            b -= lr * probs.mean(axis=0)

        # Cross-entropy over the whole training set, in chunks to bound memory.
        # Averaging the per-batch losses collected during the epoch would mix
        # together losses computed under different weights.
        total = 0.
        for begin in range(0, len(y), 4096):
            scores = x[begin:begin + 4096] @ w + b
            labels = y[begin:begin + 4096]
            peak = scores.max(axis=1)
            total += float((peak + np.log(np.exp(scores - peak[:, None]).sum(axis=1))
                            - scores[np.arange(len(labels)), labels]).sum())
        losses.append(total / len(y))

        accuracy = 100 * float(np.mean(predict_in_chunks(x_val, w, b) == y_val))
        validations.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch, best_w, best_b = accuracy, epoch, w.copy(), b.copy()
        if verbose:
            print(f'epoch {epoch:3d}/{epochs}  train CE {losses[-1]:.4f}  '
                  f'val accuracy {accuracy:.2f}%', flush=True)

    test_accuracy = 100 * float(np.mean(predict_in_chunks(x_test, best_w, best_b) == test_y))
    return {
        'seed': seed, 'alpha': lr, 'batch_size': batch_size, 'epochs': epochs, 'decay': decay,
        'training_images': int(len(y)), 'validation_images': int(len(y_val)),
        'test_images': int(len(test_y)),
        'best_epoch': best_epoch,
        'validation_accuracy_pct': best_accuracy,
        'test_accuracy_pct': test_accuracy,
        'training_loss': losses,
        'validation_curve': validations,
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the housing and MNIST experiments and save metrics to JSON')
    sub = parser.add_subparsers(dest='dataset', required=True)

    housing = sub.add_parser('housing', help='mini-batch gradient descent on housing prices')
    housing.add_argument('--data', required=True)
    housing.add_argument('--quadratic', action='store_true',
                         help='append squared features')
    housing.add_argument('--quadratic-scheme', choices=['standardized', 'raw'],
                         default='standardized', dest='quadratic_scheme',
                         help='square the standardised columns (default) or the raw ones')
    housing.add_argument('--legacy-preprocessing', action='store_true',
                         help='standardise using all 506 rows instead of the training rows')
    housing.add_argument('--search-decay', action='store_true', dest='search_decay',
                         help='sweep the L2 penalty over 3, 1, 0.3, 0.1, 0.03, 0.01 '
                              'and select on validation MAE')

    mnist = sub.add_parser('mnist', help='multinomial softmax regression on MNIST')
    mnist.add_argument('--data-dir', required=True)

    for sp in (housing, mnist):
        sp.add_argument('--lr', type=float, default=None)
        sp.add_argument('--batch-size', type=int, default=None)
        sp.add_argument('--epochs', type=int, default=None)
        sp.add_argument('--decay', type=float, default=0.)
        sp.add_argument('--seed', type=int, default=314)
        sp.add_argument('--output', default=None)

    args = parser.parse_args(argv)
    defaults = {'housing': (.001, 10, 100), 'mnist': (.1, 50, 60)}
    lr, batch, epochs = defaults[args.dataset]
    common = {'lr': args.lr if args.lr is not None else lr,
              'batch_size': args.batch_size or batch,
              'epochs': args.epochs or epochs,
              'seed': args.seed}

    if args.dataset == 'housing':
        shared = dict(common, quadratic=args.quadratic,
                      quadratic_scheme=args.quadratic_scheme,
                      legacy_preprocessing=args.legacy_preprocessing)
        result = (housing_decay_search(args.data, **shared) if args.search_decay
                  else housing_experiment(args.data, decay=args.decay, **shared))
    else:
        result = mnist_experiment(args.data_dir, decay=args.decay, **common)

    out = Path(args.output or f'{args.dataset}_metrics.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    summary = {k: v for k, v in result.items() if not isinstance(v, list)}
    print(json.dumps(summary, indent=2))
    print('Saved', out)
    return result


if __name__ == '__main__':
    main()
