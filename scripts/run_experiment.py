#!/usr/bin/env python3
"""Small synthetic demonstrations of each model, for exercising the pipeline.

These use generated data, so they run anywhere and in seconds. They show the
call pattern for each routine; the measured results are in the project
READMEs and in `results/`.
"""
import argparse
import json

import numpy as np

from ml_portfolio.core import add_intercept
from ml_portfolio.experiments import compare_directions, rotated_gaussian
from ml_portfolio.optimization import fit_linear_sgd, fit_logistic, fit_softmax_sgd
from ml_portfolio.retention import metrics, prepare_splits


def directional():
    x_to_y, y_to_x = compare_directions(rotated_gaussian())
    return {'x_to_y_intercept_slope': x_to_y.tolist(),
            'y_to_x_intercept_slope': y_to_x.tolist()}


def housing_demo(rng):
    X = rng.normal(size=(500, 13))
    y = X @ rng.normal(size=13) + rng.normal(size=500)
    ix = rng.permutation(len(y))
    train, val, test = ix[:350], ix[350:425], ix[425:]
    w, history = fit_linear_sgd(X[train], y[train], X[val], y[val])
    return {'best_epoch': history.best_epoch,
            'validation_mae': history.best_score,
            'test_mae': float(np.mean(abs(add_intercept(X[test]) @ w - y[test])))}


def classification_demo(rng):
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + X[:, 1] + rng.normal(size=400) * .5 > 0).astype(int)
    (Xtr, ytr), _, (Xte, yte) = prepare_splits(X, y)
    w = fit_logistic(Xtr, ytr)
    return metrics(yte, (add_intercept(Xte) @ w >= 0).astype(int))


def softmax_demo(rng):
    X = rng.normal(size=(700, 8))
    y = np.argmax(X @ rng.normal(size=(8, 3)), axis=1)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare_splits(X, y)
    W, history = fit_softmax_sgd(Xtr, ytr, Xva, yva)
    return {'best_epoch': history.best_epoch,
            'validation_accuracy': history.best_score,
            'test_accuracy': float(np.mean(np.argmax(add_intercept(Xte) @ W, axis=1) == yte))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True,
                        choices=['directional', 'housing-demo',
                                 'classification-demo', 'softmax-demo'])
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    handlers = {'directional': lambda: directional(),
                'housing-demo': lambda: housing_demo(rng),
                'classification-demo': lambda: classification_demo(rng),
                'softmax-demo': lambda: softmax_demo(rng)}
    print(json.dumps(handlers[args.study](), indent=2))


if __name__ == '__main__':
    main()
