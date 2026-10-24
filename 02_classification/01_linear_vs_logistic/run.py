#!/usr/bin/env python3
"""Thresholded least squares against logistic regression on two 2D datasets.

Dataset A is two well-separated Gaussian blobs. Dataset B is the same, except
half of the positive class is translated far away from the boundary. Those
extra points are still trivially on the correct side, so a good classifier
should be unaffected by them.

Least squares is affected, because it fits the 0/1 targets by squared error:
a point far on the correct side has a large residual, and the fit rotates the
line to reduce it. Logistic regression's loss saturates for confidently
correct points, so they stop contributing gradient and the boundary stays put.
"""
import json
from pathlib import Path

import numpy as np

from ml_portfolio.optimization import fit_logistic

SEED = 314
N_PER_CLASS = 200
SHIFT = 3.0


def main():
    rng = np.random.default_rng(SEED)
    results = []
    for case in ('A', 'B'):
        covariance = [[.06, 0], [0, .06]]
        negative = rng.multivariate_normal([-.2, -.2], covariance, size=N_PER_CLASS)
        positive = rng.multivariate_normal([.2, .2], covariance, size=N_PER_CLASS)
        if case == 'B':
            # Push half the positive class away from the boundary. These points
            # are even easier to classify, so a sound method should not degrade.
            positive[:N_PER_CLASS // 2] += SHIFT

        X = np.vstack((negative, positive))
        y = np.r_[np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)]
        Xb = np.column_stack((np.ones(len(X)), X))

        ols = np.linalg.lstsq(Xb, y, rcond=None)[0]
        logistic = fit_logistic(X, y, learning_rate=.1, epochs=2000)
        results.append({
            'dataset': case,
            'ols_training_accuracy': float(np.mean((Xb @ ols >= .5) == y)),
            'logistic_training_accuracy': float(np.mean((Xb @ logistic >= 0) == y)),
        })

    root = Path(__file__).resolve().parents[2]
    (root / 'results').mkdir(exist_ok=True)
    (root / 'results/linear_logistic_synthetic.json').write_text(
        json.dumps(results, indent=2) + '\n')
    for r in results:
        print(f"dataset {r['dataset']}: least squares {r['ols_training_accuracy']:.4f}  "
              f"logistic {r['logistic_training_accuracy']:.4f}")
    print('\nSaved results/linear_logistic_synthetic.json')


if __name__ == '__main__':
    main()
