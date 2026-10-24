#!/usr/bin/env python3
"""Compare logistic and ridge classifiers on a prepared numeric churn table.

Takes a CSV with numeric feature columns and one binary outcome column.
Splits 70/15/15 with stratification, fits the scaler on training rows,
optionally applies SMOTE to training rows only, selects on validation F1,
and scores the selected model once on test.

Input is validated rather than coerced: missing values and non-numeric
columns are rejected with a message, because silently filling or dropping
them changes the experiment without saying so.
"""
import argparse
import json

import pandas as pd

from ml_portfolio.core import add_intercept, ridge_least_squares
from ml_portfolio.optimization import fit_logistic
from ml_portfolio.retention import metrics, prepare_splits

# Ridge predicts the 0/1 target directly, so it is thresholded at 0.5.
# Logistic produces a logit, so its threshold is 0.
THRESHOLD = {'logistic': 0., 'ridge': .5}


def load(csv_path, target_column):
    frame = pd.read_csv(csv_path)
    if target_column not in frame:
        raise ValueError(f'target column {target_column!r} is not in the CSV')
    target = frame.pop(target_column)
    if target.isna().any() or frame.isna().any().any():
        raise ValueError('missing values present; handle them explicitly before training')
    non_numeric = [name for name, column in frame.items()
                   if not pd.api.types.is_numeric_dtype(column)]
    if non_numeric:
        raise ValueError(f'encode these categorical columns before running: {non_numeric}')
    classes = sorted(target.unique().tolist())
    if len(classes) != 2:
        raise ValueError(f'a binary target is required, found {len(classes)} classes')
    y = (target == classes[1]).astype(int).to_numpy()
    return frame.to_numpy(dtype=float), y, classes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--target', default='churned')
    parser.add_argument('--smote', action='store_true',
                        help='oversample the training rows only')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    X, y, classes = load(args.csv, args.target)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare_splits(
        X, y, seed=args.seed, smote=args.smote)

    candidates = []
    for kind, settings in (('logistic', [.01, .1]), ('ridge', [0., .01, .1])):
        for setting in settings:
            w = (fit_logistic(Xtr, ytr, learning_rate=setting) if kind == 'logistic'
                 else ridge_least_squares(Xtr, ytr, regularization=setting))
            predicted = (add_intercept(Xva) @ w >= THRESHOLD[kind]).astype(int)
            candidates.append({'model': kind, 'parameter': setting,
                               'validation_f1': metrics(yva, predicted)['f1'],
                               'weights': w})

    best = max(candidates, key=lambda c: c['validation_f1'])
    test_pred = (add_intercept(Xte) @ best['weights'] >= THRESHOLD[best['model']]).astype(int)

    print(json.dumps({
        'positive_label': str(classes[1]),
        'candidates': [{k: v for k, v in c.items() if k != 'weights'} for c in candidates],
        'selected_model': best['model'],
        'selected_parameter': best['parameter'],
        'validation_f1': best['validation_f1'],
        'test_metrics': metrics(yte, test_pred),
        'n_test': int(len(yte)),
        'smote': bool(args.smote),
        'seed': args.seed,
    }, indent=2))


if __name__ == '__main__':
    main()
