#!/usr/bin/env python3
"""Housing baseline and quadratic model with training-only scaling."""
import json
from pathlib import Path

from ml_portfolio.reproduction import housing_experiment

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    for name, params in (('baseline', {}),
                         ('quadratic', {'quadratic': True, 'decay': .01})):
        result = housing_experiment(root / 'data/housing.pkl', **params)
        out = root / 'results' / f'housing_{name}_project.json'
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + '\n')
        print(f"{name:10s} validation MAE {result['validation_mae']:.5f}  "
              f"test MAE {result['test_mae']:.5f}  (epoch {result['best_epoch']})")
