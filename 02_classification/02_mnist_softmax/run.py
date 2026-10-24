#!/usr/bin/env python3
"""Train the MNIST softmax model for 60 epochs and save the metrics."""
import json
from pathlib import Path

from ml_portfolio.reproduction import mnist_experiment

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    result = mnist_experiment(root / 'data', epochs=60)
    out = root / 'results/mnist_project.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(f"validation {result['validation_accuracy_pct']:.2f}%  "
          f"test {result['test_accuracy_pct']:.2f}%  "
          f"(best epoch {result['best_epoch']})")
