import numpy as np
from ml_portfolio.reproduction import housing_experiment,read_mnist
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_housing_validation_and_test_are_finite():
    r=housing_experiment(ROOT/'data/housing.pkl',epochs=2)
    assert np.isfinite(r['validation_mae']) and np.isfinite(r['test_mae'])
    assert 1 <= r['best_epoch'] <= 2
def test_mnist_idx_integrity():
    a,b,c,d=read_mnist(ROOT/'data')
    assert a.shape==(60000,784) and c.shape==(10000,784)
    assert b.shape==(60000,) and d.shape==(10000,)
    assert np.all((b>=0)&(b<=9))
