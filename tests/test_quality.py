"""Regression checks covering easily overlooked experimental protocol defects."""
import numpy as np
import pytest
from pathlib import Path
from ml_portfolio.core import stratified_split, ridge_least_squares
from ml_portfolio.reproduction import housing_experiment, read_mnist
ROOT = Path(__file__).resolve().parents[1]

def test_split_does_not_drop_samples():
    x=np.arange(40).reshape(20,2); y=np.repeat([0,1],10)
    chunks=stratified_split(x,y,seed=9)
    assert sum(len(yy) for _,yy in chunks)==20
    assert all(set(yy)=={0,1} for _,yy in chunks)
    assert set(np.concatenate([xx[:,0] for xx,_ in chunks])) == set(x[:,0])

def test_split_rejects_too_few_class_samples():
    with pytest.raises(ValueError): stratified_split(np.arange(8).reshape(4,2),np.array([0,1,1,1]))

def test_ridge_does_not_regularize_intercept():
    x=np.zeros((8,1)); y=np.full(8,7.)
    assert np.isclose(ridge_least_squares(x,y,regularization=10)[0],7.)

def test_housing_input_validation():
    with pytest.raises(ValueError): housing_experiment(ROOT/'data/housing.pkl',epochs=0)

def test_mnist_header_and_labels():
    a,b,c,d=read_mnist(ROOT/'data')
    assert a.dtype==np.uint8 and c.dtype==np.uint8
    assert a.shape[1]==784 and set(np.unique(d))==set(range(10))
