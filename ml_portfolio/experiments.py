"""Directional least squares on a rotated bivariate Gaussian."""
import numpy as np

from .core import ordinary_least_squares


def rotated_gaussian(n=5000, var1=1., var2=.3, angle=45., seed=42):
    """Sample a 2D Gaussian with given axis variances, rotated by `angle`.

    Drawing along the axes and then rotating is equivalent to sampling from a
    covariance with eigenvalues `var1`, `var2` and eigenvectors at `angle`,
    and it makes the two controls in this study independent: `var2` sets how
    tight the cloud is about its major axis, `angle` sets its orientation.
    """
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n, 2)) * np.sqrt([var1, var2])
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return samples @ rotation.T


def compare_directions(data):
    """Fit x -> y and y -> x on the same cloud and return both coefficient pairs.

    These minimise different residuals: vertical distance in one case,
    horizontal in the other. The slopes are not reciprocals of each other
    except in the degenerate noiseless case, which is the point of the study.
    """
    x, y = data[:, 0], data[:, 1]
    return ordinary_least_squares(x[:, None], y), ordinary_least_squares(y[:, None], x)
