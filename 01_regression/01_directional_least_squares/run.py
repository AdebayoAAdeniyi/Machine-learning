#!/usr/bin/env python3
"""Fit x -> y and y -> x on the same rotated Gaussian cloud and compare.

Varies the minor-axis variance and the rotation angle over a 3 x 3 grid and
records both fitted slopes, both intercepts, their product, and the closed
form each should match.

The product of the two slopes is exactly the squared correlation:

    slope(y|x) = cov / var_x,   slope(x|y) = cov / var_y
    product    = cov^2 / (var_x var_y) = r^2

so the two directions give reciprocal slopes only when r^2 = 1, that is when
the cloud is a perfect line. Everything else in this study follows from that
one identity, and the run checks it against the sampled estimates.
"""
import json
from pathlib import Path

import numpy as np

# `var1` and `var2` are variances, matching the exercise and
# `ml_portfolio.experiments.rotated_gaussian`. Passing them to `np.random.normal`
# as `scale`, which expects a standard deviation, would silently square the
# minor axis and change every number below.
MAJOR_VARIANCE = 1.0
MINOR_VARIANCES = (0.1, 0.3, 0.8)
ANGLES_DEG = (30, 45, 60)
N_SAMPLES = 5000
SEED = 314


def theoretical(angle_deg, minor_variance, major_variance=MAJOR_VARIANCE):
    """Population slopes and r^2 for a rotated Gaussian with these axes."""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    var_x = major_variance * c * c + minor_variance * s * s
    var_y = major_variance * s * s + minor_variance * c * c
    covariance = c * s * (major_variance - minor_variance)
    return {'x_to_y_slope': covariance / var_x,
            'y_to_x_slope': covariance / var_y,
            'r_squared': covariance ** 2 / (var_x * var_y)}


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for angle in ANGLES_DEG:
        for minor_variance in MINOR_VARIANCES:
            major = rng.normal(scale=np.sqrt(MAJOR_VARIANCE), size=N_SAMPLES)
            minor = rng.normal(scale=np.sqrt(minor_variance), size=N_SAMPLES)
            theta = np.deg2rad(angle)
            x = major * np.cos(theta) - minor * np.sin(theta)
            y = major * np.sin(theta) + minor * np.cos(theta)

            fit_xy = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
            fit_yx = np.linalg.lstsq(np.column_stack((y, np.ones(len(y)))), x, rcond=None)[0]
            expected = theoretical(angle, minor_variance)
            rows.append({
                'angle_deg': angle,
                'minor_axis_variance': minor_variance,
                'x_to_y_slope': float(fit_xy[0]),
                'y_to_x_slope': float(fit_yx[0]),
                'x_to_y_intercept': float(fit_xy[1]),
                'y_to_x_intercept': float(fit_yx[1]),
                'slope_product': float(fit_xy[0] * fit_yx[0]),
                'theoretical_x_to_y_slope': expected['x_to_y_slope'],
                'theoretical_r_squared': expected['r_squared'],
            })

    root = Path(__file__).resolve().parents[2]
    (root / 'results').mkdir(exist_ok=True)
    (root / 'results/directional_regression.json').write_text(json.dumps(rows, indent=2) + '\n')

    header = f"{'angle':>6} {'minor var':>10} {'x->y':>8} {'y->x':>8} {'product':>9} {'r^2 (theory)':>13}"
    print(header)
    for r in rows:
        print(f"{r['angle_deg']:>6} {r['minor_axis_variance']:>10} "
              f"{r['x_to_y_slope']:>8.4f} {r['y_to_x_slope']:>8.4f} "
              f"{r['slope_product']:>9.4f} {r['theoretical_r_squared']:>13.4f}")
    print('\nSaved results/directional_regression.json')


if __name__ == '__main__':
    main()
