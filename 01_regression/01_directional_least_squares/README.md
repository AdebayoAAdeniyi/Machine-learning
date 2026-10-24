# Directional least squares: why `x → y` is not `y → x`

**Question.** Fitting a line to predict `y` from `x` and fitting one to
predict `x` from `y` describe the same cloud of points. Do they recover the
same relationship?

## Why they differ

The two fits minimise different residuals:

```
x → y:   min over (a, b)  Σ (y_i − a x_i − b)²        vertical distance
y → x:   min over (c, d)  Σ (x_i − c y_i − d)²        horizontal distance
```

so there is no reason for `a` and `1/c` to agree. The exact relationship is
short and worth deriving, because it explains every number in this study:

```
a = cov(x, y) / var(x)        c = cov(x, y) / var(y)
a · c = cov(x, y)² / (var(x) var(y)) = r²
```

The product of the two slopes is the squared correlation. The two directions
give reciprocal slopes exactly when `r² = 1`, that is when the points lie on
a perfect line. Everything else is a consequence of how much scatter there
is about the major axis.

## Design

Sample 5,000 points from a bivariate Gaussian with major-axis variance 1 and
minor-axis variance `v`, then rotate by `θ`. Sweep `v ∈ {0.1, 0.3, 0.8}` and
`θ ∈ {30°, 45°, 60°}`, fit both directions, and compare the product of the
slopes against the `r²` predicted by the covariance algebra above.

Rotating a cloud drawn along its own axes is equivalent to sampling from a
covariance matrix with those eigenvalues and that orientation, and it keeps
the two controls independent: `v` sets the scatter, `θ` sets the direction.

## Results

Seed 314, from [`results/directional_regression.json`](../../results/directional_regression.json):

| angle | minor variance | `x → y` slope | `y → x` slope | product | `r²` (theory) |
|---:|---:|---:|---:|---:|---:|
| 30° | 0.1 | 0.5067 | 1.1822 | 0.5990 | 0.6030 |
| 30° | 0.3 | 0.3746 | 0.6400 | 0.2398 | 0.2344 |
| 30° | 0.8 | 0.0895 | 0.1039 | 0.0093 | 0.0093 |
| 45° | 0.1 | 0.8282 | 0.8016 | 0.6639 | 0.6694 |
| 45° | 0.3 | 0.5466 | 0.5241 | 0.2865 | 0.2899 |
| 45° | 0.8 | 0.0985 | 0.0989 | 0.0097 | 0.0123 |
| 60° | 0.1 | 1.1963 | 0.4952 | 0.5924 | 0.6030 |
| 60° | 0.3 | 0.6304 | 0.3507 | 0.2211 | 0.2344 |
| 60° | 0.8 | 0.1136 | 0.0989 | 0.0112 | 0.0093 |

The sampled slope products track the predicted `r²` across the whole grid,
which confirms the derivation rather than merely illustrating it.

**The scatter, not the angle, drives the asymmetry.** At minor variance 0.1
the two slopes at 30° differ by a factor of 2.3. At 0.8 they agree to three
decimal places, because `r²` has collapsed to 0.01 and both fits have
flattened towards zero slope: when the cloud is nearly circular, the best
linear prediction in either direction is close to "predict the mean".

**At 45° the two slopes nearly coincide**, since the symmetry makes
`var(x) = var(y)` and the two denominators equal. That is a property of the
orientation, not evidence that the directions agree in general: at 30° and
60° the same variances give very different pairs.

**A note on units.** The variance parameters are variances, not standard
deviations. Passing them to a sampler that expects a scale would square the
minor axis and change every number here: at 45° with `v = 0.3` the slope is
0.5385 if `v` is a variance and 0.8349 if it is read as a standard
deviation. The script fixes the convention explicitly for that reason.

## Takeaway

"The regression of y on x" is a specific claim about which variable carries
the error, not a symmetric statement about association. Reporting a slope
without saying which direction was fitted leaves the number ambiguous by a
factor that is `1/r²`, which on the 0.8-variance rows here is about a
hundred.

## Reproduce

```bash
python 01_regression/01_directional_least_squares/run.py
```

Implementation: [`run.py`](run.py) and
[`ml_portfolio/experiments.py`](../../ml_portfolio/experiments.py).
