# When least squares fails as a classifier

**Question.** Thresholded least squares and logistic regression both produce
a linear decision boundary. When do they disagree, and why?

## The setup

Two 2D Gaussian classes, 200 points each.

- **Dataset A**: the classes sit either side of the origin, well separated.
- **Dataset B**: identical, except half of the positive class is translated
  by +3 in both coordinates, moving it *further onto its own correct side*.

Those shifted points are easier, not harder. Any sound classifier should be
unaffected by them. That is what makes B a diagnostic rather than just a
different dataset.

## Results

Training accuracy, seed 314, from
[`results/linear_logistic_synthetic.json`](../../results/linear_logistic_synthetic.json):

| Dataset | Thresholded least squares | Logistic regression |
|---|---:|---:|
| A, separated classes | 0.8925 | 0.8975 |
| B, half the positives moved further away | **0.7500** | **0.9075** |

On A the two methods are within half a point of each other. On B, adding
points that are easier to classify costs least squares 14 points of accuracy
while logistic regression is unchanged.

## Why

Least squares fits the 0/1 labels by squared error. A point sitting at
distance 4 on the correct side of the boundary still has a large residual
against its target of 1, and squared error grows with the square of that
residual. The fit therefore rotates the line to reduce an error on points
that were never in any danger of being misclassified, and the boundary
drifts off the region where the classes actually meet.

Logistic regression optimises cross-entropy on `σ(w·x)`. For a confidently
correct point the predicted probability is near 1, the gradient contribution
`(σ(w·x) − y)` is near zero, and the point stops influencing the fit. The
loss saturates exactly where least squares keeps accumulating penalty.

This is the standard argument for why a classification loss should be used
for classification, and dataset B is constructed so the mechanism is visible
in a single number rather than only in a plot.

## A note on the metric

These are **training** accuracies. The point being made is about the fitted
decision boundary on the data the models were given, so held-out performance
is not the relevant quantity here. Any claim about generalisation would need
a test split, and nothing in this study makes one.

## Techniques demonstrated

Choosing a loss function to match the task; deriving and implementing the
logistic gradient; constructing a synthetic stress test where the expected
failure is isolated to one cause; and reporting the metric that matches the
claim rather than the one that sounds strongest.

## Reproduce

```bash
python 02_classification/01_linear_vs_logistic/run.py
```

Logistic regression comes from the shared
[`fit_logistic`](../../ml_portfolio/optimization.py); no external ML library
is required.
