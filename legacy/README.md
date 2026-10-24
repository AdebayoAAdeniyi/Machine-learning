# Original coursework scripts

These are the scripts as I first wrote them for CMPUT 566 and the banking
retention project, kept unchanged so the provenance of the work is visible.
They contain the machine-specific paths and helper imports they ran with at
the time, so they will not run as-is elsewhere.

The version to read and run is the package in
[`ml_portfolio/`](../ml_portfolio/), which implements the same models with
portable I/O, tests and command-line runners.

| Script | What it does |
|---|---|
| `01_regression/01_directional_least_squares/original_Q1.py` | Least squares fitted in both directions on a rotated Gaussian |
| `01_regression/01_directional_least_squares/original_variance_rotation_study.py` | Sweep over minor-axis variance and rotation angle |
| `01_regression/02_housing_gradient_descent/original_baseline.py` | Mini-batch gradient descent on housing prices, validation checkpointing |
| `01_regression/02_housing_gradient_descent/original_quadratic_regularization.py` | Squared features with an L2 sweep |
| `01_regression/02_housing_gradient_descent/original_hyperparameter_search.py` | Wider search over learning rate, batch size and penalty |
| `02_classification/01_linear_vs_logistic/original_comparison.py` | Thresholded least squares against logistic regression |
| `02_classification/02_mnist_softmax/original_softmax.py` | Multinomial softmax regression on MNIST |
| `02_classification/02_mnist_softmax/original_search.py` | Hyperparameter search for the softmax model |
| `03_capstone/customer_retention_prediction/original_project.py` | SQL feature integration, SMOTE, and the three-model churn comparison |

## What the rewrite changed

The reasoning for each decision is in
[`docs/METHOD_NOTES.md`](../docs/METHOD_NOTES.md). In brief:

- portable paths and a command-line interface in place of a hard-coded
  working directory;
- numerically stable sigmoid, softmax and cross-entropy;
- gradients divided by the actual batch length rather than the nominal
  batch size;
- epoch training loss recomputed over the whole training set rather than
  accumulated from per-batch values;
- scalers fitted on training rows by default, with the all-rows protocol
  still available through `--legacy-preprocessing` so the two can be
  compared on identical folds;
- the squared-feature parameterisation made explicit through
  `--quadratic-scheme`, because the two natural choices span the same
  column space but reach different points under a fixed optimisation
  budget.

The reported housing figures reproduce exactly through the package:
baseline validation 3.32873 and test 3.29626, quadratic validation 2.66053
and test 2.79900 at L2 = 0.01.
