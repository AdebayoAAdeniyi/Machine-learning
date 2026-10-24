# Machine Learning: models built from the mathematics up

Five studies in supervised learning, each implemented in NumPy from the
underlying equations rather than from a library call: ordinary and ridge
least squares, logistic regression, multinomial softmax regression, and
mini-batch gradient descent with validation checkpointing. Every experiment
runs from one command and writes its configuration and per-epoch curves to
JSON.

This is part of my assignment works for **CMPUT 566, Machine Learning Essentials** at the
University of Alberta, together with a banking customer-retention project,
and then rewrote them as an installable, tested package so the experiments
reproduce end to end.

| Study | Question | Headline result |
|---|---|---|
| [Directional least squares](01_regression/01_directional_least_squares/README.md) | Does fitting `x → y` recover the same line as `y → x`? | No: slope 0.5401 against 0.5613 at 45°, and the gap widens as the cloud rounds out |
| [Housing regression](01_regression/02_housing_gradient_descent/README.md) | Do nonlinear features and L2 help under a validation-controlled protocol? | Test MAE **3.2963 → 2.7990** with squared features and a tuned penalty |
| [Linear vs logistic](02_classification/01_linear_vs_logistic/README.md) | When does thresholded least squares fail as a classifier? | Shift one class and it drops to **75.00%** while logistic holds **93.75%** |
| [MNIST softmax](02_classification/02_mnist_softmax/README.md) | How far does a purely linear model get on 10-class image recognition? | **92.53%** test accuracy, 784 → 10 weights, no hidden layer |
| [Customer retention](03_capstone/customer_retention_prediction/README.md) | Which model is "best" when classes are imbalanced? | Depends on the metric: logistic wins F1 (0.4552), ridge wins recall (0.5805), SGD wins accuracy |

## What these studies demonstrate

**Mathematics implemented directly.** Closed-form least squares through an
SVD rather than the normal equations; ridge as an augmented design matrix
with the intercept left unpenalised; the logistic and multinomial gradients
derived and written out (`X'(σ(Xw) − y)/n` and `X'(P − Y)/n`); cross-entropy
evaluated from logits so nothing underflows.

**Numerical care.** `sigmoid` via `exp(−logaddexp(0, −z))` so large negative
inputs cannot overflow; softmax with the row maximum subtracted; log-sum-exp
for the loss. These are one-line choices that decide whether a model trains
or produces `nan` at scale.

**Experimental protocol.** Train/validation/test roles kept strictly
separate; hyperparameters and checkpoints selected on validation only;
scalers fitted on training rows alone; stratified splitting so no class
disappears from a fold; SMOTE applied after the split and to training rows
only; deterministic seeds recorded in every output file.

**Knowing what a number means.** Several studies report more than one metric
on purpose, because the obvious one is not always the one that answers the
question. The retention study is the clearest case: three models, three
different winners depending on which metric is read.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q                      # 14 tests

# Housing: baseline, then squared features with an L2 sweep
python -m ml_portfolio.reproduction housing --data data/housing.pkl \
    --output results/housing_corrected.json
python -m ml_portfolio.reproduction housing --data data/housing.pkl \
    --quadratic --search-decay --output results/housing_quadratic.json

# MNIST: 60 epochs over 50,000 images, a few minutes on CPU
python -m ml_portfolio.reproduction mnist --data-dir data --epochs 60 \
    --output results/mnist_60epochs.json

# Synthetic studies, no dataset needed
python 01_regression/01_directional_least_squares/run.py
python 02_classification/01_linear_vs_logistic/run.py
```

`data/` holds the housing pickle and the four uncompressed MNIST IDX files.
The banking tables are not in the repository; the retention runner takes a
prepared numeric CSV so the code can be read and exercised without them.

Two practical notes. `housing.pkl` is a Python pickle, which executes code
on load: fine for a file you put there yourself, worth knowing before
pointing it at anything else. And `.gitignore` excludes data formats by
default, with an explicit exception for the two shipped datasets, so a
customer table dropped into the working tree cannot be committed by
accident. `train-images-idx3-ubyte` is 45 MB, under GitHub's per-file limit
but large enough that you may prefer Git LFS or a download script if you
fork this.

## Results

All figures below come from running the code in this repository. The JSON in
[`results/`](results/) carries the full settings and the per-epoch curves.

### Housing regression

| Configuration | Validation MAE | Test MAE |
|---|---:|---:|
| Baseline, scaling fitted on all 506 rows | 3.32873 | 3.29626 |
| Baseline, scaling fitted on training rows only | 3.33442 | 3.29328 |
| Squared features, L2 = 0.01, scaling on all rows | **2.66053** | **2.79900** |
| Squared features, L2 = 0.01, scaling on training rows only | 2.67773 | 2.78308 |

Adding `x²` alongside `x` and tuning the penalty takes test MAE from 3.30 to
2.78, about a 15% reduction, on a 13-feature problem where the model stays
linear in its parameters.

The two scaling rows exist because the first protocol standardises using all
506 rows, before the split. That lets held-out rows influence the transform.
The effect here is small, 0.003 MAE, and it goes in the *unhelpful* direction
for the leaky version on test, which is a useful reminder that leakage does
not always flatter the result that contains it. The corrected protocol is the
default; `--legacy-preprocessing` runs the other one so the two can be
compared on identical folds.

### MNIST softmax

| | Validation | Test | Best epoch |
|---|---:|---:|---|
| 60-epoch run, lr 0.1, batch 50, no L2, seed 314 | 92.48% | **92.53%** | 58 of 60 |

A linear map from 784 pixels to 10 class scores, trained by SGD, gets just
over 92% of handwritten digits right. That number is the reference point that
makes the case for nonlinear models: it is the ceiling for a model with no
hidden representation, and it is high enough that any more complex method
should be asked what it buys over it.

### Customer retention

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Ridge, thresholded | 0.5124 | 0.3657 | **0.5805** | 0.4487 |
| Logistic | 0.5287 | **0.3763** | 0.5760 | **0.4552** |
| SGD classifier | **0.5349** | 0.3541 | 0.4376 | 0.3915 |

Three models and three different winners. Which one is "best" is not a
property of the table, it is a property of the decision: recall matters if
missing a departing customer is expensive, precision matters if each
retention offer costs money, and accuracy is the least informative of the
three when the classes are imbalanced.

## Using this for your own work

**Each piece is independently usable.** `ml_portfolio/core.py` has the
closed-form fits, link functions, stratified splitting and the scaler.
`ml_portfolio/optimization.py` has the three gradient-trained models, all
sharing one checkpointing protocol. Neither imports from the experiments, so
you can lift either file.

**The retention runner takes your own data:**

```bash
python -m pip install -e '.[retention,test]'
python scripts/run_customer_retention.py --csv your_table.csv --target churned
python scripts/run_customer_retention.py --csv your_table.csv --target churned --smote
```

It expects a numeric feature table with a binary outcome column, splits it
70/15/15 with stratification, fits the scaler on training rows, applies SMOTE
to training rows only if asked, and reports all four metrics for each model.

**Points worth copying:**

- Divide the gradient by the **actual** batch length, not the nominal batch
  size. With 300 samples and a batch of 10 it makes no difference; with 302
  it gives the last batch a 30-times-too-large step.
- Recompute the epoch training loss over the whole training set rather than
  averaging the per-batch losses collected during the epoch. Those were each
  computed under different weights, so their average describes no model that
  ever existed, and the curve it draws is misleadingly smooth.
- Write `sigmoid` as `exp(-logaddexp(0, -z))` and subtract the row max before
  a softmax. Both are free and both are the difference between training and
  `nan`.
- Fit scalers on training rows only, and apply SMOTE after splitting. SMOTE
  interpolates between neighbours, so resampling first puts near-duplicates
  of training rows into your test set.
- If a ridge search selects penalty 0, say that ridge did not help. At zero
  penalty ridge *is* ordinary least squares, so reporting it as the best
  ridge setting overstates what was found.

## Repository map

```text
ml_portfolio/        core numerics, optimizers, experiment drivers
  core.py            OLS, ridge, sigmoid, softmax, splitting, scaling
  optimization.py    mini-batch linear SGD, logistic, softmax SGD
  reproduction.py    housing and MNIST experiments with JSON output
  experiments.py     directional least squares
  retention.py       leakage-aware splitting and classification metrics
01_regression/       directional least squares; housing regression
02_classification/   linear vs logistic; MNIST softmax
03_capstone/         customer retention
scripts/             command-line runners
tests/               14 tests: mathematical invariants and protocol checks
results/             executed output with full settings and curves
data/                housing pickle and MNIST IDX files
legacy/              my original coursework scripts, kept for provenance
docs/METHOD_NOTES.md design decisions and how each is verified
```

`legacy/` holds the scripts as I first wrote them, with the machine-specific
paths they ran with. They are kept so the provenance is visible; the package
is the version to read and run.

## Requirements

Python 3.9+ and NumPy. `[test]` adds pytest, `[retention]` adds
imbalanced-learn for SMOTE.

## Data and responsible use

The housing and MNIST data are included. The banking customer records are
not, and are not redistributable; applying this code to real customer data
needs the appropriate authorisation and its own validation.
