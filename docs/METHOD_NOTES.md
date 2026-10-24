# Method notes

The decisions behind the implementations, and how each one is checked.
Written for a reader deciding whether to trust the numbers, or building
something similar and wanting the reasoning rather than only the code.

---

## 1. Numerical stability is not optional at these scales

Three places where the textbook expression fails in floating point, and the
equivalent that does not:

| Quantity | Naive form | Used here |
|---|---|---|
| Logistic | `1 / (1 + exp(-z))` | `exp(-logaddexp(0, -z))` |
| Softmax | `exp(z) / Σexp(z)` | subtract the row max first |
| Cross-entropy | `-log(p)` after softmax | log-sum-exp directly on the logits |

The naive sigmoid overflows for very negative `z`. The naive softmax
overflows on MNIST, where raw scores reach magnitudes that `exp` cannot
represent. Taking the log of a probability that has underflowed to zero
gives `-inf` and poisons the whole epoch average. All three replacements are
algebraically identical and cost nothing.

`test_softmax_stability` checks that `softmax([[1000, 1001]])` still sums to
one.

## 2. Gradients are divided by the actual batch length

```python
w -= lr * (2 * xb.T @ (xb @ w - yb) / len(xb) + 2 * decay * penalty)
```

Not by the nominal batch size. With 300 training rows and a batch of 10 the
two agree exactly. With 302 rows the final batch has 2 samples, and dividing
it by 10 gives that batch a step five times too large, every epoch, at
whatever point in the data it happens to fall.

## 3. The epoch loss is recomputed, not accumulated

The training loss for an epoch is evaluated over the whole training set
*after* the epoch finishes. The alternative, summing the per-batch losses
observed during the epoch, is cheaper and wrong: each of those was computed
under different weights, so their average describes no model that ever
existed. It also draws a smoother curve than reality, which is the worst
property a diagnostic can have.

On MNIST this is computed in 4,096-row chunks to bound memory.

## 4. Transforms are fitted on training rows only

`Standardizer` separates `fit` from `transform` so the mean and scale come
from the training split and are applied unchanged elsewhere. The housing
experiment exposes `--legacy-preprocessing`, which standardises using all 506
rows, purely so the two protocols can be compared on identical folds.

That comparison is instructive: the leaky version scores *worse* on test
(3.29626 against 3.29328). Leakage biases a result in expectation, but on a
single split it can fall either way, so "the number improved" is not
evidence that a protocol was sound. The measurement is the argument, not the
direction it happened to move.

## 5. Ordering around SMOTE

`prepare_splits` enforces split → fit scaler on train → oversample train.

Oversampling before splitting is the more damaging of the two common
mistakes. SMOTE creates new points by interpolating between near neighbours,
so if it runs first, synthetic near-duplicates of training rows land in
validation and test. The model is then scored partly on data it has
effectively already seen, and the inflation can be large enough to reverse a
model ranking.

## 6. Feature parameterisation changes the answer under a fixed budget

The housing study builds squared features two ways:

- `standardized`: square the standardised columns, design `[1, z, z²]`
- `raw`: square the raw columns then standardise, design `[1, std(x), std(x²)]`

These span the **same column space**, because `z² = ((x − μ)/σ)²` is a linear
combination of `x²`, `x` and `1`. An exact least-squares solve returns
identical predictions from either.

100 epochs of gradient descent at a fixed learning rate does not:

| Scheme | Validation MAE | Test MAE |
|---|---:|---:|
| standardized | 2.6605 | 2.7990 |
| raw | 3.1953 | 3.1014 |

0.30 MAE, larger than the effect of the regularisation being tuned. The two
parameterisations are conditioned differently, so the optimiser lands
somewhere different within the same budget. Both are available through
`--quadratic-scheme` rather than one being silently assumed.

The general point: "the model can represent it" and "this optimisation run
will find it" are different claims, and a fixed iteration budget turns a
cosmetic choice into a substantive one.

## 7. Selection happens on validation, reporting on test

Every hyperparameter and every checkpoint is chosen on validation. The test
split is scored once, with the selected configuration.

The housing L2 sweep shows why this discipline costs something and is still
right. Validation selects penalty 0.01 (validation 2.6605, test 2.7990), but
the best test MAE in the sweep is at penalty 0.1 (2.7321). Quoting 2.7321
would require having looked at test to choose it, so the reported figure is
2.7990. The gap is the price of a 100-row validation split, and it is real.

## 8. Ridge at zero penalty is ordinary least squares

If a penalty search selects 0, the correct statement is that regularisation
did not help, not that a small amount of ridge was optimal. The two readings
sound similar and imply different things about the problem.

`ridge_least_squares` also leaves the intercept unpenalised, implemented by
zeroing the first diagonal entry of the penalty block. Shrinking the
intercept would make the fit depend on where the target happens to be
centred. `test_ridge_does_not_regularize_intercept` checks that a constant
target of 7 still returns an intercept of 7 at penalty 10.

## 9. Closed forms go through `lstsq`, not the normal equations

`np.linalg.lstsq` uses an SVD. Forming and inverting `X'X` squares the
condition number and fails outright when the design is rank deficient, where
`lstsq` returns the minimum-norm solution. Ridge is implemented by stacking
`sqrt(lambda) I` beneath the design rather than adding `lambda I` to `X'X`,
for the same reason.

## 10. Stratified splitting, and what it refuses to do

`stratified_split` splits within each class so every fold contains every
class. A plain random split can leave a rare class out of validation or test
entirely, which makes that class's metric meaningless without any error
being raised. A class with fewer than three examples cannot be split three
ways, and the function says so rather than silently producing an empty fold.

## 11. Input validation that fails early

The MNIST reader checks each file's magic number and that the declared count
matches the bytes present. A truncated or wrong-endian file then fails at
load with a message naming the file, instead of training for several minutes
on garbage. The housing reader rejects non-finite values and misaligned
shapes for the same reason.

`housing.pkl` is a Python pickle, which executes code on load. That is noted
wherever it is read, because the right precaution is knowing the file's
provenance, not a code change.

## 12. Determinism

Every experiment takes an explicit seed and records it, along with the full
configuration and per-epoch curves, in its JSON output. The housing split is
a seeded permutation held fixed across configurations so comparisons are on
identical folds rather than on folds that differ along with the setting
being tested.

## 13. Scope

The directional study is a controlled simulation, so its conclusion follows
from the covariance algebra and the samples confirm it. The housing and MNIST
results are single splits of single datasets with one seed: no confidence
intervals are reported, and none should be inferred from differences of a
few hundredths. The retention figures come from data not distributed here,
so the code is provided to be read and reused rather than as an executable
reproduction of those numbers.
