# Housing prices: gradient descent, feature engineering and generalisation

**Question.** Can squared features and an L2 penalty improve housing-price
prediction under a protocol where every choice is made on validation data?

**Data.** 506 rows, 13 features, split by a seeded permutation into 300
training, 100 validation and 106 test rows. The same permutation is used for
every configuration so all comparisons are on identical folds.

**Model.** Linear in its parameters, trained by mini-batch gradient descent
on squared error over standardised targets, scored as mean absolute error in
the original price units. Weights are checkpointed at the epoch with the
lowest validation MAE.

```bash
python -m ml_portfolio.reproduction housing --data data/housing.pkl \
    --quadratic --search-decay --output results/housing_quadratic.json
```

## Results

| Configuration | Validation MAE | Test MAE |
|---|---:|---:|
| Baseline, scaling fitted on all 506 rows | 3.32873 | 3.29626 |
| Baseline, scaling fitted on training rows only | 3.33442 | 3.29328 |
| Squared features, L2 = 0.01, scaling on all rows | **2.66053** | **2.79900** |
| Squared features, L2 = 0.01, scaling on training rows only | 2.67773 | 2.78308 |

The L2 sweep over `{3, 1, 0.3, 0.1, 0.03, 0.01}`, selected on validation:

| L2 penalty | Validation MAE | Test MAE |
|---:|---:|---:|
| 3 | 4.3833 | 3.7311 |
| 1 | 3.3986 | 3.0626 |
| 0.3 | 2.9430 | 2.7355 |
| 0.1 | 2.7346 | 2.7321 |
| 0.03 | 2.6707 | 2.7778 |
| **0.01** | **2.6605** | 2.7990 |

**Squared features are worth about 15% of the error.** Test MAE falls from
3.30 to 2.78, on a model that remains linear in its parameters and therefore
still trains by the same gradient step. This is the cheapest kind of
capacity increase available: no new optimiser, no new loss, just a wider
design matrix.

**The L2 sweep shows validation and test disagreeing at the margin.**
Validation MAE improves monotonically as the penalty shrinks, so 0.01 is
selected. Test MAE is lowest at 0.1 (2.7321) and slightly worse at the
selected 0.01 (2.7990). That gap is the cost of selecting on a 100-row
validation split, and it is the honest reason not to quote the best test
number in a sweep as if it were achievable: choosing it would require having
looked at test.

## Two methodological points

### Where the scaler is fitted

Standardising with statistics computed over all 506 rows lets the held-out
rows influence the transform. Fitting on the 300 training rows alone is the
correct protocol and is the default here; `--legacy-preprocessing` runs the
other one on the same folds so the two can be compared directly.

The difference is small, 0.003 MAE, and on test it runs *against* the leaky
version. That is worth noting rather than hiding: leakage of this kind
biases a result in expectation, but on a single split it can land either
way, so "the number went down" is not evidence that the protocol was sound.

### How the squared features are parameterised

There are two natural ways to build `x²` features, and they are not
interchangeable under gradient descent:

- `--quadratic-scheme standardized` squares the already-standardised columns
  and appends them, giving the design `[1, z, z²]`.
- `--quadratic-scheme raw` squares the raw columns first and then
  standardises everything, giving `[1, std(x), std(x²)]`.

These span the **same column space**, because `z²` is a linear combination of
`x²`, `x` and `1`. An exact least-squares solve would return identical
predictions from either. Gradient descent run for a fixed 100 epochs at a
fixed learning rate does not:

| Scheme | Validation MAE | Test MAE |
|---|---:|---:|
| standardized | 2.6605 | 2.7990 |
| raw | 3.1953 | 3.1014 |

A 0.30 MAE difference, which is larger than the entire effect of the L2
penalty being tuned. The two parameterisations are conditioned differently,
so the optimiser reaches a different point within the same budget. The
lesson generalises: with a fixed iteration budget, how features are scaled
is not cosmetic, and two models that are mathematically equivalent can be
practically very different.

## Techniques demonstrated

Analytic gradients implemented and vectorised; mini-batch optimisation with
the gradient divided by the actual batch length; validation checkpointing;
feature augmentation; L2 regularisation with the intercept excluded by
default; identification and measurement of a preprocessing leak; and the
distinction between the column space a model can represent and the point in
it that a finite optimisation run actually reaches.

## Reproduce

```bash
python -m ml_portfolio.reproduction housing --data data/housing.pkl \
    --output results/housing_corrected.json
python -m ml_portfolio.reproduction housing --data data/housing.pkl \
    --legacy-preprocessing --output results/housing_legacy_scaling.json
python -m ml_portfolio.reproduction housing --data data/housing.pkl \
    --quadratic --search-decay --output results/housing_quadratic_corrected.json
python -m ml_portfolio.reproduction housing --data data/housing.pkl \
    --quadratic --legacy-preprocessing --search-decay \
    --output results/housing_quadratic_legacy.json
```

Implementation: [`ml_portfolio/reproduction.py`](../../ml_portfolio/reproduction.py).
Saved runs with full per-epoch curves: [`results/`](../../results/).
