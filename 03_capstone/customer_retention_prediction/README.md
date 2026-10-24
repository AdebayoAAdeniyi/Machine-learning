# Customer retention: choosing a model when the classes are imbalanced

**Question.** Three classical classifiers are trained on the same churn data
and each one wins on a different metric. Which is "best", and what does the
question actually depend on?

## Design

Three customer tables joined on customer ID in SQL, producing 18 behavioural
features covering purchase quantity, spending, recency and tenure. A
stratified 70/15/15 train/validation/test split, SMOTE applied to the
training rows only to address class imbalance, and three models compared:
thresholded ridge regression, logistic regression, and an SGD-trained
sigmoid classifier.

The customer tables are not in this repository and are not redistributable.
The runner therefore takes a prepared numeric table, so the pipeline can be
read, tested and applied to other data.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Ridge, thresholded | 0.5124 | 0.3657 | **0.5805** | 0.4487 |
| Logistic | 0.5287 | **0.3763** | 0.5760 | **0.4552** |
| SGD classifier | **0.5349** | 0.3541 | 0.4376 | 0.3915 |

Three models, three different winners. That is the result, not an
inconvenience to be resolved by picking one column.

**Accuracy is the least informative column here.** SGD leads it at 0.5349
while having the worst recall by 14 points and the worst F1 by 6. With
imbalanced classes, accuracy rewards a model for being right about the
majority, which is rarely the decision anyone cares about in churn.

**Precision and recall trade against each other, and the exchange rate is a
business quantity, not a statistical one.** Ridge catches more departing
customers (recall 0.5805) at the cost of more false alarms. Logistic is more
often right when it does flag someone (precision 0.3763). Which is better
depends on the cost of a retention offer against the value of a retained
customer, and no amount of model comparison settles it.

**F1 picks logistic**, but only because it weights precision and recall
equally, which is a choice rather than a default. If missing a churner costs
five times what a wasted offer costs, the F-beta that encodes that ranks
ridge first.

**One caveat worth stating.** Which class is labelled 1 determines what
"precision" and "recall" refer to. Before any of these numbers drives an
operational decision, that mapping needs confirming against the source data.

## Techniques demonstrated

SQL feature integration across multiple tables; behavioural feature
construction; stratified splitting so no class vanishes from a fold; class
imbalance handling with SMOTE applied in the correct order; comparison of
three model families under one protocol; confusion-matrix derived metrics;
and selecting a metric from the decision being supported rather than from
the model that happens to win.

## The ordering that matters

`prepare_splits` enforces: split first, then fit the scaler on training rows,
then oversample training rows only.

Both of the other orderings are tempting and both inflate the result. Scaling
before the split lets held-out rows influence the transform. Oversampling
before the split is worse: SMOTE synthesises new points by interpolating
between neighbours, so near-duplicates of training rows end up in validation
and test, and the model is scored partly on data it has effectively seen.

## Run it on your own data

```bash
python -m pip install -e '.[retention,test]'
python scripts/run_customer_retention.py --csv your_table.csv --target churned
python scripts/run_customer_retention.py --csv your_table.csv --target churned --smote
```

Expects a numeric feature table with a binary outcome column. It reports all
four metrics for each model so the trade-off above is visible rather than
collapsed into one figure.

Implementation: [`ml_portfolio/retention.py`](../../ml_portfolio/retention.py)
and [`scripts/run_customer_retention.py`](../../scripts/run_customer_retention.py).

## Data handling

No customer records are included in this repository, and none should be
committed to it. Applying this pipeline to real customer data requires the
appropriate authorisation and its own validation.
