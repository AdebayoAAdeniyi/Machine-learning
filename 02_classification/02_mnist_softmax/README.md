# MNIST: how far does a purely linear model get?

**Question.** With no hidden layer and no convolution, just 784 pixels mapped
to 10 class scores, how accurately can handwritten digits be classified, and
what do the learning rate, batch size and epoch choice contribute?

**Data.** The four uncompressed MNIST IDX files: 60,000 labelled training
images and 10,000 test images, 28 × 28 pixels each. The training set is split
50,000 for fitting and 10,000 for validation; the separate test set is scored
once, with the weights from the best validation epoch.

## The model

A linear map `s = Wx + b` from 784 pixel intensities to 10 class scores,
turned into a distribution by softmax and trained on multinomial
cross-entropy by mini-batch SGD. The gradient is `X'(P − Y)/n`, built in
place by subtracting 1 from the true-class column of the predicted
probabilities rather than materialising the one-hot target matrix.

Three implementation points that matter at this scale:

- **Softmax with the row maximum subtracted.** Raw scores reach magnitudes
  where `exp` overflows; subtracting the max is algebraically free and makes
  every exponent at most zero.
- **Cross-entropy from logits via log-sum-exp**, so no probability is ever
  logged after underflowing to zero.
- **Epoch loss recomputed over the whole training set**, in chunks, rather
  than averaging the per-batch losses seen during the epoch. Those were each
  computed under different weights, so their average describes no model that
  ever existed and draws a misleadingly smooth curve.

The bias is kept as a separate vector so weight decay never shrinks it.

## Results

| | Validation | Test | Best epoch |
|---|---:|---:|---|
| lr 0.1, batch 50, no L2, 60 epochs, seed 314 | 92.48% | **92.53%** | 58 of 60 |

Full per-epoch cross-entropy and validation accuracy in
[`results/mnist_60epochs.json`](../../results/mnist_60epochs.json).

**92.5% is the honest ceiling for a model with no hidden representation**,
and that is what makes it useful. It is the number any more complex
architecture should be asked to justify itself against: a convolutional
network reaching 99% is buying 6.5 points for a large increase in
complexity, and knowing the linear baseline is what makes that trade visible.

**The best epoch is 58 of 60**, so validation accuracy was still improving
when training stopped. The checkpoint mechanism means the reported model is
the best one seen rather than the last one, but the curve indicates this run
was budget-limited rather than converged.

**Accuracy and the training objective are different quantities.** The model
minimises cross-entropy because accuracy is piecewise constant and provides
no gradient; selection and reporting then use accuracy because that is what
the task is judged on. Those two roles are kept separate throughout, and the
JSON records both curves.

## Techniques demonstrated

Multinomial logistic regression derived and implemented; numerically stable
softmax and log-sum-exp; vectorised mini-batch SGD over 50,000 examples;
binary IDX file parsing with header and shape validation; train/validation/
test discipline with checkpoint selection; and awareness of the difference
between the optimised loss and the reported metric.

## Reproduce

```bash
python -m ml_portfolio.reproduction mnist --data-dir data \
    --lr 0.1 --batch-size 50 --epochs 60 --seed 314 \
    --output results/mnist_60epochs.json
```

A few minutes on CPU. The runner validates the IDX magic numbers and
declared array shapes before training, so a truncated or wrong-endian file
fails immediately rather than producing a quietly wrong model.

Implementation: [`ml_portfolio/reproduction.py`](../../ml_portfolio/reproduction.py).
