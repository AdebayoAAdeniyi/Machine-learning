"""Churn modelling helpers: leakage-aware splitting and threshold metrics.

Takes a prepared numeric table with a binary outcome column. No customer
data is bundled with this repository.
"""
import numpy as np

from .core import Standardizer, stratified_split


def metrics(y_true, y_pred, positive_label=1):
    """Accuracy, precision, recall and F1 for a binary prediction.

    All four are reported together because they rank models differently under
    class imbalance, and which one matters depends on the relative cost of a
    missed churner against a wasted retention offer. Precision and recall are
    defined as zero when their denominator is empty rather than raising, so a
    degenerate model still produces a comparable row.
    """
    actual = np.asarray(y_true) == positive_label
    predicted = np.asarray(y_pred) == positive_label
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {'accuracy': float(np.mean(actual == predicted)),
            'precision': precision,
            'recall': recall,
            'f1': f1}


def prepare_splits(X, y, seed=42, smote=False):
    """Split, then scale on training only, then optionally oversample training only.

    The order matters, which is why this is one function rather than three
    inline steps. Scaling before the split lets held-out rows influence the
    transform. Oversampling before the split is worse: SMOTE synthesises new
    points by interpolating between neighbours, so near-duplicates of training
    rows leak into validation and test and the reported scores come out
    optimistic.
    """
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(X, y, seed=seed)
    scaler = Standardizer().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)
    if smote:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "SMOTE needs the optional extra: pip install -e '.[retention]'") from exc
        Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    return (Xtr, ytr), (Xva, yva), (Xte, yte)
