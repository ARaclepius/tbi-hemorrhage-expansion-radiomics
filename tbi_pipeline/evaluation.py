"""Stratified CV with pooled out-of-fold (OOF) metrics."""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .pipeline import build_pipeline


def make_folds(X, y, n_splits=C.N_SPLITS, seed=C.RANDOM_STATE):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    safe = lambda a, b: a / b if b > 0 else 0.0
    return {
        "oof_auc": roc_auc_score(y_true, y_prob),
        "oof_acc": accuracy_score(y_true, y_pred),
        "oof_pr_auc": average_precision_score(y_true, y_prob),
        "oof_f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "ppv": safe(tp, tp + fp),
        "npv": safe(tn, tn + fn),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cm": cm,
    }


def cross_validate_oof(X, y, classifier, folds, seed=C.RANDOM_STATE):
    """Fit the full pipeline on every training fold; pool validation predictions."""
    oof_pred = np.zeros(len(X), dtype=int)
    oof_prob = np.zeros(len(X), dtype=float)
    n_components = []
    for k, (tr, va) in enumerate(folds, start=1):
        pipe = build_pipeline(X.iloc[tr], clone(classifier), seed=seed)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof_pred[va] = pipe.predict(X.iloc[va]).astype(int)
        oof_prob[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        n_components.append(int(pipe.named_steps["pca"].n_components_))
        print(f"    fold {k}/{len(folds)}: train={len(tr)} val={len(va)} "
              f"PCs={n_components[-1]}")
    metrics = calculate_metrics(y, oof_pred, oof_prob)
    fpr, tpr, _ = roc_curve(y, oof_prob)
    return metrics, {"fpr": fpr, "tpr": tpr}, n_components, oof_prob
