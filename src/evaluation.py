"""
Model evaluation.

Pulls together the metrics that matter for an imbalanced binary
classification task like injury prediction:

- ROC-AUC (threshold-independent ranking quality)
- PR-AUC / average precision (more informative than ROC under heavy imbalance)
- F1 at the chosen decision threshold
- Confusion matrix at the threshold
- Brier score and a calibration check

Plus a grouped cross-validation harness that splits *by athlete*, not by
row, so the model has to generalize to new people - the row-level CV
default would leak per-athlete signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Metrics for a single test set."""

    n_samples: int
    n_positives: int
    base_rate: float
    threshold: float
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    brier: float
    confusion: List[List[int]]   # [[TN, FP], [FN, TP]]
    threshold_curve: Optional[Dict] = None  # for plotting

    def as_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
            "base_rate": self.base_rate,
            "threshold": self.threshold,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "brier": self.brier,
            "confusion": self.confusion,
        }


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    include_curves: bool = False,
) -> EvalReport:
    """
    Compute a full evaluation report from probabilities.

    Parameters
    ----------
    y_true : array of {0, 1}
    y_proba : array of probabilities for the positive class
    threshold : decision threshold for hard predictions and F1
    include_curves : if True, the report includes ROC and PR curves
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    if y_true.shape != y_proba.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_proba {y_proba.shape}"
        )
    if not set(np.unique(y_true).tolist()).issubset({0, 1}):
        raise ValueError("y_true must be binary 0/1")

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Some metrics require both classes to be present.
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_proba))

    curves = None
    if include_curves and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
        curves = {
            "roc_fpr": fpr.tolist(),
            "roc_tpr": tpr.tolist(),
            "pr_precision": prec_curve.tolist(),
            "pr_recall": rec_curve.tolist(),
        }

    return EvalReport(
        n_samples=int(len(y_true)),
        n_positives=int(y_true.sum()),
        base_rate=float(y_true.mean()) if len(y_true) else 0.0,
        threshold=float(threshold),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        f1=f1,
        precision=prec,
        recall=rec,
        brier=brier,
        confusion=cm.tolist(),
        threshold_curve=curves,
    )


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------


def best_f1_threshold(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Tuple[float, float]:
    """
    Return the (threshold, F1) that maximizes F1 over the validation set.

    Useful when the default 0.5 cutoff isn't appropriate (which is almost
    always the case for imbalanced problems).
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one fewer threshold than points; align.
    f1s = np.zeros_like(prec)
    denom = prec + rec
    f1s[denom > 0] = 2 * prec[denom > 0] * rec[denom > 0] / denom[denom > 0]
    if len(thresholds) == 0:
        return 0.5, float(f1s.max() if len(f1s) else 0.0)
    # Best F1 over the matching threshold positions.
    # f1s has len == len(thresholds) + 1; align by chopping the trailing point.
    f1_aligned = f1s[:-1]
    if len(f1_aligned) == 0:
        return 0.5, float(f1s.max())
    idx = int(np.argmax(f1_aligned))
    return float(thresholds[idx]), float(f1_aligned[idx])


# ---------------------------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------------------------


def group_cross_validate(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    threshold: float = 0.5,
) -> Dict[str, List[float]]:
    """
    Run grouped CV: each fold's validation set has *unseen athletes*.

    Returns a dict of metric_name -> per-fold values.

    The pipeline is cloned per fold (sklearn does this inside
    ``cross_val_predict``-equivalent code, but here we do it explicitly
    so we can run the full evaluate() report on each fold, including
    pr_auc which sklearn's ``cross_val_score`` doesn't surface neatly).
    """
    from sklearn.base import clone

    cv = GroupKFold(n_splits=n_splits)
    out: Dict[str, List[float]] = {
        "roc_auc": [], "pr_auc": [], "f1": [], "precision": [],
        "recall": [], "brier": [],
    }

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_tr = X.iloc[tr_idx]; y_tr = y[tr_idx]
        X_va = X.iloc[va_idx]; y_va = y[va_idx]

        model = clone(pipeline)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        rep = evaluate(y_va, proba, threshold=threshold)
        for k in out:
            out[k].append(getattr(rep, k))

    return out


def summarize_cv(cv_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Summarize CV results as mean/std per metric."""
    out: Dict[str, Dict[str, float]] = {}
    for metric, vals in cv_results.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            out[metric] = {"mean": float("nan"), "std": float("nan")}
            continue
        out[metric] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }
    return out
