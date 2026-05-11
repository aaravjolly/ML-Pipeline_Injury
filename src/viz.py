"""
Visualization helpers.

Plots for: ROC curve, precision-recall curve, calibration plot,
confusion matrix, and feature importance.

All matplotlib, no seaborn dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_roc(y_true, y_proba, out_path: Optional[Path] = None,
             title: str = "ROC curve"):
    """Plot the receiver-operating-characteristic curve."""
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#205493", lw=2,
            label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig


def plot_pr(y_true, y_proba, out_path: Optional[Path] = None,
            title: str = "precision-recall curve"):
    """Plot the precision-recall curve - more informative than ROC under imbalance."""
    import matplotlib.pyplot as plt
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    base_rate = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec, color="#205493", lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(base_rate, color="gray", lw=1, ls="--",
               label=f"base rate = {base_rate:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig


def plot_calibration(y_true, y_proba, out_path: Optional[Path] = None,
                     n_bins: int = 10, title: str = "calibration"):
    """Reliability diagram: predicted probability vs observed frequency."""
    import matplotlib.pyplot as plt
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins,
                                             strategy="uniform")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", label="perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#205493", lw=2, label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig


def plot_confusion_matrix(y_true, y_pred,
                           out_path: Optional[Path] = None,
                           title: str = "confusion matrix"):
    """Heatmap of confusion counts and rates."""
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["healthy", "injured"])
    ax.set_yticklabels(["healthy", "injured"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig


def plot_feature_importance(
    pipeline,
    top_n: int = 15,
    out_path: Optional[Path] = None,
    title: str = "feature importance",
):
    """
    Plot feature importances if the model exposes them.

    Tries (in order): ``feature_importances_`` (RF/GB), absolute
    ``coef_`` (logistic regression). Returns None if neither is available.
    """
    import matplotlib.pyplot as plt
    clf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocess"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        # Fallback: index strings.
        names = np.asarray([f"f{i}" for i in range(getattr(clf, "n_features_in_", 0))])

    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_, dtype=float)
        label = "importance"
    elif hasattr(clf, "coef_"):
        importances = np.abs(np.asarray(clf.coef_).ravel())
        label = "|coefficient|"
    else:
        return None

    if len(importances) != len(names):
        # Defensive: align lengths.
        m = min(len(importances), len(names))
        importances, names = importances[:m], names[:m]

    order = np.argsort(importances)[::-1][:top_n]
    importances = importances[order]
    names = np.asarray(names)[order]

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.3 * len(order))))
    y_pos = np.arange(len(order))[::-1]
    ax.barh(y_pos, importances, color="#205493")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(label)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig
