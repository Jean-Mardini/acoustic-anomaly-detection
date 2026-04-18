"""Extra metrics for anomaly detection (e.g. partial AUC)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve


def partial_auc_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    max_fpr: float = 0.1,
) -> float:
    """
    Area under ROC curve for FPR in [0, max_fpr], normalized by max_fpr to [0, 1].

    Matches common DCASE-style reporting when max_fpr=0.1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("max_fpr must be in (0, 1].")
    # integrate only up to max_fpr
    idx = fpr <= max_fpr + 1e-12
    f_clip = fpr[idx]
    t_clip = tpr[idx]
    if f_clip[-1] < max_fpr:
        t_end = np.interp(max_fpr, fpr, tpr)
        f_clip = np.append(f_clip, max_fpr)
        t_clip = np.append(t_clip, t_end)
    area = auc(f_clip, t_clip)
    return float(area / max_fpr)


def roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
