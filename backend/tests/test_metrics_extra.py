import numpy as np

from backend.ml.metrics_extra import partial_auc_roc, roc_auc_safe


def test_roc_auc_safe_binary() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    a = roc_auc_safe(y, s)
    assert a == 1.0


def test_partial_auc_finite() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    p = partial_auc_roc(y, s, max_fpr=0.1)
    assert np.isfinite(p)
