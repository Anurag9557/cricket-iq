"""Metrics for win-probability evaluation.

We score PROBABILITY quality, not accuracy: Brier (mean squared error of the
probability), log-loss (punishes confident mistakes), AUC (ranking/discrimination),
and ECE (calibration — does "70%" actually happen 70% of the time?).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Mean gap between predicted confidence and observed frequency, across prob bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    ids = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            ece += mask.sum() / len(y_prob) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def evaluate(y_true, y_prob) -> dict:
    """Headline metrics. y_true in {0,1}, y_prob in [0,1]."""
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": int(len(y_true)),
    }