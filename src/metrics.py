"""Evaluation helpers used across case studies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def held_out_auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def brier(y_true, y_score) -> float:
    return float(brier_score_loss(y_true, y_score))


def precision_at_k(y_true, y_score, k: float = 0.1) -> float:
    """Fraction of positives among top-k scored rows (k as fraction of n)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    take = max(1, int(round(n * k)))
    idx = np.argsort(-y_score)[:take]
    return float(y_true[idx].mean())


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index on a numeric feature."""
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected.dropna(), quantiles))
    if len(cuts) < 3:
        return 0.0
    e_counts = np.histogram(expected, bins=cuts)[0].astype(float)
    a_counts = np.histogram(actual, bins=cuts)[0].astype(float)
    e_pct = (e_counts + 1e-6) / (e_counts.sum() + 1e-6 * len(e_counts))
    a_pct = (a_counts + 1e-6) / (a_counts.sum() + 1e-6 * len(a_counts))
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def leakage_audit(df: pd.DataFrame, feature_cols: list[str], label_col: str, forbidden: list[str]) -> dict:
    """Flag columns that must not appear at prediction time."""
    present = [c for c in forbidden if c in feature_cols]
    return {
        "ok": len(present) == 0,
        "forbidden_present": present,
        "n_features": len(feature_cols),
        "label": label_col,
    }
