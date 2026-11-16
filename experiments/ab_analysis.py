"""Lightweight experiment analysis: difference in means, CUPED, power sketch."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def diff_in_means(df: pd.DataFrame, arm_col: str = "arm", y_col: str = "outcome") -> dict:
    a = df.loc[df[arm_col] == "treatment", y_col].to_numpy()
    b = df.loc[df[arm_col] == "control", y_col].to_numpy()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "lift": float(a.mean() - b.mean()),
        "t": float(t),
        "p": float(p),
        "n_t": int(len(a)),
        "n_c": int(len(b)),
    }


def cuped(df: pd.DataFrame, arm_col: str = "arm", y_col: str = "outcome", x_col: str = "pre_metric") -> dict:
    """CUPED-adjusted lift using pre-period covariate."""
    y = df[y_col].to_numpy(dtype=float)
    x = df[x_col].to_numpy(dtype=float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    adj = df.copy()
    adj["_y_adj"] = y_adj
    out = diff_in_means(adj, arm_col=arm_col, y_col="_y_adj")
    out["theta"] = float(theta)
    out["method"] = "cuped"
    return out


def rough_power(effect: float, sd: float, n_per_arm: int, alpha: float = 0.05) -> float:
    """Two-sided two-sample power sketch (normal approx)."""
    se = sd * np.sqrt(2 / n_per_arm)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z = effect / se
    return float(stats.norm.cdf(z - z_alpha) + stats.norm.cdf(-z - z_alpha))
