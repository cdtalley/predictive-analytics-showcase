"""Synthetic data generators shaped like resume projects (no real client data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_retention(path: Path, n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Higher-ed style retention / outreach risk (Collegis twin shape)."""
    rng = np.random.default_rng(seed)
    gpa = rng.normal(3.0, 0.55, n).clip(0.5, 4.0)
    credits = rng.integers(0, 120, n)
    logins_30d = rng.poisson(8, n)
    crm_touches = rng.poisson(2, n)
    days_since_login = rng.integers(0, 90, n)
    # Leakage-like column that must be stripped at prediction time
    post_outcome_survey = rng.normal(0, 1, n)

    logit = (
        -0.8
        + (-0.55) * (gpa - 3.0)
        + 0.012 * (90 - credits).clip(0, 90)
        + 0.04 * days_since_login
        + (-0.08) * logins_30d
        + (-0.05) * crm_touches
        + 0.35 * post_outcome_survey  # leakage signal
    )
    p = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, p)

    df = pd.DataFrame(
        {
            "student_id": [f"S{i:05d}" for i in range(n)],
            "gpa": gpa.round(2),
            "credits_earned": credits,
            "logins_30d": logins_30d,
            "crm_touches_30d": crm_touches,
            "days_since_login": days_since_login,
            "channel": rng.choice(["organic", "paid", "referral", "partner"], n),
            "program": rng.choice(["undergrad", "grad", "certificate"], n, p=[0.6, 0.25, 0.15]),
            "post_outcome_survey": post_outcome_survey.round(3),  # FORBIDDEN at score time
            "churned_90d": churned,
        }
    )
    df.to_csv(path, index=False)
    return df


def generate_device_risk(path: Path, n: int = 5000, seed: int = 43) -> pd.DataFrame:
    """Field-device / IoT-style failure risk (predictive maintenance twin)."""
    rng = np.random.default_rng(seed)
    age_days = rng.integers(30, 1800, n)
    battery_mv = rng.normal(3200, 180, n)
    open_events_7d = rng.poisson(12, n)
    error_codes_7d = rng.poisson(1.2, n)
    temp_c = rng.normal(22, 8, n)

    logit = (
        -3.0
        + 0.0015 * age_days
        + (-0.002) * (battery_mv - 3000)
        + 0.35 * error_codes_7d
        + 0.02 * np.abs(temp_c - 22)
        + 0.01 * open_events_7d
    )
    p = 1 / (1 + np.exp(-logit))
    failed_30d = rng.binomial(1, p)

    df = pd.DataFrame(
        {
            "device_id": [f"D{i:05d}" for i in range(n)],
            "age_days": age_days,
            "battery_mv": battery_mv.round(0),
            "open_events_7d": open_events_7d,
            "error_codes_7d": error_codes_7d,
            "temp_c": temp_c.round(1),
            "region": rng.choice(["NE", "SE", "MW", "W"], n),
            "failed_30d": failed_30d,
        }
    )
    df.to_csv(path, index=False)
    return df


def generate_fraud(path: Path, n: int = 20000, seed: int = 44) -> pd.DataFrame:
    """Payment / transaction fraud scoring twin (imbalanced)."""
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.5, 1.1, n)
    hour = rng.integers(0, 24, n)
    is_intl = rng.binomial(1, 0.08, n)
    device_age_days = rng.integers(1, 2000, n)
    velocity_1h = rng.poisson(1.5, n)

    logit = (
        -6.5
        + 0.15 * np.log1p(amount)
        + 0.08 * ((hour < 5) | (hour > 22)).astype(float)
        + 1.8 * is_intl
        + 0.25 * velocity_1h
        + (-0.0004) * device_age_days
    )
    p = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, p)

    df = pd.DataFrame(
        {
            "txn_id": [f"T{i:06d}" for i in range(n)],
            "amount": amount.round(2),
            "hour": hour,
            "is_intl": is_intl,
            "device_age_days": device_age_days,
            "velocity_1h": velocity_1h,
            "mcc": rng.choice(["5411", "5812", "5999", "4111", "6011"], n),
            "fraud": fraud,
        }
    )
    df.to_csv(path, index=False)
    return df


def generate_product_events(path: Path, n_users: int = 3000, seed: int = 45) -> pd.DataFrame:
    """Product analytics events for funnel / cohort KPIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        cohort = rng.choice(["2025-Q3", "2025-Q4", "2026-Q1"])
        plan = rng.choice(["free", "pro", "enterprise"], p=[0.55, 0.35, 0.10])
        signup_day = int(rng.integers(0, 90))
        activated = rng.random() < (0.45 if plan == "free" else 0.7)
        if not activated:
            rows.append(
                {
                    "user_id": f"U{u:05d}",
                    "cohort": cohort,
                    "plan": plan,
                    "event": "signup",
                    "day": signup_day,
                    "value": 1,
                }
            )
            continue
        rows.append(
            {
                "user_id": f"U{u:05d}",
                "cohort": cohort,
                "plan": plan,
                "event": "signup",
                "day": signup_day,
                "value": 1,
            }
        )
        rows.append(
            {
                "user_id": f"U{u:05d}",
                "cohort": cohort,
                "plan": plan,
                "event": "activate",
                "day": signup_day + int(rng.integers(0, 7)),
                "value": 1,
            }
        )
        if rng.random() < 0.35:
            rows.append(
                {
                    "user_id": f"U{u:05d}",
                    "cohort": cohort,
                    "plan": plan,
                    "event": "paid_convert",
                    "day": signup_day + int(rng.integers(7, 45)),
                    "value": 1,
                }
            )
        for _ in range(int(rng.poisson(4))):
            rows.append(
                {
                    "user_id": f"U{u:05d}",
                    "cohort": cohort,
                    "plan": plan,
                    "event": "feature_use",
                    "day": signup_day + int(rng.integers(0, 60)),
                    "value": 1,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def generate_experiment(path: Path, n: int = 12000, seed: int = 46) -> pd.DataFrame:
    """A/B experiment with a small true lift (for power / CUPED demos)."""
    rng = np.random.default_rng(seed)
    arm = rng.choice(["control", "treatment"], n)
    pre_metric = rng.normal(10, 3, n)
    noise = rng.normal(0, 2.5, n)
    true_lift = 0.35
    outcome = pre_metric + noise + np.where(arm == "treatment", true_lift, 0.0)
    converted = (outcome + rng.normal(0, 1, n)) > 11.0

    df = pd.DataFrame(
        {
            "user_id": [f"E{i:05d}" for i in range(n)],
            "arm": arm,
            "pre_metric": pre_metric.round(3),
            "outcome": outcome.round(3),
            "converted": converted.astype(int),
        }
    )
    df.to_csv(path, index=False)
    return df
