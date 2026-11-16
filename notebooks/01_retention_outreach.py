"""Case 01 — Retention / outreach risk (Collegis-shaped synthetic twin).

Demonstrates: leakage audit, prediction-time feature set, held-out AUC, SHAP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generators import generate_retention
from src.metrics import held_out_auc, leakage_audit, precision_at_k

DATA = ROOT / "data" / "synthetic" / "retention_students.csv"
FORBIDDEN = ["post_outcome_survey"]


def main() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        generate_retention(DATA)

    df = pd.read_csv(DATA)
    feature_cols = [
        "gpa",
        "credits_earned",
        "logins_30d",
        "crm_touches_30d",
        "days_since_login",
        "channel",
        "program",
    ]
    audit = leakage_audit(df, feature_cols + FORBIDDEN, "churned_90d", FORBIDDEN)
    print("Leakage audit (must be ok=False if forbidden left in):", audit)
    assert audit["ok"] is False
    audit_clean = leakage_audit(df, feature_cols, "churned_90d", FORBIDDEN)
    assert audit_clean["ok"] is True
    print("Prediction-time feature set cleared:", audit_clean)

    X = df[feature_cols]
    y = df["churned_90d"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        [
            ("num", "passthrough", ["gpa", "credits_earned", "logins_30d", "crm_touches_30d", "days_since_login"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["channel", "program"]),
        ]
    )
    clf = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    print(f"Held-out AUC: {held_out_auc(y_test, proba):.3f}")
    print(f"Precision@10%: {precision_at_k(y_test, proba, 0.10):.3f}")
    print("Ready for SHAP on transformed matrix in an interactive session.")


if __name__ == "__main__":
    main()
