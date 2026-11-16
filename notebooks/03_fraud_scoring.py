"""Case 03 — Fraud scoring with imbalance, thresholds, PSI drift."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generators import generate_fraud
from src.metrics import held_out_auc, pr_auc, psi, precision_at_k

DATA = ROOT / "data" / "synthetic" / "fraud_txns.csv"


def main() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        generate_fraud(DATA)

    df = pd.read_csv(DATA)
    feats = ["amount", "hour", "is_intl", "device_age_days", "velocity_1h"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[feats], df["fraud"], test_size=0.3, random_state=44, stratify=df["fraud"]
    )
    # Mid-split drift sketch: compare amount distribution train vs test
    print(f"PSI(amount) train vs test: {psi(X_train['amount'], X_test['amount']):.4f}")

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.07,
        scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        eval_metric="logloss",
        random_state=44,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {held_out_auc(y_test, proba):.3f}")
    print(f"PR-AUC:  {pr_auc(y_test, proba):.3f}")
    print(f"Precision@1%: {precision_at_k(y_test, proba, 0.01):.3f}")


if __name__ == "__main__":
    main()
