"""Case 02 — Device failure risk (predictive maintenance twin)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generators import generate_device_risk
from src.metrics import held_out_auc, precision_at_k

DATA = ROOT / "data" / "synthetic" / "device_risk.csv"


def main() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        generate_device_risk(DATA)

    df = pd.read_csv(DATA)
    feats = ["age_days", "battery_mv", "open_events_7d", "error_codes_7d", "temp_c"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[feats], df["failed_30d"], test_size=0.25, random_state=43, stratify=df["failed_30d"]
    )
    model = GradientBoostingClassifier(random_state=43)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"Held-out AUC: {held_out_auc(y_test, proba):.3f}")
    print(f"Precision@5% (ops queue): {precision_at_k(y_test, proba, 0.05):.3f}")
    print("Top importances:", sorted(zip(feats, model.feature_importances_), key=lambda t: -t[1])[:3])


if __name__ == "__main__":
    main()
