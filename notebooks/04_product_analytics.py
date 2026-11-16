"""Case 04 — Product analytics KPI tree / funnel (Looker-shaped twin)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generators import generate_product_events

DATA = ROOT / "data" / "synthetic" / "product_events.csv"


def main() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        generate_product_events(DATA)

    df = pd.read_csv(DATA)
    users = df.groupby(["cohort", "plan", "user_id"])["event"].apply(set).reset_index(name="events")
    users["signup"] = users["events"].apply(lambda s: "signup" in s)
    users["activated"] = users["events"].apply(lambda s: "activate" in s)
    users["paid"] = users["events"].apply(lambda s: "paid_convert" in s)

    funnel = (
        users.groupby(["cohort", "plan"])
        .agg(
            n_users=("user_id", "count"),
            activation_rate=("activated", "mean"),
            paid_rate=("paid", "mean"),
        )
        .reset_index()
    )
    print(funnel.to_string(index=False))
    print("\nKPI tree: signup -> activate -> paid_convert; feature_use as engagement intensity.")


if __name__ == "__main__":
    main()
