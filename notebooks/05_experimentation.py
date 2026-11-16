"""Case 05 — Experiment rigor: naive lift vs CUPED, power sketch."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.ab_analysis import cuped, diff_in_means, rough_power
from src.generators import generate_experiment

DATA = ROOT / "data" / "synthetic" / "experiment_ab.csv"


def main() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        generate_experiment(DATA)

    df = pd.read_csv(DATA)
    naive = diff_in_means(df)
    adjusted = cuped(df)
    print("Naive difference-in-means:", naive)
    print("CUPED-adjusted:", adjusted)

    sd = df["outcome"].std()
    n = (df["arm"] == "treatment").sum()
    print(f"Rough power for lift=0.35, sd={sd:.2f}, n/arm={n}: {rough_power(0.35, sd, int(n)):.2f}")
    print("Teaching point: design must answer the question; peeking without alpha spend lies.")


if __name__ == "__main__":
    main()
