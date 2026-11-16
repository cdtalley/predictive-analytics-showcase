"""Generate all synthetic datasets (fixed seeds, resume-shaped schemas)."""

from __future__ import annotations

from pathlib import Path

from .generators import (
    generate_retention,
    generate_device_risk,
    generate_fraud,
    generate_product_events,
    generate_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generate_retention(OUT / "retention_students.csv", n=8000, seed=42)
    generate_device_risk(OUT / "device_risk.csv", n=5000, seed=43)
    generate_fraud(OUT / "fraud_txns.csv", n=20000, seed=44)
    generate_product_events(OUT / "product_events.csv", n_users=3000, seed=45)
    generate_experiment(OUT / "experiment_ab.csv", n=12000, seed=46)
    print(f"Wrote synthetic tables under {OUT}")


if __name__ == "__main__":
    main()
