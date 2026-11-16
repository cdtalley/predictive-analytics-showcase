# Predictive Analytics Showcase

Public portfolio of **predictive analytics and product data science** work, using **synthetic data only**.

Shapes mirror production problems from Drake Talley's resume (higher-ed retention, device risk, fraud scoring, product analytics, experiment rigor). No client PII. No employer data.

**Author:** Drake Talley · [draketalley.ai](https://draketalley.ai) · [github.com/cdtalley](https://github.com/cdtalley)

## Why this repo exists

Hiring managers for Staff / Senior Data Scientist roles often want to see:

1. Can you turn an ambiguous business question into a measurable outcome?
2. Can you keep evaluation honest (leakage, held-out metrics, calibration)?
3. Can you stand up product analytics and experiments, not only a model notebook?

This repo answers those with runnable notebooks and small Python packages.

## Case studies (synthetic twins)

| Case | Resume anchor | What you will see |
|------|---------------|-------------------|
| `01_retention_outreach` | Collegis / prismbase churn + outreach | Leakage audit, prediction-time features, held-out AUC, intervention bands |
| `02_device_risk` | Field-device predictive maintenance | Survival-style risk scores, precision@k for ops, failure reduction framing |
| `03_fraud_scoring` | Federal / bank fraud lift | Class imbalance, threshold tuning with ops cost, SHAP, drift (PSI) |
| `04_product_analytics` | Subscriber / Looker product analytics | KPI tree, funnel, cohort retention, dashboard-ready aggregates |
| `05_experimentation` | Evidence bar / kill bad hypotheses | Power, CUPED-style variance reduction, sequential peeking caution |

Related public twins you may already know:

- [Student-Prediction](https://github.com/cdtalley/Student-Prediction)
- [SentinelAI](https://github.com/cdtalley/SentinelAI)
- [Data-Science-Portfolio](https://github.com/cdtalley/Data-Science-Portfolio)

## Quickstart

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m src.generate_all
jupyter lab notebooks
```

## Layout

```
data/synthetic/     generated CSVs (gitignored large dumps; regenerable)
src/                generators + metrics helpers
notebooks/          case studies
experiments/        reusable A/B analysis helpers
dashboards/         optional Streamlit stub for product KPIs
```

## Honesty rules

- All tables are synthetic. Seeds are fixed for reproducibility.
- Metrics are illustrative of method, not client results.
- Client names and proprietary schemas are not reproduced.

## License

MIT for code. Synthetic data is generated locally; treat as CC0 for reuse in demos.
