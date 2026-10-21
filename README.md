# Credit Risk: Default Prediction

Predicts whether a customer will be **90+ days past due within the next 2 years**.

## Status

🚧 Work in progress

- [ ] 01 — Exploratory data analysis
- [ ] 02 — Data preparation + baseline (Logistic Regression)
- [ ] 03 — Main model (LightGBM + MLflow)
- [ ] 04 — Explainability (SHAP) + business impact

## Data

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle), ~150k customers.
Target: `SeriousDlqin2yrs`.

Data is **not versioned in this repo**. It is stored in a Databricks Unity Catalog Volume:
`/Volumes/workspace/risco_credito/raw/cs-training.csv`