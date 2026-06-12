# Attrition Risk Predictor — UI

An interactive Streamlit app that takes an employee's raw details and predicts
whether they're likely to leave, with the SHAP drivers behind the score.

## What it does

The deployed model (`notebooks/processed/models/final_model.joblib`) is a tuned
`RandomForestClassifier` that scores a **56-feature, scaled** vector and flags a
leaver when `P(Leave) ≥ 0.49` (the recall-first threshold fixed in Phase 6).

The form collects ~30 *raw* fields. The hard part — and the bulk of the code —
is faithfully reproducing the training-time transform so those raw inputs become
the exact feature vector the model expects:

```
raw inputs ──encode──► ──engineer 13 features──► ──scale 30 cols──► RandomForest ──► P(Leave) ≥ 0.49?
```

Two engineered features (`PeerRelativeIncome`, `IsLowIncome`) depend on
train-only statistics that were never saved as artifacts. `src/preprocess.py`
recomputes them by reproducing the exact deterministic train split
(`random_state=42`) from the raw CSV — no hardcoded tables.

## Layout

| File | Role |
|---|---|
| `src/preprocess.py` | raw dict → scaled 56-column vector (mirrors `notebooks/feature-engineering.ipynb`) |
| `src/predict.py` | loads model + scaler, returns `{probability, prediction, label, drivers}` (SHAP per `notebooks/interpretation.ipynb`) |
| `app.py` | Streamlit form → verdict + driver chart |
| `tests/test_parity.py` | proves the UI transform reproduces `X_test.parquet` exactly |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501).

## Verify correctness

```bash
py tests/test_parity.py
```

This reproduces the test split, runs every raw test row through the UI
transform, and asserts the result matches the persisted `X_test.parquet` and the
model's `predict_proba` to within 1e-9 (currently: exact, 0.0 diff).

## Notes

- The model/scaler were pickled with scikit-learn 1.7.x. Loading under a newer
  minor version works (parity is exact) but logs an `InconsistentVersionWarning`;
  pin `scikit-learn==1.7.*` to silence it.
- Predictions are decision support for HR retention review, not a verdict on any
  individual employee.
