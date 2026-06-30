# Employee Attrition Prediction (HR Analytics)

A binary-classification project that predicts whether an employee is likely to
**leave the organization**, so HR can intervene with retention strategies for
at-risk, high-value staff — paired with an interactive UI that scores a single
employee and explains *why*.

The pipeline is optimized for **recall on leavers** (missing someone about to
quit is the costly error) and surfaces the **key drivers** of attrition to guide
retention policy.

---

## Results

The deployed model is a tuned **RandomForestClassifier** that flags a leaver
when `P(Leave) ≥ 0.49` (the recall-first threshold fixed in Phase 6).

| Metric (held-out test set, 294 rows, ~16% real attrition) | Value |
|---|---|
| **Recall** (leavers caught) | **0.830** — catches 39 of 47 actual leavers |
| Precision | 0.315 |
| F1 | 0.456 |
| ROC-AUC | 0.766 |
| PR-AUC | 0.397 |

**Top attrition drivers** (SHAP, Phase 8): **JobHoppingIndex**, **OverTime**,
**MonthlyIncome** — with OverTime the single strongest signal on permutation-recall.

The model is tuned to catch most leavers by flagging ~26% of the workforce for
retention review. It is **decision support for HR**, not a verdict on any
individual employee.

---

## Dataset

[IBM HR Analytics Employee Attrition dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
— `WA_Fn-UseC_-HR-Employee-Attrition.csv` at the repo root, **1,470 rows × 35
columns**. Features span job details, compensation, satisfaction, work patterns,
and demographics. The target `Attrition` (Yes/No) is imbalanced at **~16%
attrition (5.2:1)**, so accuracy is misleading and recall is the priority metric.

---

## Pipeline

The project is implemented as **per-stage notebooks** in [notebooks/](notebooks/).
Stages communicate only via files persisted to `processed/` — there is no
in-memory hand-off between notebooks.

| Phase | Notebook | What it does |
|---|---|---|
| 1. EDA | [eda-presentation.ipynb](notebooks/eda-presentation.ipynb) | Attrition patterns, driver discovery, class-imbalance check |
| 2–3. Preprocessing + Feature Engineering | [feature-engineering.ipynb](notebooks/feature-engineering.ipynb) | Encode, scale, 13 engineered features → 56-col matrices (train-fit only) |
| 4. Handle Imbalance | [handle-imbalance.ipynb](notebooks/handle-imbalance.ipynb) | SMOTE on the training set (50/50) |
| 5. Modeling | [modeling.ipynb](notebooks/modeling.ipynb) | 5-fold CV bake-off: LogReg → RandomForest → XGBoost → LightGBM |
| 6. Tuning | [tuning.ipynb](notebooks/tuning.ipynb) | GridSearchCV (F2) + recall-first threshold sweep → `final_model.joblib` |
| 7. Evaluation | [evaluation.ipynb](notebooks/evaluation.ipynb) | Held-out test bake-off; confirms RandomForest; confusion/ROC/PR figures |
| 8. Interpretation | [interpretation.ipynb](notebooks/interpretation.ipynb) | SHAP drivers cross-checked 3 ways; driver→retention-lever synthesis |
| 9. Prediction UI | [app.py](app.py) | Streamlit single-employee scorer (see below) |

See [CLAUDE.md](CLAUDE.md) for the full pipeline architecture and the exact
input/output artifacts of each stage, and [insights.md](insights.md) for
phase-by-phase findings and numbers.

---

## Repository layout

```
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Raw dataset (1,470 × 35)
├── app.py                                  # Streamlit prediction UI (Phase 9)
├── requirements.txt                        # Runtime dependencies
├── src/                                    # UI support package
│   ├── preprocess.py                       # raw inputs → scaled 56-feature vector
│   └── predict.py                          # model + scaler → {probability, label, SHAP drivers}
├── tests/
│   └── test_parity.py                      # proves the UI transform reproduces X_test exactly
├── notebooks/                              # Per-stage pipeline (Phases 1–8)
│   ├── *.ipynb                             # one notebook per stage
│   ├── processed/                          # persisted splits, scaler, SMOTE matrices, models/
│   └── outputs/                            # evaluation & interpretation figures
├── CLAUDE.md                               # Architecture, progress, conventions
├── insights.md                             # Detailed phase-by-phase insights
└── README_UI.md                            # Prediction UI documentation
```

---

## Getting started

```bash
pip install -r requirements.txt
```

**Reproduce the pipeline:** run the notebooks in order — `eda-presentation` →
`feature-engineering` → `handle-imbalance` → `modeling` → `tuning` →
`evaluation` → `interpretation`. Each stage reads from and writes to
`notebooks/processed/`.

**Run the prediction UI:**

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501). The
form collects ~30 raw employee fields and returns a Leave/Stay verdict, the
`P(Leave)` score, and the SHAP drivers (with suggested retention levers) behind
it. The hard part — faithfully reproducing the training-time transform so raw
inputs become the exact 56-feature vector the model expects — lives in
[src/preprocess.py](src/preprocess.py). See [README_UI.md](README_UI.md) for details.

---

## Tests

```bash
python tests/test_parity.py
```

This reproduces the test split, pushes every raw test row through the UI
transform, and asserts the result matches the persisted `X_test.parquet` and the
model's `predict_proba` — currently **exact (0.0 diff over all 294 rows)**.

---

## Documentation

- [CLAUDE.md](CLAUDE.md) — pipeline architecture, file structure, phase progress, conventions
- [insights.md](insights.md) — detailed phase-by-phase findings and metrics
- [README_UI.md](README_UI.md) — prediction UI usage and correctness verification

> **Note:** Predictions are decision support for HR retention review, not a
> judgement on any individual employee.
