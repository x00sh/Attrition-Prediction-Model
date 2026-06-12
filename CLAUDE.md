### Project: Employee Attrition Prediction (HR Analytics)

**Problem Statement:** A company is losing valuable employees and wants to retain its workforce. Build a classification model that predicts whether an employee is likely to **leave the organization**, so HR can intervene with retention strategies for at-risk, high-value staff.

**Type:** Binary classification (Attrition: Yes / No)

**IMPORTANT:** Each time a new phase is implemented, update implementation progress and write a brief summary with the key insights below.

---

### Dataset

Use the **IBM HR Analytics Employee Attrition dataset** (Kaggle) — `WA_Fn-UseC_-HR-Employee-Attrition.csv` at repo root, 1,470 rows × 35 columns. Features include:

- **Job details:** job role, department, job level, years at company, years in current role
- **Compensation:** monthly income, salary hike percentage, stock options
- **Satisfaction:** job satisfaction, environment satisfaction, work-life balance, relationship satisfaction
- **Work patterns:** overtime, business travel frequency, distance from home
- **Demographics:** age, education, marital status
- **Target:** Attrition (Yes / No)

---

### File Structure

```
├── WA_Fn-UseC_-HR-Employee-Attrition.csv       # Raw dataset (1,470 rows × 35 cols)
├── notebooks/
│   ├── eda-presentation.ipynb
│   ├── feature-engineering.ipynb
│   ├── handle-imbalance.ipynb
│   ├── modeling.ipynb
│   ├── tuning.py, tuning.md                    # Phase 6 source cells (merge via combine_notebook.py)
│   ├── tuning.ipynb                            # Phase 6 — complete
│   ├── evaluation.ipynb                        # Phase 7 — complete
│   └── outputs/                                # Phase 7 figures (confusion_matrices, roc_curves, pr_curves, threshold_tradeoff .png)
├── processed/                                  # Persisted outputs (train/test splits, scaler, SMOTE matrices)
│   ├── X_train.parquet, X_test.parquet
│   ├── y_train.parquet, y_test.parquet
│   ├── X_train_smote.parquet, y_train_smote.parquet
│   ├── scaler.joblib, scale_cols.joblib
│   └── ...
└── TO BE DELETED/                              # Deprecated; do not use
    └── attrition-prediction-model.ipynb        # (Original monolithic notebook; superseded by split pipeline)
```

---

### Objectives

1. Predict attrition with strong recall (catching employees likely to leave matters most).
2. Identify the key drivers of attrition to guide retention policy.

---

### Repository Structure & Pipeline Architecture

The project is implemented as **per-stage notebooks** in `notebooks/`. Stages communicate **only via files persisted to `processed/`** — there is no in-memory hand-off between notebooks. Helper functions (e.g. an `evaluate_model()` utility) must be defined in the notebook that uses them; they cannot carry across stages.

| Stage notebook | Phases | Input | Output to `processed/` |
|---|---|---|---|
| `notebooks/eda-presentation.ipynb` | 1. EDA | raw CSV | none (presentation only) |
| `notebooks/feature-engineering.ipynb` | 2–3. Preprocessing + Feature Engineering | raw CSV | `X_train.parquet` (1176×56), `X_test.parquet` (294×56), `y_train.parquet`, `y_test.parquet`, `scaler.joblib`, `scale_cols.joblib` |
| `notebooks/handle-imbalance.ipynb` | 4. Handle Imbalance | the four matrices above | `X_train_smote.parquet` (1972×56), `y_train_smote.parquet` (50/50) |
| `notebooks/modeling.ipynb` | 5. Modeling | all four matrices + SMOTE matrices | `cv_results.parquet`, `model_selection.joblib`, fitted models in `models/` (logreg_smote.joblib, random_forest.joblib, xgboost.joblib, lightgbm.joblib) |
| `notebooks/tuning.ipynb` (source: `tuning.py` + `tuning.md`) | 6. Tuning | `X_train`/`y_train`/`X_test`/`y_test` + `cv_results.parquet` (SMOTE matrices NOT used — SMOTE lives inside the LogReg pipeline) | `tuning_results.parquet`, `threshold_sweep.parquet`, `best_params.joblib`, `tuning_selection.joblib` (leader + threshold), `models/*_tuned.joblib`, `models/final_model.joblib` |
| `notebooks/evaluation.ipynb` | 7. Evaluation | `X_test`/`y_test` + `models/*_tuned.joblib` + `tuning_selection.joblib`, `tuning_results.parquet`, `threshold_sweep.parquet`, `cv_results.parquet` | `evaluation_results.parquet` (test bake-off), `final_evaluation.joblib` (confirmed model + test metrics), 4 PNGs in `outputs/`; `models/final_model.joblib` re-pointed only if test contradicts OOF leader |

**Convention for future stages (5–8):** load everything you need from `processed/`, persist anything a downstream stage will need.

**Deprecated:** `TO BE DELETED/attrition-prediction-model.ipynb` is the original monolithic notebook. Do not extend it — its feature matrix (76 cols) and results are superseded by the split pipeline (56 cols).

---

### Suggested Workflow

|Phase|Tasks|
|---|---|
|**1. EDA**|Attrition rate by overtime/role/satisfaction/income, age and tenure patterns, class imbalance check|
|**2. Preprocessing**|Encode categoricals (role, travel, marital status), scale numerics, drop constant/ID columns|
|**3. Feature Engineering**|Income-per-level ratios, tenure ratios, satisfaction composite scores|
|**4. Handle Imbalance**|Attrition is the minority (~16%) — use SMOTE, class weights, or threshold tuning|
|**5. Modeling**|Logistic Regression (interpretable baseline) → Random Forest → XGBoost/LightGBM|
|**6. Tuning**|GridSearchCV; tune threshold to favor recall on leavers|
|**7. Evaluation**|Recall, Precision, F1, ROC-AUC, PR-AUC; confusion matrix|
|**8. Interpretation**|SHAP / feature importance — which factors most drive employees to leave|

---

### Implementation Progress

| Phase | Status | Notes |
|---|---|---|
| **1. EDA** | Complete | `notebooks/eda-presentation.ipynb` — key drivers identified; class imbalance confirmed |
| **2. Preprocessing** | Complete | `notebooks/feature-engineering.ipynb` — split, encoding, scaling |
| **3. Feature Engineering** | Complete | `notebooks/feature-engineering.ipynb` — 13 new features across 4 groups |
| **4. Handle Imbalance** | Complete | `notebooks/handle-imbalance.ipynb` — SMOTE applied, outputs persisted |
| **5. Modeling** | Complete | `notebooks/modeling.ipynb` — 4 models compared; LogReg (SMOTE) selected as leader |
| **6. Tuning** | Complete | `notebooks/tuning.ipynb` — RandomForest selected as tuned leader (recall 0.716 OOF); all four families tuned via GridSearchCV on F2; thresholds swept on OOF probabilities |
| **7. Evaluation** | Complete | `notebooks/evaluation.ipynb` — held-out bake-off; RandomForest confirmed (test recall 0.830, precision 0.315, ROC-AUC 0.766); OOF→test gains generalise; deployed model unchanged |
| **8. Interpretation** | Pending | — |

---

### Detailed Phase Insights

See **[insights.md](insights.md)** for comprehensive phase-by-phase summaries, including:
- Key findings and patterns discovered in each phase
- Dataset characteristics and preprocessing decisions
- Feature engineering rationale and engineering choices
- Imbalance handling strategies and results
- Model performance comparisons and selection criteria
- Hyperparameter tuning results and threshold optimization
- Artifacts persisted for downstream stages
