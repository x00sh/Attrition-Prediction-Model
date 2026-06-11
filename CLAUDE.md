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
│   └── tuning.ipynb                            # (generated after the user confirms the run)
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
| **6. Tuning** | Source written | `notebooks/tuning.py` + `tuning.md` — awaiting user run; merge to `.ipynb` after confirming. Insights to be filled in from the run results |
| **7. Evaluation** | Pending | — |
| **8. Interpretation** | Pending | — |

---

#### Phase 1 — EDA Insights

- Dataset: 1,470 rows × 35 cols; 4 constant/ID cols dropped → 31 usable features
- No missing values or duplicates
- **Class imbalance**: 83.9% stayed / 16.1% left (5.2:1 ratio) — accuracy is misleading; recall on leavers is the priority metric
- **Top attrition drivers**:
  - OverTime: 30% attrition (Yes) vs 10% (No) — strongest single signal
  - MaritalStatus: Single 25.5% > Divorced 20.6% > Married 12.3%
  - BusinessTravel: Frequent 24.9% > Rarely 14.9% > Non-Travel 8.1%
  - JobRole: Sales Representatives and Lab Technicians highest; Managers lowest
  - JobInvolvement = 1 (Low): 33.3% attrition
  - WorkLifeBalance = 1 (Bad): 31.4% attrition
  - MonthlyIncome: leavers earn significantly less (corr –0.16 with attrition)
  - Early tenure (≤2 yrs): 29.8% attrition vs 12.0% beyond 2 years
  - YearsSinceLastPromotion: non-linear pattern — both 0 yrs (20.2%) and 6+ yrs (21.8%) are high-risk
- **Multicollinearity**: YearsAtCompany, TotalWorkingYears, YearsInCurrentRole, YearsWithCurrManager are highly intercorrelated; MonthlyIncome correlates strongly with JobLevel

---

#### Phase 2 — Preprocessing Insights

- Dropped 4 constant/ID columns: `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`
- Target encoded: `Attrition` → Yes=1 / No=0
- Deterministic encodings run **pre-split** (no data statistics involved):
  - Binary: `Gender` (Female=0, Male=1), `OverTime` (No=0, Yes=1)
  - Ordinal: `BusinessTravel` (Non-Travel=0, Travel_Rarely=1, Travel_Frequently=2)
  - One-hot (with dropped reference, `drop_first`): `Department`, `EducationField`, `JobRole`, `MaritalStatus`
  - → 45 encoded columns before feature engineering
- Stratified 80/20 train/test split preserves the ~16% attrition rate in both sets
  - Train: 1,176 rows (16.2% attrition) | Test: 294 rows (16.0% attrition)
- **Everything statistical is fit on train only** (leakage guardrail): every threshold, peer median, and the scaler are computed from `X_train` and applied to `X_test`
- StandardScaler fit on the 30 continuous columns; the 26 flag/dummy ({0,1}) columns are left unscaled; scaler persisted as `processed/scaler.joblib` + `processed/scale_cols.joblib`
- **Final shapes**: X_train (1176, 56), X_test (294, 56); zero NaN/inf; DataFrames persisted as parquet (column names preserved for SHAP in Phase 8)

---

#### Phase 3 — Feature Engineering Insights

13 new features created across 4 groups, all driven by EDA signals and computed from **train-fit statistics only**:

| Group | Features |
|---|---|
| OverTime interactions | `Single_OT`, `FreqTravel_OT`, `IsHighRisk` (OverTime × IsLowIncome × EarlyTenure) |
| Pay vs peers | `PeerRelativeIncome` (vs JobRole × JobLevel peer median), `IsLowIncome` (train 33rd pct = $3,733), `IncomePerLevel` |
| Tenure / career | `EarlyTenure` (≤2 yrs), `TenureRatio`, `PromotionOverdue` (≥6 yrs), `JobHoppingIndex` |
| Engagement / well-being | `SatisfactionComposite`, `EngagementScore`, `IsLowEngagement` |

- Raw `JobRole` / `MaritalStatus` are stashed as `_JobRole` / `_MaritalStatus` helper columns before one-hot encoding (the peer-income and Single-flag features need them), then dropped after engineering
- Ratio features guard against divide-by-zero (`TotalWorkingYears` is 0 for 11 employees)
- Peer-income groups: 26 (JobRole × JobLevel) medians from train; unseen groups fall back to the train global median

---

#### Phase 4 — Handle Imbalance Insights

- **Implemented in `notebooks/handle-imbalance.ipynb`**: loads the four matrices persisted by `notebooks/feature-engineering.ipynb` from `processed/`
- **Imbalance confirmed**: 986 No / 190 Yes in training set (5.2:1 ratio); a constant "No" classifier scores 83.8% accuracy — accuracy is not a useful metric
- **SMOTE applied** (`k_neighbors=5`, `random_state=42`) to training data only → `X_train_smote` (1,972 × 56), `y_train_smote` (986 No / 986 Yes, 50/50)
  - 796 synthetic minority examples generated by interpolation in feature space
  - `X_train_smote` kept as a DataFrame (column names preserved for SHAP in Phase 8)
- **Test set untouched**: `X_test` (294 × 56) and `y_test` retain the original 16% attrition rate — evaluation always reflects real-world class frequencies
- **Persisted for the modeling stage**: `processed/X_train_smote.parquet`, `processed/y_train_smote.parquet`
- **Two training datasets available for Phase 5**:
  - `X_train_smote` / `y_train_smote` — for models without a native class-weight parameter (e.g. base Logistic Regression)
  - `X_train` / `y_train` (already in `processed/`) — used with `class_weight='balanced'` / `scale_pos_weight` in tree models (Random Forest, XGBoost, LightGBM)
- **Threshold tuning** deferred to Phase 6 after model selection
- **Priority metrics established**: Recall (Yes) → F1 (Yes) → ROC-AUC → PR-AUC

---

#### Phase 5 — Modeling Insights

- **Implemented in `notebooks/modeling.ipynb`**: Compares four classifier families using 5-fold stratified cross-validation on `X_train` (never mixing in synthetic SMOTE samples during CV to avoid fold leakage)
- **Imbalance handling strategy per model**:
  - Logistic Regression: SMOTE inside an `imblearn` pipeline (refit on each training fold)
  - Random Forest: `class_weight='balanced'` to penalize minority misclassification
  - XGBoost & LightGBM: `scale_pos_weight ≈ 5.19` (ratio of negatives to positives)
- **Cross-validated performance (5-fold, on original ~16% attrition rate in each fold)**:
  | Model | Recall | F1 | ROC-AUC | PR-AUC |
  |---|---|---|---|---|
  | LogReg (SMOTE) | 0.516 | 0.511 | 0.793 | 0.562 |
  | LightGBM | 0.437 | 0.522 | 0.808 | 0.587 |
  | XGBoost | 0.426 | 0.520 | 0.785 | 0.563 |
  | RandomForest | 0.384 | 0.461 | 0.797 | 0.551 |
- **Leader selected**: LogReg (SMOTE) by cross-validated recall (0.516), then F1 (0.511) — highest recall captures more at-risk leavers, aligning with business priority
- **Test-set sanity check** (indicative only; authoritative evaluation deferred to Phase 7):
  - LogReg (SMOTE): 0.468 recall, 0.463 F1 — confirms CV picture transfers to unseen data
  - Other models show lower recall on test, consistent with their CV rankings
- **Artifacts persisted**:
  - `cv_results.parquet`: Full cross-validation comparison table
  - `model_selection.joblib`: Leader name + selection metric metadata
  - `processed/models/`: All four fitted models (logreg_smote.joblib, random_forest.joblib, xgboost.joblib, lightgbm.joblib) pre-fitted for Phase 6 tuning
