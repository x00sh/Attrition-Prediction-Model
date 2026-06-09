### Project: Employee Attrition Prediction (HR Analytics)

**Problem Statement:** A company is losing valuable employees and wants to retain its workforce. Build a classification model that predicts whether an employee is likely to **leave the organization**, so HR can intervene with retention strategies for at-risk, high-value staff.

**Type:** Binary classification (Attrition: Yes / No)

**IMPORTANT:** Each time a new phase is implemented, update implementation progress and write a brief summary with the key insights below.

---

### Dataset

Use the **IBM HR Analytics Employee Attrition dataset** (Kaggle, ~1,470 rows, 35 features).

#### Columns used

| Column | Type | Notes |
|---|---|---|
| **Age** | Numeric | Employee age |
| **Attrition** | Categorical | Target variable — Yes / No |
| **BusinessTravel** | Categorical | Non-Travel / Travel_Rarely / Travel_Frequently |
| **DailyRate** | Numeric | Daily pay rate |
| **Department** | Categorical | HR / R&D / Sales |
| **DistanceFromHome** | Numeric | Miles from home |
| **Education** | Ordinal (1–5) | 1=Below College … 5=Doctor |
| **EducationField** | Categorical | HR / Life Sciences / Marketing / Medical / Other / Technical Degree |
| **EnvironmentSatisfaction** | Ordinal (1–4) | 1=Low … 4=Very High |
| **Gender** | Categorical | Female / Male |
| **HourlyRate** | Numeric | Hourly pay rate |
| **JobInvolvement** | Ordinal (1–4) | 1=Low … 4=Very High |
| **JobLevel** | Ordinal (1–5) | Seniority level |
| **JobRole** | Categorical | 9 roles (e.g. Manager, Sales Exec, Research Scientist) |
| **JobSatisfaction** | Ordinal (1–4) | 1=Low … 4=Very High |
| **MaritalStatus** | Categorical | Divorced / Married / Single |
| **MonthlyIncome** | Numeric | Monthly salary |
| **MonthlyRate** | Numeric | Monthly rate (different from income) |
| **NumCompaniesWorked** | Numeric | Prior employers count |
| **OverTime** | Categorical | Yes / No |
| **PercentSalaryHike** | Numeric | Last raise percentage |
| **PerformanceRating** | Ordinal (1–4) | 1=Low … 4=Outstanding |
| **RelationshipSatisfaction** | Ordinal (1–4) | 1=Low … 4=Very High |
| **StockOptionLevel** | Ordinal (0–3) | Stock option grant level |
| **TotalWorkingYears** | Numeric | Total career experience |
| **TrainingTimesLastYear** | Numeric | Training sessions attended |
| **WorkLifeBalance** | Ordinal (1–4) | 1=Bad … 4=Best |
| **YearsAtCompany** | Numeric | Tenure at current company |
| **YearsInCurrentRole** | Numeric | Time in current role |
| **YearsSinceLastPromotion** | Numeric | Time since last promotion |
| **YearsWithCurrManager** | Numeric | Time with current manager |

---

### Objectives

1. Predict attrition with strong recall (catching employees likely to leave matters most).
2. Identify the key drivers of attrition to guide retention policy.

---

### Suggested Workflow

|Phase|Tasks|
|---|---|
|**1. EDA**|Attrition rate by overtime/role/satisfaction/income, age and tenure patterns, class imbalance check|
|**2. Preprocessing**|Encode categoricals (role, travel, marital status), scale numerics, drop constant/ID columns|
|**3. Feature Engineering**|Income-per-level ratios, tenure ratios, satisfaction composite scores, compensation scores|
|**4. Handle Imbalance**|Attrition is the minority (~16%) — use SMOTE, class weights, or threshold tuning|
|**5. Modeling**|Logistic Regression (interpretable baseline) → Random Forest → XGBoost/LightGBM|
|**6. Tuning**|GridSearchCV; tune threshold to favor recall on leavers|
|**7. Evaluation**|Recall, Precision, F1, ROC-AUC, PR-AUC; confusion matrix|
|**8. Interpretation**|SHAP / feature importance — which factors most drive employees to leave|

---

### Implementation Progress

| Phase | Status | Notes |
|---|---|---|
| **1. EDA** | Complete | Key drivers identified; class imbalance confirmed |
| **2. Preprocessing** | Complete | Train/test split, encoding, scaling done |
| **3. Feature Engineering** | Complete | 46 new features across 6 groups |
| **4. Handle Imbalance** | Complete | SMOTE applied; evaluate_model() helper defined |
| **5. Modeling** | Complete | LR / RF / XGB / LGBM baselines; LR selected (Recall 0.468, F1 0.512) |
| **6. Tuning** | Pending | — |
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
- Stratified 80/20 train/test split preserves 16% attrition rate in both sets
  - Train: 1,176 rows (16.2% attrition) | Test: 294 rows (16.0% attrition)
- Categorical encoding strategy:
  - Binary: `Gender` (Female=0, Male=1), `OverTime` (No=0, Yes=1)
  - Ordinal: `BusinessTravel` (Non-Travel=0, Travel_Rarely=1, Travel_Frequently=2)
  - One-hot (with dropped reference): `Department` (ref=HR), `EducationField` (ref=Below College), `JobRole` (ref=Sales Executive), `MaritalStatus` (ref=Divorced)
- StandardScaler fit on X_train only, applied to X_test — no leakage
- **Final shapes**: X_train (1176, 76), X_test (294, 76); 59 scaled floats + 17 boolean flags; zero NaN values

---

#### Phase 3 — Feature Engineering Insights

46 new features created across 6 groups. Key signals found:

**Strongest attrition signals from engineered features:**
- `LowIncome_OT` (IsLowIncome × OverTime): **54.1% attrition** — highest interaction signal in dataset
- `Single_OT` (Single × OverTime): 49.6% attrition (131 employees)
- `IsHighRisk` (OverTime + IsLowIncome + EarlyTenure, 3-way flag): 49.6% attrition
- `FreqTravel_LowWLB` (Frequent travel × WLB=1): 46.2% attrition (13 employees)

**Compensation (4 features):** `IncomePerLevel`, `CompensationScore`, `IncomeVsExperience`, `RateDiscrepancy`
- Leavers earn ~8% less per job level when controlling for seniority

**Tenure & Career (5 features):** `TenureRatio`, `PromotionStagnationRatio`, `RoleStability` (corr –0.16), `ManagerStability`, `JobHoppingIndex`
- TenureRatio (YearsAtCompany / TotalWorkingYears): leavers median 0.50 vs stayers 0.67
- JobHoppingIndex (NumCompaniesWorked / TotalWorkingYears): leavers median 0.33 vs stayers 0.17

**Satisfaction Composites (3 features):** `SatisfactionComposite` (corr –0.16), `EngagementScore`, `WellbeingScore`

**Risk Flags (4 features):** `EarlyTenure`, `LowWLBFlag`, `IsLowEngagement`, `PromotionOverdue`

**Post-split features (computed from train stats only to prevent leakage):** `IsLowIncome` (threshold: $3,733 = 33rd pct of train), `LowIncome_OT`, `IsHighRisk`, `PeerRelativeIncome` (vs JobRole × JobLevel peer median)

---

#### Phase 4 — Handle Imbalance Insights

- **Imbalance confirmed**: 986 No / 190 Yes in training set (5.2:1 ratio); a constant "No" classifier scores 83.8% accuracy — accuracy is not a useful metric
- **SMOTE applied** (`k_neighbors=5`, `random_state=42`) to training data only → `X_train_smote` (1,972 × 76), `y_train_smote` (986 No / 986 Yes, 50/50)
  - 796 synthetic minority examples generated by interpolation in feature space
  - `X_train_smote` returned as a DataFrame (column names preserved for SHAP in Phase 8)
- **Test set untouched**: `X_test` (294 × 76) and `y_test` retain the original 16% attrition rate — evaluation always reflects real-world class frequencies
- **Two training datasets available for Phase 5**:
  - `X_train_smote` / `y_train_smote` — for models without a native class-weight parameter (e.g. base Logistic Regression)
  - `X_train` / `y_train` — used with `class_weight='balanced'` in sklearn estimators (Random Forest, XGBoost, LightGBM)
- **Threshold tuning** deferred to Phase 6 after model selection
- **`evaluate_model(model, X, y, label="")`** helper defined — prints classification report (precision / recall / F1 per class), ROC-AUC, PR-AUC, and confusion matrix; reused in Phases 5–7
- **Priority metrics established**: Recall (Yes) → F1 (Yes) → ROC-AUC → PR-AUC

#### Phase 5 — Modeling Insights

- Four classifiers trained at default hyperparameters on test set (294 samples, 16% attrition)
- Training data policy:
  - Logistic Regression: X_train_smote / y_train_smote (SMOTE 50/50)
  - Random Forest: X_train + class_weight=balanced
  - XGBoost: X_train + scale_pos_weight=5.19
  - LightGBM: X_train + is_unbalance=True
- **Phase 5 winner:** Logistic Regression (highest Recall (Yes) = 0.468 on test set; best F1 0.512)
- Test-set results:

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| LR | 0.468 | 0.564 | 0.512 | 0.795 | 0.547 |
| RF | 0.447 | 0.488 | 0.467 | 0.782 | 0.446 |
| XGBoost | 0.362 | 0.586 | 0.447 | 0.791 | 0.530 |
| LGBM | 0.277 | 0.650 | 0.388 | 0.797 | 0.537 |

- Tree models (RF, XGB, LGBM) sacrifice recall for precision at default thresholds; LR with SMOTE balances both better pre-tuning
- LGBM leads on ROC-AUC (0.797) and Precision (0.650) but trails badly on Recall — threshold tuning in Phase 6 could shift this ranking
- Threshold tuning and GridSearchCV deferred to Phase 6