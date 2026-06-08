### Project: Employee Attrition Prediction (HR Analytics)

**Problem Statement:** A company is losing valuable employees and wants to retain its workforce. Build a classification model that predicts whether an employee is likely to **leave the organization**, so HR can intervene with retention strategies for at-risk, high-value staff.

**Type:** Binary classification (Attrition: Yes / No)

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
|**3. Feature Engineering**|Income-per-level ratios, tenure ratios, satisfaction composite scores|
|**4. Handle Imbalance**|Attrition is the minority (~16%) — use SMOTE, class weights, or threshold tuning|
|**5. Modeling**|Logistic Regression (interpretable baseline) → Random Forest → XGBoost/LightGBM|
|**6. Tuning**|GridSearchCV; tune threshold to favor recall on leavers|
|**7. Evaluation**|Recall, Precision, F1, ROC-AUC, PR-AUC; confusion matrix|
|**8. Interpretation**|SHAP / feature importance — which factors most drive employees to leave|