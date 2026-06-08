### Project: Employee Attrition Prediction (HR Analytics)

**Problem Statement:** A company is losing valuable employees and wants to retain its workforce. Build a classification model that predicts whether an employee is likely to **leave the organization**, so HR can intervene with retention strategies for at-risk, high-value staff.

**Type:** Binary classification (Attrition: Yes / No)

---

### Dataset

Use the **IBM HR Analytics Employee Attrition dataset** (Kaggle, ~1,470 rows, 35 features). Features include:

- **Job details:** job role, department, job level, years at company, years in current role
- **Compensation:** monthly income, salary hike percentage, stock options
- **Satisfaction:** job satisfaction, environment satisfaction, work-life balance, relationship satisfaction
- **Work patterns:** overtime, business travel frequency, distance from home
- **Demographics:** age, education, marital status
- **Target:** Attrition (Yes / No)

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