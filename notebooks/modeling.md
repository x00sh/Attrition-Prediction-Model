# Modeling

**Goal:** train baseline classifiers that predict employee attrition, and pick a leader to carry into tuning. The business cost of missing a leaver outweighs a false alarm, so **recall on the "Yes" (left) class is the priority metric**, with F1, ROC-AUC and PR-AUC as supporting evidence. Accuracy is deliberately ignored — a constant "No" classifier already scores ~84%.

Four families are compared, spanning the interpretable-to-powerful range: Logistic Regression, Random Forest, XGBoost, and LightGBM.

This stage stays scoped to model **selection**. Hyperparameter search and threshold tuning are Phase 6, the full evaluation (confusion matrix, ROC/PR curves) is Phase 7, and SHAP interpretation is Phase 8.

###

## 1. Setup

Standard imports plus the four estimators and the imbalanced-learn pipeline. `RANDOM_STATE = 42` matches every earlier stage, and `PROC` points at the `processed/` directory that the preprocessing and imbalance stages wrote to.

###

## 2. Load the modeling matrices

All inputs come from `processed/` — there is no in-memory hand-off between stage notebooks.

- `X_train` / `X_test` (1176 × 56 / 294 × 56) and their targets carry the real ~16% attrition rate.
- `X_train_smote` / `y_train_smote` (1972 × 56, balanced 50/50) are loaded **only for the final full-train fit of Logistic Regression**. They are never passed to cross-validation: doing so would let synthetic rows leak across folds and inflate the scores.

###

## 3. Evaluation helper

`evaluate_model()` is defined locally because helper functions do not carry across stage notebooks. It reports the recall-first metric set — recall, precision and F1 on the positive class, plus ROC-AUC and PR-AUC — at the **default 0.5 threshold**. Moving that threshold to favour recall is a Phase 6 task.

###

## 4. Imbalance weight for the tree models

The tree models handle the 5.2:1 class imbalance natively rather than by resampling. `scale_pos_weight = #negatives / #positives` is computed from the training labels (≈ 5.19) and fed to XGBoost and LightGBM; Random Forest uses `class_weight='balanced'`, which derives the same idea internally.

###

## 5. Model specifications

Each model is paired with the imbalance strategy that suits it:

| Model | Imbalance strategy |
|---|---|
| Logistic Regression | SMOTE **inside a pipeline** — resampling is refit on each training fold |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` |
| LightGBM | `scale_pos_weight` |

Wrapping SMOTE in an `imblearn` pipeline is what makes the cross-validation in the next cell leakage-free: the validation fold is always real, untouched data.

The feature matrix is already standardized (the scaler was fit in the feature-engineering stage), so Logistic Regression consumes it directly with no re-scaling.

###

## 6. Cross-validate every model

5-fold `StratifiedKFold` cross-validation on `X_train` — **not** the precomputed SMOTE matrix — scored on recall, F1, ROC-AUC and PR-AUC (`average_precision`, the area under the precision-recall curve). Stratification preserves the ~16% attrition rate in every fold.

The results are sorted by mean recall, then F1, matching the priority order set in Phase 4. This cross-validated comparison — not the test set — is what selects the leader, keeping the test set untouched for Phase 7.

###

## 7. Light test-set readout (context only)

For a sanity check, each model is fit on the full training data and read off against the held-out test set. Logistic Regression is fit on the precomputed SMOTE matrix; the tree models on the original training data.

This readout is **indicative only** — it exists to confirm the CV picture transfers to unseen data and to catch anything obviously broken. The authoritative, presentation-grade test evaluation (confusion matrix, curves, final metric table) is Phase 7.

###

## 8. Select the leader and persist

The leader is the top row of the CV table (recall, then F1). Every fitted model is saved to `processed/models/` so Phase 6 can tune the chosen one without re-running the comparison, alongside:

- `cv_results.parquet` — the cross-validated comparison table
- `model_selection.joblib` — a small record of the leader and the selection metric

Downstream: Phase 6 loads the leader for GridSearch + threshold tuning, Phase 7 evaluates it formally on the test set, and Phase 8 runs SHAP on it (column names were preserved through every stage for exactly this).