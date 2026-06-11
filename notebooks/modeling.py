# Cell 1 — Imports & setup
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)

RANDOM_STATE = 42
PROC = Path('../processed')

###

# Cell 2 — Load the modeling matrices
X_train = pd.read_parquet(PROC / 'X_train.parquet')
X_test = pd.read_parquet(PROC / 'X_test.parquet')
y_train = pd.read_parquet(PROC / 'y_train.parquet')['Attrition']
y_test = pd.read_parquet(PROC / 'y_test.parquet')['Attrition']

# SMOTE matrices are used only for the FINAL full-train fit of Logistic Regression.
# They are never fed to cross-validation (that would leak synthetic rows across folds).
X_train_smote = pd.read_parquet(PROC / 'X_train_smote.parquet')
y_train_smote = pd.read_parquet(PROC / 'y_train_smote.parquet')['Attrition']

print(f'X_train       {X_train.shape} | attrition {y_train.mean():.3f}')
print(f'X_test        {X_test.shape}  | attrition {y_test.mean():.3f}')
print(f'X_train_smote {X_train_smote.shape} | attrition {y_train_smote.mean():.3f}')

###

# Cell 3 — Evaluation helper
# Defined here because helpers do not carry across stage notebooks.
# Reports the recall-first metric set at the default 0.5 threshold;
# threshold tuning is deferred to Phase 6.
def evaluate_model(model, X, y, name='Model'):
    """Return a dict of recall/precision/F1 (on the positive 'Yes' class),
    ROC-AUC and PR-AUC for a fitted model on (X, y)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        'model': name,
        'recall_yes': recall_score(y, y_pred),
        'precision_yes': precision_score(y, y_pred),
        'f1_yes': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba),
        'pr_auc': average_precision_score(y, y_proba),
    }

###

# Cell 4 — Imbalance weight for the tree models
# scale_pos_weight = #negatives / #positives, computed from the training labels.
spw = (y_train == 0).sum() / (y_train == 1).sum()
print(f'scale_pos_weight = {spw:.3f}  ({(y_train == 0).sum()} No / {(y_train == 1).sum()} Yes)')

###

# Cell 5 — Model specifications
# Each entry pairs an estimator with its imbalance strategy:
#   - LogReg uses SMOTE *inside* a pipeline so resampling happens per CV fold.
#   - Tree models handle imbalance natively (class_weight / scale_pos_weight).
models = {
    'LogReg (SMOTE)': ImbPipeline([
        ('smote', SMOTE(k_neighbors=5, random_state=RANDOM_STATE)),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    'RandomForest': RandomForestClassifier(
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
    ),
    'XGBoost': XGBClassifier(
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    'LightGBM': LGBMClassifier(
        scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    ),
}

###

# Cell 6 — Cross-validate every model on the training set
# 5-fold stratified CV on X_train (NOT the precomputed SMOTE matrix).
# The SMOTE step inside the LogReg pipeline is refit on each training fold only.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',  # area under the precision-recall curve
}

rows = []
for name, est in models.items():
    cvres = cross_validate(est, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    rows.append({
        'model': name,
        'recall': cvres['test_recall'].mean(),
        'recall_std': cvres['test_recall'].std(),
        'f1': cvres['test_f1'].mean(),
        'roc_auc': cvres['test_roc_auc'].mean(),
        'pr_auc': cvres['test_pr_auc'].mean(),
    })

cv_results = (
    pd.DataFrame(rows)
    .sort_values(['recall', 'f1'], ascending=False)
    .reset_index(drop=True)
)
cv_results

###

# Cell 7 — Light test-set readout (context only)
# Fit each model on the FULL training data and read off test metrics.
# This is indicative only — the authoritative test evaluation is Phase 7.
# LogReg is fit on the precomputed SMOTE matrix; trees on the original train.
fitted = {}
test_rows = []
for name, est in models.items():
    if name == 'LogReg (SMOTE)':
        # Use the plain LR (no in-pipeline SMOTE) on the precomputed SMOTE matrix.
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_train_smote, y_train_smote)
    else:
        model = est
        model.fit(X_train, y_train)
    fitted[name] = model
    test_rows.append(evaluate_model(model, X_test, y_test, name=name))

test_results = (
    pd.DataFrame(test_rows)
    .sort_values(['recall_yes', 'f1_yes'], ascending=False)
    .reset_index(drop=True)
)
test_results

###

# Cell 8 — Select the leader and persist artifacts
# Leader chosen by cross-validated recall, then F1 (the priority order from Phase 4).
leader = cv_results.iloc[0]['model']
print(f'CV leader (recall -> F1): {leader}')

OUT = PROC / 'models'
OUT.mkdir(parents=True, exist_ok=True)

# Persist every fitted model so Phase 6 can tune the leader without re-deciding.
slug = {
    'LogReg (SMOTE)': 'logreg_smote',
    'RandomForest': 'random_forest',
    'XGBoost': 'xgboost',
    'LightGBM': 'lightgbm',
}
for name, model in fitted.items():
    joblib.dump(model, OUT / f'{slug[name]}.joblib')

cv_results.to_parquet(PROC / 'cv_results.parquet', index=False)
joblib.dump({'leader': leader, 'metric': 'cv_recall_then_f1'}, PROC / 'model_selection.joblib')

print(f'Saved {len(fitted)} models to {OUT.resolve()}')
print(f'Saved cv_results.parquet and model_selection.joblib to {PROC.resolve()}')