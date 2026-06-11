# Cell 1 — Imports & setup
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_predict, ParameterGrid,
)
from sklearn.metrics import (
    recall_score, precision_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, make_scorer,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)

RANDOM_STATE = 42
PROC = Path('../processed')

###

# Cell 2 — Load the modeling matrices and the Phase 5 baseline
# The precomputed SMOTE matrices are NOT loaded here: in this stage SMOTE always
# lives inside the LogReg pipeline, so resampling is refit on each CV training fold.
X_train = pd.read_parquet(PROC / 'X_train.parquet')
X_test = pd.read_parquet(PROC / 'X_test.parquet')
y_train = pd.read_parquet(PROC / 'y_train.parquet')['Attrition']
y_test = pd.read_parquet(PROC / 'y_test.parquet')['Attrition']

# Phase 5 cross-validated results, for the before/after-tuning comparison.
baseline_cv = pd.read_parquet(PROC / 'cv_results.parquet')

# scale_pos_weight = #negatives / #positives (same computation as Phase 5).
spw = (y_train == 0).sum() / (y_train == 1).sum()

print(f'X_train {X_train.shape} | attrition {y_train.mean():.3f}')
print(f'X_test  {X_test.shape}  | attrition {y_test.mean():.3f}')
print(f'scale_pos_weight = {spw:.3f}')

###

# Cell 3 — Helpers, scorers and the search wrapper
# Defined here because helpers do not carry across stage notebooks.
def metrics_at_threshold(y_true, proba, threshold):
    """Recall/precision/F1/F2 on the positive 'Yes' class at a probability threshold."""
    pred = (proba >= threshold).astype(int)
    return {
        'recall': recall_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred),
        'f2': fbeta_score(y_true, pred, beta=2),
    }


def sweep_thresholds(y_true, proba, thresholds):
    """One row of threshold metrics per candidate threshold."""
    rows = [{'threshold': t, **metrics_at_threshold(y_true, proba, t)} for t in thresholds]
    return pd.DataFrame(rows)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'f2': make_scorer(fbeta_score, beta=2),
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',
}


def run_search(estimator, param_grid, name):
    """Exhaustive GridSearchCV refit on F2; prints candidate count and elapsed time."""
    n_candidates = len(ParameterGrid(param_grid))
    print(f'{name}: {n_candidates:,} candidates x 5 folds = {n_candidates * 5:,} fits')
    search = GridSearchCV(
        estimator, param_grid, scoring=scoring, refit='f2',
        cv=cv, n_jobs=-1, verbose=1,
    )
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f'{name}: done in {elapsed / 60:.1f} min | best CV F2 = {search.best_score_:.3f}')
    print(f'{name}: best params = {search.best_params_}')
    return search


searches = {}

###

# Cell 4 — Logistic Regression search (SMOTE-on and SMOTE-off variants)
# The grid is a list of dicts so invalid combinations (e.g. l1_ratio without
# elasticnet) are never generated. saga supports all three penalties.
C_LIST = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

logreg_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('clf', LogisticRegression(solver='saga', max_iter=5000, random_state=RANDOM_STATE)),
])

smote_on = {
    'smote__sampling_strategy': [0.5, 0.65, 0.8, 1.0],
    'smote__k_neighbors': [3, 5, 7, 9],
    'clf__class_weight': [None, 'balanced'],
}
smote_off = {
    'smote': ['passthrough'],
    'clf__class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}],
}
penalties = [
    {'clf__penalty': ['l2'], 'clf__C': C_LIST},
    {'clf__penalty': ['l1'], 'clf__C': C_LIST},
    {'clf__penalty': ['elasticnet'], 'clf__C': C_LIST, 'clf__l1_ratio': [0.2, 0.5, 0.8]},
]

logreg_grid = [
    {**variant, **penalty}
    for variant in (smote_on, smote_off)
    for penalty in penalties
]

searches['LogReg (SMOTE)'] = run_search(logreg_pipe, logreg_grid, 'LogReg (SMOTE)')

###

# Cell 5 — Random Forest search
# n_jobs=1 on the estimator: GridSearchCV already parallelises across candidates,
# and nesting parallelism oversubscribes the CPU.
rf_grid = {
    'n_estimators': [300, 600],
    'max_depth': [None, 4, 6, 8, 12, 16],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 0.3, 0.5],
    'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 4}, {0: 1, 1: 6}, {0: 1, 1: 8}],
}

searches['RandomForest'] = run_search(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    rf_grid, 'RandomForest',
)

###

# Cell 6 — XGBoost search
xgb_grid = {
    'n_estimators': [200, 400, 800],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 6],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
    'gamma': [0, 1],
    'reg_alpha': [0, 1],
    'reg_lambda': [1, 5],
    'scale_pos_weight': [0.5 * spw, spw, 1.5 * spw],
}

searches['XGBoost'] = run_search(
    XGBClassifier(eval_metric='logloss', tree_method='hist',
                  random_state=RANDOM_STATE, n_jobs=1),
    xgb_grid, 'XGBoost',
)

###

# Cell 7 — LightGBM search
# subsample_freq=1 is required for subsample to take effect in LightGBM.
lgbm_grid = {
    'n_estimators': [200, 400, 800],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [7, 15, 31, 63],
    'max_depth': [-1, 4, 8],
    'min_child_samples': [5, 10, 20, 40],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
    'reg_alpha': [0, 1],
    'reg_lambda': [0, 5],
    'scale_pos_weight': [0.5 * spw, spw, 1.5 * spw],
}

searches['LightGBM'] = run_search(
    LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, subsample_freq=1),
    lgbm_grid, 'LightGBM',
)

###

# Cell 8 — Tuned vs Phase 5 baseline (default 0.5 threshold)
# What hyperparameter search alone bought, before any threshold movement.
rows = []
for name, search in searches.items():
    res = search.cv_results_
    i = search.best_index_
    rows.append({
        'model': name,
        'recall': res['mean_test_recall'][i],
        'precision': res['mean_test_precision'][i],
        'f1': res['mean_test_f1'][i],
        'f2': res['mean_test_f2'][i],
        'roc_auc': res['mean_test_roc_auc'][i],
        'pr_auc': res['mean_test_pr_auc'][i],
    })

tuned_cv = pd.DataFrame(rows)
compare = tuned_cv.merge(
    baseline_cv[['model', 'recall', 'f1']], on='model', suffixes=('_tuned', '_phase5'),
)
compare['recall_gain'] = compare['recall_tuned'] - compare['recall_phase5']
compare['f1_gain'] = compare['f1_tuned'] - compare['f1_phase5']
print(compare.sort_values('f2', ascending=False).to_string(index=False))

###

# Cell 9 — Out-of-fold probabilities and the threshold sweep
# OOF probabilities come from cross_val_predict with the same 5-fold splitter, so
# every probability is produced by a model that never saw that row. The test set
# plays no part in choosing the threshold.
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)

oof_proba = {}
tuned_threshold = {}
sweeps = []
criteria_rows = []
for name, search in searches.items():
    proba = cross_val_predict(
        search.best_estimator_, X_train, y_train,
        cv=cv, method='predict_proba', n_jobs=-1,
    )[:, 1]
    oof_proba[name] = proba

    sweep = sweep_thresholds(y_train, proba, THRESHOLDS)
    sweep.insert(0, 'model', name)
    sweeps.append(sweep)

    crit_list = [
        ('max F2', sweep.loc[sweep['f2'].idxmax(), 'threshold']),
        ('max F1', sweep.loc[sweep['f1'].idxmax(), 'threshold']),
    ]
    feasible = sweep[sweep['precision'] >= 0.30]
    if len(feasible):
        crit_list.append((
            'max recall @ precision >= 0.30',
            feasible.loc[feasible['recall'].idxmax(), 'threshold'],
        ))
    crit_list.append(('default 0.50', 0.50))

    tuned_threshold[name] = crit_list[0][1]  # primary criterion: max OOF F2
    for crit, t in crit_list:
        criteria_rows.append({
            'model': name, 'criterion': crit, 'threshold': t,
            **metrics_at_threshold(y_train, proba, t),
        })

threshold_sweep = pd.concat(sweeps, ignore_index=True)
criteria_table = pd.DataFrame(criteria_rows)
print(criteria_table.to_string(index=False))

###

# Cell 10 — Final comparison and leader selection
# OOF metrics at each model's tuned (max-F2) threshold, ranked by the project rule:
# recall first, F1 tiebreak.
final_rows = []
for name in searches:
    t = tuned_threshold[name]
    proba = oof_proba[name]
    final_rows.append({
        'model': name,
        'threshold': t,
        **metrics_at_threshold(y_train, proba, t),
        'roc_auc': roc_auc_score(y_train, proba),
        'pr_auc': average_precision_score(y_train, proba),
    })

final_results = (
    pd.DataFrame(final_rows)
    .sort_values(['recall', 'f1'], ascending=False)
    .reset_index(drop=True)
)
leader = final_results.iloc[0]['model']
print(final_results.to_string(index=False))
print(f'\nTuned leader (OOF recall -> F1 at tuned threshold): {leader}')

###

# Cell 11 — Light test-set readout (context only)
# Indicative only — the authoritative test evaluation is Phase 7.
# best_estimator_ was already refit on the full training set by GridSearchCV.
leader_model = searches[leader].best_estimator_
leader_t = tuned_threshold[leader]

test_proba = leader_model.predict_proba(X_test)[:, 1]
sanity = pd.DataFrame([
    {'setting': f'tuned threshold ({leader_t:.2f})',
     **metrics_at_threshold(y_test, test_proba, leader_t)},
    {'setting': 'default threshold (0.50)',
     **metrics_at_threshold(y_test, test_proba, 0.50)},
])
sanity['roc_auc'] = roc_auc_score(y_test, test_proba)
sanity['pr_auc'] = average_precision_score(y_test, test_proba)
print(f'Leader on test (indicative only): {leader}')
print(sanity.to_string(index=False))

###

# Cell 12 — Persist artifacts for Phases 7 and 8
OUT = PROC / 'models'
OUT.mkdir(parents=True, exist_ok=True)

slug = {
    'LogReg (SMOTE)': 'logreg_smote',
    'RandomForest': 'random_forest',
    'XGBoost': 'xgboost',
    'LightGBM': 'lightgbm',
}
for name, search in searches.items():
    joblib.dump(search.best_estimator_, OUT / f'{slug[name]}_tuned.joblib')
joblib.dump(leader_model, OUT / 'final_model.joblib')

summary_rows = []
for name, search in searches.items():
    t = tuned_threshold[name]
    at_t = metrics_at_threshold(y_train, oof_proba[name], t)
    at_default = metrics_at_threshold(y_train, oof_proba[name], 0.50)
    base = baseline_cv.loc[baseline_cv['model'] == name].iloc[0]
    summary_rows.append({
        'model': name,
        'n_candidates': len(ParameterGrid(search.param_grid)),
        'best_cv_f2': search.best_score_,
        'threshold': t,
        'oof_recall': at_t['recall'],
        'oof_precision': at_t['precision'],
        'oof_f1': at_t['f1'],
        'oof_f2': at_t['f2'],
        'oof_roc_auc': roc_auc_score(y_train, oof_proba[name]),
        'oof_pr_auc': average_precision_score(y_train, oof_proba[name]),
        'recall_at_05': at_default['recall'],
        'f1_at_05': at_default['f1'],
        'phase5_recall': base['recall'],
        'phase5_f1': base['f1'],
    })

tuning_results = pd.DataFrame(summary_rows)
tuning_results.to_parquet(PROC / 'tuning_results.parquet', index=False)
threshold_sweep.to_parquet(PROC / 'threshold_sweep.parquet', index=False)

joblib.dump({name: search.best_params_ for name, search in searches.items()},
            PROC / 'best_params.joblib')

joblib.dump({
    'leader': leader,
    'slug': slug[leader],
    'threshold': float(leader_t),
    'threshold_criterion': 'max OOF F2',
    'refit_metric': 'f2',
    'selection_rule': 'oof_recall_then_f1_at_tuned_threshold',
    'best_params': searches[leader].best_params_,
    'oof_metrics': metrics_at_threshold(y_train, oof_proba[leader], leader_t),
}, PROC / 'tuning_selection.joblib')

print(f'Saved 4 tuned models + final_model.joblib to {OUT.resolve()}')
print(f'Saved tuning_results.parquet, threshold_sweep.parquet, best_params.joblib '
      f'and tuning_selection.joblib to {PROC.resolve()}')