# Tuning

**Goal:** improve on the Phase 5 baselines with two levers:

1. **Hyperparameters** — exhaustive `GridSearchCV` over all four model families (~50,000 candidate configurations, ~254,000 individual fits), refit on **F2**.
2. **Operating point** — the decision threshold is tuned on out-of-fold training probabilities to favour recall on leavers, per the project's priority order.

Every family from Phase 5 is re-tuned, not just the leader: tuning routinely reshuffles rankings, and the comparison is cheap relative to the searches themselves. The data is small (1,176 × 56), so the full run is expected to take roughly 30–70 minutes on a modern multi-core desktop.

Scope boundary unchanged: the authoritative test evaluation (confusion matrix, curves) is Phase 7, and SHAP interpretation is Phase 8. This stage ends by persisting a final tuned model and its threshold for those stages to load.

###

## 1. Setup

The Phase 5 import set plus the tuning machinery: `GridSearchCV`, `cross_val_predict`, `ParameterGrid`, and `fbeta_score`/`make_scorer` for the F2 metric. `RANDOM_STATE = 42` and `PROC` match every earlier stage.

###

## 2. Load the modeling matrices and the Phase 5 baseline

All inputs come from `processed/` — no in-memory hand-off between stage notebooks.

Two differences from Phase 5's load cell:

- The **precomputed SMOTE matrices are not loaded**. In this stage SMOTE always sits inside the Logistic Regression pipeline, so it is refit on each CV training fold — including for the final full-train refit — and the standalone matrices are unnecessary.
- `cv_results.parquet` (the Phase 5 cross-validated table) is loaded so every tuned model can be compared against its own baseline.

`scale_pos_weight` (≈ 5.19) is recomputed from the training labels; the grids search a band around it rather than treating it as fixed.

###

## 3. Helpers, scorers and the search wrapper

Helpers are defined locally because functions do not carry across stage notebooks.

- `metrics_at_threshold()` / `sweep_thresholds()` — recall, precision, F1 and F2 on the positive class at an arbitrary probability threshold; used by the threshold sweep and every readout table.
- **Scoring:** multi-metric (recall, precision, F1, F2, ROC-AUC, PR-AUC) with **`refit='f2'`**. Refitting on pure recall would be degenerate — across ~50k candidates, the heaviest class weighting wins by predicting nearly everyone as a leaver (recall → 1.0 at base-rate precision). F2 weights recall four times as heavily as precision, so it leans the search toward the business priority while a predict-all-positive solution (F2 ≈ 0.49 at 16% prevalence) cannot win. Pushing the recall/precision operating point further is the threshold's job, not the grid's.
- `run_search()` — wraps `GridSearchCV(n_jobs=-1)`, prints the candidate count up front and the elapsed time and best parameters when done. Because `refit='f2'`, each search ends with `best_estimator_` already refit on the full training set.

The CV splitter is the same `StratifiedKFold(5, shuffle=True, random_state=42)` as Phase 5, so tuned and baseline scores are directly comparable.

###

## 4. Logistic Regression search

The Phase 5 leader gets the most structurally varied grid. It is a **list of six dicts** — {SMOTE-on, SMOTE-off} × {l2, l1, elasticnet} — so invalid combinations (such as `l1_ratio` outside elasticnet) are never generated:

- **SMOTE-on:** `sampling_strategy` 0.5–1.0 and `k_neighbors` 3–9 are searched as pipeline parameters (`smote__…`), optionally combined with `class_weight='balanced'`.
- **SMOTE-off:** `smote: 'passthrough'` disables the resampler entirely, testing whether class weights alone (including explicit 1:3 … 1:6 dicts) beat synthetic oversampling.
- Both variants cross 11 values of `C` with all three penalties (`solver='saga'` supports them all; `max_iter=5000` because saga converges slowly — the features were standardized in Phase 2, so it is well-conditioned).

≈ 2,035 candidates → ~10,200 fits.

###

## 5. Random Forest search

Tree shape (`max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`), forest size, and five imbalance treatments — `'balanced'`, `'balanced_subsample'`, and explicit 1:4 / 1:6 / 1:8 weight dicts that punish missed leavers harder than the ~5.2:1 base rate.

The estimator runs with `n_jobs=1`: `GridSearchCV` already parallelises across candidates, and nesting parallelism inside each fit oversubscribes the CPU.

2,160 candidates → 10,800 fits. This is the slowest family per fit.

###

## 6. XGBoost search

The full regularisation surface: boosting size vs learning rate, tree complexity (`max_depth`, `min_child_weight`, `gamma`), row/column subsampling, and both L1/L2 penalties — crossed with three `scale_pos_weight` settings (half, exact, and 1.5× the 5.19 base ratio). `tree_method='hist'` keeps each fit fast on this data size.

15,552 candidates → 77,760 fits.

###

## 7. LightGBM search

The leaf-wise analogue of the XGBoost grid: `num_leaves` is LightGBM's primary complexity control, searched alongside `max_depth`, `min_child_samples`, subsampling, regularisation, and the same three `scale_pos_weight` settings. `subsample_freq=1` is required — without it LightGBM silently ignores `subsample`.

31,104 candidates → 155,520 fits; LightGBM is the fastest family per fit.

###

## 8. Tuned vs Phase 5 baseline

Cross-validated metrics of each family's best configuration at the **default 0.5 threshold**, joined against the Phase 5 baseline table. This isolates what hyperparameter search alone bought, before the threshold moves — the two levers are reported separately so their contributions aren't conflated.

###

## 9. Out-of-fold probabilities and the threshold sweep

The threshold cannot be chosen on the test set (that would leak it into model selection), and the full-train fit's in-sample probabilities would be optimistic. Instead, `cross_val_predict` produces **out-of-fold probabilities**: every training row is scored by a model that never saw it. The LogReg pipeline re-applies SMOTE inside each fold, so this stays leakage-free.

Each model's probabilities are swept across 91 thresholds (0.05–0.95, step 0.01) and four criteria are tabulated:

| Criterion | Rationale |
|---|---|
| **max F2** (primary) | recall-weighted, no arbitrary constants, comparable across models |
| max F1 | the balanced reference point |
| max recall @ precision ≥ 0.30 | an HR-capacity framing — flag as many leavers as possible while ~1 in 3 flags is real (≈ 2× the 16% base rate) |
| default 0.50 | what the model does untouched |

The **max-F2 threshold** is the one carried forward per model; the other rows show the trade-off space.

###

## 10. Final comparison and leader selection

Each model's out-of-fold metrics at its own tuned threshold, ranked by the project rule from Phase 4: **recall first, F1 tiebreak**. The top row is the tuned leader. ROC-AUC and PR-AUC are threshold-free and reported for context.

###

## 11. Light test-set readout (context only)

The Phase 5 pattern: the leader (already refit on the full training set by `GridSearchCV`) is read off against the held-out test set at both the tuned and the default threshold. **Indicative only** — it confirms the out-of-fold picture transfers to unseen data; the authoritative evaluation is Phase 7.

###

## 12. Persist artifacts

Everything Phases 7 and 8 need, written to `processed/`:

| Artifact | Contents |
|---|---|
| `models/*_tuned.joblib` | each family's `best_estimator_`, refit on the full training set |
| `models/final_model.joblib` | the tuned leader — the single entry point for Phases 7–8 |
| `tuning_results.parquet` | per model: search size, best CV F2, tuned threshold, OOF metrics at that threshold and at 0.5, Phase 5 baseline |
| `threshold_sweep.parquet` | the full 4 × 91 threshold sweep, so Phase 7 can plot it without recomputing |
| `best_params.joblib` | winning hyperparameters for all four families |
| `tuning_selection.joblib` | leader name, slug, **chosen threshold**, criteria and the selection rule |

Phase 7 contract: load `models/final_model.joblib` and `tuning_selection.joblib['threshold']`, then classify the test set with `proba >= threshold`.