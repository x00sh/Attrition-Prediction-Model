"""Load the deployed model and turn a raw employee record into a verdict.

The deployed estimator (``notebooks/processed/models/final_model.joblib``) is a
bare RandomForestClassifier scoring the scaled 56-feature vector. An employee is
flagged a leaver when ``P(Leave) >= threshold`` (0.49, fixed on OOF data in
Phase 6). Per-prediction drivers are produced with the same SHAP TreeExplainer
pattern used in ``notebooks/interpretation.ipynb``.
"""

from functools import lru_cache

import joblib
import numpy as np

from . import preprocess

PROC = preprocess.PROC

# Driver -> retention lever, copied from notebooks/interpretation.ipynb (Cell 10)
# so the UI can suggest an action for each risk-increasing factor.
LEVER = {
    "OverTime": "Cap sustained overtime; redistribute load on over-extended roles.",
    "MonthlyIncome": "Review pay for below-band staff in high-attrition roles.",
    "IncomePerLevel": "Audit pay-vs-job-level compression; fix under-levelled high performers.",
    "PeerRelativeIncome": "Benchmark pay against role+level peers; close large negative gaps.",
    "IsLowIncome": "Prioritise the lowest-income tercile for retention review.",
    "Age": "Tailor offers: career pathing for younger staff, flexibility for older.",
    "TotalWorkingYears": "Watch early-career, low-experience joiners most closely.",
    "YearsAtCompany": "Strengthen first-two-year onboarding and check-ins.",
    "EarlyTenure": "Front-load mentoring during the first two years.",
    "YearsInCurrentRole": "Offer lateral moves / growth before stagnation sets in.",
    "YearsWithCurrManager": "Provide manager-relationship and transition support.",
    "YearsSinceLastPromotion": "Set clear promotion timelines; flag overdue cases.",
    "PromotionOverdue": "Review staff 6+ years without a promotion.",
    "StockOptionLevel": "Extend equity to retain at-risk mid-level staff.",
    "JobSatisfaction": "Act on low job-satisfaction survey scores.",
    "EnvironmentSatisfaction": "Improve team/environment for low scorers.",
    "JobInvolvement": "Increase ownership and meaningful work.",
    "WorkLifeBalance": "Protect work-life balance, especially under overtime.",
    "SatisfactionComposite": "Target the lowest composite-satisfaction segment.",
    "EngagementScore": "Boost involvement and balance jointly.",
    "IsLowEngagement": "Intervene on the low-involvement / low-balance group.",
    "DistanceFromHome": "Offer remote/hybrid or relocation support for long commutes.",
    "Single_OT": "Single employees on overtime are highest-risk - protect their balance.",
    "FreqTravel_OT": "Frequent travel plus overtime compounds risk; rebalance both.",
    "IsHighRisk": "Composite high-risk flag - prioritise for proactive outreach.",
    "MaritalStatus_Single": "Single staff churn more; tailor engagement.",
    "NumCompaniesWorked": "Prior job-hopping signals flight risk; invest early.",
    "JobHoppingIndex": "High job-hop rate - secure with growth and recognition.",
    "BusinessTravel": "Reduce travel burden where it predicts exits.",
}


@lru_cache(maxsize=1)
def _model():
    return joblib.load(PROC / "models" / "final_model.joblib")


@lru_cache(maxsize=1)
def threshold():
    """Deployment threshold fixed on OOF data in Phase 6 (0.49)."""
    return float(joblib.load(PROC / "final_evaluation.joblib")["threshold"])


@lru_cache(maxsize=1)
def _explainer():
    import shap  # imported lazily — only needed when drivers are requested

    return shap.TreeExplainer(_model())


def _normalize_shap(raw):
    """Positive-class SHAP values as a 2-D array, across shap versions.

    (Copied from notebooks/interpretation.ipynb Cell 3.)
    """
    if isinstance(raw, list):  # old API: [class0, class1]
        return np.asarray(raw[1])
    arr = np.asarray(raw)
    if arr.ndim == 3:  # new API: (n, features, n_classes)
        return arr[:, :, 1]
    return arr


def _normalize_base(raw):
    """Scalar positive-class base value, across shap versions."""
    if isinstance(raw, (list, np.ndarray)):
        vals = np.ravel(np.asarray(raw, dtype=float))
        return float(vals[1]) if vals.size > 1 else float(vals[0])
    return float(raw)


def predict(raw, top_n=6, explain=True):
    """Score one raw employee record.

    Returns a dict with:
      probability  - P(Leave) in [0, 1]
      prediction   - 1 (leave risk) / 0 (likely to stay)
      label        - human-readable verdict
      threshold    - the decision threshold applied
      base_value   - mean P(Leave) baseline (only if explain)
      drivers      - top_n features by |SHAP|, each:
                     {feature, value, shap, direction, lever}
    """
    X = preprocess.build_feature_vector(raw)
    X_scaled = preprocess.scale(X)

    proba = float(_model().predict_proba(X_scaled)[:, 1][0])
    thr = threshold()
    pred = int(proba >= thr)

    result = {
        "probability": proba,
        "prediction": pred,
        "label": "LEAVE RISK" if pred else "LIKELY TO STAY",
        "threshold": thr,
    }
    if not explain:
        return result

    sv = _normalize_shap(_explainer().shap_values(X_scaled))[0]  # (56,)
    base = _normalize_base(_explainer().expected_value)
    display_vals = preprocess.inverse_scale(X_scaled).iloc[0]  # human-readable units
    features = preprocess.feature_columns()

    order = np.argsort(np.abs(sv))[::-1][:top_n]
    drivers = []
    for i in order:
        feat = features[i]
        contrib = float(sv[i])
        drivers.append({
            "feature": feat,
            "value": float(display_vals[feat]),
            "shap": contrib,
            "direction": "increases risk" if contrib > 0 else "reduces risk",
            "lever": LEVER.get(feat, "Review in context."),
        })

    result["base_value"] = base
    result["drivers"] = drivers
    return result
