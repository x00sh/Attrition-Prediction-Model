"""Raw employee inputs -> model-ready 56-feature vector.

This module reproduces, step for step, the transform built in
``notebooks/feature-engineering.ipynb`` so a single employee entered in the UI
becomes the *exact* scaled feature vector the deployed RandomForest was trained
on. Getting this wrong silently corrupts predictions, so the logic here is a
faithful copy of the notebook's `engineer()` / scaling cells.

Pipeline (same order as training): encode -> engineer 13 features -> scale 30
continuous columns. Two engineered features (`PeerRelativeIncome`, `IsLowIncome`)
depend on statistics computed on the *train split only*. Those constants were
never persisted as artifacts, so we recompute them here by reproducing the exact
deterministic split (``random_state=42``) from the raw CSV — guaranteed identical
train rows, no hardcoded lookup tables.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Paths (resolved relative to the repo root = parent of this file's dir) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROC = REPO_ROOT / "notebooks" / "processed"

RANDOM_STATE = 42
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

# The 13 engineered feature names (added by `engineer`); everything else in the
# 56-col matrix is a base encoded column present before engineering.
ENGINEERED = [
    "Single_OT", "FreqTravel_OT", "PeerRelativeIncome", "IsLowIncome",
    "IncomePerLevel", "EarlyTenure", "TenureRatio", "PromotionOverdue",
    "JobHoppingIndex", "IsHighRisk", "SatisfactionComposite", "EngagementScore",
    "IsLowEngagement",
]

# One-hot fields and the deterministic binary/ordinal maps (copied from the
# feature-engineering notebook, cell 6).
ONEHOT_FIELDS = ["Department", "EducationField", "JobRole", "MaritalStatus"]
TRAVEL_MAP = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}

# Categorical option lists (full level sets, used by the UI form and to build the
# correct dummy column names).
CATEGORY_OPTIONS = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": [
        "Human Resources", "Life Sciences", "Marketing", "Medical", "Other",
        "Technical Degree",
    ],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative",
    ],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}

# UI form layout: (raw column, group, human label, kind). kind is "cat" for a
# dropdown (options from CATEGORY_OPTIONS) or "int" for a numeric input whose
# range/default are derived from the dataset at load time.
FIELD_SPEC = [
    # Job
    ("Department", "Job", "Department", "cat"),
    ("JobRole", "Job", "Job role", "cat"),
    ("JobLevel", "Job", "Job level (1-5)", "int"),
    ("JobInvolvement", "Job", "Job involvement (1-4)", "int"),
    ("BusinessTravel", "Job", "Business travel", "cat"),
    ("OverTime", "Job", "Works overtime", "cat"),
    # Compensation
    ("MonthlyIncome", "Compensation", "Monthly income ($)", "int"),
    ("DailyRate", "Compensation", "Daily rate ($)", "int"),
    ("HourlyRate", "Compensation", "Hourly rate ($)", "int"),
    ("MonthlyRate", "Compensation", "Monthly rate ($)", "int"),
    ("PercentSalaryHike", "Compensation", "Percent salary hike (%)", "int"),
    ("StockOptionLevel", "Compensation", "Stock option level (0-3)", "int"),
    # Satisfaction
    ("JobSatisfaction", "Satisfaction", "Job satisfaction (1-4)", "int"),
    ("EnvironmentSatisfaction", "Satisfaction", "Environment satisfaction (1-4)", "int"),
    ("RelationshipSatisfaction", "Satisfaction", "Relationship satisfaction (1-4)", "int"),
    ("WorkLifeBalance", "Satisfaction", "Work-life balance (1-4)", "int"),
    ("PerformanceRating", "Satisfaction", "Performance rating (1-4)", "int"),
    # Tenure & career
    ("YearsAtCompany", "Tenure & career", "Years at company", "int"),
    ("YearsInCurrentRole", "Tenure & career", "Years in current role", "int"),
    ("YearsSinceLastPromotion", "Tenure & career", "Years since last promotion", "int"),
    ("YearsWithCurrManager", "Tenure & career", "Years with current manager", "int"),
    ("TotalWorkingYears", "Tenure & career", "Total working years", "int"),
    ("NumCompaniesWorked", "Tenure & career", "Number of companies worked", "int"),
    ("TrainingTimesLastYear", "Tenure & career", "Trainings last year", "int"),
    # Demographics
    ("Age", "Demographics", "Age", "int"),
    ("Gender", "Demographics", "Gender", "cat"),
    ("MaritalStatus", "Demographics", "Marital status", "cat"),
    ("Education", "Demographics", "Education (1=Below College ... 5=Doctor)", "int"),
    ("EducationField", "Demographics", "Education field", "cat"),
    ("DistanceFromHome", "Demographics", "Distance from home", "int"),
]

GROUP_ORDER = ["Job", "Compensation", "Satisfaction", "Tenure & career", "Demographics"]

# Dataset-derived ranges don't always cover the full valid domain for these fields.
FIELD_RANGE_OVERRIDES = {
    "PerformanceRating": {"min": 1, "max": 4},
    "PercentSalaryHike": {"min": 1, "max": 25},
}


@lru_cache(maxsize=1)
def _raw_df():
    """Cleaned raw frame (4 constant/ID columns dropped), target separated out."""
    df = pd.read_csv(CSV_PATH).drop(columns=DROP_COLS)
    return df


@lru_cache(maxsize=1)
def _train_constants():
    """Recompute the train-only stats by reproducing the exact stratified split.

    Returns (low_income_thresh, peer_median Series indexed by (JobRole, JobLevel),
    global_income_median). Row membership depends only on (n, y, random_state), so
    splitting the raw frame yields the identical train rows the notebook used.
    """
    df = _raw_df()
    y = (df["Attrition"] == "Yes").astype(int)
    X = df.drop(columns=["Attrition"])
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    low_income_thresh = X_train["MonthlyIncome"].quantile(0.33)
    peer_median = X_train.groupby(["JobRole", "JobLevel"])["MonthlyIncome"].median()
    global_income_median = X_train["MonthlyIncome"].median()
    return low_income_thresh, peer_median, global_income_median


@lru_cache(maxsize=1)
def feature_columns():
    """Canonical 56-column order, read from the persisted training matrix."""
    return list(pd.read_parquet(PROC / "X_train.parquet").columns)


@lru_cache(maxsize=1)
def _base_columns():
    """The post-encoding, pre-engineering columns (56 minus the 13 engineered)."""
    eng = set(ENGINEERED)
    return [c for c in feature_columns() if c not in eng]


@lru_cache(maxsize=1)
def _scaler():
    return joblib.load(PROC / "scaler.joblib")


@lru_cache(maxsize=1)
def scale_columns():
    return list(joblib.load(PROC / "scale_cols.joblib"))


@lru_cache(maxsize=1)
def field_defaults():
    """Per-field UI metadata: {name: {"kind", "label", "group", default, min, max, options}}.

    Numeric ranges/defaults come from the full dataset (form ergonomics only —
    they do not affect the transform). Categorical defaults use the column mode.
    """
    df = _raw_df()
    meta = {}
    for name, group, label, kind in FIELD_SPEC:
        entry = {"kind": kind, "label": label, "group": group}
        if kind == "cat":
            entry["options"] = CATEGORY_OPTIONS[name]
            entry["default"] = str(df[name].mode().iloc[0])
        else:
            col = df[name]
            entry["min"] = int(col.min())
            entry["max"] = int(col.max())
            entry["default"] = int(round(col.median()))
            if name in FIELD_RANGE_OVERRIDES:
                entry.update(FIELD_RANGE_OVERRIDES[name])
        meta[name] = entry
    return meta


def engineer(X):
    """Apply the 13 EDA-driven features — verbatim copy of the notebook's logic.

    Expects a frame already carrying the base encoded columns plus the `_JobRole`
    / `_MaritalStatus` helper columns. Uses train-fit constants from
    `_train_constants()`.
    """
    low_income_thresh, peer_median, global_income_median = _train_constants()
    X = X.copy()

    # OverTime interactions (strongest signal)
    X["Single_OT"] = (X["MaritalStatus_Single"] * X["OverTime"]).astype(int)
    X["FreqTravel_OT"] = ((X["BusinessTravel"] == 2).astype(int) * X["OverTime"]).astype(int)

    # Pay vs peers (below-peer pay = flight risk)
    peer = X.set_index(["_JobRole", "JobLevel"]).index.map(peer_median)
    peer = pd.Series(peer, index=X.index).fillna(global_income_median)
    X["PeerRelativeIncome"] = X["MonthlyIncome"] / peer
    X["IsLowIncome"] = (X["MonthlyIncome"] < low_income_thresh).astype(int)
    X["IncomePerLevel"] = X["MonthlyIncome"] / X["JobLevel"]  # JobLevel >= 1, safe

    # Tenure / career (early tenure + stalled promotion)
    twy = X["TotalWorkingYears"].replace(0, np.nan)  # divide-by-zero guard
    X["EarlyTenure"] = (X["YearsAtCompany"] <= 2).astype(int)
    X["TenureRatio"] = (X["YearsAtCompany"] / twy).fillna(0.0)
    X["PromotionOverdue"] = (X["YearsSinceLastPromotion"] >= 6).astype(int)
    X["JobHoppingIndex"] = (X["NumCompaniesWorked"] / twy).fillna(0.0)

    # Combined high-risk flag (OverTime + low income + early tenure)
    X["IsHighRisk"] = (
        (X["OverTime"] == 1) & (X["IsLowIncome"] == 1) & (X["EarlyTenure"] == 1)
    ).astype(int)

    # Engagement / well-being composites
    X["SatisfactionComposite"] = X[
        ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]
    ].mean(axis=1)
    X["EngagementScore"] = X["JobInvolvement"] + X["WorkLifeBalance"]
    X["IsLowEngagement"] = (
        (X["JobInvolvement"] <= 2) & (X["WorkLifeBalance"] <= 2)
    ).astype(int)

    return X.drop(columns=["_JobRole", "_MaritalStatus"])


def _encode_base(raw):
    """Build the post-encoding / pre-engineering single-row frame from raw inputs.

    Dummies are written directly against the canonical column names (single-row
    `pd.get_dummies` cannot infer the full level set), then reindexed so absent
    dummies — including each one-hot reference category — become 0.
    """
    raw = dict(raw)
    row = {}

    # Binary + ordinal (notebook cell 6)
    row["Gender"] = 1 if raw["Gender"] == "Male" else 0
    row["OverTime"] = 1 if raw["OverTime"] == "Yes" else 0
    row["BusinessTravel"] = TRAVEL_MAP[raw["BusinessTravel"]]

    # Numeric passthroughs: every base column that is neither encoded above nor a
    # one-hot dummy is a raw numeric carried straight through.
    dummy_prefixes = tuple(f"{f}_" for f in ONEHOT_FIELDS)
    for col in _base_columns():
        if col in ("Gender", "OverTime", "BusinessTravel") or col.startswith(dummy_prefixes):
            continue
        row[col] = raw[col]

    # One-hot: set the matching dummy to 1 (reference categories have no column).
    for field in ONEHOT_FIELDS:
        row[f"{field}_{raw[field]}"] = 1

    df = pd.DataFrame([row]).reindex(columns=_base_columns(), fill_value=0)
    # Helper columns the engineered features need (dropped inside `engineer`).
    df["_JobRole"] = raw["JobRole"]
    df["_MaritalStatus"] = raw["MaritalStatus"]
    return df


def build_feature_vector(raw):
    """Raw input dict -> unscaled 56-column DataFrame in canonical order."""
    df = engineer(_encode_base(raw))
    return df.reindex(columns=feature_columns(), fill_value=0)


def scale(df):
    """Apply the persisted StandardScaler to the 30 continuous columns only."""
    out = df.copy()
    cols = scale_columns()
    out[cols] = _scaler().transform(out[cols])
    return out


def to_model_input(raw):
    """Convenience: raw dict -> scaled, model-ready 56-column DataFrame."""
    return scale(build_feature_vector(raw))


def inverse_scale(df):
    """Map a scaled feature frame back to human-readable units (for display)."""
    out = df.copy()
    cols = scale_columns()
    out[cols] = _scaler().inverse_transform(out[cols])
    return out
