"""Parity test: the UI transform must reproduce the training matrices exactly.

Reproduce the deterministic split to recover the *raw* test rows, push each
through ``preprocess.build_feature_vector`` + ``scale``, and assert the result
equals ``notebooks/processed/X_test.parquet`` within float tolerance. Then
confirm ``predict.predict`` probabilities match a direct ``predict_proba`` on the
persisted scaled test matrix. If this passes, the UI scores employees with the
exact pipeline the model was trained and evaluated on.

Run:  py tests/test_parity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import predict, preprocess  # noqa: E402

RAW_FIELDS = [name for name, *_ in preprocess.FIELD_SPEC]


def _raw_test_rows():
    """Recover the raw (un-encoded) test split rows, indexed as in X_test.parquet."""
    df = pd.read_csv(preprocess.CSV_PATH).drop(columns=preprocess.DROP_COLS)
    y = (df["Attrition"] == "Yes").astype(int)
    X = df.drop(columns=["Attrition"])
    _, X_test_raw, _, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=preprocess.RANDOM_STATE
    )
    return X_test_raw


def main():
    X_test_raw = _raw_test_rows()
    X_test_expected = pd.read_parquet(preprocess.PROC / "X_test.parquet")
    cols = preprocess.feature_columns()

    # Sanity: same rows, same order.
    assert list(X_test_raw.index) == list(X_test_expected.index), "test index mismatch"

    # 1. Transform parity — every raw row -> identical scaled 56-vector.
    rebuilt = []
    for _, row in X_test_raw.iterrows():
        raw = {f: row[f] for f in RAW_FIELDS}
        rebuilt.append(preprocess.scale(preprocess.build_feature_vector(raw)).iloc[0])
    rebuilt = pd.DataFrame(rebuilt, index=X_test_raw.index)[cols]

    max_diff = float(np.abs(rebuilt.to_numpy() - X_test_expected[cols].to_numpy()).max())
    print(f"[1] transform parity: max abs diff over {len(rebuilt)} rows x {len(cols)} cols = {max_diff:.3e}")
    assert max_diff < 1e-9, f"transform diverges from training matrix (max diff {max_diff})"

    # 2. Prediction parity — UI predict() matches direct predict_proba on stored matrix.
    model = predict._model()
    direct = model.predict_proba(X_test_expected)[:, 1]
    ui = np.array([
        predict.predict({f: row[f] for f in RAW_FIELDS}, explain=False)["probability"]
        for _, row in X_test_raw.iterrows()
    ])
    pmax = float(np.abs(direct - ui).max())
    print(f"[2] probability parity: max abs diff = {pmax:.3e}")
    assert pmax < 1e-9, f"predict() diverges from predict_proba (max diff {pmax})"

    # 3. Threshold + directional sanity on two hand-built profiles.
    thr = predict.threshold()
    print(f"[3] threshold = {thr:.2f}")
    assert abs(thr - 0.49) < 1e-9, f"expected threshold 0.49, got {thr}"

    high = dict(zip(RAW_FIELDS, [X_test_raw.iloc[0][f] for f in RAW_FIELDS]))
    high.update({
        "OverTime": "Yes", "MaritalStatus": "Single", "MonthlyIncome": 2000,
        "JobLevel": 1, "YearsAtCompany": 1, "TotalWorkingYears": 1,
        "JobSatisfaction": 1, "EnvironmentSatisfaction": 1, "WorkLifeBalance": 1,
        "JobInvolvement": 1, "BusinessTravel": "Travel_Frequently", "Age": 24,
    })
    low = dict(high)
    low.update({
        "OverTime": "No", "MaritalStatus": "Married", "MonthlyIncome": 19000,
        "JobLevel": 5, "YearsAtCompany": 10, "TotalWorkingYears": 20,
        "JobSatisfaction": 4, "EnvironmentSatisfaction": 4, "WorkLifeBalance": 4,
        "JobInvolvement": 4, "BusinessTravel": "Non-Travel", "Age": 45,
    })
    p_high = predict.predict(high, explain=False)["probability"]
    p_low = predict.predict(low, explain=False)["probability"]
    print(f"[3] high-risk P(Leave)={p_high:.3f}  vs  low-risk P(Leave)={p_low:.3f}")
    assert p_high > p_low, "high-risk profile should score higher than low-risk"

    # 4. SHAP drivers produced without error.
    drivers = predict.predict(high, top_n=6)["drivers"]
    print(f"[4] top drivers for high-risk profile:")
    for d in drivers:
        print(f"      {d['feature']:24s} value={d['value']:>10.2f}  shap={d['shap']:+.3f}  ({d['direction']})")

    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
