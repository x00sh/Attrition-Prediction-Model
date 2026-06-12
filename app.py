"""Streamlit UI for the employee attrition model.

HR enters an employee's details; the app reproduces the training-time transform
(see ``src/preprocess.py``), scores them with the deployed RandomForest, and
shows a Leave/Stay verdict plus the SHAP drivers behind it.

Run:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import streamlit as st

from src import predict, preprocess

st.set_page_config(page_title="Attrition Risk Predictor", page_icon="🧭", layout="wide")

META = preprocess.field_defaults()

RATE_FIELDS = ("HourlyRate", "DailyRate", "MonthlyRate")

DEPT_ROLES = {
    "Sales": ["Sales Representative", "Sales Executive", "Manager"],
    "Human Resources": ["Human Resources", "Manager"],
    "Research & Development": [
        "Research Scientist", "Research Director", "Manager",
        "Laboratory Technician", "Manufacturing Director", "Healthcare Representative",
    ],
}


def _infer_rates(monthly_income):
    """Map MonthlyIncome's percentile position onto each rate field's [min, max] range."""
    mi = META["MonthlyIncome"]
    pos = (monthly_income - mi["min"]) / max(mi["max"] - mi["min"], 1)
    return {
        n: int(round(META[n]["min"] + pos * (META[n]["max"] - META[n]["min"])))
        for n in RATE_FIELDS
    }


def _on_income_change():
    """When MonthlyIncome changes, update any rate fields the user hasn't overridden."""
    inferred = _infer_rates(st.session_state["MonthlyIncome"])
    for name, val in inferred.items():
        if not st.session_state.get(f"_rate_custom_{name}", False):
            st.session_state[name] = val


def _on_rate_change(name):
    """Mark a rate field as manually overridden so income changes don't reset it."""
    st.session_state[f"_rate_custom_{name}"] = True


def _on_dept_change():
    dept = st.session_state["Department"]
    valid = DEPT_ROLES[dept]
    if st.session_state.get("JobRole") not in valid:
        st.session_state["JobRole"] = valid[0]


def render_field(name):
    """Render the right widget for a raw field, returning its value."""
    m = META[name]
    if m["kind"] == "cat":
        opts = m["options"]
        return st.selectbox(m["label"], opts, index=opts.index(m["default"]), key=name)
    return st.number_input(
        m["label"], min_value=m["min"], max_value=m["max"],
        value=m["default"], step=1, key=name,
    )


def collect_inputs():
    """Render all fields reactively. Returns (raw_dict, was_submitted)."""
    groups = {g: [] for g in preprocess.GROUP_ORDER}
    for name, group, *_ in preprocess.FIELD_SPEC:
        groups[group].append(name)

    # Seed rate defaults in session_state before first render so the expander
    # opens with values inferred from the default MonthlyIncome, not from META defaults.
    if "MonthlyIncome" not in st.session_state:
        for name, val in _infer_rates(META["MonthlyIncome"]["default"]).items():
            if name not in st.session_state:
                st.session_state[name] = val

    raw = {}
    for group in preprocess.GROUP_ORDER:
        st.subheader(group)
        fields = groups[group]

        if group == "Compensation":
            mi_meta = META["MonthlyIncome"]
            raw["MonthlyIncome"] = st.number_input(
                mi_meta["label"],
                min_value=mi_meta["min"],
                max_value=mi_meta["max"],
                value=mi_meta["default"],
                step=1,
                key="MonthlyIncome",
                on_change=_on_income_change,
            )

            with st.expander(
                "Rate details — inferred from monthly income (expand to override)",
                expanded=False,
            ):
                st.caption(
                    "HourlyRate, DailyRate, and MonthlyRate are approximated from "
                    "monthly income. The model uses these values — open to set exact "
                    "figures if available."
                )
                rate_cols = st.columns(3)
                for i, name in enumerate(RATE_FIELDS):
                    m = META[name]
                    with rate_cols[i]:
                        raw[name] = st.number_input(
                            m["label"],
                            min_value=m["min"],
                            max_value=m["max"],
                            value=m["default"],
                            step=1,
                            key=name,
                            on_change=_on_rate_change,
                            args=(name,),
                        )

            other_comp = [n for n in fields if n not in RATE_FIELDS and n != "MonthlyIncome"]
            comp_cols = st.columns(3)
            for i, name in enumerate(other_comp):
                with comp_cols[i]:
                    raw[name] = render_field(name)

        else:
            cols = st.columns(3)
            for i, name in enumerate(fields):
                with cols[i % 3]:
                    if name == "Department":
                        m = META["Department"]
                        raw["Department"] = st.selectbox(
                            m["label"], m["options"],
                            index=m["options"].index(
                                st.session_state.get("Department", m["default"])
                            ),
                            key="Department",
                            on_change=_on_dept_change,
                        )
                    elif name == "JobRole":
                        dept = st.session_state.get("Department", META["Department"]["default"])
                        valid_roles = DEPT_ROLES[dept]
                        current = st.session_state.get("JobRole", valid_roles[0])
                        if current not in valid_roles:
                            current = valid_roles[0]
                        raw["JobRole"] = st.selectbox(
                            META["JobRole"]["label"],
                            valid_roles,
                            index=valid_roles.index(current),
                            key="JobRole",
                        )
                    else:
                        raw[name] = render_field(name)

    submitted = st.button("Predict attrition risk", type="primary")
    return raw, submitted


def show_verdict(res):
    p = res["probability"]
    thr = res["threshold"]
    leaving = res["prediction"] == 1

    left, right = st.columns([1, 1])
    with left:
        if leaving:
            st.error(f"### ⚠️ {res['label']}")
        else:
            st.success(f"### ✅ {res['label']}")
        st.metric("Predicted P(Leave)", f"{p:.1%}")
        st.caption(
            f"Flagged a leaver when P(Leave) ≥ {thr:.0%} — the recall-first "
            "threshold fixed in tuning (Phase 6)."
        )
    with right:
        st.progress(min(max(p, 0.0), 1.0))
        st.caption(f"Decision threshold: {thr:.0%}")


def show_drivers(res):
    """Horizontal SHAP bar — red pushes toward leaving, blue toward staying."""
    drivers = list(reversed(res["drivers"]))  # largest at top of chart
    names = [d["feature"] for d in drivers]
    vals = [d["shap"] for d in drivers]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(drivers) + 1))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to P(Leave)")
    ax.set_title("Top drivers for this employee")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("**What's pushing risk up — and what to do about it:**")
    for d in res["drivers"]:
        if d["shap"] > 0:
            st.markdown(
                f"- **{d['feature']}** (value {d['value']:.2f}) → {d['lever']}"
            )


def main():
    st.title("🧭 Employee Attrition Risk Predictor")
    st.markdown(
        "Enter an employee's details to estimate their likelihood of leaving. "
        "The model is a tuned RandomForest optimised for **recall** (catching "
        "likely leavers), evaluated on a held-out test set. Predictions are "
        "decision support, not a verdict on any individual."
    )

    raw, submitted = collect_inputs()
    if submitted:
        with st.spinner("Scoring employee…"):
            st.session_state["_result"] = predict.predict(raw, top_n=6)

    if "_result" in st.session_state:
        st.divider()
        show_verdict(st.session_state["_result"])
        st.divider()
        show_drivers(st.session_state["_result"])
    elif not submitted:
        st.info("Fill in the form above and click **Predict attrition risk**.")


if __name__ == "__main__":
    main()
