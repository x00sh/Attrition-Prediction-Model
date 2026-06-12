"""Headless end-to-end smoke test of the Streamlit app via AppTest.

Runs app.py in-process, submits the input form, and asserts the prediction path
renders a verdict with no uncaught exceptions.

Run:  py tests/test_app_smoke.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console may be cp1252
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def main():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, f"initial render raised: {at.exception}"
    # Initial state: no submission yet, info prompt shown.
    assert any("Predict attrition risk" in b.label for b in at.button), "submit button missing"
    print(f"[1] initial render OK | {len(at.selectbox)} dropdowns, {len(at.number_input)} numeric inputs")

    # Submit the form with default values and re-run.
    at.button[0].click().run()
    assert not at.exception, f"prediction path raised: {at.exception}"

    verdict_texts = [e.value for e in at.success] + [e.value for e in at.error]
    assert any("LEAVE RISK" in t or "LIKELY TO STAY" in t for t in verdict_texts), \
        f"no verdict rendered; success/error blocks = {verdict_texts}"
    metrics = [(m.label, m.value) for m in at.metric]
    print(f"[2] after submit: verdict={verdict_texts} | metrics={metrics}")

    print("\nAPP SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
