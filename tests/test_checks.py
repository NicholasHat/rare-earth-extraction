"""validation.checks: each known failure mode trips the right flag (README §9)."""
import pandas as pd

from validation import checks, schema
from validation.report import Severity

EL = schema.ELEMENT_COLUMN


def _df(rows: list[dict]) -> pd.DataFrame:
    return schema.coerce_schema(pd.DataFrame(rows))


def _good_curve(element="La", n=12, start=10.0, step=7.0):
    """A clean, monotonic, in-bounds %E-vs-pH series with >= 8 points."""
    return [
        {EL: element, "pH": 1.0 + 0.3 * i, "Extract%": min(99.0, start + step * i)}
        for i in range(n)
    ]


def test_clean_extraction_passes_green():
    report = checks.run(_df(_good_curve()), [], figure_is_curve=True)
    assert report.verdict is Severity.GREEN
    assert report.passed


def test_sparse_result_is_red():
    rows = [{EL: "Yb", "pH": 1.0, "Extract%": 20.0}, {EL: "Yb", "pH": 5.0, "Extract%": 90.0}]
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert not report.passed
    assert any(f.check == "row_count_sanity" for f in report.reds)


def test_sparse_result_not_flagged_when_not_a_curve():
    rows = [{EL: "Yb", "pH": 1.0, "Extract%": 20.0}, {EL: "Yb", "pH": 5.0, "Extract%": 90.0}]
    report = checks.run(_df(rows), [], figure_is_curve=False)
    assert not any(f.check == "row_count_sanity" for f in report.flags)


def test_axis_out_of_bounds_is_red():
    rows = _good_curve()
    rows[0]["Extract%"] = 150.0  # impossible %E
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert any(f.check == "axis_bounds" for f in report.reds)


def test_garbled_numeric_is_red():
    report = checks.run(_df(_good_curve()), [], figure_is_curve=True, coercion_failures=3)
    assert any(f.check == "schema_conformance" for f in report.reds)


def test_text_endpoint_mismatch_is_red():
    # Curve only reaches ~50% at pH 3, but the paper claims 95% there.
    rows = [
        {EL: "Nd", "pH": 1.0 + 0.25 * i, "Extract%": min(55.0, 10.0 + 4.5 * i)}
        for i in range(12)
    ]
    endpoints = [
        {"element": "Nd", "x_value": 3.0, "x_basis": "pH", "y_value": 95.0, "y_metric": "Extract%"}
    ]
    report = checks.run(_df(rows), endpoints, figure_is_curve=True)
    assert any(f.check == "text_endpoint_cross_check" for f in report.reds)


def test_text_endpoint_match_passes():
    rows = [
        {EL: "Nd", "pH": 1.0 + 0.25 * i, "Extract%": min(96.0, 10.0 + 8.0 * i)}
        for i in range(12)
    ]
    # At pH ~3.0 (i=8) Extract% is ~74; claim 74 should match within tolerance.
    endpoints = [
        {"element": "Nd", "x_value": 3.0, "x_basis": "pH", "y_value": 74.0, "y_metric": "Extract%"}
    ]
    report = checks.run(_df(rows), endpoints, figure_is_curve=True)
    assert not any(f.check == "text_endpoint_cross_check" for f in report.reds)


def test_vocabulary_drift_is_amber_not_blocking():
    rows = _good_curve()
    for r in rows:
        r["Extractant type"] = "totally-new-type"
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert report.passed  # amber doesn't block
    assert any(f.check == "vocabulary" and f.severity is Severity.AMBER for f in report.flags)


def test_empty_extraction_is_red():
    report = checks.run(_df([]), [], figure_is_curve=True)
    assert not report.passed
    assert any(f.check == "schema_conformance" for f in report.reds)


def _two_experiment_rows(element="Nd"):
    """A pH-sweep experiment (Extractant Conc. fixed at 500) plus a second,
    separate concentration-sweep experiment for the same element (pH fixed
    at 1.75) — the OUTPUT CONTRACT combines every experiment into one flat
    `rows` list, so both legitimately share the element column."""
    ph_sweep = [
        {EL: element, "pH": 0.87 + 0.15 * i, "Extract%": min(99.0, 2.0 + 6.0 * i),
         "Extractant Conc. (mM)": 500.0}
        for i in range(14)
    ]
    conc_sweep = [
        {EL: element, "pH": 1.75, "Extract%": pct, "Extractant Conc. (mM)": conc}
        for conc, pct in [(50, 0.24), (100, 25.66), (250, 7.38), (500, 14.1), (1000, 26.92)]
    ]
    return ph_sweep + conc_sweep


def test_monotonicity_does_not_pool_a_second_experiment():
    # The conc-sweep block alone (all at pH=1.75, %E jumping 0.24->26.92) would
    # look wildly non-monotonic if pooled with the pH-sweep by element alone.
    report = checks.run(_df(_two_experiment_rows()), [], figure_is_curve=True)
    assert not any(f.check == "monotonicity" for f in report.flags)


def test_text_endpoint_cross_check_picks_matching_experiment_not_nearest_tie():
    # Five conc-sweep rows all share pH=1.75 exactly (a tie on x-distance).
    # The paper's stated point (Extract% 14.02 at pH 1.75) matches the
    # Extractant Conc.=500 row (14.1) — not the Conc.=50 row (0.24), which an
    # arbitrary "first nearest x" tie-break would previously have picked.
    endpoints = [
        {"element": "Nd", "x_value": 1.75, "x_basis": "pH", "y_value": 14.02, "y_metric": "Extract%"}
    ]
    report = checks.run(_df(_two_experiment_rows()), endpoints, figure_is_curve=True)
    assert not any(f.check == "text_endpoint_cross_check" for f in report.reds)


def test_text_endpoint_cross_check_still_flags_a_real_mismatch():
    # None of the candidate rows at pH=1.75 comes anywhere near a claimed 60%.
    endpoints = [
        {"element": "Nd", "x_value": 1.75, "x_basis": "pH", "y_value": 60.0, "y_metric": "Extract%"}
    ]
    report = checks.run(_df(_two_experiment_rows()), endpoints, figure_is_curve=True)
    assert any(f.check == "text_endpoint_cross_check" for f in report.reds)
