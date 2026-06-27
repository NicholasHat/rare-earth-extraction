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
