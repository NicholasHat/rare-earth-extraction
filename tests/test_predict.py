"""Tests for calculator.predict — experiment conditions → interpolated Extract%.

Reads v_current_best only (same seam as calculator.sanity); fixtures go through
the real merge path so the view resolution is exercised, not mocked.
"""
import pandas as pd
import pytest

from calculator.predict import _evaluate_series, predict_extract_pct
from database import merge
from validation import schema


def _sweep_df(element="La", conc=500.0, ppm=100.0, points=((1.0, 10.0), (2.0, 40.0), (3.0, 90.0))):
    rows = [
        {
            schema.ELEMENT_COLUMN: element,
            "pH": p,
            "Extract%": pct,
            "Extractant": "Cyanex 272",
            "Extractant Conc. (mM)": conc,
            "RRE composition (ppm)": ppm,
        }
        for p, pct in points
    ]
    return schema.coerce_schema(pd.DataFrame(rows))


def _commit(conn, df, sha="hash1", doi="10.1/x"):
    return merge.commit_extraction(
        conn,
        content_sha256=sha,
        pdf_path=f"data/incoming/{sha}.pdf",
        df=df,
        text_endpoints=[],
        prompt_version="extraction_v9",
        prompt_sha256="psha",
        model="claude-opus-4-8",
        qa_passed=True,
        qa_report_json="[]",
        raw_response="{}",
        doi=doi,
    )


def test_empty_db_yields_empty_report(conn):
    r = predict_extract_pct(conn, element="La", extractant="Cyanex 272", ph=2.0)
    assert r.matched == [] and r.off_condition == []
    assert r.best_estimate is None


def test_interpolates_between_bracketing_points(conn):
    _commit(conn, _sweep_df())
    r = predict_extract_pct(conn, element="La", extractant="Cyanex 272", ph=2.5)
    assert len(r.matched) == 1
    s = r.matched[0]
    assert s.predicted_extract_pct == pytest.approx(65.0)   # midway 40 → 90
    assert s.bracket == ((2.0, 40.0), (3.0, 90.0))
    assert r.best_estimate == pytest.approx(65.0)


def test_exact_measured_ph_returns_measured_value(conn):
    _commit(conn, _sweep_df())
    r = predict_extract_pct(conn, element="La", extractant="Cyanex 272", ph=2.0)
    assert r.matched[0].predicted_extract_pct == pytest.approx(40.0)


def test_ph_outside_sweep_is_reported_not_extrapolated(conn):
    _commit(conn, _sweep_df())
    r = predict_extract_pct(conn, element="La", extractant="Cyanex 272", ph=5.0)
    assert len(r.matched) == 1
    assert r.matched[0].predicted_extract_pct is None
    assert r.matched[0].bracket is None
    assert r.best_estimate is None
    assert (r.matched[0].ph_min, r.matched[0].ph_max) == (1.0, 3.0)


def test_condition_tolerance_splits_matched_from_off_condition(conn):
    _commit(conn, _sweep_df(conc=500.0), sha="hashA", doi="10.1/a")
    _commit(conn, _sweep_df(conc=100.0), sha="hashB", doi="10.1/b")
    r = predict_extract_pct(
        conn, element="La", extractant="Cyanex 272", ph=2.5, extractant_conc_mM=500.0
    )
    assert [s.extractant_conc_mM for s in r.matched] == [500.0]
    assert [s.extractant_conc_mM for s in r.off_condition] == [100.0]
    # No condition filter → everything matches.
    r_all = predict_extract_pct(conn, element="La", extractant="Cyanex 272", ph=2.5)
    assert len(r_all.matched) == 2 and r_all.off_condition == []


def test_short_series_are_not_sweeps(conn):
    # Two distinct pH points only — below _MIN_SWEEP_POINTS, can't interpolate.
    _commit(conn, _sweep_df(points=((1.0, 10.0), (2.0, 40.0))))
    r = predict_extract_pct(conn, element="La", extractant="Cyanex 272", ph=1.5)
    assert r.matched == [] and r.off_condition == []


def test_element_and_extractant_are_filtered(conn):
    _commit(conn, _sweep_df(element="La"))
    r = predict_extract_pct(conn, element="Lu", extractant="Cyanex 272", ph=2.0)
    assert r.matched == []
    r2 = predict_extract_pct(conn, element="La", extractant="D2EHPA", ph=2.0)
    assert r2.matched == []


def test_best_estimate_is_median_across_papers(conn):
    _commit(conn, _sweep_df(points=((1.0, 10.0), (2.0, 40.0), (3.0, 90.0))), sha="hashA", doi="10.1/a")
    _commit(conn, _sweep_df(points=((1.0, 20.0), (2.0, 60.0), (3.0, 95.0))), sha="hashB", doi="10.1/b")
    r = predict_extract_pct(conn, element="La", extractant="Cyanex 272", ph=2.0)
    assert r.best_estimate == pytest.approx(50.0)   # median of 40 and 60


def test_evaluate_series_averages_duplicate_ph_readings():
    # e.g. a pH-sweep point and a conc-sweep row measured at the same pH.
    pred, bracket = _evaluate_series([(1.0, 10.0), (2.0, 40.0), (2.0, 42.0), (3.0, 90.0)], 2.0)
    assert pred == pytest.approx(41.0)
    assert bracket == ((2.0, 41.0), (2.0, 41.0))
