"""Tests for calculator.sanity.typical_ranges — reads v_current_best only."""
import pandas as pd

from calculator.sanity import typical_ranges
from database import merge
from validation import schema


def _sample_df(extractant="Cyanex 272", element="La", n=5):
    rows = [
        {
            schema.ELEMENT_COLUMN: element,
            "pH": 1.0 + 0.5 * i,
            "Extract%": 10.0 + 15 * i,
            "Extractant": extractant,
            "Extractant Conc. (mM)": 500.0,
        }
        for i in range(n)
    ]
    return schema.coerce_schema(pd.DataFrame(rows))


def _commit(conn, sha="hash1", doi="10.1/x", extractant="Cyanex 272", element="La"):
    return merge.commit_extraction(
        conn,
        content_sha256=sha,
        pdf_path=f"data/incoming/{sha}.pdf",
        df=_sample_df(extractant, element),
        text_endpoints=[],
        prompt_version="extraction_v5.2",
        prompt_sha256="psha",
        model="claude-opus-4-8",
        qa_passed=True,
        qa_report_json="[]",
        raw_response="{}",
        doi=doi,
    )


def test_typical_ranges_returns_none_when_no_data(conn):
    assert typical_ranges(conn, "Cyanex 272", "La") is None


def test_typical_ranges_summarizes_one_paper(conn):
    _commit(conn)
    summary = typical_ranges(conn, "Cyanex 272", "La")
    assert summary.n_papers == 1
    assert summary.n_rows == 5
    assert summary.pH_min == 1.0
    assert summary.pH_max == 3.0
    assert summary.extract_pct_min == 10.0
    assert summary.extract_pct_max == 70.0


def test_typical_ranges_counts_distinct_papers(conn):
    _commit(conn, sha="hashA", doi="10.1/a")
    _commit(conn, sha="hashB", doi="10.1/b")
    summary = typical_ranges(conn, "Cyanex 272", "La")
    assert summary.n_papers == 2
    assert summary.n_rows == 10


def test_typical_ranges_filters_by_extractant_and_element(conn):
    _commit(conn, sha="hashA", doi="10.1/a", extractant="Cyanex 272", element="La")
    _commit(conn, sha="hashB", doi="10.1/b", extractant="D2EHPA", element="La")
    summary = typical_ranges(conn, "Cyanex 272", "La")
    assert summary.n_papers == 1
    assert typical_ranges(conn, "D2EHPA", "La").n_papers == 1
    assert typical_ranges(conn, "Cyanex 272", "Lu") is None
