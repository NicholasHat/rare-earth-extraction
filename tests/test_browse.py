"""Tests for database.browse — the read-only queries behind the DB viewer page."""
import pandas as pd

from database import browse, merge
from validation import schema


def _sample_df(element="La", n=3):
    rows = [
        {schema.ELEMENT_COLUMN: element, "pH": 1.0 + i, "Extract%": 10.0 + 10 * i}
        for i in range(n)
    ]
    return schema.coerce_schema(pd.DataFrame(rows))


def _commit(conn, sha="hash1", doi="10.1/x", prompt_version="extraction_v5.1", note=None):
    return merge.commit_extraction(
        conn,
        content_sha256=sha,
        pdf_path=f"data/incoming/{sha}.pdf",
        df=_sample_df(),
        text_endpoints=[],
        prompt_version=prompt_version,
        prompt_sha256="psha",
        model="claude-opus-4-8",
        qa_passed=True,
        qa_report_json="[]",
        raw_response="{}",
        doi=doi,
        title="A Test Paper",
        note=note,
    )


def test_list_papers_empty(conn):
    assert browse.list_papers(conn).empty


def test_list_papers_includes_counts(conn):
    _commit(conn)
    df = browse.list_papers(conn)
    assert len(df) == 1
    assert df.iloc[0]["approved_runs"] == 1
    assert df.iloc[0]["current_best_rows"] == 3
    assert df.iloc[0]["title"] == "A Test Paper"


def test_list_prompt_runs_returns_history_newest_first(conn):
    _commit(conn, sha="a", doi="10.1/a", prompt_version="extraction_v5")
    _commit(conn, sha="a", doi="10.1/a", prompt_version="extraction_v5.1")
    df = browse.list_prompt_runs(conn)
    assert len(df) == 2
    # newest (highest prompt_run_id) first
    assert df.iloc[0]["prompt_run_id"] > df.iloc[1]["prompt_run_id"]
    assert set(df["prompt_version"]) == {"extraction_v5", "extraction_v5.1"}


def test_list_prompt_runs_filters_by_paper(conn):
    s1 = _commit(conn, sha="a", doi="10.1/a")
    _commit(conn, sha="b", doi="10.1/b")
    df = browse.list_prompt_runs(conn, paper_id=s1["paper_id"])
    assert len(df) == 1
    assert df.iloc[0]["paper_id"] == s1["paper_id"]


def test_list_review_log_records_approve_action(conn):
    _commit(conn, note="looks good")
    df = browse.list_review_log(conn)
    assert len(df) == 1
    assert df.iloc[0]["action"] == "approve"
    assert df.iloc[0]["note"] == "looks good"


def test_list_review_log_filters_by_paper(conn):
    s1 = _commit(conn, sha="a", doi="10.1/a")
    _commit(conn, sha="b", doi="10.1/b")
    df = browse.list_review_log(conn, paper_id=s1["paper_id"])
    assert len(df) == 1
