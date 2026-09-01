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


def test_paper_summary_empty(conn):
    assert browse.paper_summary(conn).empty


def test_paper_summary_aggregates_elements_extractants_and_ph(conn):
    rows = [
        {schema.ELEMENT_COLUMN: "La", "pH": 1.5, "Extractant": "D2EHPA", "Extractant type": "acidic"},
        {schema.ELEMENT_COLUMN: "La", "pH": 3.0, "Extractant": "D2EHPA", "Extractant type": "acidic"},
        {schema.ELEMENT_COLUMN: "Ce", "pH": 2.0, "Extractant": "Cyanex 572", "Extractant type": "acidic"},
    ]
    merge.commit_extraction(
        conn,
        content_sha256="hash1",
        pdf_path="data/incoming/hash1.pdf",
        df=schema.coerce_schema(pd.DataFrame(rows)),
        text_endpoints=[],
        prompt_version="extraction_v9",
        prompt_sha256="psha",
        model="claude-sonnet-5",
        qa_passed=True,
        qa_report_json="[]",
        raw_response="{}",
        doi="10.1/x",
        title="A Test Paper",
    )
    df = browse.paper_summary(conn)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["elements"] == "Ce, La"
    assert row["rows_per_element"] == "Ce (1), La (2)"
    assert row["data_rows"] == 3
    assert row["extractants"] == "Cyanex 572, D2EHPA"
    assert row["extractant_types"] == "acidic"
    assert row["ph_min"] == 1.5
    assert row["ph_max"] == 3.0
    assert row["prompt_version"] == "extraction_v9"


def test_paper_summary_reflects_the_current_best_run_only(conn):
    _commit(conn, prompt_version="extraction_v8")
    _commit(conn, prompt_version="extraction_v9")  # supersedes v8 for this paper
    df = browse.paper_summary(conn)
    assert len(df) == 1
    assert df.iloc[0]["prompt_version"] == "extraction_v9"
    assert df.iloc[0]["data_rows"] == 3  # v8's rows are not double-counted


def test_paper_summary_includes_unapproved_papers(conn):
    from database import papers_repo

    papers_repo.insert(conn, content_sha256="orphan", pdf_path="data/incoming/orphan.pdf",
                       doi="10.1/orphan", title="Uploaded, not approved")
    df = browse.paper_summary(conn)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Uploaded, not approved"
    assert pd.isna(df.iloc[0]["data_rows"])


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
