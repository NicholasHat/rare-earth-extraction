"""database.merge: an approved extraction round-trips into v_current_best."""
import pandas as pd

from database import extractions_repo, merge
from validation import schema


def _sample_df():
    rows = [
        {schema.ELEMENT_COLUMN: "La", "pH": 1.0 + 0.3 * i, "Extract%": 10.0 + 7 * i}
        for i in range(10)
    ]
    return schema.coerce_schema(pd.DataFrame(rows))


def _commit(conn, sha="hash1", doi="10.1/x", **kw):
    return merge.commit_extraction(
        conn,
        content_sha256=sha,
        pdf_path=f"data/incoming/{sha}.pdf",
        df=_sample_df(),
        text_endpoints=[{"element": "La", "x_value": 3.0, "x_basis": "pH",
                         "y_value": 60.0, "y_metric": "Extract%", "source_quote": "q"}],
        prompt_version=kw.get("prompt_version", "extraction_v5.1"),
        prompt_sha256="psha",
        model="claude-opus-4-8",
        qa_passed=True,
        qa_report_json="[]",
        raw_response="{}",
        doi=doi,
    )


def test_commit_populates_current_best(conn):
    summary = _commit(conn)
    assert summary["rows_merged"] == 10
    best = extractions_repo.current_best(conn)
    assert len(best) == 10
    # text endpoint recorded too
    n_ep = conn.execute("SELECT COUNT(*) FROM text_endpoints").fetchone()[0]
    assert n_ep == 1
    # an approved run exists, with no reviewer identity stored
    run = conn.execute("SELECT * FROM prompt_runs").fetchone()
    assert run["status"] == "approved"
    assert run["reviewed_at"] is not None


def test_reupload_same_paper_reuses_paper_row(conn):
    s1 = _commit(conn, sha="hashA", doi="10.1/a", prompt_version="extraction_v5")
    # New prompt version, same paper (same hash) -> coexistence, not a new paper.
    s2 = _commit(conn, sha="hashA", doi="10.1/a", prompt_version="extraction_v5.1")
    assert s1["paper_id"] == s2["paper_id"]
    assert s1["prompt_run_id"] != s2["prompt_run_id"]
    n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert n_papers == 1
    # current-best returns ONE version's rows (the latest approved), not both.
    best = extractions_repo.current_best(conn)
    assert len(best) == 10
    assert best["prompt_run_id"].nunique() == 1


def test_current_best_survives_two_digit_version_numbers(conn):
    """v10 must beat v9. Sorting prompt_version as TEXT gets this backwards."""
    _commit(conn, sha="hashA", doi="10.1/a", prompt_version="extraction_v9")
    newer = _commit(conn, sha="hashA", doi="10.1/a", prompt_version="extraction_v10")
    best = extractions_repo.current_best(conn)
    assert best["prompt_run_id"].unique().tolist() == [newer["prompt_run_id"]]


def test_current_best_follows_approval_order_so_a_version_can_be_rolled_back(conn):
    """Re-approving an older version makes it current again — the rollback path."""
    older = _commit(conn, sha="hashA", doi="10.1/a", prompt_version="extraction_v5")
    _commit(conn, sha="hashA", doi="10.1/a", prompt_version="extraction_v9")
    conn.execute(
        "UPDATE prompt_runs SET reviewed_at = datetime('now', '+1 hour') "
        "WHERE prompt_run_id = ?",
        (older["prompt_run_id"],),
    )
    best = extractions_repo.current_best(conn)
    assert best["prompt_run_id"].unique().tolist() == [older["prompt_run_id"]]
