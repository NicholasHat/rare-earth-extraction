"""Writes to `prompt_runs` (status + provenance) and `review_log` (audit trail)."""
from __future__ import annotations

import sqlite3


def insert_prompt_run(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    prompt_version: str,
    prompt_sha256: str,
    model: str,
    n_rows_returned: int,
    qa_passed: bool,
    qa_report_json: str,
    raw_response: str | None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> int:
    """Record an extraction attempt (status starts 'pending'). Returns prompt_run_id."""
    cur = conn.execute(
        """
        INSERT INTO prompt_runs (
            paper_id, prompt_version, prompt_sha256, model, status,
            n_rows_returned, qa_passed, qa_report_json, raw_response,
            input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens
        ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            prompt_version,
            prompt_sha256,
            model,
            n_rows_returned,
            1 if qa_passed else 0,
            qa_report_json,
            raw_response,
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
        ),
    )
    return int(cur.lastrowid)


def set_run_status(conn: sqlite3.Connection, prompt_run_id: int, status: str) -> None:
    """Set a run to 'approved' or 'rejected' and stamp reviewed_at."""
    if status not in ("approved", "rejected"):
        raise ValueError(f"invalid status: {status}")
    conn.execute(
        "UPDATE prompt_runs SET status = ?, reviewed_at = datetime('now') "
        "WHERE prompt_run_id = ?",
        (status, prompt_run_id),
    )


def log_review(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    prompt_run_id: int,
    action: str,
    note: str | None = None,
    edited_diff_json: str | None = None,
) -> None:
    """Append an approve/edit/reject record (no reviewer identity)."""
    if action not in ("approve", "edit", "reject"):
        raise ValueError(f"invalid action: {action}")
    conn.execute(
        """
        INSERT INTO review_log (paper_id, prompt_run_id, action, note, edited_diff_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (paper_id, prompt_run_id, action, note, edited_diff_json),
    )
