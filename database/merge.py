"""Atomic merge of a reviewed extraction into the master DB (README §6, Phase A1).

`commit_extraction` ties the per-table writes (papers, prompt_runs, extractions,
text_endpoints, review_log) into one transaction so a partially-merged paper can
never exist. It is the single write path used by the review UI on 'approve'.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from . import extractions_repo, papers_repo, review_repo


def commit_extraction(
    conn: sqlite3.Connection,
    *,
    content_sha256: str,
    pdf_path: str,
    df: pd.DataFrame,
    text_endpoints: list[dict],
    prompt_version: str,
    prompt_sha256: str,
    model: str,
    qa_passed: bool,
    qa_report_json: str,
    raw_response: str | None,
    doi: str | None = None,
    reference_no: str | None = None,
    title: str | None = None,
    original_filename: str | None = None,
    figure_type: str | None = None,
    is_raster_figure: int | None = None,
    note: str | None = None,
    override: bool = False,
) -> dict:
    """Insert/locate the paper, record an approved run, and merge its rows.

    If a paper with the same hash/DOI already exists, it is reused (the new run
    is recorded against it — the coexistence path). Returns a summary dict with
    the new paper_id and prompt_run_id.

    The whole thing runs in one transaction; any error rolls everything back.
    """
    try:
        with conn:  # BEGIN ... COMMIT on success, ROLLBACK on exception
            existing = papers_repo.find_by_hash(conn, content_sha256) or (
                papers_repo.find_by_doi(conn, doi)
            )
            if existing is not None:
                paper_id = int(existing["paper_id"])
            else:
                paper_id = papers_repo.insert(
                    conn,
                    content_sha256=content_sha256,
                    pdf_path=pdf_path,
                    doi=doi,
                    reference_no=reference_no,
                    title=title,
                    original_filename=original_filename,
                    figure_type=figure_type,
                    is_raster_figure=is_raster_figure,
                )

            run_id = review_repo.insert_prompt_run(
                conn,
                paper_id=paper_id,
                prompt_version=prompt_version,
                prompt_sha256=prompt_sha256,
                model=model,
                n_rows_returned=len(df),
                qa_passed=qa_passed,
                qa_report_json=qa_report_json,
                raw_response=raw_response,
            )

            n_rows = extractions_repo.insert_extractions(conn, paper_id, run_id, df)
            extractions_repo.insert_text_endpoints(conn, paper_id, run_id, text_endpoints)

            review_repo.set_run_status(conn, run_id, "approved")
            log_note = note
            if override and not qa_passed:
                prefix = "[RED-flag override] "
                log_note = prefix + (note or "approved despite red QA flags")
            review_repo.log_review(
                conn,
                paper_id=paper_id,
                prompt_run_id=run_id,
                action="approve",
                note=log_note,
            )
        return {"paper_id": paper_id, "prompt_run_id": run_id, "rows_merged": n_rows}
    except sqlite3.IntegrityError as e:
        raise RuntimeError(f"merge failed (constraint): {e}") from e
