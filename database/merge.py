"""Atomic merge of a reviewed extraction into the master DB (plan §6).

`commit_extraction` ties the per-table writes (papers, prompt_runs, extractions,
text_endpoints, review_log) into one transaction so a partially-merged paper can
never exist. It is the single write path used by the review UI on 'approve'.

It is also the open-licence gate: a paper whose recorded licence is not open
(config.OPEN_LICENSES) — or that has none — is refused before anything is
written, and the first merge of each day leaves a backup of the DB first.

Given the table as extracted, it also records what the reviewer changed
(`edit_diff.table_diff`) on the approval's review_log row, in the same
transaction as the rows themselves.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

import config
from ingestion.licensing import License, is_open

from . import connection, edit_diff, extractions_repo, papers_repo, review_repo


class LicenseRequired(RuntimeError):
    """The paper has no recorded open licence, so it may not be merged."""


def commit_extraction(
    conn: sqlite3.Connection,
    *,
    content_sha256: str,
    pdf_path: str,
    df: pd.DataFrame,
    extracted: pd.DataFrame | None = None,
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
    tracking: dict[str, str | None] | None = None,
    note: str | None = None,
    override: bool = False,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    license: License | None = None,
) -> dict:
    """Insert/locate the paper, record an approved run, and merge its rows.

    If a paper with the same hash/DOI already exists, it is reused (the new run
    is recorded against it — the coexistence path). Approving a paper under a
    prompt version it was already approved under retires the earlier run rather
    than failing; its rows stay in the DB but stop being current. Returns a
    summary dict with the paper_id, prompt_run_id, and the superseded
    prompt_run_id (None on a first approval).

    `license` is the paper's licence (from the open-access manifest or set at
    review); it is recorded on the paper. Without one, the licence already on
    the paper's row is used. Either way it must be open, or LicenseRequired is
    raised and nothing is written.

    `df` is the table to merge; `extracted` is the same paper's table as the
    model produced it, before review. When given, the reviewer's edits between
    the two (rows deleted, rows added, cells changed — see `edit_diff`) are
    stored as the approval's `review_log.edited_diff_json` (NULL for no edits),
    and an edited table with no note is noted "edited in review".

    The whole thing runs in one transaction; any error rolls everything back.
    """
    existing = papers_repo.find_by_hash(conn, content_sha256) or papers_repo.find_by_doi(conn, doi)
    effective = license or papers_repo.get_license(existing)
    if effective is None or not is_open(effective):
        what = f"licence '{effective.id}'" if effective is not None else "no recorded licence"
        raise LicenseRequired(
            f"Not merged: this paper has {what}, and only openly-licensed papers "
            f"({', '.join(sorted(config.OPEN_LICENSES))}) may enter the database. Record its licence "
            "(the open-access manifest, or set it by hand at review) — nothing was written."
        )
    diff_json = edit_diff.to_json(edit_diff.table_diff(extracted, df)) if extracted is not None else None
    connection.backup_once_a_day(conn)
    try:
        with conn:  # BEGIN ... COMMIT on success, ROLLBACK on exception
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
            if license is not None:
                papers_repo.set_license(conn, paper_id, license)
            if tracking:
                # Same call for new and reused papers: on a re-approval the
                # reviewer's (non-empty) tracking edits update the paper row.
                papers_repo.update_tracking(conn, paper_id, **tracking)

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
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            )

            n_rows = extractions_repo.insert_extractions(conn, paper_id, run_id, df)
            extractions_repo.insert_text_endpoints(conn, paper_id, run_id, text_endpoints)

            # Only one approved run per (paper, prompt_version) is allowed, so
            # retire the incumbent before approving this one — otherwise
            # re-approving the same version raises a bare UNIQUE violation.
            superseded_run_id = review_repo.supersede_approved_run(
                conn,
                paper_id=paper_id,
                prompt_version=prompt_version,
                replaced_by=run_id,
            )
            review_repo.set_run_status(conn, run_id, "approved")
            log_note = note or ("edited in review" if diff_json else None)
            if override and not qa_passed:
                prefix = "[RED-flag override] "
                log_note = prefix + (log_note or "approved despite red QA flags")
            review_repo.log_review(
                conn,
                paper_id=paper_id,
                prompt_run_id=run_id,
                action="approve",
                note=log_note,
                edited_diff_json=diff_json,
            )
        return {
            "paper_id": paper_id,
            "prompt_run_id": run_id,
            "rows_merged": n_rows,
            "superseded_run_id": superseded_run_id,
        }
    except sqlite3.IntegrityError as e:
        raise RuntimeError(f"merge failed (constraint): {e}") from e

