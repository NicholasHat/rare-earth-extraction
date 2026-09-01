"""Read-only browsing queries for the master-DB viewer page (README §5).

Pure functions over an existing connection — no writes — so they're testable
against the real schema fixture like the rest of the repo's data-access layer.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from . import naming


def list_papers(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per paper, with a count of approved runs and current-best rows."""
    return pd.read_sql_query(
        """
        SELECT
            p.paper_id, p.doi, p.title, p.original_filename, p.is_raster_figure,
            p.uploaded_at,
            (SELECT COUNT(*) FROM prompt_runs pr
             WHERE pr.paper_id = p.paper_id AND pr.status = 'approved') AS approved_runs,
            (SELECT COUNT(*) FROM v_current_best vb WHERE vb.paper_id = p.paper_id) AS current_best_rows
        FROM papers p
        ORDER BY p.paper_id
        """,
        conn,
    )


def paper_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per paper from v_paper_summary — the dedup sheet.

    Elements, per-element row counts, extractants, and pH range of each paper's
    current-best data; uploaded-but-unapproved papers appear with null data
    columns so they too can be checked before re-extracting.
    """
    return pd.read_sql_query("SELECT * FROM v_paper_summary", conn)


# Column order of the external "REE Paper Tracking" spreadsheet — tracking_sheet
# must produce exactly these, in this order, so its rows paste straight in.
TRACKING_SHEET_COLUMNS = [
    "Ref No.",
    "Status",
    "Date processed",
    "Short citation",
    "Year",
    "DOI / link",
    "Output file",
    "Figures / tables used",
    "Rows extracted",
    "No. of elements",
    "Extractant",
    "Extractant type",
    "Known issues / caveats",
    "Short description",
]


def tracking_sheet(conn: sqlite3.Connection) -> pd.DataFrame:
    """v_paper_summary reshaped into the external tracking spreadsheet's columns.

    `Output file` is computed with the same helper that names the actual export
    written on approval, so the sheet and data/exports/ can never disagree.
    """
    v = paper_summary(conn)
    if v.empty:
        return pd.DataFrame(columns=TRACKING_SHEET_COLUMNS)
    out = pd.DataFrame(
        {
            "Ref No.": v["reference_no"],
            "Status": v["status"],
            "Date processed": v["date_processed"],
            "Short citation": v["short_citation"],
            "Year": v["pub_year"],
            "DOI / link": v["doi"].map(naming.doi_link),
            "Output file": [
                naming.export_filename(r.short_citation, r.paper_id, int(r.data_rows))
                if pd.notna(r.data_rows)
                else None
                for r in v.itertuples()
            ],
            "Figures / tables used": v["figures_used"],
            "Rows extracted": v["data_rows"].astype("Int64"),
            "No. of elements": v["n_elements"].astype("Int64"),
            "Extractant": v["extractants"],
            "Extractant type": v["extractant_types"],
            "Known issues / caveats": v["known_issues"],
            "Short description": v["short_description"],
        }
    )
    return out


def list_prompt_runs(conn: sqlite3.Connection, paper_id: int | None = None) -> pd.DataFrame:
    """Extraction run history — every attempt (pending/approved/rejected), not just approved."""
    sql = """
        SELECT pr.prompt_run_id, pr.paper_id, p.doi, pr.prompt_version, pr.model,
               pr.status, pr.run_timestamp, pr.reviewed_at, pr.n_rows_returned, pr.qa_passed
        FROM prompt_runs pr
        JOIN papers p ON p.paper_id = pr.paper_id
    """
    params: tuple = ()
    if paper_id is not None:
        sql += " WHERE pr.paper_id = ?"
        params = (paper_id,)
    sql += " ORDER BY pr.prompt_run_id DESC"
    return pd.read_sql_query(sql, conn, params=params)


def list_review_log(conn: sqlite3.Connection, paper_id: int | None = None) -> pd.DataFrame:
    """The append-only approve/edit/reject audit trail (no reviewer identity, by design)."""
    sql = """
        SELECT rl.review_id, rl.paper_id, p.doi, rl.prompt_run_id, rl.action,
               rl.note, rl.created_at
        FROM review_log rl
        JOIN papers p ON p.paper_id = rl.paper_id
    """
    params: tuple = ()
    if paper_id is not None:
        sql += " WHERE rl.paper_id = ?"
        params = (paper_id,)
    sql += " ORDER BY rl.review_id DESC"
    return pd.read_sql_query(sql, conn, params=params)
