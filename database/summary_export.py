"""Refresh the one-row-per-paper summary spreadsheet (v_paper_summary).

The review UI calls `refresh_summary_xlsx()` after every successful approval so
the file at `config.SUMMARY_XLSX_PATH` always mirrors the DB — pointing that
path into a cloud-synced folder (Google Drive for Desktop etc.) makes the cloud
copy auto-update too. Reads through a read-only connection like every other
consumer; the DB write path (merge.py) stays file-free.
"""
from __future__ import annotations

from pathlib import Path

import config
from database import browse, connection


def refresh_summary_xlsx(path: Path | None = None) -> Path:
    """Rewrite the paper-summary XLSX from v_paper_summary; return its path."""
    path = Path(path or config.SUMMARY_XLSX_PATH)
    conn = connection.get_readonly_conn()
    try:
        df = browse.paper_summary(conn)
    finally:
        conn.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, engine="openpyxl")
    return path
