"""CRUD for the `papers` table — the dedup + provenance anchor."""
from __future__ import annotations

import sqlite3


def find_by_hash(conn: sqlite3.Connection, content_sha256: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM papers WHERE content_sha256 = ?", (content_sha256,)
    ).fetchone()


def find_by_doi(conn: sqlite3.Connection, doi: str | None) -> sqlite3.Row | None:
    if not doi:
        return None
    return conn.execute(
        "SELECT * FROM papers WHERE doi = ?", (doi.strip().lower(),)
    ).fetchone()


def get(conn: sqlite3.Connection, paper_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()


def insert(
    conn: sqlite3.Connection,
    *,
    content_sha256: str,
    pdf_path: str,
    doi: str | None = None,
    reference_no: str | None = None,
    title: str | None = None,
    original_filename: str | None = None,
    figure_type: str | None = None,
    is_raster_figure: int | None = None,
) -> int:
    """Insert a paper and return its paper_id. DOI is canonicalised to lowercase."""
    cur = conn.execute(
        """
        INSERT INTO papers (
            reference_no, doi, title, content_sha256, original_filename,
            pdf_path, figure_type, is_raster_figure
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            reference_no,
            doi.strip().lower() if doi else None,
            title,
            content_sha256,
            original_filename,
            pdf_path,
            figure_type,
            is_raster_figure,
        ),
    )
    return int(cur.lastrowid)
