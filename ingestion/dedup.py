"""Deduplication check, run BEFORE extraction to avoid wasted API spend.

Checks the content hash (dedup key #2) and the parsed DOI (dedup key #1) against
the `papers` table. A match doesn't hard-block — it surfaces the existing paper
so the reviewer can choose to skip, or re-extract as a new run under a newer
prompt version (the coexistence path, README §6 Phase A3).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from database import papers_repo


@dataclass
class DuplicateMatch:
    paper_id: int
    reason: str          # 'content_hash' | 'doi'
    doi: str | None
    pdf_path: str | None


def find_existing(
    conn: sqlite3.Connection, content_sha256: str, doi: str | None
) -> DuplicateMatch | None:
    """Return a DuplicateMatch if this paper is already in the DB, else None."""
    row = papers_repo.find_by_hash(conn, content_sha256)
    if row is not None:
        return DuplicateMatch(int(row["paper_id"]), "content_hash", row["doi"], row["pdf_path"])

    row = papers_repo.find_by_doi(conn, doi)
    if row is not None:
        return DuplicateMatch(int(row["paper_id"]), "doi", row["doi"], row["pdf_path"])

    return None
