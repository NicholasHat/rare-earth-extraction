"""Accept an uploaded PDF and persist it under its content hash.

PDFs are kept (README §5, decision 2) so a paper can be re-extracted when the
prompt improves, and so the text-endpoint QA check has the source to verify
against. Naming the file by sha256 makes the on-disk store self-deduplicating.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import config


def content_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def save_pdf(pdf_bytes: bytes) -> tuple[str, Path]:
    """Persist the PDF to data/incoming/<sha256>.pdf. Returns (sha256, path).

    Idempotent: the same bytes always map to the same path, so re-uploading the
    same file overwrites identical content rather than duplicating it.
    """
    config.ensure_dirs()
    sha = content_hash(pdf_bytes)
    path = config.INCOMING_DIR / f"{sha}.pdf"
    if not path.exists():
        path.write_bytes(pdf_bytes)
    return sha, path
