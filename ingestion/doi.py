"""Best-effort DOI extraction from a PDF (metadata first, then first-page text).

A parsed DOI is dedup key #1; the content hash is the always-present fallback.
"""
from __future__ import annotations

import io
import re

from pypdf import PdfReader

# DOIs: 10.<registrant>/<suffix>. Suffix is permissive but stops at whitespace/quotes.
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)


def _clean(doi: str) -> str:
    # Trailing punctuation from running text is common; strip it.
    return doi.rstrip(".,;)]>").lower()


def parse_doi(pdf_bytes: bytes) -> str | None:
    """Return the first DOI found in metadata or the first two pages, else None."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return None

    # 1) Document metadata (some publishers set /doi or embed it in the subject).
    meta = reader.metadata or {}
    for value in meta.values():
        if isinstance(value, str):
            m = _DOI_RE.search(value)
            if m:
                return _clean(m.group(0))

    # 2) First couple of pages of text (DOI usually sits in the header/footer).
    for page in reader.pages[:2]:
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        m = _DOI_RE.search(text)
        if m:
            return _clean(m.group(0))
    return None
