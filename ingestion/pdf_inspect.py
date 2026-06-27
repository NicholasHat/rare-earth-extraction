"""Lightweight PDF triage: raster-vs-vector hint and a title guess (README §6).

Phase A1 only needs a coarse hint. `is_raster_figure` flags papers whose figures
are bitmap images (higher digitizing risk per README §9) vs. vector graphics.
The heuristic: a page carrying embedded image XObjects but little extractable
text is likely a scanned/raster figure page. This is a hint, not a guarantee —
the reviewer confirms.
"""
from __future__ import annotations

import io

from pypdf import PdfReader


def inspect(pdf_bytes: bytes) -> dict:
    """Return {'is_raster_figure': 0/1/None, 'title': str|None, 'n_pages': int}."""
    result: dict = {"is_raster_figure": None, "title": None, "n_pages": 0}
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return result

    result["n_pages"] = len(reader.pages)

    meta = reader.metadata or {}
    title = getattr(meta, "title", None)
    if isinstance(title, str) and title.strip():
        result["title"] = title.strip()

    pages_with_images = 0
    pages_with_text = 0
    for page in reader.pages:
        try:
            if page.images:  # embedded raster XObjects
                pages_with_images += 1
        except Exception:
            pass
        try:
            if (page.extract_text() or "").strip():
                pages_with_text += 1
        except Exception:
            pass

    if pages_with_images and pages_with_text == 0:
        result["is_raster_figure"] = 1
    elif pages_with_images:
        # Mixed: some raster content present; flag as possibly-raster (1) so the
        # reviewer applies extra scrutiny. Pure-text papers -> 0.
        result["is_raster_figure"] = 1 if pages_with_images >= pages_with_text else 0
    else:
        result["is_raster_figure"] = 0
    return result
