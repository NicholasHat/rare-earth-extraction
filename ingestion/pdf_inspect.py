"""Lightweight PDF triage: raster-vs-vector hint and a title guess (README §6).

`is_raster_figure` flags papers whose figures are bitmap images (higher
digitizing risk per plan §9) vs. vector graphics — a coarse upload-time hint
persisted on the paper; the deterministic curve pre-pass makes the real
per-page vector/raster call at extraction time. This is a hint, not a
guarantee — the reviewer confirms.

The verdict is taken from the figure pages themselves, not from how much text
the paper has: a page is a *raster figure page* when it carries a sizeable
embedded image and (next to) no vector drawing, and a *vector figure page*
when it carries hundreds of drawing objects (plot markers, ticks, curves).
The paper is flagged raster when its raster figure pages outnumber its vector
ones. (An earlier heuristic compared image-bearing pages against text-bearing
pages; every page of a journal article has text, so it said "vector" for every
raster paper it met — Quinn 2015 and three MDPI/Sci Rep papers.)
"""
from __future__ import annotations

import io

import pdfplumber

# A raster figure occupies a real share of the page; logos and icons don't.
_MIN_IMAGE_PAGE_FRACTION = 0.05
# Below this many drawing objects a page has no plotted data in vector form
# (a frame, a few rules); a vector plot has hundreds (one per marker/tick).
_VECTOR_PAGE_MIN_CURVES = 100


def _page_kind(page) -> str | None:
    """'raster' | 'vector' | None for one page (see module docstring)."""
    page_area = float(page.width) * float(page.height)
    big_images = [
        im for im in page.images
        if (im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) >= _MIN_IMAGE_PAGE_FRACTION * page_area
    ]
    n_curves = len(page.curves)
    if n_curves >= _VECTOR_PAGE_MIN_CURVES:
        return "vector"
    if big_images:
        return "raster"
    return None


def inspect(pdf_bytes: bytes) -> dict:
    """Return {'is_raster_figure': 0/1/None, 'title': str|None, 'n_pages': int}."""
    result: dict = {"is_raster_figure": None, "title": None, "n_pages": 0}
    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception:
        return result
    with pdf:
        result["n_pages"] = len(pdf.pages)
        title = (pdf.metadata or {}).get("Title")
        if isinstance(title, str) and title.strip():
            result["title"] = title.strip()

        kinds = {"raster": 0, "vector": 0}
        for page in pdf.pages:
            try:
                kind = _page_kind(page)
            except Exception:
                continue
            if kind:
                kinds[kind] += 1

    if kinds["raster"] or kinds["vector"]:
        result["is_raster_figure"] = 1 if kinds["raster"] > kinds["vector"] else 0
    return result
