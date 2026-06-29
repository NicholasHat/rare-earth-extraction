"""Vector path — detect drawing objects from pdfplumber geometry (plan §3, §4.1).

Only used when a page actually has vector content. Markers are the *filled*
path objects (circles/diamonds/triangles drawn as one closed fill each) and the
assembled *stroked* glyphs (×/+/✶ built from line segments); the connecting
lines and the plot frame are stroked, unfilled, and long, so they're excluded by
the filled/short heuristics here and in markers.py.
"""
from __future__ import annotations

import pdfplumber


def open_page(pdf_bytes: bytes, page_index: int):
    pdf = pdfplumber.open(_BytesIO(pdf_bytes))
    return pdf, pdf.pages[page_index]


def _BytesIO(b: bytes):
    import io
    return io.BytesIO(b)


def page_is_vector(page) -> bool:
    """A figure page is 'vector' for our purposes if it carries filled path
    objects (markers). Pure-raster figure pages have images but no curves."""
    return any(c.get("fill") for c in page.curves)


def find_plot_frame(page) -> tuple[float, float, float, float] | None:
    """Largest rectangle on the page (the plot border). Falls back to the
    bounding box of all filled markers if no clear frame rect exists."""
    rects = [r for r in page.rects if r["width"] > 50 and r["height"] > 50]
    if rects:
        big = max(rects, key=lambda r: r["width"] * r["height"])
        return (big["x0"], big["top"], big["x1"], big["bottom"])
    filled = [c for c in page.curves if c.get("fill")]
    if not filled:
        return None
    return (
        min(c["x0"] for c in filled),
        min(c["top"] for c in filled),
        max(c["x1"] for c in filled),
        max(c["bottom"] for c in filled),
    )


def collect_filled_markers(page, *, legend_bbox=None) -> list[dict]:
    """All filled path objects, optionally excluding those inside a legend box."""
    out = []
    for c in page.curves:
        if not c.get("fill"):
            continue
        if legend_bbox and _inside(c, legend_bbox):
            continue
        out.append(c)
    return out


def collect_stroked_segments(page, frame, *, max_len_pt=12.0, legend_bbox=None) -> list[dict]:
    """Short line segments — candidate fragments of ×/+/✶ glyphs (monochrome path).

    Long lines (connecting curves, axis frame, gridlines) are excluded by length.
    """
    out = []
    for ln in page.lines:
        length = ((ln["x1"] - ln["x0"]) ** 2 + (ln["bottom"] - ln["top"]) ** 2) ** 0.5
        if length > max_len_pt:
            continue
        if legend_bbox and _inside(ln, legend_bbox):
            continue
        out.append(ln)
    return out


def _inside(obj, bbox) -> bool:
    x0, top, x1, bottom = bbox
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom
