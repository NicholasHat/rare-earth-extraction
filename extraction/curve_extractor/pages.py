"""Render a region of a PDF page to a grayscale array, with whichever PDF
library the environment has.

The code-execution sandbox the extraction model works in has changed under
this pipeline: PyMuPDF worked and pdfplumber did not on 2026-09-25; on
2026-10-07 PyMuPDF was gone and pdfplumber and pypdfium2 worked. The prompt
and toolkit guide had named PyMuPDF, so every run spent its first turns
finding that out. Code that renders goes through `render`, which tries
pypdfium2, then PyMuPDF, then pdfplumber, so the instructions never name a
library again.

Coordinates are pdfplumber's — PDF points from the top-left of the media box —
the same space as the curve pre-pass's figure bboxes. The rendering libraries
measure from the crop box instead; on a PDF whose crop box trims the media box
(publisher margins: Draa 2004 is offset by (42, 351) pt) the difference is
added here, so a bbox from the pre-pass renders the region it names.
"""
from __future__ import annotations

import numpy as np


def render(pdf_path: str, page_index: int, bbox: tuple[float, float, float, float] | None = None,
           dpi: int = 300) -> np.ndarray:
    """`bbox` (x0, top, x1, bottom) of page `page_index` (0-based) as uint8
    grayscale, 0 = ink. None renders the whole page."""
    errors = []
    for backend in (_pdfium, _mupdf, _plumber):
        try:
            page_arr, origin = backend(pdf_path, page_index, dpi)
        except ImportError as e:
            errors.append(str(e))
            continue
        if bbox is None:
            return page_arr
        scale = dpi / 72.0
        ox, oy = origin
        x0, top, x1, bottom = (int(round((v - o) * scale)) for v, o in zip(bbox, (ox, oy, ox, oy)))
        h, w = page_arr.shape
        return page_arr[max(top, 0):min(bottom, h), max(x0, 0):min(x1, w)]
    raise ImportError("no PDF renderer available (pypdfium2, PyMuPDF, pdfplumber): " + "; ".join(errors))


def _pdfium(pdf_path, page_index, dpi):
    import pypdfium2 as pdfium
    page = pdfium.PdfDocument(pdf_path)[page_index]
    # Bottom-left-origin boxes; the render covers the crop box.
    media_l, _, _, media_t = page.get_mediabox()
    crop_l, _, _, crop_t = page.get_cropbox()
    arr = page.render(scale=dpi / 72.0, grayscale=True).to_numpy()
    return arr.reshape(arr.shape[0], arr.shape[1]), (crop_l - media_l, media_t - crop_t)


def _mupdf(pdf_path, page_index, dpi):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    page = fitz.open(pdf_path)[page_index]
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    return arr, (page.cropbox.x0, page.cropbox.y0)   # top-left-origin, relative to the media box


def _plumber(pdf_path, page_index, dpi):
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        img = page.to_image(resolution=dpi).original.convert("L")   # the crop box, like the others
        return np.asarray(img), crop_origin(page)


def crop_origin(page) -> tuple[float, float]:
    """Top-left of a pdfplumber page's crop box in its own coordinates — what
    a render of that page starts at."""
    box = getattr(page, "cropbox", None) or page.bbox
    return box[0], box[1]
