"""Raster path — classical CV marker detection on a rendered figure image
(plan §6/§8: the fallback for papers whose figures are embedded images, e.g.
Quinn et al. 2015 — 200 DPI grayscale, monochrome, marker-shape-coded).

Pipeline: render the figure region at high DPI → threshold → suppress the thin
connecting/axis lines with a morphological opening (so markers sitting ON a line
don't merge into it) → connected-component blobs → size-filter to marker scale →
classify shape (filled ■●▲ vs stroked ×/+/✶) → return MarkerRecords in pixel
coords. Grouping by element (shape→element) and calibration are left to the
caller/LLM, same seam as the vector path.

This is a best-effort detector: monochrome shape coding at ~15px with crowded
panels is the hardest figure class, so callers should treat its counts as
review-gated, not ground truth (the warnings surface low-confidence groups).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .types import MarkerRecord

_RENDER_DPI = 300
_DARK_THRESHOLD = 128          # 8-bit grayscale; below = ink
_MIN_BLOB_PX = 12              # smaller = text speckle / noise
_MAX_BLOB_PX = 600             # larger = merged line/frame, not a single marker
_OPEN_ITERS = 1                # erosion iterations to break thin lines off markers


def render_region(page, bbox, dpi: int = _RENDER_DPI) -> np.ndarray:
    """Render `bbox` (pdf points) of `page` to a grayscale numpy array."""
    pil = page.to_image(resolution=dpi).original.convert("L")
    sc = dpi / 72.0
    x0, top, x1, bottom = bbox
    crop = pil.crop((int(x0 * sc), int(top * sc), int(x1 * sc), int(bottom * sc)))
    return np.asarray(crop)


def _ink_mask(arr: np.ndarray) -> np.ndarray:
    return arr < _DARK_THRESHOLD


def detect_blobs(arr: np.ndarray) -> list[dict]:
    """Find marker-scale blobs after suppressing thin lines.

    Returns dicts: {cx, cy, area, bbox_w, bbox_h, fill_ratio, solidity_mask}.
    """
    ink = _ink_mask(arr)
    # Morphological opening removes structures thinner than the marker core
    # (connecting curves, axis frame, gridlines) while keeping marker bodies.
    opened = ndimage.binary_opening(ink, iterations=_OPEN_ITERS)
    lbl, n = ndimage.label(opened)
    if n == 0:
        return []
    objs = ndimage.find_objects(lbl)
    blobs = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        sub = lbl[ys, xs] == i
        area = int(sub.sum())
        if not (_MIN_BLOB_PX <= area <= _MAX_BLOB_PX):
            continue
        h, w = sub.shape
        cy = (ys.start + ys.stop - 1) / 2.0
        cx = (xs.start + xs.stop - 1) / 2.0
        fill_ratio = area / (w * h) if w * h else 0.0
        blobs.append({
            "cx": float(cx), "cy": float(cy), "area": area,
            "bbox_w": w, "bbox_h": h, "fill_ratio": fill_ratio,
        })
    return blobs


def classify_blob_shape(blob: dict) -> tuple[str, str]:
    """Coarse (marker_type, shape) from blob geometry.

    Filled shapes (■●▲) have a high fill ratio; stroked glyphs (×/+/✶) are thin
    strokes with a low fill ratio. Finer shape ID at ~15px is unreliable, so we
    distinguish at the type level (the discriminative, robust split) and leave
    exact glyph naming to the legend-reading LLM.
    """
    fr = blob["fill_ratio"]
    aspect = blob["bbox_w"] / blob["bbox_h"] if blob["bbox_h"] else 1.0
    if fr >= 0.6:
        return "filled", "filled_blob"
    if fr <= 0.45 and 0.6 <= aspect <= 1.7:
        return "stroked", "stroked_glyph"
    return "filled", "ambiguous"


def detect_markers(page, bbox, dpi: int = _RENDER_DPI) -> tuple[list[MarkerRecord], list[str]]:
    """Full raster detection for one figure region. Returns (markers, warnings)."""
    arr = render_region(page, bbox, dpi)
    blobs = detect_blobs(arr)
    warnings: list[str] = []
    if not blobs:
        return [], ["raster: no marker-scale blobs found after line suppression"]

    records = []
    for b in blobs:
        mtype, shape = classify_blob_shape(b)
        records.append(MarkerRecord(group_key=shape, marker_type=mtype,
                                    pixel_x=b["cx"], pixel_y=b["cy"]))

    n_amb = sum(1 for b in blobs if classify_blob_shape(b)[1] == "ambiguous")
    if n_amb > 0.3 * len(blobs):
        warnings.append(
            f"raster: {n_amb}/{len(blobs)} blobs ambiguous shape — monochrome "
            "shape coding at this resolution is low-confidence; review required."
        )
    return records, warnings
