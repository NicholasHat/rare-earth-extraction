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

# A data marker has a roughly square, modest-sized bounding box; connecting-line
# fragments are elongated and most text characters are taller-than-wide or part of
# a tight horizontal run. These two filters cut the bulk of the contamination that
# blob detection alone leaves (validated on Quinn Fig. 2: 826 raw -> ~300 -> ~184).
_MARKER_MIN_SIDE = 6
_MARKER_MAX_SIDE = 26
_MARKER_ASPECT_LO = 0.55
_MARKER_ASPECT_HI = 1.8
# Text-row removal (conservative): a horizontal run of many tightly-spaced,
# similar blobs is an axis-label / legend / title row, not data.
_TEXTROW_Y_TOL = 8
_TEXTROW_MIN_BLOBS = 8
_TEXTROW_MAX_MEDIAN_GAP = 36


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


def _is_marker_shaped(b: dict) -> bool:
    """Square-ish, modest-sized bbox — rejects line fragments and oversized merges."""
    w, h = b["bbox_w"], b["bbox_h"]
    if not (_MARKER_MIN_SIDE <= w <= _MARKER_MAX_SIDE and _MARKER_MIN_SIDE <= h <= _MARKER_MAX_SIDE):
        return False
    aspect = w / h if h else 1.0
    return _MARKER_ASPECT_LO <= aspect <= _MARKER_ASPECT_HI


def _remove_text_rows(blobs: list[dict]) -> tuple[list[dict], int]:
    """Drop blobs that sit in a dense horizontal run (axis labels / legend / title).

    Conservative: requires many (>= _TEXTROW_MIN_BLOBS) tightly-spaced blobs sharing
    a y-baseline, so a sparse near-horizontal stretch of real curve points (a few
    markers, wider spacing) is preserved.
    """
    if not blobs:
        return blobs, 0
    order = sorted(range(len(blobs)), key=lambda i: blobs[i]["cy"])
    rows: list[list[int]] = [[order[0]]]
    for i in order[1:]:
        if blobs[i]["cy"] - blobs[rows[-1][-1]]["cy"] <= _TEXTROW_Y_TOL:
            rows[-1].append(i)
        else:
            rows.append([i])
    text_idx: set[int] = set()
    for r in rows:
        if len(r) < _TEXTROW_MIN_BLOBS:
            continue
        xs = sorted(blobs[i]["cx"] for i in r)
        gaps = np.diff(xs)
        if len(gaps) and float(np.median(gaps)) < _TEXTROW_MAX_MEDIAN_GAP:
            text_idx.update(r)
    kept = [b for i, b in enumerate(blobs) if i not in text_idx]
    return kept, len(text_idx)


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
    """Best-effort raster marker detection for one figure region.

    Returns (markers, warnings). The count is an ESTIMATE — monochrome shape coding
    in a multi-panel figure with baked-in text is the hardest case, so callers must
    treat the result as a lower-confidence hint (the pre-pass never marks raster
    pages authoritative) and the warnings flag it for manual digitisation.
    """
    arr = render_region(page, bbox, dpi)
    raw = detect_blobs(arr)
    warnings: list[str] = []
    if not raw:
        return [], ["raster: no marker-scale blobs found after line suppression"]

    shaped = [b for b in raw if _is_marker_shaped(b)]
    kept, n_text = _remove_text_rows(shaped)

    records = []
    for b in kept:
        mtype, shape = classify_blob_shape(b)
        records.append(MarkerRecord(group_key=shape, marker_type=mtype,
                                    pixel_x=b["cx"], pixel_y=b["cy"]))

    warnings.append(
        f"raster: ESTIMATE only — {len(raw)} raw blobs → {len(shaped)} marker-shaped "
        f"→ {len(kept)} after removing {n_text} text-row blobs. Multi-panel monochrome "
        "raster; verify counts by visual digitisation."
    )
    n_amb = sum(1 for b in kept if classify_blob_shape(b)[1] == "ambiguous")
    if kept and n_amb > 0.3 * len(kept):
        warnings.append(
            f"raster: {n_amb}/{len(kept)} kept blobs ambiguous shape — low-confidence."
        )
    return records, warnings
