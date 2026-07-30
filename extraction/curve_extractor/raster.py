"""Raster path — classical CV marker detection on a rendered figure image
(plan §6/§8: the fallback for papers whose figures are embedded images, e.g.
Quinn et al. 2015 — 200 DPI grayscale, monochrome, marker-shape-coded).

Pipeline: render the figure region at high DPI → threshold → suppress the thin
connecting/axis lines with a morphological opening (so markers sitting ON a line
don't merge into it) → connected-component blobs → size-filter to marker scale →
classify shape family (■ vs ● vs ▲/◆ by extent + mass offset; stroked ×/+/✶ by
fill ratio) → template-matching recovery (scikit-image `match_template`): markers
that overlap a curve, another marker, or baked-in text merge into blobs the size/
aspect filters rightly reject, so for each filled family with enough clean
detections one exemplar is cross-correlated over the full ink mask and new
correlation peaks become recovered markers. MarkerRecords come back in pixel
coords, grouped by shape family. Element mapping (shape→element via the legend)
and calibration are left to the caller/LLM, same seam as the vector path.

This is a best-effort detector: monochrome shape coding at ~15px with crowded
panels is the hardest figure class, so callers should treat its counts as
review-gated, not ground truth (the warnings surface low-confidence groups).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import ndimage
from skimage.feature import match_template, peak_local_max

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
# Filled shape families by extent (= area / bbox area, on the thresholded render):
# square ≈ 1.0, circle ≈ π/4 ≈ 0.79, triangle/diamond ≈ 0.5. Triangle vs diamond
# splits on vertical mass offset — a triangle's centroid sits off the bbox centre,
# a diamond's on it. Bands are loose to absorb anti-aliasing at ~15-30px.
_SQUARE_MIN_EXTENT = 0.87
_CIRCLE_MIN_EXTENT = 0.62
_FILLED_MIN_EXTENT = 0.42
_TRIANGLE_CENTROID_OFFSET = 0.08
# Template-matching recovery: needs enough clean same-family detections to trust
# an exemplar patch, and a correlation peak this strong to call it a marker.
_TEMPLATE_MIN_EXEMPLARS = 4
_TEMPLATE_MATCH_THRESHOLD = 0.6


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
        # Vertical mass offset within the bbox, in [-0.5, 0.5]: ~0 for centred
        # shapes (square/circle/diamond), off-centre for triangles.
        ys_idx = np.nonzero(sub)[0]
        centroid_dy = float(ys_idx.mean() / (h - 1) - 0.5) if h > 1 else 0.0
        blobs.append({
            "cx": float(cx), "cy": float(cy), "area": area,
            "bbox_w": w, "bbox_h": h, "fill_ratio": fill_ratio,
            "centroid_dy": centroid_dy,
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
    """(marker_type, shape family) from blob geometry.

    Filled shapes split into families by extent — square ≈ 1.0, circle ≈ 0.79,
    triangle/diamond ≈ 0.5 (triangle vs diamond by vertical mass offset) — so
    each family becomes its own group_key and per-series counts survive the
    raster path instead of collapsing into one "filled_blob" bucket. Stroked
    glyphs (×/+/✶) are thin strokes with a low fill ratio. Mapping family →
    element stays with the legend-reading LLM, as for vector colours.
    """
    fr = blob["fill_ratio"]
    aspect = blob["bbox_w"] / blob["bbox_h"] if blob["bbox_h"] else 1.0
    if fr >= _FILLED_MIN_EXTENT:
        if fr >= _SQUARE_MIN_EXTENT:
            return "filled", "filled_square"
        if fr >= _CIRCLE_MIN_EXTENT:
            return "filled", "filled_circle"
        if abs(blob.get("centroid_dy", 0.0)) > _TRIANGLE_CENTROID_OFFSET:
            return "filled", "filled_triangle"
        return "filled", "filled_diamond"
    if 0.6 <= aspect <= 1.7:
        return "stroked", "stroked_glyph"
    return "filled", "ambiguous"


def _recover_missed_markers(
    ink: np.ndarray, kept: list[dict]
) -> list[tuple[str, float, float]]:
    """Template-matching recovery pass (scikit-image).

    A marker that overlaps a curve, another marker, or baked-in text merges into
    a blob the size/aspect filters rightly reject — blob detection alone
    under-counts exactly where the figure is densest. For each filled shape
    family with enough clean detections, cross-correlate a median-area exemplar
    patch over the full ink mask and accept correlation peaks that don't
    coincide with an already-kept blob. Returns (family, cx, cy) per recovered
    marker; still estimate-tier — the pre-pass never marks raster pages
    authoritative.
    """
    by_family: dict[str, list[dict]] = defaultdict(list)
    for b in kept:
        mtype, family = classify_blob_shape(b)
        if mtype == "filled" and family != "ambiguous":
            by_family[family].append(b)

    occupied = [(b["cx"], b["cy"]) for b in kept]
    img = ink.astype(float)
    recovered: list[tuple[str, float, float]] = []
    for family, blobs in sorted(by_family.items()):
        if len(blobs) < _TEMPLATE_MIN_EXEMPLARS:
            continue
        exemplar = sorted(blobs, key=lambda b: b["area"])[len(blobs) // 2]
        w, h = exemplar["bbox_w"], exemplar["bbox_h"]
        y0 = max(int(round(exemplar["cy"] - h / 2)), 0)
        x0 = max(int(round(exemplar["cx"] - w / 2)), 0)
        template = img[y0:y0 + h, x0:x0 + w]
        if template.size == 0 or not template.any() or \
                template.shape[0] >= img.shape[0] or template.shape[1] >= img.shape[1]:
            continue
        response = match_template(img, template, pad_input=True)
        min_dist = max(3, int(0.7 * max(w, h)))
        peaks = peak_local_max(
            response, min_distance=min_dist, threshold_abs=_TEMPLATE_MATCH_THRESHOLD
        )
        for py, px in peaks:
            if all(math.hypot(px - ox, py - oy) > min_dist for ox, oy in occupied):
                occupied.append((float(px), float(py)))
                recovered.append((family, float(px), float(py)))
    return recovered


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

    recovered = _recover_missed_markers(_ink_mask(arr), kept)
    for family, cx, cy in recovered:
        records.append(MarkerRecord(group_key=family, marker_type="filled",
                                    pixel_x=cx, pixel_y=cy))
    if recovered:
        warnings.append(
            f"raster: template matching recovered {len(recovered)} marker(s) that "
            "blob detection lost to overlaps — verify visually."
        )

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
