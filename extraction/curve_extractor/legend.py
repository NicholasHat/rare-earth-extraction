"""Markers by the figure's own legend swatches, and tick-label positions, for
scanned (raster) figures. Ported from the sibling ree-extraction-local project
(local_extract/raster_glyphs.py), where it finds ~82-90 % of Quinn et al.
2015 Fig. 1's markers with no false positives on the panel checked by hand.

Classifying each blob's shape fails on monochrome scans — a fitted line
through a square changes its outline, and anti-aliasing blurs square vs
circle — so markers are found by **template matching against the legend's
swatches**: the legend draws each series' exact marker at the same scale,
with no line through it. Every location goes to whichever swatch correlates
best, so markers come back already grouped by series. Tick labels are found
as text positions beside each axis (), the raster counterpart
of reading label positions from a vector PDF; pair them with the values read
off the figure and fit with calibrate.fit_ticks (calibrate.align_ticks pairs
tick *marks* instead, when the labels can't be located).

All coordinates are pixels of the figure rendered at 300 dpi, with panels
from raster.find_panels. scikit-image is imported lazily (match_markers only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Scans are anti-aliased and, here, upsampled: a thin hollow-marker outline
# sits mostly at gray 130-190, so a stricter cutoff breaks it open.
_INK = 190
_STRAIGHT_LINE_PX = 120      # ≈ 0.4 in: straight runs this long are axes/gridlines, erased before matching
_MIN_SIDE_PX = 9             # smaller: grid dots, text specks
_MAX_SIDE_PX = 40            # larger: merged markers, text runs
_HOLLOW_INK_FRAC = 0.72      # ink / filled area below this ⇒ hollow swatch
# Tick labels and axis titles sit left of / below a panel; legend swatches
# are searched only outside these zones.
_AXIS_ZONE_LEFT_PX = 150
_AXIS_ZONE_BELOW_PX = 110
_MATCH_THRESHOLD = 0.55      # normalised cross-correlation to accept a marker
_EDGE_MARGIN_PX = 14
_MATCH_SCALES = (0.8, 0.9, 1.0, 1.15)
# Body-area tie-break (solid square vs circle): opening strips fitted lines up
# to ~6 px thick; applied identically to swatch and marker, so it cancels.
_AREA_OPEN_ITERS = 3


@dataclass(frozen=True)
class Swatch:
    x: float
    y: float
    shape: str | None       # square / circle / triangle / diamond (None: unclear)
    fill: str               # solid / open
    template: np.ndarray    # float ink mask, the swatch as drawn

    @property
    def description(self) -> str:
        return f"{self.fill} {self.shape or 'marker'}"


@dataclass(frozen=True)
class Marker:
    x: float
    y: float
    swatch: int             # index into the swatch list it matched
    score: float


def ink(arr: np.ndarray) -> np.ndarray:
    return arr < _INK


def classify(mask: np.ndarray) -> str | None:
    """Shape of a clean filled mask from which bounding-box corners it
    occupies: square all four, triangle the two at its base, circle/diamond
    none (split by extent: circle ≈ π/4, diamond ≈ 1/2)."""
    h, w = mask.shape
    k = max(2, min(h, w) // 5)
    full = [mask[:k, :k].mean() > 0.5, mask[:k, -k:].mean() > 0.5,
            mask[-k:, :k].mean() > 0.5, mask[-k:, -k:].mean() > 0.5]
    if all(full):
        return "square"
    if full in ([False, False, True, True], [True, True, False, False]):
        return "triangle"
    if not any(full):
        return "circle" if mask.mean() >= 0.64 else "diamond"
    return None


def _in_axis_zone(x: float, y: float, frames) -> bool:
    return any(x0 - _AXIS_ZONE_LEFT_PX <= x <= x1 + 10 and top - 10 <= y <= bottom + _AXIS_ZONE_BELOW_PX
               for x0, top, x1, bottom in frames)


def find_swatches(arr: np.ndarray, frames) -> list[Swatch]:
    """Legend swatches: marker-sized glyphs outside every panel and its axis
    labels, standing apart from their neighbours on the left (a legend
    marker is followed by its line/label but not preceded by text, unlike a
    letter inside a word). Returned in reading order."""
    m = ink(arr)
    filled = ndimage.binary_fill_holes(m)
    labels, _ = ndimage.label(filled)
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (_MIN_SIDE_PX <= min(h, w) and max(h, w) <= _MAX_SIDE_PX and 0.6 <= w / h <= 1.7):
            continue
        x, y = (xs.start + xs.stop - 1) / 2, (ys.start + ys.stop - 1) / 2
        if _in_axis_zone(x, y, frames):
            continue
        if m[ys.start:ys.stop, max(0, xs.start - w):max(0, xs.start - 2)].any():
            continue
        mask = labels[sl] == i
        fill = "open" if m[sl][mask].mean() < _HOLLOW_INK_FRAC else "solid"
        pad = 2
        tpl = m[max(0, ys.start - pad):ys.stop + pad, max(0, xs.start - pad):xs.stop + pad]
        out.append(Swatch(x, y, classify(mask), fill, tpl.astype(float)))
    return sorted(out, key=lambda s: (round(s.y / 25), s.x))


def _straight_lines(m: np.ndarray) -> np.ndarray:
    return (ndimage.binary_opening(m, structure=np.ones((1, _STRAIGHT_LINE_PX)))
            | ndimage.binary_opening(m, structure=np.ones((_STRAIGHT_LINE_PX, 1))))


def _text_mask(m: np.ndarray) -> np.ndarray:
    """In-plot text (panel titles, "Slope 3", "1.0 M EHEHPA"): a horizontal run
    of three or more letter-sized glyphs spaced closer than a letter height.
    Data markers are rarely packed that tightly along one baseline."""
    labels, _ = ndimage.label(m)
    boxes = [(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
             for sl in ndimage.find_objects(labels)
             if 12 <= sl[0].stop - sl[0].start <= 40 and 2 <= sl[1].stop - sl[1].start <= 40]
    boxes.sort()
    text = np.zeros_like(m)
    used = [False] * len(boxes)
    for i, (x0, t0, x1, b0) in enumerate(boxes):
        if used[i]:
            continue
        run, right = [i], x1
        for j in range(i + 1, len(boxes)):
            bx0, bt, bx1, bb = boxes[j]
            height = b0 - t0
            overlap = min(b0, bb) - max(t0, bt)
            if bx0 - right > 0.8 * height:
                if bx0 - right > 3 * height:
                    break
                continue
            if overlap >= 0.5 * min(height, bb - bt):
                run.append(j)
                right = max(right, bx1)
        if len(run) >= 3:
            for k in run:
                used[k] = True
                bx0, bt, bx1, bb = boxes[k]
                text[max(bt - 2, 0):bb + 2, max(bx0 - 2, 0):bx1 + 2] = True
    return text


def _body_area(m: np.ndarray, x: float, y: float, half: int) -> int:
    """Filled area of the marker body at (x, y), lines stripped by opening."""
    y0, x0 = max(int(y) - half, 0), max(int(x) - half, 0)
    patch = ndimage.binary_opening(ndimage.binary_fill_holes(m[y0:int(y) + half + 1, x0:int(x) + half + 1]),
                                   iterations=_AREA_OPEN_ITERS)
    labels, _ = ndimage.label(patch)
    centre = labels[min(int(y) - y0, patch.shape[0] - 1), min(int(x) - x0, patch.shape[1] - 1)]
    return int((labels == centre).sum()) if centre else 0


def match_markers(arr: np.ndarray, box, swatches: list[Swatch]) -> list[Marker]:
    """Every marker inside `box`, assigned to its best-matching swatch.

    Before matching, straight axis/grid lines are erased (a tick crossing the
    x-axis otherwise reads as a "+") and so is in-plot text. Normalised
    cross-correlation of each swatch over the remaining ink gives candidates;
    overlapping candidates from different swatches resolve to the highest
    score — except between solid swatches, where a circle sits inside a
    square and their scores are close: there the marker's body area,
    relative to each swatch's own, decides."""
    from skimage.feature import match_template, peak_local_max

    # Match over the box grown by a marker's size, so a marker touching the
    # frame edge is matched whole; only centres inside the box are kept.
    bx0, btop, bx1, bbottom = box
    x0, top = max(bx0 - _EDGE_MARGIN_PX, 0), max(btop - _EDGE_MARGIN_PX, 0)
    x1, bottom = min(bx1 + _EDGE_MARGIN_PX, arr.shape[1] - 1), min(bbottom + _EDGE_MARGIN_PX, arr.shape[0] - 1)
    m = ink(arr[top:bottom + 1, x0:x1 + 1])
    m = m & ~_straight_lines(m)
    m = m & ~_text_mask(m)
    plot = m.astype(float)
    candidates = []
    for idx, sw in enumerate(swatches):
        th, tw = sw.template.shape
        if th >= plot.shape[0] or tw >= plot.shape[1]:
            continue
        # A legend swatch can be drawn a little larger or smaller than the
        # data markers (Quinn Fig. 4), and a thin hollow outline only
        # correlates at the right scale — so hollow swatches take the best
        # response over a few scales. Solid swatches stay at 1:1: a solid
        # body correlates across scales anyway, and an enlarged square would
        # swallow the circles.
        scales = _MATCH_SCALES if sw.fill == "open" else (1.0,)
        response = np.max([match_template(plot, ndimage.zoom(sw.template, s, order=0), pad_input=True)
                           for s in scales], axis=0)
        for py, px in peak_local_max(response, min_distance=max(3, int(0.5 * min(th, tw))),
                                     threshold_abs=_MATCH_THRESHOLD):
            candidates.append((float(response[py, px]), int(px), int(py), idx, min(th, tw)))

    swatch_area = [int(ndimage.binary_opening(ndimage.binary_fill_holes(s.template > 0),
                                              iterations=_AREA_OPEN_ITERS).sum())
                   for s in swatches]
    accepted: list[Marker] = []
    for score, px, py, idx, size in sorted(candidates, reverse=True):
        if not (bx0 <= px + x0 <= bx1 and btop <= py + top <= bbottom):
            continue
        if any(np.hypot(px + x0 - a.x, py + top - a.y) <= 0.6 * size for a in accepted):
            continue
        if swatches[idx].fill == "solid":
            rivals = [c for c in candidates
                      if swatches[c[3]].fill == "solid" and c[0] >= score - 0.15
                      and np.hypot(c[1] - px, c[2] - py) <= 0.6 * size]
            if len({c[3] for c in rivals}) > 1:
                area = _body_area(m, px, py, size)
                if area:
                    idx = min({c[3] for c in rivals},
                              key=lambda k: abs(np.log(area / max(swatch_area[k], 1))))
        accepted.append(Marker(float(px + x0), float(py + top), idx, round(score, 3)))
    return accepted


def label_centres(arr: np.ndarray, frame, axis: str) -> list[float]:
    """Pixel centres of the tick labels printed beside an axis — the raster
    counterpart of reading tick-label text positions from a vector PDF.

    y: labels are rows of glyphs right-aligned against the left axis; each
    row's vertical centre is its tick position (the rotated axis title sits
    further left and is excluded by requiring rows to end near the axis).
    x: labels are clusters of glyphs under the axis, separated by wider gaps
    than the characters inside one label; each cluster's horizontal centre
    is its tick position."""
    m = ink(arr)
    x0, top, x1, bottom = frame
    if axis == "y":
        region = m[max(top - 15, 0):bottom + 16, max(x0 - 130, 0):max(x0 - 4, 0)]
        oy, ox = max(top - 15, 0), max(x0 - 130, 0)
    elif axis == "top":                        # a second x scale printed above the frame
        region = m[max(top - 55, 0):max(top - 6, 0), max(x0 - 40, 0):x1 + 41]
        oy, ox = max(top - 55, 0), max(x0 - 40, 0)
    else:
        region = m[bottom + 6:bottom + 70, max(x0 - 40, 0):x1 + 41]
        oy, ox = bottom + 6, max(x0 - 40, 0)
    labels, _ = ndimage.label(region)
    boxes = [(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop) for sl in ndimage.find_objects(labels)
             if 3 <= sl[0].stop - sl[0].start <= 45]
    if not boxes:
        return []
    if axis == "y":
        # Group glyphs into text rows by vertical overlap.
        rows: list[list[tuple]] = []
        for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
            cy = (b[1] + b[3]) / 2
            if rows and abs(cy - np.mean([(r[1] + r[3]) / 2 for r in rows[-1]])) < 8:
                rows[-1].append(b)
            else:
                rows.append([b])
        right_edge = region.shape[1]
        centres = []
        for r in rows:
            if right_edge - max(b[2] for b in r) > 30:      # not against the axis: title / stray ink
                continue
            heights = [b[3] - b[1] for b in r]
            if max(heights) < 10:                          # a lone minus sign / speck
                continue
            tall = [b for b in r if b[3] - b[1] >= 0.6 * max(heights)]
            centres.append(oy + (min(b[1] for b in tall) + max(b[3] for b in tall)) / 2)
        return centres
    # x: only the first text line under the axis (the title is lower down).
    first_top = min(b[1] for b in boxes)
    line = [b for b in boxes if b[1] < first_top + 12 and b[3] - b[1] >= 10 or
            (b[1] < first_top + 25 and b[3] - b[1] < 10 and b[1] > first_top)]
    line.sort()
    clusters: list[list[tuple]] = []
    for b in line:
        if clusters and b[0] - max(c[2] for c in clusters[-1]) < 14:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    return [ox + (min(b[0] for b in c) + max(b[2] for b in c)) / 2 for c in clusters]


def has_top_scale(arr: np.ndarray, frame) -> bool:
    """A second, evenly spaced row of tick labels above the frame (a
    mixture-composition plot's reversed co-extractant scale)."""
    centres = label_centres(arr, frame, "top")
    if len(centres) < 4:
        return False
    gaps = np.diff(sorted(centres))
    return bool(gaps.std() / gaps.mean() < 0.1)


def even_subset(centres: list[float], n: int) -> list[float]:
    """The n of `centres` that are most evenly spaced — tick labels are;
    a stray row (a rotated title's subscript near the axis) is not."""
    pts = sorted(centres)
    while len(pts) > n:
        def spread(k):
            rest = pts[:k] + pts[k + 1:]
            gaps = np.diff(rest)
            return float(gaps.std() / gaps.mean())
        pts.pop(min(range(len(pts)), key=spread))
    return pts
