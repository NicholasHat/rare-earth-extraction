"""Vector path — classify, group, and assemble markers (plan §4.1–4.3).

Grouping is by deterministic visual identity (colour for colour figures, shape
for monochrome); assembly uses an `eps` derived from each group's own MARKER
GEOMETRY (constant across the curve), not inter-point spacing — that's the fix
for the global-tolerance under-counting bug.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .types import MarkerRecord

# Groups smaller than this are treated as legend swatches / reference marks,
# not data series, and dropped (with a warning).
MIN_MARKERS_PER_GROUP = 3


def _hex(colour) -> str:
    if not colour:
        return "#000000"
    r, g, b = (int(round(v * 255)) for v in colour[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def centroid(obj) -> tuple[float, float]:
    return ((obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2)


def classify_marker_type(obj) -> str:
    """'filled' for a closed fill path; 'stroked' for an unfilled line fragment."""
    return "filled" if obj.get("fill") else "stroked"


def group_filled_by_colour(markers) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in markers:
        groups[_hex(m.get("non_stroking_color"))].append(m)
    return dict(groups)


def _calibrate_eps_filled(group_objs) -> float:
    """Dedup eps for filled markers — a SMALL fraction of marker size, so it only
    merges truly coincident paths (a marker drawn as outline+fill, ~0px apart),
    never distinct neighbours. Grounded on real geometry: in Swain Fig. 2 the
    closest two *distinct* same-series markers are ~1.3px apart, while this eps is
    ~0.8px, so dense-zone neighbours are preserved (the under-count fix, plan §4.2)."""
    diags = [math.hypot(o["width"], o["height"]) for o in group_objs]
    return 0.1 * float(np.median(diags)) if diags else 0.5


def _single_linkage(points: np.ndarray, eps: float) -> list[list[int]]:
    """Group point indices whose pairwise gap <= eps (single-linkage, union-find)."""
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # O(n^2) is fine for the few-hundred markers per figure we see in practice.
    for i in range(n):
        for j in range(i + 1, n):
            if math.dist(points[i], points[j]) <= eps:
                union(i, j)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return list(clusters.values())


def assemble_filled(group_key: str, group_objs: list[dict]) -> list[MarkerRecord]:
    """Dedupe near-coincident filled paths (a marker drawn as outline+fill = 2
    objects) into one marker each. Cleanly-separated markers are never merged
    because eps is marker-size-scaled, far below inter-marker spacing."""
    pts = np.array([centroid(o) for o in group_objs])
    eps = _calibrate_eps_filled(group_objs)
    records = []
    for idxs in _single_linkage(pts, eps):
        cx = float(np.mean([pts[i][0] for i in idxs]))
        cy = float(np.mean([pts[i][1] for i in idxs]))
        records.append(MarkerRecord(group_key=group_key, marker_type="filled",
                                    pixel_x=cx, pixel_y=cy))
    return records


def _segment_midpoint(seg) -> tuple[float, float]:
    return ((seg["x0"] + seg["x1"]) / 2, (seg["top"] + seg["bottom"]) / 2)


def _segment_length(seg) -> float:
    return math.hypot(seg["x1"] - seg["x0"], seg["bottom"] - seg["top"])


def _calibrate_eps_stroked(segments: list[dict]) -> float:
    """Assembly eps for stroked glyphs — a fraction of the fragment's own arm
    length (plan §4.2), so it stays constant across a curve's density instead
    of scaling with inter-marker spacing like the originally-diagnosed bug."""
    lengths = [_segment_length(s) for s in segments]
    return 0.6 * float(np.median(lengths)) if lengths else 3.0


def _segment_angle_deg(seg) -> float:
    return math.degrees(math.atan2(seg["bottom"] - seg["top"], seg["x1"] - seg["x0"])) % 180


def _round_to(angle_deg: float, step: int) -> int:
    return round(angle_deg / step) % (180 // step) * step


def classify_glyph_shape(segs: list[dict]) -> str | None:
    """Shape class of an assembled stroked glyph, from its fragments' angles
    (mod 180 — a line has no direction): '+' = one ~0deg + one ~90deg segment,
    '×' = one ~45deg + one ~135deg segment, '✶' = three fragments ~60deg apart.
    Anything else (e.g. two near-parallel fragments, a coincidental crossing of
    unrelated strokes like axis ticks) isn't a recognized glyph -> None, so
    assemble_stroked discards it as noise rather than mislabelling it."""
    if len(segs) == 2:
        angles = sorted(_round_to(_segment_angle_deg(s), 45) for s in segs)
        if angles == [0, 90]:
            return "plus"
        if angles == [45, 135]:
            return "cross"
        return None
    if len(segs) == 3:
        angles = sorted(_round_to(_segment_angle_deg(s), 60) for s in segs)
        if angles == [0, 60, 120]:
            return "star"
        return None
    return None


def assemble_stroked(segments: list[dict]) -> list[MarkerRecord]:
    """Assemble monochrome ×/+/✶ glyphs from their 2-3 line-segment fragments
    and classify each assembled glyph's shape (plan §4.2, §2 "monochrome
    figures: identity = marker shape class"). Unlike the filled path, identity
    isn't known until after assembly, so this clusters fragments globally on a
    single geometry-derived eps first, then classifies each resulting glyph."""
    if not segments:
        return []
    eps = _calibrate_eps_stroked(segments)
    pts = np.array([_segment_midpoint(s) for s in segments])
    records = []
    for idxs in _single_linkage(pts, eps):
        if len(idxs) < 2:
            continue  # a lone unmatched fragment is noise, not an assembled glyph
        glyph_segs = [segments[i] for i in idxs]
        shape = classify_glyph_shape(glyph_segs)
        if shape is None:
            continue  # fragments crossed/clustered but don't form a known glyph
        cx = float(np.mean([pts[i][0] for i in idxs]))
        cy = float(np.mean([pts[i][1] for i in idxs]))
        records.append(MarkerRecord(group_key=shape, marker_type="stroked",
                                    pixel_x=cx, pixel_y=cy))
    return records


def group_stroked_by_shape(records: list[MarkerRecord]) -> dict[str, list[MarkerRecord]]:
    groups: dict[str, list[MarkerRecord]] = defaultdict(list)
    for r in records:
        groups[r.group_key].append(r)
    return dict(groups)


def detect_merge_warnings(records: list[MarkerRecord], per_group_counts: dict[str, int]) -> list[str]:
    """Flag groups whose count is a low outlier vs siblings — the deterministic
    analogue of the row_count_sanity QA check, raised at the source (plan §4.3)."""
    warnings = []
    counts = [n for n in per_group_counts.values()]
    if len(counts) >= 3:
        med = float(np.median(counts))
        for key, n in per_group_counts.items():
            if med > 0 and n < 0.6 * med:
                warnings.append(
                    f"group {key} has {n} markers vs median {med:.0f} across series — "
                    "possible under-detection or a non-data group."
                )
    return warnings
