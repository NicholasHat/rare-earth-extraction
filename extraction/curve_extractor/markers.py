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
