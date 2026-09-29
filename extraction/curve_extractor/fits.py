"""Robust straight-line fits for log D series (ported from ree-extraction-local).

For the cation-exchange systems these papers study, log D is linear in pH
(−log[H⁺]) and in log[extractant] (the extraction prompt's Step 6). That prior is
used twice: to drop points that sit far off their own series' line (in-plot
text, a stray match) and to hand a point to a neighbouring series whose line
it sits on (a solid circle matched as a solid square).
"""
from __future__ import annotations

import numpy as np

Line = tuple[float, float]   # (intercept, slope)
_INLIER_TOL = 0.12           # log units — digitisation scatter of a clean point
_DROP_FLOOR = 0.5
_TARGET_TOL = 0.15           # a moved point must sit this close to its new series' line


def theil_sen(xy: list[tuple[float, float]]) -> Line | None:
    """Median-of-slopes line; tolerant of up to ~29 % outliers."""
    if len(xy) < 3:
        return None
    x, y = np.array(xy).T
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
    if not slopes:
        return None
    b = float(np.median(slopes))
    return float(np.median(y - b * x)), b


def consensus_line(xy: list[tuple[float, float]], tol: float = _INLIER_TOL) -> Line | None:
    """The line through a pair of points that most other points agree with
    (exhaustive RANSAC — series are small), refit by least squares on its
    inliers. Survives close to half the points being intruders — a series
    that absorbed in-plot text and a neighbour's markers, which breaks a
    median-slope fit (Quinn Fig. 1 Nd: 3 of 8)."""
    if len(xy) < 3:
        return theil_sen(xy)
    pts = np.array(xy)
    best, best_key = None, None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (x1, y1), (x2, y2) = pts[i], pts[j]
            if x1 == x2:
                continue
            b = (y2 - y1) / (x2 - x1)
            a = y1 - b * x1
            resid = np.abs(pts[:, 1] - (a + b * pts[:, 0]))
            inliers = resid <= tol
            key = (int(inliers.sum()), -float(resid[inliers].sum()))
            if best_key is None or key > best_key:
                best, best_key = inliers, key
    if best is None or best.sum() < 2:
        return theil_sen(xy)
    b, a = np.polyfit(pts[best, 0], pts[best, 1], 1)
    return float(a), float(b)


def residual(line: Line, x: float, y: float) -> float:
    a, b = line
    return abs(y - (a + b * x))


def drop_off_line(xy: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], int]:
    """Drop points beyond 3·1.4826·MAD of the series' line, never closer than
    0.5 log units: real series curve a little off the ideal slope (≈0.3 seen
    on Quinn's Cyanex 272 data), while text and stray matches land far off."""
    line = consensus_line(xy) if len(xy) >= 5 else None
    if line is None:
        return xy, 0
    resid = np.array([residual(line, x, y) for x, y in xy])
    limit = max(4.4478 * float(np.median(resid)), _DROP_FLOOR)
    kept = [p for p, r in zip(xy, resid) if r <= limit]
    return kept, len(xy) - len(kept)


def reassign(series_xy: dict[str, list[tuple[float, float]]],
             same_kind: dict[str, str]) -> tuple[dict[str, list[tuple[float, float]]], int]:
    """Move a point to another series (of the same marker fill, the only
    confusable kind) when it lies on that series' line and clearly off its own.
    Lines are consensus fits on the original assignment, so intruders
    don't bend the line they are judged against."""
    lines = {k: consensus_line(v) for k, v in series_xy.items() if len(v) >= 3}
    out = {k: [] for k in series_xy}
    moved = 0
    for k, pts in series_xy.items():
        for x, y in pts:
            own = residual(lines[k], x, y) if lines.get(k) else 0.0
            best = min(((residual(line, x, y), j) for j, line in lines.items()
                        if j != k and line and same_kind.get(j) == same_kind.get(k)), default=None)
            if best and own > 0.25 and best[0] < _TARGET_TOL and best[0] < own / 3:
                out[best[1]].append((x, y))
                moved += 1
            else:
                out[k].append((x, y))
    return {k: sorted(v) for k, v in out.items()}, moved
