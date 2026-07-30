"""Experiment conditions → Extract% prediction from prior approved data (Pillar B).

The bench question this answers: "100 ppm feed, 500 mM Cyanex 272 — what
extraction % should I expect for La at pH 3.00?" Deterministically, from data:
find every pH-sweep series in `v_current_best` measured under similar
conditions (same extractant and element; extractant concentration and feed
within tolerance when supplied), then linearly interpolate each series at the
requested pH between its two bracketing measured points. Every prediction is
traceable to one paper's series and its bracket; a pH outside a series' measured
range yields no number for that series (reported as out-of-range, never
extrapolated), and series measured under clearly different conditions are
returned separately rather than silently pooled — the same "no invented
numbers" rule the assistant pillar follows.

Read-only by design: callers pass a `get_readonly_conn()` connection and this
module only queries `v_current_best` (never the raw extractions table).
"""
from __future__ import annotations

import sqlite3
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

# A series only counts as a pH sweep if it has at least this many distinct pH
# points — singletons (e.g. one row of a concentration sweep at fixed pH) can't
# be interpolated.
_MIN_SWEEP_POINTS = 3
# Condition tolerances (relative). Reported concentrations step coarsely
# (50/100/250/500/1000 mM), so ±25% separates neighbouring levels; feed
# compositions vary more between papers, so ±50%.
_CONC_TOL_FRAC = 0.25
_FEED_TOL_FRAC = 0.50


@dataclass
class SeriesPrediction:
    """One paper's pH-sweep series evaluated at the requested pH."""

    paper_id: int
    extractant_conc_mM: float | None
    feed_ppm: float | None
    n_points: int
    ph_min: float
    ph_max: float
    predicted_extract_pct: float | None            # None when pH outside the sweep
    # The two measured (pH, Extract%) points the prediction interpolates
    # between (equal when the requested pH hits a measured point exactly).
    bracket: tuple[tuple[float, float], tuple[float, float]] | None


@dataclass
class PredictionReport:
    element: str
    extractant: str
    ph: float
    matched: list[SeriesPrediction] = field(default_factory=list)
    off_condition: list[SeriesPrediction] = field(default_factory=list)

    @property
    def best_estimate(self) -> float | None:
        """Median across the condition-matched series that bracket the pH."""
        values = [s.predicted_extract_pct for s in self.matched
                  if s.predicted_extract_pct is not None]
        return median(values) if values else None


def _within(value: float | None, target: float, tol_frac: float) -> bool:
    if value is None:
        return False
    return abs(value - target) <= tol_frac * target


def _evaluate_series(rows: list[tuple[float, float]], ph: float) -> tuple[float | None, tuple | None]:
    """Linear interpolation at `ph` over measured (pH, Extract%) points.

    Duplicate-pH readings are averaged first. Returns (prediction, bracket);
    both None when `ph` falls outside the measured range.
    """
    by_ph: dict[float, list[float]] = defaultdict(list)
    for p, pct in rows:
        by_ph[p].append(pct)
    points = sorted((p, sum(v) / len(v)) for p, v in by_ph.items())
    phs = [p for p, _ in points]
    if not (phs[0] <= ph <= phs[-1]):
        return None, None
    i = bisect_left(phs, ph)
    if phs[i] == ph:
        lo = hi = points[i]
        return points[i][1], (lo, hi)
    lo, hi = points[i - 1], points[i]
    frac = (ph - lo[0]) / (hi[0] - lo[0])
    return lo[1] + frac * (hi[1] - lo[1]), (lo, hi)


def predict_extract_pct(
    conn: sqlite3.Connection,
    *,
    element: str,
    extractant: str,
    ph: float,
    extractant_conc_mM: float | None = None,
    feed_ppm: float | None = None,
) -> PredictionReport:
    """Predict Extract% at `ph` from prior approved pH sweeps.

    `extractant_conc_mM` / `feed_ppm` are optional condition filters: when
    given, series within tolerance land in `matched` and the rest in
    `off_condition` (still shown — a nearby condition is informative even when
    it isn't the requested one); when omitted, every sweep matches.
    """
    rows = conn.execute(
        """
        SELECT paper_id,
               "Extractant Conc. (mM)" AS conc,
               "RRE composition (ppm)" AS ppm,
               pH,
               "Extract%" AS pct
        FROM v_current_best
        WHERE "Extractant" = ?
          AND "Rare Earth Elements (REY:La, Ce, Nd)" LIKE '%' || ? || '%'
          AND pH IS NOT NULL
          AND "Extract%" IS NOT NULL
        """,
        (extractant, element),
    ).fetchall()

    series: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        series[(r["paper_id"], r["conc"], r["ppm"])].append((r["pH"], r["pct"]))

    report = PredictionReport(element=element, extractant=extractant, ph=ph)
    for (paper_id, conc, ppm), pts in sorted(series.items(), key=lambda kv: str(kv[0])):
        if len({p for p, _ in pts}) < _MIN_SWEEP_POINTS:
            continue
        prediction, bracket = _evaluate_series(pts, ph)
        sp = SeriesPrediction(
            paper_id=paper_id,
            extractant_conc_mM=conc,
            feed_ppm=ppm,
            n_points=len(pts),
            ph_min=min(p for p, _ in pts),
            ph_max=max(p for p, _ in pts),
            predicted_extract_pct=prediction,
            bracket=bracket,
        )
        on_condition = (
            (extractant_conc_mM is None or _within(conc, extractant_conc_mM, _CONC_TOL_FRAC))
            and (feed_ppm is None or _within(ppm, feed_ppm, _FEED_TOL_FRAC))
        )
        (report.matched if on_condition else report.off_condition).append(sp)
    return report
