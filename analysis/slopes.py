"""Slope analysis: straight-line fits of log D against pH or log[extractant].

For a cation-exchange extraction M³⁺ + n(HA) ⇌ MA₃(HA)ₙ₋₃ + 3H⁺,
log D = const + 3·pH + n·log[HA]: at fixed extractant concentration the
slope of log D vs pH counts the H⁺ released per metal ion, and at fixed
equilibrium pH the slope of log D vs log[extractant] counts the extractant
units (monomer or dimer, as the paper's concentration counts them) per metal
ion. The page states those readings; this module only fits the lines.

What is fitted, per curve:
  * y is log D where D is reported (or derivable from %E and the phase ratio),
    else log[%E/(100 − %E)], which differs from log D by log(O/A) — a constant
    within one curve, so the slope is the same;
  * only points inside derived.RELIABLE_PCT when %E is known (log D runs off
    to ±∞ at the 0 % and 100 % plateaus, where a small reading error moves it
    by tenths of a decade);
  * x is pH, or log10 of the extractant concentration (x <= 0 cannot be);
  * at least MIN_FIT_POINTS distinct x values, or no fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import derived
from .curves import FIGURE, CurveSet
from .data import ELEMENT, PAPER

MIN_FIT_POINTS = 4

BASIS_D = "log D"
BASIS_RATIO = "log[%E/(100 − %E)]"

# Below this extractant/metal mole ratio, extractant bound to the metal is
# not negligible, so free extractant < total and log[total] misplaces x.
EXCESS_RATIO = 10.0


@dataclass
class Fit:
    slope: float
    intercept: float
    slope_se: float
    r2: float
    n: int


def fit_line(x, y) -> Fit | None:
    """Ordinary least squares y = a + b·x with the standard error of b
    and R². None with fewer than MIN_FIT_POINTS distinct x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(np.unique(x)) < MIN_FIT_POINTS:
        return None
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ssr = float((resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    sxx = float(((x - x.mean()) ** 2).sum())
    se = float(np.sqrt(ssr / (len(x) - 2) / sxx))
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    return Fit(float(b), float(a), se, r2, len(x))


def _basis(pts: pd.DataFrame) -> str | None:
    """log D if every point has it (reported or derived), else the %E ratio
    if every point has %E; mixing the two within one curve would mix
    offsets."""
    if pts["log_d"].notna().all():
        return BASIS_D
    if pts["pct"].notna().all():
        return BASIS_RATIO
    return None


def analyse(cs: CurveSet, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every curve of `cs` (built along pH for kind 'pH', along
    `Extractant Conc. (mM)` for kind 'extractant', with y EXTRACTION).

    Returns (points, fits). points: one row per point with `fx` (pH or
    log10 conc), `fy`, `basis`, `used` (bool) and `why` (reason a point was
    left out). fits: one row per curve with slope, its standard error, R²,
    n, the fitted x range, and a `note` when no fit was made or a caveat applies.
    """
    if kind not in ("pH", "extractant"):
        raise ValueError(kind)
    point_rows, fit_rows = [], []
    lo, hi = derived.RELIABLE_PCT
    by_curve = dict(tuple(cs.points.groupby("curve")))
    for c in cs.curves.to_dict("records"):
        pts = by_curve[c["curve"]].copy()
        if kind == "extractant":
            positive = pts["x"] > 0
            pts["fx"] = np.log10(pts["x"].where(positive))
        else:
            positive = pd.Series(True, index=pts.index)
            pts["fx"] = pts["x"]
        basis = _basis(pts)
        pts["basis"] = basis
        pts["fy"] = pts["log_d"] if basis == BASIS_D else pts["log_ratio"]
        in_band = pts["pct"].isna() | derived.in_reliable_band(pts["pct"])
        pts["used"] = positive & in_band & np.isfinite(pts["fy"]) & (basis is not None)
        pts["why"] = np.select(
            [~positive, ~in_band, ~np.isfinite(pts["fy"].astype(float))],
            ["concentration 0 has no logarithm", f"outside {lo:g}–{hi:g} % extraction", "no value"],
            default="",
        )
        point_rows.append(pts)

        used = pts[pts["used"]]
        fit = fit_line(used["fx"], used["fy"]) if basis is not None else None
        notes = []
        if basis is None:
            notes.append("mixes reported D and %E without a phase ratio")
        elif fit is None:
            notes.append(f"only {used['fx'].nunique()} usable point(s); a fit needs {MIN_FIT_POINTS}")
        if kind == "pH" and c.get("pH type") == "initial":
            notes.append("initial pH: the method assumes equilibrium pH")
        if kind == "extractant":
            ratio = pd.to_numeric(used["Molar ratio of EX/REE"], errors="coerce")
            low = int((ratio < EXCESS_RATIO).sum())
            if low:
                notes.append(f"extractant/metal below {EXCESS_RATIO:g} at {low} fitted point(s): "
                             "free extractant is less than the total, which biases the slope")
        fit_rows.append({
            "curve": c["curve"], "label": c["label"], PAPER: c[PAPER], FIGURE: c[FIGURE],
            ELEMENT: c[ELEMENT], "conditions": c["conditions"], "pH type": c.get("pH type"),
            "basis": basis,
            "slope": fit.slope if fit else None, "slope_se": fit.slope_se if fit else None,
            "intercept": fit.intercept if fit else None, "r2": fit.r2 if fit else None,
            "n_fit": fit.n if fit else len(used), "n_left_out": int((~pts["used"]).sum()),
            "x_min": float(used["fx"].min()) if fit else None,
            "x_max": float(used["fx"].max()) if fit else None,
            "note": "; ".join(notes),
        })
    points = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    return points, pd.DataFrame(fit_rows)
