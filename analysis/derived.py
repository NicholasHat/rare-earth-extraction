"""Quantities computed from reported ones — every one of them is marked as
computed wherever the page shows it.

Conventions (the same ones validation.checks uses for its %E-vs-D check):

  %E = 100·D / (D + A/O)   =>   D = [%E / (100 − %E)] / (O/A)

with O/A the `Phase ratio (O/A)` column (a solid sorbent's m/V in g/mL, D in
mL/g, so the same formula holds). Where the phase ratio is not recorded D
cannot be derived, but log[%E/(100 − %E)] = log D + log(O/A) still can: it
differs from log D only by a constant within one experiment, so slopes and
ratios between curves measured at the same phase ratio are unaffected.

Near 0 % and 100 % extraction, %E/(100 − %E) is ill-defined: a 0.5 % reading
error at 99 % moves log D by 0.2. RELIABLE_PCT is the band inside which a
%E-based value takes part in a fit or a ratio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RELIABLE_PCT = (5.0, 95.0)

D_REPORTED = "reported"
D_DERIVED = "derived from %E and O/A"


def distribution_ratio(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """D per row and where it came from: the reported `Distribution ratio
    (D)` when present, else D derived from `Extract%` and `Phase ratio (O/A)`
    when both are known and 0 < %E < 100. Returns (D, source); rows with
    neither get NaN and None."""
    reported = pd.to_numeric(df["Distribution ratio (D)"], errors="coerce")
    reported = reported.where(reported > 0)
    pct = pd.to_numeric(df["Extract%"], errors="coerce")
    oa = pd.to_numeric(df["Phase ratio (O/A)"], errors="coerce")
    ok = (pct > 0) & (pct < 100) & (oa > 0)
    derived = (pct / (100.0 - pct) / oa).where(ok)
    d = reported.fillna(derived)
    source = pd.Series(None, index=df.index, dtype=object)
    source[derived.notna()] = D_DERIVED
    source[reported.notna()] = D_REPORTED
    return d, source


def log_pct_ratio(pct) -> np.ndarray:
    """log10(%E / (100 − %E)); NaN where %E is not strictly inside 0..100."""
    p = np.asarray(pct, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(p / (100.0 - p))
    return np.where((p > 0) & (p < 100), out, np.nan)


def in_reliable_band(pct) -> np.ndarray:
    """True where %E lies inside RELIABLE_PCT (inclusive)."""
    p = np.asarray(pct, dtype=float)
    lo, hi = RELIABLE_PCT
    return (p >= lo) & (p <= hi)


def average_repeats(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Points sorted by x with repeated x values averaged (the same rule
    calculator.predict applies to duplicate-pH readings)."""
    s = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    s = s.groupby("x", sort=True)["y"].mean()
    return s.index.to_numpy(float), s.to_numpy(float)


@dataclass
class Half:
    """The x at which a curve passes 50 % extraction, or why there is none."""
    x: float | None
    note: str
    bracket: tuple[tuple[float, float], tuple[float, float]] | None = None


MIN_HALF_POINTS = 3


def x_at_half(x, pct, *, x_name: str = "pH") -> Half:
    """Where a measured curve crosses 50 % extraction, found between its two
    bracketing measured points and never extrapolated.

    Interpolation is linear in log[%E/(100 − %E)] (log D up to a constant),
    which is close to linear in pH over a bracket; if a bracketing point sits
    at exactly 0 or 100 % it falls back to linear in %E. A curve that crosses
    50 % more than once (scatter around the midpoint, or two experiments in
    one curve) gets no value rather than a guess.
    """
    xs, ps = average_repeats(x, pct)
    if len(xs) < MIN_HALF_POINTS:
        return Half(None, f"only {len(xs)} measured {x_name} value(s)")
    s = ps - 50.0
    nonzero = np.sign(s[s != 0])
    changes = int((nonzero[1:] != nonzero[:-1]).sum()) if len(nonzero) > 1 else 0
    if changes > 1:
        return Half(None, "crosses 50 % more than once")
    zeros = np.flatnonzero(s == 0)
    if changes == 0:
        if len(zeros) == 1:
            i = int(zeros[0])
            return Half(float(xs[i]), "measured at 50 %", ((xs[i], ps[i]), (xs[i], ps[i])))
        if len(zeros) > 1:
            return Half(None, "at 50 % over more than one measured point")
        i_max, i_min = int(np.argmax(ps)), int(np.argmin(ps))
        if ps[i_max] < 50:
            return Half(None, f"below 50 % throughout (max {ps[i_max]:.3g} % at {x_name} {xs[i_max]:g})")
        return Half(None, f"above 50 % throughout (min {ps[i_min]:.3g} % at {x_name} {xs[i_min]:g})")
    if len(zeros):
        i = int(zeros[0])
        return Half(float(xs[i]), "measured at 50 %", ((xs[i], ps[i]), (xs[i], ps[i])))
    i = int(np.flatnonzero(s[:-1] * s[1:] < 0)[0])
    (x0, p0), (x1, p1) = (xs[i], ps[i]), (xs[i + 1], ps[i + 1])
    l0, l1 = log_pct_ratio([p0, p1])
    if np.isfinite(l0) and np.isfinite(l1):
        value = x0 + (0.0 - l0) / (l1 - l0) * (x1 - x0)
        how = "interpolated in log[%E/(100 − %E)]"
    else:
        value = x0 + (50.0 - p0) / (p1 - p0) * (x1 - x0)
        how = "interpolated linearly in %E"
    return Half(float(value), how, ((float(x0), float(p0)), (float(x1), float(p1))))


def interpolate_inside(x_known, y_known, x_at: float, *, log_x: bool = False) -> tuple[float | None, bool]:
    """y at `x_at` by linear interpolation between the bracketing known points
    (in log x when `log_x`), or None outside the known range — never
    extrapolated. Returns (value, exact) where exact means x_at is a known x."""
    xs, ys = average_repeats(x_known, y_known)
    if len(xs) == 0 or not (xs[0] <= x_at <= xs[-1]):
        return None, False
    hit = np.flatnonzero(np.isclose(xs, x_at, rtol=0, atol=1e-9))
    if len(hit):
        return float(ys[hit[0]]), True
    i = int(np.searchsorted(xs, x_at))
    x0, x1 = xs[i - 1], xs[i]
    if log_x:
        x0, x1, xa = np.log10(x0), np.log10(x1), np.log10(x_at)
    else:
        xa = x_at
    return float(ys[i - 1] + (xa - x0) / (x1 - x0) * (ys[i] - ys[i - 1])), False
