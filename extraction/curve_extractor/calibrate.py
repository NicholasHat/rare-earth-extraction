"""Axis calibration — pure least-squares pixel→data fit (plan §4.4).

Shared by both paths. Each path detects tick **pixel positions** and their
**data values** in its own way, then calls `fit_axis` here. The fit tries both a
linear and a log10 model and keeps whichever has the lower residual, so a log
axis (e.g. the 0.05→1.0 concentration sweep) is detected automatically.

Tick *values* are the one genuinely-OCR part (plan §4.4): `auto_ticks` makes a
best-effort read from pdfplumber chars, and the caller falls back to
LLM-supplied tick values when it returns too few.
"""
from __future__ import annotations

import re

import numpy as np

from .types import AxisCalibration

_RESIDUAL_FRAC_THRESHOLD = 0.02
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _fit_linear(pixels: np.ndarray, values: np.ndarray):
    A = np.vstack([pixels, np.ones_like(pixels)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, values, rcond=None)
    pred = slope * pixels + intercept
    resid = values - pred
    rms = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), rms, r2


def fit_axis(axis: str, tick_pixels: list[float], tick_values: list[float]) -> AxisCalibration:
    px = np.asarray(tick_pixels, dtype=float)
    val = np.asarray(tick_values, dtype=float)
    if len(px) < 2:
        raise ValueError(f"axis {axis!r} needs >= 2 ticks, got {len(px)}")

    slope, intercept, rms, r2 = _fit_linear(px, val)
    model = "linear"
    span = float(val.max() - val.min()) or 1.0

    if np.all(val > 0):
        ls, li, _, lr2 = _fit_linear(px, np.log10(val))
        pred_lin = 10.0 ** (ls * px + li)
        log_rms = float(np.sqrt(np.mean((val - pred_lin) ** 2)))
        if log_rms < rms:
            model, slope, intercept, rms, r2 = "log10", ls, li, log_rms, lr2

    return AxisCalibration(
        axis=axis, model=model, slope=slope, intercept=intercept,
        residual_rms=rms, r_squared=r2, n_ticks=len(px),
        tick_values=list(val), ok=rms <= _RESIDUAL_FRAC_THRESHOLD * span,
    )


def auto_ticks(page, frame, axis: str) -> tuple[list[float], list[float]] | None:
    """Best-effort read of (tick_pixels, tick_values) from numeric chars just
    outside the plot frame. Returns None if fewer than 3 monotone numeric labels
    are found (caller then uses an LLM-supplied mapping)."""
    x0, top, x1, bottom = frame
    found: list[tuple[float, float]] = []
    for w in page.extract_words():
        if not _NUM_RE.match(w["text"]):
            continue
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        if axis == "x" and bottom - 2 < cy < bottom + 28 and x0 - 25 < cx < x1 + 25:
            found.append((cx, float(w["text"])))
        elif axis == "y" and x0 - 50 < cx < x0 + 4 and top - 12 < cy < bottom + 12:
            found.append((cy, float(w["text"])))
    # dedupe by pixel, require monotone value sequence
    found = sorted(set(found))
    if len(found) < 3:
        return None
    pixels = [p for p, _ in found]
    values = [v for _, v in found]
    return pixels, values
