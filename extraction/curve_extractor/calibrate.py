"""Axis calibration — pure least-squares pixel→data fit (plan §4.4).

Shared by both paths. Each path detects tick **pixel positions** and their
**data values** in its own way, then calls `fit_axis` here. The fit tries both a
linear and a log10 model and keeps whichever has the lower residual, so a log
axis (e.g. the 0.05→1.0 concentration sweep) is detected automatically.

Tick *values* are the one genuinely-OCR part (plan §4.4): `auto_ticks` reads
them from the PDF's own characters beside the frame, and `fit_ticks` fits them,
dropping a stray number that isn't a tick label. When too few labels are
found, the caller falls back to LLM-supplied tick values. Both follow the
sibling ree-extraction-local project, which calibrates every Swain & Otu
vector panel this way.
"""
from __future__ import annotations

import re

import numpy as np

from .types import AxisCalibration

_RESIDUAL_FRAC_THRESHOLD = 0.02
_NUM_RE = re.compile(r"^-?\d*\.?\d+$")
# Characters join into one label when on the same line and this close: a PDF
# may store "30" as "3" + "0" as separate text runs, and word extraction then
# reads two labels, "3" and "0" (why Swain & Otu's axes never calibrated).
_JOIN_GAP_PT = 1.2
_SAME_LINE_PT = 1.5
# Where tick labels sit relative to the frame, in PDF points.
_X_LABEL_BAND_PT = 14
_Y_LABEL_BAND_PT = 30
_MIN_LABELS = 3


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


def _numeric_labels(chars: list[dict]) -> list[tuple[float, float, float]]:
    """Join characters into numbers: (x centre, y centre, value)."""
    chars = sorted(chars, key=lambda c: (round(c["top"]), c["x0"]))
    runs: list[list[dict]] = []
    for c in chars:
        if runs and abs(c["top"] - runs[-1][-1]["top"]) < _SAME_LINE_PT \
                and c["x0"] - runs[-1][-1]["x1"] < _JOIN_GAP_PT:
            runs[-1].append(c)
        else:
            runs.append([c])
    out = []
    for run in runs:
        text = "".join(c["text"] for c in run).strip().replace("\u2212", "-").replace("\u2013", "-")
        if _NUM_RE.match(text):
            out.append(((run[0]["x0"] + run[-1]["x1"]) / 2,
                        (min(c["top"] for c in run) + max(c["bottom"] for c in run)) / 2,
                        float(text)))
    return out


def auto_ticks(page, frame, axis: str) -> tuple[list[float], list[float]] | None:
    """Best-effort read of (tick_pixels, tick_values) from the numeric labels
    printed just outside the plot frame: under it for x, left of it (or, when
    the left has too few, right of it) for y. Positions are the labels' own
    centres. Returns None if fewer than 3 are found (caller then uses an
    LLM-supplied mapping)."""
    x0, top, x1, bottom = frame
    if axis == "x":
        chars = [c for c in page.chars
                 if bottom < c["top"] < bottom + _X_LABEL_BAND_PT
                 and x0 - 10 < (c["x0"] + c["x1"]) / 2 < x1 + 10]
        found = sorted({(x, v) for x, _, v in _numeric_labels(chars)})
    else:
        found = []
        for side in ("left", "right"):
            if side == "left":
                chars = [c for c in page.chars if x0 - _Y_LABEL_BAND_PT < c["x0"] and c["x1"] < x0 - 0.5
                         and top - 6 < (c["top"] + c["bottom"]) / 2 < bottom + 6]
            else:
                chars = [c for c in page.chars if x1 + 0.5 < c["x0"] < x1 + _Y_LABEL_BAND_PT
                         and top - 6 < (c["top"] + c["bottom"]) / 2 < bottom + 6]
            found = sorted({(y, v) for _, y, v in _numeric_labels(chars)})
            if len(found) >= _MIN_LABELS:
                break
    if len(found) < _MIN_LABELS:
        return None
    return [p for p, _ in found], [v for _, v in found]


def fit_ticks(axis: str, tick_pixels: list[float], tick_values: list[float]) -> AxisCalibration | None:
    """fit_axis, dropping the worst-fitting label while the fit is poor — a
    stray number in the label band (a condition in an axis title, "at pH
    1.75") is not a tick. None when fewer than 3 labels remain."""
    pairs = list(zip(tick_pixels, tick_values))
    while len(pairs) >= _MIN_LABELS:
        cal = fit_axis(axis, [p for p, _ in pairs], [v for _, v in pairs])
        if cal.ok:
            return cal
        pairs.remove(max(pairs, key=lambda pv: abs(cal.pixel_to_data(pv[0]) - pv[1])))
    return None
