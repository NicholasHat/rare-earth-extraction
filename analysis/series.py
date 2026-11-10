"""Across the rare earth series: the pH of 50 % extraction per element, and
separation factors between elements of one experiment.

Both work on a CurveSet (analysis.curves) and compare curves of one *set*:
the same paper, figure and fixed conditions, differing only in the element
(and that element's own feed concentration). Curves from different
experiments are never paired or averaged together.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import derived
from .curves import FIGURE, CurveSet
from .data import ATOMIC_NUMBER, ELEMENT, LANTHANIDES, PAPER, series_position


def half_extraction(cs: CurveSet) -> pd.DataFrame:
    """pH at 50 % extraction (pH½) for every %E-vs-pH curve of `cs`.

    One row per curve: its pH½ (None when the measured curve does not cross
    50 %, with the reason in `note`) and the two measured points it lies
    between. Interpolated inside the measured sweep only (derived.x_at_half).
    """
    if cs.x != "pH" or cs.y != "Extract%":
        raise ValueError("half_extraction needs a CurveSet of Extract% along pH")
    rows = []
    by_curve = dict(tuple(cs.points.groupby("curve")))
    for c in cs.curves.to_dict("records"):
        pts = by_curve[c["curve"]]
        half = derived.x_at_half(pts["x"], pts["y"], x_name="pH")
        lo, hi = half.bracket if half.bracket else (None, None)
        rows.append({
            "curve": c["curve"], "set": c["set"], "set_label": c["set_label"],
            PAPER: c[PAPER], FIGURE: c[FIGURE], ELEMENT: c[ELEMENT],
            "conditions": c["conditions"], "pH type": c.get("pH type"),
            "pH_half": half.x, "note": half.note,
            "between": (f"pH {lo[0]:g} ({lo[1]:.3g} %) and pH {hi[0]:g} ({hi[1]:.3g} %)"
                        if lo is not None and lo != hi else ""),
            "n_points": c["n_points"],
            "pH range": f"{pts['x'].min():g}–{pts['x'].max():g}",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_order"] = out[ELEMENT].map(series_position)
    return out.sort_values(["set", "_order"], kind="stable").drop(columns="_order").reset_index(drop=True)


def _pairs(elements: list[str], how: str) -> list[tuple[str, str, str]]:
    """(lighter, heavier, skipped) element pairs in series order. 'adjacent':
    consecutive lanthanides present (`skipped` names measured-elsewhere
    lanthanides in between; Pm is never measured and not counted);
    'all': every pair, Y and Sc included."""
    ordered = sorted(set(elements), key=series_position)
    if how == "all":
        return [(a, b, "") for a, b in combinations(ordered, 2)]
    lns = [e for e in ordered if e in LANTHANIDES]
    out = []
    for a, b in zip(lns, lns[1:]):
        between = [e for e in LANTHANIDES if ATOMIC_NUMBER[a] < ATOMIC_NUMBER[e] < ATOMIC_NUMBER[b]]
        out.append((a, b, ", ".join(between)))
    return out


def _basis(pts_a: pd.DataFrame, pts_b: pd.DataFrame) -> str | None:
    """What β is computed from for one pair of curves of one set: reported D
    when both curves report D on every point, else log[%E/(100 − %E)] when both
    have %E (the phase ratio is part of the set's key, so it is the same for
    both curves and cancels in the ratio). None when neither works."""
    if (pts_a["d_source"] == derived.D_REPORTED).all() and (pts_b["d_source"] == derived.D_REPORTED).all():
        return "D"
    if pts_a["pct"].notna().all() and pts_b["pct"].notna().all():
        return "%E"
    return None


def _usable(pts: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Points that take part in a ratio: inside the reliable %E band wherever
    %E is known, with a finite log value on the chosen basis."""
    value = pts["log_d"] if basis == "D" else pts["log_ratio"]
    band = pts["pct"].isna() | derived.in_reliable_band(pts["pct"])
    keep = band & np.isfinite(value)
    return pd.DataFrame({"x": pts["x"][keep], "v": value[keep], "pct": pts["pct"][keep]})


def separation_factors(cs: CurveSet, *, pairs: str = "adjacent", log_x: bool = False):
    """β = D(heavier) / D(lighter) for element pairs of one set, at each
    measured x of the heavier element's curve where the lighter one's
    measured range brackets it (its log value interpolated there in x, or log
    x when `log_x`; never extrapolated). Only points inside the reliable %E
    band count.

    Returns (points, summary, unavailable): one row per matched x; one row
    per pair with the median and range of β; and plain sentences for pairs
    that could not be computed.
    """
    point_rows, summary_rows, unavailable = [], [], []
    by_curve = dict(tuple(cs.points.groupby("curve")))
    for set_id, set_curves in cs.curves.groupby("set", sort=True):
        set_label = set_curves["set_label"].iloc[0]
        paper = set_curves[PAPER].iloc[0]
        counts = set_curves[ELEMENT].value_counts()
        for light, heavy, skipped in _pairs(set_curves[ELEMENT].tolist(), pairs):
            name = f"{heavy}/{light}"
            where = f"{name} in {set_label}"
            if counts[light] > 1 or counts[heavy] > 1:
                dup = light if counts[light] > 1 else heavy
                unavailable.append(f"{where}: {dup} has {counts[dup]} curves in this experiment "
                                   "(different feed levels), so the pair is ambiguous.")
                continue
            ca = set_curves.loc[set_curves[ELEMENT] == heavy, "curve"].iloc[0]
            cb = set_curves.loc[set_curves[ELEMENT] == light, "curve"].iloc[0]
            pa, pb = by_curve[ca], by_curve[cb]
            basis = _basis(pa, pb)
            if basis is None:
                unavailable.append(f"{where}: one curve reports only D and the other only %E, and the "
                                   "phase ratio needed to convert between them is not recorded.")
                continue
            ua, ub = _usable(pa, basis), _usable(pb, basis)
            matched = []
            for xa, va, pct_a in ua.itertuples(index=False):
                vb, exact = derived.interpolate_inside(ub["x"], ub["v"], xa, log_x=log_x)
                if vb is None:
                    continue
                matched.append((xa, va, vb, exact))
            if not matched:
                unavailable.append(f"{where}: the two curves have no measured {cs.x} in common inside "
                                   f"{derived.RELIABLE_PCT[0]:g}–{derived.RELIABLE_PCT[1]:g} % extraction.")
                continue
            betas = []
            for xa, va, vb, exact in matched:
                beta = float(10 ** (va - vb))
                betas.append(beta)
                point_rows.append({
                    "set": set_id, "set_label": set_label, PAPER: paper, "pair": name,
                    "heavier": heavy, "lighter": light, "x": xa, "beta": beta,
                    "matched": "same x" if exact else f"{light} interpolated",
                    "basis": basis,
                })
            summary_rows.append({
                "set": set_id, "set_label": set_label, PAPER: paper, "pair": name,
                "order": series_position(light) * 100 + series_position(heavy),
                "beta_median": float(np.median(betas)), "beta_min": float(min(betas)),
                "beta_max": float(max(betas)), "n": len(betas),
                "x_min": float(min(m[0] for m in matched)), "x_max": float(max(m[0] for m in matched)),
                "basis": "reported D" if basis == "D" else "%E (phase ratio cancels)",
                "skips": skipped,
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["set", "order"], kind="stable").reset_index(drop=True)
    return pd.DataFrame(point_rows), summary, unavailable
