"""What a curve is — the one definition the QA checks (validation.checks) and
the Analysis page (analysis.curves) share.

A curve is one element's rows in one `Figure/Table` under one set of fixed
conditions, along the condition that varies: the x axis. Its key is the
element plus every condition column except x and the conditions that move
with x (pH follows an acid concentration; per-element feeds follow the total
feed). "Every condition column" is meant literally: the 2026-11 ten-paper
trial's key lacked `mixing method`, `Material Process`, `Sources`,
`Acid Solution` and `Diluent`, so a figure's 100–400 rpm series and its
saponified-vs-not series each pooled into one zig-zag "curve", and QA called
real data non-monotonic or duplicated.

Within one `Figure/Table` the x axis is found, not assumed: it is the numeric
condition along which the figure's rows form the longest curves (`figure_x`).
A figure where no numeric condition varies within a series — a bar chart with
one bar per extractant, a table of single conditions — has no x axis; its
"curves" are single points.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import COLUMNS, ELEMENT_COLUMN, FIGURE_COLUMN

# What a row measures, or derives from what it measures and its conditions —
# never part of a key. (`Extractant type` follows `Extractant`; the molar
# ratio follows the extractant concentration and the feed.)
_NOT_CONDITIONS = {
    "Reference No.", "DOI", ELEMENT_COLUMN, "Extractant type", "Molar ratio of EX/REE",
    "Extract%", "Recovery %", "Separation factor (SF%)", "Distribution ratio (D)",
}
CONDITION_COLUMNS: list[str] = [FIGURE_COLUMN] + [
    c for c in COLUMNS if c not in _NOT_CONDITIONS and c != FIGURE_COLUMN
]

# Numeric conditions a figure can sweep, in the order a tie between two of
# them is resolved. Time first: a grid of contact times × pH values (or
# temperatures) is a kinetics plot, one curve per pH.
X_COLUMNS: list[str] = [
    "Contact time (minute)",
    "Leaching time (minute)",
    "pH",
    "Extractant Conc. (mM)",
    "Acid Solution conc. (M)",
    "Extract Temperature (oC)",
    "Stripping Temperature (oC)",
    "Phase ratio (O/A)",
    "Co-extractant Conc. (mM)",
    "RRE composition (mM)",
    "Total REE feed (mM)",
]

FEED_COLUMNS = ("RRE composition (ppm)", "RRE composition (mM)")

# Conditions that move with the x axis, so they leave the key along with it.
LINKED: dict[str, set[str]] = {
    "Acid Solution conc. (M)": {"pH"},                       # pH follows the acid
    "Total REE feed (mM)": set(FEED_COLUMNS),
    "RRE composition (mM)": {"RRE composition (ppm)", "Total REE feed (mM)"},
}


def curve_key(x: str | None) -> list[str]:
    """The columns that identify a curve along `x` (None: a single point —
    every condition): the element, then every condition but x and what
    moves with it."""
    gone = LINKED.get(x, set()) | {x}
    return [ELEMENT_COLUMN] + [c for c in CONDITION_COLUMNS if c not in gone]


def _key(df: pd.DataFrame, x: str | None) -> list[str]:
    return [c for c in curve_key(x) if c in df.columns]


def _mean_curve_length(rows: pd.DataFrame, x: str) -> float:
    xs = pd.to_numeric(rows[x], errors="coerce")
    on = rows[xs.notna()]
    if on.empty:
        return 0.0
    return float(on.groupby(_key(on, x), dropna=False)[x].nunique().mean())


def figure_x(rows: pd.DataFrame) -> str | None:
    """The x axis of one figure's (or table's) rows: of the numeric conditions
    that vary, the one along which the rows form the longest curves (mean
    distinct x per curve). Longest, not merely most varied: when an acid
    concentration sweep also records each point's measured pH, keying on the
    acid leaves one-point "curves" along pH. None when no condition varies
    within a series."""
    best, best_len = None, 1.0
    for x in X_COLUMNS:
        if x in rows.columns and pd.to_numeric(rows[x], errors="coerce").nunique() > 1:
            n = _mean_curve_length(rows, x)
            if n > best_len:
                best, best_len = x, n
    return best


@dataclass
class Curve:
    label: str            # how a reviewer names it: "Dy (stirring 300 rpm) in Fig. 4(b)"
    x: str | None         # the varied column; None for a single-condition point
    figure: str | None
    rows: pd.DataFrame    # the curve's rows, in table order


def _describe(column: str, v) -> str:
    if isinstance(v, float):
        return f"{v:g} mM" if column == "Extractant Conc. (mM)" else f"{column} {v:g}"
    return str(v)


def _label(element, named: dict, varying: list[str], figure) -> str:
    conds = [_describe(c, named[c]) for c in varying if pd.notna(named.get(c))]
    label = f"{element} ({', '.join(conds)})" if conds else str(element)
    return label + (f" in {figure}" if figure is not None else "")


def curves(df: pd.DataFrame, x: str | None = None) -> list[Curve]:
    """Every curve of `df`: per `Figure/Table`, along that figure's own x
    (`figure_x`), or along `x` for every figure when it is given (a caller
    asking "which curves run along pH?"). Rows with no `Figure/Table`
    (extracted before prompt v15) form one group."""
    if ELEMENT_COLUMN not in df.columns or df.empty:
        return []
    work = df.copy()
    work[ELEMENT_COLUMN] = work[ELEMENT_COLUMN].fillna("(unspecified)")
    by_figure = work.groupby(FIGURE_COLUMN, dropna=False, sort=False) if FIGURE_COLUMN in work.columns \
        else [(None, work)]
    out: list[Curve] = []
    for figure, rows in by_figure:
        figure = None if (figure is None or pd.isna(figure)) else figure
        fx = x if x is not None else figure_x(rows)
        key = _key(rows, fx)
        groups = list(rows.groupby(key, dropna=False, sort=False))
        # Name a curve by the conditions that tell it apart from the same
        # element's other curves in the figure.
        firsts = pd.DataFrame([g.iloc[0][key] for _, g in groups]).astype(object)
        firsts = firsts.where(firsts.notna(), "\0")
        varying = [c for c in key[1:] if c != FIGURE_COLUMN
                   and (firsts.groupby(ELEMENT_COLUMN)[c].nunique() > 1).any()]
        if not varying:   # one series per element: name the extractant, as reviewers do
            varying = [c for c in ("Extractant", "Extractant Conc. (mM)") if c in key]
        for _, sub in groups:
            first = sub.iloc[0]
            out.append(Curve(_label(first[ELEMENT_COLUMN], first.to_dict(), varying, figure), fx, figure, sub))
    return out
