"""Approved rows -> curves.

A curve is one paper's rows for one element in one figure/table under one set
of fixed conditions, along whichever condition varies (the x axis). The key
is validation.curves' curve key — the definition the QA checks use, every
condition column but x and what moves with it (descriptors such as `Sources`
included: paper 7 of the 2026-10 trial ran the same LiCl sweep in four
solvents, recorded only there) — plus the paper.

Rows extracted before `Figure/Table` existed (prompt ≤ v14) group by their
conditions alone. A curve whose x repeats is the visible symptom of two
experiments that differ only in something the schema did not record; it is
drawn as points, never joined into a zig-zag line, and `repeated_x` says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from validation import schema
from validation.curves import FEED_COLUMNS, curve_key

from . import derived
from .data import ELEMENT, PAPER, series_position

FIGURE = schema.FIGURE_COLUMN
PH_TYPE = schema.PH_TYPE_COLUMN


@dataclass(frozen=True)
class Axis:
    column: str
    title: str
    log_scale: bool      # a log axis is the chemistry convention when the data span a decade


# Display of each x axis validation.curves knows, in the order the page offers
# them: column -> (title, log scale).
_AXIS_STYLE = {
    "pH": ("pH", False),
    "Extractant Conc. (mM)": ("Extractant conc. (mM)", True),
    "Acid Solution conc. (M)": ("Acid conc. (M)", False),
    "Contact time (minute)": ("Contact time (min)", False),
    "Leaching time (minute)": ("Leaching time (min)", False),
    "Extract Temperature (oC)": ("Temperature (°C)", False),
    "Stripping Temperature (oC)": ("Stripping temperature (°C)", False),
    "Phase ratio (O/A)": ("Phase ratio (O/A)", True),
    "Co-extractant Conc. (mM)": ("Co-extractant conc. (mM)", True),
    "RRE composition (mM)": ("Element feed (mM)", True),
    "Total REE feed (mM)": ("Total REE feed (mM)", True),
}
X_AXES: list[Axis] = [Axis(c, *style) for c, style in _AXIS_STYLE.items()]
AXIS_BY_COLUMN = {a.column: a for a in X_AXES}

# Pseudo-measure for the derived views: a row counts if it has %E or D.
EXTRACTION = "Extract% or D"

# y measures: name -> axis title.
Y_MEASURES = {
    "Extract%": "Extraction (%)",
    "Recovery %": "Recovery (%)",
    "D": "D",
}


def key_columns(x_col: str) -> list[str]:
    """The columns that identify a curve along `x_col` (x itself excluded)."""
    return ["paper_id"] + curve_key(x_col)


def set_columns(x_col: str) -> list[str]:
    """The columns that identify one experiment across elements: a curve's
    key minus the element and the per-element feed (a mixed feed holds each
    element at its own concentration)."""
    return [c for c in key_columns(x_col) if c not in (ELEMENT, *FEED_COLUMNS)]


# --------------------------------------------------------------------------- #
# Describing conditions
# --------------------------------------------------------------------------- #

def _fmt(col: str, v) -> str:
    if col == "Extractant Conc. (mM)":
        return f"{v:g} mM extractant"
    if col == "Co-extractant":
        return f"with {v}"
    if col == "Co-extractant Conc. (mM)":
        return f"{v:g} mM co-extractant"
    if col == "Diluent":
        return f"in {v}"
    if col == "RRE composition (ppm)":
        return f"{v:g} ppm feed"
    if col == "RRE composition (mM)":
        return f"{v:g} mM feed"
    if col == "Total REE feed (mM)":
        return f"{v:g} mM total REE"
    if col == "Extract Temperature (oC)":
        return f"{v:g} °C"
    if col == "Acid Solution conc. (M)":
        return f"{v:g} M acid"
    if col == "Leaching time (minute)":
        return f"{v:g} min leaching"
    if col == "Contact time (minute)":
        return f"{v:g} min contact"
    if col == "Phase ratio (O/A)":
        return f"O/A {v:g}"
    if col == "Stripping Temperature (oC)":
        return f"stripping {v:g} °C"
    if col == PH_TYPE:
        return f"{v} pH"
    if col == "pH":
        return f"pH {v:g}"
    return str(v)


def describe(values: dict, columns: list[str]) -> list[str]:
    """Readable parts for the non-null `columns` of one curve (paper_id is
    shown as the paper label; mM feed only when no ppm feed is given)."""
    parts = []
    for c in columns:
        v = values.get(c)
        if c == "paper_id" or v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if c == "RRE composition (mM)" and pd.notna(values.get("RRE composition (ppm)")):
            continue
        parts.append(_fmt(c, v))
    return parts


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

@dataclass
class CurveSet:
    """Curves along one x for one y measure.

    points: one row per plotted point — curve, label, PAPER, ELEMENT, FIGURE,
            x, y, y_source ('reported' or how it was derived), pct, log_ratio,
            log_d, d_source, plus the key columns.
    curves: one row per curve — curve, label, conditions, set, set_label,
            n_points, n_x, repeated_x, draw ('line' | 'points'), and the key.
    shared: conditions every curve has in common (for a caption).
    notes:  plain sentences about rows that could not be plotted.
    """
    x: str
    y: str
    points: pd.DataFrame
    curves: pd.DataFrame
    shared: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.points.empty


def _sort_key(s: pd.Series) -> pd.Series:
    """Numbers sort as numbers (50 before 1000), text as text, nulls last."""
    if s.name == "_order" or s.name in schema.NUMERIC_COLUMNS or s.name == "paper_id":
        return pd.to_numeric(s, errors="coerce")
    return s.astype("string")


def _null_safe(v):
    return None if (v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA) else v


def _varying(curves: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if curves[c].astype(object).where(curves[c].notna(), "\0").nunique() > 1]


def _labels(curves: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Each curve named by only what differs between the curves shown."""
    varying = _varying(curves, columns)
    order = [PAPER] + [c for c in varying if c != "paper_id"]
    labels = []
    for row in curves.to_dict("records"):
        parts = [row[PAPER]] if "paper_id" in varying else []
        parts += describe(row, [c for c in order if c != PAPER])
        labels.append(", ".join(parts) or str(row[ELEMENT] or row[PAPER]))
    out = pd.Series(labels, index=curves.index)
    dup = out.duplicated(keep=False)
    out[dup] = out[dup] + " (#" + curves.loc[dup, "curve"].astype(str) + ")"
    return out


SWEEP_MIN_POINTS = 3


def _also_on(df: pd.DataFrame, x: str) -> pd.Series:
    """For rows with no recorded source figure: the other axes along which the
    row also sits on a sweep of SWEEP_MIN_POINTS or more values. Without the
    figure such a row joins every sweep whose fixed conditions it matches —
    Swain & Otu's concentration sweep at pH 1.75 shares its 500 mM point with
    the pH sweep at 500 mM, and where the two measurements disagree the pH
    curve jumps. Which sweep the row came from is not recorded, so it is
    shown on both and named, never silently assigned to one."""
    out = pd.Series("", index=df.index, dtype=object)
    no_fig = df[FIGURE].isna()
    if not no_fig.any():
        return out
    for axis in X_AXES:
        if axis.column == x:
            continue
        v = pd.to_numeric(df[axis.column], errors="coerce")
        sub = df[v.notna()]
        if sub.empty:
            continue
        n = sub.groupby(key_columns(axis.column), dropna=False)[axis.column].transform("nunique")
        on = n.index[(n >= SWEEP_MIN_POINTS).to_numpy()]
        hit = out.index.isin(on) & no_fig.to_numpy()
        out[hit] = out[hit].where(out[hit] == "", out[hit] + ", ") + f"{axis.title} sweep"
    return out


def assemble(df: pd.DataFrame, x: str, y: str, *, min_x: int = 2) -> CurveSet:
    """Group rows into curves along `x` for measure `y` ('Extract%',
    'Recovery %' or 'D').

    A group with fewer than `min_x` distinct x values is not a curve along
    this axis — typically one point of a sweep along another axis (a
    concentration sweep's rows seen along pH) — and is left out, counted in
    the notes."""
    d, d_source = derived.distribution_ratio(df)
    work = df.copy()
    work["pct"] = pd.to_numeric(work["Extract%"], errors="coerce")
    work["d"] = d
    work["d_source"] = d_source
    work["log_d"] = np.log10(d.where(d > 0))
    work["log_ratio"] = derived.log_pct_ratio(work["pct"])
    work["x"] = pd.to_numeric(work[x], errors="coerce")
    work["also_on"] = _also_on(df, x)
    notes: list[str] = []
    if y == "D":
        work["y"] = work["d"]
        work["y_source"] = work["d_source"]
        no_d = work["x"].notna() & work["d"].isna() & work["pct"].between(0, 100, inclusive="neither")
        if no_d.any():
            notes.append(
                f"{int(no_d.sum())} point(s) have %E but no recorded phase ratio, so D cannot be derived "
                "for them; the Slope analysis tab plots them as log[%E/(100 − %E)]."
            )
    elif y == EXTRACTION:
        # Any measure of extraction: %E, else D (reported or derived). Used by
        # the derived views, which read pct / log_d / log_ratio, never y.
        work["y"] = work["pct"].fillna(work["d"])
        work["y_source"] = derived.D_REPORTED
    else:
        work["y"] = pd.to_numeric(work[y], errors="coerce")
        work["y_source"] = derived.D_REPORTED
    work = work[work["x"].notna() & work["y"].notna()]

    key = key_columns(x)
    if not work.empty:
        n_x = work.groupby(key, dropna=False, sort=False)["x"].transform("nunique")
        lone = n_x < min_x
        if lone.any():
            notes.append(
                f"{int(lone.sum())} point(s) measured at a single "
                f"{AXIS_BY_COLUMN[x].title if x in AXIS_BY_COLUMN else x} under their conditions are not shown "
                "(they belong to sweeps along another axis)."
            )
        work = work[~lone]
    empty_curves = pd.DataFrame(columns=["curve", "label", "conditions", "set", "set_label", "n_points",
                                         "n_x", "repeated_x", "draw", PAPER, *key])
    if work.empty:
        return CurveSet(x, y, work.assign(curve=pd.Series(dtype=int), label=pd.Series(dtype=str)), empty_curves, [], notes)

    rows = []
    groups = work.groupby(key, dropna=False, sort=False)
    for gid, (_, sub) in enumerate(groups):
        first = sub.iloc[0]
        rec = {c: _null_safe(first[c]) for c in key}
        rec[PAPER] = first[PAPER]
        rec["_gid"] = gid
        rec["n_points"] = len(sub)
        rec["n_x"] = int(sub["x"].nunique())
        rows.append(rec)
    curves = pd.DataFrame(rows)
    curves["_order"] = curves[ELEMENT].map(series_position)
    sort_cols = [PAPER, FIGURE, "_order"] + [c for c in key if c not in (PAPER, FIGURE, ELEMENT, "paper_id")]
    curves = curves.sort_values(sort_cols, key=_sort_key, na_position="last", kind="stable").reset_index(drop=True)
    curves["curve"] = range(len(curves))
    curves["repeated_x"] = curves["n_points"] > curves["n_x"]
    curves["draw"] = np.where((curves["n_x"] >= 3) & ~curves["repeated_x"], "line", "points")
    curves["label"] = _labels(curves, key)
    curves["conditions"] = [", ".join(describe(r, [c for c in key if c != ELEMENT])) for r in curves.to_dict("records")]

    set_cols = set_columns(x)
    set_ids = curves.groupby(set_cols, dropna=False, sort=False).ngroup()
    curves["set"] = set_ids
    sets = curves.drop_duplicates("set")
    set_names = _labels(sets.assign(**{ELEMENT: None}), set_cols)
    curves["set_label"] = curves["set"].map(dict(zip(sets["set"], set_names)))

    gid_to_curve = dict(zip(curves["_gid"], curves["curve"]))
    work["curve"] = groups.ngroup().map(gid_to_curve)
    points = work.merge(curves[["curve", "label", "conditions", "set", "set_label", "draw"]], on="curve")
    points = points.sort_values(["curve", "x"], kind="stable").reset_index(drop=True)

    shared_cols = [c for c in key if c not in _varying(curves, key)]
    shared = describe(curves.iloc[0].to_dict(), shared_cols)
    if PAPER in curves and curves[PAPER].nunique() == 1:
        shared = [curves[PAPER].iloc[0]] + shared
    curves = curves.drop(columns=["_gid", "_order"])
    if curves[FIGURE].isna().any():
        missing = sorted(curves.loc[curves[FIGURE].isna(), PAPER].unique())
        notes.append(
            f"Source figure not recorded for {', '.join(missing)} (extracted before prompt v15): "
            "those curves are grouped by their recorded conditions only."
        )
    shared_pts = points[(points["also_on"] != "") & points["curve"].map(curves.set_index("curve")["n_points"]).gt(1)]
    if not shared_pts.empty:
        notes.append(
            f"{len(shared_pts)} of those point(s) also sit on a sweep along another axis with the same fixed "
            "conditions (named in the hover); which experiment they came from is not recorded, so they are "
            "drawn on both, and a curve may jump where the two experiments disagree."
        )
    if curves["repeated_x"].any():
        n = int(curves["repeated_x"].sum())
        notes.append(
            f"{n} curve(s) have more than one point at the same {AXIS_BY_COLUMN[x].title if x in AXIS_BY_COLUMN else x}: "
            "probably two experiments that differ in something the database does not record. "
            "They are drawn as unconnected points."
        )
    return CurveSet(x, y, points, curves, shared, notes)


def varying_axes(df: pd.DataFrame, y: str) -> list[Axis]:
    """The x axes along which at least one curve of `df` has two or more
    distinct values of `y` measured — the only axes worth offering."""
    if y in ("D", EXTRACTION):
        d, _ = derived.distribution_ratio(df)
        has_y = d.notna() | (pd.to_numeric(df["Extract%"], errors="coerce").notna() if y == EXTRACTION else False)
    else:
        has_y = pd.to_numeric(df[y], errors="coerce").notna()
    out = []
    for axis in X_AXES:
        x = pd.to_numeric(df[axis.column], errors="coerce")
        sub = df[has_y & x.notna()]
        if sub.empty:
            continue
        n_x = sub.groupby(key_columns(axis.column), dropna=False)[axis.column].nunique()
        if (n_x >= 2).any():
            out.append(axis)
    return out


def ph_type_groups(cs: CurveSet) -> list[tuple[str | None, list[int]]]:
    """Curves split by pH type when the x axis is pH and the curves shown mix
    types (equilibrium and initial pH are not the same axis). One group with
    title None when there is nothing to split."""
    if cs.x != "pH" or cs.curves.empty:
        return [(None, cs.curves["curve"].tolist())]
    kinds = cs.curves[PH_TYPE].astype(object).where(cs.curves[PH_TYPE].notna(), None)
    distinct = list(dict.fromkeys(kinds.tolist()))
    if len(distinct) <= 1:
        return [(None, cs.curves["curve"].tolist())]
    titles = {"equilibrium": "Equilibrium pH", "initial": "Initial pH", None: "pH type not recorded"}
    order = sorted(distinct, key=lambda k: ("equilibrium", "initial", None).index(k) if k in titles else 1.5)
    return [(titles.get(k, f"pH type: {k}"), cs.curves.loc[kinds == k, "curve"].tolist()) for k in order]
