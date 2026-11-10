"""Analysis — curves and derived quantities over the approved data.

UI only: curve assembly, derived quantities and fits live in `analysis/`
(pure, unit-tested). Reads `v_current_best` on a read-only connection; this
page never writes to the master DB.
"""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from analysis import charts, curves, data, derived, series, slopes
from analysis.curves import AXIS_BY_COLUMN, EXTRACTION, FIGURE, Y_MEASURES
from analysis.data import ELEMENT, PAPER
from database import connection

st.set_page_config(page_title="Analysis", layout="wide")
st.title("Analysis")
st.caption(
    "Curves and quantities derived from approved data. Read-only. "
    "Anything computed rather than reported by a paper is marked as computed."
)

# Required before the read-only open below: SQLite's mode=ro will not create a
# missing file (same as the Calculator page).
connection.init_db()
conn = connection.get_readonly_conn()
try:
    rows = data.load(conn)
finally:
    conn.close()

if rows.excluded:
    st.caption(
        "Not shown: "
        + ", ".join(f"{n} row(s) for {el}" for el, n in rows.excluded.items())
        + " (not a rare earth element)."
    )
if rows.df.empty:
    st.info("No approved data yet. Papers appear here once they are approved and merged.")
    st.stop()

DARK = st.context.theme.type == "dark"
LO, HI = derived.RELIABLE_PCT

# --------------------------------------------------------------------------- #
# Filters (apply to every tab)
# --------------------------------------------------------------------------- #

df_all = rows.df
f1, f2, f3, f4 = st.columns(4)
sel_papers = f1.multiselect("Paper", sorted(df_all[PAPER].unique()), placeholder="All papers")
sel_extractants = f2.multiselect("Extractant", sorted(df_all["Extractant"].dropna().unique()), placeholder="All")
sel_elements = f3.multiselect(
    "Element", sorted(df_all[ELEMENT].dropna().unique(), key=data.series_position), placeholder="All"
)
sel_treatments = f4.multiselect("Treatment", sorted(df_all["Treatment"].dropna().unique()), placeholder="All")

df = df_all
for col, chosen in ((PAPER, sel_papers), ("Extractant", sel_extractants), (ELEMENT, sel_elements),
                    ("Treatment", sel_treatments)):
    if chosen:
        df = df[df[col].isin(chosen)]
if df.empty:
    st.info("No approved rows match these filters.")
    st.stop()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _axis_title(column: str) -> str:
    return AXIS_BY_COLUMN[column].title if column in AXIS_BY_COLUMN else column


def _log_default(column: str, values: pd.Series) -> bool:
    """Log x when the chemistry convention says so and the data span a decade."""
    v = pd.to_numeric(values, errors="coerce")
    v = v[v > 0]
    return bool(AXIS_BY_COLUMN[column].log_scale and len(v) and v.max() / v.min() >= 10)


def _limit(curve_ids: list[int], what: str = "curves") -> list[int]:
    if len(curve_ids) > charts.MAX_SERIES:
        st.caption(
            f"Showing the first {charts.MAX_SERIES} of {len(curve_ids)} {what} (in paper order). "
            "Narrow the filters to see the rest."
        )
        return curve_ids[: charts.MAX_SERIES]
    return curve_ids


def _download(frame: pd.DataFrame, name: str, key: str) -> None:
    st.download_button(
        "Download plotted data (CSV)", frame.to_csv(index=False).encode("utf-8"),
        file_name=name, mime="text/csv", key=key,
    )


def _curve_table(points: pd.DataFrame, x_title: str, y_title: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Paper": points[PAPER], "Figure/Table": points[FIGURE], "Element": points[ELEMENT],
        "Curve": points["label"], "Conditions": points["conditions"], "Also on": points["also_on"],
        x_title: points["x"], y_title: points["y"],
        "Value": points["y_source"].where(points["y_source"] == derived.D_REPORTED,
                                          "computed: " + points["y_source"].astype(str)),
    })


def _draw_curves(cs: curves.CurveSet, *, log_x: bool, key: str) -> None:
    """Chart(s) of a CurveSet: split by pH type when the x axis is pH and the
    curves mix types; capped at MAX_SERIES; points at x <= 0 left off a log axis."""
    x_title, y = _axis_title(cs.x), cs.y
    y_title = Y_MEASURES.get(y, y)
    if cs.shared:
        st.caption("All curves shown: " + "; ".join(cs.shared) + ".")
    for note in cs.notes:
        st.caption(note)
    plotted = []
    for title, ids in curves.ph_type_groups(cs):
        if title:
            st.markdown(f"**{title}**")
        ids = _limit(ids)
        pts = cs.points[cs.points["curve"].isin(ids)]
        crv = cs.curves[cs.curves["curve"].isin(ids)]
        if log_x and (pts["x"] <= 0).any():
            st.caption(f"{int((pts['x'] <= 0).sum())} point(s) at {x_title} = 0 cannot sit on a log axis and are not drawn.")
            pts = pts[pts["x"] > 0]
        st.altair_chart(
            charts.curve_chart(pts, crv, x_title=x_title, y_title=y_title, log_x=log_x,
                               log_y=(y == "D"), percent=(y != "D"), dark=DARK),
            width="stretch",
        )
        plotted.append(pts)
    if (cs.points["y_source"] != derived.D_REPORTED).any():
        st.caption("Hollow markers: D computed from %E and the phase ratio, D = [%E/(100 − %E)] / (O/A). "
                   "Filled markers: values as reported.")
    if (cs.curves["draw"] == "points").any():
        st.caption("Curves with fewer than three points, or with two points at one x, are drawn as points without a line.")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{y_title} vs {x_title}").strip("_").lower()
    _download(_curve_table(pd.concat(plotted), x_title, y_title), f"curves_{slug}.csv", key)


def _measures(frame: pd.DataFrame) -> list[str]:
    out = [m for m in ("Extract%", "Recovery %") if pd.to_numeric(frame[m], errors="coerce").notna().any()]
    if derived.distribution_ratio(frame)[0].notna().any():
        out.append("D")
    return out


def _curve_explorer(frame: pd.DataFrame, key: str) -> None:
    measures = _measures(frame)
    if not measures:
        st.info("No %E, recovery or D values in this selection.")
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    y = c1.radio("Measure", measures, format_func=lambda m: Y_MEASURES[m], horizontal=True, key=f"{key}_y")
    if "D" not in measures:
        c1.caption("D is not offered: no row here reports D or the phase ratio needed to compute it.")
    axes = curves.varying_axes(frame, y)
    if not axes:
        st.info(f"No curves in this selection: {Y_MEASURES[y].lower()} is not measured at two or more values "
                "of any one condition.")
        return
    axis = c2.selectbox("x axis", axes, format_func=lambda a: a.title, key=f"{key}_x")
    if axis.column == "Leaching time (minute)":
        c2.caption("Before prompt v15, extraction contact time was recorded as leaching time.")
    cs = curves.assemble(frame, axis.column, y)
    log_x = c3.checkbox("Log x", value=_log_default(axis.column, cs.points["x"]), key=f"{key}_logx_{axis.column}",
                        disabled=not AXIS_BY_COLUMN[axis.column].log_scale)
    if cs.empty:
        st.info("No curves to draw for this choice.")
        return
    _draw_curves(cs, log_x=log_x, key=f"{key}_csv")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tab_curves, tab_series, tab_beta, tab_slope, tab_compare = st.tabs(
    ["Curves", "Across the series", "Separation factors", "Slope analysis", "Compare papers"]
)

with tab_curves:
    _curve_explorer(df, "curves")

with tab_series:
    cs = curves.assemble(df, "pH", "Extract%")
    if cs.empty:
        st.info("No extraction-vs-pH curves in this selection.")
    else:
        table = series.half_extraction(cs)
        st.caption(
            "pH½ is the pH at 50 % extraction, interpolated between the two measured points either side of it "
            "(linearly in log[%E/(100 − %E)]) and never extrapolated beyond a measured sweep. "
            "At a phase ratio of 1 it is the pH where D = 1. Computed, not reported."
        )
        if table["pH_half"].isna().all():
            st.info(f"None of the {len(table)} extraction-vs-pH curves here crosses 50 % inside its measured range.")
        else:
            for title, ids in curves.ph_type_groups(cs):
                sub = table[table["curve"].isin(ids)]
                if title:
                    st.markdown(f"**{title}**")
                sets = list(dict.fromkeys(sub.loc[sub["pH_half"].notna(), "set"]))
                sub = sub[sub["set"].isin(_limit(sets, "experiments"))]
                if sub["pH_half"].notna().any():
                    st.altair_chart(charts.half_chart(sub, dark=DARK), width="stretch")
        shown = table.rename(columns={"set_label": "Experiment", "pH_half": "pH½ (computed)",
                                      "between": "Between measured points", "note": "Note"})
        shown = shown[["Experiment", PAPER, FIGURE, ELEMENT, "pH½ (computed)", "Between measured points", "Note",
                       "pH type", "pH range", "n_points"]].rename(columns={ELEMENT: "Element", "n_points": "Points"})
        st.dataframe(shown, width="stretch", hide_index=True,
                     column_config={"pH½ (computed)": st.column_config.NumberColumn(format="%.2f")})
        _download(shown, "pH_half.csv", "series_csv")

with tab_beta:
    axes = curves.varying_axes(df, EXTRACTION)
    if not axes:
        st.info("No curves in this selection to compare elements along.")
    else:
        c1, c2 = st.columns(2)
        axis = c1.selectbox("Matched along", axes, format_func=lambda a: a.title, key="beta_x")
        how = c2.radio("Pairs", ["adjacent", "all"], horizontal=True, key="beta_pairs",
                       format_func={"adjacent": "Adjacent lanthanides", "all": "Every pair"}.get)
        cs = curves.assemble(df, axis.column, EXTRACTION)
        pts, summary, unavailable = series.separation_factors(cs, pairs=how, log_x=axis.log_scale)
        st.caption(
            "β = D(heavier) / D(lighter), for two elements of the same experiment (same paper, figure and "
            f"conditions). Evaluated at each measured {axis.title} of the heavier element's curve where the "
            f"lighter element's curve brackets it (its log D interpolated linearly{' in log x' if axis.log_scale else ''}, "
            f"never extrapolated), using only points between {LO:g} and {HI:g} % extraction. Where D is not reported "
            "it comes from %E: the phase ratio is the same for both elements and cancels. Marker: median; "
            "bar: range over the matched points. Computed, not reported."
        )
        if how == "adjacent":
            st.caption("Y and Sc are not in the adjacent-lanthanide sequence; choose Every pair to include them.")
        if summary.empty:
            st.info("No element pair in this selection was measured in the same experiment over a common "
                    f"{axis.title} range." + (" Reasons are listed below." if unavailable else ""))
        else:
            sets = list(dict.fromkeys(summary["set"]))
            shown_sets = _limit(sets, "experiments")
            st.altair_chart(charts.beta_chart(summary[summary["set"].isin(shown_sets)], dark=DARK), width="stretch")
            table = summary.assign(
                **{"β median": summary["beta_median"], "β range": summary.apply(
                    lambda r: f"{r['beta_min']:.3g}–{r['beta_max']:.3g}", axis=1),
                   f"{axis.title} range": summary.apply(lambda r: f"{r['x_min']:g}–{r['x_max']:g}", axis=1)}
            )[["pair", "set_label", "β median", "β range", "n", f"{axis.title} range", "basis", "skips"]].rename(
                columns={"pair": "Pair", "set_label": "Experiment", "n": "Matched points", "basis": "From",
                         "skips": "Lanthanides between not measured"})
            st.dataframe(table, width="stretch", hide_index=True,
                         column_config={"β median": st.column_config.NumberColumn(format="%.3g")})
            _download(pts.rename(columns={"x": axis.title, "beta": "beta (computed)"}).drop(columns=["set"]),
                      "separation_factors.csv", "beta_csv")
        if unavailable:
            with st.expander(f"Pairs not computed ({len(unavailable)})"):
                for line in unavailable:
                    st.write(line)
        reported = df[pd.to_numeric(df["Separation factor (SF%)"], errors="coerce").notna()]
        if not reported.empty:
            st.markdown("**Separation factors as reported**")
            st.caption("The database does not record which element pair a reported value refers to.")
            st.dataframe(reported[[PAPER, FIGURE, ELEMENT, "Extractant", "pH", "Extractant Conc. (mM)",
                                   "Separation factor (SF%)"]], width="stretch", hide_index=True)

with tab_slope:
    kind = st.radio("Fit", ["pH", "extractant"], horizontal=True, key="slope_kind",
                    format_func={"pH": "log D vs pH", "extractant": "log D vs log[extractant]"}.get)
    x_col = "pH" if kind == "pH" else "Extractant Conc. (mM)"
    cs = curves.assemble(df, x_col, EXTRACTION)
    if kind == "pH":
        st.caption("At fixed extractant concentration, the slope of log D vs pH is read as the number of H⁺ "
                   "released per metal ion (3 for a trivalent rare earth exchanging with an acidic extractant).")
    else:
        st.caption("At fixed equilibrium pH and with extractant in large excess, the slope of log D vs "
                   "log[extractant] is read as the number of extractant units per metal ion — monomer or dimer, "
                   "as the paper's concentration counts them.")
    st.caption(
        f"Fitted by least squares per curve, on points between {LO:g} and {HI:g} % extraction only "
        f"(log D is ill-defined on the 0 % and 100 % plateaus), and only with at least {slopes.MIN_FIT_POINTS} "
        "such points. Hollow markers are left out of the fit. Where D is neither reported nor computable "
        "(no phase ratio recorded), y is log[%E/(100 − %E)] = log D + log(O/A): shifted by a constant, same slope."
    )
    if cs.empty:
        st.info(f"No curves along {_axis_title(x_col)} in this selection.")
    else:
        pts, fits = slopes.analyse(cs, kind)
        x_title = "pH" if kind == "pH" else "log10 extractant conc. (mM)"
        bases = set(fits["basis"].dropna())
        y_title = (slopes.BASIS_D if bases == {slopes.BASIS_D} else slopes.BASIS_RATIO if bases == {slopes.BASIS_RATIO}
                   else f"{slopes.BASIS_D} or {slopes.BASIS_RATIO} (see table)")
        for note in cs.notes:
            st.caption(note)
        if fits["slope"].isna().all():
            st.info(f"None of the {len(fits)} curve(s) here has {slopes.MIN_FIT_POINTS} or more points between "
                    f"{LO:g} and {HI:g} % extraction, so nothing is fitted. The points are still shown.")
        for title, ids in curves.ph_type_groups(cs):
            if title:
                st.markdown(f"**{title}**")
            ids = _limit(ids)
            st.altair_chart(
                charts.slope_chart(pts[pts["curve"].isin(ids)], fits[fits["curve"].isin(ids)],
                                   cs.curves[cs.curves["curve"].isin(ids)], x_title=x_title, y_title=y_title, dark=DARK),
                width="stretch",
            )
        table = fits.assign(
            **{"Slope (computed)": fits["slope"], "± SE": fits["slope_se"], "R²": fits["r2"],
               "Fitted range": fits.apply(lambda r: f"{r['x_min']:.3g}–{r['x_max']:.3g}" if pd.notna(r["slope"]) else "", axis=1)}
        )[["label", "Slope (computed)", "± SE", "R²", "n_fit", "n_left_out", "Fitted range", "basis", "note"]].rename(
            columns={"label": "Curve", "n_fit": "Points fitted", "n_left_out": "Left out", "basis": "y", "note": "Note"})
        st.dataframe(table, width="stretch", hide_index=True, column_config={
            "Slope (computed)": st.column_config.NumberColumn(format="%.2f"),
            "± SE": st.column_config.NumberColumn(format="%.2f"),
            "R²": st.column_config.NumberColumn(format="%.3f"),
        })
        csv = pts.merge(fits[["curve", "slope", "slope_se", "r2"]], on="curve")[
            ["label", PAPER, FIGURE, ELEMENT, "x", "fx", "fy", "basis", "used", "why", "slope", "slope_se", "r2"]
        ].rename(columns={"label": "Curve", "x": _axis_title(x_col), "fx": x_title, "fy": "y (computed)",
                          "used": "in fit", "why": "left out because", "slope": "slope (computed)"})
        _download(csv, f"slopes_{kind}.csv", "slope_csv")

with tab_compare:
    pairs = (df.dropna(subset=["Extractant"]).groupby([ELEMENT, "Extractant"])[PAPER].nunique())
    pairs = pairs[pairs >= 2]
    if pairs.empty:
        st.info("No element and extractant in this selection were measured by two or more papers. "
                "Extractant names are matched exactly as recorded.")
    else:
        options = sorted(pairs.index, key=lambda p: (data.series_position(p[0]), p[1]))
        el, ex = st.selectbox("Element and extractant", options, key="cmp_pair",
                              format_func=lambda p: f"{p[0]} with {p[1]} ({pairs[p]} papers)")
        st.caption(
            "Each paper's curves are drawn separately, never pooled. Their conditions are listed below the chart; "
            "'not recorded' means neither the paper's row nor its extraction gives the value. "
            "Extractant names are matched exactly as recorded."
        )
        sub = df[(df[ELEMENT] == el) & (df["Extractant"] == ex)]
        _curve_explorer(sub, "cmp")
        cond_cols = ["Diluent", "Extractant Conc. (mM)", "Co-extractant", "Co-extractant Conc. (mM)",
                     "RRE composition (ppm)", "RRE composition (mM)", "Total REE feed (mM)", "pH type",
                     "Extract Temperature (oC)", "Phase ratio (O/A)", "Contact time (minute)", "Acid Solution",
                     "Acid Solution conc. (M)", "Treatment"]
        conds = (sub.groupby([PAPER, FIGURE], dropna=False)[cond_cols]
                 .agg(lambda s: ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in s.dropna().unique())
                      or "not recorded")
                 .reset_index())
        conds[FIGURE] = conds[FIGURE].fillna("not recorded")
        st.markdown("**Conditions by paper and figure**")
        st.dataframe(conds, width="stretch", hide_index=True)
