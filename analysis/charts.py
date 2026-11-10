"""Altair specs for the Analysis page — form and colour only; every number
comes from the other analysis modules.

Series styling: eight categorical hues (a palette validated for colour-vision
deficiency on adjacent pairs, light and dark steps), used once with filled
circles and solid lines and once more with squares and dashed lines, so up to
MAX_SERIES curves stay distinguishable without colour alone. When each curve
shown is a different element, an element keeps the same style whatever else is
on the chart (La is always the first style, Lu the fourteenth); otherwise
styles follow the curves' sort order.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

from . import derived
from .curves import FIGURE
from .data import ELEMENT, LANTHANIDES, PAPER, SERIES_ORDER, series_position

LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SURFACE = {False: "#ffffff", True: "#0e1117"}      # Streamlit's light / dark backgrounds
MUTED = "#898781"
MAX_SERIES = 2 * len(LIGHT)
PLOT_HEIGHT = 360
LEGEND_ROW = 20


def style_indices(labels: list[str], elements: list[str] | None = None) -> list[int]:
    """Style slot per series: fixed by element when every series is a
    distinct element, else by position."""
    if elements is not None and len(set(elements)) == len(elements) and all(e in SERIES_ORDER for e in elements):
        return [series_position(e) for e in elements]
    return list(range(len(labels)))


def _scales(labels: list[str], elements: list[str] | None, dark: bool):
    palette = DARK if dark else LIGHT
    idx = style_indices(labels, elements)
    colors = [palette[i % len(palette)] for i in idx]
    shapes = ["circle" if i < len(palette) else "square" for i in idx]
    dashes = [[1, 0] if i < len(palette) else [6, 3] for i in idx]
    return (alt.Scale(domain=labels, range=colors), alt.Scale(domain=labels, range=shapes),
            alt.Scale(domain=labels, range=dashes))


def _legend_columns(labels: list[str]) -> int:
    longest = max((len(s) for s in labels), default=0)
    return 1 if longest > 45 else 2 if longest > 22 else 4 if longest > 10 else 8


def _legend(labels: list[str]) -> alt.Legend:
    # Entries run row by row in series order (no `direction`: Vega would reorder them).
    return alt.Legend(title=None, orient="bottom", columns=_legend_columns(labels),
                      labelLimit=900, symbolStrokeWidth=2)


def height(labels: list[str]) -> int:
    """Total chart height. Streamlit sizes a chart to fit its height
    *including* the legend, so a bottom legend must add to it, not eat the plot."""
    rows = -(-len(labels) // _legend_columns(labels)) if len(labels) > 1 else 0
    return PLOT_HEIGHT + 60 + LEGEND_ROW * rows


def _series_encodings(field: str, labels: list[str], elements: list[str] | None, dark: bool):
    color, shape, dash = _scales(labels, elements, dark)
    legend = _legend(labels) if len(labels) > 1 else None    # one series: the caption names it
    return (alt.Color(f"{field}:N", scale=color, legend=legend),
            alt.Shape(f"{field}:N", scale=shape, legend=legend),
            alt.StrokeDash(f"{field}:N", scale=dash, legend=None))


def _x(field: str, title: str, log: bool) -> alt.X:
    return alt.X(f"{field}:Q", title=title,
                 scale=alt.Scale(type="log", nice=False) if log else alt.Scale(zero=False, nice=True))


def curve_chart(points: pd.DataFrame, curves: pd.DataFrame, *, x_title: str, y_title: str,
                log_x: bool, log_y: bool, percent: bool, dark: bool = False) -> alt.LayerChart:
    """One line + marker series per curve. Curves drawn as 'points' (fewer
    than three x values, or a repeated x) get markers only. Hollow markers
    are computed values, filled ones reported."""
    labels = curves["label"].tolist()
    elements = curves[ELEMENT].tolist()
    color, shape, dash = _series_encodings("label", labels, elements, dark)
    data = points.assign(
        figure=points[FIGURE].fillna("not recorded"),
        value=points["y_source"].where(points["y_source"] == derived.D_REPORTED, "computed: " + points["y_source"].astype(str)),
        derived=points["y_source"] != derived.D_REPORTED,
    )[["curve", "label", "x", "y", "draw", "derived", "value", PAPER, "figure", ELEMENT, "conditions", "also_on"]]
    data = data.rename(columns={PAPER: "paper", ELEMENT: "element"})
    if percent:
        y_scale = alt.Scale(domain=[0, 100])
    elif log_y:
        y_scale = alt.Scale(type="log")
    else:
        y_scale = alt.Scale(zero=False)
    y = alt.Y("y:Q", title=y_title, scale=y_scale)
    x = _x("x", x_title, log_x)
    tooltip = [
        alt.Tooltip("paper:N", title="Paper"), alt.Tooltip("figure:N", title="Figure/Table"),
        alt.Tooltip("element:N", title="Element"), alt.Tooltip("conditions:N", title="Conditions"),
        alt.Tooltip("x:Q", title=x_title, format=".4g"), alt.Tooltip("y:Q", title=y_title, format=".4g"),
        alt.Tooltip("value:N", title="Value"),
    ]
    if (data["also_on"] != "").any():
        tooltip.append(alt.Tooltip("also_on:N", title="Also on"))
    base = alt.Chart(data)
    lines = base.transform_filter("datum.draw == 'line'").mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(
        x=x, y=y, color=color, strokeDash=dash, detail="curve:N", order="x:Q")
    reported = base.transform_filter("!datum.derived").mark_point(
        filled=True, size=70, opacity=1, stroke=SURFACE[dark], strokeWidth=1).encode(
        x=x, y=y, color=color, shape=shape, tooltip=tooltip)
    computed = base.transform_filter("datum.derived").mark_point(
        filled=False, size=70, opacity=1, strokeWidth=1.75).encode(
        x=x, y=y, color=color, shape=shape, tooltip=tooltip)
    return alt.layer(lines, reported, computed).properties(height=height(labels))


def half_chart(table: pd.DataFrame, dark: bool = False) -> alt.LayerChart:
    """pH at 50 % extraction against the element, in series order, one series
    per experiment (set)."""
    data = table[table["pH_half"].notna()].rename(columns={PAPER: "paper", ELEMENT: "element"})
    data = data.assign(figure=data[FIGURE].fillna("not recorded"))
    # Join an experiment's lanthanides with a line (Y and Sc stand apart: they
    # are not the next step of the series), and only when each element appears once.
    single = data.groupby("set_label")["element"].transform(lambda e: e[e.isin(LANTHANIDES)].is_unique)
    data = data.assign(connect=single.astype(bool) & data["element"].isin(LANTHANIDES))
    labels = list(dict.fromkeys(data["set_label"]))
    color, shape, dash = _series_encodings("set_label", labels, None, dark)
    order = [e for e in SERIES_ORDER if e in set(data["element"])]
    x = alt.X("element:N", sort=order, title=None, axis=alt.Axis(labelAngle=0))
    y = alt.Y("pH_half:Q", title="pH at 50 % extraction (computed)", scale=alt.Scale(zero=False))
    tooltip = [
        alt.Tooltip("paper:N", title="Paper"), alt.Tooltip("figure:N", title="Figure/Table"),
        alt.Tooltip("element:N", title="Element"), alt.Tooltip("pH_half:Q", title="pH½", format=".3f"),
        alt.Tooltip("between:N", title="Between measured"), alt.Tooltip("conditions:N", title="Conditions"),
    ]
    base = alt.Chart(data)
    lines = base.transform_filter("datum.connect").mark_line(strokeWidth=2).encode(
        x=x, y=y, color=color, strokeDash=dash, detail="set_label:N")
    pts = base.mark_point(filled=False, size=80, strokeWidth=2, opacity=1).encode(
        x=x, y=y, color=color, shape=shape, tooltip=tooltip)
    return alt.layer(lines, pts).properties(height=height(labels))


def beta_chart(summary: pd.DataFrame, dark: bool = False) -> alt.LayerChart:
    """Median β per element pair (marker) with the range over the matched
    points (rule), on a log axis with β = 1 marked."""
    data = summary.rename(columns={PAPER: "paper"})
    labels = list(dict.fromkeys(data["set_label"]))
    color, shape, _ = _series_encodings("set_label", labels, None, dark)
    order = list(dict.fromkeys(data.sort_values("order", kind="stable")["pair"]))
    x = alt.X("pair:N", sort=order, title=None, axis=alt.Axis(labelAngle=0))
    dodge = {"xOffset": alt.XOffset("set_label:N", scale=alt.Scale(domain=labels))} if len(labels) > 1 else {}
    y_scale = alt.Scale(type="log")
    tooltip = [
        alt.Tooltip("pair:N", title="Pair"), alt.Tooltip("paper:N", title="Paper"),
        alt.Tooltip("set_label:N", title="Experiment"),
        alt.Tooltip("beta_median:Q", title="β median", format=".3g"),
        alt.Tooltip("beta_min:Q", title="β min", format=".3g"), alt.Tooltip("beta_max:Q", title="β max", format=".3g"),
        alt.Tooltip("n:Q", title="Matched points"), alt.Tooltip("basis:N", title="From"),
    ]
    base = alt.Chart(data)
    one = alt.Chart(pd.DataFrame({"one": [1.0]})).mark_rule(color=MUTED, strokeWidth=1).encode(y=alt.Y("one:Q", scale=y_scale))
    rng = base.mark_rule(strokeWidth=2).encode(
        x=x, **dodge, y=alt.Y("beta_min:Q", scale=y_scale, title="β = D(heavier) / D(lighter) (computed)"),
        y2="beta_max:Q", color=color)
    pts = base.mark_point(filled=True, size=80, opacity=1, stroke=SURFACE[dark], strokeWidth=1).encode(
        x=x, **dodge, y=alt.Y("beta_median:Q", scale=y_scale), color=color, shape=shape, tooltip=tooltip)
    return alt.layer(one, rng, pts).properties(height=height(labels))


def slope_chart(points: pd.DataFrame, fits: pd.DataFrame, curves: pd.DataFrame, *, x_title: str,
                y_title: str, dark: bool = False) -> alt.LayerChart:
    """Points of each curve (filled: in the fit; hollow: left out) and the
    fitted line drawn over the fitted range only."""
    labels = curves["label"].tolist()
    color, shape, dash = _series_encodings("label", labels, curves[ELEMENT].tolist(), dark)
    data = points[points["fy"].notna() & points["fx"].notna()].rename(columns={PAPER: "paper", ELEMENT: "element"})
    data = data.assign(figure=data[FIGURE].fillna("not recorded"),
                       status=data["why"].where(~data["used"], "fitted"))
    data = data[["label", "fx", "fy", "used", "status", "paper", "figure", "element", "basis"]]
    lines = []
    for f in fits[fits["slope"].notna()].to_dict("records"):
        for xv in (f["x_min"], f["x_max"]):
            lines.append({"label": f["label"], "fx": xv, "fy": f["intercept"] + f["slope"] * xv,
                          "fit": f"slope {f['slope']:.2f} ± {f['slope_se']:.2f}, R² {f['r2']:.3f}, n {f['n_fit']}"})
    x = alt.X("fx:Q", title=x_title, scale=alt.Scale(zero=False))
    y = alt.Y("fy:Q", title=y_title, scale=alt.Scale(zero=False))
    tooltip = [
        alt.Tooltip("paper:N", title="Paper"), alt.Tooltip("figure:N", title="Figure/Table"),
        alt.Tooltip("element:N", title="Element"), alt.Tooltip("fx:Q", title=x_title, format=".3g"),
        alt.Tooltip("fy:Q", title="y", format=".3g"), alt.Tooltip("basis:N", title="y is"),
        alt.Tooltip("status:N", title="Point"),
    ]
    base = alt.Chart(data)
    used = base.transform_filter("datum.used").mark_point(
        filled=True, size=70, opacity=1, stroke=SURFACE[dark], strokeWidth=1).encode(
        x=x, y=y, color=color, shape=shape, tooltip=tooltip)
    left_out = base.transform_filter("!datum.used").mark_point(
        filled=False, size=70, opacity=0.8, strokeWidth=1.5).encode(x=x, y=y, color=color, shape=shape, tooltip=tooltip)
    layers = [used, left_out]
    if lines:
        fit_lines = alt.Chart(pd.DataFrame(lines)).mark_line(strokeWidth=2).encode(
            x=x, y=y, color=color, strokeDash=dash, detail="label:N",
            tooltip=[alt.Tooltip("label:N", title="Curve"), alt.Tooltip("fit:N", title="Fit")])
        layers.insert(0, fit_lines)
    return alt.layer(*layers).properties(height=height(labels))
