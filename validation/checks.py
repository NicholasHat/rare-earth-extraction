"""Automatic QA checks run after every extraction, before review (README §9).

All checks operate on the coerced DataFrame (validation/schema.py) plus the
captured text_endpoints, so they are pure and unit-testable without spending
API tokens. Each known failure mode maps to a check here:

  - silent under-extraction ("stopped at 2 endpoints")  -> row_count_sanity (RED at <= 2 points per curve)
  - one series digitised at a subset of its markers     -> sparse_sibling (AMBER)
  - a varied axis lost (sweep rows with equal conditions) -> repeated_x (RED)
  - stripping/scrubbing percentages in Extract%         -> treatment_y_column (RED)
  - points sampled off a fitted line, not digitised     -> sampled_from_line (RED)
  - derived columns that contradict each other          -> arithmetic (RED): RRE mM vs ppm, molar ratio,
                                                           and %E vs D at the row's phase ratio
  - two elements' values swapped in a separation pair   -> separation_factor (AMBER)
  - a feed stated as a total recorded per element       -> feed_total (AMBER)
  - axis calibration drift                              -> text_endpoint_cross_check / axis_bounds (RED)
  - OCR-garbled numeric tables                          -> schema_conformance (RED)
  - monochrome series merged/dropped                    -> row_count_sanity + duplicate_rows + monotonicity
  - in-plot text / another series' markers digitised in -> off_curve (AMBER)
  - vocabulary drift (incl. pH type)                    -> vocabulary / ph_type (AMBER)
  - rows that don't say which figure/table they're from -> provenance (AMBER)

Curve-shape checks work per curve (validation/curves.py — the definition the
Analysis page shares): one element in one figure/table under one set of fixed
conditions, along that figure's own x axis. Two experiments on the same
element, the same experiment re-plotted in another figure, and two series of
one figure that differ only in stirring speed are never pooled.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from calculator.atomic_mass import ATOMIC_MASS_G_PER_MOL, REE_ELEMENTS

from . import curves as curve_def
from . import vocab
from .curves import FEED_COLUMNS, X_COLUMNS, Curve
from .report import QAReport, Severity
from .schema import ELEMENT_COLUMN, FIGURE_COLUMN, PH_TYPE_COLUMN, PH_TYPES

# Tuning constants (README §11 item 4 — revisit against the first dozen papers).
SPARSE_RED_MAX_ROWS = 2      # this few points on a curve is the "stopped at the text's endpoints" failure (RED)
SIBLING_MIN_FRAC = 0.5       # a series with under this share of its fullest sibling's points (AMBER)
PH_TOL = 0.3                 # text-endpoint x-match tolerance on pH
PCT_TOL = 10.0               # text-endpoint y-mismatch tolerance on %-type metrics
CONC_TOL_RATIO = 2.0         # text-endpoint x-match tolerance on other x (a ratio: sweeps are log-spaced)
MONOTONICITY_NOISE = 5.0     # %E reversal smaller than this is treated as noise (also: repeated_x, plateau)
OFF_CURVE_FIT_PCT = (1.0, 97.0)  # only %E in this band carries slope information in log D; outside it a
                                 # digitised curve is a saturation plateau (or censored at the frame)
OFF_CURVE_INLIER_LOGD = 0.5      # |log D residual| for a point to count as ON the fitted line
OFF_CURVE_FLAG_LOGD = 0.8        # flag needs BOTH: this far off in log D (factor ~6 in D) ...
OFF_CURVE_FLAG_PCT = 25.0        # ... and this far off in %E (so a 1 % vs 3 % scatter never trips it)
OFF_CURVE_MIN_INLIER_FRAC = 0.5  # a line must explain at least this share of the band (and >= 4 points)
PLATEAU_POINTS = 3               # a curve whose last this-many points sit within MONOTONICITY_NOISE
                                 # inside the band has stopped rising: not a log D line
SAMPLED_MIN_POINTS = 6           # sampled_from_line: judged only on curves with this many points in the band
SAMPLED_MAX_SPACING_CV = 0.03    # pH steps this even (std / mean) ...
SAMPLED_MAX_RESID_LOGD = 0.01    # ... AND every point this close to one log D line => sampled, not digitised
ARITH_REL_TOL = 0.02             # derived columns must agree with the columns they derive from within this
ARITH_PCT_ABS_TOL = 0.5          # %E vs D: a %E printed as 100 for D = 500 is rounding, not an error
FEED_TOTAL_MIN_ELEMENTS = 3      # this many elements sharing one feed mM, with no ppm, is ambiguous
SF_TOL_RATIO = 1.25              # a separation factor and the β its pair's %E/D give agree within this

SF_COLUMN = "Separation factor (SF%)"
# `Treatment` of a back-extraction row: `Stripping` / `Scrubbing` (not "Extraction + Stripping").
_BACK_EXTRACTION = re.compile(r"^\s*(?:strip|scrub)", re.IGNORECASE)
_ELEMENT_GROUPS = re.compile(r"^(L|M|H|T)?REEs?$|total", re.IGNORECASE)
# "SF Dy/Nd" in Material Process: the pair a row's separation factor is for, numerator first.
_SF_PAIR = re.compile(r"\bSF\s*([A-Z][a-z]?)\s*/\s*([A-Z][a-z]?)\b")

# Map a text-endpoint y_metric / x_basis to the schema column it lives in.
# extraction_v16+ writes the schema column name itself; the aliases are what
# v13–v15 wrote.
_Y_METRIC_TO_COL = {
    "Extract%": "Extract%",
    "extract%": "Extract%",
    "Recovery %": "Recovery %",
    "recovery %": "Recovery %",
    "recovery": "Recovery %",
}
_X_BASIS_TO_COL = {c.casefold(): c for c in X_COLUMNS} | {
    "extractant_conc_mm": "Extractant Conc. (mM)",
    "extractant_conc": "Extractant Conc. (mM)",
}


def run(
    df: pd.DataFrame,
    text_endpoints: list[dict] | None = None,
    *,
    figure_is_curve: bool = True,
    coercion_failures: int = 0,
    deterministic_counts: list[int] | None = None,
) -> QAReport:
    """Run all checks and return a QAReport.

    `figure_is_curve` says whether the source figure is a multi-point curve
    (vs. a single-condition table) — it gates the sparse-result check.
    `coercion_failures` is the count of cells that were non-null in the model
    output but failed to parse as numbers (the OCR-garbled signal).
    `deterministic_counts` are the authoritative per-series marker counts from
    the curve pre-pass (extraction/curve_prepass.py); when present they drive a
    cross-check that the model didn't under-digitise vs the real markers.
    """
    text_endpoints = text_endpoints or []
    report = QAReport()

    _schema_conformance(df, coercion_failures, report)
    if len(df) == 0:
        return report  # nothing else is meaningful on an empty extraction

    cs = curve_def.curves(df)
    _element_scope(df, report)
    _treatment_y_column(df, report)
    _short_curves(df, cs, figure_is_curve, report)
    _repeated_x(df, cs, report)
    _axis_bounds(df, report)
    _monotonicity(df, cs, report)
    _off_curve(df, cs, report)
    _sampled_from_line(df, cs, report)
    _arithmetic(df, report)
    _distribution_ratio(df, report)
    _separation_factor(df, report)
    _feed_total(df, report)
    _duplicate_rows(df, report)
    _vocabulary(df, report)
    _ph_type(df, report)
    _provenance(df, report)
    _text_endpoint_cross_check(df, text_endpoints, report)
    _deterministic_curve_count(df, deterministic_counts, report)
    return report


def _schema_conformance(df: pd.DataFrame, coercion_failures: int, report: QAReport) -> None:
    if len(df) == 0:
        report.add(
            "schema_conformance",
            Severity.RED,
            "Extraction returned 0 rows — nothing to review.",
        )
        return
    if coercion_failures > 0:
        report.add(
            "schema_conformance",
            Severity.RED,
            f"{coercion_failures} numeric cell(s) could not be parsed as numbers "
            "(possible OCR garble or wrong units) — they were stored as null.",
        )


def _element_groups(df: pd.DataFrame):
    """Yield (element_label, sub_df) for each non-empty element series."""
    if ELEMENT_COLUMN not in df.columns:
        return
    for label, sub in df.groupby(df[ELEMENT_COLUMN].fillna("(unspecified)")):
        yield str(label), sub


def _row_numbers(df: pd.DataFrame, index) -> np.ndarray:
    """1-based positions in `df` — the row numbers the reviewer sees in the
    review editor and in the exported CSV."""
    return df.index.get_indexer(index) + 1


def _rows(rows) -> str:
    rows = [int(r) for r in rows]
    shown = ", ".join(str(r) for r in rows[:8])
    return shown + (f", … ({len(rows)} total)" if len(rows) > 8 else "")


def _is_back_extraction(df: pd.DataFrame) -> pd.Series:
    """Stripping / scrubbing rows: their percentage is a back-extraction."""
    if "Treatment" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["Treatment"].astype("string").str.contains(_BACK_EXTRACTION, na=False).astype(bool)


def _y_column(rows: pd.DataFrame) -> str | None:
    """The percentage a curve plots: Extract%, else Recovery %."""
    for c in ("Extract%", "Recovery %"):
        if c in rows.columns and pd.to_numeric(rows[c], errors="coerce").notna().any():
            return c
    return None


def _curve_points(df: pd.DataFrame, curve: Curve) -> pd.DataFrame:
    """A curve's numeric (x, y) points sorted by x, with `row` giving each
    point's 1-based position in `df`."""
    y = _y_column(curve.rows)
    if curve.x is None or y is None:
        return pd.DataFrame(columns=["x", "y", "row"])
    s = pd.DataFrame({"x": pd.to_numeric(curve.rows[curve.x], errors="coerce"),
                      "y": pd.to_numeric(curve.rows[y], errors="coerce")}).dropna()
    s["row"] = _row_numbers(df, s.index)
    return s.sort_values("x", kind="stable")


def _is_figure(figure) -> bool:
    """Rows digitised from a figure (or of unknown source) — not a table or
    the prose, which state their values and have no markers to miss."""
    return figure is None or bool(re.match(r"\s*fig", str(figure), re.IGNORECASE))


def _short_curves(df: pd.DataFrame, cs: list[Curve], figure_is_curve: bool, report: QAReport) -> None:
    """Curves of a figure with too few points, along the figure's x axis.

    row_count_sanity (RED): two points — the "stopped at the text's stated
    endpoints" failure. Per curve there is no absolute "too few" beyond that:
    real series have 4–7 markers as often as 10–20 (most curves of every
    paper on the 2026-11 trial), and padding a short series is the failure
    sampled_from_line catches.

    sparse_sibling (AMBER): under half the points of the fullest series of
    the same figure. Series of one figure share its x values, so one that
    stops short was digitised at a subset of its markers (the trial: Ce 1 of
    5 kinetics points while La/Pr/Nd/Y had all 5).

    A figure with no x axis (one bar per extractant, one point per element)
    has no curves to be short, and a lone point in a figure that names no
    source is not a curve — it is one point of a sweep along another axis."""
    by_figure: dict[str, list[tuple[Curve, int, np.ndarray]]] = {}
    red = []
    for c in cs:
        if c.x is None or not _is_figure(c.figure):
            continue
        s = _curve_points(df, c)
        n = s["x"].nunique()
        if figure_is_curve and 2 <= n <= SPARSE_RED_MAX_ROWS:
            red.append(f"{c.label} ({n})")
        elif c.figure is not None:   # rows that name no figure can't be known to share one
            by_figure.setdefault(str(c.figure), []).append((c, n, s["row"].to_numpy()))
    if red:
        report.add(
            "row_count_sanity",
            Severity.RED,
            f"{len(red)} curve(s) have only {SPARSE_RED_MAX_ROWS} or fewer digitized points along their "
            f"x axis — {'; '.join(red[:6])}{' …' if len(red) > 6 else ''} — the signature of stopping "
            "at the text's stated endpoints instead of digitizing the figure's curve.",
        )
    for members in by_figure.values():
        full = max(n for _, n, _ in members)
        short = [(c, n) for c, n, _ in members if n < SIBLING_MIN_FRAC * full]
        if short:
            report.add(
                "sparse_sibling",
                Severity.AMBER,
                f"{'; '.join(f'{c.label} has {n} point(s)' for c, n in short)} — against {full} on the "
                "fullest series of the same figure. Check whether markers were skipped (hidden behind "
                "another series, or only some x values digitised) and add them, or say in `figures` why "
                "they are missing.",
            )


def _repeated_x(df: pd.DataFrame, cs: list[Curve], report: QAReport) -> None:
    """Rows of one curve at the same x — every recorded condition equal — whose
    percentages clearly differ. Something that varied between them has no
    column: the sweep's own axis was lost (the trial: a Cyanex 923 wt% sweep
    recorded as 12 rows with identical conditions and %E from 8 to 71)."""
    hits: list[tuple[Curve, np.ndarray, float, float]] = []
    for c in cs:
        y = _y_column(c.rows)
        if y is None:
            continue
        ys = pd.to_numeric(c.rows[y], errors="coerce")
        # A blank x is a value like any other here: rows of a sweep that all lack it are the failure.
        at = pd.to_numeric(c.rows[c.x], errors="coerce").fillna(-np.inf) if c.x \
            else pd.Series(0.0, index=c.rows.index)
        frame = pd.DataFrame({"x": at, "y": ys}).dropna()
        spread = frame.groupby("x")["y"].agg(lambda v: v.max() - v.min())
        bad_x = spread.index[spread > MONOTONICITY_NOISE]
        if len(bad_x):
            idx = frame.index[frame["x"].isin(bad_x)]
            sel = frame.loc[idx, "y"]
            hits.append((c, _row_numbers(df, idx), float(sel.min()), float(sel.max())))
    if not hits:
        return
    rows = np.concatenate([r for _, r, _, _ in hits])
    names = "; ".join(f"{c.label}: {lo:g}–{hi:g}" for c, _, lo, hi in hits[:6])
    more = f" (and {len(hits) - 6} more)" if len(hits) > 6 else ""
    report.add(
        "repeated_x",
        Severity.RED,
        f"{len(rows)} row(s) share every recorded condition with another row of their curve but "
        f"report clearly different values — {names}{more}. The varied quantity has no column on "
        f"these rows (rows {_rows(sorted(rows))}): record it (convert it to a schema column), or "
        "give the sweep no rows and say why in `figures`.",
    )


def _treatment_y_column(df: pd.DataFrame, report: QAReport) -> None:
    """A back-extraction percentage lives in `Recovery %`, an extraction one in
    `Extract%`: a stripping row with Extract% (the trial: one paper's 70
    stripping rows) would be read as an extraction by every consumer."""
    back = _is_back_extraction(df)
    pct, rec = _num(df, "Extract%").notna(), _num(df, "Recovery %").notna()
    for bad, what in (
        (back & pct, "are Stripping/Scrubbing rows with `Extract%` filled — a back-extraction "
                     "percentage belongs in `Recovery %`, with `Extract%` blank"),
        (~back & rec & ~pct, "are extraction rows with only `Recovery %` — an extraction percentage "
                             "belongs in `Extract%`; if these are stripping or scrubbing, say so in `Treatment`"),
    ):
        if bad.any():
            rows = _row_numbers(df, df.index[bad])
            # Rows named in the message, not as droppable: the values are right, their column is not.
            report.add("treatment_y_column", Severity.RED, f"{len(rows)} row(s) {what}: {_rows(rows)}.")


def _deterministic_curve_count(
    df: pd.DataFrame, deterministic_counts: list[int] | None, report: QAReport
) -> None:
    """Cross-check the model's per-element row counts against the authoritative
    per-series marker counts the deterministic pre-pass found in the PDF geometry.

    Conservative on purpose: the pre-pass counts are per *figure series* and the
    DataFrame is per *element* (one element may appear across several figures), so
    we compare sorted-descending and only flag when the model falls materially
    short (< 80%) of a known count — a robust signal of under-digitisation that
    doesn't penalise the fuzzy figure↔element correspondence. AMBER, not RED,
    because the multi-figure mapping is approximate.
    """
    if not deterministic_counts:
        return
    llm_counts = sorted((len(sub) for _, sub in _element_groups(df)), reverse=True)
    det = sorted(deterministic_counts, reverse=True)
    short = []
    for i, dc in enumerate(det):
        lc = llm_counts[i] if i < len(llm_counts) else 0
        if lc < 0.8 * dc:
            short.append((dc, lc))
    if short:
        report.add(
            "deterministic_curve_count",
            Severity.AMBER,
            f"Deterministic geometry found series with marker counts {det}, but the "
            f"model's largest per-element counts {llm_counts[:len(det)]} fall short on "
            f"{len(short)} of them (e.g. expected ~{short[0][0]}, got {short[0][1]}). "
            "Likely under-digitisation — verify those curves captured every point.",
        )


def _axis_bounds(df: pd.DataFrame, report: QAReport) -> None:
    bounds = {
        "Extract%": (0.0, 100.0),
        "Recovery %": (0.0, 100.0),
        "pH": (-1.0, 14.0),
        SF_COLUMN: (0.0, math.inf),
        # A negative "D" is a log D that was never converted (D = 10^log D).
        "Distribution ratio (D)": (0.0, math.inf),
        "Phase ratio (O/A)": (1e-9, math.inf),
    }
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = df[(vals < lo) | (vals > hi)]
        if len(bad) > 0:
            example = pd.to_numeric(bad[col], errors="coerce").dropna()
            sample = "" if example.empty else f" (e.g. {example.iloc[0]})"
            report.add(
                "axis_bounds",
                Severity.RED,
                f"{len(bad)} value(s) of '{col}' fall outside the plausible range "
                f"[{lo}, {hi}]{sample} — likely axis calibration drift.",
            )


def _monotonicity(df: pd.DataFrame, cs: list[Curve], report: QAReport) -> None:
    """A curve that zig-zags — rises and falls beyond noise, changing direction
    at least twice — has misread points or two series merged into it. One
    change of direction is a maximum, which is chemistry (a neutral
    extractant against acid concentration, a kinetics curve whose emulsion
    breaks), not a misread."""
    for c in cs:
        s = _curve_points(df, c)
        if len(s) < 4:
            continue
        deltas = s["y"].diff()
        moves = np.sign(deltas[deltas.abs() > MONOTONICITY_NOISE].to_numpy())
        ups, downs = int((moves > 0).sum()), int((moves < 0).sum())
        reversals = int((np.diff(moves) != 0).sum())
        if ups >= 2 and downs >= 2 and reversals >= 2:
            report.add(
                "monotonicity",
                Severity.AMBER,
                f"{c.label}: the curve along {c.x} zig-zags "
                f"({ups} rises, {downs} falls beyond noise; falls land on row(s) "
                f"{_rows(s.loc[deltas < -MONOTONICITY_NOISE, 'row'])}) — check for "
                "misread points or two series merged into one.",
            )


def _log_d(pct: np.ndarray) -> np.ndarray:
    """%E -> log10 of the distribution ratio (callers keep %E strictly inside 0..100)."""
    return np.log10(pct / (100.0 - pct))


def _off_curve_points(x: np.ndarray, pct: np.ndarray) -> tuple[list[int], int]:
    """Indices of points far off the straight line that the majority of the
    curve lies on, plus how many points that line explains (0 => no such line).

    log D is linear in equilibrium pH for the cation-exchange systems this
    database covers — that linearity is why papers plot it. So the curve is
    the line through the most points (every pair proposes one; deterministic,
    n is tens of points), refined by least squares on its inliers. A point
    well off that line is not on the curve, whatever its neighbours look like
    — which is what defeats local tests: a panel title digitised as three
    adjacent markers, or an artifact sitting right next to a real point.

    Two guards keep this honest on real curves. Only %E inside
    OFF_CURVE_FIT_PCT takes part: above it a digitised curve is a saturation
    plateau (98 % for two pH units), which is not a line in log D and would
    otherwise out-vote the rising part. And a flag needs the point to be far
    off in %E as well as in log D, because near the ends of the band a small
    %E scatter is a large log D one. Nothing is reported for a curve no line
    explains (OFF_CURVE_MIN_INLIER_FRAC): non-linearity is not evidence of
    artifacts."""
    lo, hi = OFF_CURVE_FIT_PCT
    band = np.flatnonzero((pct >= lo) & (pct <= hi))
    if len(band) < 5:
        return [], 0
    xb, yb = x[band], _log_d(pct[band])
    n = len(band)
    best_inliers: np.ndarray | None = None
    best_score = (0, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            if xb[j] == xb[i]:
                continue
            slope = (yb[j] - yb[i]) / (xb[j] - xb[i])
            resid = np.abs(yb - (yb[i] + slope * (xb - xb[i])))
            inliers = resid <= OFF_CURVE_INLIER_LOGD
            score = (int(inliers.sum()), -float(resid[inliers].sum()))
            if score > best_score:
                best_score, best_inliers = score, inliers
    if best_inliers is None or best_inliers.sum() < max(4, OFF_CURVE_MIN_INLIER_FRAC * n):
        return [], 0
    slope, intercept = np.polyfit(xb[best_inliers], yb[best_inliers], 1)
    line_logd = intercept + slope * xb
    line_pct = 100.0 / (1.0 + 10.0 ** (-line_logd))
    off = (np.abs(yb - line_logd) > OFF_CURVE_FLAG_LOGD) & (np.abs(pct[band] - line_pct) > OFF_CURVE_FLAG_PCT)
    return [int(i) for i in band[off]], int(best_inliers.sum())


def _plateaus(pct: np.ndarray) -> bool:
    """The curve's last PLATEAU_POINTS points (by x) sit level inside the fit
    band: extraction stopped rising below saturation (a loading limit, an
    initial-pH curve) — log D is no straight line there, so off_curve has no
    line to judge against. Saturation near 100 % is outside the band and
    already ignored."""
    tail = pct[-PLATEAU_POINTS:]
    return (len(tail) == PLATEAU_POINTS and tail.max() - tail.min() <= MONOTONICITY_NOISE
            and tail.max() <= OFF_CURVE_FIT_PCT[1])


def _log_d_line_curves(df: pd.DataFrame, cs: list[Curve], *, equilibrium_only: bool):
    """(curve, points) for curves along pH — the axis log D is linear in.
    `equilibrium_only` drops initial-pH curves (the line holds for the
    equilibrium pH; rows that don't say which kind are kept)."""
    for c in cs:
        if c.x != "pH" or _y_column(c.rows) != "Extract%":
            continue
        if equilibrium_only and PH_TYPE_COLUMN in c.rows.columns \
                and str(c.rows[PH_TYPE_COLUMN].iloc[0]).strip().lower() == "initial":
            continue
        yield c, _curve_points(df, c)


def _off_curve(df: pd.DataFrame, cs: list[Curve], report: QAReport) -> None:
    """Points that sit far off their own curve. Extraction curves don't do that;
    in-plot text does — a panel title or annotation digitised as a marker lands
    at whatever %E its pixels sit at (seen live: Quinn 2015's panel titles at
    log D ≈ 0.7 became runs of 83–85 %E points in the Lu series, three of them
    side by side). So does a neighbouring series' marker assigned to this one.
    Judged only on equilibrium-pH sweeps that keep rising (_plateaus): on the
    2026-11 trial, initial-pH curves levelling off at 35 % were called off-curve."""
    for c, s in _log_d_line_curves(df, cs, equilibrium_only=True):
        x, pct = s["x"].to_numpy(float), s["y"].to_numpy(float)
        if _plateaus(pct):
            continue
        off, n_on = _off_curve_points(x, pct)
        if off:
            rows = s["row"].to_numpy()[off]
            report.add(
                "off_curve",
                Severity.AMBER,
                f"{c.label}: {len(rows)} point(s) at row(s) {_rows(rows)} lie > "
                f"{OFF_CURVE_FLAG_LOGD:g} decades in log D off the straight line through "
                f"the other {n_on} points of this curve — typical of in-plot text "
                "(panel titles, annotations) or another series' markers digitised into "
                "this one. Delete unless the figure really shows them.",
                rows=rows,
            )


def _sampled_from_line(df: pd.DataFrame, cs: list[Curve], report: QAReport) -> None:
    """Curves whose points were generated rather than digitised: evenly spaced
    in pH AND lying on one straight log D line more exactly than reading
    markers off a figure can (a pixel is ~0.01 log D on a typical panel, and
    real digitised series scatter ~0.07 around their line). Seen live: a
    Quinn et al. 2015 extraction (extraction_v10) returned 13 points per
    series at 0.1-pH steps, sampled off the fitted lines, where the figure has
    5-7 markers — spacing CV 0.005 and max residual 0.0015 log D, against
    >= 0.4 and ~0.07 for real series. Either condition alone is innocent
    (experiments are often run at even pH setpoints; a clean series is close
    to linear); both together are not."""
    lo, hi = OFF_CURVE_FIT_PCT
    for c, s in _log_d_line_curves(df, cs, equilibrium_only=False):
        x, pct = s["x"].to_numpy(float), s["y"].to_numpy(float)
        band = (pct >= lo) & (pct <= hi)
        if band.sum() < SAMPLED_MIN_POINTS:
            continue
        steps = np.diff(x)
        if steps.mean() <= 0 or steps.std() / steps.mean() > SAMPLED_MAX_SPACING_CV:
            continue
        xb, yb = x[band], _log_d(pct[band])
        resid = np.abs(yb - np.polyval(np.polyfit(xb, yb, 1), xb))
        if resid.max() <= SAMPLED_MAX_RESID_LOGD:
            rows = s["row"].to_numpy()
            report.add(
                "sampled_from_line",
                Severity.RED,
                f"{c.label}: all {len(rows)} points are evenly spaced in pH and sit on one "
                f"log D line within {resid.max():.4f} decades — generated from a fitted "
                "line, not digitised from the figure's markers. Check the figure's real "
                "marker count; delete and re-digitise.",
                rows=rows,
            )


def _element_scope(df: pd.DataFrame, report: QAReport) -> None:
    """The database holds rare earth elements (La–Lu, Y, Sc) only; a row for
    any other metal is out of scope (the 2026-10-07 batch extracted 84 rows
    from a Pb paper), and so is a row for an element group (the 2026-11 trial
    extracted 106 LREE/MREE/HREE rows)."""
    if ELEMENT_COLUMN not in df.columns:
        return
    elements = df[ELEMENT_COLUMN].astype("string").str.strip()
    bad = elements.notna() & ~elements.isin(REE_ELEMENTS)
    if bad.any():
        rows = _row_numbers(df, df.index[bad])
        names = sorted(set(elements[bad]))
        groups = any(_ELEMENT_GROUPS.search(n) for n in names)
        report.add(
            "element_scope",
            Severity.RED,
            f"{len(rows)} row(s) are for {', '.join(names)} — not a rare earth element "
            f"(La–Lu, Y, Sc); this database holds single rare earths only"
            + (" (an element group's values get no rows)" if groups else "")
            + f": {_rows(rows)}.",
            rows=rows,
        )


def _rounding_half_unit(values: pd.Series) -> pd.Series:
    """Half a unit in the last decimal each value is written to (0.2 -> 0.05,
    0.178 -> 0.0005): how far a correctly rounded value may sit from the
    exact one. A whole number gets no allowance — "1" for 0.72 mM, or a ratio
    of 9 for 8.79, could be rounding or an error, and the prompt asks for
    three significant figures, so it stays flagged."""
    def half(v):
        if pd.isna(v):
            return np.nan
        text = f"{float(v):.10g}"
        if "." not in text or "e" in text:
            return 0.0
        return 0.5 * 10.0 ** -len(text.split(".")[1])
    return values.map(half)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def _arithmetic(df: pd.DataFrame, report: QAReport) -> None:
    """Derived columns must agree with what they derive from, row by row:
    RRE mM = ppm / the element's atomic mass, and molar ratio = extractant
    mM / RRE mM (per element — the prompt's Step 8). The model computes both;
    a row where they disagree has at least one wrong number. A value may sit
    off by its own rounding — a ratio of 0.178 written as 0.2 is not an error
    (four papers in the 2026-10-07 batch were flagged red for exactly that)."""
    mass = df[ELEMENT_COLUMN].map(ATOMIC_MASS_G_PER_MOL) if ELEMENT_COLUMN in df.columns else None
    if mass is None:
        return
    ppm, mm = _num(df, "RRE composition (ppm)"), _num(df, "RRE composition (mM)")
    conc, ratio = _num(df, "Extractant Conc. (mM)"), _num(df, "Molar ratio of EX/REE")
    for check, got, expected, what in (
        ("mM", mm, ppm / mass, "RRE composition (mM) ≠ ppm ÷ atomic mass"),
        ("ratio", ratio, conc / mm, "Molar ratio of EX/REE ≠ Extractant Conc. (mM) ÷ RRE composition (mM)"),
    ):
        allowed = np.maximum(ARITH_REL_TOL * expected.abs(), _rounding_half_unit(got))
        bad = ((got - expected).abs() > allowed) & got.notna() & expected.notna()
        if bad.any():
            rows = _row_numbers(df, df.index[bad])
            report.add(
                "arithmetic",
                Severity.RED,
                f"{what} on {len(rows)} row(s): {_rows(rows)} — one of those numbers is wrong.",
                rows=rows,
            )


def _distribution_ratio(df: pd.DataFrame, report: QAReport) -> None:
    """%E, D and the phase ratio on one row must agree: %E = 100·D/(D + A/O),
    with A/O = 1/'Phase ratio (O/A)' (for a solid sorbent O/A is m/V in g/mL
    and D is in mL/g, so the same formula holds). This is the conversion the
    v13 resin paper got wrong (100·D/(1+D) at V/m = 100 mL/g: up to 5x high).
    Judged only where all three are present, on extraction rows (a stripping
    D is the paper's own quantity, not this one); D's printed rounding and 2 %
    either side are allowed, and %E to its own rounding or ARITH_PCT_ABS_TOL."""
    d, oa = _num(df, "Distribution ratio (D)"), _num(df, "Phase ratio (O/A)")
    pct = _num(df, "Extract%")
    present = d.notna() & oa.notna() & pct.notna() & (d >= 0) & (oa > 0) & ~_is_back_extraction(df)
    if not present.any():
        return
    spread = np.maximum(_rounding_half_unit(d).fillna(0), ARITH_REL_TOL * d)
    d_lo, d_hi = (d - spread).clip(lower=0), d + spread
    a_over_o = 1.0 / oa
    lo, hi = 100 * d_lo / (d_lo + a_over_o), 100 * d_hi / (d_hi + a_over_o)
    tol = np.maximum(_rounding_half_unit(pct).fillna(0), ARITH_PCT_ABS_TOL)
    bad = present & ((pct < lo - tol) | (pct > hi + tol))
    if bad.any():
        rows = _row_numbers(df, df.index[bad])
        first = bad.idxmax()
        expected = 100 * d[first] / (d[first] + a_over_o[first])
        report.add(
            "arithmetic",
            Severity.RED,
            f"Extract% ≠ 100·D/(D + A/O) at the row's phase ratio on {len(rows)} row(s): {_rows(rows)} "
            f"(e.g. D {d[first]:g}, O/A {oa[first]:g} gives {expected:.3g} %, row says {pct[first]:g}) — "
            "a D converted with the wrong phase ratio, or a log D stored as D.",
            rows=rows,
        )


def _ratio(a: float, b: float) -> float:
    """How far apart two positive magnitudes are, as a ratio >= 1 (inf if signs differ or one is zero)."""
    if a == b:
        return 1.0
    if a <= 0 or b <= 0:
        return math.inf
    return max(a / b, b / a)


def _separation_factor(df: pd.DataFrame, report: QAReport) -> None:
    """A separation factor next to its pair's own numbers: two elements
    measured together (same figure, x and conditions) give β = D_a / D_b
    (from reported D, else from %E — the shared phase ratio cancels). The
    flag is narrow on purpose: a row whose SF matches β only with the two
    elements exchanged, against the direction the figure's other points (or
    a stated "SF Dy/Nd" in Material Process) set — values swapped between the
    elements (the trial: Y 98.2 / Er 97.0 under SF 1.537, where every other
    point of the figure had Er above Y). An SF that fits neither direction is
    not flagged: on the trial that was a paper whose own plotted β disagrees
    with its plotted %E, faithfully copied."""
    sf = _num(df, SF_COLUMN)
    if not sf.notna().any() or ELEMENT_COLUMN not in df.columns:
        return
    pct, d = _num(df, "Extract%"), _num(df, "Distribution ratio (D)")
    ratio = (pct / (100.0 - pct)).where((pct > 0) & (pct < 100))
    key = [c for c in curve_def.curve_key(None) if c in df.columns and c not in (ELEMENT_COLUMN, *FEED_COLUMNS)]
    pairs = []   # (figure, a, b, sf, beta_a_over_b, stated (num, den) or None, row indices)
    for _, g in df.groupby(key, dropna=False, sort=False):
        g_sf = sf[g.index].dropna()
        if g_sf.empty:
            continue
        stated = None
        if "Material Process" in g.columns:
            m = next((_SF_PAIR.search(str(v)) for v in g.loc[g_sf.index, "Material Process"]
                      if _SF_PAIR.search(str(v))), None)
            stated = m.groups() if m else None
        els = g[ELEMENT_COLUMN].astype("string").str.strip()
        names = sorted(set(stated) if stated else set(els.dropna()))
        if len(names) != 2:
            continue
        ia, ib = (g.index[(els == n).fillna(False).to_numpy(bool)] for n in names)
        if len(ia) != 1 or len(ib) != 1:
            continue
        both_d = d[ia[0]] > 0 and d[ib[0]] > 0
        da, db = (d[ia[0]], d[ib[0]]) if both_d else (ratio[ia[0]], ratio[ib[0]])
        if not (da > 0 and db > 0):
            continue
        for v in g_sf.unique():
            pairs.append((g[FIGURE_COLUMN].iloc[0] if FIGURE_COLUMN in g.columns else None,
                          names[0], names[1], float(v), da / db, stated, (ia[0], ib[0])))
    flagged = []
    for fig, a, b, v, beta, stated, idx in pairs:
        fits_ab, fits_ba = bool(_ratio(v, beta) <= SF_TOL_RATIO), bool(_ratio(v, 1.0 / beta) <= SF_TOL_RATIO)
        if stated:
            direction = "ab" if stated == (a, b) else "ba"
        else:   # the direction the same figure's other unambiguous points of this pair agree on
            votes = [int(_ratio(w, bt) <= SF_TOL_RATIO) - int(_ratio(w, 1.0 / bt) <= SF_TOL_RATIO)
                     for f2, a2, b2, w, bt, _, i2 in pairs if (f2, a2, b2) == (fig, a, b) and i2 != idx]
            net = sum(votes)
            direction = "ab" if net > 0 else "ba" if net < 0 else None
        if direction == "ab" and fits_ba and not fits_ab:
            flagged.append((idx, a, b, v, beta, a, b))
        elif direction == "ba" and fits_ab and not fits_ba:
            flagged.append((idx, a, b, v, 1.0 / beta, b, a))
    if flagged:
        rows = sorted({int(r) for idx, *_ in flagged for r in _row_numbers(df, list(idx))})
        (_, _, _, v, beta, num, den) = flagged[0]
        report.add(
            "separation_factor",
            Severity.AMBER,
            f"{len(flagged)} separation factor(s) contradict their pair's own values: e.g. SF {v:g} "
            f"where the {num}/{den} ratio of D (or %E) on the same rows is {beta:.3g} — they fit only with "
            f"the two elements' values exchanged, against the direction the figure's other points (or the "
            f"stated pair) set. Check which element each value belongs to (rows {_rows(rows)}).",
        )


def _feed_total(df: pd.DataFrame, report: QAReport) -> None:
    """The same RRE mM on several elements with no ppm: either the paper gave
    "x mM each", or a *total* ("1.9 mM total RE") was copied onto every
    element instead of divided among them. Seen live: extraction_v10 on Quinn
    et al. 2015 put the 1.9 mM total on each of Ce/Pr/Nd/Sm (0.475 mM each).
    Per element in ppm (a common "0.1 g/L each") gives each element a
    different mM, so it never trips this."""
    if ELEMENT_COLUMN not in df.columns:
        return
    mm, ppm = _num(df, "RRE composition (mM)"), _num(df, "RRE composition (ppm)")
    sub = df[mm.notna() & ppm.isna()]
    if sub.empty:
        return
    by_value = sub.groupby(mm[sub.index].round(4))[ELEMENT_COLUMN].agg(lambda e: sorted(set(e)))
    for value, elements in by_value.items():
        if len(elements) >= FEED_TOTAL_MIN_ELEMENTS:
            rows = _row_numbers(df, sub.index[mm[sub.index].round(4) == value])
            report.add(
                "feed_total",
                Severity.AMBER,
                f"RRE composition (mM) = {value:g} on {len(elements)} elements ({', '.join(elements)}) "
                "with no ppm — check the paper states it per element; a total "
                f"feed belongs in 'Total REE feed (mM)' (an even split would be {value / len(elements):.4g} mM each).",
                rows=rows,
            )


def _duplicate_rows(df: pd.DataFrame, report: QAReport) -> None:
    """A row that repeats an earlier row of the same curve — same element,
    figure/table and every condition, same values — adds nothing: the mark
    of a digitising loop, and safe to drop by row.

    Identical (element, pH, %E) across *different* curves is not flagged: on
    the 2026-10 trial every such case was real data — a concentration sweep's
    below-detection 0.02 %E at every concentration, a kinetics plateau, two
    synergist concentrations with the same tabulated %E, or one measurement
    plotted in two figures."""
    if ELEMENT_COLUMN not in df.columns:
        return
    values = [c for c in ("Extract%", "Recovery %", "Distribution ratio (D)") if c in df.columns]
    key = [c for c in curve_def.curve_key(None) if c in df.columns] + values
    measured = df[values].notna().any(axis=1)
    repeats = df.duplicated(subset=key, keep="first") & measured
    rows = [int(i) + 1 for i in np.flatnonzero(repeats.to_numpy())]
    if rows:
        report.add(
            "duplicate_rows",
            Severity.AMBER,
            f"{len(rows)} row(s) repeat an earlier row of the same curve (same element, "
            f"figure/table, conditions and values) — a digitizing loop or copy error; "
            f"row(s) {_rows(rows)} can be dropped.",
            rows=rows,
        )


def _vocabulary(df: pd.DataFrame, report: QAReport) -> None:
    for field in ("Extractant type", "mixing method"):
        if field not in df.columns:
            continue
        novel = vocab.unknown_values(field, df[field].tolist())
        if novel:
            report.add(
                "vocabulary",
                Severity.AMBER,
                f"New '{field}' value(s) not seen before: {', '.join(novel)} — "
                "confirm these are legitimate and not typos.",
            )


def _ph_type(df: pd.DataFrame, report: QAReport) -> None:
    """`pH type` is a two-word vocabulary: an equilibrium-pH curve and an
    initial-pH curve of the same system are not comparable, so a value that
    is neither would silently fall out of every comparison."""
    if PH_TYPE_COLUMN not in df.columns:
        return
    values = df[PH_TYPE_COLUMN].dropna().astype(str).str.strip()
    odd = sorted({v for v in values if v and v.lower() not in PH_TYPES})
    if odd:
        report.add(
            "ph_type",
            Severity.AMBER,
            f"'pH type' must be {' or '.join(PH_TYPES)} (blank when the paper doesn't say); "
            f"found {', '.join(repr(v) for v in odd)}.",
        )


def _provenance(df: pd.DataFrame, report: QAReport) -> None:
    """Every row should say which figure or table it came from: that is how a
    reviewer traces it back, and part of what identifies a curve."""
    if FIGURE_COLUMN not in df.columns:
        return
    missing = df[FIGURE_COLUMN].isna() | (df[FIGURE_COLUMN].astype("string").str.strip() == "")
    if missing.any():
        n = int(missing.sum())
        report.add(
            "provenance",
            Severity.AMBER,
            f"{'All' if n == len(df) else n} of {len(df)} row(s) don't say which figure or table they came "
            "from ('Figure/Table'), so they can't be traced back to the paper or told apart by source.",
        )


def _norm_name(value) -> str:
    """Spelling-normalised name: 'Cyanex272' == 'cyanex 272'."""
    return "".join(str(value).lower().split())


def _text_endpoint_cross_check(
    df: pd.DataFrame, text_endpoints: list[dict], report: QAReport
) -> None:
    """Compare each of the paper's stated (x, y) claims with the digitised rows
    of that element (and extractant, when the endpoint names one).

    `x_basis` must name the schema column of the x (any of X_COLUMNS); an
    endpoint on any other basis can't be compared, which is reported rather
    than skipped (the trial wrote ~10 invented bases). So is one whose x is
    its own y value. A statement with no x (`x_value` null) has nothing to
    compare.

    A statement is only a merge gate (RED) when it can be pinned to one curve.
    Text endpoints carry no loading, figure or synergist, so when the element
    has several curves along that x (several extractants, loadings, figures)
    the statement may describe a curve that wasn't digitised or another
    condition — on the 2026-10 trial "about 70 % for Cyanex 272 at pH 4.5–4.75"
    was compared with a different resin loading's 96 % — and a mismatch is
    reported as a warning (AMBER) instead."""
    unknown_bases: list[str] = []
    x_is_y: list[str] = []
    for ep in text_endpoints:
        x_val, y_val = ep.get("x_value"), ep.get("y_value")
        if x_val is None:
            continue
        basis = str(ep.get("x_basis", "")).strip()
        x_col = _X_BASIS_TO_COL.get(basis.casefold())
        if x_col is None:
            if basis not in unknown_bases:
                unknown_bases.append(basis)
            continue
        element = ep.get("element")
        try:
            x_val, y_val = float(x_val), (None if y_val is None else float(y_val))
        except (TypeError, ValueError):
            continue
        if x_val == y_val:
            x_is_y.append(f"{element} {ep.get('y_metric')} {y_val:g}")
            continue
        y_col = _Y_METRIC_TO_COL.get(str(ep.get("y_metric", "")))
        if not y_col or y_val is None:
            continue  # not a numerically checkable endpoint

        sub = df
        if element and ELEMENT_COLUMN in df.columns:
            # Exact symbol match: a substring test matched 'Y' inside 'Yb'.
            mask = df[ELEMENT_COLUMN].astype("string").str.strip().str.casefold() == str(element).strip().casefold()
            if not mask.any():
                continue  # nothing extracted for that element (out of scope, or a skipped figure)
            sub = df[mask]
        extractant = ep.get("extractant")
        if extractant and "Extractant" in sub.columns:
            mask = sub["Extractant"].map(lambda v: pd.notna(v) and _norm_name(v) == _norm_name(extractant))
            if mask.any():
                sub = sub[mask]

        xs = pd.to_numeric(sub[x_col], errors="coerce")
        ys = pd.to_numeric(sub[y_col], errors="coerce")
        valid = xs.notna() & ys.notna()
        if not valid.any():
            continue
        # Curves that actually sweep this x; a lone point is another
        # experiment's fixed condition (a concentration sweep at pH 1.75).
        n_curves = sum(1 for c in curve_def.curves(sub[valid], x_col)
                       if pd.to_numeric(c.rows[x_col], errors="coerce").nunique() > 1)
        ambiguous = not element or n_curves > 1
        severity = Severity.AMBER if ambiguous else Severity.RED
        caveat = (f" The element has {n_curves} curves along {x_col} here and the statement names "
                  "no loading or figure, so it may describe another of them." if ambiguous else "")
        xs, ys = xs[valid], ys[valid]
        who = f"'{element}'" + (f" with {extractant}" if extractant else "")

        x_dist = (xs - x_val).abs()
        nearest_x = xs.loc[x_dist.idxmin()]
        if x_col == "pH":
            if x_dist.min() > PH_TOL:
                # No digitized point near the stated pH at all — the curve
                # may not reach the paper's stated endpoint.
                report.add(
                    "text_endpoint_cross_check",
                    severity,
                    f"Paper states {y_col} {y_val:g} at {x_col} {x_val:g} for {who}, but no "
                    f"digitized point is within {PH_TOL} of {x_col}={x_val:g} (nearest is "
                    f"{nearest_x:g}). Possible truncated curve.{caveat}",
                )
                continue
            candidates = x_dist <= PH_TOL
        else:
            if _ratio(nearest_x, x_val) > CONC_TOL_RATIO:
                # Nothing within a factor of CONC_TOL_RATIO of the stated
                # value: comparing y against the nearest row would compare
                # against the wrong point. The usual cause is the endpoint
                # captured in M while the rows are in mM, which is
                # unverifiable here rather than a digitization error — so
                # this is a warning for the reviewer, not a merge gate.
                report.add(
                    "text_endpoint_cross_check",
                    Severity.AMBER,
                    f"Paper states {y_col} {y_val:g} at {x_col} {x_val:g} for {who}, but no "
                    f"digitized point is within a factor of {CONC_TOL_RATIO:g} of {x_col}={x_val:g} "
                    f"(nearest is {nearest_x:g}). Endpoint may be in different units (e.g. M vs mM) "
                    f"or the sweep may be truncated — value not compared.",
                )
                continue
            candidates = x_dist == x_dist.min()

        # Several rows can legitimately share the same (or nearly the same) x —
        # e.g. a paper's separate concentration-sweep experiment holds pH fixed
        # while %E varies with concentration, so "nearest x" alone can land on
        # an unrelated point from a different experiment. Among every candidate
        # row, the one whose y best matches the stated y is the real match.
        best_idx = (ys[candidates] - y_val).abs().idxmin()
        y_near = ys.loc[best_idx]
        if abs(y_near - y_val) > PCT_TOL:
            report.add(
                "text_endpoint_cross_check",
                severity,
                f"Paper states {y_col} {y_val:g} at {x_col} {x_val:g} for {who}, but the "
                f"digitized value there is {y_near:g} (off by > {PCT_TOL}). Possible "
                f"calibration drift or under-extraction.{caveat}",
            )
    if unknown_bases:
        report.add(
            "text_endpoint_cross_check",
            Severity.AMBER,
            f"Text endpoint(s) on an x_basis that is no schema column ({', '.join(repr(b) for b in unknown_bases)}) "
            f"could not be compared. x_basis must be the column the x lives in ({', '.join(X_COLUMNS)}); "
            "a stated x no column can hold means the sweep itself has no place in the table.",
        )
    if x_is_y:
        report.add(
            "text_endpoint_cross_check",
            Severity.AMBER,
            f"Text endpoint(s) whose x_value equals their y_value ({', '.join(x_is_y)}) — the x was "
            "copied from the y; not compared.",
        )
