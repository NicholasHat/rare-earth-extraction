"""Automatic QA checks run after every extraction, before review (README §9).

All checks operate on the coerced 26-column DataFrame plus the captured
text_endpoints, so they are pure and unit-testable without spending API tokens.
Each known failure mode maps to a check here:

  - silent under-extraction ("stopped at 2 endpoints")  -> row_count_sanity (RED)
  - axis calibration drift                              -> text_endpoint_cross_check / axis_bounds (RED)
  - OCR-garbled numeric tables                          -> schema_conformance (RED)
  - monochrome series merged/dropped                    -> row_count_sanity + duplicate_rows + monotonicity
  - in-plot text / another series' markers digitised in -> off_curve (AMBER)
  - vocabulary drift                                    -> vocabulary (AMBER)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import vocab
from .report import QAReport, Severity
from .schema import ELEMENT_COLUMN

# Tuning constants (README §11 item 4 — revisit against the first dozen papers).
SPARSE_MIN_ROWS = 8          # a curve-type figure should yield >= this many points/element
PH_TOL = 0.3                 # text-endpoint x-match tolerance on pH
PCT_TOL = 10.0               # text-endpoint y-mismatch tolerance on %-type metrics
CONC_TOL_RATIO = 2.0         # text-endpoint x-match tolerance on concentration (a ratio: sweeps are log-spaced)
MONOTONICITY_NOISE = 5.0     # %E reversal smaller than this is treated as noise
OFF_CURVE_FIT_PCT = (1.0, 97.0)  # only %E in this band carries slope information in log D; outside it a
                                 # digitised curve is a saturation plateau (or censored at the frame)
OFF_CURVE_INLIER_LOGD = 0.5      # |log D residual| for a point to count as ON the fitted line
OFF_CURVE_FLAG_LOGD = 0.8        # flag needs BOTH: this far off in log D (factor ~6 in D) ...
OFF_CURVE_FLAG_PCT = 25.0        # ... and this far off in %E (so a 1 % vs 3 % scatter never trips it)
OFF_CURVE_MIN_INLIER_FRAC = 0.5  # a line must explain at least this share of the band (and >= 4 points)

# Map a text-endpoint y_metric / x_basis to the schema column it lives in.
_Y_METRIC_TO_COL = {
    "Extract%": "Extract%",
    "extract%": "Extract%",
    "Recovery %": "Recovery %",
    "recovery %": "Recovery %",
    "recovery": "Recovery %",
}
_X_BASIS_TO_COL = {
    "pH": "pH",
    "ph": "pH",
    "extractant_conc_mM": "Extractant Conc. (mM)",
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

    _row_count_sanity(df, figure_is_curve, report)
    _axis_bounds(df, report)
    _monotonicity(df, report)
    _off_curve(df, report)
    _duplicate_rows(df, report)
    _vocabulary(df, report)
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


# Columns besides a varied x-axis that define which experiment a row belongs
# to. The prompt combines every experiment in the paper into one flat `rows`
# list (extraction_v5.1+ OUTPUT CONTRACT rule "one combined rows list across
# all experiments") — a paper commonly reports a *second* experiment for the
# same element (e.g. a concentration sweep at fixed pH) alongside the primary
# one (e.g. a pH sweep at fixed concentration). Grouping by element alone
# pools both together, corrupting any check of one curve's shape.
_CURVE_KEY_COLUMNS = [
    "Extractant",
    "Extractant Conc. (mM)",
    "Extract Temperature (oC)",
    "Acid Solution conc. (M)",
    "Leaching time (minute)",
    "Stripping Temperature (oC)",
]


def _curve_groups(df: pd.DataFrame):
    """Yield (element_label, sub_df) per distinct curve: same element AND the
    same fixed experimental conditions, so two different experiments for the
    same element are never pooled into one curve."""
    if ELEMENT_COLUMN not in df.columns:
        return
    key_cols = [ELEMENT_COLUMN] + [c for c in _CURVE_KEY_COLUMNS if c in df.columns]
    grouped = df.copy()
    grouped[ELEMENT_COLUMN] = grouped[ELEMENT_COLUMN].fillna("(unspecified)")
    for key, sub in grouped.groupby(key_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        # "Lu" alone is ambiguous once a paper has several extractants for the
        # same element; name the curve the way the reviewer would.
        conds = [f"{v:g} mM" if c == "Extractant Conc. (mM)" else str(v)
                 for c, v in zip(key_cols[1:], key[1:])
                 if c in ("Extractant", "Extractant Conc. (mM)") and pd.notna(v)]
        label = f"{key[0]} ({', '.join(conds)})" if conds else str(key[0])
        yield label, sub


def _row_count_sanity(df: pd.DataFrame, figure_is_curve: bool, report: QAReport) -> None:
    if not figure_is_curve:
        return
    for label, sub in _element_groups(df):
        n = len(sub)
        if n < SPARSE_MIN_ROWS:
            report.add(
                "row_count_sanity",
                Severity.RED,
                f"Element '{label}' has only {n} digitized point(s); a multi-point "
                f"curve should have >= {SPARSE_MIN_ROWS}. Possible silent "
                "under-extraction (model stopped early instead of digitizing the "
                "whole curve).",
            )


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
        "Separation factor (SF%)": (0.0, math.inf),
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


def _curve_points(df: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """A curve's numeric (pH, Extract%) points sorted by pH, with a `row`
    column giving each point's 1-based position in `df` — the row number the
    reviewer sees in the review editor and in the exported CSV."""
    s = sub[["pH", "Extract%"]].apply(pd.to_numeric, errors="coerce").dropna()
    s["row"] = df.index.get_indexer(s.index) + 1
    return s.sort_values("pH")


def _rows(rows) -> str:
    rows = [int(r) for r in rows]
    shown = ", ".join(str(r) for r in rows[:8])
    return shown + (f", … ({len(rows)} total)" if len(rows) > 8 else "")


def _monotonicity(df: pd.DataFrame, report: QAReport) -> None:
    if "pH" not in df.columns or "Extract%" not in df.columns:
        return
    for label, sub in _curve_groups(df):
        s = _curve_points(df, sub)
        if len(s) < 4:
            continue
        deltas = s["Extract%"].diff()
        ups = int((deltas > MONOTONICITY_NOISE).sum())
        downs = int((deltas < -MONOTONICITY_NOISE).sum())
        # Broadly monotonic (or a plateau) means movement is essentially one-way.
        if ups >= 2 and downs >= 2:
            report.add(
                "monotonicity",
                Severity.AMBER,
                f"Element '{label}' %E-vs-pH curve is non-monotonic "
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

    log D is linear in pH for the cation-exchange systems this database
    covers — that linearity is why papers plot it. So the curve is the line
    through the most points (every pair proposes one; deterministic, n is tens
    of points), refined by least squares on its inliers. A point well off that
    line is not on the curve, whatever its neighbours look like — which is
    what defeats local tests: a panel title digitised as three adjacent
    markers, or an artifact sitting right next to a real point.

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


def _off_curve(df: pd.DataFrame, report: QAReport) -> None:
    """Points that sit far off their own curve. Extraction curves don't do that;
    in-plot text does — a panel title or annotation digitised as a marker lands
    at whatever %E its pixels sit at (seen live: Quinn 2015's panel titles at
    log D ≈ 0.7 became runs of 83–85 %E points in the Lu series, three of them
    side by side). So does a neighbouring series' marker assigned to this one."""
    if "pH" not in df.columns or "Extract%" not in df.columns:
        return
    for label, sub in _curve_groups(df):
        s = _curve_points(df, sub)
        off, n_on = _off_curve_points(s["pH"].to_numpy(float), s["Extract%"].to_numpy(float))
        if off:
            rows = s["row"].to_numpy()[off]
            report.add(
                "off_curve",
                Severity.AMBER,
                f"{label}: {len(rows)} point(s) at row(s) {_rows(rows)} lie > "
                f"{OFF_CURVE_FLAG_LOGD:g} decades in log D off the straight line through "
                f"the other {n_on} points of this curve — typical of in-plot text "
                "(panel titles, annotations) or another series' markers digitised into "
                "this one. Delete unless the figure really shows them.",
                rows=rows,
            )


def _duplicate_rows(df: pd.DataFrame, report: QAReport) -> None:
    key = [ELEMENT_COLUMN, "pH", "Extract%"]
    if not all(c in df.columns for c in key):
        return
    dups = df.duplicated(subset=key, keep=False) & df[key].notna().all(axis=1)
    n = int(dups.sum())
    if n == 0:
        return
    # A repeat of an earlier row *within the same curve* (same element and
    # experimental conditions) is a digitising loop: the later copies add
    # nothing and are safe to drop by row. The same triple under two different
    # extractants is a point assigned to two series — one copy is real and
    # only the reviewer can say which, so those are reported but not named.
    curve_key = [ELEMENT_COLUMN] + [c for c in _CURVE_KEY_COLUMNS if c in df.columns]
    same_curve = df.duplicated(subset=curve_key + ["pH", "Extract%"], keep="first") & dups
    repeat_rows = [int(i) + 1 for i in np.flatnonzero(same_curve.to_numpy())]
    report.add(
        "duplicate_rows",
        Severity.AMBER,
        f"{n} row(s) share an identical (element, pH, %E) triple — possible "
        "digitizing loop or copy error."
        + (f" {len(repeat_rows)} of them repeat an earlier row of the same curve "
           f"(row(s) {_rows(repeat_rows)}) and can be dropped." if repeat_rows else ""),
        rows=repeat_rows,
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


def _text_endpoint_cross_check(
    df: pd.DataFrame, text_endpoints: list[dict], report: QAReport
) -> None:
    for ep in text_endpoints:
        x_col = _X_BASIS_TO_COL.get(str(ep.get("x_basis", "")))
        y_col = _Y_METRIC_TO_COL.get(str(ep.get("y_metric", "")))
        x_val = ep.get("x_value")
        y_val = ep.get("y_value")
        element = ep.get("element")
        if not (x_col and y_col) or x_val is None or y_val is None:
            continue  # not a numerically checkable endpoint

        sub = df
        if element and ELEMENT_COLUMN in df.columns:
            mask = df[ELEMENT_COLUMN].astype("string").str.contains(
                str(element), case=False, na=False
            )
            if mask.any():
                sub = df[mask]

        xs = pd.to_numeric(sub[x_col], errors="coerce")
        ys = pd.to_numeric(sub[y_col], errors="coerce")
        valid = xs.notna() & ys.notna()
        if not valid.any():
            continue
        xs, ys = xs[valid], ys[valid]

        x_dist = (xs - float(x_val)).abs()
        nearest_x = xs.loc[x_dist.idxmin()]
        if x_col == "pH":
            if x_dist.min() > PH_TOL:
                # No digitized point near the stated pH at all — the curve
                # may not reach the paper's stated endpoint.
                report.add(
                    "text_endpoint_cross_check",
                    Severity.RED,
                    f"Paper states {y_col} {y_val} at {x_col} {x_val} for "
                    f"'{element}', but no digitized point is within {PH_TOL} of "
                    f"{x_col}={x_val} (nearest is {nearest_x:g}). Possible truncated curve.",
                )
                continue
            candidates = x_dist <= PH_TOL
        else:
            if _ratio(nearest_x, float(x_val)) > CONC_TOL_RATIO:
                # Nothing within a factor of CONC_TOL_RATIO of the stated
                # concentration: comparing y against the nearest row would
                # compare against the wrong point. The usual cause is the
                # endpoint captured in M while the rows are in mM, which is
                # unverifiable here rather than a digitization error — so
                # this is a warning for the reviewer, not a merge gate.
                report.add(
                    "text_endpoint_cross_check",
                    Severity.AMBER,
                    f"Paper states {y_col} {y_val} at {x_col} {x_val} for "
                    f"'{element}', but no digitized point is within a factor of "
                    f"{CONC_TOL_RATIO:g} of {x_col}={x_val} (nearest is {nearest_x:g}). "
                    f"Endpoint may be in different units (e.g. M vs mM) or the sweep "
                    f"may be truncated — value not compared.",
                )
                continue
            candidates = x_dist == x_dist.min()

        # Several rows can legitimately share the same (or nearly the same) x —
        # e.g. a paper's separate concentration-sweep experiment holds pH fixed
        # while %E varies with concentration, so "nearest x" alone can land on
        # an unrelated point from a different experiment. Among every candidate
        # row, the one whose y best matches the stated y is the real match.
        best_idx = (ys[candidates] - float(y_val)).abs().idxmin()
        y_near = ys.loc[best_idx]
        if abs(y_near - float(y_val)) > PCT_TOL:
            report.add(
                "text_endpoint_cross_check",
                Severity.RED,
                f"Paper states {y_col} {y_val} at {x_col} {x_val} for "
                f"'{element}', but the digitized value there is {y_near:g} "
                f"(off by > {PCT_TOL}). Possible calibration drift or under-extraction.",
            )


def _ratio(a: float, b: float) -> float:
    """How far apart two positive magnitudes are, as a ratio >= 1 (inf if signs differ or one is zero)."""
    if a == b:
        return 1.0
    if a <= 0 or b <= 0:
        return math.inf
    return max(a / b, b / a)
