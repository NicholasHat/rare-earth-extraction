"""Automatic QA checks run after every extraction, before review (README §9).

All checks operate on the coerced 26-column DataFrame plus the captured
text_endpoints, so they are pure and unit-testable without spending API tokens.
Each known failure mode maps to a check here:

  - silent under-extraction ("stopped at 2 endpoints")  -> row_count_sanity (RED)
  - axis calibration drift                              -> text_endpoint_cross_check / axis_bounds (RED)
  - OCR-garbled numeric tables                          -> schema_conformance (RED)
  - monochrome series merged/dropped                    -> row_count_sanity + duplicate_rows + monotonicity
  - vocabulary drift                                    -> vocabulary (AMBER)
"""
from __future__ import annotations

import math

import pandas as pd

from . import vocab
from .report import QAReport, Severity
from .schema import ELEMENT_COLUMN

# Tuning constants (README §11 item 4 — revisit against the first dozen papers).
SPARSE_MIN_ROWS = 8          # a curve-type figure should yield >= this many points/element
PH_TOL = 0.3                 # text-endpoint x-match tolerance on pH
PCT_TOL = 10.0               # text-endpoint y-mismatch tolerance on %-type metrics
MONOTONICITY_NOISE = 5.0     # %E reversal smaller than this is treated as noise

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
        label = key[0] if isinstance(key, tuple) else key
        yield str(label), sub


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


def _monotonicity(df: pd.DataFrame, report: QAReport) -> None:
    if "pH" not in df.columns or "Extract%" not in df.columns:
        return
    for label, sub in _curve_groups(df):
        s = sub[["pH", "Extract%"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(s) < 4:
            continue
        s = s.sort_values("pH")
        deltas = s["Extract%"].diff().dropna()
        ups = (deltas > MONOTONICITY_NOISE).sum()
        downs = (deltas < -MONOTONICITY_NOISE).sum()
        # Broadly monotonic (or a plateau) means movement is essentially one-way.
        if ups >= 2 and downs >= 2:
            report.add(
                "monotonicity",
                Severity.AMBER,
                f"Element '{label}' %E-vs-pH curve is non-monotonic "
                f"({ups} rises, {downs} falls beyond noise) — check for misread "
                "points or two series merged into one.",
            )


def _duplicate_rows(df: pd.DataFrame, report: QAReport) -> None:
    key = [ELEMENT_COLUMN, "pH", "Extract%"]
    if not all(c in df.columns for c in key):
        return
    dups = df.duplicated(subset=key, keep=False) & df[key].notna().all(axis=1)
    n = int(dups.sum())
    if n > 0:
        report.add(
            "duplicate_rows",
            Severity.AMBER,
            f"{n} row(s) share an identical (element, pH, %E) triple — possible "
            "digitizing loop or copy error.",
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
        if x_dist.min() > PH_TOL and x_col == "pH":
            # No digitized point near the stated x at all — the curve may not
            # reach the paper's stated endpoint.
            report.add(
                "text_endpoint_cross_check",
                Severity.RED,
                f"Paper states {y_col} {y_val} at {x_col} {x_val} for "
                f"'{element}', but no digitized point is within {PH_TOL} of "
                f"{x_col}={x_val} (nearest is {nearest_x:g}). Possible truncated curve.",
            )
            continue

        # Several rows can legitimately share the same (or nearly the same) x —
        # e.g. a paper's separate concentration-sweep experiment holds pH fixed
        # while %E varies with concentration, so "nearest x" alone can land on
        # an unrelated point from a different experiment. Among every row
        # within tolerance of the stated x, the one whose y best matches the
        # stated y is the real match.
        candidates = x_dist <= (PH_TOL if x_col == "pH" else x_dist.min())
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
