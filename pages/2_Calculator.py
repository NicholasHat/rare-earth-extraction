"""Pillar B — Extractant calculator (README §7).

Deterministic, solve-for-the-blank conversions usable at the bench without
touching the Anthropic API, plus an optional cross-reference against prior
*approved* extractions (read-only; this page never writes to the master DB).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from calculator.atomic_mass import REE_ELEMENTS
from calculator.predict import predict_extract_pct
from calculator.sanity import typical_ranges
from calculator.solve import CalculatorInputs, solve
from database import connection

st.set_page_config(page_title="Extractant Calculator", layout="wide")
st.title("Extractant Calculator")
st.caption("Pillar B — solve-for-the-blank conversions. No API calls, no write access required.")

# Ensure the data dir + schema exist. Required before the read-only open below:
# SQLite's mode=ro will not create a missing file, so a fresh install would
# otherwise fail here rather than showing an empty extractant list.
connection.init_db()

conn = connection.get_readonly_conn()
try:
    known_extractants = [
        r[0]
        for r in conn.execute(
            'SELECT DISTINCT "Extractant" FROM v_current_best '
            'WHERE "Extractant" IS NOT NULL ORDER BY 1'
        ).fetchall()
    ]
finally:
    conn.close()

col1, col2 = st.columns(2)
with col1:
    extractant_choice = st.selectbox(
        "Extractant identity", known_extractants + ["Other (type below)"]
    )
    extractant = (
        st.text_input("Extractant name", placeholder="e.g. Cyanex 272")
        if extractant_choice == "Other (type below)"
        else extractant_choice
    )
    element = st.selectbox("REE element", REE_ELEMENTS)
with col2:
    feed_value = st.number_input("Feed metal concentration", min_value=0.0, value=0.0, step=1.0)
    feed_unit = st.radio("Feed unit", ["ppm", "mM"], horizontal=True)
    has_pH = st.checkbox("Check against a target pH")
    target_pH = st.number_input(
        "Target pH", min_value=-1.0, max_value=14.0, value=2.0, step=0.1, disabled=not has_pH
    )
    volume_ratio = st.number_input(
        "Phase volume ratio (org:aq)", min_value=0.1, value=1.0, step=0.1,
        help="The extraction schema doesn't record phase ratio, so prior data "
             "can't be filtered by it — predictions assume the papers' own "
             "(usually 1:1) ratio.",
    )

st.divider()
st.write("Fill in **one** of the two fields below (leave the other at 0) and the engine solves it.")
c1, c2, c3 = st.columns(3)
with c1:
    molar_ratio = st.number_input("Target molar ratio EX/REE", min_value=0.0, value=0.0, step=1.0)
with c2:
    extractant_conc = st.number_input(
        "Target extractant conc. (mM)", min_value=0.0, value=0.0, step=1.0
    )
with c3:
    volume_mL = st.number_input("Solution volume (mL, optional)", min_value=0.0, value=0.0, step=1.0)

if st.button("Solve", type="primary"):
    inputs = CalculatorInputs(
        element=element,
        feed_value=feed_value or None,
        feed_unit=feed_unit,
        target_molar_ratio=molar_ratio or None,
        target_extractant_conc_mM=extractant_conc or None,
        volume_mL=volume_mL or None,
    )
    result = solve(inputs)

    for w in result.warnings:
        st.warning(w)

    st.subheader("Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Feed (ppm)", f"{result.ree_ppm:.3g}" if result.ree_ppm is not None else "—")
    r1.metric("Feed (mM)", f"{result.ree_mM:.4g}" if result.ree_mM is not None else "—")
    r2.metric(
        "Extractant conc. (mM)",
        f"{result.extractant_conc_mM:.3g}" if result.extractant_conc_mM is not None else "—",
    )
    r2.metric(
        "Molar ratio EX/REE", f"{result.molar_ratio:.3g}" if result.molar_ratio is not None else "—"
    )
    if volume_mL:
        r3.metric(
            "REE mass (mg)", f"{result.ree_mass_mg:.3g}" if result.ree_mass_mg is not None else "—"
        )
        r3.metric(
            "Extractant (mmol)",
            f"{result.extractant_mmol_total:.3g}" if result.extractant_mmol_total is not None else "—",
        )

    if extractant:
        conn = connection.get_readonly_conn()
        try:
            summary = typical_ranges(conn, extractant, element)
        finally:
            conn.close()

        if summary is None:
            st.info(f"No prior approved data for {extractant} + {element} yet.")
        else:
            st.subheader(
                f"Cross-reference: {extractant} + {element} "
                f"({summary.n_papers} paper(s), {summary.n_rows} rows)"
            )
            st.write(
                f"pH range {summary.pH_min:.2g}–{summary.pH_max:.2g} "
                f"(median {summary.pH_median:.2g}); "
                f"Extract% range {summary.extract_pct_min:.1f}–{summary.extract_pct_max:.1f} "
                f"(median {summary.extract_pct_median:.1f})"
            )
            if has_pH and summary.pH_min is not None and not (
                summary.pH_min <= target_pH <= summary.pH_max
            ):
                st.warning(
                    f"Across {summary.n_papers} paper(s), {extractant} extraction of {element} runs "
                    f"at pH {summary.pH_min:.2g}–{summary.pH_max:.2g} "
                    f"(median {summary.pH_median:.2g}). Your input of pH {target_pH:g} is outside "
                    "this range — double-check; the dataset may simply be incomplete."
                )

    # Predicted Extract% at the target pH, interpolated inside real measured
    # sweeps from prior approved papers (calculator.predict — deterministic,
    # per-paper, never extrapolated beyond a sweep's measured pH range).
    if extractant and has_pH:
        conn = connection.get_readonly_conn()
        try:
            prediction = predict_extract_pct(
                conn,
                element=element,
                extractant=extractant,
                ph=target_pH,
                extractant_conc_mM=result.extractant_conc_mM,
                feed_ppm=result.ree_ppm,
            )
        finally:
            conn.close()

        st.subheader(f"Predicted Extract% — {element} with {extractant} at pH {target_pH:g}")
        if volume_ratio != 1.0:
            st.warning(
                f"Phase ratio {volume_ratio:g}:1 requested, but the schema doesn't record "
                "phase ratio — prior data generally assumes ~1:1, so treat the prediction "
                "as approximate at other ratios."
            )

        def _series_rows(series_list):
            return pd.DataFrame([
                {
                    "paper_id": s.paper_id,
                    "extractant conc. (mM)": s.extractant_conc_mM,
                    "feed (ppm)": s.feed_ppm,
                    "sweep points": s.n_points,
                    "sweep pH range": f"{s.ph_min:g}–{s.ph_max:g}",
                    "predicted Extract%": (
                        f"{s.predicted_extract_pct:.1f}"
                        if s.predicted_extract_pct is not None
                        else "pH outside measured sweep"
                    ),
                    "interpolated between (pH, %)": (
                        f"{s.bracket[0]} ↔ {s.bracket[1]}" if s.bracket else "—"
                    ),
                }
                for s in series_list
            ])

        if not prediction.matched and not prediction.off_condition:
            st.info(
                f"No approved pH-sweep data for {extractant} + {element} yet — "
                "the prediction needs at least one merged paper measuring it."
            )
        else:
            if prediction.best_estimate is not None:
                st.metric(
                    "Best estimate (median of matched sweeps)",
                    f"{prediction.best_estimate:.1f}%",
                )
            elif prediction.matched:
                st.info(
                    "Matching sweeps exist but none covers this pH — see their "
                    "measured ranges below (predictions are never extrapolated)."
                )
            if prediction.matched:
                st.dataframe(_series_rows(prediction.matched), use_container_width=True)
            if prediction.off_condition:
                with st.expander(
                    f"Sweeps under other conditions ({len(prediction.off_condition)}) — "
                    "same pair, different conc./feed"
                ):
                    st.dataframe(
                        _series_rows(prediction.off_condition), use_container_width=True
                    )
