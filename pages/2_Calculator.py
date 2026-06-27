"""Pillar B — Extractant calculator (README §7).

Deterministic, solve-for-the-blank conversions usable at the bench without
touching the Anthropic API, plus an optional cross-reference against prior
*approved* extractions (read-only; this page never writes to the master DB).
"""
from __future__ import annotations

import streamlit as st

from calculator.atomic_mass import REE_ELEMENTS
from calculator.sanity import typical_ranges
from calculator.solve import CalculatorInputs, solve
from database import connection

st.set_page_config(page_title="Extractant Calculator", layout="wide")
st.title("Extractant Calculator")
st.caption("Pillar B — solve-for-the-blank conversions. No API calls, no write access required.")

# Ensure the data dir + schema exist (also lets the DISTINCT query below run on a fresh DB).
connection.init_db()

conn = connection.get_conn()
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
