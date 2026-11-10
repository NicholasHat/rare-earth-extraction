"""The 35-column extraction schema contract (see README §5).

This is the single source of truth for column names, order, and dtypes. The
column names are the *canonical* strings produced by the extraction prompt and
stored verbatim (as quoted identifiers) in the `extractions` table, so nothing
downstream has to rename or re-learn them.

The first 26 columns (`LAB_COLUMNS`) mirror the lab's own spreadsheet and never
change name or order. `ADDED_COLUMNS` are appended after them: the 2026-10
ten-paper trial showed rows losing information the lab's columns have no home
for (see prompts/CHANGELOG.md, extraction_v15). Model output written before
they existed (extraction_v7–v14) carries the 26 lab columns only; those rows
load with the added columns null.
"""
from __future__ import annotations

import pandas as pd

# The lab spreadsheet's 26 columns, in its order.
LAB_COLUMNS: list[str] = [
    "Reference No.",
    "DOI",
    "Treatment",
    "Sources",
    "Material Process",
    "Si (%)",
    "Al (%)",
    "Zn (%)",
    "Fe (%)",
    "Rare Earth Elements (REY:La, Ce, Nd)",
    "RRE composition (ppm)",
    "RRE composition (mM)",
    "Extractant",
    "Extractant type",
    "Extractant Conc. (mM)",
    "Molar ratio of EX/REE",
    "Extract%",
    "Extract Temperature (oC)",
    "pH",
    "Separation factor (SF%)",
    "Acid Solution",
    "Acid Solution conc. (M)",
    "mixing method",
    "Stripping Temperature (oC)",
    "Leaching time (minute)",
    "Recovery %",
]

# Appended by extraction_v15 (2026-11). Order is part of the contract.
ADDED_COLUMNS: list[str] = [
    "Diluent",                      # organic diluent (kerosene, n-heptane, ...)
    "Co-extractant",                # synergist / modifier of a mixed system
    "Co-extractant Conc. (mM)",
    "Phase ratio (O/A)",            # V_org/V_aq; a solid sorbent's m/V in g/mL
    "Contact time (minute)",        # extraction contact time ("Leaching time" is leaching only)
    "Distribution ratio (D)",       # D as reported (log D converted to D)
    "Total REE feed (mM)",          # total feed of a mixed-REE solution
    "pH type",                      # 'equilibrium' | 'initial'
    "Figure/Table",                 # source of the row: 'Fig. 2a', 'Table 3', 'text'
]

COLUMNS: list[str] = LAB_COLUMNS + ADDED_COLUMNS

# Columns stored as REAL in SQLite — coerced to float, non-numeric -> NaN.
NUMERIC_COLUMNS: list[str] = [
    "Si (%)",
    "Al (%)",
    "Zn (%)",
    "Fe (%)",
    "RRE composition (ppm)",
    "RRE composition (mM)",
    "Extractant Conc. (mM)",
    "Molar ratio of EX/REE",
    "Extract%",
    "Extract Temperature (oC)",
    "pH",
    "Separation factor (SF%)",
    "Acid Solution conc. (M)",
    "Stripping Temperature (oC)",
    "Leaching time (minute)",
    "Recovery %",
    "Co-extractant Conc. (mM)",
    "Phase ratio (O/A)",
    "Contact time (minute)",
    "Distribution ratio (D)",
    "Total REE feed (mM)",
]

# The remaining columns are free TEXT.
TEXT_COLUMNS: list[str] = [c for c in COLUMNS if c not in NUMERIC_COLUMNS]

# The element identity column drives per-element grouping in the QA checks.
ELEMENT_COLUMN = "Rare Earth Elements (REY:La, Ce, Nd)"
# Where each row came from. A curve is (paper, Figure/Table, element, series).
FIGURE_COLUMN = "Figure/Table"
PH_TYPE_COLUMN = "pH type"
PH_TYPES = ("equilibrium", "initial")


def sql_type(column: str) -> str:
    """The SQLite column type of a schema column."""
    return "REAL" if column in NUMERIC_COLUMNS else "TEXT"


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exactly the schema's columns, in order, dtypes coerced.

    Missing columns are added as null; extra columns are dropped. Numeric
    columns are coerced with errors -> NaN (an OCR-garbled "1.2x" surfaces as
    NaN here, which the schema-conformance check then flags — see README §9).
    """
    out = pd.DataFrame(index=df.index)
    for col in COLUMNS:
        series = df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index, dtype="object")
        if col in NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(series, errors="coerce")
        else:
            out[col] = series.astype("object").where(series.notna(), None)
    return out
