"""The 26-column extraction schema contract (see README §5).

This is the single source of truth for column names, order, and dtypes. The
column names are the *canonical* strings produced by the extraction prompt and
stored verbatim (as quoted identifiers) in the `extractions` table, so nothing
downstream has to rename or re-learn them.
"""
from __future__ import annotations

import pandas as pd

# Canonical 26 columns, in order.
COLUMNS: list[str] = [
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
]

# The remaining columns are free TEXT.
TEXT_COLUMNS: list[str] = [c for c in COLUMNS if c not in NUMERIC_COLUMNS]

# The element identity column drives per-element grouping in the QA checks.
ELEMENT_COLUMN = "Rare Earth Elements (REY:La, Ce, Nd)"


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exactly the 26 columns, in order, dtypes coerced.

    Missing columns are added as null; extra columns are dropped. Numeric
    columns are coerced with errors -> NaN (an OCR-garbled "1.2x" surfaces as
    NaN here, which the schema-conformance check then flags — see README §9).
    """
    out = pd.DataFrame()
    for col in COLUMNS:
        series = df[col] if col in df.columns else pd.Series([None] * len(df))
        if col in NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(series, errors="coerce")
        else:
            out[col] = series.astype("object").where(series.notna(), None)
    return out
