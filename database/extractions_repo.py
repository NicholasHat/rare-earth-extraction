"""Append `extractions` / `text_endpoints` rows and query via `v_current_best`."""
from __future__ import annotations

import sqlite3

import pandas as pd

from validation.schema import COLUMNS

# Pre-built column lists for the parametrised INSERT (26 schema cols + 2 FKs).
_INSERT_COLS = ["paper_id", "prompt_run_id"] + COLUMNS
_QUOTED = ", ".join(f'"{c}"' for c in _INSERT_COLS)
_PLACEHOLDERS = ", ".join(["?"] * len(_INSERT_COLS))
_INSERT_SQL = f"INSERT INTO extractions ({_QUOTED}) VALUES ({_PLACEHOLDERS})"


def insert_extractions(
    conn: sqlite3.Connection, paper_id: int, prompt_run_id: int, df: pd.DataFrame
) -> int:
    """Insert the 26-column rows for one run. Returns the number of rows inserted."""
    rows = []
    for _, r in df.iterrows():
        values = [paper_id, prompt_run_id]
        for col in COLUMNS:
            v = r.get(col)
            # Normalise pandas NaN/NA to SQL NULL.
            values.append(None if pd.isna(v) else v)
        rows.append(values)
    conn.executemany(_INSERT_SQL, rows)
    return len(rows)


def insert_text_endpoints(
    conn: sqlite3.Connection,
    paper_id: int,
    prompt_run_id: int,
    endpoints: list[dict],
) -> int:
    """Insert captured text endpoints (may be empty). Returns count inserted."""
    rows = [
        (
            paper_id,
            prompt_run_id,
            ep.get("element"),
            ep.get("x_value"),
            ep.get("x_basis"),
            ep.get("y_value"),
            ep.get("y_metric"),
            ep.get("source_quote"),
        )
        for ep in endpoints
        if ep.get("element")  # element is NOT NULL
    ]
    conn.executemany(
        """
        INSERT INTO text_endpoints
            (paper_id, prompt_run_id, element, x_value, x_basis, y_value, y_metric, source_quote)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def current_best(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the current-best (latest approved version per paper) extractions."""
    return pd.read_sql_query("SELECT * FROM v_current_best", conn)


def count_current_best(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM v_current_best").fetchone()[0])
