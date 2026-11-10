"""The approved rows the Analysis page plots.

Read-only by design: callers pass a `connection.get_readonly_conn()`
connection, and only `v_current_best` (each paper's current approved run) is
read — joined to `papers` for a readable label, never to the raw
`extractions` table.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from validation import schema

ELEMENT = schema.ELEMENT_COLUMN
PAPER = "Paper"          # readable paper label, added to every row

# Atomic numbers: the order the lanthanide series is plotted in. Pm is
# radioactive and never measured in these papers; Y and Sc are rare earths
# but not lanthanides, so they come after Lu wherever the series is drawn.
ATOMIC_NUMBER: dict[str, int] = {
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Y": 39, "Sc": 21,
}
LANTHANIDES: list[str] = [e for e in ATOMIC_NUMBER if e not in ("Y", "Sc")]
# Plot order along the series: La..Lu, then Y, then Sc.
SERIES_ORDER: list[str] = LANTHANIDES + ["Y", "Sc"]


def series_position(element: str) -> int:
    """Position of an element along SERIES_ORDER (unknown elements last)."""
    try:
        return SERIES_ORDER.index(element)
    except ValueError:
        return len(SERIES_ORDER)


@dataclass
class Rows:
    """The approved rows, ready to plot."""
    df: pd.DataFrame                       # schema columns + paper_id + PAPER
    excluded: dict[str, int] = field(default_factory=dict)   # non-REE element -> rows left out


def paper_label(short_citation, pub_year, title, paper_id) -> str:
    """'Swain & Otu 2011'; else the title (shortened); else 'Paper 7'."""
    if isinstance(short_citation, str) and short_citation.strip():
        year = f" {pub_year}" if isinstance(pub_year, str) and pub_year.strip() else ""
        return short_citation.strip() + year
    if isinstance(title, str) and title.strip():
        t = title.strip()
        return t if len(t) <= 50 else t[:49].rstrip() + "…"
    return f"Paper {paper_id}"


def load(conn: sqlite3.Connection) -> Rows:
    """Every current approved row of a rare earth element, with a paper label.

    Columns are read by name and coerced through validation.schema, so a DB
    written before the 2026-11 columns existed loads with them null.
    Rows for any other element (the DB is meant to hold REE only, but QA can
    be overridden at approval) are left out and counted in `excluded`.
    """
    raw = pd.read_sql_query(
        "SELECT v.*, p.short_citation AS _citation, p.pub_year AS _year, p.title AS _title "
        "FROM v_current_best v JOIN papers p ON p.paper_id = v.paper_id",
        conn,
    )
    df = schema.coerce_schema(raw)
    df.insert(0, "paper_id", raw["paper_id"].astype("Int64") if len(raw) else pd.Series(dtype="Int64"))
    labels = {
        int(pid): paper_label(c, y, t, int(pid))
        for pid, c, y, t in raw[["paper_id", "_citation", "_year", "_title"]].drop_duplicates("paper_id").itertuples(index=False)
    }
    # Two papers with one label (same citation) stay distinguishable.
    counts = pd.Series(list(labels.values())).value_counts()
    labels = {pid: (f"{lab} (#{pid})" if counts[lab] > 1 else lab) for pid, lab in labels.items()}
    df.insert(1, PAPER, df["paper_id"].map(labels))

    element = df[ELEMENT].astype("string").str.strip()
    df[ELEMENT] = element.astype(object).where(element.notna(), None)
    is_ree = element.isin(ATOMIC_NUMBER).fillna(False).astype(bool)
    excluded = element[~is_ree].fillna("(none)").value_counts().to_dict()
    return Rows(df=df[is_ree].reset_index(drop=True), excluded={str(k): int(v) for k, v in excluded.items()})
