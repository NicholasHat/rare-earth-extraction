"""DB cross-reference for the calculator (README §7) — optional, high-value.

Reads `v_current_best`, never the raw `extractions` table, so a superseded
prompt-version's rows never pollute the typical-range stats — and only
through `get_readonly_conn()`, so a calculator query can never mutate the
master DB (README §5).
"""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass


@dataclass
class RangeSummary:
    n_papers: int
    n_rows: int
    pH_min: float | None
    pH_median: float | None
    pH_max: float | None
    conc_min: float | None
    conc_median: float | None
    conc_max: float | None
    extract_pct_min: float | None
    extract_pct_median: float | None
    extract_pct_max: float | None


def _stats(values: list[float | None]) -> tuple[float | None, float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, None
    return min(clean), statistics.median(clean), max(clean)


def typical_ranges(conn: sqlite3.Connection, extractant: str, element: str) -> RangeSummary | None:
    """Typical pH / extractant-conc. / Extract% for a prior (extractant, element) pair.

    `element` is matched with LIKE so it still works if a row's element field
    ever holds more than the bare symbol; in current data it's an exact symbol.
    """
    rows = conn.execute(
        """
        SELECT paper_id, pH, "Extractant Conc. (mM)" AS conc, "Extract%" AS pct
        FROM v_current_best
        WHERE "Extractant" = ?
          AND "Rare Earth Elements (REY:La, Ce, Nd)" LIKE '%' || ? || '%'
        """,
        (extractant, element),
    ).fetchall()
    if not rows:
        return None

    pH_min, pH_median, pH_max = _stats([r["pH"] for r in rows])
    conc_min, conc_median, conc_max = _stats([r["conc"] for r in rows])
    pct_min, pct_median, pct_max = _stats([r["pct"] for r in rows])

    return RangeSummary(
        n_papers=len({r["paper_id"] for r in rows}),
        n_rows=len(rows),
        pH_min=pH_min, pH_median=pH_median, pH_max=pH_max,
        conc_min=conc_min, conc_median=conc_median, conc_max=conc_max,
        extract_pct_min=pct_min, extract_pct_median=pct_median, extract_pct_max=pct_max,
    )
