"""Plain, testable implementations of the Pillar C agent's tools (README §8).

These take an explicit `sqlite3.Connection` and return JSON strings (never
raise on bad input — a tool result the model can read and recover from beats
a crashed turn). The @beta_tool-decorated wrappers the agent actually calls
live in assistant/agent.py and just add connection-management glue here.
"""
from __future__ import annotations

import json
import sqlite3
import time

from calculator import conversions
from calculator.atomic_mass import atomic_mass

from . import sql_guard

_QUERY_TIMEOUT_S = 5.0

_CALC_OPS = {
    "ppm_to_mM",
    "mM_to_ppm",
    "extractant_conc_from_ratio",
    "molar_ratio_from_conc",
    "volume_for_target_moles",
    "mmol_in_volume",
    "mass_mg_in_volume",
}


def query_database(conn: sqlite3.Connection, sql: str) -> str:
    """Run a guarded, read-only SELECT. Returns a JSON array of row objects,
    or `{"error": ...}` if the query was rejected or failed."""
    try:
        safe_sql = sql_guard.guard(sql)
    except sql_guard.SQLGuardError as e:
        return json.dumps({"error": str(e)})

    deadline = time.monotonic() + _QUERY_TIMEOUT_S
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    try:
        rows = conn.execute(safe_sql).fetchall()
    except sqlite3.Error as e:
        return json.dumps({"error": f"query failed: {e}"})
    finally:
        conn.set_progress_handler(None, 0)

    return json.dumps([dict(r) for r in rows], default=str)


def list_extractants(conn: sqlite3.Connection) -> str:
    """Distinct extractant names already present in the approved dataset, as a JSON array."""
    rows = conn.execute(
        'SELECT DISTINCT "Extractant" FROM v_current_best '
        'WHERE "Extractant" IS NOT NULL ORDER BY 1'
    ).fetchall()
    return json.dumps([r[0] for r in rows])


def calculator(
    operation: str,
    *,
    element: str | None = None,
    ppm: float | None = None,
    mM: float | None = None,
    ree_mM: float | None = None,
    molar_ratio_ex_per_ree: float | None = None,
    extractant_mM: float | None = None,
    target_mmol: float | None = None,
    conc_mM: float | None = None,
    volume_mL: float | None = None,
) -> str:
    """Thin wrapper over calculator/conversions.py (README §8). Returns
    `{"result": ...}` or `{"error": ...}` as a JSON string — never raises."""
    if operation not in _CALC_OPS:
        return json.dumps({"error": f"unknown operation {operation!r}; choices: {sorted(_CALC_OPS)}"})
    try:
        mass = atomic_mass(element) if element else None
        if operation == "ppm_to_mM":
            result = conversions.ppm_to_mM(ppm, mass)
        elif operation == "mM_to_ppm":
            result = conversions.mM_to_ppm(mM, mass)
        elif operation == "extractant_conc_from_ratio":
            result = conversions.extractant_conc_from_ratio(ree_mM, molar_ratio_ex_per_ree)
        elif operation == "molar_ratio_from_conc":
            result = conversions.molar_ratio_from_conc(extractant_mM, ree_mM)
        elif operation == "volume_for_target_moles":
            result = conversions.volume_for_target_moles(target_mmol, conc_mM)
        elif operation == "mmol_in_volume":
            result = conversions.mmol_in_volume(conc_mM, volume_mL)
        else:  # mass_mg_in_volume
            result = conversions.mass_mg_in_volume(conc_mM, volume_mL, mass)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": str(e)})
    return json.dumps({"result": result})
