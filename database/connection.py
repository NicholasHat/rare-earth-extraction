"""SQLite connection helpers.

`get_conn()` returns a read/write connection and ensures the schema exists.
`get_readonly_conn()` opens the DB in read-only mode — used by Pillars B and C
(calculator, assistant) so a bad query can never mutate the dataset.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import config

_SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"

# Columns added to prompt_runs after its CREATE TABLE first shipped. SQLite's
# ALTER TABLE has no ADD COLUMN IF NOT EXISTS, so pre-existing DBs (whose
# CREATE TABLE IF NOT EXISTS is a no-op) need this applied in Python instead.
_PROMPT_RUN_USAGE_COLUMNS = {
    "input_tokens": "INTEGER",
    "output_tokens": "INTEGER",
    "cache_creation_input_tokens": "INTEGER",
    "cache_read_input_tokens": "INTEGER",
}

# Tracking-sheet columns added to papers after its CREATE TABLE first shipped
# (same pre-existing-DB story as the prompt_runs usage columns above).
_PAPER_TRACKING_COLUMNS = {
    "short_citation": "TEXT",
    "pub_year": "TEXT",
    "figures_used": "TEXT",
    "known_issues": "TEXT",
    "short_description": "TEXT",
}


# Views defined in schema.sql. Their CREATE ... IF NOT EXISTS is a no-op on a DB
# that already has them, so an edited view definition would never reach an
# existing master.db. Views hold no data — dropping them first is free and makes
# schema.sql authoritative for every future edit.
_VIEWS = ("v_current_best", "v_paper_summary")


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, sql_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def _ensure_added_columns(conn: sqlite3.Connection) -> None:
    """Apply every post-launch ALTER TABLE (no-op on a brand-new DB, where
    PRAGMA table_info of a not-yet-created table is empty)."""
    _ensure_columns(conn, "prompt_runs", _PROMPT_RUN_USAGE_COLUMNS)
    _ensure_columns(conn, "papers", _PAPER_TRACKING_COLUMNS)


def _apply_schema(conn: sqlite3.Connection) -> None:
    for view in _VIEWS:
        conn.execute(f"DROP VIEW IF EXISTS {view}")
    # Added columns must exist before the script recreates views that read them.
    _ensure_added_columns(conn)
    conn.executescript(_SCHEMA_SQL.read_text())
    conn.commit()


def init_db() -> None:
    """Create the data dir and apply the (idempotent) schema."""
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    try:
        _apply_schema(conn)
    finally:
        conn.close()


def get_conn() -> sqlite3.Connection:
    """Read/write connection. Ensures the schema exists on first use."""
    config.ensure_dirs()
    first_time = not config.DB_PATH.exists()
    conn = sqlite3.connect(config.DB_PATH)
    _apply_pragmas(conn)
    if first_time:
        _apply_schema(conn)
    else:
        # Cheap and idempotent — covers DBs created before a column was added.
        _ensure_added_columns(conn)
        conn.commit()
    return conn


def get_readonly_conn() -> sqlite3.Connection:
    """Read-only connection (Pillars B & C). Raises if the DB doesn't exist yet."""
    if not config.DB_PATH.exists():
        raise FileNotFoundError(
            f"master DB not found at {config.DB_PATH}; run an extraction first."
        )
    uri = f"file:{config.DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn
