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


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row


def _ensure_prompt_run_usage_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(prompt_runs)")}
    for name, sql_type in _PROMPT_RUN_USAGE_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE prompt_runs ADD COLUMN {name} {sql_type}")


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL.read_text())
    _ensure_prompt_run_usage_columns(conn)
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
        _ensure_prompt_run_usage_columns(conn)
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
