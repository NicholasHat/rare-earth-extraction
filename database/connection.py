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


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row


def init_db() -> None:
    """Create the data dir and apply the (idempotent) schema."""
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.executescript(_SCHEMA_SQL.read_text())
        conn.commit()
    finally:
        conn.close()


def get_conn() -> sqlite3.Connection:
    """Read/write connection. Ensures the schema exists on first use."""
    config.ensure_dirs()
    first_time = not config.DB_PATH.exists()
    conn = sqlite3.connect(config.DB_PATH)
    _apply_pragmas(conn)
    if first_time:
        conn.executescript(_SCHEMA_SQL.read_text())
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
