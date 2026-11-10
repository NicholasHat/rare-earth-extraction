"""SQLite connection helpers.

`get_conn()` returns a read/write connection and ensures the schema exists.
`get_readonly_conn()` opens the DB in read-only mode — used by Pillars B and C
(calculator, assistant) so a bad query can never mutate the dataset.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import config
from validation import schema

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

# Licence provenance added to papers 2026-11 (the merge gate's record).
_PAPER_LICENSE_COLUMNS = {
    "license": "TEXT",
    "license_evidence": "TEXT",
    "license_url": "TEXT",
    "license_origin": "TEXT",
}

# The extraction columns appended after the lab's 26 (extraction_v15). Derived
# from validation/schema.py, the single source of truth for the column list.
_EXTRACTION_ADDED_COLUMNS = {c: schema.sql_type(c) for c in schema.ADDED_COLUMNS}


# Views defined in schema.sql. Their CREATE ... IF NOT EXISTS is a no-op on a DB
# that already has them, so an edited view definition would never reach an
# existing master.db. Views hold no data — dropping them first is free and makes
# schema.sql authoritative for every future edit.
_VIEWS = ("v_current_best", "v_paper_summary")


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if not existing:
        return  # table not created yet: schema.sql creates it with every column
    for name, sql_type in columns.items():
        if name not in existing:
            # Quoted: extraction column names carry spaces, '%', '/' and parentheses.
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {_quote(name)} {sql_type}")


def _ensure_added_columns(conn: sqlite3.Connection) -> None:
    """Apply every post-launch ALTER TABLE (no-op on a brand-new DB, where
    PRAGMA table_info of a not-yet-created table is empty)."""
    _ensure_columns(conn, "prompt_runs", _PROMPT_RUN_USAGE_COLUMNS)
    _ensure_columns(conn, "papers", _PAPER_TRACKING_COLUMNS)
    _ensure_columns(conn, "papers", _PAPER_LICENSE_COLUMNS)
    _ensure_columns(conn, "extractions", _EXTRACTION_ADDED_COLUMNS)


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


# Keep this many daily backups of the master DB (see backup_once_a_day).
_BACKUPS_KEPT = 14


def backup_once_a_day(conn: sqlite3.Connection) -> Path | None:
    """Copy the connection's database file to `<db dir>/backups/<stem>-<date>.db`
    unless today's copy already exists, and prune all but the newest
    _BACKUPS_KEPT. Called by the one write path (merge.commit_extraction)
    before it writes, so the first merge of each day leaves a restorable
    copy of the DB as it was. Uses SQLite's online backup API (consistent
    under WAL). A no-op for an in-memory DB; returns the copy's path when
    one was made."""
    row = conn.execute("PRAGMA database_list").fetchone()
    db_file = row[2] if row is not None else ""
    if not db_file:
        return None
    db_path = Path(db_file)
    backups = db_path.parent / "backups"
    target = backups / f"{db_path.stem}-{date.today().isoformat()}.db"
    if target.exists():
        return None
    backups.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".db.partial")
    dest = sqlite3.connect(partial)
    try:
        conn.backup(dest)
    finally:
        dest.close()
    partial.replace(target)
    for old in sorted(backups.glob(f"{db_path.stem}-????-??-??.db"))[:-_BACKUPS_KEPT]:
        old.unlink(missing_ok=True)
    return target
