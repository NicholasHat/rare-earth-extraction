"""Publish a copy of the master DB for the hosted read-only viewer (viewer.py).

    python -m database.snapshot [DEST]        # default: viewer_data/master.db

The copy is taken with SQLite's online backup (consistent even while the app
is open), then trimmed of what a viewer has no use for and what shouldn't
leave this machine: each run's raw model output (kept in the working DB and
under data/transcripts/) and the local directory of each PDF path. Every
table, view and row the Database and Analysis pages read is unchanged.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import config

DEFAULT_DEST = config.ROOT / "viewer_data" / "master.db"


def write(dest: Path = DEFAULT_DEST, source: Path | None = None) -> Path:
    source = source or config.DB_PATH
    if not source.exists():
        raise FileNotFoundError(f"no master DB at {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    out = sqlite3.connect(tmp)
    try:
        src.backup(out)
        out.execute("UPDATE prompt_runs SET raw_response = NULL")
        for paper_id, pdf_path in out.execute("SELECT paper_id, pdf_path FROM papers").fetchall():
            out.execute("UPDATE papers SET pdf_path = ? WHERE paper_id = ?", (Path(pdf_path).name, paper_id))
        out.commit()
        out.execute("PRAGMA journal_mode = DELETE")   # one file, no -wal beside it
        out.execute("VACUUM")
    finally:
        out.close()
        src.close()
    os.replace(tmp, dest)
    for stale in ("-wal", "-shm"):   # left beside the old copy by a local viewer run
        dest.with_name(dest.name + stale).unlink(missing_ok=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    dest = write(Path(args[0]) if args else DEFAULT_DEST)
    conn = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        rows = conn.execute("SELECT COUNT(*) FROM v_current_best").fetchone()[0]
    finally:
        conn.close()
    print(f"{dest}: {papers} papers, {rows} rows, {dest.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
