"""validation/schema.py is the single source of truth for the extraction
columns: schema.sql, the startup migration, and the assistant's prompt must
agree with it, and an existing master.db must pick the added columns up."""
import sqlite3
from pathlib import Path

import pandas as pd

from database import connection
from validation import schema

ROOT = Path(__file__).resolve().parents[1]


def test_schema_is_the_labs_26_then_the_nine_added():
    assert len(schema.LAB_COLUMNS) == 26 and len(schema.ADDED_COLUMNS) == 9
    assert schema.COLUMNS[:26] == schema.LAB_COLUMNS and schema.LAB_COLUMNS[-1] == "Recovery %"
    assert set(schema.NUMERIC_COLUMNS) | set(schema.TEXT_COLUMNS) == set(schema.COLUMNS)


def test_schema_sql_creates_exactly_the_schema_columns_in_order(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(extractions)")]
    assert cols[3:-1] == schema.COLUMNS
    types = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(extractions)")}
    assert all(types[c] == schema.sql_type(c) for c in schema.COLUMNS)


def test_coerce_keeps_the_index_and_fills_absent_columns():
    df = pd.DataFrame({"pH": ["1.5", "x"], "Diluent": ["kerosene", None]}, index=[4, 7])
    out = schema.coerce_schema(df)
    assert list(out.columns) == schema.COLUMNS and list(out.index) == [4, 7]
    assert out["pH"].tolist()[0] == 1.5 and pd.isna(out["pH"].tolist()[1])
    assert out["Figure/Table"].isna().all() and out["Diluent"].tolist() == ["kerosene", None]


def _old_db(path):
    """A master.db as it was before 2026-11: 26 extraction columns, no licence."""
    conn = sqlite3.connect(path)
    conn.executescript((ROOT / "database" / "schema.sql").read_text())
    conn.execute("DROP VIEW v_paper_summary")
    conn.execute("DROP VIEW v_current_best")
    for c in schema.ADDED_COLUMNS:
        conn.execute(f'ALTER TABLE extractions DROP COLUMN "{c}"')
    for c in ("license", "license_evidence", "license_url", "license_origin"):
        conn.execute(f"ALTER TABLE papers DROP COLUMN {c}")
    conn.execute("INSERT INTO papers (content_sha256, pdf_path) VALUES ('swain', 's.pdf')")
    conn.execute("INSERT INTO prompt_runs (paper_id, prompt_version, prompt_sha256, model, status, reviewed_at) "
                 "VALUES (1, 'extraction_v5.1', 's', 'm', 'approved', datetime('now'))")
    conn.execute('INSERT INTO extractions (paper_id, prompt_run_id, "pH", "Extract%") VALUES (1, 1, 2.0, 40.0)')
    conn.commit()
    assert "Figure/Table" not in [r[1] for r in conn.execute("PRAGMA table_info(extractions)")]
    conn.close()


def test_an_existing_db_gains_the_columns_and_keeps_its_rows(tmp_path, monkeypatch):
    import config
    db = tmp_path / "master.db"
    _old_db(db)
    monkeypatch.setattr(config, "DB_PATH", db)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    connection.init_db()
    connection.init_db()     # idempotent
    conn = sqlite3.connect(db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(extractions)")]
    assert set(schema.COLUMNS) <= set(cols)
    assert conn.execute('SELECT "pH", "Extract%", "Figure/Table" FROM v_current_best').fetchall() == [(2.0, 40.0, None)]
    assert conn.execute("SELECT license FROM papers").fetchall() == [(None,)]
    assert conn.execute("SELECT license, data_rows FROM v_paper_summary").fetchall() == [(None, 1)]


def test_a_migration_file_mirrors_the_added_columns():
    text = (ROOT / "database" / "migrations" / "0006_extraction_v15_columns.sql").read_text()
    assert all(f'"{c}"' in text for c in schema.ADDED_COLUMNS)
    assert "license" in (ROOT / "database" / "migrations" / "0007_paper_license.sql").read_text()


def test_the_assistant_is_told_every_column():
    prompt = (ROOT / "assistant" / "system_prompt.md").read_text()
    assert all(f"`{c}`" in prompt for c in schema.COLUMNS)
