"""The hosted read-only viewer: database.snapshot publishes a trimmed copy of
the master DB, and viewer.py serves the Database and Analysis pages over it."""
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config
from database import connection, merge, snapshot
from ingestion.licensing import License
from validation import schema

VIEWER = str(Path(__file__).resolve().parents[1] / "viewer.py")


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "master.db")
    for name in ("INCOMING_DIR", "STAGING_DIR", "EXPORTS_DIR", "TRANSCRIPTS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    conn = connection.get_conn()
    try:
        merge.commit_extraction(
            conn, license=License("cc-by", origin="manifest"), content_sha256="abc",
            pdf_path="/Users/someone/project/data/incoming/abc.pdf",
            df=schema.coerce_schema(pd.DataFrame(
                [{schema.ELEMENT_COLUMN: "La", "pH": 1.0 + i, "Extract%": 20.0 * i, schema.FIGURE_COLUMN: "Fig. 1"}
                 for i in range(4)])),
            text_endpoints=[], prompt_version="extraction_v16", prompt_sha256="p", model="m",
            qa_passed=True, qa_report_json="[]", raw_response="the model's whole output", doi="10.1/abc",
            tracking={"short_citation": "Abc 2020", "pub_year": "2020"},
        )
    finally:
        conn.close()
    return tmp_path / "master.db"


def test_snapshot_keeps_every_row_and_drops_raw_output_and_local_paths(db, tmp_path):
    dest = snapshot.write(tmp_path / "out" / "master.db")
    conn = sqlite3.connect(dest)
    assert conn.execute("SELECT COUNT(*) FROM v_current_best").fetchone()[0] == 4
    assert conn.execute("SELECT pdf_path, license FROM papers").fetchall() == [("abc.pdf", "cc-by")]
    assert conn.execute("SELECT raw_response, prompt_version FROM prompt_runs").fetchall() == [(None, "extraction_v16")]
    assert sorted(p.name for p in dest.parent.iterdir()) == ["master.db"]
    # the working DB is untouched
    assert sqlite3.connect(db).execute("SELECT raw_response FROM prompt_runs").fetchone()[0] == "the model's whole output"
    assert snapshot.main([str(dest)]) == 0     # re-publishing replaces the file


def test_viewer_serves_only_the_database_and_analysis_pages(db, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", snapshot.write(tmp_path / "viewer" / "master.db"))
    at = AppTest.from_file(VIEWER, default_timeout=30).run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Master Database"]
    assert [m.value for m in at.metric][:3] == ["1", "1", "4"]
    source = Path(VIEWER).read_text()
    assert "app.py" not in source.split('"""')[2] and "Assistant" not in source.split('"""')[2]
