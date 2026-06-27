"""Shared fixtures: an in-memory master DB built from the real schema."""
import sqlite3
from pathlib import Path

import pytest

_SCHEMA = Path(__file__).resolve().parents[1] / "database" / "schema.sql"


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA.read_text())
    c.execute("PRAGMA foreign_keys = ON;")
    yield c
    c.close()
