"""Tests for assistant.tools — the plain functions behind the agent's tools."""
import json
import sqlite3

import pandas as pd

from assistant import tools
from database import merge
from validation import schema


def _sample_df(extractant="Cyanex 272", element="La", n=3):
    rows = [
        {
            schema.ELEMENT_COLUMN: element,
            "pH": 1.0 + 0.5 * i,
            "Extract%": 10.0 + 15 * i,
            "Extractant": extractant,
            "Extractant Conc. (mM)": 500.0,
        }
        for i in range(n)
    ]
    return schema.coerce_schema(pd.DataFrame(rows))


def _commit(conn, sha="hash1", doi="10.1/x", extractant="Cyanex 272", element="La"):
    return merge.commit_extraction(
        conn,
        content_sha256=sha,
        pdf_path=f"data/incoming/{sha}.pdf",
        df=_sample_df(extractant, element),
        text_endpoints=[],
        prompt_version="extraction_v5.2",
        prompt_sha256="psha",
        model="claude-opus-4-8",
        qa_passed=True,
        qa_report_json="[]",
        raw_response="{}",
        doi=doi,
    )


# --- query_database -------------------------------------------------------- #

def test_query_database_returns_rows_as_json(conn):
    _commit(conn)
    out = json.loads(tools.query_database(conn, 'SELECT "Extract%" FROM v_current_best ORDER BY 1'))
    assert out == [{"Extract%": 10.0}, {"Extract%": 25.0}, {"Extract%": 40.0}]


def test_query_database_rejects_disallowed_table():
    out = json.loads(tools.query_database(sqlite3.connect(":memory:"), "SELECT * FROM extractions"))
    assert "error" in out
    assert "allow-list" in out["error"]


def test_query_database_rejects_mutation(conn):
    out = json.loads(tools.query_database(conn, "DELETE FROM papers"))
    assert "error" in out


def test_query_database_handles_sql_error_gracefully(conn):
    out = json.loads(tools.query_database(conn, "SELECT nonexistent_column FROM papers"))
    assert "error" in out


# --- list_extractants ------------------------------------------------------- #

def test_list_extractants_empty(conn):
    assert json.loads(tools.list_extractants(conn)) == []


def test_list_extractants_returns_sorted_distinct(conn):
    _commit(conn, sha="a", doi="10.1/a", extractant="D2EHPA")
    _commit(conn, sha="b", doi="10.1/b", extractant="Cyanex 272")
    assert json.loads(tools.list_extractants(conn)) == ["Cyanex 272", "D2EHPA"]


# --- calculator -------------------------------------------------------------- #

def test_calculator_ppm_to_mM():
    out = json.loads(tools.calculator("ppm_to_mM", element="La", ppm=100.0))
    assert out["result"] == 100.0 / 138.91


def test_calculator_extractant_conc_from_ratio():
    out = json.loads(tools.calculator("extractant_conc_from_ratio", ree_mM=0.72, molar_ratio_ex_per_ree=694.4))
    assert out["result"] == 0.72 * 694.4


def test_calculator_unknown_operation_returns_error():
    out = json.loads(tools.calculator("not_a_real_op"))
    assert "error" in out


def test_calculator_unknown_element_returns_error_not_raise():
    out = json.loads(tools.calculator("ppm_to_mM", element="Xx", ppm=100.0))
    assert "error" in out


def test_calculator_missing_required_arg_returns_error_not_raise():
    out = json.loads(tools.calculator("ppm_to_mM", element="La"))  # ppm missing -> None
    assert "error" in out
