"""Exhaustive tests for assistant.sql_guard — the hard backstop independent
of the system prompt (README §8)."""
import json

import pytest

from assistant import sql_guard, tools
from assistant.sql_guard import SQLGuardError, guard


def test_simple_select_passes_and_gets_limit_appended():
    out = guard('SELECT * FROM v_current_best')
    assert out == 'SELECT * FROM v_current_best LIMIT 500'


def test_existing_limit_not_duplicated():
    out = guard('SELECT * FROM v_current_best LIMIT 10')
    assert out.count("LIMIT") == 1
    assert "LIMIT 10" in out


def test_with_cte_passes():
    sql = 'WITH x AS (SELECT pH FROM v_current_best) SELECT * FROM x'
    out = guard(sql)
    assert out.startswith("WITH")


def test_trailing_semicolon_allowed():
    out = guard('SELECT 1;')
    assert ";" not in out


def test_lowercase_select_passes():
    out = guard('select * from papers')
    assert "LIMIT" in out


@pytest.mark.parametrize("table", ["extractions", "review_log", "sqlite_master"])
def test_disallowed_table_rejected(table):
    with pytest.raises(SQLGuardError, match="allow-list"):
        guard(f"SELECT * FROM {table}")


@pytest.mark.parametrize(
    "table", ["v_current_best", "v_paper_summary", "papers", "text_endpoints", "prompt_runs"]
)
def test_allowed_tables_pass(table):
    guard(f"SELECT * FROM {table}")  # should not raise


def test_join_table_also_checked():
    with pytest.raises(SQLGuardError, match="allow-list"):
        guard('SELECT * FROM papers JOIN review_log ON 1=1')


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO papers (doi) VALUES ('x')",
        "UPDATE papers SET doi='x'",
        "DELETE FROM papers",
        "DROP TABLE papers",
        "ALTER TABLE papers ADD COLUMN x",
        "CREATE TABLE evil (x INT)",
        "PRAGMA table_info(papers)",
        "ATTACH DATABASE 'x.db' AS x",
        "VACUUM",
        "BEGIN; SELECT 1",
    ],
)
def test_mutating_or_pragma_statements_rejected(sql):
    with pytest.raises(SQLGuardError):
        guard(sql)


def test_stacked_statements_rejected():
    with pytest.raises(SQLGuardError, match="multiple statements"):
        guard("SELECT 1 FROM papers; DROP TABLE papers")


def test_non_select_leading_statement_rejected():
    with pytest.raises(SQLGuardError, match="only SELECT"):
        guard("EXPLAIN SELECT * FROM papers")


def test_empty_query_rejected():
    with pytest.raises(SQLGuardError, match="empty"):
        guard("   ")


def test_sql_comment_does_not_smuggle_forbidden_keyword_past_detection():
    # The forbidden-keyword check still scans inside the (stripped) body; this
    # just confirms comments are stripped before the rest of the guard runs,
    # not used to hide a second statement.
    with pytest.raises(SQLGuardError):
        guard("SELECT 1 FROM papers /* comment */; DROP TABLE papers")


def test_quoted_column_name_with_percent_does_not_break_guard():
    out = guard('SELECT "Extract%" FROM v_current_best WHERE "Extract%" > 90')
    assert "LIMIT" in out


def test_replace_function_is_not_mistaken_for_a_write():
    """`replace()` is an ordinary string function; only `REPLACE INTO` writes,
    and that can't get past the leading SELECT/WITH check."""
    out = guard("""SELECT replace("Extractant", 'a', 'b') FROM v_current_best""")
    assert "LIMIT" in out


def test_cte_may_not_shadow_an_allow_listed_relation():
    """SQLite lets a CTE shadow a real view, and the authorizer would then
    report reads inside it as coming from that view."""
    with pytest.raises(SQLGuardError, match="may not reuse"):
        guard("WITH v_current_best AS (SELECT * FROM extractions) SELECT * FROM v_current_best")


# --- the authorizer: enforcement the regex pre-check cannot do ----------------

def _run(conn, sql):
    with sql_guard.authorized(conn):
        return conn.execute(guard(sql)).fetchall()


@pytest.mark.parametrize(
    "sql",
    [
        # Old-style comma joins: _TABLE_REF_RE only sees the table after FROM.
        "SELECT * FROM v_current_best, extractions",
        "SELECT * FROM papers, sqlite_master",
        "SELECT * FROM papers p, review_log r",
        # Table-valued function: no FROM/JOIN keyword precedes it.
        "SELECT * FROM papers, pragma_table_info('extractions')",
        # Correlated subquery.
        "SELECT (SELECT COUNT(*) FROM extractions) AS n FROM papers",
    ],
)
def test_authorizer_denies_relations_the_regex_pre_check_misses(conn, sql):
    with pytest.raises(SQLGuardError, match="allow-list"):
        _run(conn, sql)


def test_authorizer_allows_reading_through_an_allow_listed_view(conn):
    """v_current_best is a view over `extractions`; expanding it reads the raw
    table, which must stay allowed even though a direct read is not."""
    assert _run(conn, "SELECT * FROM v_current_best") == []


def test_authorizer_allows_the_doubly_nested_summary_view(conn):
    """v_paper_summary expands v_current_best, which expands `extractions` —
    every level of that chain must clear the authorizer."""
    assert _run(conn, "SELECT * FROM v_paper_summary") == []


def test_authorizer_allows_a_plain_cte(conn):
    assert _run(conn, "WITH x AS (SELECT paper_id FROM papers) SELECT * FROM x") == []


def test_authorizer_is_uninstalled_after_a_denial(conn):
    """A denial must not leave the shared connection unusable."""
    with pytest.raises(SQLGuardError):
        _run(conn, "SELECT * FROM papers, review_log")
    assert conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0] == 0


def test_query_database_reports_a_denied_relation_instead_of_crashing(conn):
    out = json.loads(tools.query_database(conn, "SELECT * FROM v_current_best, extractions"))
    assert "allow-list" in out["error"]
