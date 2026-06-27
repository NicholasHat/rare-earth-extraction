"""Exhaustive tests for assistant.sql_guard — the hard backstop independent
of the system prompt (README §8)."""
import pytest

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


@pytest.mark.parametrize("table", ["v_current_best", "papers", "text_endpoints", "prompt_runs"])
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
