"""Backstop SQL guard for assistant.tools.query_database (README §8).

This is independent of the system prompt: even a jailbroken model cannot write
through the `query_database` tool, and cannot read outside the allow-list,
because of three independent layers:

1. the connection is opened read-only via `get_readonly_conn()`;
2. `guard()` rejects anything but a single SELECT (optionally a leading WITH)
   and appends a LIMIT;
3. `authorized()` installs a SQLite authorizer for the duration of the query.

Layer 3 is the authoritative allow-list. It is a callback SQLite invokes while
*compiling* the statement, so it sees every relation the query actually touches
regardless of how it was written — comma joins, table-valued functions,
correlated subqueries, nested views. `guard()` also does a regex pre-check on
table names, but only to produce a friendlier error before execution: a
reference it misses is still denied by the authorizer.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager

# The only relations the assistant is allowed to read. v_current_best (not the
# raw `extractions` table) is the one the agent is told to use for "what data
# do we have" questions, so superseded prompt-version rows never appear.
ALLOWED_TABLES = {"v_current_best", "papers", "text_endpoints", "prompt_runs"}

DEFAULT_LIMIT = 500

# Only these can follow a WITH clause and write; every other DDL/DML verb
# (DROP, PRAGMA, ATTACH, VACUUM, ...) can only appear at the start of the
# statement, where the SELECT/WITH check below already rejects it. Keeping the
# list this short avoids matching ordinary SQL functions -- the previous list
# included REPLACE, which rejected any query using the replace() function.
_WRITE_AFTER_WITH = ("INSERT", "UPDATE", "DELETE")

_COMMENT_RE = re.compile(r"--.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+\"?(\w+)\"?", re.IGNORECASE)
_CTE_NAME_RE = re.compile(r"\b(\w+)\s+AS\s*\(", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)
_LEADING_RE = re.compile(r"^(SELECT|WITH)\b", re.IGNORECASE)

# Actions that carry no relation to check: SQLITE_SELECT is emitted once per
# SELECT, SQLITE_FUNCTION once per scalar/aggregate call. Everything the
# allow-list cares about arrives as SQLITE_READ, and every write action falls
# through to the default deny.
_UNCONDITIONAL_ACTIONS = frozenset({sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION})


class SQLGuardError(ValueError):
    """Raised when a query fails the read-only / allow-list guard."""


def _allow_list_error(names) -> str:
    return (
        f"query references table(s) not on the allow-list: {sorted(names)}. "
        f"Allowed: {sorted(ALLOWED_TABLES)}. Use v_current_best for extraction data."
    )


def guard(sql: str) -> str:
    """Validate `sql`, returning a (possibly LIMIT-augmented) safe query string.

    Raises SQLGuardError on anything but a single SELECT (optionally a leading
    WITH/CTE). Run the result inside `authorized(conn)` -- the table allow-list
    is enforced there, not here.
    """
    clean = _COMMENT_RE.sub(" ", sql).strip()
    if not clean:
        raise SQLGuardError("empty query")

    # Allow exactly one optional trailing semicolon; reject stacked statements.
    body = clean[:-1].strip() if clean.endswith(";") else clean
    if ";" in body:
        raise SQLGuardError("multiple statements are not allowed")

    if not _LEADING_RE.match(body):
        raise SQLGuardError("only SELECT statements (optionally a leading WITH/CTE) are allowed")

    for kw in _WRITE_AFTER_WITH:
        if re.search(rf"\b{kw}\b", body, re.IGNORECASE):
            raise SQLGuardError(f"forbidden keyword: {kw}")

    # A WITH clause defines its own CTE names, which are local aliases rather
    # than real tables. A CTE may NOT reuse an allow-listed name: SQLite lets a
    # CTE shadow a real view, and the authorizer reports reads inside it as
    # coming from that name -- so `WITH v_current_best AS (SELECT * FROM
    # extractions) ...` would otherwise launder a raw-table read.
    cte_names = {m.group(1).lower() for m in _CTE_NAME_RE.finditer(body)}
    shadowed = cte_names & ALLOWED_TABLES
    if shadowed:
        raise SQLGuardError(
            f"a CTE may not reuse the name of an allow-listed table: {sorted(shadowed)}"
        )

    # Advisory only -- see the module docstring. Catches the common case with a
    # clear message before the query runs; the authorizer is what guarantees it.
    tables = {m.group(1).lower() for m in _TABLE_REF_RE.finditer(body)}
    disallowed = tables - ALLOWED_TABLES - cte_names
    if disallowed:
        raise SQLGuardError(_allow_list_error(disallowed))

    if not _LIMIT_RE.search(body):
        body = f"{body.rstrip()} LIMIT {DEFAULT_LIMIT}"

    return body


def _make_authorizer(denied: set[str]):
    """Build a SQLite authorizer callback, recording what it rejected.

    The callback must not raise -- sqlite3 turns an exception into a bare deny
    and loses the reason -- so refusals are collected in `denied` and turned
    into a SQLGuardError by `authorized()`.
    """

    def authorizer(action, arg1, arg2, db_name, trigger_or_view):
        if action in _UNCONDITIONAL_ACTIONS:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = (arg1 or "").lower()
            # `trigger_or_view` names the view whose body is being expanded, so
            # a read of `extractions` reached through v_current_best is allowed
            # while a direct one is not. guard() has already rejected a CTE that
            # shadows an allow-listed name, which is the only way this could be
            # something other than a genuine view expansion.
            if table in ALLOWED_TABLES or (trigger_or_view or "").lower() in ALLOWED_TABLES:
                return sqlite3.SQLITE_OK
            denied.add(arg1 or "?")
            return sqlite3.SQLITE_DENY
        # Default deny: writes, PRAGMA, ATTACH, transactions -- anything that
        # isn't a read of an allow-listed relation.
        denied.add(f"action {action}")
        return sqlite3.SQLITE_DENY

    return authorizer


@contextmanager
def authorized(conn: sqlite3.Connection):
    """Enforce the table allow-list on `conn` for the duration of the block.

    Translates SQLite's opaque "access to X.Y is prohibited" into the same
    SQLGuardError message `guard()` raises, and always uninstalls the
    authorizer so the connection stays usable afterwards.
    """
    denied: set[str] = set()
    conn.set_authorizer(_make_authorizer(denied))
    try:
        yield
    except sqlite3.DatabaseError as e:
        if denied:
            raise SQLGuardError(_allow_list_error(denied)) from e
        raise
    finally:
        conn.set_authorizer(None)
