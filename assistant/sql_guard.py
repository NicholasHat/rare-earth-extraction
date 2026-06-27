"""Backstop SQL guard for assistant.tools.query_database (README §8).

This is independent of the system prompt: even a jailbroken model cannot
write through the `query_database` tool, because (1) the connection it runs
against is opened read-only via `get_readonly_conn()`, and (2) this guard
rejects anything but a single SELECT/CTE over a whitelisted set of tables
before the query ever reaches SQLite.
"""
from __future__ import annotations

import re

# The only relations the assistant is allowed to read. v_current_best (not the
# raw `extractions` table) is the one the agent is told to use for "what data
# do we have" questions, so superseded prompt-version rows never appear.
ALLOWED_TABLES = {"v_current_best", "papers", "text_endpoints", "prompt_runs"}

DEFAULT_LIMIT = 500

_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "TRIGGER",
    "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
)

_COMMENT_RE = re.compile(r"--.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+\"?(\w+)\"?", re.IGNORECASE)
_CTE_NAME_RE = re.compile(r"\b(\w+)\s+AS\s*\(", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)
_LEADING_RE = re.compile(r"^(SELECT|WITH)\b", re.IGNORECASE)


class SQLGuardError(ValueError):
    """Raised when a query fails the read-only / whitelist guard."""


def guard(sql: str) -> str:
    """Validate `sql`, returning a (possibly LIMIT-augmented) safe query string.

    Raises SQLGuardError on anything but a single SELECT (optionally a WITH
    CTE ending in one) over the whitelisted tables.
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

    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", body, re.IGNORECASE):
            raise SQLGuardError(f"forbidden keyword: {kw}")

    # A WITH clause can define its own CTE names; those are local aliases,
    # not real tables, so they're allowed alongside the real whitelist.
    cte_names = {m.group(1).lower() for m in _CTE_NAME_RE.finditer(body)}

    tables = {m.group(1).lower() for m in _TABLE_REF_RE.finditer(body)}
    disallowed = tables - ALLOWED_TABLES - cte_names
    if disallowed:
        raise SQLGuardError(
            f"query references table(s) not on the allow-list: {sorted(disallowed)}. "
            f"Allowed: {sorted(ALLOWED_TABLES)}. Use v_current_best for extraction data."
        )

    if not _LIMIT_RE.search(body):
        body = f"{body.rstrip()} LIMIT {DEFAULT_LIMIT}"

    return body
