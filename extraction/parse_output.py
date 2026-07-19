"""Parse the model's raw response into a (DataFrame, text_endpoints) pair.

The OUTPUT CONTRACT asks for a single JSON object with `rows` and
`text_endpoints`. Two `rows` shapes are supported:
  - extraction_v5.1–v6: a list of objects, one per row, keyed by column name.
  - extraction_v7+: a compact positional form — a top-level `columns` list of
    the 26 column names plus `rows` as a list of same-length value arrays —
    to avoid paying output tokens for 26 verbose keys on every single row.
This parser is deliberately tolerant: it pulls the JSON out of a ```json
fenced block when present, else falls back to the outermost {...} span, so a
stray sentence around the block doesn't break the run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pandas as pd

from validation import schema

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ParsedExtraction:
    df: pd.DataFrame                 # coerced to the 26-column schema
    text_endpoints: list[dict]
    coercion_failures: int           # numeric cells that were non-null but unparseable
    raw_text: str


class ParseError(ValueError):
    """Raised when no JSON object can be recovered from the model output."""


def _extract_json_object(text: str) -> dict:
    # Prefer the last fenced ```json block (the model may show work first).
    matches = _FENCE_RE.findall(text)
    candidates = list(matches)
    if not candidates:
        # Fallback: outermost brace span.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidates = [text[start : end + 1]]
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ParseError("no parseable JSON object found in model output")


def _count_coercion_failures(raw_rows: list[dict]) -> int:
    """Count cells that held a non-empty value but won't parse as a number."""
    failures = 0
    for row in raw_rows:
        for col in schema.NUMERIC_COLUMNS:
            v = row.get(col)
            if v is None or v == "":
                continue
            if pd.isna(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]):
                failures += 1
    return failures


def _expand_positional_rows(columns: object, rows: list) -> list[dict]:
    """extraction_v7+ compact form: {"columns": [...], "rows": [[v0, v1, ...], ...]}.

    Validated against the canonical schema before use. Without this, a
    typo'd/renamed column would silently read as "missing" once
    schema.coerce_schema matches by exact name — the data isn't dropped by
    this function, but it's orphaned under the wrong key and coerce_schema
    nulls it out with no error. A row whose length doesn't match `columns`
    would silently misalign every value after the gap via zip() truncation.
    Both are now hard failures instead of silent data loss.
    """
    if not isinstance(columns, list):
        raise ParseError("array-shaped 'rows' requires a top-level 'columns' list")
    if set(columns) != set(schema.COLUMNS):
        missing = sorted(set(schema.COLUMNS) - set(columns))
        unexpected = sorted(set(columns) - set(schema.COLUMNS))
        detail = "; ".join(
            part for part in (
                f"missing: {missing}" if missing else "",
                f"unexpected: {unexpected}" if unexpected else "",
            ) if part
        )
        raise ParseError(f"'columns' does not match the 26-column schema ({detail})")

    expanded = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            got = len(row) if isinstance(row, list) else type(row).__name__
            raise ParseError(
                f"row length {got} does not match 'columns' length {len(columns)}"
            )
        expanded.append(dict(zip(columns, row)))
    return expanded


def parse(raw_text: str) -> ParsedExtraction:
    obj = _extract_json_object(raw_text)
    raw_rows = obj.get("rows") or []
    if not isinstance(raw_rows, list):
        raise ParseError("'rows' is not a list")
    if raw_rows and isinstance(raw_rows[0], list):
        raw_rows = _expand_positional_rows(obj.get("columns"), raw_rows)
    endpoints = obj.get("text_endpoints") or []
    if not isinstance(endpoints, list):
        endpoints = []

    coercion_failures = _count_coercion_failures(raw_rows)
    raw_df = pd.DataFrame(raw_rows) if raw_rows else pd.DataFrame(columns=schema.COLUMNS)
    df = schema.coerce_schema(raw_df)

    return ParsedExtraction(
        df=df,
        text_endpoints=[ep for ep in endpoints if isinstance(ep, dict)],
        coercion_failures=coercion_failures,
        raw_text=raw_text,
    )
