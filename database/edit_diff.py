"""What the reviewer changed: the staged extraction vs the rows actually merged.

`table_diff` is pure. `merge.commit_extraction` calls it on approval and stores
the result as `review_log.edited_diff_json` in the merge's own transaction, so
the audit trail says not just that a paper was approved but what was edited,
deleted or added on the way.

Row identity is the DataFrame index label. The review page keeps the staged
table's labels through everything a reviewer can do — dropping QA-flagged
rows (`DataFrame.drop`) and editing in `st.data_editor`, which carries labels
through cell edits and deletions and gives added rows new ones — so a label
present on both sides is the same row, and a reorder is no change. Rows are
reported by their 1-based position in the staged table, the numbering QA
flags and the review note's "dropped rows" use. One limit: Streamlit numbers
an added row one past the highest label left, so a row added after deleting
the last staged row reuses that row's label and is reported as that row's
cells changing rather than as a delete plus an add — still exactly what
reconstructs the merged table.

Both tables pass through `schema.coerce_schema` first, so dtype noise is no
change: NaN, None and blank text are all "empty"; numbers compare to 12
significant digits (1, 1.0 and "1.0" in a numeric column are equal); and a
text cell equals a number that prints the same (a Reference No. of 1 and "1").
An untouched paper yields None.

Approvals from before this was written have a NULL diff. Their edits can be
recovered, read-only, from each run's stored `raw_response` (what the model
returned, i.e. the staged table) against its merged rows:

    python -m database.edit_diff [path/to/master.db]

(`recover`, which has to re-find row identity by content — see `realign`.)

Shape (fixed; deleted and changed in staged-row order, added in table order,
cells in schema column order; empty cells left out of `values`):

    {"deleted": [{"row": 7, "values": {"pH": 2.5, ...}}],
     "added":   [{"values": {...}}],
     "changed": [{"row": 3, "column": "pH", "old": 1.2, "new": 1.5}]}
"""
from __future__ import annotations

import difflib
import json
import sqlite3
import sys

import pandas as pd

from validation import schema


def _cell(v):
    """A cell as plain JSON-able Python, None for anything empty."""
    if v is None:
        return None
    if isinstance(v, str):
        return v if v.strip() else None
    if pd.isna(v):
        return None
    if hasattr(v, "item"):          # numpy scalar
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _canon(v) -> str | None:
    """What two cells must share to be equal: numbers to 12 significant digits
    (float round-trip noise and 1 vs 1.0 vanish), everything as text (so a
    Reference No. read back from SQLite as '1' equals the model's 1)."""
    v = _cell(v)
    if v is None:
        return None
    return format(v, ".12g") if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _values(row: pd.Series) -> dict:
    return {c: v for c in schema.COLUMNS if (v := _cell(row[c])) is not None}


def table_diff(extracted: pd.DataFrame, merged: pd.DataFrame) -> dict | None:
    """The reviewer's edits turning `extracted` into `merged`, or None if none."""
    before, after = schema.coerce_schema(extracted), schema.coerce_schema(merged)
    if not (before.index.is_unique and after.index.is_unique):
        # No usable identity: fall back to position.
        before, after = before.reset_index(drop=True), after.reset_index(drop=True)
    row_no = {label: i + 1 for i, label in enumerate(before.index)}

    deleted = [{"row": row_no[k], "values": _values(before.loc[k])}
               for k in before.index if k not in after.index]
    added = [{"values": _values(after.loc[k])} for k in after.index if k not in row_no]
    changed = []
    for k in before.index:
        if k not in after.index:
            continue
        for c in schema.COLUMNS:
            old, new = before.at[k, c], after.at[k, c]
            if _canon(old) != _canon(new):
                changed.append({"row": row_no[k], "column": c, "old": _cell(old), "new": _cell(new)})

    if not (deleted or added or changed):
        return None
    return {"deleted": deleted, "added": added, "changed": changed}


def to_json(diff: dict | None) -> str | None:
    """Compact, deterministic JSON for review_log.edited_diff_json (None stays NULL)."""
    if diff is None:
        return None
    return json.dumps(diff, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def summary(diff_json: str | None) -> str:
    """One line for a review-log listing: '2 rows deleted, 1 row added, 3 cells changed'."""
    if not isinstance(diff_json, str) or not diff_json:   # NULL reads back as None or NaN
        return ""
    d = json.loads(diff_json)
    return ", ".join(f"{len(d[k])} {noun}{'s' if len(d[k]) != 1 else ''} {verb}"
                     for k, noun, verb in (("deleted", "row", "deleted"), ("added", "row", "added"),
                                           ("changed", "cell", "changed")) if d[k])


def realign(extracted: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """`merged` relabelled with `extracted`'s row labels, for tables whose row
    identity was lost — rows read back from the DB, where the labels the review
    page carried are gone. Only for recovering edits made before
    edited_diff_json was written; live approvals have real labels.

    Rows are matched by content in order (difflib over whole rows); within a
    stretch that differs, a merged row is paired, in order, with the staged
    row it shares the most cells with, if that is at least half the cells
    either has filled — a cell edit — and is otherwise an added row. A
    heuristic, exact for what the review page is used for: deleting rows and
    correcting a few cells of a row.
    """
    before = schema.coerce_schema(extracted).reset_index(drop=True)
    after = schema.coerce_schema(merged).reset_index(drop=True)

    def keys(df):
        return [tuple(map(_canon, row)) for row in df[schema.COLUMNS].itertuples(index=False)]

    def likeness(a, b) -> float:
        filled = sum(x is not None or y is not None for x, y in zip(a, b))
        return sum(x == y and x is not None for x, y in zip(a, b)) / max(filled, 1)

    a, b = keys(before), keys(after)
    labels: list[int] = []
    fresh = len(a)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            labels += range(i1, i2)
            continue
        for j in range(j1, j2):
            best = max(range(i1, i2), key=lambda i: likeness(a[i], b[j]), default=None)
            if best is not None and likeness(a[best], b[j]) >= 0.5:
                labels.append(best)
                i1 = best + 1           # keep the pairing in order
            else:
                labels.append(fresh)
                fresh += 1
    return after.set_axis(labels)


def recover(conn: sqlite3.Connection, prompt_run_id: int) -> dict | None:
    """The edits made at review of an approved run, rebuilt from its stored
    raw_response and merged rows — for approvals from before edited_diff_json
    was written. Read-only."""
    from extraction import parse_output   # the DB layer doesn't otherwise parse model output

    (raw,) = conn.execute("SELECT raw_response FROM prompt_runs WHERE prompt_run_id = ?",
                          (prompt_run_id,)).fetchone()
    cols = ", ".join(f'"{c}"' for c in schema.COLUMNS)
    merged = pd.read_sql_query(
        f"SELECT {cols} FROM extractions WHERE prompt_run_id = ? ORDER BY extraction_id",
        conn, params=(prompt_run_id,))
    extracted = parse_output.parse(raw).df
    return table_diff(extracted, realign(extracted, merged))


def main(argv: list[str] | None = None) -> int:
    """Print the recovered edits of every approved run (read-only)."""
    import config

    args = sys.argv[1:] if argv is None else argv
    path = args[0] if args else config.DB_PATH
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        runs = conn.execute(
            "SELECT pr.prompt_run_id, pr.paper_id, rl.edited_diff_json FROM prompt_runs pr "
            "LEFT JOIN review_log rl ON rl.prompt_run_id = pr.prompt_run_id AND rl.action = 'approve' "
            "WHERE pr.status = 'approved' AND pr.raw_response IS NOT NULL ORDER BY pr.prompt_run_id"
        ).fetchall()
        for run_id, paper_id, recorded in runs:
            diff = to_json(recover(conn, run_id))
            source = "recorded" if recorded else "recovered"
            print(f"paper {paper_id} run {run_id} ({source}): {summary(recorded or diff) or 'no edits'}")
            if diff and not recorded:
                print(f"  {diff}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
