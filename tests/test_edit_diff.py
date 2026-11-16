"""database.edit_diff: what a reviewer changed between the staged extraction
and the merged rows — and merge recording it atomically on the approval."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from database import edit_diff, extractions_repo, merge, review_repo
from ingestion.licensing import License
from validation import schema

EL = schema.ELEMENT_COLUMN


def _table(n=5):
    return schema.coerce_schema(pd.DataFrame(
        [{EL: "La", "pH": 1.0 + 0.5 * i, "Extract%": 10.0 * (i + 1), "Figure/Table": "Fig. 2"}
         for i in range(n)]))


def test_an_untouched_table_is_no_diff():
    t = _table()
    assert edit_diff.table_diff(t, t.copy()) is None
    assert edit_diff.to_json(None) is None


def test_dtype_noise_is_no_diff():
    t = _table()
    noisy = t.copy()
    noisy["pH"] = noisy["pH"].astype(object).map(str)           # "1.0" for 1.0
    noisy["Extract%"] = noisy["Extract%"].astype(np.float32)     # float round trip
    noisy["Reference No."] = ""                                 # blank text is empty
    noisy["Diluent"] = np.nan                                   # NaN is empty
    t["Extract%"] = t["Extract%"] + 1e-13
    assert edit_diff.table_diff(t, noisy) is None


def test_a_text_number_equals_the_same_number():
    t = _table()
    t["Reference No."] = [1] * len(t)
    back = t.copy()
    back["Reference No."] = "1"                                 # as SQLite returns it
    assert edit_diff.table_diff(t, back) is None


def test_a_cell_edit_is_reported_by_staged_row_and_column():
    t = _table()
    e = t.copy()
    e.loc[2, "pH"] = 2.25
    e.loc[2, "Diluent"] = "kerosene"
    assert edit_diff.table_diff(t, e) == {
        "deleted": [], "added": [],
        "changed": [{"row": 3, "column": "pH", "old": 2, "new": 2.25},
                    {"row": 3, "column": "Diluent", "old": None, "new": "kerosene"}],
    }


def test_deleted_rows_keep_their_staged_numbers_and_values():
    t = _table()
    e = t.drop(index=[0, 3])             # what dropping QA-flagged rows does
    d = edit_diff.table_diff(t, e)
    assert [x["row"] for x in d["deleted"]] == [1, 4]
    assert d["deleted"][0]["values"] == {EL: "La", "pH": 1, "Extract%": 10, "Figure/Table": "Fig. 2"}
    assert d["added"] == [] and d["changed"] == []


def test_an_added_row_is_reported_with_its_values():
    t = _table()
    new = schema.coerce_schema(pd.DataFrame([{EL: "Nd", "pH": 3.0, "Extract%": 55.5}], index=[5]))
    d = edit_diff.table_diff(t, pd.concat([t, new]))
    assert d == {"deleted": [], "added": [{"values": {EL: "Nd", "pH": 3, "Extract%": 55.5}}], "changed": []}


def test_a_reorder_is_no_diff_but_a_cell_edit_under_it_is_found():
    t = _table()
    e = t.iloc[::-1].copy()
    assert edit_diff.table_diff(t, e) is None
    e.loc[4, "Extract%"] = 49.0
    assert edit_diff.table_diff(t, e)["changed"] == [{"row": 5, "column": "Extract%", "old": 50, "new": 49}]


def test_the_json_is_compact_deterministic_and_summarised():
    t = _table()
    e = t.drop(index=[1])
    e.loc[0, "pH"] = 0.9
    s = edit_diff.to_json(edit_diff.table_diff(t, e))
    assert s == edit_diff.to_json(edit_diff.table_diff(t.copy(), e.copy()))
    assert ", " not in s.replace("(REY:La, Ce, Nd)", "") and '": ' not in s
    assert json.loads(s)["changed"][0] == {"row": 1, "column": "pH", "old": 1, "new": 0.9}
    assert edit_diff.summary(s) == "1 row deleted, 1 cell changed"
    assert edit_diff.summary(None) == "" and edit_diff.summary(float("nan")) == ""


def test_realign_recovers_identity_lost_on_the_way_through_the_db():
    t = _table(8)
    e = t.drop(index=[2, 5])
    e.loc[6, "Extract%"] = 61.0
    lost = e.reset_index(drop=True)                     # rows read back from SQLite
    assert edit_diff.table_diff(t, edit_diff.realign(t, lost)) == edit_diff.table_diff(t, e)


def test_realign_reports_an_unrelated_row_as_added_not_as_an_edit():
    t = _table(4)
    other = schema.coerce_schema(pd.DataFrame([{EL: "Yb", "pH": 9.0, "Extract%": 1.0, "Diluent": "x"}]))
    lost = pd.concat([t.drop(index=[1]), other]).reset_index(drop=True)
    d = edit_diff.table_diff(t, edit_diff.realign(t, lost))
    assert [x["row"] for x in d["deleted"]] == [2]
    assert d["added"][0]["values"][EL] == "Yb" and d["changed"] == []


# --------------------------------------------------------------------------- #
# merge records the diff in its own transaction
# --------------------------------------------------------------------------- #
def _commit(conn, df, extracted, **kw):
    return merge.commit_extraction(
        conn, license=License("cc-by", origin="manifest"), content_sha256="h", pdf_path="h.pdf",
        df=df, extracted=extracted, text_endpoints=[], prompt_version="extraction_v16",
        prompt_sha256="p", model="m", qa_passed=True, qa_report_json="[]", raw_response="{}", **kw)


def _log(conn):
    return conn.execute("SELECT action, note, edited_diff_json FROM review_log").fetchall()


def test_merge_records_the_edits_on_the_approval(conn):
    t = _table()
    e = t.drop(index=[4])
    e.loc[0, "pH"] = 1.1
    _commit(conn, e, t)
    [(action, note, diff)] = _log(conn)
    assert action == "approve" and note == "edited in review"
    assert diff == edit_diff.to_json(edit_diff.table_diff(t, e))
    assert len(extractions_repo.current_best(conn)) == 4


def test_merge_of_an_untouched_table_records_no_diff(conn):
    t = _table()
    _commit(conn, t.copy(), t, note="checked")
    assert [tuple(r) for r in _log(conn)] == [("approve", "checked", None)]


def test_merge_without_the_extracted_table_records_no_diff(conn):
    t = _table()
    _commit(conn, t.drop(index=[0]), None)
    assert [tuple(r) for r in _log(conn)] == [("approve", None, None)]


def test_the_diff_and_the_rows_commit_or_roll_back_together(conn, monkeypatch):
    def fail(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(review_repo, "log_review", fail)
    t = _table()
    with pytest.raises(sqlite3.OperationalError):
        _commit(conn, t.drop(index=[0]), t)
    assert conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM prompt_runs").fetchone()[0] == 0
    assert _log(conn) == []


def test_past_edits_are_recovered_from_the_raw_response_and_listed(conn, tmp_path, capsys):
    """Approvals from before the diff was recorded: rebuilt read-only from the
    run's raw_response against its merged rows."""
    t = _table(6)
    rows = t.astype(object).where(t.notna(), None).values.tolist()
    raw = json.dumps({"columns": schema.COLUMNS, "rows": rows})
    e = t.drop(index=[1])
    e.loc[4, "Extract%"] = 44.0
    run = merge.commit_extraction(
        conn, license=License("cc-by", origin="manifest"), content_sha256="h", pdf_path="h.pdf",
        df=e, text_endpoints=[], prompt_version="extraction_v16", prompt_sha256="p", model="m",
        qa_passed=True, qa_report_json="[]", raw_response=raw)["prompt_run_id"]   # no `extracted`: as before
    assert edit_diff.recover(conn, run) == edit_diff.table_diff(t, e)

    db = tmp_path / "copy.db"
    conn.commit()
    conn.backup(sqlite3.connect(db))
    assert edit_diff.main([str(db)]) == 0
    out = capsys.readouterr().out
    assert f"run {run} (recovered): 1 row deleted, 1 cell changed" in out
