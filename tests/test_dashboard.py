"""The Streamlit pages, run headless with AppTest against a temporary data
directory: the review flow (licence gate, QA of the table as edited, merge),
the batch-jobs section and the read-only pages. No API calls — every paid or
networked entry point is replaced by one that fails the test."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config
from database import connection
from extraction import batch_jobs, runner, staging
from extraction.batch_jobs import CollectOutcome, PaperOutcome
from extraction.runner import BatchItem, ExtractionResult
from extraction.staging import BatchJob, PaperRef
from ingestion.licensing import License
from validation import schema
from validation.report import QAReport

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    for name, path in (
        ("DATA_DIR", tmp_path), ("INCOMING_DIR", tmp_path / "incoming"),
        ("STAGING_DIR", tmp_path / "staging"), ("EXPORTS_DIR", tmp_path / "exports"),
        ("DB_PATH", tmp_path / "master.db"), ("TRANSCRIPTS_DIR", tmp_path / "transcripts"),
        ("OPEN_ACCESS_MANIFEST", tmp_path / "open_access" / "manifest.csv"),
    ):
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(config, "REQUIRE_PASSWORD", False)

    def paid(*a, **k):
        raise AssertionError("a paid or networked call was made")

    for mod, names in ((batch_jobs, ("collect", "discard", "reconcile", "submit")),
                       (runner, ("extract_paper", "submit_batch", "collect_batch", "batch_status"))):
        for n in names:
            monkeypatch.setattr(mod, n, paid)
    return tmp_path


def _paper(c="a", license=None) -> PaperRef:
    return PaperRef(sha=c * 64, pdf_path=f"{c}.pdf", doi=None, filename=f"{c}.pdf",
                    meta={"title": "T", "n_pages": 5, "is_raster_figure": 0}, license=license)


def _result(rows, report=None) -> ExtractionResult:
    return ExtractionResult(
        df=schema.coerce_schema(pd.DataFrame(rows)), text_endpoints=[], qa_report=report or QAReport(),
        prompt_version="extraction_v15", prompt_sha256="s", model="claude-sonnet-5-5", raw_response="{}",
        coercion_failures=0, input_tokens=1000, output_tokens=2000, via_batch=True,
        transcript="transcripts/msgbatch_1/aaa.json.gz",
    )


_CLEAN = [{schema.ELEMENT_COLUMN: "La", "pH": 1.0 + i, "Extract%": 10.0 * i, "Figure/Table": "Fig. 2"}
          for i in range(10)]


def _run(path=APP) -> AppTest:
    at = AppTest.from_file(path, default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _merged_papers():
    conn = connection.get_conn()
    try:
        return [dict(r) for r in conn.execute("SELECT paper_id, license, license_origin FROM papers")]
    finally:
        conn.close()


def _review_list(at) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if d.key == "review_list")


def _browser_sends(monkeypatch, key, value):
    """The widget state the browser sends for an element AppTest can't drive
    (a row clicked in st.dataframe, a cell typed in st.data_editor): once, on
    the next run, as a JSON string under the element's id."""
    from streamlit.elements.lib.utils import user_key_from_element_id
    from streamlit.testing.v1.element_tree import ElementTree
    original = ElementTree.get_widget_states

    def states(tree):
        ws = original(tree)
        for node in tree:
            pid = getattr(getattr(node, "proto", None), "id", "")
            if pid and user_key_from_element_id(pid) == key:
                w = ws.widgets.add()
                w.id, w.string_value = pid, json.dumps(value)
                monkeypatch.setattr(ElementTree, "get_widget_states", original)
        return ws
    monkeypatch.setattr(ElementTree, "get_widget_states", states)


def _click_row(at, monkeypatch, i):
    _browser_sends(monkeypatch, "review_list", {"selection": {"rows": [i], "columns": []}})
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def _editor_key(at, sha):
    keys = [k for k in at.session_state.filtered_state if str(k).startswith(f"editor_{sha}")]
    assert len(keys) == 1, keys
    return keys[0]


def test_a_paper_with_no_licence_can_be_viewed_but_not_approved():
    staging.stage(_paper(), False, _result(_CLEAN))
    at = _run()

    assert _review_list(at).iloc[0]["Licence"] == "none"
    assert any("No licence recorded" in w.value for w in at.warning)
    assert _button(at, "Approve & merge").disabled
    assert len(at.dataframe) >= 1   # the table is still there to review
    assert _merged_papers() == []


def test_recording_a_licence_at_review_unblocks_the_merge_and_is_recorded():
    staging.stage(_paper(), False, _result(_CLEAN))
    at = _run()
    assert _button(at, "Save licence").disabled          # no evidence, no licence
    at.text_input(key=f"lic_ev_{'a' * 64}").input("CC BY 4.0 line on page 1").run()
    _button(at, "Save licence").click().run()
    assert not at.exception

    assert staging.load_all()["a" * 64].paper.license == License("cc-by", "CC BY 4.0 line on page 1", None, "manual")
    assert not _button(at, "Approve & merge").disabled
    assert any(c.value.startswith("Licence: cc-by (manual)") for c in at.caption)

    _button(at, "Approve & merge").click().run()
    assert not at.exception
    assert _merged_papers() == [{"paper_id": 1, "license": "cc-by", "license_origin": "manual"}]
    assert staging.load_all() == {}


def test_approval_goes_by_the_qa_of_the_table_as_it_stands_not_the_staged_report():
    # Staged with a clean report, but today's checks find an arithmetic red
    # (100 ppm La is 0.72 mM, not 5).
    rows = [dict(r, **{"RRE composition (ppm)": 100.0, "RRE composition (mM)": 5.0}) for r in _CLEAN]
    staged = staging.stage(_paper(license=License("cc-by", origin="manifest")), False, _result(rows))
    assert staged.result.qa_report.passed
    assert not runner.rerun_qa(staged.result, figure_is_curve=False).qa_report.passed

    at = _run()
    assert any("approval goes by this" in c.value for c in at.caption)
    assert any("[arithmetic]" in e.value for e in at.error)
    assert at.checkbox(key=f"override_{'a' * 64}")   # offered because the current table has reds

    _button(at, "Approve & merge").click().run()
    assert any("red QA flags" in e.value for e in at.error)
    assert _merged_papers() == []

    at.checkbox(key=f"override_{'a' * 64}").check().run()
    _button(at, "Approve & merge").click().run()
    assert not at.exception
    assert len(_merged_papers()) == 1


def test_every_staged_paper_is_listed_and_any_one_can_be_reviewed(monkeypatch):
    for c in "abc":
        staging.stage(_paper(c, License("cc-by", origin="manifest") if c != "b" else None), False,
                      _result(_CLEAN))
    at = _run()

    listed = _review_list(at)
    assert list(listed["Paper"]) == ["a.pdf", "b.pdf", "c.pdf"]
    assert list(listed["Licence"]) == ["cc-by", "none", "cc-by"]
    assert {"Rows", "Red", "Amber", "Cost (USD)"} <= set(listed.columns)
    assert _button(at, "Previous").disabled
    assert at.session_state["review_sha"] == "a" * 64

    _click_row(at, monkeypatch, 2)                 # straight to the last one
    assert at.session_state["review_sha"] == "c" * 64
    assert _button(at, "Next").disabled
    _button(at, "Previous").click().run()
    assert at.session_state["review_sha"] == "b" * 64
    assert _button(at, "Approve & merge").disabled   # b's own licence state, not a's
    _editor_key(at, "b" * 64)                         # only the selected paper's editor is drawn


def test_edits_to_one_paper_are_kept_while_another_is_reviewed(monkeypatch):
    for c in "ab":
        staging.stage(_paper(c, License("cc-by", origin="manifest")), False, _result(_CLEAN))
    a = "a" * 64
    at = _run()
    _browser_sends(monkeypatch, _editor_key(at, a),
                   {"edited_rows": {"0": {"pH": 1.5}}, "added_rows": [], "deleted_rows": []})
    at.run()
    at.text_input(key=f"note_{a}").input("checked Fig. 2").run()
    at.text_input(key=f"cit_{a}").input("Someone et al.").run()
    assert _review_list(at).iloc[0]["Edited"] == "yes"

    _click_row(at, monkeypatch, 1)
    assert at.session_state["review_sha"] == "b" * 64
    assert _review_list(at).iloc[0]["Edited"] == "yes"
    _click_row(at, monkeypatch, 0)
    assert not [w.value for w in at.warning if "Session State" in w.value]
    assert next(d.value for d in at.dataframe if d.key == _editor_key(at, a))["pH"].iloc[0] == 1.5
    assert at.text_input(key=f"note_{a}").value == "checked Fig. 2"
    assert at.text_input(key=f"cit_{a}").value == "Someone et al."

    _button(at, "Approve & merge").click().run()
    assert not at.exception
    conn = connection.get_conn()
    try:
        ph = [r[0] for r in conn.execute('SELECT "pH" FROM extractions ORDER BY rowid')]
        note = conn.execute("SELECT note FROM review_log").fetchone()[0]
    finally:
        conn.close()
    assert ph[0] == 1.5 and len(ph) == len(_CLEAN)
    assert note == "checked Fig. 2"


def test_what_the_reviewer_changed_is_in_the_audit_log_and_on_the_database_page(monkeypatch):
    staging.stage(_paper("a", License("cc-by", origin="manifest")), False, _result(_CLEAN))
    a = "a" * 64
    at = _run()
    _browser_sends(monkeypatch, _editor_key(at, a), {
        "edited_rows": {"0": {"pH": 1.5}},
        "deleted_rows": [3],
        "added_rows": [{schema.ELEMENT_COLUMN: "Nd", "pH": 2.0, "Extract%": 40.0, "Figure/Table": "Table 1"}],
    })
    at.run()
    _button(at, "Approve & merge").click().run()
    assert not at.exception
    conn = connection.get_conn()
    try:
        note, diff = conn.execute("SELECT note, edited_diff_json FROM review_log").fetchone()
    finally:
        conn.close()
    diff = json.loads(diff)
    assert note == "edited in review"
    assert diff["changed"] == [{"row": 1, "column": "pH", "old": 1, "new": 1.5}]
    assert [d["row"] for d in diff["deleted"]] == [4]
    assert diff["added"] == [{"values": {schema.ELEMENT_COLUMN: "Nd", "pH": 2, "Extract%": 40,
                                         "Figure/Table": "Table 1"}}]

    log = next(d.value for d in _run(str(ROOT / "pages" / "1_Database.py")).dataframe
               if "edits" in d.value.columns)
    assert log.iloc[0]["edits"] == "1 row deleted, 1 row added, 1 cell changed"


def test_after_approve_or_reject_the_next_paper_in_the_list_is_selected(monkeypatch):
    for c in "abcd":
        staging.stage(_paper(c, License("cc-by", origin="manifest")), False, _result(_CLEAN))
    at = _run()
    _click_row(at, monkeypatch, 1)
    _button(at, "Reject").click().run()
    assert set(staging.load_all()) == {c * 64 for c in "acd"}
    assert at.session_state["review_sha"] == "c" * 64

    _button(at, "Next").click().run()                 # d, the last one
    _button(at, "Approve & merge").click().run()
    assert not at.exception
    assert len(_merged_papers()) == 1
    assert at.session_state["review_sha"] == "c" * 64   # nothing after it: the one before
    assert list(_review_list(at)["Paper"]) == ["a.pdf", "c.pdf"]


def test_batch_jobs_show_expiry_and_keep_the_collection_outcome_on_screen(monkeypatch):
    p = _paper("b")
    job = BatchJob(batch_id="msgbatch_1", file_ids={p.sha: "file_b"},
                   items={p.sha: BatchItem(p.sha, True, "", [], "v", "s", "m")}, papers={p.sha: p},
                   created_at=(datetime.now(timezone.utc) - timedelta(days=25)).isoformat())
    job.save()
    assert p.sha in staging.in_flight_shas()

    at = _run()
    assert any("deletes this batch's results" in w.value for w in at.warning)

    def collect(batch_id):
        job.delete()
        return CollectOutcome(batch_id, "collected", "ended", papers=[
            PaperOutcome(p.sha, p.filename, True, rows=12, reds=0, ambers=2, cost_usd=1.5,
                         transcript="transcripts/msgbatch_1/b.json.gz")])
    monkeypatch.setattr(batch_jobs, "collect", collect)
    _button(at, "Check status").click().run()
    assert not at.exception
    assert any("1/1 paper(s) staged" in s.value for s in at.success)
    assert at.dataframe[0].value.iloc[0]["Rows"] == 12


def test_with_a_write_password_set_nothing_is_rejected_collected_or_discarded_while_locked(monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_PASSWORD", True)
    monkeypatch.setattr(config, "WRITE_PASSWORD", "secret")
    staging.stage(_paper("a", License("cc-by", origin="manifest")), False, _result(_CLEAN))
    p = _paper("b")
    BatchJob(batch_id="msgbatch_1", file_ids={p.sha: "file_b"},
             items={p.sha: BatchItem(p.sha, True, "", [], "v", "s", "m")}, papers={p.sha: p},
             errors={p.sha: "boom"}).save()
    at = _run()

    # batch_jobs.collect / discard fail the test if called (the data_dir fixture).
    for label in ("Reject", "Check status", "Discard failed paper(s)", "Re-run QA checks"):
        _button(at.run(), label).click().run()   # the prompt stops the run: start each one afresh
        assert not at.exception, label
        assert any("requires the shared write password" in i.value for i in at.info), label
    assert set(staging.load_all()) == {"a" * 64}
    assert set(staging.load_batch_jobs()) == {"msgbatch_1"}


def test_failed_runs_list_their_transcript_and_credit_state():
    from extraction.anthropic_client import CreditExhausted
    staging.record_failed_run(_paper(), CreditExhausted("credit balance is too low"))
    at = _run()
    df = at.dataframe[-1].value
    assert {"Transcript", "Credit exhausted"} <= set(df.columns)
    assert bool(df.iloc[0]["Credit exhausted"])


def test_database_page_shows_the_added_columns_and_a_missing_licence():
    from database import papers_repo
    connection.init_db()
    conn = connection.get_conn()
    with conn:
        papers_repo.insert(conn, content_sha256="c" * 64, doi=None, reference_no=None, title="Swain",
                           pdf_path="c.pdf", original_filename="swain.pdf", figure_type=None,
                           is_raster_figure=0)
    conn.close()

    at = _run(str(ROOT / "pages" / "1_Database.py"))
    papers = next(d.value for d in at.dataframe if "license_origin" in d.value.columns)
    assert papers.iloc[0]["license"] == "none recorded"
    assert any("can't be re-approved" in c.value for c in at.caption)


@pytest.mark.parametrize("page", ["2_Calculator.py", "3_Lab_Assistant.py"])
def test_read_only_pages_render(page):
    _run(str(ROOT / "pages" / page))
