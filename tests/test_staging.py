"""Tests for extraction/staging.py — the on-disk review queue and Batch API
job sidecars that let both survive a Streamlit server restart. Pure file I/O
against a temporary STAGING_DIR; no Streamlit, no API."""
import json

import pandas as pd
import pytest

import config
from extraction import staging
from extraction.runner import BatchItem, ExtractionResult
from extraction.staging import BatchJob, PaperRef
from validation import schema
from validation.report import QAReport, Severity


@pytest.fixture(autouse=True)
def _staging_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    return tmp_path


def _paper(sha="a" * 64) -> PaperRef:
    return PaperRef(
        sha=sha, pdf_path=f"data/incoming/{sha}.pdf", doi="10.1016/x",
        filename="swain.pdf", meta={"title": "T", "n_pages": 9, "is_raster_figure": 0},
    )


def _result() -> ExtractionResult:
    df = schema.coerce_schema(pd.DataFrame([
        {"Rare Earth Elements (REY:La, Ce, Nd)": "La", "pH": 2.0, "Extract%": 40.5},
        {"Rare Earth Elements (REY:La, Ce, Nd)": "Ce", "pH": 2.5, "Extract%": 61.0},
    ]))
    report = QAReport()
    report.add("monotonicity", Severity.AMBER, "check La")
    return ExtractionResult(
        df=df, text_endpoints=[{"element": "La", "x_value": 2.0}], qa_report=report,
        prompt_version="extraction_v9", prompt_sha256="abc", model="claude-sonnet-5",
        raw_response="{}", coercion_failures=0, curve_analysis="## block",
        deterministic_counts=[19, 19], input_tokens=10, output_tokens=20,
        cache_creation_input_tokens=5, cache_read_input_tokens=100,
    )


# --------------------------------------------------------------------------- #
# Review queue
# --------------------------------------------------------------------------- #

def test_stage_then_load_round_trips_every_field(_staging_dir):
    staged = staging.stage(_paper(), False, _result())
    loaded = staging.load_all()

    assert list(loaded) == [staged.paper.sha]
    got = loaded[staged.paper.sha]
    assert got.paper == staged.paper
    assert got.figure_is_curve is False
    # The .xlsx round trip turns empty text cells into NaN; coerce_schema (which
    # the approve path runs anyway) normalises both sides to the schema's None.
    pd.testing.assert_frame_equal(
        schema.coerce_schema(got.result.df), schema.coerce_schema(staged.result.df)
    )
    for name in (
        "text_endpoints", "prompt_version", "prompt_sha256", "model", "raw_response",
        "coercion_failures", "curve_analysis", "deterministic_counts", "input_tokens",
        "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
    ):
        assert getattr(got.result, name) == getattr(staged.result, name), name
    assert got.result.qa_report.to_json() == staged.result.qa_report.to_json()


def test_restaging_the_same_paper_replaces_it(_staging_dir):
    staging.stage(_paper(), True, _result())
    second = _result()
    second.prompt_version = "extraction_v10"
    staging.stage(_paper(), True, second)

    loaded = staging.load_all()
    assert len(loaded) == 1
    assert loaded[_paper().sha].result.prompt_version == "extraction_v10"
    assert len(list(_staging_dir.glob("*.*"))) == 2  # one .xlsx + one .meta.json


def test_discard_removes_both_files(_staging_dir):
    staging.stage(_paper(), True, _result())
    staging.discard(_paper().sha)
    assert [p for p in _staging_dir.iterdir() if p.is_file()] == []
    assert staging.load_all() == {}


def test_load_skips_orphaned_sidecar_and_corrupt_json_but_deletes_nothing(_staging_dir):
    staging.stage(_paper("b" * 64), True, _result())
    # An old-style sidecar (no table inside) whose .xlsx is gone.
    (_staging_dir / ("c" * 64 + ".meta.json")).write_text("{}")
    # Table with an unparseable sidecar.
    (_staging_dir / ("d" * 64 + ".xlsx")).write_bytes(b"not really xlsx")
    (_staging_dir / ("d" * 64 + ".meta.json")).write_text("{not json")

    loaded = staging.load_all()
    assert list(loaded) == ["b" * 64]
    assert (_staging_dir / ("c" * 64 + ".meta.json")).exists()   # left for inspection


def test_the_sidecar_is_the_record_so_a_lost_or_stale_xlsx_loses_nothing(_staging_dir):
    staged = staging.stage(_paper(), True, _result())
    staging.table_path(_paper().sha).unlink()
    got = staging.load_all()[_paper().sha]
    pd.testing.assert_frame_equal(got.result.df, schema.coerce_schema(staged.result.df))


def test_a_26_column_staged_result_loads_into_the_35_column_schema(_staging_dir):
    """The 2026-10 trial's sidecars: table in an .xlsx with the lab's 26
    columns, provenance as a parallel row_figures list in the sidecar."""
    sha = "e" * 64
    df = _result().df[schema.LAB_COLUMNS]
    df.to_excel(_staging_dir / f"{sha}.xlsx", index=False, engine="openpyxl")
    (_staging_dir / f"{sha}.meta.json").write_text(json.dumps({
        "sha": sha, "pdf_path": "p.pdf", "doi": None, "filename": "old.pdf", "meta": {},
        "prompt_version": "extraction_v14", "prompt_sha256": "x", "model": "m",
        "raw_response": "{}", "coercion_failures": 0, "text_endpoints": [], "qa_report_json": "[]",
        "row_figures": ["Fig. 2", "Fig. 3"], "figures": [],
    }))
    got = staging.load_all()[sha].result
    assert list(got.df.columns) == schema.COLUMNS
    assert got.row_figures == ["Fig. 2", "Fig. 3"]
    assert got.df["Diluent"].isna().all()


def test_a_re_extraction_keeps_the_result_it_replaces(_staging_dir):
    staging.stage(_paper(), True, _result())
    staging.stage(_paper(), True, _result())            # the same run re-staged (Re-run QA): nothing kept
    assert not (_staging_dir / "_superseded").exists()
    second = _result()
    second.raw_response = '{"rows": []}'
    staging.stage(_paper(), True, second)
    kept = list((_staging_dir / "_superseded").glob("*.meta.json"))
    assert len(kept) == 1 and json.loads(kept[0].read_text())["raw_response"] == "{}"
    assert staging.load_all()[_paper().sha].result.raw_response == '{"rows": []}'


def test_licence_round_trips_and_can_be_set_by_hand(_staging_dir):
    from ingestion.licensing import License
    staging.stage(_paper(), True, _result())
    assert staging.load_all()[_paper().sha].paper.license is None
    staging.set_license(_paper().sha, License.of("CC BY 4.0", "article page footer", "https://x"))
    got = staging.load_all()[_paper().sha].paper.license
    assert got == License("cc-by", "article page footer", "https://x", "manual")


def test_load_tolerates_a_sidecar_written_before_new_fields_existed(_staging_dir):
    """Older sidecars lack figure_is_curve and some result fields; they must
    still load with defaults rather than being silently dropped."""
    import json
    sha = "e" * 64
    _result().df.to_excel(_staging_dir / f"{sha}.xlsx", index=False, engine="openpyxl")
    (_staging_dir / f"{sha}.meta.json").write_text(json.dumps({
        "sha": sha, "pdf_path": "p.pdf", "doi": None, "filename": "old.pdf", "meta": {},
        "prompt_version": "extraction_v7", "prompt_sha256": "x", "model": "m",
        "raw_response": "{}", "coercion_failures": 0, "text_endpoints": [],
        "qa_report_json": "[]",
    }))
    got = staging.load_all()[sha]
    assert got.figure_is_curve is True
    assert got.result.deterministic_counts == []
    assert got.result.input_tokens == 0
    assert got.result.qa_report.passed


# --------------------------------------------------------------------------- #
# Batch API jobs
# --------------------------------------------------------------------------- #

def _job(batch_id="msgbatch_1") -> BatchJob:
    item = BatchItem(
        custom_id="a" * 64, figure_is_curve=True, analysis_block="", deterministic_counts=[],
        prompt_version="extraction_v9", prompt_sha256="abc", model="claude-sonnet-5",
    )
    return BatchJob(
        batch_id=batch_id, file_ids={"a" * 64: "file_1"},
        items={"a" * 64: item}, papers={"a" * 64: _paper()},
    )


def test_batch_job_round_trips(_staging_dir):
    job = _job()
    job.save()
    loaded = staging.load_batch_jobs()
    assert list(loaded) == ["msgbatch_1"]
    got = loaded["msgbatch_1"]
    assert got == job


def test_batch_job_round_trips_a_re_extraction_and_loads_older_sidecars(_staging_dir):
    """qa_feedback (a batched re-extraction) survives the sidecar; a sidecar
    written before the field existed still loads, as a first extraction."""
    job = _job()
    job.items["a" * 64].qa_feedback = "## QA FEEDBACK"
    job.toolkit_file_id = "file_kit"
    job.save()
    loaded = staging.load_batch_jobs()["msgbatch_1"]
    assert loaded.items["a" * 64].qa_feedback == "## QA FEEDBACK"
    assert loaded.toolkit_file_id == "file_kit"

    payload = json.loads(job.path.read_text())
    del payload["items"]["a" * 64]["qa_feedback"]
    del payload["toolkit_file_id"]
    job.path.write_text(json.dumps(payload))
    loaded = staging.load_batch_jobs()["msgbatch_1"]
    assert loaded.items["a" * 64].qa_feedback is None
    assert loaded.toolkit_file_id is None      # continuation rebuilds the request as it was sent


def test_batch_job_delete_and_corrupt_sidecar(_staging_dir):
    job = _job()
    job.save()
    (_staging_dir / "_batch_bad.batch.json").write_text("{not json")
    assert list(staging.load_batch_jobs()) == ["msgbatch_1"]
    job.delete()
    assert staging.load_batch_jobs() == {}


def test_record_failures_keeps_only_failed_papers(_staging_dir):
    job = _job()
    ok_sha, bad_sha = "a" * 64, "f" * 64
    job.items[bad_sha] = BatchItem(**{**vars(job.items[ok_sha]), "custom_id": bad_sha})
    job.papers[bad_sha] = _paper(bad_sha)

    job.record_failures({bad_sha: "could not parse"})

    got = staging.load_batch_jobs()[job.batch_id]
    assert list(got.items) == [bad_sha]
    assert list(got.papers) == [bad_sha]
    assert got.errors == {bad_sha: "could not parse"}
    assert got.file_ids == job.file_ids  # every upload is still cleaned up on discard


def test_collection_lock_excludes_a_second_collector_and_dies_with_its_holder(_staging_dir):
    """An OS lock: a second collector (another click, the CLI) is refused
    while the first holds it, and it frees itself when the holder exits —
    here in a child process killed mid-collection — with nothing to clean up."""
    import subprocess
    import sys
    from pathlib import Path

    job = _job()
    job.save()
    assert not job.collection_in_progress()
    with staging.collection_lock(job.batch_id):
        assert job.collection_in_progress()
        with pytest.raises(staging.Busy):
            with staging.collection_lock(job.batch_id):
                pass
    assert not job.collection_in_progress()

    root = Path(__file__).resolve().parents[1]
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time, config; from pathlib import Path; config.STAGING_DIR = Path(sys.argv[1]); "
         "from extraction import staging\n"
         "with staging.collection_lock('msgbatch_1'):\n    print('ready', flush=True); time.sleep(60)",
         str(_staging_dir)],
        cwd=root, stdout=subprocess.PIPE, text=True,
    )
    assert child.stdout.readline().strip() == "ready"
    assert job.collection_in_progress()
    child.kill()
    child.wait(10)
    assert not job.collection_in_progress()


def test_a_pending_submission_holds_its_papers_and_yields_to_its_accepted_job(_staging_dir):
    job = _job(batch_id=None)
    job.save()
    assert job.path.name.startswith("_pending_")
    assert staging.load_pending_submissions()[0].job_id == job.job_id
    assert "a" * 64 in staging.in_flight_shas()
    # Accepted: the journal entry becomes the job.
    job.batch_id = "msgbatch_9"
    job.save()
    assert staging.load_pending_submissions() == []
    assert list(staging.load_batch_jobs()) == ["msgbatch_9"]
    # A crash between writing the job and removing the journal leaves both; the job wins.
    stale = BatchJob.from_dict({**json.loads(job.path.read_text()), "batch_id": None})
    staging._write_json(stale.path, json.loads(job.path.read_text()) | {"batch_id": None})
    assert staging.load_pending_submissions() == [] and list(staging.load_batch_jobs()) == ["msgbatch_9"]


def test_a_claimed_paper_is_in_flight_only_while_claimed(_staging_dir):
    with staging.claim("c" * 64):
        assert "c" * 64 in staging.in_flight_shas()
        with pytest.raises(staging.Busy):
            with staging.claim("c" * 64):
                pass
    assert "c" * 64 not in staging.in_flight_shas()


# --------------------------------------------------------------------------- #
# Failed-runs log
# --------------------------------------------------------------------------- #

def test_failed_runs_log_round_trips_newest_first_and_skips_corrupt_lines(_staging_dir):
    class _Failed(RuntimeError):
        usage = {"input_tokens": 1, "output_tokens": 2,
                 "cache_creation_input_tokens": 3, "cache_read_input_tokens": 4}
        turns_completed = 2

    assert staging.load_failed_runs() == []
    staging.record_failed_run(_paper(), ValueError("first, no usage"))
    staging.record_failed_run(_paper(), _Failed("second, with usage"))
    with staging._failed_runs_path().open("a") as f:
        f.write("{not json\n")

    got = staging.load_failed_runs()
    assert [g["error"] for g in got] == ["second, with usage", "first, no usage"]
    assert got[0]["usage"]["output_tokens"] == 2 and got[0]["turns_completed"] == 2
    assert got[1]["usage"] is None
    assert got[0]["sha"] == _paper().sha and got[0]["filename"] == _paper().filename
    assert staging.load_failed_runs(limit=1) == got[:1]


def test_a_failed_run_records_its_transcript_and_whether_credit_ran_out(_staging_dir):
    from extraction.anthropic_client import CreditExhausted, ExtractionFailed
    err = ExtractionFailed(CreditExhausted("no credit"), {"input_tokens": 0, "output_tokens": 5,
                                                         "cache_creation_input_tokens": 0,
                                                         "cache_read_input_tokens": 0})
    err.transcript = "transcripts/msgbatch_1/aaa.json.gz"
    staging.record_failed_run(_paper(), err, batch_id="msgbatch_1")
    (got,) = staging.load_failed_runs()
    assert got["transcript"] == "transcripts/msgbatch_1/aaa.json.gz" and got["credit_exhausted"] is True
