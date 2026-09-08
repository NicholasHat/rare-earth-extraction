"""Tests for extraction/staging.py — the on-disk review queue and Batch API
job sidecars that let both survive a Streamlit server restart. Pure file I/O
against a temporary STAGING_DIR; no Streamlit, no API."""
from datetime import datetime, timedelta, timezone

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
    assert len(list(_staging_dir.glob("*"))) == 2  # one .xlsx + one .meta.json


def test_discard_removes_both_files(_staging_dir):
    staging.stage(_paper(), True, _result())
    staging.discard(_paper().sha)
    assert list(_staging_dir.iterdir()) == []
    assert staging.load_all() == {}


def test_load_skips_orphaned_sidecar_and_corrupt_json(_staging_dir):
    staging.stage(_paper("b" * 64), True, _result())
    # Sidecar without its table (e.g. a crash between the two writes).
    (_staging_dir / ("c" * 64 + ".meta.json")).write_text("{}")
    # Table with an unparseable sidecar.
    (_staging_dir / ("d" * 64 + ".xlsx")).write_bytes(b"not really xlsx")
    (_staging_dir / ("d" * 64 + ".meta.json")).write_text("{not json")

    loaded = staging.load_all()
    assert list(loaded) == ["b" * 64]
    assert not (_staging_dir / ("c" * 64 + ".meta.json")).exists()


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


def test_batch_job_delete_and_corrupt_sidecar(_staging_dir):
    job = _job()
    job.save()
    (_staging_dir / "_batch_bad.batch.json").write_text("{not json")
    assert list(staging.load_batch_jobs()) == ["msgbatch_1"]
    job.delete()
    assert staging.load_batch_jobs() == {}


def test_record_failures_keeps_only_failed_papers_and_releases_lock(_staging_dir):
    job = _job()
    ok_sha, bad_sha = "a" * 64, "f" * 64
    job.items[bad_sha] = BatchItem(**{**vars(job.items[ok_sha]), "custom_id": bad_sha})
    job.papers[bad_sha] = _paper(bad_sha)
    job.begin_collection()
    assert job.collection_in_progress()

    job.record_failures({bad_sha: "could not parse"})

    got = staging.load_batch_jobs()[job.batch_id]
    assert list(got.items) == [bad_sha]
    assert list(got.papers) == [bad_sha]
    assert got.errors == {bad_sha: "could not parse"}
    assert got.file_ids == job.file_ids  # every upload is still cleaned up on discard
    assert not got.collection_in_progress()


def test_collection_lock_self_heals_after_staleness_window():
    job = _job()
    assert not job.collection_in_progress()
    job.collection_started_at = datetime.now(timezone.utc).isoformat()
    assert job.collection_in_progress()
    stale = datetime.now(timezone.utc) - staging.COLLECTION_LOCK_STALE_AFTER - timedelta(minutes=1)
    job.collection_started_at = stale.isoformat()
    assert not job.collection_in_progress()
