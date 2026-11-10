"""extraction/batch_jobs.py: submit journals before it pays, reconcile finds
or cleans up what an interrupted submission left, collect is safe to repeat.
No API calls — a fake client and patched runner calls."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pandas as pd
import pytest

import config
from extraction import anthropic_client, batch_jobs, runner, staging
from extraction.anthropic_client import BatchInfo
from extraction.runner import BatchItem, ExtractionResult
from extraction.staging import BatchJob, PaperRef
from validation import schema
from validation.report import QAReport, Severity


@pytest.fixture(autouse=True)
def _dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(config, "ensure_dirs", lambda: (tmp_path / "staging").mkdir(exist_ok=True))
    (tmp_path / "staging").mkdir()
    # Never the real prompt's prepass; never a real PDF.
    monkeypatch.setattr(runner, "_prepass", lambda pdf: ("", []))
    return tmp_path


def _paper(c):
    return PaperRef(sha=c * 64, pdf_path=f"{c}.pdf", doi=None, filename=f"{c}.pdf", meta={})


def _status_error(code, message="nope", kind="invalid_request_error"):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches")
    return anthropic.APIStatusError(message, response=httpx.Response(code, request=request),
                                    body={"type": "error", "error": {"type": kind, "message": message}})


def _fake_client(create):
    """Uploads succeed with sequential ids; batches.create does `create`."""
    client = MagicMock()
    client.with_options.return_value = client
    counter = iter(range(1, 100))
    client.beta.files.upload.side_effect = lambda file: SimpleNamespace(id=f"file_{next(counter)}")
    client.beta.messages.batches.create.side_effect = create
    return client


def _snapshot_create(seen):
    def create(**kwargs):
        # What is on disk at the moment the paid call goes out.
        (job,) = staging.load_pending_submissions()
        seen.update(create_attempted=job.create_attempted_at, file_ids=dict(job.file_ids),
                    toolkit=job.toolkit_file_id, items=set(job.items))
        return SimpleNamespace(id="msgbatch_new", created_at=datetime.now(timezone.utc))
    return create


def test_submit_journals_every_upload_and_the_create_attempt_before_paying():
    seen = {}
    with patch("anthropic.Anthropic", return_value=_fake_client(_snapshot_create(seen))):
        outcome = batch_jobs.submit([(_paper("a"), b"%PDF-a"), (_paper("b"), b"%PDF-b")])
    assert seen["create_attempted"] and seen["toolkit"] == "file_1"
    assert seen["file_ids"] == {"a" * 64: "file_2", "b" * 64: "file_3"} and seen["items"] == {"a" * 64, "b" * 64}
    assert outcome.job.batch_id == "msgbatch_new"
    assert staging.load_pending_submissions() == []
    assert set(staging.load_batch_jobs()["msgbatch_new"].items) == {"a" * 64, "b" * 64}


def test_submit_never_sends_a_paper_that_is_staged_or_already_in_a_job():
    BatchJob(batch_id="msgbatch_old", file_ids={}, items={}, papers={"a" * 64: _paper("a")}).save()
    df = schema.coerce_schema(pd.DataFrame([{schema.ELEMENT_COLUMN: "La", "pH": 2.0, "Extract%": 40.0}]))
    staging.stage(_paper("b"), True, ExtractionResult(df=df, text_endpoints=[], qa_report=QAReport(),
                  prompt_version="v", prompt_sha256="s", model="m", raw_response="{}", coercion_failures=0))
    client = _fake_client(lambda **kw: SimpleNamespace(id="msgbatch_x", created_at=None))
    with patch("anthropic.Anthropic", return_value=client):
        outcome = batch_jobs.submit([(_paper("a"), b"a"), (_paper("b"), b"b")])
    assert outcome.job is None and set(outcome.skipped) == {"a" * 64, "b" * 64}
    client.beta.files.upload.assert_not_called()
    # A QA-feedback re-extraction may replace a staged paper (but not one in a job).
    with patch("anthropic.Anthropic", return_value=client):
        outcome = batch_jobs.submit([(_paper("b"), b"b")], qa_feedback="## QA", replacing_staged=True)
    assert outcome.job.batch_id == "msgbatch_x"


def test_a_submission_refused_for_credit_leaves_nothing_behind():
    def create(**kwargs):
        raise _status_error(402, "Your credit balance is too low to access the Anthropic API.", "billing_error")
    client = _fake_client(create)
    with patch("anthropic.Anthropic", return_value=client), pytest.raises(anthropic_client.CreditExhausted):
        batch_jobs.submit([(_paper("a"), b"a")])
    assert staging.load_pending_submissions() == [] and staging.load_batch_jobs() == {}
    assert sorted(c.args[0] for c in client.beta.files.delete.call_args_list) == ["file_1", "file_2"]
    assert staging.in_flight_shas() == set()


def test_an_unknown_create_outcome_keeps_the_journal_and_uploads_for_reconcile():
    def create(**kwargs):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    client = _fake_client(create)
    with patch("anthropic.Anthropic", return_value=client), pytest.raises(anthropic_client.BatchSubmissionUnknown):
        batch_jobs.submit([(_paper("a"), b"a")])
    client.beta.files.delete.assert_not_called()
    (pending,) = staging.load_pending_submissions()
    assert pending.create_attempted_at and pending.file_ids == {"a" * 64: "file_2"}
    assert "a" * 64 in staging.in_flight_shas()     # not resubmittable until resolved


def _pending(created_ago=None, n=1):
    job = BatchJob(batch_id=None, file_ids={_paper("a").sha: "file_2"}, toolkit_file_id="file_1",
                   items={(_paper(c).sha): BatchItem(_paper(c).sha, True, "", [], "v", "s", "m")
                          for c in "abc"[:n]},
                   papers={_paper(c).sha: _paper(c) for c in "abc"[:n]})
    if created_ago is not None:
        job.create_attempted_at = (datetime.now(timezone.utc) - created_ago).isoformat()
    job.save()
    return job


def _info(batch_id, created, requests=1, status="in_progress"):
    return BatchInfo(batch_id, status, created, None, requests, 0, 0, 0, 0)


def test_reconcile_adopts_the_batch_an_interrupted_submission_created():
    job = _pending(timedelta(minutes=30))
    attempted = datetime.fromisoformat(job.create_attempted_at)
    listed = [_info("msgbatch_other", attempted - timedelta(hours=2)),
              _info("msgbatch_found", attempted + timedelta(seconds=3))]
    with patch.object(anthropic_client, "list_batches", return_value=listed), \
         patch.object(anthropic_client, "batch_info"):
        report = batch_jobs.reconcile()
    assert [j.batch_id for j in report.adopted] == ["msgbatch_found"]
    assert list(staging.load_batch_jobs()) == ["msgbatch_found"] and staging.load_pending_submissions() == []
    assert [b.batch_id for b in report.untracked] == ["msgbatch_other"]


def test_reconcile_cleans_up_a_submission_that_never_created_a_batch():
    _pending(None)                                   # died before the create call
    _pending(timedelta(hours=1))                     # create attempted an hour ago, no batch exists
    young = _pending(timedelta(seconds=10))          # create attempted just now: wait
    with patch.object(anthropic_client, "list_batches", return_value=[]), \
         patch.object(anthropic_client, "cleanup_batch_files") as cleanup:
        report = batch_jobs.reconcile()
    assert len(report.abandoned) == 2 and cleanup.call_count == 2
    assert [j.job_id for j, _ in report.unresolved] == [young.job_id]
    assert [j.job_id for j in staging.load_pending_submissions()] == [young.job_id]


def test_reconcile_leaves_a_submission_whose_process_is_still_running():
    job = _pending(None)
    with staging.intent_lock(job.job_id), patch.object(anthropic_client, "list_batches", return_value=[]):
        report = batch_jobs.reconcile()
    assert report.abandoned == [] and report.unresolved[0][0].job_id == job.job_id


def test_reconcile_reports_how_long_results_remain():
    job = BatchJob(batch_id="msgbatch_1", file_ids={}, items={}, papers={},
                   created_at=(datetime.now(timezone.utc) - timedelta(days=25)).isoformat())
    job.save()
    with patch.object(anthropic_client, "list_batches", return_value=[]), \
         patch.object(anthropic_client, "batch_info", side_effect=RuntimeError("offline")):
        (st,) = batch_jobs.reconcile().jobs
    assert timedelta(days=3) < st.expires_in < timedelta(days=5) and st.error == "offline"


def _staged_result(n_rows=3):
    df = schema.coerce_schema(pd.DataFrame([{schema.ELEMENT_COLUMN: "La", "pH": 1.0 + i, "Extract%": 10.0 * i}
                                            for i in range(n_rows)]))
    report = QAReport()
    report.add("monotonicity", Severity.AMBER, "x")
    return ExtractionResult(df=df, text_endpoints=[], qa_report=report, prompt_version="extraction_v15",
                            prompt_sha256="s", model="claude-sonnet-5-5", raw_response="{}", coercion_failures=0,
                            output_tokens=1000, via_batch=True)


def _recording(results, seen):
    """A runner.collect_batch stand-in that records which papers it was asked for."""
    def fake(batch_id, items, file_ids, toolkit_file_id, *, store=None):
        seen.append((set(items), store))
        return results
    return fake


def _job_with(*chars):
    job = BatchJob(batch_id="msgbatch_1", file_ids={_paper(c).sha: f"file_{c}" for c in chars},
                   items={_paper(c).sha: BatchItem(_paper(c).sha, True, "", [], "v", "s", "m") for c in chars},
                   papers={_paper(c).sha: _paper(c) for c in chars}, toolkit_file_id="file_kit")
    job.save()
    return job


def test_collect_stages_successes_keeps_failures_and_cleans_up_only_when_done():
    _job_with("a", "b")
    failure = anthropic_client.ExtractionFailed(ValueError("no JSON"), {"input_tokens": 0, "output_tokens": 9,
                                                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})
    seen = []
    with patch.object(runner, "batch_status", return_value="ended"), \
         patch.object(runner, "collect_batch", _recording({_paper("a").sha: _staged_result(),
                                                           _paper("b").sha: failure}, seen)), \
         patch.object(anthropic_client, "cleanup_batch_files") as cleanup:
        outcome = batch_jobs.collect("msgbatch_1")
    assert seen[0][1].run_id == "msgbatch_1"
    (ok,) = [p for p in outcome.papers if p.staged]
    assert ok.rows == 3 and ok.ambers == 1 and ok.cost_usd is not None
    assert [p.sha for p in outcome.failures] == [_paper("b").sha]
    assert set(staging.load_all()) == {_paper("a").sha}
    job = staging.load_batch_jobs()["msgbatch_1"]
    assert set(job.items) == {_paper("b").sha} and "no JSON" in job.errors[_paper("b").sha]
    cleanup.assert_not_called()
    assert staging.load_failed_runs()[0]["batch_id"] == "msgbatch_1"

    # Collect again: only the failed paper is retried; when it lands, the job and its uploads go.
    seen = []
    with patch.object(runner, "batch_status", return_value="ended"), \
         patch.object(runner, "collect_batch", _recording({_paper("b").sha: _staged_result()}, seen)), \
         patch.object(anthropic_client, "cleanup_batch_files") as cleanup:
        batch_jobs.collect("msgbatch_1")
    assert seen[0][0] == {_paper("b").sha}
    assert staging.load_batch_jobs() == {} and cleanup.call_count == 1
    assert set(staging.load_all()) == {_paper("a").sha, _paper("b").sha}


def test_collect_waits_for_the_batch_and_yields_to_another_collector():
    _job_with("a")
    with patch.object(runner, "batch_status", return_value="in_progress"), \
         patch.object(runner, "collect_batch") as collect:
        assert batch_jobs.collect("msgbatch_1").status == "not_ended"
        with staging.collection_lock("msgbatch_1"):
            assert batch_jobs.collect("msgbatch_1").status == "busy"
    collect.assert_not_called()
    assert batch_jobs.collect("msgbatch_nope").status == "gone"


def test_a_paper_staged_before_a_crash_is_not_staged_again():
    """mark_staged drops each paper from the job as it lands, so a collection
    interrupted half-way never re-stages (or re-offers for approval) a paper."""
    job = _job_with("a", "b")
    job.mark_staged(_paper("a").sha)
    seen = []
    with patch.object(runner, "batch_status", return_value="ended"), \
         patch.object(runner, "collect_batch", _recording({_paper("b").sha: _staged_result()}, seen)), \
         patch.object(anthropic_client, "cleanup_batch_files"):
        batch_jobs.collect("msgbatch_1")
    assert seen[0][0] == {_paper("b").sha}


def test_discard_frees_the_papers():
    _job_with("a")
    with patch.object(anthropic_client, "cleanup_batch_files") as cleanup:
        batch_jobs.discard("msgbatch_1")
    assert cleanup.call_args.args == ({_paper("a").sha: "file_a"}, "file_kit")
    assert staging.in_flight_shas() == set()
