"""python -m extraction.bulk: which manifest papers get submitted, and that
the commands drive extraction/batch_jobs.py (patched — no API calls)."""
import hashlib
from unittest.mock import patch

import pytest

import config
from extraction import batch_jobs, bulk, staging
from extraction.batch_jobs import CollectOutcome, PaperOutcome, ReconcileReport, SubmitOutcome
from extraction.staging import BatchJob

from test_licensing import write_manifest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "INCOMING_DIR", tmp_path / "incoming")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "master.db")    # absent: nothing approved
    for d in ("staging", "incoming", "open_access/pdfs"):
        (tmp_path / d).mkdir(parents=True)
    return tmp_path


def _pdf(env, body):
    data = b"%PDF-1.4 " + body
    sha = hashlib.sha256(data).hexdigest()
    (env / "open_access" / "pdfs" / f"{sha}.pdf").write_bytes(data)
    return sha


def _manifest(env):
    a, b, c, d, e = (_pdf(env, x) for x in (b"a", b"b", b"c", b"d", b"e"))
    rows = [
        {"sha256": a, "license": "cc-by", "tier": "A", "first_batch": "1", "title": "Alpha"},
        {"sha256": b, "license": "cc-by-nc-nd", "tier": "B", "first_batch": "1", "title": "Beta"},
        {"sha256": c, "license": "all-rights-reserved", "tier": "A", "first_batch": "1", "title": "Closed"},
        {"sha256": d, "license": "cc0", "tier": "reject", "first_batch": "1", "title": "Rejected"},
        {"sha256": e, "license": "cc-by", "tier": "A", "first_batch": "", "title": "Later"},
        {"sha256": "f" * 64, "license": "cc-by", "tier": "A", "first_batch": "1", "title": "No PDF"},
    ]
    return write_manifest(env / "open_access" / "manifest.csv", rows), (a, b, c, d, e)


def test_selection_takes_only_open_unreviewed_papers_with_their_pdf(env):
    m, (a, b, c, d, e) = _manifest(env)
    chosen, left_out = bulk.select(bulk.manifest.read(m), first_batch=True, tiers=[], limit=None)
    assert [x.sha256 for x in chosen] == [a, b]
    reasons = {x.title: why for x, why in left_out}
    assert "not an open licence" in reasons["Closed"] and "tier is reject" in reasons["Rejected"]
    assert "PDF not found" in reasons["No PDF"] and "Later" not in reasons

    chosen, _ = bulk.select(bulk.manifest.read(m), first_batch=False, tiers=["A"], limit=1)
    assert [x.sha256 for x in chosen] == [a]


def test_selection_never_resubmits_a_staged_or_in_flight_paper(env):
    m, (a, b, *_rest) = _manifest(env)
    BatchJob(batch_id="msgbatch_1", file_ids={}, items={}, papers={a: staging.PaperRef(a, "p", None, "f", {})}).save()
    chosen, left_out = bulk.select(bulk.manifest.read(m), first_batch=True, tiers=[], limit=None)
    assert [x.sha256 for x in chosen] == [b]
    assert any("already in a batch job" in why for x, why in left_out if x.sha256 == a)


def test_reextract_submits_a_staged_paper_by_sha_prefix_and_nothing_else(env, capsys):
    m, (a, b, c, *_rest) = _manifest(env)
    with patch.object(staging, "load_all", return_value={a: None}):
        chosen, _ = bulk.select(bulk.manifest.read(m), first_batch=True, tiers=[], limit=None)
        assert [x.sha256 for x in chosen] == [b]                      # staged: normally left alone
        chosen, left_out = bulk.select(bulk.manifest.read(m), first_batch=False, tiers=[], limit=None,
                                       reextract=[a[:12].upper(), c[:12]])
        assert [x.sha256 for x in chosen] == [a]
        assert ["not an open licence" in why for _, why in left_out] == [True]   # the gate still holds
        with patch.object(batch_jobs, "submit", return_value=SubmitOutcome(None, {})) as submit:
            assert bulk.main(["submit", "--manifest", str(m), "--reextract", a[:12]]) == 0
            assert submit.call_args.kwargs["replacing_staged"] is True
            assert bulk.main(["submit", "--manifest", str(m), "--reextract", "0000"]) == 1
    assert "no manifest row" in capsys.readouterr().err


def test_dry_run_submits_nothing(env, capsys):
    m, _ = _manifest(env)
    with patch.object(batch_jobs, "submit") as submit:
        assert bulk.main(["submit", "--manifest", str(m), "--first-batch", "--dry-run"]) == 0
    submit.assert_not_called()
    assert "2 paper(s) to submit" in capsys.readouterr().out


def test_submit_ingests_with_the_manifest_licence_and_reports_the_batch(env, capsys):
    m, (a, b, *_rest) = _manifest(env)
    seen = {}

    def fake_submit(papers, **kw):
        seen["papers"] = papers
        return SubmitOutcome(BatchJob(batch_id="msgbatch_9", file_ids={}, items={p.sha: None for p, _ in papers},
                                      papers={}), {})
    with patch.object(batch_jobs, "submit", side_effect=fake_submit):
        assert bulk.main(["submit", "--manifest", str(m), "--tier", "A", "--tier", "B", "--limit", "2"]) == 0
    refs = [p for p, _ in seen["papers"]]
    assert [r.sha for r in refs] == [a, b] and refs[0].license.id == "cc-by" and refs[0].license.origin == "manifest"
    assert refs[0].filename == "Alpha" and (env / "incoming" / f"{a}.pdf").exists()
    assert "msgbatch_9" in capsys.readouterr().out


def test_submit_out_of_credit_exits_3(env):
    m, _ = _manifest(env)
    with patch.object(batch_jobs, "submit", side_effect=bulk.anthropic_client.CreditExhausted("no credit")):
        assert bulk.main(["submit", "--manifest", str(m), "--first-batch"]) == bulk.EXIT_NO_CREDIT


def test_collect_prints_each_paper_and_flags_failures(env, capsys):
    BatchJob(batch_id="msgbatch_1", file_ids={}, items={}, papers={}).save()
    outcome = CollectOutcome("msgbatch_1", "collected", "ended", [
        PaperOutcome("a" * 64, "Alpha", True, rows=120, reds=0, ambers=2, cost_usd=0.31),
        PaperOutcome("b" * 64, "Beta", False, error="no parseable JSON", transcript="transcripts/x/b.json.gz"),
    ])
    with patch.object(batch_jobs, "collect", return_value=outcome):
        assert bulk.main(["collect"]) == bulk.EXIT_FAILED
    out = capsys.readouterr().out
    assert "120 rows" in out and "0 red / 2 amber" in out and "$0.31" in out
    assert "FAILED" in out and "data/transcripts/x/b.json.gz" in out and "discard --batch msgbatch_1" in out


def test_status_reconciles_and_shows_the_queue(env, capsys):
    with patch.object(batch_jobs, "reconcile", return_value=ReconcileReport(api_error="no key")):
        assert bulk.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "could not list batches" in out and "Review queue (0 staged)" in out
