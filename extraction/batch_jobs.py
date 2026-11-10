"""Batch API jobs end to end: submit, reconcile, collect, discard.

The one implementation behind both the dashboard's batch section (app.py) and
the headless runner (`python -m extraction.bulk`), so a batch submitted from
either shows up in, and can be collected from, the other. It composes the
runner's batch calls (runner.submit_batch / runner.collect_batch) with the
on-disk records in extraction/staging.py and never writes the master DB —
results land in the review queue, and only a human approval merges them.

What it guarantees, and how:

- **No untracked paid batch, no orphaned upload.** A submission is journaled
  (a `_pending_` BatchJob) before the first upload, updated after every
  upload, and marked just before the create call, all while holding an OS
  lock. If the process dies anywhere in there, `reconcile` finds the journal
  with its lock free and resolves it against the API's batch list: a batch
  created in the window is adopted (same request count, created after the
  attempt, tracked nowhere else); if none was, its uploads are deleted.
- **Never the same paper twice.** Submission checks and records under one
  lock: a paper staged for review, in any batch job (submitted, being
  submitted, or failed and not yet discarded) or in a running synchronous
  extraction is skipped with the reason.
- **Collection is safe to interrupt and repeat.** One collector per batch (an
  OS lock, released if the process dies — no stale lock to clear). Results
  are saved to data/transcripts/ before they are processed, a continuation
  saves every turn and resumes from the last one, and each paper is dropped
  from the job as soon as it is staged — so a re-run re-pays nothing and
  never stages a paper twice.
- **Credit exhaustion is a pause, not a mess.** A submission refused for
  credit leaves nothing behind; a collection that runs out keeps every
  completed turn and every failed paper on its job, and says so.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import config

from . import anthropic_client, runner, staging
from .anthropic_client import BatchInfo, CreditExhausted, is_credit_exhausted
from .staging import BatchJob, PaperRef
from .transcripts import TranscriptStore

# A journaled submission whose create call was attempted but matches no batch
# on the API this long afterwards was never created.
_ABANDON_AFTER = timedelta(minutes=15)
# How far either side of the create attempt an adoptable batch's creation time may lie.
_ADOPT_SKEW = timedelta(minutes=5)
# Warn when a batch's results are this close to the API's 29-day expiry.
EXPIRY_WARNING = timedelta(days=7)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


# --------------------------------------------------------------------------- #
# Submit
# --------------------------------------------------------------------------- #

@dataclass
class SubmitOutcome:
    job: BatchJob | None                 # None when nothing was submitted
    skipped: dict[str, str] = field(default_factory=dict)   # sha -> why it was not submitted


def submit(
    papers: list[tuple[PaperRef, bytes]],
    *,
    figure_is_curve: bool = True,
    qa_feedback: str | None = None,
    replacing_staged: bool = False,
    prompt_version: str | None = None,
    model: str | None = None,
) -> SubmitOutcome:
    """Submit the papers that are free to extract as one Batches API job.

    A paper already staged, in a batch job or in a running extraction is
    skipped (reported in `skipped`) — except that `replacing_staged` (a
    QA-feedback re-extraction of a staged paper) allows a staged one.

    Raises CreditExhausted (nothing submitted, nothing left behind) when the
    account has no credit, anthropic_client.BatchSubmissionUnknown when the
    create call's outcome is unknown (the journal is kept; `reconcile`
    resolves it), and re-raises anything else after cleaning up."""
    with ExitStack() as stack:
        with staging.submission_lock():
            busy = staging.in_flight_shas()
            staged = set() if replacing_staged else set(staging.load_all())
            skipped: dict[str, str] = {}
            chosen: list[tuple[PaperRef, bytes]] = []
            for paper, pdf in papers:
                if paper.sha in busy:
                    skipped[paper.sha] = "already in a batch job (or being extracted)"
                elif paper.sha in staged:
                    skipped[paper.sha] = "already staged for review"
                elif any(paper.sha == p.sha for p, _ in chosen):
                    skipped[paper.sha] = "listed twice"
                else:
                    chosen.append((paper, pdf))
            if not chosen:
                return SubmitOutcome(None, skipped)
            job = BatchJob(batch_id=None, file_ids={}, items={}, papers={p.sha: p for p, _ in chosen})
            stack.enter_context(staging.intent_lock(job.job_id))
            job.save()   # the journal entry: these papers are now in flight

        def on_upload(custom_id, file_id):
            if custom_id is None:
                job.toolkit_file_id = file_id
            else:
                job.file_ids[custom_id] = file_id
            job.save()

        def on_prepared(items):
            job.items = items
            job.save()

        def before_create():
            job.create_attempted_at = _now().isoformat()
            job.save()

        try:
            submission, items = runner.submit_batch(
                [(p.sha, pdf) for p, pdf in chosen],
                figure_is_curve=figure_is_curve, qa_feedback=qa_feedback,
                prompt_version=prompt_version, model=model,
                on_prepared=on_prepared, on_upload=on_upload, before_create=before_create,
            )
        except anthropic_client.BatchSubmissionUnknown:
            raise   # keep the journal and its uploads: the batch may exist
        except Exception as e:
            job.delete()   # nothing was created; anthropic_client deleted the uploads
            if is_credit_exhausted(e):
                raise CreditExhausted(
                    "The Anthropic credit balance is exhausted — the batch was not submitted and "
                    "nothing was left behind. Top up, then submit again."
                ) from e
            raise
        job.batch_id = submission.batch_id
        job.created_at = submission.created_at
        job.items = items
        job.file_ids = dict(submission.file_ids)
        job.toolkit_file_id = submission.toolkit_file_id
        job.save()   # becomes _batch_<id>; the journal entry is removed
    return SubmitOutcome(job, skipped)


# --------------------------------------------------------------------------- #
# Reconcile (status)
# --------------------------------------------------------------------------- #

@dataclass
class JobStatus:
    job: BatchJob
    info: BatchInfo | None = None
    error: str | None = None             # couldn't ask the API about it

    @property
    def expires_in(self) -> timedelta:
        return self.job.results_expire_at() - _now()


@dataclass
class ReconcileReport:
    jobs: list[JobStatus] = field(default_factory=list)          # submitted jobs, with the API's view
    adopted: list[BatchJob] = field(default_factory=list)        # interrupted submissions whose batch was found
    abandoned: list[BatchJob] = field(default_factory=list)      # interrupted before a batch existed: cleaned up
    unresolved: list[tuple[BatchJob, str]] = field(default_factory=list)
    untracked: list[BatchInfo] = field(default_factory=list)     # batches on the account no local job tracks
    api_error: str | None = None


def _abandon(job: BatchJob) -> None:
    anthropic_client.cleanup_batch_files(job.file_ids, job.toolkit_file_id)
    job.delete()


def reconcile(*, now: datetime | None = None) -> ReconcileReport:
    """Square the local job records with the API (free metadata calls only).

    Resolves interrupted submissions (adopt the batch they created, or clean
    up after one that never was), reports every submitted job's API status
    and how long its results remain downloadable, and lists batches on the
    account (from the last 29 days) that no local record tracks — e.g. one
    submitted from before this journal existed, or by another tool on the
    same key."""
    now = now or _now()
    report = ReconcileReport()
    try:
        listed = anthropic_client.list_batches(now - timedelta(days=config.BATCH_RESULTS_RETENTION_DAYS))
    except Exception as e:
        listed = None
        report.api_error = str(e)

    jobs = staging.load_batch_jobs()
    tracked = set(jobs)
    for job in staging.load_pending_submissions():
        if staging.intent_busy(job.job_id):
            report.unresolved.append((job, "the submission is still running"))
            continue
        if job.create_attempted_at is None:
            # Stopped before the create call: no batch can exist for it.
            _abandon(job)
            report.abandoned.append(job)
            continue
        if listed is None:
            report.unresolved.append((job, "can't reach the API to check whether its batch was created"))
            continue
        attempted = datetime.fromisoformat(job.create_attempted_at)
        candidates = [
            b for b in listed
            if b.batch_id not in tracked
            and attempted - _ADOPT_SKEW <= b.created_at <= attempted + _ADOPT_SKEW
            and b.requests == len(job.items)
        ]
        if len(candidates) == 1:
            job.batch_id = candidates[0].batch_id
            job.created_at = _iso(candidates[0].created_at)
            job.save()
            tracked.add(job.batch_id)
            jobs[job.batch_id] = job
            report.adopted.append(job)
        elif not candidates and now - attempted > _ABANDON_AFTER:
            _abandon(job)
            report.abandoned.append(job)
        elif not candidates:
            report.unresolved.append((job, "its create call was moments ago — check again shortly"))
        else:
            report.unresolved.append((
                job, f"{len(candidates)} untracked batches match it "
                     f"({', '.join(b.batch_id for b in candidates)}) — resolve by hand"))

    by_id = {b.batch_id: b for b in listed or []}
    for batch_id, job in jobs.items():
        info = by_id.get(batch_id)
        error = None
        if info is None:
            try:
                info = anthropic_client.batch_info(batch_id)
            except Exception as e:
                error = str(e)
        report.jobs.append(JobStatus(job, info, error))
    report.untracked = [b for b in listed or [] if b.batch_id not in tracked]
    return report


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #

@dataclass
class PaperOutcome:
    sha: str
    filename: str
    staged: bool
    rows: int = 0
    reds: int = 0
    ambers: int = 0
    cost_usd: float | None = None
    error: str | None = None
    transcript: str | None = None


@dataclass
class CollectOutcome:
    batch_id: str
    status: str                  # 'collected' | 'not_ended' | 'busy' | 'gone'
    processing_status: str | None = None
    papers: list[PaperOutcome] = field(default_factory=list)
    credit_exhausted: bool = False

    @property
    def failures(self) -> list[PaperOutcome]:
        return [p for p in self.papers if not p.staged]


def collect(batch_id: str) -> CollectOutcome:
    """Fold an ended batch's results into the review queue, exactly as the
    dashboard always has: parse + QA each result and stage it. Paused items
    are finished with synchronous continuations that resume from their saved
    turns. Failed papers stay on the job (visible, re-collectable,
    discardable); a job whose every paper is staged is removed and its
    uploads deleted. Never touches the master DB."""
    from evaluation.cost import result_usd

    try:
        with staging.collection_lock(batch_id):
            job = staging.load_batch_jobs().get(batch_id)   # fresh: another collector may have finished it
            if job is None:
                return CollectOutcome(batch_id, "gone")
            status = runner.batch_status(batch_id)
            if status != "ended":
                return CollectOutcome(batch_id, "not_ended", status)

            store = TranscriptStore(batch_id)
            results = runner.collect_batch(
                batch_id, job.items, job.file_ids, job.toolkit_file_id, store=store
            )
            outcome = CollectOutcome(batch_id, "collected", status)
            errors: dict[str, str] = {}
            for sha, result in results.items():
                paper = job.papers[sha]
                if isinstance(result, Exception):
                    staging.record_failed_run(paper, result, batch_id=batch_id)
                    errors[sha] = str(result)
                    outcome.credit_exhausted |= is_credit_exhausted(result)
                    outcome.papers.append(PaperOutcome(
                        sha, paper.filename, False, error=str(result),
                        transcript=getattr(result, "transcript", None)))
                    continue
                staging.stage(paper, job.items[sha].figure_is_curve, result)
                job.mark_staged(sha)
                report = result.qa_report
                outcome.papers.append(PaperOutcome(
                    sha, paper.filename, True, rows=len(result.df), reds=len(report.reds),
                    ambers=len(report.ambers), cost_usd=result_usd(result), transcript=result.transcript))
            if errors:
                job.record_failures(errors)
            else:
                anthropic_client.cleanup_batch_files(job.file_ids, job.toolkit_file_id)
                job.delete()
            return outcome
    except staging.Busy:
        return CollectOutcome(batch_id, "busy")


def discard(batch_id: str) -> None:
    """Give up on a job's remaining (failed) papers: delete its uploads and its
    record, which frees those papers to be submitted again. Their transcripts
    and failed-run entries stay. Refused (staging.Busy) while it is being collected."""
    with staging.collection_lock(batch_id):
        job = staging.load_batch_jobs().get(batch_id)
        if job is None:
            return
        anthropic_client.cleanup_batch_files(job.file_ids, job.toolkit_file_id)
        job.delete()
