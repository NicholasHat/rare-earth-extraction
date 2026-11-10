"""On-disk persistence for the review queue and in-flight Batch API jobs.

Everything the extraction page keeps in Streamlit session state is also written
under `data/staging/` so a server restart, idle timeout, or browser refresh
never loses an extraction that already cost API money:

  <sha>.meta.json            a staged paper: identity, licence, the extracted
                             table, QA report, usage — the record
  <sha>.xlsx                 the same table as a spreadsheet, for opening by
                             hand (written first; the sidecar is the commit point)
  _batch_<id>.batch.json     a submitted-but-not-yet-collected Batches API job
  _pending_<job>.batch.json  a batch submission in progress (or interrupted):
                             written before anything is uploaded or paid for, so
                             a crash can never leave an untracked batch or upload
  _failed_runs.jsonl         append-only log of extractions that failed after
                             spending API money (what failed, tokens billed,
                             where the transcript is)
  _superseded/               a staged result that a re-extraction replaced
  _locks/                    OS file locks (released by the OS when the
                             holding process dies — no stale locks to clear)

Pure file I/O — no Streamlit, no DB — so it is unit-testable and app.py stays
a thin UI layer over it. Every file is written to a temporary name and
os.replace'd into place, so a crash leaves the previous version or the new
one, never half of either. Sidecar layouts are flat JSON, and loading
tolerates missing keys, so sidecars written by earlier versions of the app
keep loading (a 26-column table loads with the added columns null).
"""
from __future__ import annotations

import fcntl
import io
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
from ingestion.licensing import License
from validation import schema
from validation.report import QAReport

from .parse_output import fill_figures
from .runner import BatchItem, ExtractionResult


@dataclass
class PaperRef:
    """Identity of one uploaded PDF, as the review/merge step needs it."""
    sha: str
    pdf_path: str
    doi: str | None
    filename: str
    meta: dict            # ingestion.pdf_inspect.inspect() output (title, n_pages, ...)
    license: License | None = None   # from the open-access manifest, or set by hand at review


@dataclass
class StagedPaper:
    """One extraction awaiting review."""
    paper: PaperRef
    figure_is_curve: bool
    result: ExtractionResult


# --------------------------------------------------------------------------- #
# Atomic writes and OS file locks
# --------------------------------------------------------------------------- #

def _write_atomic(path: Path, data: bytes) -> None:
    """Write `data` to `path` so a reader (or a crash) sees the old file or the
    new one, never a torn mix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json(path: Path, payload) -> None:
    _write_atomic(path, json.dumps(payload).encode("utf-8"))


class Busy(RuntimeError):
    """Another process (or another click) is already doing this."""


def _lock_path(name: str) -> Path:
    return config.STAGING_DIR / "_locks" / f"{name}.lock"


@contextmanager
def _flock(name: str, what: str, *, wait: bool = False):
    """Hold an exclusive OS lock named `name` for the duration of the block.
    Raises Busy (naming `what`) if someone else holds it, unless `wait`.

    flock locks belong to the open file, so they are released when the holder
    exits for any reason — a crash, a kill, a laptop shutdown — which is what
    makes them safe for guarding paid work: no staleness window to guess, no
    lock file to delete by hand. Two opens in the same process (two Streamlit
    sessions) exclude each other too. The lock file itself is left in place:
    unlinking it could let a waiter lock an inode nobody else sees."""
    path = _lock_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise Busy(f"{what} is already in progress in another process or session") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} since {_now_iso()}\n".encode())
        yield
    finally:
        os.close(fd)


def _is_locked(name: str) -> bool:
    path = _lock_path(name)
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def collection_lock(batch_id: str):
    """Exclusive right to collect one batch (download, continue, stage). A
    second collector — another click, the CLI and the dashboard at once —
    gets Busy instead of paying for the same continuation twice."""
    return _flock(f"collect_{batch_id}", f"collecting batch {batch_id}")


def collection_busy(batch_id: str) -> bool:
    return _is_locked(f"collect_{batch_id}")


def claim(sha: str):
    """Exclusive right to run a synchronous extraction of one paper; held for
    the whole call, so a double click (or a second session) is refused rather
    than paid for twice."""
    return _flock(f"claim_{sha}", f"an extraction of {sha[:12]}…")


def submission_lock():
    """Serialises the check-then-record step of every batch submission (see
    extraction/batch_jobs.py). Short; waits rather than failing."""
    return _flock("submit", "a batch submission", wait=True)


def intent_lock(job_id: str):
    """Held by the process submitting a batch for the whole submission; free
    means the submitter finished or died (batch_jobs.reconcile checks)."""
    return _flock(f"intent_{job_id}", f"submission {job_id}")


def intent_busy(job_id: str) -> bool:
    return _is_locked(f"intent_{job_id}")


def claimed_shas() -> set[str]:
    """Papers a synchronous extraction is running for right now."""
    lock_dir = config.STAGING_DIR / "_locks"
    if not lock_dir.exists():
        return set()
    out = set()
    for f in lock_dir.glob("claim_*.lock"):
        name = f.name.removesuffix(".lock")
        if _is_locked(name):
            out.add(name.removeprefix("claim_"))
    return out


# --------------------------------------------------------------------------- #
# ExtractionResult <-> JSON. The QAReport has its own JSON form and the table
# is stored in pandas' "split" JSON; every other field is a plain JSON value.
# --------------------------------------------------------------------------- #

_RESULT_JSON_FIELDS = [
    f.name for f in fields(ExtractionResult) if f.name not in ("df", "qa_report")
]


def _result_payload(result: ExtractionResult) -> dict:
    payload = {name: getattr(result, name) for name in _RESULT_JSON_FIELDS}
    payload["qa_report_json"] = result.qa_report.to_json()
    payload["table"] = result.df.to_json(orient="split", date_format="iso")
    return payload


def _table_from_payload(sha: str, payload: dict) -> pd.DataFrame:
    """The staged table: from the sidecar itself, or — for a sidecar written
    before tables lived there — from its .xlsx. Coerced to today's schema; a
    v13–v14 result's parallel `row_figures` list becomes its Figure/Table column."""
    if payload.get("table") is not None:
        df = pd.read_json(io.StringIO(payload["table"]), orient="split", dtype=False, convert_dates=False)
    else:
        df = pd.read_excel(table_path(sha), engine="openpyxl")
    return fill_figures(schema.coerce_schema(df), payload.get("row_figures"))


def _result_from_payload(df: pd.DataFrame, payload: dict) -> ExtractionResult:
    kwargs = {name: payload[name] for name in _RESULT_JSON_FIELDS if name in payload}
    return ExtractionResult(
        df=df, qa_report=QAReport.from_json(payload.get("qa_report_json")), **kwargs
    )


def _paper_payload(paper: PaperRef) -> dict:
    return asdict(paper)


def _paper_from_payload(sha: str, payload: dict) -> PaperRef:
    lic = payload.get("license")
    return PaperRef(
        sha=sha,
        pdf_path=payload["pdf_path"],
        doi=payload.get("doi"),
        filename=payload["filename"],
        meta=payload.get("meta") or {},
        license=License(**lic) if isinstance(lic, dict) else None,
    )


# --------------------------------------------------------------------------- #
# Review queue
# --------------------------------------------------------------------------- #

def _meta_path(sha: str) -> Path:
    return config.STAGING_DIR / f"{sha}.meta.json"


def table_path(sha: str) -> Path:
    return config.STAGING_DIR / f"{sha}.xlsx"


def _superseded_dir() -> Path:
    return config.STAGING_DIR / "_superseded"


def _keep_superseded(sha: str, incoming: ExtractionResult) -> None:
    """Before a different result replaces a staged one (a re-extraction), move
    the old one aside rather than destroying output that was paid for.
    Re-staging the same run (Re-run QA, a licence set) keeps nothing."""
    meta = _meta_path(sha)
    if not meta.exists():
        return
    try:
        old = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        old = {}
    if old.get("raw_response") == incoming.raw_response:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = _superseded_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if table_path(sha).exists():
        os.replace(table_path(sha), dest / f"{sha}.{stamp}.xlsx")
    os.replace(meta, dest / f"{sha}.{stamp}.meta.json")


def stage(paper: PaperRef, figure_is_curve: bool, result: ExtractionResult) -> StagedPaper:
    """Persist an extraction for review (replacing any earlier staging of the
    same paper, e.g. after a QA-feedback re-extraction — whose old result is
    kept under _superseded/) and return its queue entry. The new result is
    fully on disk before the sidecar, the commit point, is swapped in."""
    config.ensure_dirs()
    xlsx = io.BytesIO()
    result.df.to_excel(xlsx, index=False, engine="openpyxl")
    payload = {**_paper_payload(paper), "figure_is_curve": figure_is_curve, **_result_payload(result)}
    data = json.dumps(payload).encode("utf-8")
    _keep_superseded(paper.sha, result)
    _write_atomic(table_path(paper.sha), xlsx.getvalue())
    _write_atomic(_meta_path(paper.sha), data)
    return StagedPaper(paper=paper, figure_is_curve=figure_is_curve, result=result)


def set_license(sha: str, license: License | None) -> StagedPaper:
    """Record (or clear) the licence of a staged paper — the review-time
    control for a paper the open-access manifest doesn't cover. The merge
    gate (database/merge.py) reads it on approval."""
    staged = load_all().get(sha)
    if staged is None:
        raise KeyError(f"no staged paper {sha}")
    staged.paper.license = license
    return stage(staged.paper, staged.figure_is_curve, staged.result)


def discard(sha: str) -> None:
    """Remove a paper's staging files (after approve or reject). Its run's
    transcript stays under data/transcripts/."""
    for p in (_meta_path(sha), table_path(sha)):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def load_all() -> dict[str, StagedPaper]:
    """Every staged paper on disk, keyed by sha. A sidecar that doesn't load
    (corrupt, or an old one whose .xlsx is gone) is skipped and left in place
    for inspection — never deleted."""
    config.ensure_dirs()
    out: dict[str, StagedPaper] = {}
    for meta_file in sorted(config.STAGING_DIR.glob("*.meta.json")):
        sha = meta_file.name.removesuffix(".meta.json")
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
            df = _table_from_payload(sha, payload)
            out[sha] = StagedPaper(
                paper=_paper_from_payload(sha, payload),
                figure_is_curve=payload.get("figure_is_curve", True),
                result=_result_from_payload(df, payload),
            )
        except Exception:
            continue  # leave it for manual inspection, don't crash the page
    return out


# --------------------------------------------------------------------------- #
# Failed-runs log
# --------------------------------------------------------------------------- #

def _failed_runs_path() -> Path:
    return config.STAGING_DIR / "_failed_runs.jsonl"


def record_failed_run(paper: PaperRef, error: Exception, *, batch_id: str | None = None) -> None:
    """Append one failed extraction to the failed-runs log. Money was spent and
    nothing was staged, so this is the attempt's record: what failed, for an
    anthropic_client.ExtractionFailed the tokens billed before it did, and
    where the run's transcript (its raw output) is saved.

    `batch_id` names the Batches API job for a batch item. Retrying a failed
    collection re-reports that batch turn's usage, so entries sharing a
    batch_id are one run, not several (evaluation.sources counts it once)."""
    from .anthropic_client import is_credit_exhausted

    config.ensure_dirs()
    entry = {
        "at": _now_iso(),
        "sha": paper.sha,
        "filename": paper.filename,
        "error": str(error),
        "usage": getattr(error, "usage", None),
        "turns_completed": getattr(error, "turns_completed", None),
        "batch_id": batch_id,
        "transcript": getattr(error, "transcript", None),
        "credit_exhausted": is_credit_exhausted(error),
    }
    path = _failed_runs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # One write of one line in append mode: a crash can at worst leave a torn
    # last line, which load_failed_runs skips.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_failed_runs(limit: int = 10) -> list[dict]:
    """The most recent failed runs, newest first (corrupt lines skipped)."""
    path = _failed_runs_path()
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    return entries[::-1][:limit]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Batch API jobs
# --------------------------------------------------------------------------- #

@dataclass
class BatchJob:
    """A Batches API job, persisted from before its first upload until every
    paper in it has either landed in the review queue or been explicitly
    discarded. `batch_id` is None while the submission is in progress (or was
    interrupted before the API answered): that record is the journal that
    batch_jobs.reconcile resolves against the API's batch list."""
    batch_id: str | None
    file_ids: dict[str, str]            # sha -> Files API id, deleted after collection
    items: dict[str, BatchItem]         # sha -> what runner.collect_batch needs
    papers: dict[str, PaperRef]         # sha -> identity for the review/merge step
    submitted_at: str = field(default_factory=_now_iso)
    errors: dict[str, str] = field(default_factory=dict)   # sha -> why collection failed
    toolkit_file_id: str | None = None  # the job's sandbox-toolkit upload; None before it existed
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str | None = None       # the API's creation time of the batch
    create_attempted_at: str | None = None   # set just before the (paid) create call

    @property
    def pending(self) -> bool:
        """Submission not (known to be) accepted by the API yet."""
        return self.batch_id is None

    @property
    def path(self) -> Path:
        if self.batch_id:
            return config.STAGING_DIR / f"_batch_{self.batch_id}.batch.json"
        return self._pending_path

    @property
    def _pending_path(self) -> Path:
        return config.STAGING_DIR / f"_pending_{self.job_id}.batch.json"

    @property
    def shas(self) -> set[str]:
        """Every paper this job still holds (in flight, or failed and not discarded)."""
        return set(self.papers) | set(self.items)

    def results_expire_at(self) -> datetime:
        """When the API stops serving this batch's results (29 days after creation)."""
        start = datetime.fromisoformat(self.created_at or self.submitted_at)
        return start + timedelta(days=config.BATCH_RESULTS_RETENTION_DAYS)

    def collection_in_progress(self) -> bool:
        return bool(self.batch_id) and collection_busy(self.batch_id)

    def mark_staged(self, sha: str) -> None:
        """One paper's result is in the review queue: drop it from the job, so
        an interrupted collection never stages it a second time."""
        self.items.pop(sha, None)
        self.papers.pop(sha, None)
        self.errors.pop(sha, None)
        self.save()

    def record_failures(self, errors: dict[str, str]) -> None:
        """Keep only the papers that failed, so they stay visible across reruns
        and restarts until the user discards them (or collects again)."""
        self.items = {sha: v for sha, v in self.items.items() if sha in errors}
        self.papers = {sha: v for sha, v in self.papers.items() if sha in errors}
        self.errors = errors
        self.save()

    def save(self) -> None:
        config.ensure_dirs()
        payload = asdict(self)
        _write_json(self.path, payload)
        if self.batch_id:
            # The submission was accepted: its journal entry is superseded.
            self._pending_path.unlink(missing_ok=True)

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)
        self._pending_path.unlink(missing_ok=True)

    @classmethod
    def from_dict(cls, payload: dict) -> "BatchJob":
        batch_id = payload.get("batch_id")
        return cls(
            batch_id=batch_id,
            file_ids=payload.get("file_ids") or {},
            items={sha: BatchItem(**d) for sha, d in (payload.get("items") or {}).items()},
            papers={
                sha: _paper_from_payload(sha, d) for sha, d in (payload.get("papers") or {}).items()
            },
            submitted_at=payload.get("submitted_at") or _now_iso(),
            errors=payload.get("errors") or {},
            toolkit_file_id=payload.get("toolkit_file_id"),
            job_id=payload.get("job_id") or batch_id or uuid.uuid4().hex,
            created_at=payload.get("created_at"),
            create_attempted_at=payload.get("create_attempted_at"),
        )


def _load_jobs() -> list[BatchJob]:
    config.ensure_dirs()
    jobs: dict[str, BatchJob] = {}
    files = sorted(config.STAGING_DIR.glob("_batch_*.batch.json")) + sorted(
        config.STAGING_DIR.glob("_pending_*.batch.json"))
    for f in files:
        try:
            job = BatchJob.from_dict(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
        # A submission accepted just before a crash can leave both its
        # journal entry and its job; the job (with a batch id) wins.
        if job.job_id in jobs and jobs[job.job_id].batch_id:
            continue
        jobs[job.job_id] = job
    return list(jobs.values())


def load_batch_jobs() -> dict[str, BatchJob]:
    """Every submitted batch job on disk, keyed by batch_id (corrupt sidecars skipped)."""
    return {j.batch_id: j for j in _load_jobs() if j.batch_id}


def load_pending_submissions() -> list[BatchJob]:
    """Batch submissions in progress, or interrupted before the API answered
    (resolved by batch_jobs.reconcile)."""
    return [j for j in _load_jobs() if not j.batch_id]


def in_flight_shas() -> set[str]:
    """Papers that must not be extracted again right now: held by a batch job
    (submitted, being submitted, or failed and not yet discarded) or by a
    running synchronous extraction."""
    return {sha for job in _load_jobs() for sha in job.shas} | claimed_shas()
