"""On-disk persistence for the review queue and in-flight Batch API jobs.

Everything the extraction page keeps in Streamlit session state is also written
under `data/staging/` so a server restart, idle timeout, or browser refresh
never loses an extraction that already cost API money:

  <sha>.xlsx                 the extracted 26-column table (also what the
                             reviewer edits)
  <sha>.meta.json            everything else about that staged paper
  _batch_<id>.batch.json     a submitted-but-not-yet-collected Batches API job

Pure file I/O — no Streamlit, no DB — so it is unit-testable and app.py stays
a thin UI layer over it. Sidecar layouts are flat JSON, and loading tolerates
missing keys, so sidecars written by earlier versions of the app keep loading.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from validation.report import QAReport

from .runner import BatchItem, ExtractionResult


@dataclass
class PaperRef:
    """Identity of one uploaded PDF, as the review/merge step needs it."""
    sha: str
    pdf_path: str
    doi: str | None
    filename: str
    meta: dict            # ingestion.pdf_inspect.inspect() output (title, n_pages, ...)


@dataclass
class StagedPaper:
    """One extraction awaiting review."""
    paper: PaperRef
    figure_is_curve: bool
    result: ExtractionResult


# --------------------------------------------------------------------------- #
# ExtractionResult <-> JSON. The DataFrame lives in its own .xlsx and the
# QAReport has its own JSON form; every other field is a plain JSON value.
# --------------------------------------------------------------------------- #

_RESULT_JSON_FIELDS = [
    f.name for f in fields(ExtractionResult) if f.name not in ("df", "qa_report")
]


def _result_payload(result: ExtractionResult) -> dict:
    payload = {name: getattr(result, name) for name in _RESULT_JSON_FIELDS}
    payload["qa_report_json"] = result.qa_report.to_json()
    return payload


def _result_from_payload(df: pd.DataFrame, payload: dict) -> ExtractionResult:
    kwargs = {name: payload[name] for name in _RESULT_JSON_FIELDS if name in payload}
    return ExtractionResult(
        df=df, qa_report=QAReport.from_json(payload.get("qa_report_json")), **kwargs
    )


def _paper_from_payload(sha: str, payload: dict) -> PaperRef:
    return PaperRef(
        sha=sha,
        pdf_path=payload["pdf_path"],
        doi=payload.get("doi"),
        filename=payload["filename"],
        meta=payload.get("meta") or {},
    )


# --------------------------------------------------------------------------- #
# Review queue
# --------------------------------------------------------------------------- #

def _meta_path(sha: str):
    return config.STAGING_DIR / f"{sha}.meta.json"


def table_path(sha: str):
    return config.STAGING_DIR / f"{sha}.xlsx"


def stage(paper: PaperRef, figure_is_curve: bool, result: ExtractionResult) -> StagedPaper:
    """Persist an extraction for review (overwriting any earlier staging of the
    same paper, e.g. after a QA-feedback re-extraction) and return its queue entry."""
    config.ensure_dirs()
    result.df.to_excel(table_path(paper.sha), index=False, engine="openpyxl")
    payload = {**asdict(paper), "figure_is_curve": figure_is_curve, **_result_payload(result)}
    _meta_path(paper.sha).write_text(json.dumps(payload), encoding="utf-8")
    return StagedPaper(paper=paper, figure_is_curve=figure_is_curve, result=result)


def discard(sha: str) -> None:
    """Remove a paper's staging files (after approve or reject)."""
    for p in (_meta_path(sha), table_path(sha)):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def load_all() -> dict[str, StagedPaper]:
    """Every staged paper on disk, keyed by sha. A sidecar without its table, or
    one that no longer parses, is skipped (and an orphaned sidecar removed)."""
    config.ensure_dirs()
    out: dict[str, StagedPaper] = {}
    for meta_file in sorted(config.STAGING_DIR.glob("*.meta.json")):
        sha = meta_file.name.removesuffix(".meta.json")
        xlsx = table_path(sha)
        if not xlsx.exists():
            meta_file.unlink(missing_ok=True)
            continue
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
            df = pd.read_excel(xlsx, engine="openpyxl")
            out[sha] = StagedPaper(
                paper=_paper_from_payload(sha, payload),
                figure_is_curve=payload.get("figure_is_curve", True),
                result=_result_from_payload(df, payload),
            )
        except Exception:
            continue  # corrupt sidecar — leave it for manual inspection, don't crash the page
    return out


# --------------------------------------------------------------------------- #
# Batch API jobs
# --------------------------------------------------------------------------- #

# A paused batch item is finished off with a synchronous pause_turn
# continuation (anthropic_client._continue_until_done), which can take a while
# and has no visual feedback of its own — long enough that a user clicking
# "Check status" again before it finishes previously restarted the whole
# (expensive) continuation from scratch. `collection_started_at` is a lock
# against that second attempt; the staleness window lets it self-heal if a
# prior attempt crashed without clearing it.
COLLECTION_LOCK_STALE_AFTER = timedelta(minutes=10)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BatchJob:
    """A submitted Batches API job, persisted until every paper in it has
    either landed in the review queue or been explicitly discarded."""
    batch_id: str
    file_ids: dict[str, str]            # sha -> Files API id, deleted after collection
    items: dict[str, BatchItem]         # sha -> what runner.collect_batch needs
    papers: dict[str, PaperRef]         # sha -> identity for the review/merge step
    submitted_at: str = field(default_factory=_now_iso)
    errors: dict[str, str] = field(default_factory=dict)   # sha -> why collection failed
    collection_started_at: str | None = None

    @property
    def path(self):
        return config.STAGING_DIR / f"_batch_{self.batch_id}.batch.json"

    def collection_in_progress(self) -> bool:
        if not self.collection_started_at:
            return False
        started = datetime.fromisoformat(self.collection_started_at)
        return datetime.now(timezone.utc) - started < COLLECTION_LOCK_STALE_AFTER

    def begin_collection(self) -> None:
        """Take the collection lock (persisted, so it holds across reruns)."""
        self.collection_started_at = _now_iso()
        self.save()

    def record_failures(self, errors: dict[str, str]) -> None:
        """Keep only the papers that failed, so they stay visible across reruns
        and restarts until the user discards them; release the lock."""
        self.items = {sha: v for sha, v in self.items.items() if sha in errors}
        self.papers = {sha: v for sha, v in self.papers.items() if sha in errors}
        self.errors = errors
        self.collection_started_at = None
        self.save()

    def release_lock(self) -> None:
        self.collection_started_at = None
        self.save()

    def save(self) -> None:
        config.ensure_dirs()
        self.path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)

    @classmethod
    def from_dict(cls, payload: dict) -> "BatchJob":
        return cls(
            batch_id=payload["batch_id"],
            file_ids=payload["file_ids"],
            items={sha: BatchItem(**d) for sha, d in payload["items"].items()},
            papers={
                sha: _paper_from_payload(sha, d) for sha, d in payload["papers"].items()
            },
            submitted_at=payload.get("submitted_at") or _now_iso(),
            errors=payload.get("errors") or {},
            collection_started_at=payload.get("collection_started_at"),
        )


def load_batch_jobs() -> dict[str, BatchJob]:
    """Every in-flight batch job on disk, keyed by batch_id (corrupt sidecars skipped)."""
    config.ensure_dirs()
    out: dict[str, BatchJob] = {}
    for f in sorted(config.STAGING_DIR.glob("_batch_*.batch.json")):
        try:
            job = BatchJob.from_dict(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
        out[job.batch_id] = job
    return out
