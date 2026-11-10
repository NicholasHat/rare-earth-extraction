"""Orchestrate extraction runs end to end (plan §6).

  load pinned prompt -> call Anthropic API -> parse output -> run QA checks

Returns ExtractionResult(s) for the review UI. It does NOT touch the DB —
merge happens only on human approval (database.merge.commit_extraction). The
one thing it writes is each run's transcript (extraction/transcripts.py):
every paid turn, saved as it completes.

Two entry points, sharing the same pre-pass + parse + QA logic (_prepass,
_postprocess):
  - `extract_paper()` — synchronous, one paper, blocks until done.
  - `submit_batch()` / `batch_status()` / `collect_batch()` — the Anthropic
    Message Batches API (50% cheaper, asynchronous). app.py's batch-jobs UI
    drives this across Streamlit reruns so a submitted batch survives a
    server restart while it's still processing.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import pandas as pd

import config
from validation import checks
from validation.report import QAReport

from . import anthropic_client, curve_prepass, parse_output, prompt_loader, provenance
from .transcripts import TranscriptStore


@dataclass
class ExtractionResult:
    df: pd.DataFrame
    text_endpoints: list[dict]
    qa_report: QAReport
    prompt_version: str
    prompt_sha256: str
    model: str
    raw_response: str
    coercion_failures: int
    curve_analysis: str = ""          # the injected deterministic pre-pass block
    deterministic_counts: list[int] = field(default_factory=list)  # authoritative per-series marker counts
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0  # high value here confirms the cache breakpoint is working
    figures: list[dict] = field(default_factory=list)      # every figure: digitised or why not (v13+)
    via_batch: bool = False           # billed at the Batches API's 50% rate (a paused item's synchronous continuation, if any, at full rate)
    continuation_usage: dict = field(default_factory=dict)  # tokens of turns after the first (a batch item's full-price continuation)
    transcript: str = ""              # where every paid turn of this run is saved, relative to data/

    @property
    def row_figures(self) -> list[str]:
        """The source figure/table of each row — its `Figure/Table` column."""
        return provenance.row_figures(self.df)


def _prepass(pdf_bytes: bytes) -> tuple[str, list[int]]:
    """Deterministic pre-pass: count markers from the PDF's own vector geometry and
    return the grounding-anchor text + authoritative counts (plan §6). Pure / no
    API; a failure here must never block the extraction, so it degrades to
    "no anchor"."""
    try:
        prepass = curve_prepass.analyze(pdf_bytes)
        return prepass.to_prompt_block(), prepass.authoritative_counts
    except Exception:
        return "", []


def rerun_qa(result: ExtractionResult, *, figure_is_curve: bool) -> ExtractionResult:
    """The same QA the extraction ran, re-applied to an already-staged result —
    pure, no API call. Lets a paper staged before a check existed (or before a
    check learned to name rows) pick up today's checks without paying for
    another extraction. Everything but the report is carried over unchanged."""
    report = checks.run(
        result.df,
        result.text_endpoints,
        figure_is_curve=figure_is_curve,
        coercion_failures=result.coercion_failures,
        deterministic_counts=result.deterministic_counts,
    )
    return replace(result, qa_report=report)


def _postprocess(
    response: anthropic_client.ExtractResponse,
    *,
    figure_is_curve: bool,
    deterministic_counts: list[int],
    analysis_block: str,
    prompt_version: str,
    prompt_sha256: str,
    model: str,
    via_batch: bool = False,
    transcript: str = "",
) -> ExtractionResult:
    """Parse a model response and run QA — shared by the sync and batch paths."""
    parsed = parse_output.parse(response.text)
    report = checks.run(
        parsed.df,
        parsed.text_endpoints,
        figure_is_curve=figure_is_curve,
        coercion_failures=parsed.coercion_failures,
        deterministic_counts=deterministic_counts,
    )
    return ExtractionResult(
        df=parsed.df,
        text_endpoints=parsed.text_endpoints,
        qa_report=report,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        model=model,
        raw_response=parsed.raw_text,
        coercion_failures=parsed.coercion_failures,
        curve_analysis=analysis_block,
        deterministic_counts=deterministic_counts,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cache_creation_input_tokens=response.cache_creation_input_tokens,
        cache_read_input_tokens=response.cache_read_input_tokens,
        figures=parsed.figures,
        via_batch=via_batch,
        continuation_usage=dict(response.continuation_usage),
        transcript=transcript,
    )


def qa_feedback_block(report: QAReport, row_count: int) -> str:
    """Render a previous attempt's QA failures as a user-turn feedback block for
    an on-demand re-extraction (the review UI's re-extract action). Injected
    per-run like the curve pre-pass block — never folded into the pinned,
    versioned prompt file."""
    lines = [
        "## QA FEEDBACK ON A PREVIOUS EXTRACTION ATTEMPT",
        f"A previous extraction of this same paper produced {row_count} rows, but "
        "automated QA raised the findings below. Redo the full extraction per your "
        "instructions, specifically resolving each finding — keep whatever the "
        "findings don't dispute.",
    ]
    lines += [f"- [{f.severity.value.upper()}] {f.check}: {f.message}" for f in report.flags]
    return "\n".join(lines)


def must_use_batch(paper_meta: dict) -> bool:
    """Whether a paper is only ever extracted through the Batches API.

    Raster-figure papers are. Their code-execution loop runs far longer than a
    vector paper's (Quinn et al. 2015 was still digitizing after 51 iterations
    under extraction_v9), and every iteration re-reads the whole growing
    transcript, so cost grows faster than the loop does. Batch requests are
    half price for the same 50-iteration turn, and the first turn runs on
    Anthropic's side, so it survives the browser session or the laptop going
    away; the synchronous path holds one streamed connection per turn for as
    long as the loop runs (2026-09-22: a ~$7.50 synchronous run).
    `paper_meta` is ingestion.pdf_inspect.inspect()'s dict.
    """
    return bool(paper_meta.get("is_raster_figure"))


def extract_paper(
    pdf_bytes: bytes,
    *,
    figure_is_curve: bool = True,
    prompt_version: str | None = None,
    model: str | None = None,
    qa_feedback: str | None = None,
) -> ExtractionResult:
    """Run extraction + QA on one PDF's bytes (synchronous — blocks until done).

    Every completed turn is saved to a transcript (data/transcripts/sync_*/)
    as it arrives; a failure's ExtractionFailed names it in `.transcript`."""
    bundle = prompt_loader.load_prompt(prompt_version)
    model = model or config.EXTRACTION_MODEL
    analysis_block, deterministic_counts = _prepass(pdf_bytes)
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    store = TranscriptStore.for_sync_run()

    try:
        response = anthropic_client.extract(
            bundle.text, pdf_bytes, model=model,
            analysis_block=analysis_block or None, qa_feedback=qa_feedback,
            on_turn=lambda chain: store.save(sha, chain),
        )
        try:
            return _postprocess(
                response,
                figure_is_curve=figure_is_curve,
                deterministic_counts=deterministic_counts,
                analysis_block=analysis_block,
                prompt_version=bundle.version,
                prompt_sha256=bundle.sha256,
                model=model,
                transcript=store.relpath(sha) if store.path(sha).exists() else "",
            )
        except Exception as e:
            # The call succeeded and was billed; a parse/QA failure must still say so.
            raise anthropic_client.ExtractionFailed(e, response.usage()) from e
    except anthropic_client.ExtractionFailed as e:
        if store.path(sha).exists():
            e.transcript = store.relpath(sha)
        raise


# --------------------------------------------------------------------------- #
# Message Batches API path
# --------------------------------------------------------------------------- #


@dataclass
class BatchItem:
    """One paper queued for the Batches API, keyed by content sha256."""
    custom_id: str
    figure_is_curve: bool
    analysis_block: str
    deterministic_counts: list[int]
    prompt_version: str
    prompt_sha256: str
    model: str
    qa_feedback: str | None = None   # set on a batched re-extraction (see submit_batch)


def submit_batch(
    papers: list[tuple[str, bytes]],
    *,
    figure_is_curve: bool = True,
    prompt_version: str | None = None,
    model: str | None = None,
    qa_feedback: str | None = None,
    on_prepared=None,
    on_upload=None,
    before_create=None,
) -> tuple[anthropic_client.BatchSubmission, dict[str, "BatchItem"]]:
    """Run the deterministic pre-pass for each paper and submit one Batches API
    job covering all of them.

    `papers` is a list of (custom_id, pdf_bytes); custom_id is the paper's
    content sha256. Returns the submission (batch id, per-paper file ids, the
    toolkit file id) and fills `items` — the caller persists all of it so the
    batch can be checked/collected later, even across a server restart.

    `qa_feedback` is the previous attempt's QA block (qa_feedback_block) for an
    on-demand re-extraction, exactly as extract_paper takes it; it applies to
    every paper in the job, so a re-extraction submits a one-paper job. It is
    kept on the BatchItem because a paused item is rebuilt from the item when
    its continuation is sent.

    `on_prepared(items)`, `on_upload(custom_id, file_id)` and `before_create()`
    are journaling hooks (extraction/batch_jobs.py records each step before
    the next, so a crash never leaves an untracked paid batch or upload); see
    anthropic_client.submit_batch for the failure modes.
    """
    bundle = prompt_loader.load_prompt(prompt_version)
    model = model or config.EXTRACTION_MODEL

    items: dict[str, BatchItem] = {}
    requests: list[anthropic_client.BatchRequest] = []
    pdfs: dict[str, bytes] = {}
    for custom_id, pdf_bytes in papers:
        analysis_block, deterministic_counts = _prepass(pdf_bytes)
        items[custom_id] = BatchItem(
            custom_id=custom_id,
            figure_is_curve=figure_is_curve,
            analysis_block=analysis_block,
            deterministic_counts=deterministic_counts,
            prompt_version=bundle.version,
            prompt_sha256=bundle.sha256,
            model=model,
            qa_feedback=qa_feedback,
        )
        requests.append(_batch_request(items[custom_id], bundle.text))
        pdfs[custom_id] = pdf_bytes
    if on_prepared is not None:
        on_prepared(items)

    submission = anthropic_client.submit_batch(
        requests, pdfs, on_upload=on_upload, before_create=before_create
    )
    return submission, items


def _batch_request(item: BatchItem, prompt_text: str) -> anthropic_client.BatchRequest:
    return anthropic_client.BatchRequest(
        custom_id=item.custom_id,
        prompt_text=prompt_text,
        model=item.model,
        analysis_block=item.analysis_block or None,
        qa_feedback=item.qa_feedback,
    )


def batch_status(batch_id: str) -> str:
    """'in_progress' | 'canceling' | 'ended'."""
    return anthropic_client.poll_batch_status(batch_id)


def collect_batch(
    batch_id: str,
    items: dict[str, BatchItem],
    file_ids: dict[str, str],
    toolkit_file_id: str | None = None,
    *,
    store: TranscriptStore | None = None,
) -> dict[str, ExtractionResult | Exception]:
    """Once the batch has ended, parse + QA every succeeded result.

    A paper whose batch item errored, or whose output fails to parse, is
    surfaced as an Exception value rather than raised, so one bad paper
    doesn't lose the rest of the batch. A paper whose server-side tool loop
    paused (stop_reason=pause_turn) is transparently finished off with a
    synchronous continuation inside collect_batch_results — see there. That
    requires re-sending the original request, so we reload each item's pinned
    prompt text by version (cheap — a local file read, not an API call) and
    reuse the already-uploaded `file_id` rather than re-uploading the PDF.

    With a `store` (the batch's TranscriptStore) every result is saved to disk
    before it is worked on and a continuation resumes from its saved turns —
    see anthropic_client.collect_batch_results. A failure's ExtractionFailed
    names the transcript in `.transcript`.
    """
    prompt_text_by_version: dict[str, str] = {}
    requests = []
    for item in items.values():
        if item.prompt_version not in prompt_text_by_version:
            prompt_text_by_version[item.prompt_version] = prompt_loader.load_prompt(
                item.prompt_version
            ).text
        requests.append(_batch_request(item, prompt_text_by_version[item.prompt_version]))

    raw_results = anthropic_client.collect_batch_results(
        batch_id, requests, file_ids, toolkit_file_id, store=store
    )
    out: dict[str, ExtractionResult | Exception] = {}
    for custom_id, item in items.items():
        raw = raw_results.get(custom_id)
        transcript = store.relpath(custom_id) if store is not None and store.path(custom_id).exists() else ""
        if raw is None:
            out[custom_id] = RuntimeError("no result returned for this paper")
            continue
        if isinstance(raw, Exception):
            if transcript:
                raw.transcript = transcript
            out[custom_id] = raw
            continue
        try:
            out[custom_id] = _postprocess(
                raw,
                figure_is_curve=item.figure_is_curve,
                deterministic_counts=item.deterministic_counts,
                analysis_block=item.analysis_block,
                prompt_version=item.prompt_version,
                prompt_sha256=item.prompt_sha256,
                model=item.model,
                via_batch=True,
                transcript=transcript,
            )
        except Exception as e:
            # Broad for the same reason as collect_batch_results: _postprocess
            # does parsing, schema coercion and QA, and anything it raises for
            # ONE paper must not discard the papers already parsed above (a
            # re-collect re-bills any synchronous continuations that succeeded).
            failed = anthropic_client.ExtractionFailed(e, raw.usage())
            failed.transcript = transcript
            out[custom_id] = failed
    return out


def cleanup_batch_files(file_ids: dict[str, str], toolkit_file_id: str | None = None) -> None:
    """Delete the Files API uploads made for a batch, once results are collected."""
    anthropic_client.cleanup_batch_files(file_ids, toolkit_file_id)
