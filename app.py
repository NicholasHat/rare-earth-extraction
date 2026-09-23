"""REE Extraction Dashboard — home page (Pillar A).

Batch PDF upload: one or more PDFs in -> one 26-column table per paper out,
each with automatic QA and manual review before merging into the master DB.
Run with:  streamlit run app.py

This page is the only writer of the master DB (plan §6); the Database,
Calculator and Lab Assistant pages under pages/ are read-only consumers.
Everything that must survive a server restart — the review queue and any
in-flight Batch API job — is persisted through extraction/staging.py; this
module is UI only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

import auth
import config
from database import connection, merge, naming, papers_repo
from extraction import runner, staging
from extraction.prompt_loader import PromptNotReadyError
from extraction.runner import ExtractionResult
from extraction.staging import BatchJob, PaperRef, StagedPaper
from ingestion import dedup, doi as doi_mod, pdf_inspect, upload
from validation import schema
from validation.report import ReviewTier, Severity, review_tier

st.set_page_config(page_title="REE Extraction Dashboard", layout="wide")

# Ensure the data dir + schema exist before anything reads the DB.
connection.init_db()


def _pending() -> dict[str, StagedPaper]:
    """The review queue, read from disk on every script run. Staging is the one
    source of truth: a result can be staged by a run whose browser session is
    gone (see _run_batch), so a per-session cache would never show it."""
    return staging.load_all()


def _first_value(df: pd.DataFrame, col: str):
    if col not in df.columns or df.empty:
        return None
    s = df[col].dropna()
    return None if s.empty else str(s.iloc[0])


# --------------------------------------------------------------------------- #
# QA report rendering
# --------------------------------------------------------------------------- #
def _format_usage(result: ExtractionResult) -> str:
    total_in = result.input_tokens + result.cache_creation_input_tokens + result.cache_read_input_tokens
    cached_pct = f"{100 * result.cache_read_input_tokens / total_in:.0f}%" if total_in else "0%"
    return (
        f"{total_in:,} input tokens ({cached_pct} served from cache) · "
        f"{result.output_tokens:,} output tokens"
    )


def _tier_for(staged: StagedPaper) -> ReviewTier:
    """How much attention this staged paper needs (validation.report.review_tier).
    'Anchor' = the deterministic pre-pass verified marker counts for at least
    one figure page; 'raster' from the upload triage or the pre-pass's own
    per-page verdict, whichever noticed."""
    result = staged.result
    is_raster = bool(staged.paper.meta.get("is_raster_figure")) or "(raster image" in result.curve_analysis
    return review_tier(result.qa_report.verdict, bool(result.deterministic_counts), is_raster)


def render_qa(report) -> None:
    verdict = report.verdict
    if verdict is Severity.GREEN:
        st.success("QA passed — no flags.")
        return
    header = "🔴 QA flagged issues (merge gated)" if not report.passed else "🟠 QA warnings"
    (st.error if not report.passed else st.warning)(header)
    for flag in report.reds:
        st.error(f"**[{flag.check}]** {flag.message}")
    for flag in report.ambers:
        st.warning(f"**[{flag.check}]** {flag.message}")


# --------------------------------------------------------------------------- #
# Batch ingestion preview (cheap, no API calls)
# --------------------------------------------------------------------------- #
@dataclass
class _Preview:
    paper: PaperRef
    pdf_bytes: bytes
    status: str          # 'new' | 'existing (paper_id=N)'


def _preview(uploaded_file, conn) -> _Preview:
    pdf_bytes = uploaded_file.getvalue()
    sha, pdf_path = upload.save_pdf(pdf_bytes)
    parsed_doi = doi_mod.parse_doi(pdf_bytes)
    existing = dedup.find_existing(conn, sha, parsed_doi)
    return _Preview(
        paper=PaperRef(
            sha=sha,
            pdf_path=str(pdf_path),
            doi=parsed_doi,
            filename=uploaded_file.name,
            meta=pdf_inspect.inspect(pdf_bytes),
        ),
        pdf_bytes=pdf_bytes,
        status=f"existing (paper_id={existing.paper_id})" if existing else "new",
    )


def _run_batch(selected: list[_Preview], figure_is_curve: bool) -> None:
    errors = []
    for i, p in enumerate(selected, start=1):
        with st.status(f"[{i}/{len(selected)}] Extracting {p.paper.filename}…", expanded=False):
            try:
                result = runner.extract_paper(p.pdf_bytes, figure_is_curve=figure_is_curve)
            except PromptNotReadyError as e:
                errors.append((p.paper.filename, f"Prompt not ready: {e}"))
                continue
            except Exception as e:  # API/parse/QA — money may have been spent: log it, keep going
                staging.record_failed_run(p.paper, e)
                errors.append((p.paper.filename, f"Extraction failed: {e}"))
                continue
            # Persist before touching Streamlit again. Closing the status box
            # (or any st.* call) raises Streamlit's stop signal if the browser
            # session that started this run has since been replaced — a tab
            # reload, a sleep/wake reconnect, a page switch — and a result
            # that only exists in this thread dies with it. On disk, it is
            # picked up by the review queue on the next page load.
            staging.stage(p.paper, figure_is_curve, result)
    n_ok = len(selected) - len(errors)
    if n_ok:
        st.success(f"Extracted {n_ok}/{len(selected)} paper(s) — ready for review below.")
    for filename, msg in errors:
        st.error(f"{filename}: {msg}")


def _submit_batch_job(
    papers: list[tuple[PaperRef, bytes]], figure_is_curve: bool, qa_feedback: str | None = None
) -> None:
    """Submit papers as one Message Batches API job (50% cheaper, asynchronous).
    Persists a sidecar so the job can be checked/collected later, including
    across a server restart. With `qa_feedback` it is a re-extraction: the
    job's result replaces the paper's staged one when it is collected."""
    try:
        submission, items = runner.submit_batch(
            [(paper.sha, pdf_bytes) for paper, pdf_bytes in papers],
            figure_is_curve=figure_is_curve, qa_feedback=qa_feedback,
        )
    except PromptNotReadyError as e:
        st.error(f"Prompt not ready: {e}")
        return
    except Exception as e:
        st.error(f"Batch submission failed: {e}")
        return
    BatchJob(
        batch_id=submission.batch_id,
        file_ids=submission.file_ids,
        toolkit_file_id=submission.toolkit_file_id,
        items=items,
        papers={paper.sha: paper for paper, _ in papers},
    ).save()
    st.success(
        f"Batch submitted: {len(papers)} paper(s), batch_id={submission.batch_id}. Batches usually "
        "finish within an hour (up to 24h) — come back to 'Batch API jobs' and click "
        "'Check status' to retrieve results once it's done."
    )


def _collect_batch_job(job: BatchJob) -> None:
    """Once a batch has ended, parse + QA its results and fold successes into
    the normal staging/review queue — same shape _run_batch produces.

    A paper whose result fails to parse/QA is recorded on the job (which keeps
    only the failed papers) so it stays visible across reruns and server
    restarts, instead of a one-shot st.error that flashes and is gone the
    moment the immediate st.rerun() below fires. The sidecar and its Files
    API uploads are only cleaned up once every item has landed in the review
    queue — a batch with failures is never silently discarded.

    Caller must hold the collection lock (job.begin_collection()) — see
    render_batch_jobs.
    """
    try:
        results = runner.collect_batch(job.batch_id, job.items, job.file_ids, job.toolkit_file_id)
    except Exception as e:
        job.release_lock()
        st.error(f"Could not collect batch results: {e}")
        return

    n_ok = 0
    errors: dict[str, str] = {}
    for sha, result in results.items():
        if isinstance(result, Exception):
            staging.record_failed_run(job.papers[sha], result)
            errors[sha] = str(result)
            continue
        staging.stage(job.papers[sha], job.items[sha].figure_is_curve, result)
        n_ok += 1

    if errors:
        job.record_failures(errors)
    else:
        runner.cleanup_batch_files(job.file_ids, job.toolkit_file_id)
        job.delete()

    if n_ok:
        st.success(
            f"Batch {job.batch_id}: {n_ok}/{len(results)} paper(s) extracted — ready for review below."
        )


def render_failed_runs() -> None:
    """Extractions that failed after spending API money, newest first — the
    only record of what such a run cost (staging.record_failed_run)."""
    failed = staging.load_failed_runs()
    if not failed:
        return
    with st.expander(f"Failed runs ({len(failed)} most recent)"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "When": f["at"],
                        "File": f["filename"],
                        "Error": f["error"],
                        **{k: (f.get("usage") or {}).get(k) for k in (
                            "input_tokens", "output_tokens",
                            "cache_creation_input_tokens", "cache_read_input_tokens",
                        )},
                    }
                    for f in failed
                ]
            ),
            width="stretch", hide_index=True,
        )


def render_batch_jobs() -> None:
    jobs = staging.load_batch_jobs()
    if not jobs:
        return
    st.divider()
    st.subheader(f"Batch API jobs ({len(jobs)} in flight)")
    for batch_id, job in jobs.items():
        with st.container(border=True):
            st.write(f"**{batch_id}** — {len(job.items)} paper(s), submitted {job.submitted_at}")
            for sha, msg in job.errors.items():
                filename = job.papers[sha].filename if sha in job.papers else sha
                st.error(f"{filename}: {msg}")
            in_progress = job.collection_in_progress()
            if in_progress:
                st.info(
                    "A previous check is still finishing up — this can take a while "
                    "if a paper needs a synchronous pause_turn continuation. Please "
                    "wait rather than checking again; re-checking now would restart "
                    "that (expensive) continuation from scratch."
                )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Check status", key=f"batch_status_{batch_id}", disabled=in_progress):
                    try:
                        status = runner.batch_status(batch_id)
                    except Exception as e:
                        st.error(f"Could not check batch status: {e}")
                        continue
                    if status != "ended":
                        st.info(f"Still processing (status: {status}). Check back later.")
                        continue
                    job.begin_collection()
                    with st.spinner(
                        "Collecting batch results — this can take a while if any "
                        "paper needs an extra synchronous digitization round..."
                    ):
                        _collect_batch_job(job)
                    st.rerun()
            with col2:
                if job.errors and st.button("Discard failed paper(s)", key=f"batch_discard_{batch_id}"):
                    runner.cleanup_batch_files(job.file_ids, job.toolkit_file_id)
                    job.delete()
                    st.rerun()


# --------------------------------------------------------------------------- #
# Review + merge queue (one paper at a time, picked from the pending batch)
# --------------------------------------------------------------------------- #
def _stored_tracking(sha: str, doi: str | None) -> dict[str, str]:
    """Tracking-sheet fields already on this paper (re-extraction case), '' if none."""
    conn = connection.get_conn()
    try:
        row = papers_repo.find_by_hash(conn, sha) or papers_repo.find_by_doi(conn, doi)
    finally:
        conn.close()
    return {
        field: (row[field] or "") if row is not None else ""
        for field in papers_repo.TRACKING_FIELDS
    }


def render_tracking_inputs(sha: str, doi: str | None) -> dict[str, str]:
    """Reviewer-entered fields for the tracking sheet; persisted on approval."""
    stored = _stored_tracking(sha, doi)
    with st.expander("Tracking info (short citation also names the export file)", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        tracking = {
            "short_citation": col1.text_input(
                "Short citation", value=stored["short_citation"],
                placeholder="e.g. Swain & Otu", key=f"cit_{sha}",
            ),
            "pub_year": col2.text_input(
                "Year", value=stored["pub_year"], placeholder="e.g. 2011", key=f"year_{sha}",
            ),
            "figures_used": col3.text_input(
                "Figures / tables used", value=stored["figures_used"],
                placeholder="e.g. Fig. 2, Fig. 4", key=f"figs_{sha}",
            ),
            "known_issues": st.text_area(
                "Known issues / caveats", value=stored["known_issues"], key=f"issues_{sha}",
            ),
            "short_description": st.text_area(
                "Short description", value=stored["short_description"], key=f"desc_{sha}",
            ),
        }
    return tracking


def _approve(sha: str, staged: StagedPaper, edited: pd.DataFrame,
             tracking: dict[str, str], note: str, override: bool) -> None:
    paper, result = staged.paper, staged.result
    edited_clean = schema.coerce_schema(edited)
    was_edited = not edited_clean.reset_index(drop=True).equals(result.df.reset_index(drop=True))
    merge_note = note or ("edited in review" if was_edited else None)
    conn = connection.get_conn()
    try:
        summary = merge.commit_extraction(
            conn,
            content_sha256=paper.sha,
            pdf_path=paper.pdf_path,
            df=edited_clean,
            text_endpoints=result.text_endpoints,
            prompt_version=result.prompt_version,
            prompt_sha256=result.prompt_sha256,
            model=result.model,
            qa_passed=result.qa_report.passed,
            qa_report_json=result.qa_report.to_json(),
            raw_response=result.raw_response,
            doi=paper.doi,
            reference_no=_first_value(edited_clean, "Reference No."),
            title=paper.meta.get("title"),
            original_filename=paper.filename,
            figure_type=None,
            is_raster_figure=paper.meta.get("is_raster_figure"),
            tracking=tracking,
            note=merge_note,
            override=override,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_creation_input_tokens=result.cache_creation_input_tokens,
            cache_read_input_tokens=result.cache_read_input_tokens,
        )
    finally:
        conn.close()
    # Export the approved per-paper artifact, named to match the tracking
    # sheet's `Output file` column (e.g. Swain_&_Otu_322.csv).
    export_path = config.EXPORTS_DIR / naming.export_filename(
        tracking["short_citation"], summary["paper_id"], summary["rows_merged"]
    )
    edited_clean.to_csv(export_path, index=False)
    merged_msg = (
        f"Merged {summary['rows_merged']} rows → paper_id={summary['paper_id']}, "
        f"prompt_run_id={summary['prompt_run_id']}. Export: {export_path.name}"
    )
    if summary["superseded_run_id"] is not None:
        # Retiring a previously approved run changes what the calculator and
        # assistant see, so don't let it happen silently.
        merged_msg += (
            f" Superseded the earlier approved run "
            f"(prompt_run_id={summary['superseded_run_id']}) for this prompt version; "
            "its rows stay in the DB but are no longer current."
        )
    st.success(merged_msg)
    staging.discard(sha)


def render_review_queue() -> None:
    pending = _pending()
    if not pending:
        return

    st.divider()
    st.subheader(f"Review queue ({len(pending)} pending)")
    sha = st.selectbox(
        "Paper to review",
        list(pending),
        format_func=lambda s: (
            f"{_tier_for(pending[s]).icon} {pending[s].paper.filename} "
            f"({len(pending[s].result.df)} rows · {_tier_for(pending[s]).label})"
        ),
    )
    staged = pending[sha]
    paper, result = staged.paper, staged.result

    tier = _tier_for(staged)
    {"Fast track": st.success, "Standard": st.info, "Full review": st.error}[tier.label](
        f"{tier.icon} **{tier.label}** — {tier.reason}"
    )
    render_qa(result.qa_report)
    st.caption(_format_usage(result))

    # One click removes every row a QA check named as individually wrong
    # (QAReport.flagged_rows — off-curve points, same-curve repeats). Series-
    # level flags never name rows, so they can't be dropped this way; that is
    # deliberate. The drop lives in this session only and is recorded on
    # approval as an edit, with the rows in the review note. It resets the
    # editor (new key), so drop first, then edit cells.
    drop_key = f"dropped_{sha}"
    dropped: list[int] = st.session_state.get(drop_key, [])
    droppable = [r for r in result.qa_report.flagged_rows if r not in dropped]
    col_drop, col_qa = st.columns(2)
    if droppable and col_drop.button(
        f"🧹 Drop the {len(droppable)} QA-flagged row(s)", key=f"drop_{sha}",
        help="Rows named by the off-curve and duplicate checks: " + ", ".join(map(str, droppable)),
    ):
        st.session_state[drop_key] = sorted(dropped + droppable)
        st.rerun()
    # QA is pure and cheap; a paper staged before a check existed can pick up
    # today's checks without another extraction. Replaces the staged report.
    if col_qa.button("♻️ Re-run QA checks", key=f"requa_{sha}",
                     help="Re-apply the current validation checks to this staged table (no API call)."):
        staging.stage(paper, staged.figure_is_curve, runner.rerun_qa(result, figure_is_curve=staged.figure_is_curve))
        st.session_state.pop(drop_key, None)
        st.rerun()
    if dropped:
        st.caption(f"Dropped {len(dropped)} QA-flagged row(s) in this session: {', '.join(map(str, dropped))}.")
    table = result.df.drop(index=[r - 1 for r in dropped if r - 1 in result.df.index])

    st.write(f"**{len(table)} rows.** Edit cells below if needed.")
    edited = st.data_editor(table, num_rows="dynamic", width="stretch", key=f"editor_{sha}_{len(dropped)}")

    if result.text_endpoints:
        with st.expander(f"Captured text endpoints ({len(result.text_endpoints)})"):
            st.dataframe(pd.DataFrame(result.text_endpoints), width="stretch")

    tracking = render_tracking_inputs(sha, paper.doi)
    note = st.text_input("Review note (optional)", key=f"note_{sha}")
    override = False
    if not result.qa_report.passed:
        override = st.checkbox(
            "Override red QA flags and merge anyway (records the override in the audit log)",
            key=f"override_{sha}",
        )

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("✅ Approve & merge", type="primary", key=f"approve_{sha}"):
        if not auth.require_write_access():
            st.stop()
        if not result.qa_report.passed and not override:
            st.error("Red QA flags present — tick the override box to merge anyway.")
            st.stop()
        if dropped:
            note = f"dropped {len(dropped)} QA-flagged row(s): {', '.join(map(str, dropped))}" + (f"; {note}" if note else "")
        _approve(sha, staged, edited, tracking, note, override)
        st.rerun()

    if col_b.button("🗑️ Reject", key=f"reject_{sha}"):
        staging.discard(sha)
        st.info(f"{paper.filename} rejected and discarded (nothing written to the master DB).")
        st.rerun()

    # The middle option between "approve anyway" and "throw it away": re-run
    # the extraction with the QA findings injected as feedback, on demand only
    # (one extra extraction; never automatic — a false-positive flag shouldn't
    # silently cost money). Replaces this paper's staged result. A raster
    # paper takes the Batch API route here exactly as on first extraction
    # (runner.must_use_batch); its result lands when the job is collected, so
    # a second click meanwhile is refused rather than paid for twice.
    batch_pending = any(sha in job.items for job in staging.load_batch_jobs().values())
    if batch_pending:
        col_c.caption("A batched re-extraction of this paper is in flight — see 'Batch API jobs'.")
    if result.qa_report.flags and col_c.button(
        "🔁 Re-extract with QA feedback", key=f"reextract_{sha}", disabled=batch_pending
    ):
        if not auth.require_write_access():
            st.stop()
        feedback = runner.qa_feedback_block(result.qa_report, len(result.df))
        if runner.must_use_batch(paper.meta):
            _submit_batch_job([(paper, Path(paper.pdf_path).read_bytes())],
                              staged.figure_is_curve, qa_feedback=feedback)
            st.stop()  # keep the submission message on screen; the job shows on the next run
        with st.spinner("Re-extracting with the QA findings as feedback — this runs one full synchronous extraction..."):
            try:
                new_result = runner.extract_paper(
                    Path(paper.pdf_path).read_bytes(),
                    figure_is_curve=staged.figure_is_curve,
                    qa_feedback=feedback,
                )
            except Exception as e:
                staging.record_failed_run(paper, e)
                st.error(f"Re-extraction failed (previous staged result kept): {e}")
                st.stop()
            staging.stage(paper, staged.figure_is_curve, new_result)  # before the spinner closes — see _run_batch
        st.rerun()


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("REE Extraction Dashboard")
    st.caption("Batch PDF upload → 26-column tables → QA → review → merge")

    uploaded_files = st.file_uploader(
        "Upload one or more research-paper PDFs", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Upload one or more PDFs to begin.")
        render_batch_jobs()
        render_review_queue()
        render_failed_runs()
        return

    conn = connection.get_conn()
    try:
        previews = [_preview(f, conn) for f in uploaded_files]
    finally:
        conn.close()

    st.subheader(f"Batch ({len(previews)} file(s))")
    table = pd.DataFrame(
        [
            {
                "Include": True,
                "File": p.paper.filename,
                "DOI": p.paper.doi or "—",
                "Pages": p.paper.meta["n_pages"],
                "Raster?": "yes" if p.paper.meta["is_raster_figure"] else "no/unknown",
                "Status": p.status,
            }
            for p in previews
        ]
    )
    edited_table = st.data_editor(
        table,
        width="stretch",
        hide_index=True,
        disabled=["File", "DOI", "Pages", "Raster?", "Status"],
        key="batch_table",
    )

    figure_is_curve = st.checkbox(
        "Primary figure(s) are multi-point curves (enables sparse-result QA check)",
        value=True,
    )
    use_batch_api = st.checkbox(
        "Use the Batch API for every paper (50% cheaper token pricing; asynchronous — "
        "usually done within an hour, up to 24h). Raster papers take it regardless.",
        value=False,
        help=(
            "Verified live once (2026-07-29, ~$2.26/paper, accuracy matching the "
            "synchronous baseline). Still worth a 1-2 paper trial before a full "
            "run after any change to the request shape. Raster-figure papers are "
            "never run synchronously: their much longer digitization loop costs "
            "several times more that way and can exhaust the sync path's "
            "continuation cap with nothing to show for the spend."
        ),
    )

    selected = [p for p, inc in zip(previews, edited_table["Include"]) if inc]
    # A second click on the run/submit button before the first has finished
    # (or a re-upload of a paper still in the queue) must not pay for the
    # same extraction twice: skip anything already in an unfinished batch job
    # or already staged for review. Both live on disk, so this holds across
    # sessions and restarts.
    in_flight = {sha for job in staging.load_batch_jobs().values() for sha in job.items}
    awaiting_review = set(staging.load_all())
    busy = [p for p in selected if p.paper.sha in in_flight | awaiting_review]
    if busy:
        st.warning(
            "Skipping "
            + ", ".join(
                f"**{p.paper.filename}** "
                f"({'in a batch job still running' if p.paper.sha in in_flight else 'awaiting review below'})"
                for p in busy
            )
            + " — approve or reject it first to extract it again."
        )
        busy_shas = {p.paper.sha for p in busy}
        selected = [p for p in selected if p.paper.sha not in busy_shas]
    dup_selected = [p for p in selected if p.status != "new"]
    if dup_selected:
        st.warning(
            f"{len(dup_selected)} selected file(s) already exist in the DB — "
            "extracting will record a **new run** against the existing paper "
            "(coexistence), not a duplicate paper."
        )

    # Route: the toggle sends everything to the Batch API; without it, raster
    # papers still go that way (runner.must_use_batch) and the rest run now.
    batched, synchronous = [], []
    for p in selected:
        (batched if use_batch_api or runner.must_use_batch(p.paper.meta) else synchronous).append(p)
    if batched and not use_batch_api:
        st.info(
            "Raster paper(s) routed to the Batch API: "
            + ", ".join(f"**{p.paper.filename}**" for p in batched)
            + " — check 'Batch API jobs' for the result."
        )
    button_label = " + ".join(
        label for label, n in (
            (f"Run extraction on {len(synchronous)} paper(s)", len(synchronous)),
            (f"Submit {len(batched)} paper(s) as a batch", len(batched)),
        ) if n
    ) or "Run extraction on 0 paper(s)"
    if st.button(button_label, type="primary", disabled=not selected):
        if not auth.require_write_access():
            st.stop()
        if batched:
            _submit_batch_job([(p.paper, p.pdf_bytes) for p in batched], figure_is_curve)
        if synchronous:
            _run_batch(synchronous, figure_is_curve)

    render_batch_jobs()
    render_review_queue()
    render_failed_runs()


# Streamlit executes this module top-to-bottom on every interaction.
main()
