"""REE Extraction Dashboard — home page (Pillar A).

Batch API jobs (from `python -m extraction.bulk` or submitted here) and one-off
PDF uploads -> one extraction table per paper, each with automatic QA and
manual review before merging into the master DB. Only openly-licensed papers
can be merged (database/merge.py). Run with:  streamlit run app.py

This page is the only writer of the master DB (plan §6); the Database,
Calculator and Lab Assistant pages under pages/ are read-only consumers.
Everything that must survive a server restart — the review queue and any
in-flight Batch API job — is persisted through extraction/staging.py; this
module is UI only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

import auth
import config
from database import connection, merge, naming, papers_repo
from extraction import anthropic_client, batch_jobs, provenance, runner, staging
from extraction.prompt_loader import PromptNotReadyError
from extraction.runner import ExtractionResult
from extraction.staging import PaperRef, StagedPaper
from evaluation.cost import result_usd
from ingestion import dedup, doi as doi_mod, licensing, pdf_inspect, upload
from ingestion.licensing import License
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
    text = (
        f"{total_in:,} input tokens ({cached_pct} served from cache) · "
        f"{result.output_tokens:,} output tokens"
    )
    cost = result_usd(result)
    if cost is not None:
        text += f" · ≈ ${cost:.2f}" + (" (Batch API)" if result.via_batch else "")
    if result.transcript:
        text += f" · transcript: {result.transcript}"
    return text


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
            license=licensing.from_manifest(sha),
        ),
        pdf_bytes=pdf_bytes,
        status=f"existing (paper_id={existing.paper_id})" if existing else "new",
    )


def _run_batch(selected: list[_Preview], figure_is_curve: bool) -> None:
    errors = []
    for i, p in enumerate(selected, start=1):
        with st.status(f"[{i}/{len(selected)}] Extracting {p.paper.filename}…", expanded=False):
            try:
                with staging.claim(p.paper.sha):   # a double click can't pay for it twice
                    result = runner.extract_paper(p.pdf_bytes, figure_is_curve=figure_is_curve)
                    # Persist before touching Streamlit again (see below), and
                    # before the claim is released.
                    staging.stage(p.paper, figure_is_curve, result)
            except staging.Busy as e:
                errors.append((p.paper.filename, str(e)))
                continue
            except PromptNotReadyError as e:
                errors.append((p.paper.filename, f"Prompt not ready: {e}"))
                continue
            except Exception as e:  # API/parse/QA — money may have been spent: log it, keep going
                staging.record_failed_run(p.paper, e)
                errors.append((p.paper.filename, f"Extraction failed: {e}"))
                continue
            # The result was staged above, before touching Streamlit again.
            # Closing the status box (or any st.* call) raises Streamlit's stop
            # signal if the browser session that started this run has since
            # been replaced — a tab reload, a sleep/wake reconnect, a page
            # switch — and a result that only exists in this thread dies with
            # it. On disk, it is picked up by the review queue on the next load.
    n_ok = len(selected) - len(errors)
    if n_ok:
        st.success(f"Extracted {n_ok}/{len(selected)} paper(s) — ready for review below.")
    for filename, msg in errors:
        st.error(f"{filename}: {msg}")


def _submit_batch_job(
    papers: list[tuple[PaperRef, bytes]], figure_is_curve: bool, qa_feedback: str | None = None
) -> None:
    """Submit papers as one Message Batches API job (50% cheaper, asynchronous)
    through extraction/batch_jobs.py — the same code path as the headless
    `python -m extraction.bulk`. With `qa_feedback` it is a re-extraction: the
    job's result replaces the paper's staged one when it is collected."""
    try:
        outcome = batch_jobs.submit(
            papers, figure_is_curve=figure_is_curve, qa_feedback=qa_feedback,
            replacing_staged=qa_feedback is not None,
        )
    except PromptNotReadyError as e:
        st.error(f"Prompt not ready: {e}")
        return
    except anthropic_client.CreditExhausted as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Batch submission failed: {e}")
        return
    for sha, reason in outcome.skipped.items():
        st.warning(f"Not submitted: {sha[:12]}… — {reason}.")
    if outcome.job is None:
        return
    st.success(
        f"Batch submitted: {len(outcome.job.items)} paper(s), batch_id={outcome.job.batch_id}. Batches usually "
        "finish within an hour (up to 24h) — come back to 'Batch API jobs' and click "
        "'Check status' to retrieve results once it's done."
    )


def _collect_batch_job(batch_id: str) -> None:
    """Once a batch has ended, parse + QA its results and fold successes into
    the normal staging/review queue (extraction/batch_jobs.collect). A paper
    that fails stays on the job, visible across reruns and restarts; the job
    and its uploads are cleaned up once every paper has landed.

    The outcome is kept in session state and shown by render_batch_jobs on the
    next run: the caller reruns straight after, which would wipe any message."""
    try:
        outcome = batch_jobs.collect(batch_id)
    except Exception as e:
        st.session_state["batch_notice"] = ("error", f"Could not collect batch results: {e}", None)
        return
    if outcome.status == "busy":
        notice = ("info", "This batch is being collected by another session or the bulk runner — wait for it.", None)
    elif outcome.status == "not_ended":
        notice = ("info", f"Batch {batch_id}: still processing (status: {outcome.processing_status}). "
                          "Check back later.", None)
    elif outcome.status == "gone":
        notice = ("info", f"Batch {batch_id} was already collected or discarded.", None)
    else:
        n_ok = sum(1 for p in outcome.papers if p.staged)
        rows = pd.DataFrame([
            {"File": p.filename, "Staged": p.staged, "Rows": p.rows, "Red": p.reds, "Amber": p.ambers,
             "Cost (USD)": p.cost_usd, "Error": p.error or "", "Transcript": p.transcript or ""}
            for p in outcome.papers
        ])
        notice = ("success" if n_ok == len(outcome.papers) else "warning",
                  f"Batch {batch_id}: {n_ok}/{len(outcome.papers)} paper(s) staged for review.", rows)
    if outcome.credit_exhausted:
        notice = ("error", notice[1] + " The Anthropic credit balance ran out. Every completed turn is "
                  "kept — top up and check status again to resume.", notice[2])
    st.session_state["batch_notice"] = notice


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
                        "Credit exhausted": bool(f.get("credit_exhausted")),
                        "Transcript": f.get("transcript") or "",
                    }
                    for f in failed
                ]
            ),
            width="stretch", hide_index=True,
        )


def _render_batch_notice() -> None:
    """The last collect/reconcile outcome of this session, shown once."""
    notice = st.session_state.pop("batch_notice", None)
    if notice is None:
        return
    kind, message, rows = notice
    getattr(st, kind)(message)
    if rows is not None and not rows.empty:
        st.dataframe(rows, width="stretch", hide_index=True)


def _reconcile() -> None:
    report = batch_jobs.reconcile()
    lines = [f"Adopted batch {j.batch_id}." for j in report.adopted]
    lines += [f"Submission {j.job_id[:8]} never created a batch — cleaned up." for j in report.abandoned]
    lines += [f"Submission {j.job_id[:8]}: {why}." for j, why in report.unresolved]
    lines += [f"Untracked batch on the account: {b.batch_id}." for b in report.untracked]
    if report.api_error:
        lines.append(f"Could not list batches: {report.api_error}")
    st.session_state["batch_notice"] = ("info", "  \n".join(lines) or "Nothing to reconcile.", None)


def render_batch_jobs() -> None:
    jobs = staging.load_batch_jobs()
    pending = staging.load_pending_submissions()
    if not jobs and not pending and "batch_notice" not in st.session_state:
        return
    st.divider()
    st.subheader(f"Batch API jobs ({len(jobs)} in flight)")
    _render_batch_notice()
    if pending:
        st.warning(
            f"{len(pending)} batch submission(s) are in progress or were interrupted before the API "
            "answered. Reconciling checks the API's batch list and adopts or cleans up each one."
        )
        if st.button("Reconcile with the API", key="batch_reconcile"):
            if not auth.require_write_access():
                st.stop()
            _reconcile()
            st.rerun()
    now = datetime.now(timezone.utc)
    for batch_id, job in jobs.items():
        with st.container(border=True):
            st.write(f"**{batch_id}** — {len(job.items)} paper(s), submitted {job.submitted_at[:16].replace('T', ' ')}")
            expires = job.results_expire_at()
            if expires - now < batch_jobs.EXPIRY_WARNING:
                st.warning(f"The API deletes this batch's results on {expires:%Y-%m-%d} — collect them before then.")
            for sha, msg in job.errors.items():
                filename = job.papers[sha].filename if sha in job.papers else sha
                st.error(f"{filename}: {msg}")
            in_progress = job.collection_in_progress()
            if in_progress:
                st.info(
                    "Being collected right now (here, in another session, or by the bulk runner) — "
                    "this can take a while if a paper needs a synchronous continuation."
                )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Check status", key=f"batch_status_{batch_id}", disabled=in_progress):
                    if not auth.require_write_access():
                        st.stop()
                    with st.spinner(
                        "Collecting batch results — this can take a while if any "
                        "paper needs an extra synchronous digitization round..."
                    ):
                        _collect_batch_job(batch_id)
                    st.rerun()
            with col2:
                if job.errors and st.button("Discard failed paper(s)", key=f"batch_discard_{batch_id}"):
                    if not auth.require_write_access():
                        st.stop()
                    try:
                        batch_jobs.discard(batch_id)
                    except staging.Busy as e:
                        st.session_state["batch_notice"] = ("error", str(e), None)
                    st.rerun()


# --------------------------------------------------------------------------- #
# Review + merge queue (one paper at a time, picked from the list of staged papers)
# --------------------------------------------------------------------------- #
def _db_paper(sha: str, doi: str | None):
    """The paper's row in the master DB (re-extraction case), or None."""
    conn = connection.get_conn()
    try:
        return papers_repo.find_by_hash(conn, sha) or papers_repo.find_by_doi(conn, doi)
    finally:
        conn.close()


def _license_for(paper: PaperRef) -> License | None:
    """The licence the merge gate will judge this paper by: the staged one,
    else the open-access manifest's, else the one on its DB row (mirrors
    merge.commit_extraction)."""
    return licensing.resolve(paper.sha, paper.license) or papers_repo.get_license(_db_paper(paper.sha, paper.doi))


def render_license(sha: str, paper: PaperRef, lic: License | None) -> None:
    """The paper's licence, and — when it has no open one — the form to record it."""
    if lic is not None and lic.is_open:
        st.caption(f"Licence: {lic.id} ({lic.origin})" + (f" — {lic.evidence}" if lic.evidence else ""))
        return
    if lic is None:
        st.warning("No licence recorded — approval is blocked until an open licence is recorded.")
    else:
        st.warning(f"Licence '{lic.id}' is not an open licence — this paper can't be merged.")
    with st.expander("Record a licence"):
        col1, col2 = st.columns([1, 3])
        lic_id = col1.selectbox("Licence", sorted(config.OPEN_LICENSES), key=f"lic_id_{sha}")
        evidence = col2.text_input("Where the paper states it", key=f"lic_ev_{sha}",
                                   placeholder="e.g. the licence line on the article's first page")
        url = st.text_input("Source URL (optional)", key=f"lic_url_{sha}")
        if st.button("Save licence", key=f"lic_save_{sha}", disabled=not evidence.strip()):
            if not auth.require_write_access():
                st.stop()
            staging.set_license(sha, License.of(lic_id, evidence.strip(), url.strip()))
            st.rerun()


def _stored_tracking(sha: str, doi: str | None) -> dict[str, str]:
    """Tracking-sheet fields already on this paper (re-extraction case), '' if none."""
    row = _db_paper(sha, doi)
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


def _edited_qa(staged: StagedPaper, edited_clean: pd.DataFrame):
    """QA of the rows that would merge — the edited ones, not the extraction's,
    so an edit (a typed 950 for 95) can't slip past the checks."""
    return runner.rerun_qa(replace(staged.result, df=edited_clean),
                           figure_is_curve=staged.figure_is_curve).qa_report


def _approve(sha: str, staged: StagedPaper, edited: pd.DataFrame,
             tracking: dict[str, str], note: str, override: bool) -> None:
    paper, result = staged.paper, staged.result
    edited_clean = schema.coerce_schema(edited)
    was_edited = not edited_clean.reset_index(drop=True).equals(result.df.reset_index(drop=True))
    merge_note = note or ("edited in review" if was_edited else None)
    qa = _edited_qa(staged, edited_clean)
    if not qa.passed and not override:
        st.error("The table as edited has red QA flags — fix them, or tick the override box: "
                 + "; ".join(f"[{f.check}] {f.message}" for f in qa.reds))
        st.stop()
    conn = connection.get_conn()
    try:
        summary = merge.commit_extraction(
            conn,
            license=licensing.resolve(paper.sha, paper.license),
            content_sha256=paper.sha,
            pdf_path=paper.pdf_path,
            df=edited_clean,
            text_endpoints=result.text_endpoints,
            prompt_version=result.prompt_version,
            prompt_sha256=result.prompt_sha256,
            model=result.model,
            qa_passed=qa.passed,
            qa_report_json=qa.to_json(),
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
    except merge.LicenseRequired as e:
        st.error(str(e))
        st.stop()
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


@dataclass
class _Editor:
    """One paper's table editor across reruns. Streamlit forgets a widget's
    state once a run doesn't draw it, and only the paper under review is drawn,
    so the edited table is kept here and the editor is re-created from it
    (under a new key) when the reviewer comes back to the paper."""
    sha: str
    base: pd.DataFrame          # the staged table minus dropped rows
    seed: pd.DataFrame          # what the live editor was created from
    draft: pd.DataFrame | None = None   # the table as last edited, if edited
    gen: int = 0

    @property
    def key(self) -> str:
        return f"editor_{self.sha}_{self.gen}"

    def changed(self) -> bool:
        delta = st.session_state.get(self.key) or {}
        return self.draft is not None or any(delta.get(k) for k in ("edited_rows", "added_rows", "deleted_rows"))


def _editors() -> dict[str, _Editor]:
    return st.session_state.setdefault("review_editors", {})


def _table_editor(sha: str, table: pd.DataFrame) -> pd.DataFrame:
    ed = _editors().get(sha)
    if ed is None or not ed.base.equals(table):
        # First look, or the table under it changed (rows dropped, QA re-run
        # undid a drop, a re-extraction landed): start from the table as is.
        ed = _editors()[sha] = _Editor(sha, table, table, gen=ed.gen + 1 if ed else 0)
    elif ed.key not in st.session_state and ed.draft is not None:
        ed.seed, ed.gen = ed.draft, ed.gen + 1
    edited = st.data_editor(ed.seed, num_rows="dynamic", width="stretch", key=ed.key)
    ed.draft = edited if ed.changed() else None
    return edited


# Per-paper inputs of the review form (tracking fields, note, override, the
# licence form), by key prefix; each key ends in the paper's sha.
_PAPER_INPUTS = ("lic_id_", "lic_ev_", "lic_url_", "cit_", "year_", "figs_", "issues_", "desc_",
                 "note_", "override_")


def _keep_inputs(pending: dict[str, StagedPaper], selected: str) -> None:
    """Keep what was typed for the papers not on screen: re-assigning a value
    through Session State stops Streamlit from dropping it with the widget."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(_PAPER_INPUTS):
            sha = k.rsplit("_", 1)[1]
            if sha != selected and sha in pending:
                st.session_state[k] = st.session_state[k]


def _select(order: list[str]) -> str:
    """The paper under review. It stays selected across reruns; a click on the
    list picks another; once it leaves the queue (approved, rejected) the one
    after it in the list takes its place, else the one before."""
    ss = st.session_state
    last_order = ss.get("review_order", [])
    sha = ss.get("review_sha")
    rows = ((ss.get("review_list") or {}).get("selection") or {}).get("rows") or []
    if rows and rows[0] != ss.get("review_row") and rows[0] < len(last_order):
        sha = last_order[rows[0]]     # clicked in the list
    if sha not in order:
        if sha in last_order:
            i = last_order.index(sha)
            sha = next((s for s in last_order[i + 1:] + last_order[:i][::-1] if s in order), order[0])
        else:
            sha = order[0]
    ss["review_sha"], ss["review_order"], ss["review_row"] = sha, order, order.index(sha)
    ss["review_list"] = {"selection": {"rows": [order.index(sha)], "columns": []}}
    return sha


def _step(by: int) -> None:
    order = st.session_state["review_order"]
    st.session_state["review_sha"] = order[order.index(st.session_state["review_sha"]) + by]


def render_review_list(pending: dict[str, StagedPaper], licences: dict[str, License | None],
                       sha: str) -> None:
    """Every staged paper, one row each; the selected row is the paper below."""
    def licence(lic: License | None) -> str:
        return "none" if lic is None else lic.id if lic.is_open else f"{lic.id} (not open)"

    editors = _editors()
    st.dataframe(
        pd.DataFrame([
            {
                "Paper": s.paper.filename,
                "Rows": len(s.result.df),
                "Red": len(s.result.qa_report.reds),
                "Amber": len(s.result.qa_report.ambers),
                "Review": _tier_for(s).label,
                "Licence": licence(licences[k]),
                "Cost (USD)": result_usd(s.result),
                "Edited": "yes" if (k in editors and editors[k].changed())
                          or st.session_state.get(f"dropped_{k}") else "",
            }
            for k, s in pending.items()
        ]),
        key="review_list", on_select="rerun", selection_mode="single-row-required",
        hide_index=True, width="stretch",
        column_config={"Cost (USD)": st.column_config.NumberColumn(format="%.2f")},
    )
    order = list(pending)
    i = order.index(sha)
    col_prev, col_next, col_where = st.columns([1, 1, 6])
    col_prev.button("Previous", key="review_prev", on_click=_step, args=(-1,), disabled=i == 0)
    col_next.button("Next", key="review_next", on_click=_step, args=(1,), disabled=i == len(order) - 1)
    col_where.write(f"**{i + 1} of {len(order)}:** {pending[sha].paper.filename}")


def render_review_queue() -> None:
    pending = _pending()
    if not pending:
        return

    st.divider()
    st.subheader(f"Review queue ({len(pending)} pending)")
    licences = {s: _license_for(p.paper) for s, p in pending.items()}
    sha = _select(list(pending))
    _keep_inputs(pending, sha)
    for gone in set(_editors()) - set(pending):
        del _editors()[gone]
    render_review_list(pending, licences, sha)
    staged = pending[sha]
    paper, result = staged.paper, staged.result
    lic = licences[sha]
    mergeable = lic is not None and lic.is_open

    tier = _tier_for(staged)
    {"Fast track": st.success, "Standard": st.info, "Full review": st.error}[tier.label](
        f"{tier.icon} **{tier.label}** — {tier.reason}"
    )
    render_license(sha, paper, lic)
    render_qa(result.qa_report)
    st.caption(_format_usage(result))

    # One click removes every row a QA check named as individually wrong
    # (QAReport.flagged_rows — off-curve points, same-curve repeats). Series-
    # level flags never name rows, so they can't be dropped this way; that is
    # deliberate. The drop lives in this session only and is recorded on
    # approval as an edit, with the rows in the review note. It resets the
    # editor (_table_editor), so drop first, then edit cells.
    drop_key = f"dropped_{sha}"
    dropped: list[int] = st.session_state.get(drop_key, [])
    droppable = [r for r in result.qa_report.flagged_rows if r not in dropped]
    col_drop, col_qa = st.columns(2)
    if droppable and col_drop.button(
        f"Drop the {len(droppable)} QA-flagged row(s)", key=f"drop_{sha}",
        help="Rows named by the off-curve and duplicate checks: " + ", ".join(map(str, droppable)),
    ):
        st.session_state[drop_key] = sorted(dropped + droppable)
        st.rerun()
    # QA is pure and cheap; a paper staged before a check existed can pick up
    # today's checks without another extraction. Replaces the staged report.
    if col_qa.button("Re-run QA checks", key=f"requa_{sha}",
                     help="Re-apply the current validation checks to this staged table (no API call)."):
        if not auth.require_write_access():
            st.stop()
        staging.stage(paper, staged.figure_is_curve, runner.rerun_qa(result, figure_is_curve=staged.figure_is_curve))
        st.session_state.pop(drop_key, None)
        st.rerun()
    if dropped:
        st.caption(f"Dropped {len(dropped)} QA-flagged row(s) in this session: {', '.join(map(str, dropped))}.")
    table = result.df.drop(index=[r - 1 for r in dropped if r - 1 in result.df.index])

    st.write(f"**{len(table)} rows.** Edit cells below if needed.")
    edited = _table_editor(sha, table)
    # What approval checks: the table as it stands (edits, dropped rows, and
    # today's checks), which can differ from the staged report above.
    qa = _edited_qa(staged, schema.coerce_schema(edited))
    if qa.to_json() != result.qa_report.to_json():
        st.caption("QA of the table below with today's checks — approval goes by this:")
        render_qa(qa)

    if result.text_endpoints:
        with st.expander(f"Captured text endpoints ({len(result.text_endpoints)})"):
            st.dataframe(pd.DataFrame(result.text_endpoints), width="stretch")
    if result.figures or result.row_figures:
        figs = provenance.summarise(result.figures, result.row_figures)
        skipped = sum(1 for f in figs if not f["digitised"])
        with st.expander(f"Figures ({len(figs) - skipped} digitised, {skipped} skipped — as extracted)"):
            st.dataframe(pd.DataFrame(figs), width="stretch", hide_index=True)

    tracking = render_tracking_inputs(sha, paper.doi)
    note = st.text_input("Review note (optional)", key=f"note_{sha}")
    override = False
    if not qa.passed:
        override = st.checkbox(
            "Override red QA flags and merge anyway (records the override in the audit log)",
            key=f"override_{sha}",
        )

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Approve & merge", type="primary", key=f"approve_{sha}", disabled=not mergeable):
        if not auth.require_write_access():
            st.stop()
        if dropped:
            note = f"dropped {len(dropped)} QA-flagged row(s): {', '.join(map(str, dropped))}" + (f"; {note}" if note else "")
        _approve(sha, staged, edited, tracking, note, override)
        st.rerun()

    if col_b.button("Reject", key=f"reject_{sha}"):
        if not auth.require_write_access():
            st.stop()
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
    batch_pending = sha in staging.in_flight_shas()
    if batch_pending:
        col_c.caption("This paper is already in a batch job or being extracted — see 'Batch API jobs'.")
    if result.qa_report.flags and col_c.button(
        "Re-extract with QA feedback", key=f"reextract_{sha}", disabled=batch_pending
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
                with staging.claim(sha):   # a double click can't pay for it twice
                    new_result = runner.extract_paper(
                        Path(paper.pdf_path).read_bytes(),
                        figure_is_curve=staged.figure_is_curve,
                        qa_feedback=feedback,
                    )
                    staging.stage(paper, staged.figure_is_curve, new_result)   # before the spinner closes — see _run_batch
            except staging.Busy as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                staging.record_failed_run(paper, e)
                st.error(f"Re-extraction failed (previous staged result kept): {e}")
                st.stop()
        st.rerun()


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("REE Extraction Dashboard")
    st.caption(
        f"Batch jobs and uploads → {len(schema.COLUMNS)}-column tables → QA → review → merge. "
        "Bulk runs over the open-access manifest: `python -m extraction.bulk`."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more research-paper PDFs", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
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
                "Licence": getattr(_license_for(p.paper), "id", None) or "none recorded",
                "Status": p.status,
            }
            for p in previews
        ]
    )
    edited_table = st.data_editor(
        table,
        width="stretch",
        hide_index=True,
        disabled=["File", "DOI", "Pages", "Raster?", "Licence", "Status"],
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
    in_flight = staging.in_flight_shas()
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
