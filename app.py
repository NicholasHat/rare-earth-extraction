"""REE Extraction Dashboard — Phase A2.

Batch PDF upload: one or more PDFs in -> one 26-column table per paper out,
each with automatic QA and manual review before merging into the master DB.
Run with:  streamlit run app.py

This is the Pillar A spine (README §6). Pillars B (calculator) and C (assistant)
come later; the schema and master DB they read from are built here.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import auth
import config
from database import connection, merge
from extraction import runner
from extraction.runner import ExtractionResult
from extraction.parse_output import ParseError
from extraction.prompt_loader import PromptNotReadyError
from ingestion import dedup, doi as doi_mod, pdf_inspect, upload
from validation import schema
from validation.report import QAReport, Severity

st.set_page_config(page_title="REE Extraction Dashboard", layout="wide")

# Ensure the data dir + schema exist before anything reads the DB.
connection.init_db()


# --------------------------------------------------------------------------- #
# Staging persistence (survive server restarts / idle timeouts)
# --------------------------------------------------------------------------- #

def _meta_path(sha: str):
    return config.STAGING_DIR / f"{sha}.meta.json"


def _staging_path(sha: str):
    return config.STAGING_DIR / f"{sha}.xlsx"


def _save_staging_meta(sha: str, stash: dict, result: ExtractionResult) -> None:
    """Write a JSON sidecar so the review queue can be restored after a restart."""
    payload = {
        "sha": sha,
        "pdf_path": stash["pdf_path"],
        "doi": stash["doi"],
        "filename": stash["filename"],
        "meta": stash["meta"],
        "prompt_version": result.prompt_version,
        "prompt_sha256": result.prompt_sha256,
        "model": result.model,
        "raw_response": result.raw_response,
        "coercion_failures": result.coercion_failures,
        "curve_analysis": result.curve_analysis,
        "deterministic_counts": result.deterministic_counts,
        "qa_report_json": result.qa_report.to_json(),
        "text_endpoints": result.text_endpoints,
    }
    _meta_path(sha).write_text(json.dumps(payload), encoding="utf-8")


def _delete_staging_files(sha: str) -> None:
    for p in (_meta_path(sha), _staging_path(sha)):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _restore_staging_queue() -> None:
    """On first load, reload any unreviewed staging items from disk into session state."""
    if st.session_state.get("_staging_restored"):
        return
    st.session_state["_staging_restored"] = True
    pending = st.session_state.setdefault("pending", {})
    for meta_file in config.STAGING_DIR.glob("*.meta.json"):
        sha = meta_file.name.removesuffix(".meta.json")
        if sha in pending:
            continue
        xlsx = _staging_path(sha)
        if not xlsx.exists():
            meta_file.unlink(missing_ok=True)
            continue
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
            df = pd.read_excel(xlsx, engine="openpyxl")
            result = ExtractionResult(
                df=df,
                text_endpoints=payload["text_endpoints"],
                qa_report=QAReport.from_json(payload["qa_report_json"]),
                prompt_version=payload["prompt_version"],
                prompt_sha256=payload["prompt_sha256"],
                model=payload["model"],
                raw_response=payload.get("raw_response"),
                coercion_failures=payload.get("coercion_failures", 0),
                curve_analysis=payload.get("curve_analysis", ""),
                deterministic_counts=payload.get("deterministic_counts"),
            )
            pending[sha] = {
                "sha": sha,
                "pdf_path": payload["pdf_path"],
                "doi": payload["doi"],
                "filename": payload["filename"],
                "meta": payload["meta"],
                "result": result,
            }
        except Exception:
            pass  # corrupt sidecar — skip silently


# --------------------------------------------------------------------------- #
# QA report rendering
# --------------------------------------------------------------------------- #
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
def _preview(uploaded_file, conn) -> dict:
    pdf_bytes = uploaded_file.getvalue()
    sha = upload.content_hash(pdf_bytes)
    _, pdf_path = upload.save_pdf(pdf_bytes)
    parsed_doi = doi_mod.parse_doi(pdf_bytes)
    meta = pdf_inspect.inspect(pdf_bytes)
    existing = dedup.find_existing(conn, sha, parsed_doi)
    return {
        "filename": uploaded_file.name,
        "sha": sha,
        "pdf_path": str(pdf_path),
        "pdf_bytes": pdf_bytes,
        "doi": parsed_doi,
        "meta": meta,
        "status": f"existing (paper_id={existing.paper_id})" if existing else "new",
    }


def _run_batch(selected: list[dict], figure_is_curve: bool) -> None:
    pending = st.session_state.setdefault("pending", {})
    errors = []
    for i, p in enumerate(selected, start=1):
        with st.status(f"[{i}/{len(selected)}] Extracting {p['filename']}…", expanded=False):
            try:
                result = runner.extract_paper(p["pdf_bytes"], figure_is_curve=figure_is_curve)
            except PromptNotReadyError as e:
                errors.append((p["filename"], f"Prompt not ready: {e}"))
                continue
            except ParseError as e:
                errors.append((p["filename"], f"Could not parse model output: {e}"))
                continue
            except Exception as e:  # API/auth/etc. — record and keep going
                errors.append((p["filename"], f"Extraction failed: {e}"))
                continue
        sha = p["sha"]
        result.df.to_excel(_staging_path(sha), index=False, engine="openpyxl")
        stash = {
            "sha": sha,
            "pdf_path": p["pdf_path"],
            "doi": p["doi"],
            "meta": p["meta"],
            "filename": p["filename"],
            "result": result,
        }
        _save_staging_meta(sha, stash, result)
        pending[sha] = stash
    n_ok = len(selected) - len(errors)
    if n_ok:
        st.success(f"Extracted {n_ok}/{len(selected)} paper(s) — ready for review below.")
    for filename, msg in errors:
        st.error(f"{filename}: {msg}")


# --------------------------------------------------------------------------- #
# Review + merge queue (one paper at a time, picked from the pending batch)
# --------------------------------------------------------------------------- #
def render_review_queue() -> None:
    pending: dict = st.session_state.get("pending", {})
    if not pending:
        return

    st.divider()
    st.subheader(f"Review queue ({len(pending)} pending)")
    options = list(pending.keys())
    sha = st.selectbox(
        "Paper to review",
        options,
        format_func=lambda s: f"{pending[s]['filename']} ({len(pending[s]['result'].df)} rows)",
    )
    stash = pending[sha]
    result = stash["result"]

    render_qa(result.qa_report)

    st.write(f"**{len(result.df)} rows extracted.** Edit cells below if needed.")
    edited = st.data_editor(result.df, num_rows="dynamic", use_container_width=True, key=f"editor_{sha}")

    if result.text_endpoints:
        with st.expander(f"Captured text endpoints ({len(result.text_endpoints)})"):
            st.dataframe(pd.DataFrame(result.text_endpoints), use_container_width=True)

    note = st.text_input("Review note (optional)", key=f"note_{sha}")
    override = False
    if not result.qa_report.passed:
        override = st.checkbox(
            "Override red QA flags and merge anyway (records the override in the audit log)",
            key=f"override_{sha}",
        )

    col_a, col_b = st.columns(2)
    if col_a.button("✅ Approve & merge", type="primary", key=f"approve_{sha}"):
        if not auth.require_write_access():
            st.stop()
        if not result.qa_report.passed and not override:
            st.error("Red QA flags present — tick the override box to merge anyway.")
            st.stop()
        edited_clean = schema.coerce_schema(edited)
        was_edited = not edited_clean.reset_index(drop=True).equals(
            result.df.reset_index(drop=True)
        )
        merge_note = note or ("edited in review" if was_edited else None)
        conn = connection.get_conn()
        try:
            summary = merge.commit_extraction(
                conn,
                content_sha256=stash["sha"],
                pdf_path=stash["pdf_path"],
                df=edited_clean,
                text_endpoints=result.text_endpoints,
                prompt_version=result.prompt_version,
                prompt_sha256=result.prompt_sha256,
                model=result.model,
                qa_passed=result.qa_report.passed,
                qa_report_json=result.qa_report.to_json(),
                raw_response=result.raw_response,
                doi=stash["doi"],
                reference_no=_first_value(edited_clean, "Reference No."),
                title=stash["meta"].get("title"),
                original_filename=stash["filename"],
                figure_type=None,
                is_raster_figure=stash["meta"].get("is_raster_figure"),
                note=merge_note,
                override=override,
            )
        finally:
            conn.close()
        # Export the approved per-paper spreadsheet artifact.
        export_path = config.EXPORTS_DIR / f"paper_{summary['paper_id']}.xlsx"
        edited_clean.to_excel(export_path, index=False, engine="openpyxl")
        st.success(
            f"Merged {summary['rows_merged']} rows → paper_id={summary['paper_id']}, "
            f"prompt_run_id={summary['prompt_run_id']}. Export: {export_path.name}"
        )
        pending.pop(sha, None)
        _delete_staging_files(sha)
        st.rerun()

    if col_b.button("🗑️ Reject", key=f"reject_{sha}"):
        pending.pop(sha, None)
        _delete_staging_files(sha)
        st.info(f"{stash['filename']} rejected and discarded (nothing written to the master DB).")
        st.rerun()


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def main() -> None:
    _restore_staging_queue()
    st.title("REE Extraction Dashboard")
    st.caption("Phase A2 — batch PDF upload → 26-column tables → review → merge")

    uploaded_files = st.file_uploader(
        "Upload one or more research-paper PDFs", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Upload one or more PDFs to begin.")
        render_review_queue()
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
                "File": p["filename"],
                "DOI": p["doi"] or "—",
                "Pages": p["meta"]["n_pages"],
                "Raster?": "yes" if p["meta"]["is_raster_figure"] else "no/unknown",
                "Status": p["status"],
            }
            for p in previews
        ]
    )
    edited_table = st.data_editor(
        table,
        use_container_width=True,
        hide_index=True,
        disabled=["File", "DOI", "Pages", "Raster?", "Status"],
        key="batch_table",
    )

    figure_is_curve = st.checkbox(
        "Primary figure(s) are multi-point curves (enables sparse-result QA check)",
        value=True,
    )

    selected = [p for p, inc in zip(previews, edited_table["Include"]) if inc]
    dup_selected = [p for p in selected if p["status"] != "new"]
    if dup_selected:
        st.warning(
            f"{len(dup_selected)} selected file(s) already exist in the DB — "
            "extracting will record a **new run** against the existing paper "
            "(coexistence), not a duplicate paper."
        )

    if st.button(
        f"Run extraction on {len(selected)} paper(s)", type="primary", disabled=not selected
    ):
        if not auth.require_write_access():
            st.stop()
        _run_batch(selected, figure_is_curve)

    render_review_queue()


def _first_value(df: pd.DataFrame, col: str):
    if col not in df.columns or df.empty:
        return None
    s = df[col].dropna()
    return None if s.empty else str(s.iloc[0])


# Streamlit executes this module top-to-bottom on every interaction.
main()
