"""Headless bulk runner: extract openly-licensed papers through the Batches API
without the dashboard.

    python -m extraction.bulk submit [--manifest PATH] (--first-batch | --tier A [--tier B]) [--limit N] [--dry-run]
    python -m extraction.bulk submit --reextract SHA [--reextract SHA ...] [--dry-run]
    python -m extraction.bulk status
    python -m extraction.bulk collect [--batch ID]
    python -m extraction.bulk discard --batch ID

`submit` picks papers from the open-access manifest (data/open_access/manifest.csv
by default) and submits them as one batch (`--reextract` instead names staged
papers, by sha256 prefix, to extract again under the current prompt — the
result each replaces is kept under data/staging/_superseded/); `status` reconciles the local job
records with the API (finding any batch an interrupted submission left
untracked) and shows the review queue; `collect` stages every ended batch's
results exactly as the dashboard does, finishing paused items; `discard`
gives up on a job's failed papers so they can be submitted again.

Every command is idempotent and safe to interrupt or re-run: a paper that is
staged, approved in the DB, or held by any batch job is never submitted
again, and collection resumes where it stopped without re-paying. Nothing
here writes the master DB — results wait in the review queue for a human to
approve them. Batches submitted here appear in the dashboard's "Batch API
jobs" section, and vice versa (same records: extraction/batch_jobs.py).

Exit status: 0 = fine, 1 = some paper failed or was refused, 3 = the
Anthropic credit balance is exhausted (everything done so far is kept).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

import config
from database import connection, papers_repo
from ingestion import manifest, pdf_inspect, upload
from ingestion.licensing import License, is_open

from . import anthropic_client, batch_jobs, staging

EXIT_OK, EXIT_FAILED, EXIT_NO_CREDIT = 0, 1, 3


def _approved_in_db() -> tuple[set[str], set[str]]:
    """(content hashes, DOIs) of papers with an approved run in the master DB."""
    try:
        conn = connection.get_readonly_conn()
    except FileNotFoundError:
        return set(), set()
    try:
        rows = conn.execute(
            "SELECT p.content_sha256, p.doi FROM papers p WHERE EXISTS "
            "(SELECT 1 FROM prompt_runs r WHERE r.paper_id = p.paper_id AND r.status = 'approved')"
        ).fetchall()
    except sqlite3.Error:
        return set(), set()
    finally:
        conn.close()
    return {r[0] for r in rows}, {r[1] for r in rows if r[1]}


def select(entries: list[manifest.ManifestEntry], *, first_batch: bool, tiers: list[str],
           limit: int | None, reextract: list[str] = ()) -> tuple[list[manifest.ManifestEntry], list[tuple[manifest.ManifestEntry, str]]]:
    """The manifest rows to submit, in manifest order, and every row of the
    requested set that was left out, with the reason. `reextract` (sha256
    prefixes) requests exactly those papers instead, staged or not."""
    approved_shas, approved_dois = _approved_in_db()
    busy = staging.in_flight_shas()
    staged = set(staging.load_all())
    wanted_tiers = {t.lower() for t in tiers}
    prefixes = tuple(p.lower() for p in reextract)

    chosen, left_out = [], []
    for e in entries:
        tier = (e.tier or "").lower()
        if prefixes and not e.sha256.startswith(prefixes):
            continue
        if first_batch and not e.first_batch:
            continue
        if wanted_tiers and tier not in wanted_tiers:
            continue
        reason = None
        if tier in ("reject", "unverified"):
            reason = f"tier is {e.tier}"
        elif not is_open(e.license):
            reason = f"licence '{e.license or 'none'}' is not an open licence"
        elif not e.pdf.exists():
            reason = f"PDF not found at {e.pdf}"
        elif upload.content_hash(e.pdf.read_bytes()) != e.sha256:
            reason = "the PDF's sha256 doesn't match the manifest"
        elif e.sha256 in approved_shas or (e.doi and papers_repo.canonical_doi(e.doi) in approved_dois):
            reason = "already approved in the database"
        elif e.sha256 in staged and not prefixes:
            reason = "already staged for review"
        elif e.sha256 in busy:
            reason = "already in a batch job"
        elif limit is not None and len(chosen) >= limit:
            reason = f"over --limit {limit}"
        if reason:
            left_out.append((e, reason))
        else:
            chosen.append(e)
    return chosen, left_out


def _paper_ref(e: manifest.ManifestEntry) -> tuple[staging.PaperRef, bytes]:
    """Ingest a manifest PDF the way the dashboard's upload does: persisted
    under data/incoming/<sha>.pdf, triaged, with the manifest's licence."""
    pdf = e.pdf.read_bytes()
    sha, path = upload.save_pdf(pdf)
    meta = pdf_inspect.inspect(pdf)
    if e.title:
        meta["title"] = e.title
    return staging.PaperRef(
        sha=sha, pdf_path=str(path), doi=e.doi, filename=e.title or e.pdf.name, meta=meta,
        license=License.of(e.license, e.license_evidence, e.source_url, origin="manifest"),
    ), pdf


def _short(text: str | None, n: int = 60) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def cmd_submit(args) -> int:
    try:
        entries = manifest.read(args.manifest)
    except manifest.ManifestError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_FAILED
    reextract = args.reextract or []
    chosen, left_out = select(entries, first_batch=args.first_batch, tiers=args.tier or [], limit=args.limit,
                              reextract=reextract)
    for prefix in reextract:
        if not any(e.sha256.startswith(prefix.lower()) for e in entries):
            print(f"error: no manifest row has a sha256 starting {prefix}", file=sys.stderr)
            return EXIT_FAILED
    for e, reason in left_out:
        print(f"  skip    {e.sha256[:12]}  {_short(e.title)} — {reason}")
    for e in chosen:
        print(f"  submit  {e.sha256[:12]}  {_short(e.title)}  [{e.license}, tier {e.tier}]")
    print(f"{len(chosen)} paper(s) to submit, {len(left_out)} left out "
          f"(prompt {config.EXTRACTION_PROMPT_VERSION}, model {config.EXTRACTION_MODEL}).")
    if args.dry_run or not chosen:
        return EXIT_OK
    try:
        outcome = batch_jobs.submit([_paper_ref(e) for e in chosen], figure_is_curve=True,
                                    replacing_staged=bool(reextract))
    except anthropic_client.CreditExhausted as e:
        print(f"CREDIT EXHAUSTED: {e}", file=sys.stderr)
        return EXIT_NO_CREDIT
    except anthropic_client.BatchSubmissionUnknown as e:
        print(f"error: {e}\nRun `python -m extraction.bulk status` — it finds the batch if it was "
              "created, or cleans up if it wasn't. Nothing will be submitted twice.", file=sys.stderr)
        return EXIT_FAILED
    for sha, reason in outcome.skipped.items():
        print(f"  skipped at submission: {sha[:12]} — {reason}")
    if outcome.job is None:
        print("Nothing submitted.")
        return EXIT_OK
    print(f"Submitted batch {outcome.job.batch_id} with {len(outcome.job.items)} paper(s). "
          "Batches usually end within an hour (at most 24h); then run "
          "`python -m extraction.bulk collect`.")
    return EXIT_OK


def _days(td: timedelta) -> str:
    return f"{td.total_seconds() / 86400:.1f} days"


def cmd_status(args) -> int:
    report = batch_jobs.reconcile()
    if report.api_error:
        print(f"warning: could not list batches from the API ({report.api_error}); local records only")
    for job in report.adopted:
        print(f"RECOVERED  {job.batch_id}: an interrupted submission's batch was found and is tracked again")
    for job in report.abandoned:
        print(f"CLEANED UP an interrupted submission of {len(job.papers)} paper(s): no batch was created; "
              "its uploads were deleted and the papers are free to submit")
    for job, why in report.unresolved:
        print(f"UNRESOLVED submission {job.job_id[:8]} ({len(job.papers)} paper(s)): {why}")

    print(f"\nBatch jobs ({len(report.jobs)}):")
    for st in report.jobs:
        job, info = st.job, st.info
        if info is not None:
            state = (f"{info.processing_status}: {info.succeeded} succeeded, {info.errored} errored, "
                     f"{info.expired} expired, {info.canceled} canceled of {info.requests}")
        else:
            state = f"unknown ({st.error})"
        warn = "  << COLLECT SOON" if st.expires_in < batch_jobs.EXPIRY_WARNING else ""
        busy = "  [being collected now]" if job.collection_in_progress() else ""
        print(f"  {job.batch_id}  submitted {job.submitted_at[:16]}  {state}  "
              f"results kept {_days(st.expires_in)} more{warn}{busy}")
        for sha, paper in job.papers.items():
            line = f"      {sha[:12]}  {_short(paper.filename)}"
            if sha in job.errors:
                line += f"  FAILED: {_short(job.errors[sha], 140)}"
            print(line)
    if report.untracked:
        print(f"\nBatches on the account that no local record tracks ({len(report.untracked)}):")
        for b in report.untracked:
            print(f"  {b.batch_id}  created {b.created_at:%Y-%m-%d %H:%M}  {b.processing_status}  "
                  f"{b.requests} request(s)")
    _print_queue()
    return EXIT_OK


def _print_queue() -> None:
    from evaluation.cost import result_usd

    queue = staging.load_all()
    print(f"\nReview queue ({len(queue)} staged):")
    total = 0.0
    for sha, s in queue.items():
        r = s.result
        cost = result_usd(r)
        total += cost or 0.0
        lic = s.paper.license.id if s.paper.license else "NO LICENCE"
        print(f"  {sha[:12]}  {_short(s.paper.filename, 50):50}  {len(r.df):4} rows  "
              f"{len(r.qa_report.reds)} red / {len(r.qa_report.ambers)} amber  "
              f"${cost if cost is not None else float('nan'):.2f}  {r.prompt_version}  {lic}")
    if queue:
        print(f"  total ≈ ${total:.2f}")
    failed = staging.load_failed_runs(limit=5)
    if failed:
        print(f"\nRecent failed runs (newest first, see data/staging/_failed_runs.jsonl):")
        for f in failed:
            print(f"  {f['at'][:16]}  {_short(f['filename'], 40)}  {_short(f['error'], 120)}")


def cmd_collect(args) -> int:
    jobs = staging.load_batch_jobs()
    ids = [args.batch] if args.batch else list(jobs)
    if not ids:
        print("No batch jobs to collect. (Run `status` to look for untracked ones.)")
        return EXIT_OK
    code = EXIT_OK
    for batch_id in ids:
        try:
            outcome = batch_jobs.collect(batch_id)
        except Exception as e:   # e.g. the connection dropped while downloading results
            no_credit = anthropic_client.is_credit_exhausted(e)
            print(f"{batch_id}: collection stopped — {e}. Everything saved so far is kept; run collect "
                  "again to carry on.", file=sys.stderr)
            code = max(code, EXIT_NO_CREDIT if no_credit else EXIT_FAILED)
            continue
        if outcome.status == "gone":
            print(f"{batch_id}: no such job (already collected or discarded)")
            continue
        if outcome.status == "busy":
            print(f"{batch_id}: being collected by another process right now — left alone")
            continue
        if outcome.status == "not_ended":
            print(f"{batch_id}: still {outcome.processing_status} — collect again later")
            continue
        print(f"{batch_id}: collected")
        for p in outcome.papers:
            if p.staged:
                cost = f"${p.cost_usd:.2f}" if p.cost_usd is not None else "$?"
                print(f"  staged  {p.sha[:12]}  {_short(p.filename, 50):50}  {p.rows:4} rows  "
                      f"{p.reds} red / {p.ambers} amber  {cost}")
            else:
                print(f"  FAILED  {p.sha[:12]}  {_short(p.filename, 50)}: {_short(p.error, 200)}"
                      + (f"\n          transcript: data/{p.transcript}" if p.transcript else ""))
                code = max(code, EXIT_FAILED)
        if outcome.credit_exhausted:
            print("  CREDIT EXHAUSTED: the failed papers above keep every completed turn; top up and run "
                  "collect again — it resumes, it doesn't restart.")
            code = EXIT_NO_CREDIT
        elif outcome.failures:
            print(f"  {len(outcome.failures)} paper(s) failed and stay on the job: collect again to retry, or "
                  f"`discard --batch {batch_id}` to free them for resubmission.")
    return code


def cmd_discard(args) -> int:
    try:
        batch_jobs.discard(args.batch)
    except staging.Busy as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_FAILED
    print(f"{args.batch}: discarded (uploads deleted; its papers can be submitted again).")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m extraction.bulk", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="submit manifest papers as one batch")
    p.add_argument("--manifest", type=Path, default=config.OPEN_ACCESS_MANIFEST)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--first-batch", action="store_true", help="the rows marked first_batch")
    which.add_argument("--tier", action="append", choices=["A", "B"], help="rows of this tier (repeatable)")
    which.add_argument("--reextract", action="append", metavar="SHA",
                       help="extract this paper again even though it is staged (sha256 prefix, repeatable)")
    p.add_argument("--limit", type=int, help="at most this many papers")
    p.add_argument("--dry-run", action="store_true", help="show the selection, submit nothing")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="reconcile with the API; show jobs and the review queue")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("collect", help="stage the results of every ended batch")
    p.add_argument("--batch", help="only this batch id")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("discard", help="give up on a job's failed papers")
    p.add_argument("--batch", required=True)
    p.set_defaults(func=cmd_discard)

    args = parser.parse_args(argv)
    config.ensure_dirs()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
