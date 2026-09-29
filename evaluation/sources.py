"""A paper's extraction result from each pipeline, loaded read-only.

This pipeline, by the paper's content sha256, first found wins: a staged
result awaiting review, else the current approved run (`v_current_best`),
else only the cost of runs that failed (`_failed_runs.jsonl`). The local
pipeline (ree-extraction-local), by the PDF's file stem: its run directory's
rows.csv, plus run.json / llm_calls.json for runtime.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config
from database import connection
from extraction import provenance, staging
from validation.schema import coerce_schema

from . import cost


@dataclass
class Result:
    source: str                      # "staged" | "approved" | "failed" | "local" | "missing"
    df: pd.DataFrame | None = None
    info: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return 0 if self.df is None else len(self.df)


def _claude_approved(sha: str) -> Result | None:
    try:
        conn = connection.get_readonly_conn()
    except FileNotFoundError:
        return None
    try:
        paper = conn.execute("SELECT paper_id FROM papers WHERE content_sha256 = ?", (sha,)).fetchone()
        if paper is None:
            return None
        df = pd.read_sql("SELECT * FROM v_current_best WHERE paper_id = ?", conn, params=(paper[0],))
        if df.empty:
            return None
        run = conn.execute("SELECT * FROM prompt_runs WHERE prompt_run_id = ?",
                           (int(df["prompt_run_id"].iloc[0]),)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    usage = {k: run[k] for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                 "cache_read_input_tokens")}
    return Result("approved", coerce_schema(df), {
        "prompt_version": run["prompt_version"], "model": run["model"], "usage": usage,
        "cost_usd": cost.usd(usage, run["model"]),   # prompt_runs doesn't record batch vs sync
    })


def _distinct_failed_runs(failed: list[dict]) -> list[tuple[dict, bool]]:
    """(usage, via_batch) per distinct run. Re-collecting a failed batch item
    logs its batch turn's usage again, plus whatever the new continuation
    attempt billed, so one batch is counted once, at its largest entry.
    Entries logged before batch_id was recorded are grouped by identical
    uncached input and turn count — a re-collection's signature — and priced
    at the full rate, since whether they were batched is unknown (an upper
    bound)."""
    groups: dict[tuple, list[dict]] = {}
    for f in failed:
        usage = f.get("usage") or {}
        key = (("batch", f["batch_id"]) if f.get("batch_id")
               else ("legacy", usage.get("input_tokens"), f.get("turns_completed")))
        groups.setdefault(key, []).append(usage)
    return [(max(us, key=lambda u: u.get("output_tokens") or 0), key[0] == "batch")
            for key, us in groups.items()]


def load_claude(sha: str) -> Result:
    staged = staging.load_all().get(sha)
    if staged is not None:
        r = staged.result
        usage = {k: getattr(r, k) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                            "cache_read_input_tokens")}
        return Result("staged", r.df, {
            "prompt_version": r.prompt_version, "model": r.model, "usage": usage, "via_batch": r.via_batch,
            "cost_usd": cost.usd(usage, r.model, via_batch=r.via_batch),
            "figures": provenance.summarise(r.figures, r.row_figures),   # empty before extraction_v13
        })
    approved = _claude_approved(sha)
    if approved is not None:
        return approved
    failed = [f for f in staging.load_failed_runs(limit=10_000) if f.get("sha") == sha]
    if failed:
        runs = _distinct_failed_runs(failed)
        spent = sum(cost.usd(usage, config.EXTRACTION_MODEL, via_batch=batch) or 0 for usage, batch in runs)
        return Result("failed", None, {"failures": len(failed), "distinct_runs": len(runs),
                                       "last_error": failed[0]["error"][:200], "cost_usd": round(spent, 3)})
    return Result("missing")


def load_local(stem: str, runs_dir: Path) -> Result:
    run = runs_dir / stem
    rows = run / "rows.csv"
    if not rows.exists():
        return Result("missing")
    info: dict = {}
    if (run / "run.json").exists():
        info.update(json.loads((run / "run.json").read_text()))
    elif (run / "llm_calls.json").exists():
        calls = json.loads((run / "llm_calls.json").read_text())
        info.update(model_calls=len(calls), model_minutes=round(sum(c["seconds"] for c in calls) / 60, 1))
    if (run / "panels.json").exists():
        panels = json.loads((run / "panels.json").read_text())
        info["panels"] = len(panels)
        info["panels_skipped"] = sum(1 for p in panels if p.get("skipped"))
        info["figures"] = _local_figures(panels, run / "rows_debug.csv")
    return Result("local", coerce_schema(pd.read_csv(rows)), info)


def _local_figures(panels: list[dict], debug_csv: Path) -> list[dict]:
    """The local run's per-figure record in provenance.summarise's shape:
    rows per figure from rows_debug.csv, and the skip reasons of its panels."""
    counts = (pd.read_csv(debug_csv)["_figure"].dropna().astype(int).astype(str).value_counts().to_dict()
              if debug_csv.exists() and debug_csv.stat().st_size else {})
    by_figure: dict[str, set[str]] = {}
    for p in panels:
        key = str(p.get("figure")) if p.get("figure") is not None else "?"
        reasons = by_figure.setdefault(key, set())
        if p.get("skipped"):
            reasons.add(p["skipped"])
    return [{"figure": f"Fig. {k}", "rows": int(counts.get(k, 0)), "digitised": counts.get(k, 0) > 0,
             "reason": "; ".join(sorted(r)) if not counts.get(k) else ""}
            for k, r in sorted(by_figure.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0))]
