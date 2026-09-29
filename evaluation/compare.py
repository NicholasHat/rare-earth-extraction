"""Compare this pipeline and the local pipeline paper by paper.

    python -m evaluation.compare PAPERS... [--local-runs DIR] [--ref-dir DIR] [--out DIR]

PAPERS are PDF files or directories of them. Each PDF is looked up by content
sha256 on this side (staged, else approved, else failed runs) and by file
stem in the local pipeline's run directory. Both tables go through the same
QA checks; then their rows are matched against each other and, when
`--ref-dir` holds `<stem>.csv`, against that reference.

Writes to --out: summary.csv (one row per paper), <stem>.json (full detail)
and report.md. Read-only: no API call, nothing staged or merged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import config
from validation import checks
from validation.schema import coerce_schema

from extraction import provenance

from . import match, sources


def _pdfs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    return out


def _qa(result: sources.Result) -> dict:
    if result.df is None:
        return {"verdict": None}
    report = checks.run(result.df, [], figure_is_curve=True)
    return {"verdict": report.verdict.value, "reds": len(report.reds), "ambers": len(report.ambers),
            "red_checks": sorted({f.check for f in report.reds})}


def compare_paper(pdf: Path, local_runs: Path | None, ref_dir: Path | None) -> dict:
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    claude = sources.load_claude(sha)
    local = sources.load_local(pdf.stem, local_runs) if local_runs else sources.Result("missing")
    detail: dict = {"paper": pdf.stem, "sha256": sha,
                    "claude": {"source": claude.source, "rows": claude.rows, **claude.info, "qa": _qa(claude)},
                    "local": {"source": local.source, "rows": local.rows, **local.info, "qa": _qa(local)}}
    if claude.df is not None and local.df is not None:
        detail["agreement"] = match.agreement(claude.df, local.df)   # a = claude, b = local
    ref_path = ref_dir / f"{pdf.stem}.csv" if ref_dir else None
    if ref_path and ref_path.exists():
        ref = coerce_schema(pd.read_csv(ref_path))
        detail["reference_rows"] = len(ref)
        for name, r in (("claude", claude), ("local", local)):
            if r.df is not None:
                detail[name]["vs_reference"] = match.score(r.df, ref)
    return detail


def _summary_row(d: dict) -> dict:
    c, l, a = d["claude"], d["local"], d.get("agreement") or {}
    row = {
        "paper": d["paper"],
        "claude_source": c["source"], "claude_rows": c["rows"], "claude_qa": c["qa"]["verdict"],
        "claude_reds": c["qa"].get("reds"), "claude_cost_usd": c.get("cost_usd"),
        "claude_prompt": c.get("prompt_version"),
        "local_source": l["source"], "local_rows": l["rows"], "local_qa": l["qa"]["verdict"],
        "local_reds": l["qa"].get("reds"), "local_minutes": l.get("elapsed_minutes", l.get("model_minutes")),
        "matched": a.get("matched"), "claude_share_matched": a.get("share_a_matched"),
        "local_share_matched": a.get("share_b_matched"),
        "median_abs_dExtract": (a.get("extract_pct_abs_diff") or {}).get("median"),
    }
    for name in ("claude", "local"):
        ref = d[name].get("vs_reference")
        if ref:
            row[f"{name}_recall"], row[f"{name}_precision"] = ref["recall"], ref["precision"]
    return row


def _fmt(v) -> str:
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def report_md(details: list[dict], summary: pd.DataFrame) -> str:
    cols = [c for c in summary.columns if c != "claude_prompt"]
    lines = [f"# Pipeline comparison — {date.today().isoformat()}", "",
             f"{len(details)} papers. `a` = this pipeline (Claude), `b` = local pipeline.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for _, r in summary.iterrows()]
    spent = pd.to_numeric(summary["claude_cost_usd"], errors="coerce")
    lines += ["", f"Claude cost across papers (estimate): ${spent.sum():.2f}"
              + (f" ({spent.isna().sum()} paper(s) with no recorded cost)" if spent.isna().any() else ""),
              "", "## Where they disagree", ""]
    for d in details:
        a = d.get("agreement")
        lines.append(f"### {d['paper']}")
        if not a:
            lines += [f"- Not comparable: Claude {d['claude']['source']}, local {d['local']['source']}.", ""]
            continue
        for label, key in (("Only Claude has", "systems_only_in_a"), ("Only local has", "systems_only_in_b")):
            if a[key]:
                lines.append(f"- {label}: " + ", ".join(f"{k} ({v})" for k, v in a[key].items()))
        if a["systems_count_differs"]:
            lines.append("- Row counts differ by 3+ (Claude, local): "
                         + ", ".join(f"{k} {v}" for k, v in a["systems_count_differs"].items()))
        lines += _figure_table(d)
        low = {k: v for k, v in (a.get("field_agreement") or {}).items() if v is not None and v < 0.9}
        if low:
            lines.append("- Fields that disagree on matched rows: " + ", ".join(f"{k} {v}" for k, v in low.items()))
        for name in ("claude", "local"):
            if d[name]["qa"].get("red_checks"):
                lines.append(f"- {name} QA RED: " + ", ".join(d[name]["qa"]["red_checks"]))
        lines.append("")
    return "\n".join(lines)


def _figure_table(d: dict) -> list[str]:
    """Rows per figure from each pipeline, with the reason a figure gave none —
    which figures each side used is often the whole story of a disagreement."""
    sides = {name: {provenance.figure_number(f["figure"]): f for f in d[name].get("figures") or []}
             for name in ("claude", "local")}
    keys = sorted(set(sides["claude"]) | set(sides["local"]),
                  key=lambda k: (not str(k).isdigit(), int(k) if str(k).isdigit() else 0, str(k)))
    if not keys:
        return []
    out = ["", "| Figure | Claude rows | Local rows | Claude: why none | Local: why none |", "|---|---|---|---|---|"]
    for k in keys:
        c, l = sides["claude"].get(k), sides["local"].get(k)
        out.append(f"| {k} | {_fmt(c and c['rows'])} | {_fmt(l and l['rows'])} | "
                   f"{(c or {}).get('reason', '') if c and not c['rows'] else ''} | "
                   f"{(l or {}).get('reason', '') if l and not l['rows'] else ''} |")
    return out + [""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("papers", nargs="+", type=Path, help="PDF files or directories of PDFs")
    ap.add_argument("--local-runs", type=Path, default=config.LOCAL_PIPELINE_RUNS_DIR,
                    help="ree-extraction-local's data/runs directory (default: LOCAL_PIPELINE_RUNS_DIR)")
    ap.add_argument("--ref-dir", type=Path, help="reference tables named <paper stem>.csv")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "comparisons" / date.today().isoformat())
    args = ap.parse_args(argv)

    pdfs = _pdfs(args.papers)
    if not pdfs:
        print("no PDFs found", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    details = [compare_paper(p, args.local_runs, args.ref_dir) for p in pdfs]
    for d in details:
        (args.out / f"{d['paper']}.json").write_text(json.dumps(d, indent=1, default=str))
    summary = pd.DataFrame([_summary_row(d) for d in details])
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "report.md").write_text(report_md(details, summary))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.to_string(index=False))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
