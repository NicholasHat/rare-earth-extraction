"""evaluation: row matching, cost accounting, failed-run dedup, and the
compare CLI end to end on temporary run directories (no API, no real PDFs)."""
import json

import pandas as pd
import pytest

from evaluation import compare, cost, match, sources
from validation import schema

EL = schema.ELEMENT_COLUMN


def _table(rows):
    return schema.coerce_schema(pd.DataFrame(rows))


def _sweep(element="Nd", extractant="Cyanex 272", n=6, shift=0.0, step=0.4):
    return [{EL: element, "Extractant": extractant, "Extractant Conc. (mM)": 500.0,
             "pH": 1.0 + step * i + shift, "Extract%": 10.0 + 12.0 * i} for i in range(n)]


def test_identical_tables_agree_completely():
    a = _table(_sweep())
    out = match.agreement(a, a.copy())
    assert out["matched"] == 6 and out["share_a_matched"] == out["share_b_matched"] == 1.0
    assert out["extract_pct_abs_diff"]["median"] == 0.0


def test_extractant_spelling_does_not_stop_a_match():
    out = match.agreement(_table(_sweep(extractant="Cyanex 272")), _table(_sweep(extractant="Cyanex272")))
    assert out["matched"] == 6


def test_points_beyond_the_ph_tolerance_do_not_match():
    out = match.agreement(_table(_sweep(step=1.0)), _table(_sweep(step=1.0, shift=0.5)))
    assert out["matched"] == 0


def test_systems_only_one_side_has_are_listed():
    a = _table(_sweep("Nd") + _sweep("La"))
    b = _table(_sweep("Nd"))
    out = match.agreement(a, b)
    assert out["systems_only_in_a"] == {"La|cyanex272": 6} and out["systems_only_in_b"] == {}


def test_score_reads_agreement_as_recall_and_precision():
    ref = _table(_sweep(n=6))
    pred = _table(_sweep(n=3) + _sweep("La", n=3))
    out = match.score(pred, ref)
    assert out["recall"] == 0.5 and out["precision"] == 0.5


def test_batch_runs_cost_half_and_unknown_models_are_unpriced():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    assert cost.usd(usage, "claude-sonnet-5") == 2.0
    assert cost.usd(usage, "claude-sonnet-5", via_batch=True) == 1.0
    assert cost.usd(usage, "some-future-model") is None


def test_a_re_collected_batch_failure_is_one_run():
    turn = {"input_tokens": 1275, "output_tokens": 88167,
            "cache_creation_input_tokens": 316036, "cache_read_input_tokens": 4533054}
    legacy = [{"usage": turn, "turns_completed": 1}, {"usage": dict(turn, output_tokens=88171), "turns_completed": 1}]
    batch = [{"usage": turn, "turns_completed": 1, "batch_id": "msgbatch_1"}] * 2
    runs = sources._distinct_failed_runs(legacy + batch)
    assert len(runs) == 2
    assert {b for _, b in runs} == {True, False}                   # legacy priced full, batch at half


def test_compare_writes_summary_and_report(tmp_path, monkeypatch):
    pdf = tmp_path / "papers" / "paper_one.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4 test")
    runs = tmp_path / "runs" / "paper_one"
    runs.mkdir(parents=True)
    _table(_sweep() + _sweep("La")).to_csv(runs / "rows.csv", index=False)
    (runs / "run.json").write_text(json.dumps({"elapsed_minutes": 4.2, "model_calls": 7}))
    monkeypatch.setattr(sources, "load_claude", lambda sha: sources.Result(
        "staged", _table(_sweep()), {"prompt_version": "extraction_v12", "cost_usd": 1.23}))
    out = tmp_path / "out"
    assert compare.main([str(pdf.parent), "--local-runs", str(tmp_path / "runs"), "--out", str(out)]) == 0
    summary = pd.read_csv(out / "summary.csv")
    row = summary.iloc[0]
    assert row["claude_rows"] == 6 and row["local_rows"] == 12 and row["matched"] == 6
    assert row["local_minutes"] == 4.2 and row["claude_cost_usd"] == 1.23
    assert "Only local has: La|cyanex272 (6)" in (out / "report.md").read_text()


def test_report_lists_rows_per_figure_and_why_a_figure_gave_none(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 x")
    run = tmp_path / "runs" / "p"
    run.mkdir(parents=True)
    rows = _table(_sweep())
    rows.to_csv(run / "rows.csv", index=False)
    rows.assign(_figure=2).to_csv(run / "rows_debug.csv", index=False)
    (run / "panels.json").write_text(json.dumps([
        {"figure": 2, "skipped": None}, {"figure": 6, "skipped": "model: not an x-y plot of data points"}]))
    monkeypatch.setattr(sources, "load_claude", lambda sha: sources.Result("staged", _table(_sweep()), {
        "figures": [{"figure": "Fig. 2", "rows": 6, "digitised": True, "reason": ""},
                    {"figure": "Fig. 6", "rows": 0, "digitised": False, "reason": "NMR spectrum"}]}))
    compare.main([str(pdf), "--local-runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out")])
    report = (tmp_path / "out" / "report.md").read_text()
    assert "| 2 | 6 | 6 |  |  |" in report
    assert "| 6 | 0 | 0 | NMR spectrum | model: not an x-y plot of data points |" in report
