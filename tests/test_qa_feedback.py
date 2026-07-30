"""Tests for the on-demand QA-feedback re-extraction plumbing (pure — no API).

The review UI's re-extract action renders the previous attempt's QA findings
(runner.qa_feedback_block) and injects them into the user turn like the curve
pre-pass block (anthropic_client._build_user_content) — never into the pinned
prompt file.
"""
from extraction import anthropic_client, runner
from validation.report import QAReport, Severity


def _report() -> QAReport:
    r = QAReport()
    r.add("deterministic_curve_count", Severity.AMBER, "expected 19 rows/series, got 15")
    r.add("monotonicity", Severity.RED, "Extract% not rising with pH for La")
    return r


def test_qa_feedback_block_lists_every_flag_with_severity():
    block = runner.qa_feedback_block(_report(), row_count=210)
    assert "QA FEEDBACK ON A PREVIOUS EXTRACTION ATTEMPT" in block
    assert "210 rows" in block
    assert "- [AMBER] deterministic_curve_count: expected 19 rows/series, got 15" in block
    assert "- [RED] monotonicity: Extract% not rising with pH for La" in block


def test_user_content_injects_feedback_after_analysis_block():
    content = anthropic_client._build_user_content(
        "file_123", "## DETERMINISTIC CURVE ANALYSIS\nx", "## QA FEEDBACK\ny"
    )
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert any("DETERMINISTIC CURVE ANALYSIS" in t for t in texts)
    assert any("QA FEEDBACK" in t for t in texts)
    # Ordering: analysis, then feedback, then the cache-controlled instruction last.
    assert texts.index(next(t for t in texts if "QA FEEDBACK" in t)) == 1
    assert "cache_control" in content[-1]


def test_user_content_unchanged_without_feedback():
    content = anthropic_client._build_user_content("file_123", None)
    assert not any("QA FEEDBACK" in c.get("text", "") for c in content)
