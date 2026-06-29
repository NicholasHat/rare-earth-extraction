"""Tests for the deterministic curve pre-pass and its pipeline wiring."""
from pathlib import Path

import pandas as pd
import pytest

from extraction import anthropic_client
from extraction.curve_prepass import CurvePrepass, FigurePage
from validation import checks
from validation.report import Severity
from validation.schema import ELEMENT_COLUMN, coerce_schema

_SWAIN = Path("data/incoming/b5a26fd1b0a4575e614a7228ddc04c760ddfc556c57d2b3302ec1031116693d9.pdf")


# --- prompt block construction (pure) --------------------------------------- #

def test_prompt_block_empty_when_nothing_detected():
    assert CurvePrepass().to_prompt_block() == ""


def test_prompt_block_marks_authoritative_and_raster():
    p = CurvePrepass(
        confident_pages=[FigurePage(2, [19] * 9, confident=True)],
        unverified_pages=[FigurePage(4, [27, 26, 12], confident=False)],
        raster_pages=[0, 6],
    )
    block = p.to_prompt_block()
    assert "Page 2 (authoritative)" in block
    assert "9 distinct data series" in block and "19 digitised markers" in block
    assert "verify visually" in block
    assert "raster images" in block


def test_authoritative_counts_only_from_confident_pages():
    p = CurvePrepass(
        confident_pages=[FigurePage(2, [19, 19], confident=True)],
        unverified_pages=[FigurePage(4, [27, 26], confident=False)],
    )
    assert p.authoritative_counts == [19, 19]


# --- client injection (no API call) ----------------------------------------- #

def test_user_content_injects_analysis_block():
    content = anthropic_client._build_user_content("file_123", "## DETERMINISTIC CURVE ANALYSIS\nx")
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert any("DETERMINISTIC CURVE ANALYSIS" in t for t in texts)
    # document + container_upload always present
    assert {c["type"] for c in content} >= {"document", "container_upload", "text"}


def test_user_content_omits_block_when_none():
    content = anthropic_client._build_user_content("file_123", None)
    assert not any("DETERMINISTIC" in c.get("text", "") for c in content)


# --- QA cross-check ---------------------------------------------------------- #

def _df_with_counts(counts_by_element: dict[str, int]) -> pd.DataFrame:
    rows = []
    for el, n in counts_by_element.items():
        for i in range(n):
            rows.append({ELEMENT_COLUMN: el, "pH": 1.0 + i, "Extract%": 10.0 + i})
    return coerce_schema(pd.DataFrame(rows))


def test_deterministic_check_silent_when_counts_match():
    df = _df_with_counts({"La": 19, "Ce": 19, "Nd": 19})
    report = checks.run(df, deterministic_counts=[19, 19, 19])
    assert not any(f.check == "deterministic_curve_count" for f in report.flags)


def test_deterministic_check_flags_undercount():
    # geometry says three 19-point series; model produced one full + two short.
    df = _df_with_counts({"La": 19, "Ce": 10, "Nd": 9})
    report = checks.run(df, deterministic_counts=[19, 19, 19])
    flags = [f for f in report.flags if f.check == "deterministic_curve_count"]
    assert len(flags) == 1
    assert flags[0].severity is Severity.AMBER


def test_deterministic_check_noop_without_counts():
    df = _df_with_counts({"La": 5})
    report = checks.run(df, deterministic_counts=None)
    assert not any(f.check == "deterministic_curve_count" for f in report.flags)


# --- integration (real PDF, skipped if absent) ------------------------------ #

@pytest.mark.skipif(not _SWAIN.exists(), reason="Swain & Otu PDF not present")
def test_prepass_on_swain_marks_clean_figure_authoritative():
    from extraction.curve_prepass import analyze
    p = analyze(_SWAIN.read_bytes())
    # The clean single-panel colour figure (page 2) -> 9 series x 19, authoritative.
    assert any(fp.series_counts == [19] * 9 for fp in p.confident_pages)
    assert 19 in p.authoritative_counts
    assert "authoritative" in p.to_prompt_block()
