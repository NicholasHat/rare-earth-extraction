"""Tests for the deterministic curve pre-pass and its pipeline wiring."""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from extraction import anthropic_client
from extraction.curve_extractor import AxisCalibration, CurveExtractionResult, MarkerRecord
from extraction.curve_prepass import (
    CurvePrepass,
    FigurePage,
    _looks_panel_merged,
    analyze,
)
from validation import checks
from validation.report import Severity
from validation.schema import ELEMENT_COLUMN, coerce_schema

_OK_CAL = AxisCalibration(
    axis="x", model="linear", slope=1.0, intercept=0.0,
    residual_rms=0.01, r_squared=0.99, n_ticks=4, tick_values=[0, 1, 2, 3], ok=True,
)
_BAD_CAL = AxisCalibration(
    axis="y", model="linear", slope=1.0, intercept=0.0,
    residual_rms=5.0, r_squared=0.2, n_ticks=4, tick_values=[0, 1, 2, 3], ok=False,
)

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


def test_prompt_block_injects_calibrated_coordinates_for_authoritative_page():
    markers = [
        MarkerRecord("#ff0000", "filled", pixel_x=1.0, pixel_y=1.0, data_x=1.234567, data_y=10.1),
        MarkerRecord("#ff0000", "filled", pixel_x=2.0, pixel_y=2.0, data_x=2.345678, data_y=20.2),
        MarkerRecord("#00ff00", "filled", pixel_x=3.0, pixel_y=3.0, data_x=3.0, data_y=30.0),
    ]
    p = CurvePrepass(confident_pages=[FigurePage(2, [2, 1], confident=True, markers=markers)])
    block = p.to_prompt_block()
    assert "DIGITIZED CURVE DATA for page 2" in block
    start = block.index("`{") + 1
    end = block.index("}`", start) + 1
    points = json.loads(block[start:end])
    assert points == {"#ff0000": [[1.235, 10.1], [2.346, 20.2]], "#00ff00": [[3.0, 30.0]]}


def test_prompt_block_omits_coordinates_when_markers_absent():
    p = CurvePrepass(confident_pages=[FigurePage(2, [19] * 9, confident=True)])
    assert "DIGITIZED CURVE DATA" not in p.to_prompt_block()


def test_prompt_block_ignores_stroked_or_uncalibrated_markers():
    markers = [
        MarkerRecord("cross", "stroked", pixel_x=1.0, pixel_y=1.0, data_x=1.0, data_y=1.0),
        MarkerRecord("#ff0000", "filled", pixel_x=1.0, pixel_y=1.0, data_x=None, data_y=None),
    ]
    p = CurvePrepass(confident_pages=[FigurePage(2, [19] * 9, confident=True, markers=markers)])
    assert "DIGITIZED CURVE DATA" not in p.to_prompt_block()


def test_unverified_pages_never_get_coordinates_even_if_present():
    # Defensive: analyze() never sets markers on unverified pages, but the
    # prompt block should ignore them there regardless.
    markers = [MarkerRecord("#ff0000", "filled", pixel_x=1.0, pixel_y=1.0, data_x=1.0, data_y=1.0)]
    p = CurvePrepass(unverified_pages=[FigurePage(4, [27, 26], confident=False, markers=markers)])
    assert "DIGITIZED CURVE DATA" not in p.to_prompt_block()


def _fake_result(markers, x_cal, y_cal, counts):
    return CurveExtractionResult(
        source="vector", is_vector=True, markers=markers,
        x_calibration=x_cal, y_calibration=y_cal, per_group_counts=counts,
        page_index=0, figure_bbox=(0, 0, 1, 1), warnings=[],
    )


def test_analyze_withholds_coordinates_when_an_axis_calibration_is_untrusted():
    # Uniform counts (page-level "confident" gate passes) but the y-axis
    # calibration residual is too high — extractor.py still writes data_x/
    # data_y in this case, so analyze() must gate on AxisCalibration.ok too.
    markers = [
        MarkerRecord("#ff0000" if i < 15 else "#00ff00", "filled",
                     pixel_x=float(i), pixel_y=float(i), data_x=float(i), data_y=float(i))
        for i in range(30)
    ]
    counts = {"#ff0000": 15, "#00ff00": 15}
    fake = _fake_result(markers, _OK_CAL, _BAD_CAL, counts)
    with patch("extraction.curve_prepass.extract_curves", return_value=fake):
        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value.pages = [None]
            p = analyze(b"fake pdf bytes")
    assert len(p.confident_pages) == 1
    assert p.confident_pages[0].markers is None
    assert "DIGITIZED CURVE DATA" not in p.to_prompt_block()


def test_analyze_carries_coordinates_when_both_axes_trusted():
    markers = [
        MarkerRecord("#ff0000" if i < 15 else "#00ff00", "filled",
                     pixel_x=float(i), pixel_y=float(i), data_x=float(i), data_y=float(i))
        for i in range(30)
    ]
    counts = {"#ff0000": 15, "#00ff00": 15}
    fake = _fake_result(markers, _OK_CAL, _OK_CAL, counts)
    with patch("extraction.curve_prepass.extract_curves", return_value=fake):
        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value.pages = [None]
            p = analyze(b"fake pdf bytes")
    assert p.confident_pages[0].markers is not None
    assert "DIGITIZED CURVE DATA" in p.to_prompt_block()


# --- panel-merge gate (pure) ------------------------------------------------- #

def _run(group_key: str, xs: list[float]) -> list[MarkerRecord]:
    return [MarkerRecord(group_key, "filled", pixel_x=x, pixel_y=50.0) for x in xs]


def test_panel_merge_flags_balanced_split_groups():
    # Two colour groups, each the same 15-point run drawn twice, offset by an
    # inter-panel gutter — the Swain & Otu Fig. 6 shape.
    one_panel = [float(x) for x in range(0, 75, 5)]
    two_panels = one_panel + [x + 120.0 for x in one_panel]
    markers = _run("#ff0000", two_panels) + _run("#00ff00", two_panels)
    assert _looks_panel_merged(markers) is True


def test_panel_merge_ignores_unbalanced_data_gap():
    # A real single-panel series with a sparse high-x tail (Swain Fig. 2 shape):
    # the biggest gap is large but splits 18|1, not ~evenly.
    xs = [float(x) for x in range(0, 90, 5)] + [200.0]
    markers = _run("#ff0000", xs) + _run("#00ff00", xs)
    assert _looks_panel_merged(markers) is False


def test_panel_merge_needs_more_than_one_suspect_group():
    one_panel = [float(x) for x in range(0, 75, 5)]
    split = one_panel + [x + 120.0 for x in one_panel]
    clean = [float(x) for x in range(0, 150, 5)]
    markers = _run("#ff0000", split) + _run("#00ff00", clean) + _run("#0000ff", clean)
    assert _looks_panel_merged(markers) is False


def test_analyze_downgrades_panel_merged_page():
    # Uniform counts (the old gate passes) but both groups are panel-split —
    # the page must land in unverified_pages with no authoritative counts.
    one_panel = [float(x) for x in range(0, 75, 5)]
    two_panels = one_panel + [x + 120.0 for x in one_panel]
    markers = _run("#ff0000", two_panels) + _run("#00ff00", two_panels)
    counts = {"#ff0000": 30, "#00ff00": 30}
    fake = _fake_result(markers, _OK_CAL, _OK_CAL, counts)
    with patch("extraction.curve_prepass.extract_curves", return_value=fake):
        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value.pages = [None]
            p = analyze(b"fake pdf bytes")
    assert p.confident_pages == []
    assert len(p.unverified_pages) == 1
    assert p.authoritative_counts == []


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


@pytest.mark.skipif(not _SWAIN.exists(), reason="Swain & Otu PDF not present")
def test_prepass_on_swain_downgrades_two_panel_fig6_page():
    # Page 5 holds Fig. 6, a two-panel figure whose same-colour curves merge
    # into inflated counts ([30, 30, 29, 26] vs the true ~15/panel). Uniformity
    # alone passed it; the panel-merge gate must keep it out of authoritative.
    from extraction.curve_prepass import analyze
    p = analyze(_SWAIN.read_bytes())
    assert all(fp.page_index != 5 for fp in p.confident_pages)
    assert any(fp.page_index == 5 for fp in p.unverified_pages)
    assert 30 not in p.authoritative_counts
