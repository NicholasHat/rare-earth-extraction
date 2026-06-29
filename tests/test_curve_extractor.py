"""Tests for the deterministic curve extractor.

The pure-logic tests (fit, classify, clustering, eps-invariance) run anywhere.
The integration test runs only if the Swain & Otu PDF is present in data/incoming
— it locks in the real validation (9 colour series × 19 markers, the count our
LLM runs under-counted).
"""
import math
from pathlib import Path

import numpy as np
import pytest

from extraction.curve_extractor import calibrate, markers, raster
from extraction.curve_extractor.types import AxisCalibration


# --- calibrate.fit_axis ----------------------------------------------------- #

def test_fit_axis_linear_recovers_mapping():
    # data = 0.01*pixel + 0.5
    pixels = [100, 200, 300, 400]
    values = [1.5, 2.5, 3.5, 4.5]
    cal = calibrate.fit_axis("x", pixels, values)
    assert cal.model == "linear"
    assert cal.ok
    assert cal.pixel_to_data(250) == pytest.approx(3.0, abs=1e-6)


def test_fit_axis_detects_log_axis():
    # values 0.05..1.0 spaced logarithmically vs pixel -> log10 chosen
    pixels = [0, 100, 200, 300]
    values = [0.05, 0.1414, 0.4, 1.131]  # ~ geometric-ish
    cal = calibrate.fit_axis("x", pixels, values)
    assert cal.model == "log10"


def test_fit_axis_flags_bad_residual():
    pixels = [0, 1, 2, 3, 4]
    values = [0, 1, 2, 9, 4]  # one wild outlier -> high residual
    cal = calibrate.fit_axis("x", pixels, values)
    assert not cal.ok


def test_fit_axis_needs_two_ticks():
    with pytest.raises(ValueError):
        calibrate.fit_axis("x", [100], [1.0])


def test_axis_calibration_pixel_to_data_log():
    cal = AxisCalibration("x", "log10", slope=0.01, intercept=-1.0,
                          residual_rms=0.0, r_squared=1.0, n_ticks=3,
                          tick_values=[0.1, 1.0], ok=True)
    assert cal.pixel_to_data(100) == pytest.approx(1.0)   # 10^(0.01*100-1)=10^0=1
    assert cal.pixel_to_data(0) == pytest.approx(0.1)     # 10^-1


# --- markers: classify / eps / clustering ----------------------------------- #

def test_classify_marker_type():
    assert markers.classify_marker_type({"fill": True}) == "filled"
    assert markers.classify_marker_type({"fill": False}) == "stroked"


def _filled(cx, cy, w=5.0, h=6.0, colour=(1.0, 0.0, 0.0)):
    return {"x0": cx - w / 2, "x1": cx + w / 2, "top": cy - h / 2, "bottom": cy + h / 2,
            "width": w, "height": h, "fill": True, "non_stroking_color": colour}


def test_assemble_does_not_merge_close_distinct_markers():
    # Two distinct markers 1.4px apart (dense-zone spacing seen in real data).
    objs = [_filled(100, 100), _filled(101.4, 100)]
    recs = markers.assemble_filled("#ff0000", objs)
    assert len(recs) == 2  # the under-count bug would merge these


def test_assemble_dedupes_coincident_paths():
    # Same marker drawn twice at ~0px apart (outline+fill) -> one record.
    objs = [_filled(100, 100), _filled(100.1, 100.05)]
    recs = markers.assemble_filled("#ff0000", objs)
    assert len(recs) == 1


def test_eps_is_density_invariant():
    # eps depends only on marker geometry, not how far apart markers are.
    sparse = [_filled(0, 0), _filled(50, 0)]
    dense = [_filled(0, 0), _filled(2, 0)]
    assert markers._calibrate_eps_filled(sparse) == markers._calibrate_eps_filled(dense)


def test_detect_merge_warnings_flags_low_outlier():
    counts = {"#a": 19, "#b": 19, "#c": 19, "#d": 6}
    warns = markers.detect_merge_warnings([], counts)
    assert any("#d" in w for w in warns)


def test_detect_merge_warnings_silent_when_uniform():
    counts = {"#a": 19, "#b": 19, "#c": 19}
    assert markers.detect_merge_warnings([], counts) == []


# --- raster shape classification -------------------------------------------- #

def test_classify_blob_shape_filled_vs_stroked():
    filled = {"fill_ratio": 0.85, "bbox_w": 12, "bbox_h": 12}
    stroked = {"fill_ratio": 0.30, "bbox_w": 12, "bbox_h": 12}
    assert raster.classify_blob_shape(filled)[0] == "filled"
    assert raster.classify_blob_shape(stroked)[0] == "stroked"


# --- integration (real PDF, skipped if absent) ------------------------------ #

_SWAIN = Path("data/incoming/b5a26fd1b0a4575e614a7228ddc04c760ddfc556c57d2b3302ec1031116693d9.pdf")


@pytest.mark.skipif(not _SWAIN.exists(), reason="Swain & Otu PDF not present")
def test_vector_path_recovers_uniform_marker_counts():
    from extraction.curve_extractor import extract_curves
    result = extract_curves(_SWAIN.read_bytes(), 2)
    assert result.is_vector
    counts = sorted(result.per_group_counts.values(), reverse=True)
    # 9 colour series, each a full 19-point curve (the LLM under-counted these).
    assert counts == [19] * 9
