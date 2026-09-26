"""Tests for the deterministic curve extractor.

The pure-logic tests (fit, classify, clustering, eps-invariance) run anywhere.
The integration test runs only if the Swain & Otu PDF is present in data/incoming
— it locks in the real validation (9 colour series × 19 markers, the count our
LLM runs under-counted).
"""
import math
from pathlib import Path
from unittest.mock import patch

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


# --- markers: stroked glyph assembly (×/+/✶) --------------------------------- #

def _seg(x0, top, x1, bottom):
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom}


def _plus_at(cx, cy, arm=3.0):
    # horizontal + vertical fragment through (cx, cy)
    return [_seg(cx - arm, cy, cx + arm, cy), _seg(cx, cy - arm, cx, cy + arm)]


def _cross_at(cx, cy, arm=3.0):
    # two diagonals through (cx, cy)
    return [_seg(cx - arm, cy - arm, cx + arm, cy + arm),
            _seg(cx - arm, cy + arm, cx + arm, cy - arm)]


def _star_at(cx, cy, arm=3.0):
    return _cross_at(cx, cy, arm) + [_seg(cx - arm, cy, cx + arm, cy)]


def test_classify_glyph_shape_distinguishes_plus_cross_star():
    assert markers.classify_glyph_shape(_plus_at(0, 0)) == "plus"
    assert markers.classify_glyph_shape(_cross_at(0, 0)) == "cross"
    assert markers.classify_glyph_shape(_star_at(0, 0)) == "star"


def test_assemble_stroked_recovers_glyphs_and_shapes():
    segs = _plus_at(10, 10) + _cross_at(40, 10) + _plus_at(70, 10)
    recs = markers.assemble_stroked(segs)
    assert len(recs) == 3
    groups = markers.group_stroked_by_shape(recs)
    assert sorted(groups) == ["cross", "plus"]
    assert len(groups["plus"]) == 2
    assert len(groups["cross"]) == 1
    assert all(r.marker_type == "stroked" for r in recs)


def test_assemble_stroked_does_not_merge_close_distinct_glyphs():
    # Two distinct + glyphs whose arms nearly touch — same dense-zone concern
    # as the filled path, but here eps is arm-length-derived, not spacing-derived.
    segs = _plus_at(0, 0, arm=3.0) + _plus_at(6.5, 0, arm=3.0)
    recs = markers.assemble_stroked(segs)
    assert len(recs) == 2


def test_assemble_stroked_eps_is_density_invariant():
    sparse = _plus_at(0, 0) + _plus_at(50, 0)
    dense = _plus_at(0, 0) + _plus_at(2, 0)
    assert markers._calibrate_eps_stroked(sparse) == markers._calibrate_eps_stroked(dense)


def test_assemble_stroked_drops_unmatched_fragment_as_noise():
    # One clean plus glyph plus a lone stray fragment (e.g. a partial gridline
    # that slipped past length filtering) with no partner nearby.
    segs = _plus_at(0, 0) + [_seg(200, 200, 203, 200)]
    recs = markers.assemble_stroked(segs)
    assert len(recs) == 1


def test_assemble_stroked_empty_input():
    assert markers.assemble_stroked([]) == []


def test_classify_glyph_shape_rejects_non_glyph_fragments():
    # Two near-parallel vertical fragments (e.g. a coincidental crossing of two
    # unrelated axis ticks) don't form any recognized glyph.
    parallel = [_seg(10, 0, 10, 6), _seg(11, 0.5, 11, 6.5)]
    assert markers.classify_glyph_shape(parallel) is None
    assert markers.classify_glyph_shape([_seg(0, 0, 5, 0)]) is None  # single fragment
    assert markers.classify_glyph_shape([_seg(0, 0, 5, 0)] * 4) is None  # 4 fragments


def test_assemble_stroked_drops_clusters_with_no_recognized_shape():
    # Two parallel fragments cluster together (within eps) but classify to
    # None — must be dropped, not mislabelled into a catch-all group.
    segs = [_seg(10, 0, 15, 0), _seg(10.5, 0.5, 15.5, 0.5)]
    assert markers.assemble_stroked(segs) == []


# --- detect: stroked-segment collection excludes ticks/legend/long lines ---- #

class _FakePage:
    def __init__(self, lines):
        self.lines = lines


def test_collect_stroked_segments_excludes_ticks_outside_frame():
    from extraction.curve_extractor import detect
    frame = (0, 0, 100, 100)
    inside = _seg(40, 40, 46, 40)          # short, inside frame -> glyph fragment
    tick = _seg(-4, 50, 0, 50)             # short, outside frame -> axis tick
    long_line = _seg(0, 0, 90, 90)         # long, inside frame -> connecting curve
    page = _FakePage([inside, tick, long_line])
    out = detect.collect_stroked_segments(page, frame)
    assert out == [inside]


# --- raster shape classification -------------------------------------------- #

def test_classify_blob_shape_filled_vs_stroked():
    filled = {"fill_ratio": 0.85, "bbox_w": 12, "bbox_h": 12}
    stroked = {"fill_ratio": 0.30, "bbox_w": 12, "bbox_h": 12}
    assert raster.classify_blob_shape(filled)[0] == "filled"
    assert raster.classify_blob_shape(stroked)[0] == "stroked"


def test_is_marker_shaped_rejects_lines_and_oversized():
    assert raster._is_marker_shaped({"bbox_w": 10, "bbox_h": 11})       # square-ish marker
    assert not raster._is_marker_shaped({"bbox_w": 4, "bbox_h": 40})    # line fragment
    assert not raster._is_marker_shaped({"bbox_w": 40, "bbox_h": 40})   # oversized merge
    assert not raster._is_marker_shaped({"bbox_w": 3, "bbox_h": 3})     # speckle


def test_remove_text_rows_drops_dense_row_keeps_sparse_curve():
    # A dense, tightly-spaced horizontal run (axis-label/legend text).
    text = [{"cx": 100 + 20 * i, "cy": 500.0} for i in range(10)]
    # A few well-separated markers along a flatter curve stretch — must survive.
    curve = [{"cx": 100 + 120 * i, "cy": 200.0} for i in range(4)]
    kept, n_text = raster._remove_text_rows(text + curve)
    assert n_text == len(text)
    assert all(b in kept for b in curve)
    assert not any(b in kept for b in text)


def test_remove_text_rows_noop_on_scattered_markers():
    blobs = [{"cx": 10.0 * i, "cy": 10.0 * i} for i in range(12)]  # diagonal, no row
    kept, n_text = raster._remove_text_rows(blobs)
    assert n_text == 0 and len(kept) == len(blobs)


# --- raster shape families + template recovery (synthetic images) ------------ #

def _draw_square(img, cx, cy, half=6):
    img[cy - half:cy + half, cx - half:cx + half] = 0


def _draw_disk(img, cx, cy, r=7):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 0


def _draw_triangle(img, cx, cy, half=7):
    for dy in range(2 * half):           # apex up: row width grows with dy
        w = int(half * dy / (2 * half - 1))
        img[cy - half + dy, cx - w:cx + w + 1] = 0


def test_filled_shape_families_get_distinct_group_keys():
    img = np.full((200, 400), 255, dtype=np.uint8)
    for i in range(5):
        _draw_square(img, 40 + 70 * i, 40)
        _draw_disk(img, 40 + 70 * i, 100)
        _draw_triangle(img, 40 + 70 * i, 160)
    blobs = raster.detect_blobs(img)
    families = {}
    for b in blobs:
        mtype, fam = raster.classify_blob_shape(b)
        assert mtype == "filled"
        families[fam] = families.get(fam, 0) + 1
    assert families == {"filled_square": 5, "filled_circle": 5, "filled_triangle": 5}


def test_template_recovery_finds_marker_lost_by_blob_filtering():
    # Six squares; blob detection "keeps" only five (simulating one lost to an
    # overlap merge). The recovery pass must find the sixth from an exemplar
    # template — and must not duplicate the five already-kept ones.
    img = np.full((120, 500), 255, dtype=np.uint8)
    centers = [(40 + 75 * i, 60) for i in range(6)]
    for cx, cy in centers:
        _draw_square(img, cx, cy)
    blobs = raster.detect_blobs(img)
    assert len(blobs) == 6
    lost = min(blobs, key=lambda b: b["cx"])
    kept = [b for b in blobs if b is not lost]
    recovered, _ = raster._recover_missed_markers(raster._ink_mask(img), kept)
    assert len(recovered) == 1
    family, cx, cy = recovered[0]
    assert family == "filled_square"
    assert abs(cx - lost["cx"]) <= 2 and abs(cy - lost["cy"]) <= 2


def test_template_recovery_needs_enough_exemplars():
    img = np.full((100, 300), 255, dtype=np.uint8)
    for i in range(3):
        _draw_square(img, 40 + 70 * i, 50)
    blobs = raster.detect_blobs(img)
    # Only 3 clean detections (< _TEMPLATE_MIN_EXEMPLARS) — no recovery attempted.
    assert raster._recover_missed_markers(raster._ink_mask(img), blobs) == ([], [])


# --- integration (real PDF, skipped if absent) ------------------------------ #

_SWAIN = Path("data/incoming/b5a26fd1b0a4575e614a7228ddc04c760ddfc556c57d2b3302ec1031116693d9.pdf")


@pytest.mark.skipif(not _SWAIN.exists(), reason="Swain & Otu PDF not present")
def test_vector_path_recovers_uniform_marker_counts():
    from extraction.curve_extractor import extract_curves
    result = extract_curves(_SWAIN.read_bytes(), 2)
    assert result.is_vector
    filled_counts = sorted(
        (n for k, n in result.per_group_counts.items()
         if any(m.group_key == k and m.marker_type == "filled" for m in result.markers)),
        reverse=True,
    )
    # 9 colour series, each a full 19-point curve (the LLM under-counted these).
    # This figure's legend actually has 14 elements — the rest (e.g. Nd = "+")
    # are monochrome stroked glyphs assemble_stroked now also partially recovers,
    # but that path has no oracle count yet (plan §5.2), so it's exercised here
    # only informally, not asserted on — same caution curve_prepass.py applies.
    assert filled_counts == [19] * 9


# --- raster image-level entry points (what the sandbox toolkit exposes) ------ #

def _figure(w=400, h=300, frame=(40, 20, 380, 260), ticks_x=(80, 160, 240, 320), ticks_y=(60, 120, 180, 240)):
    """A synthetic single-panel figure: white page, 2px frame, 8px ticks just
    inside the frame, no markers. Callers draw markers on top."""
    img = np.full((h, w), 255, dtype=np.uint8)
    x0, top, x1, bottom = frame
    img[top:top + 2, x0:x1 + 1] = 0
    img[bottom - 1:bottom + 1, x0:x1 + 1] = 0
    img[top:bottom + 1, x0:x0 + 2] = 0
    img[top:bottom + 1, x1 - 1:x1 + 1] = 0
    for tx in ticks_x:
        img[bottom - 9:bottom - 1, tx:tx + 2] = 0        # 8 px tall, 2 px wide, inside
    for ty in ticks_y:
        img[ty:ty + 2, x0 + 2:x0 + 10] = 0              # 8 px long, inside
    return img


def _square(img, cx, cy, side=12):
    img[cy - side // 2:cy + side // 2, cx - side // 2:cx + side // 2] = 0


def _circle(img, cx, cy, r=6):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 0


def test_find_frame_locates_the_plot_box():
    assert raster.find_frame(_figure()) == (40, 20, 380, 260)


def test_find_frame_none_without_frame_lines():
    assert raster.find_frame(np.full((100, 100), 255, dtype=np.uint8)) is None


def test_tick_pixels_finds_inside_ticks_on_both_axes():
    img = _figure()
    frame = raster.find_frame(img)
    assert raster.tick_pixels(img, frame, "x") == pytest.approx([80.5, 160.5, 240.5, 320.5])
    assert raster.tick_pixels(img, frame, "y") == pytest.approx([60.5, 120.5, 180.5, 240.5])


def test_tick_pixels_ignores_a_marker_sitting_on_the_axis():
    img = _figure()
    _square(img, 200, 255)                              # a 12 px marker touching the bottom edge
    frame = raster.find_frame(img)
    assert raster.tick_pixels(img, frame, "x") == pytest.approx([80.5, 160.5, 240.5, 320.5])


def test_tick_pixels_falls_back_to_outside_ticks():
    img = _figure(ticks_x=(), ticks_y=())
    for tx in (100, 200, 300):
        img[262:270, tx:tx + 2] = 0                     # below the bottom line
    frame = raster.find_frame(img)
    assert raster.tick_pixels(img, frame, "x") == pytest.approx([100.5, 200.5, 300.5])


def test_find_frame_reads_jpeg_gray_axis_lines():
    # Quinn et al. 2015's scanned axis lines are gray (~145), above the marker
    # ink threshold; the frame used to vanish.
    img = _figure()
    img[img == 0] = 145
    assert raster.find_frame(img) == (40, 20, 380, 260)


def test_find_frame_accepts_open_l_shaped_axes():
    img = np.full((300, 400), 255, dtype=np.uint8)
    img[20:261, 40:42] = 0                              # left axis only
    img[259:261, 40:381] = 0                            # bottom axis only
    assert raster.find_frame(img) == (40, 20, 380, 260)


def test_find_frame_ignores_a_neighbouring_panel_caught_in_the_crop():
    img = np.full((400, 400), 255, dtype=np.uint8)
    frame = _figure()
    img[:300] = frame
    img[330:332, 40:381] = 0                            # next panel's top edge, below the tick labels
    img[330:400, 40:42] = 0
    img[330:400, 379:381] = 0
    assert raster.find_frame(img) == (40, 20, 380, 260)


def test_find_frame_bridges_an_axis_corner_blanked_by_hollow_markers():
    img = _figure()
    img[255:265, 42:90] = 255                           # white-filled markers over the corner
    assert raster.find_frame(img) == (40, 20, 380, 260)


def test_find_frame_is_not_fooled_by_dotted_gridlines():
    img = _figure()
    img[140:142, 42:379:6] = 0                          # dotted horizontal gridline
    img[140:142, 43:379:6] = 0
    assert raster.find_frame(img) == (40, 20, 380, 260)


def test_tick_pixels_ignores_a_dotted_gridline_lying_along_the_axis():
    img = _figure(ticks_x=(), ticks_y=())
    img[255:258, 44:378:6] = 0                          # gridline dots touching the bottom line
    for tx in (100, 200, 300):
        img[261:271, tx:tx + 2] = 0                     # the real ticks, outside
    assert raster.tick_pixels(img, raster.find_frame(img), "x") == pytest.approx([100.5, 200.5, 300.5])


def test_detect_markers_in_image_groups_by_shape_family():
    img = _figure()
    for i, cx in enumerate(range(80, 340, 40)):
        _square(img, cx, 100)
        _circle(img, cx, 180)
    records, warnings = raster.detect_markers_in_image(img)
    by_family = {}
    for r in records:
        by_family.setdefault(r.group_key, []).append(r)
    assert len(by_family["filled_square"]) == 7
    assert len(by_family["filled_circle"]) == 7
    assert all(abs(r.pixel_y - 100) < 1.5 for r in by_family["filled_square"])
    assert all(abs(r.pixel_y - 180) < 1.5 for r in by_family["filled_circle"])
    assert any("ESTIMATE" in w for w in warnings)


def test_detect_markers_in_image_without_scikit_image_still_detects(monkeypatch):
    def _missing():
        raise ImportError("No module named 'skimage'")
    monkeypatch.setattr(raster, "_template_tools", _missing)
    img = _figure()
    for cx in range(80, 340, 40):
        _square(img, cx, 100)
    records, warnings = raster.detect_markers_in_image(img)
    assert len(records) == 7
    assert any("scikit-image" in w for w in warnings)


def test_detect_markers_matches_the_page_path():
    """detect_markers(page, bbox) is render + detect_markers_in_image — same result."""
    img = _figure()
    _square(img, 100, 100)
    with patch.object(raster, "render_region", return_value=img):
        assert raster.detect_markers(object(), (0, 0, 1, 1)) == raster.detect_markers_in_image(img)
