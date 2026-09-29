"""The raster tools shipped in the sandbox toolkit beyond frame/tick
detection: whole-figure panel finding, legend-swatch template matching,
tick-label positions and pairing, and the log D line clean-up. Most cases are
ported from ree-extraction-local's tests; synthetic images, no PDFs."""
import numpy as np
import pytest

from extraction.curve_extractor import calibrate, fits, legend, raster


# --- panels ---------------------------------------------------------------- #

def _panel(img, x0, top, x1, bottom):
    img[top:bottom + 1, x0:x0 + 2] = 0          # left axis
    img[bottom - 1:bottom + 1, x0:x1 + 1] = 0   # bottom axis


def test_find_panels_returns_every_panel_of_a_whole_figure_in_reading_order():
    img = np.full((900, 1400), 255, dtype=np.uint8)
    boxes = [(100, 50, 650, 400), (800, 50, 1350, 400), (100, 500, 650, 850)]
    for b in boxes:
        _panel(img, *b)
    assert raster.find_panels(img) == boxes


# --- legend swatches -> markers --------------------------------------------- #

def _square(img, cx, cy, half=8):
    img[cy - half:cy + half, cx - half:cx + half] = 0


def _ring(img, cx, cy, r=9, w=2):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    d = (yy - cy) ** 2 + (xx - cx) ** 2
    img[(d <= r * r) & (d >= (r - w) ** 2)] = 0


def test_markers_are_matched_to_the_legend_swatch_they_look_like():
    img = np.full((700, 900), 255, dtype=np.uint8)
    panel = (200, 150, 850, 600)
    _panel(img, *panel)
    _square(img, 40, 40)                         # legend: solid square ...
    _ring(img, 40, 90)                           # ... and open circle, left of the plot
    for i, x in enumerate(range(300, 800, 100)):
        _square(img, x, 500 - 50 * i)
        _ring(img, x + 20, 300 + 30 * i)
    swatches = legend.find_swatches(img, [panel])
    assert [s.description for s in swatches] == ["solid square", "open circle"]
    markers = legend.match_markers(img, panel, swatches)
    by_swatch = {i: sum(1 for m in markers if m.swatch == i) for i in range(2)}
    assert by_swatch == {0: 5, 1: 5}


@pytest.mark.parametrize("shape", ["square", "circle", "triangle", "diamond"])
def test_classify_clean_shapes(shape):
    n = 21
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2
    mask = {
        "square": np.ones((n, n), bool),
        "circle": (xx - c) ** 2 + (yy - c) ** 2 <= c ** 2,
        "triangle": np.abs(xx - c) <= yy / 2,
        "diamond": np.abs(xx - c) + np.abs(yy - c) <= c,
    }[shape]
    assert legend.classify(mask) == shape


def test_even_subset_drops_stray_row():
    rows = [58, 108, 159, 211, 260, 311, 362, 389, 412, 462]
    assert legend.even_subset(rows, 9) == [58, 108, 159, 211, 260, 311, 362, 412, 462]


# --- tick labels ------------------------------------------------------------- #

@pytest.mark.parametrize("labels, expected", [
    ([0.8, 0.6, 0.4, 0.2, 0.0], [-0.8, -0.6, -0.4, -0.2, 0.0]),    # all minus signs lost
    ([2.0, 1.5, 1.0, 0.5, 0.0, 0.5, 1.0], [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]),
    ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]),                             # already fine
])
def test_repair_signs(labels, expected):
    assert calibrate.repair_signs(labels) == expected


def test_align_ticks_adds_missing_corner_ticks():
    pairs = calibrate.align_ticks([100, 200, 300, 400, 500], [0, 0.5, 1, 1.5, 2, 2.5, 3], 0, 600)
    assert [p for p, _ in pairs] == [0, 100, 200, 300, 400, 500, 600]


def test_align_ticks_skips_minor_ticks():
    pairs = calibrate.align_ticks(list(range(50, 600, 50)), [1, 2, 3, 4, 5], 0, 600)
    assert [p for p, _ in pairs] == [100, 200, 300, 400, 500]


# --- log D line clean-up --------------------------------------------------- #

def test_drop_off_line_drops_a_text_artifact():
    xy = [(0.6, -1.6), (0.95, -0.35), (1.02, -0.05), (1.3, 0.75), (1.6, 1.7), (0.3, 1.8)]
    kept, dropped = fits.drop_off_line(xy)
    assert dropped == 1 and (0.3, 1.8) not in kept


def test_reassign_moves_a_point_onto_the_neighbouring_line_it_sits_on():
    squares = [(0.6, -1.6), (0.95, -0.35), (1.3, 0.75), (1.6, 1.7), (1.52, 0.35)]   # last is a circle
    circles = [(0.95, -1.45), (1.28, -0.45), (1.78, 1.15), (2.05, 1.95)]
    fixed, moved = fits.reassign({"sq": squares, "ci": circles}, {"sq": "solid", "ci": "solid"})
    assert moved == 1 and (1.52, 0.35) in fixed["ci"]
