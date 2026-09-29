"""The digitization toolkit shipped into the model's code-execution sandbox.

Why: on a raster paper the model spent most of its tool loop rediscovering
the page layout the pre-pass already knew, probing the sandbox environment,
and writing a blob detector, text-row filter and axis calibrator from scratch
(Quinn 2015 under extraction_v9: 51 iterations, 7 scripts, still unfinished).
Every iteration re-reads the whole growing transcript, so cost grows roughly
with the square of the loop length — the cheapest extraction is the one with
the fewest turns. This module ships the repository's own, tested
`curve_extractor` package into the sandbox as a zip (`bundle()`) next to the
PDF, and describes it to the model in a per-run user-turn block (`guide()`)
injected like the curve pre-pass block — never folded into the pinned prompt.

The guide is the contract between this code and the prompt: it names the
functions the prompt's raster steps tell the model to call, and states the
sandbox facts the model otherwise probes for (checked live 2026-09-25: PyMuPDF
1.21, numpy, scipy, PIL, scikit-image and OpenCV work; pdfplumber does not
import; there is no internet, so nothing can be installed).
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent / "curve_extractor"
FILENAME = "curve_extractor.zip"


def bundle() -> bytes:
    """The curve_extractor package as a zip, byte-identical for identical
    sources (fixed timestamps, sorted entries) so `sha256()` names a version."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(_PACKAGE_DIR.glob("*.py")):
            info = zipfile.ZipInfo(f"curve_extractor/{path.name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    return buf.getvalue()


def sha256() -> str:
    return hashlib.sha256(bundle()).hexdigest()


def guide() -> str:
    """The user-turn block describing the toolkit and the sandbox to the model."""
    return f"""## SANDBOX TOOLKIT (use it — do not write your own digitiser)
Your code-execution environment holds `$INPUT_DIR/{FILENAME}`: this pipeline's own tested curve-
digitisation package. Load it once, in your first code run, together with the paper:
```bash
unzip -oq "$INPUT_DIR/{FILENAME}" -d /tmp/toolkit && ls "$INPUT_DIR"/*.pdf
```
```python
import sys; sys.path.insert(0, "/tmp/toolkit")
from curve_extractor import raster, legend, calibrate, fits
```
**Environment facts — do not probe or reinstall:** PyMuPDF (`import fitz`), numpy, scipy, PIL,
scikit-image and OpenCV are installed and work. `pdfplumber` does not import here; do not repair it —
render with PyMuPDF. There is no internet access, so nothing can be installed.
**You cannot see images you create here.** A render is pixels for your code only; opening a PNG
with the file viewer returns base64 text that every later step re-reads. Read legends, marker
shapes, panel layout, tick labels and in-plot text from the PDF document in this conversation.

**Render the whole figure** at 300 dpi — every pixel threshold below assumes that scale (the
DETERMINISTIC CURVE ANALYSIS block gives each raster figure's bbox in PDF points, origin top-left,
same convention as `fitz.Rect`):
```python
import fitz, numpy as np
doc = fitz.open(PDF_PATH)          # keep this name: the sandbox's PyMuPDF 1.21 orphans a page
page = doc[PAGE_INDEX]             # whose document was garbage-collected
pix = page.get_pixmap(dpi=300, clip=fitz.Rect(x0, top, x1, bottom), colorspace=fitz.csGRAY)
arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)   # 0 = ink
```

**Digitise a scanned figure** — all on that one array, pixel coordinates of it:
1. `panels = raster.find_panels(arr)` — every plot panel, `(x0, top, x1, bottom)`, reading order.
   Handles closed boxes, open L-shaped axes, gray scan lines; no need to slice the figure yourself.
2. `swatches = legend.find_swatches(arr, panels)` — the legend's marker swatches, reading order,
   each with `.description` (e.g. `"solid square"`, `"open diamond"`). Map each swatch to its legend
   label from the PDF document by that description; pass only the labelled ones on.
3. Per panel: `markers = legend.match_markers(arr, panel, labelled_swatches)` — every marker,
   already assigned to a swatch (`m.swatch` = index into the list you passed, `m.x`, `m.y`). It
   erases axis/grid lines and in-plot text itself before matching.
4. Calibrate each axis from its tick-label positions: read the labels' values off the figure
   (`calibrate.repair_signs(values)` restores minus signs a scan loses), then
   `pos = legend.label_centres(arr, panel, "x" | "y")`; if `len(pos) > len(values)`,
   `pos = legend.even_subset(pos, len(values))`; y positions pair with values bottom-up, i.e.
   `sorted(pos, reverse=True)`; then `cal = calibrate.fit_ticks(axis, pos, values)` (None = not
   calibratable; `cal.pixel_to_data(px)`). Fallback when labels can't be located:
   `calibrate.align_ticks(raster.tick_pixels(arr, panel, axis), values, lo_edge, hi_edge)` gives the
   pairs for `fit_ticks`.
5. On a log D vs pH (or log[extractant]) panel, each series is a straight line:
   `series, moved = fits.reassign(series_xy, fill_by_series)` hands a point to the same-fill series
   whose line it sits on, and `xy, dropped = fits.drop_off_line(xy)` removes points far off a
   series' own line (in-plot text, stray matches).
6. `legend.has_top_scale(arr, panel)` is True for a second x scale printed along the top (a
   mixture-composition plot) — not representable in the 26-column schema; skip that panel.

A figure whose legend has no drawn swatches: `raster.detect_markers_in_image(panel_crop)` finds blobs
by shape family instead (blank legend and in-plot text first, `arr[y0:y1, x0:x1] = 255`), with
`raster.find_frame` / `raster.tick_pixels` / `calibrate.fit_axis` for one-panel crops.

**Budget per raster figure: about three code runs** — (1) render, find panels and swatches, print
them; (2) match markers and calibrate every panel; (3) clean up, convert, assign series. Do not
re-implement any of the above, do not iterate on tolerances, and do not re-render to re-confirm.
"""
