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
from curve_extractor import raster, calibrate
```
**Environment facts — do not probe or reinstall:** PyMuPDF (`import fitz`), numpy, scipy, PIL,
scikit-image and OpenCV are installed and work. `pdfplumber` does not import here; do not repair it —
render with PyMuPDF. There is no internet access, so nothing can be installed.
**You cannot see images you create here.** A render is pixels for your code only; opening a PNG
with the file viewer returns base64 text that every later step re-reads. Read legends, marker
shapes, panel layout and in-plot text from the PDF document in this conversation.

**Render a figure region** (the DETERMINISTIC CURVE ANALYSIS block gives each raster page's figure
bbox in PDF points, origin top-left, same convention as `fitz.Rect`):
```python
import fitz, numpy as np
page = fitz.open(PDF_PATH)[PAGE_INDEX]
pix = page.get_pixmap(dpi=300, clip=fitz.Rect(x0, top, x1, bottom), colorspace=fitz.csGRAY)
arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)   # 0 = ink
```
For a multi-panel figure, slice `arr` into one array per panel (you know the layout from looking at
the page) and run everything below per panel; pixel coordinates are then relative to that slice.

**Toolkit API** (all on a grayscale array; pixel coordinates of that array):
- `raster.find_frame(arr) -> (x0, top, x1, bottom) | None` — the plot frame of a single-panel region.
- `raster.tick_pixels(arr, frame, "x" | "y") -> [px, ...]` — tick-mark centres along the bottom / left
  frame edge. Pair with the tick labels you read off the figure; check the count matches the labels
  (drop unlabelled minor ticks or a marker sitting on the axis line).
- `calibrate.fit_axis("x", tick_pixels, tick_values) -> cal` — least-squares pixel→data fit, linear or
  log10 chosen automatically; `cal.pixel_to_data(px)`, `cal.model`, `cal.ok` (residual within 2 % of
  span). Do the same for `"y"`.
- `raster.detect_markers_in_image(arr) -> (records, warnings)` — line suppression, blob detection,
  marker-shape filter, text-row removal, shape classification. Each record has `group_key`
  (`filled_square` / `filled_circle` / `filled_triangle` / `filled_diamond` / `stroked_glyph` /
  `ambiguous`), `marker_type`, `pixel_x`, `pixel_y`. **Blank the legend and every in-plot text region
  first** (`arr[y0:y1, x0:x1] = 255`) — the toolkit does not know where they are; you do, from the
  PDF document. Map shape family → element from the legend, as always.

**Budget per raster figure: about three code runs** — (1) render the region and print its size and
frame, (2) blank text regions + detect markers + ticks, (3) calibrate, convert, assign series. Do not re-implement any of the
above, do not iterate on clustering tolerances, and do not re-render a panel to re-confirm a count.
"""
