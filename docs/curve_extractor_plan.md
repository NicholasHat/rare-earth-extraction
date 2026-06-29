# Development Plan — Deterministic PDF Curve Extractor

**Status:** v1 built in `extraction/curve_extractor/` (101 tests pass). Not yet wired into
the live pipeline (no `extraction_v6` prompt / runner pre-pass yet — that's the remaining M5).
**Goal:** replace LLM-guessed marker clustering / axis calibration with deterministic,
unit-tested Python, leaving the LLM only the genuinely-visual and schema-formatting work.

## Implementation status & findings from real PDFs

Two validation papers probed with `pdfplumber`, revealing a fork the plan only half-anticipated:

- **Swain & Otu 2011** — figures are **vector**, colour-coded. The vector path **works and proves
  the thesis**: it recovers **9 colour series × 19 markers each**, uniform — the exact curves our
  LLM runs under-counted at 6–19. `eps` had to be tuned to real geometry: distinct dense-zone
  markers sit as close as **1.34px**, so the dedup `eps` is ~0.8px (only true coincident paths
  merge). This is locked in by an integration test.
- **Quinn et al. 2015** (Elsevier, same publisher) — **every figure is an embedded raster image**
  (200 DPI grayscale), so the vector path cannot touch it; it routes to the **raster CV path**.
  That path is a **first cut**: it renders + suppresses connecting lines + finds blobs, but the
  count is still **contaminated by baked-in figure text** (axis numbers, legend, panel titles are
  pixels, not PDF chars) — 826 blobs vs ~280 true markers. It self-flags low confidence. **To
  reach vector-parity it needs panel-interior segmentation + text exclusion** (see §8) — not yet built.
- **Calibration:** `fit_axis` (linear/log auto-select) is done and unit-tested. Auto-reading tick
  *values* from chars is brittle on multi-panel figures, so it gracefully returns `calibration=None`
  + a warning and relies on the LLM-supplied tick seam (§3) — as the plan intended.

**Built:** `types.py`, `calibrate.py`, `detect.py`, `markers.py` (vector), `raster.py` (raster),
`extractor.py` (auto-routing on `is_vector`). **Remaining:** raster panel/text segmentation; the
M5 pipeline integration (`extraction_v6` + runner pre-pass).

---

## 1. Problem being fixed (diagnosed)

The LLM, when asked to digitize a figure, must pick a **clustering tolerance** (how close
two detected fragments must be to count as one data marker). It picks **one global value**
and applies it across:

- **mixed marker geometries** — filled shapes (circle/triangle/diamond, one fill path each)
  vs. stroked glyphs (×/+/*, built from 2–3 line segments that must be *assembled* into one
  marker), and
- **regions of very different density** — a sparse high-pH plateau vs. a dense low-pH
  transition zone where 5–7 markers sit inside a ~0.3 pH window.

One tolerance cannot be right for both: large enough to assemble a `×`'s two strokes will
also merge two adjacent circles in the dense zone; small enough to keep those circles
separate will fail to assemble the `×`. Documented result on Swain & Otu 2011 Fig. 2: 6 of
14 element-curves under-counted by 2–6 points (the lighter lanthanides La/Ce/Pr/Nd in the
dense transition region), 2 over-counted.

**Key correction to the originally-proposed fix.** The instinct was "derive `eps` from each
element's nearest-neighbour point spacing." That is still density-sensitive — in a dense
region the inter-marker spacing *is* small, so an `eps` scaled to it can still merge
neighbours. The robust quantity is **marker geometry**, which is constant across the whole
curve regardless of density: assemble fragments using an `eps` derived from the **glyph's own
size** (segment length / fill-bbox size), which is invariant to how densely markers are
packed. This plan uses marker-geometry-derived `eps`, per identity-group, and keeps the
nearest-neighbour distribution only as a *post-hoc merge detector* (see §4.3).

---

## 2. Architecture: the deterministic/LLM split

Resolves the apparent tension between "tool does the clustering per-element" and "LLM assigns
element labels." The pipeline groups markers by a **deterministic visual identity** first,
the LLM names the groups second:

```
            ┌─────────────────────── deterministic (this module) ──────────────────────┐
PDF page ─► detect drawing objects ─► group by identity ─► assemble markers ─► calibrate
            (curves/lines/rects)       (colour OR shape)    (per-group eps)     axes (LSQ)
                                             │                                     │
                                             ▼                                     ▼
                                  structured marker records  ◄───────────  data coords + residual
                                             │
            └──────────────────────────────┼───────────────────────────────────────────┘
                                            ▼
            ┌────────────────────────── LLM (downstream, unchanged scope) ──────────────┐
            read legend → map group_key→element → y-quantity conversion → 26-col rows
            └───────────────────────────────────────────────────────────────────────────┘
```

- **Colour figures:** identity = the object's fill/stroke colour tuple (deterministic from
  the PDF draw command). Each colour group ≈ one element. The LLM maps colour→element.
- **Monochrome figures:** identity = marker **shape class** (`filled_circle`, `filled_tri`,
  `cross`, `plus`, …). Each shape group ≈ one element. The LLM maps shape→element.

The LLM never sees an image *for clustering*. It receives structured JSON (group_key →
N markers with data coords) plus the PDF *only* for reading the legend and tick labels.

---

## 3. Module structure & function signatures

New subpackage `extraction/curve_extractor/` (sits beside the existing
`anthropic_client.py` / `parse_output.py` in the `extraction` package). One hard new
dependency: **`pdfplumber`** (geometry). Clustering is **numpy-native** — no `sklearn`;
optional `scipy.spatial.cKDTree` only as a speed path, guarded by a fallback.

```python
# extraction/curve_extractor/types.py
@dataclass(frozen=True)
class MarkerRecord:
    group_key: str          # "#4caf50" (colour hex) | "filled_circle" | "cross"
    marker_type: str        # "filled" | "stroked"
    pixel_x: float
    pixel_y: float
    data_x: float | None    # None until axis calibrated
    data_y: float | None

@dataclass
class AxisCalibration:
    axis: str               # "x" | "y"
    model: str              # "linear" | "log10"
    slope: float
    intercept: float
    residual_rms: float     # RMS of (fit - tick_value), in DATA units
    r_squared: float
    n_ticks: int
    tick_values: list[float]
    ok: bool                # residual_rms below threshold

@dataclass
class CurveExtractionResult:
    markers: list[MarkerRecord]
    x_calibration: AxisCalibration
    y_calibration: AxisCalibration
    per_group_counts: dict[str, int]
    page_index: int
    figure_bbox: tuple[float, float, float, float]
    is_vector: bool         # False => no drawing objects found (raster); caller falls back
    warnings: list[str]

# extraction/curve_extractor/detect.py
def load_page(pdf_bytes: bytes, page_index: int) -> "pdfplumber.page.Page": ...
def find_plot_frame(page) -> tuple[float, float, float, float]:
    """Largest rect, or 4 long perpendicular lines, bounding the plot area."""
def collect_objects(page, frame, *, legend_bbox=None) -> list[dict]:
    """curves+lines+rects strictly inside `frame`, excluding the legend region."""

# extraction/curve_extractor/markers.py
def classify_marker_type(obj: dict) -> str:                      # "filled" | "stroked"
def group_by_identity(objs, *, colour_figure: bool) -> dict[str, list[dict]]
def assemble_markers(group_objs, marker_type, group_key) -> list[MarkerRecord]
def _calibrate_eps(group_objs, marker_type) -> float            # marker-geometry derived
def _merge_warnings(markers) -> list[str]                       # post-hoc NN merge detector

# extraction/curve_extractor/calibrate.py
def detect_ticks(page, frame, axis: str) -> list[float]         # tick PIXEL positions
def read_tick_values(page, tick_pixels, axis) -> list[float] | None  # from pdfplumber chars
def fit_axis(tick_pixels, tick_values) -> AxisCalibration       # tries linear & log10, picks best

# extraction/curve_extractor/extractor.py  (orchestrator)
def extract_curves(
    pdf_bytes: bytes,
    page_index: int,
    *,
    colour_figure: bool = True,
    tick_values_x: list[float] | None = None,   # LLM-supplied fallback if chars unreadable
    tick_values_y: list[float] | None = None,
    legend_bbox: tuple | None = None,            # LLM-supplied fallback if heuristic fails
) -> CurveExtractionResult: ...
```

The optional `tick_values_*` / `legend_bbox` params are the **hybrid seam**: when
deterministic detection of those few visual facts fails, the LLM supplies them and the
geometry/clustering/fitting stays deterministic.

---

## 4. Algorithms in detail

### 4.1 Marker-type classification (`classify_marker_type`)

A checkable fact about the draw command, not a visual inference:

- `obj["object_type"] == "rect"` or a `curve` with `fill=True` / non-null
  `non_stroking_color` and a closed path → **filled**. Sub-classify shape by
  `extent = bbox_area_covered / bbox_area`: ~1.0 square, ~0.785 (π/4) circle, ~0.5 triangle;
  combine with vertex count for diamond/star.
- `line` objects with `stroke=True, fill=False` → **stroked** fragments; the marker is the
  *assembly* of 2–3 such fragments sharing a midpoint (× = 2 diagonals, + = h+v, * = 3).

Unit-testable on hand-built `dict`s mimicking pdfplumber objects — no PDF needed.

### 4.2 Self-calibrating assembly (`assemble_markers` + `_calibrate_eps`)

Per identity-group (so geometry is homogeneous within a group):

- **Filled groups:** each marker is ≈ one fill path. `eps = 0.5 × median(fill bbox diagonal)`
  of the group — large enough to dedupe a marker drawn as overlapping outline+fill paths at
  the same spot, far smaller than inter-marker spacing. Centroid = bbox centre; merge
  centroids within `eps` (single-linkage, numpy pairwise or `cKDTree` query).
- **Stroked groups:** `eps = 0.6 × median(segment length)` of the group (the glyph arm
  length). Assemble line fragments whose midpoints fall within `eps` into one glyph; glyph
  centroid = mean of fragment midpoints. This `eps` is a marker-size quantity, **constant
  across the curve**, so a dense transition zone (small inter-marker spacing) does not change
  it — the originally-diagnosed failure cannot recur for cleanly-separated markers.

Clustering is single-linkage agglomeration with a fixed `eps` (numpy `pdist`/union-find for
<~2000 points; `cKDTree.query_pairs(eps)` when available). No `sklearn`.

### 4.3 Post-hoc merge/over-count detector (`_merge_warnings`)

The independent check that turns silent errors into flags:

- After assembly, compute the centroid nearest-neighbour distribution per group. The
  **minimum** inter-marker distance should be ≳ the group's marker size. If any assembled
  cluster's own span exceeds ~1.5× the median marker bbox, two distinct markers were likely
  merged → `warning`. If a group's marker count is an outlier vs. sibling groups in the same
  figure (e.g. 6 when neighbours have 18), → `warning`. These warnings are the deterministic
  analogue of today's `row_count_sanity` QA check, raised *at the source*.

### 4.4 Axis calibration (`detect_ticks` → `read_tick_values` → `fit_axis`)

- **Tick pixels (deterministic):** ticks = short lines perpendicular to a frame edge with one
  endpoint on the frame. x-ticks share a y (frame bottom), vary in x; y-ticks vice-versa.
- **Tick values:** first try `page.chars` / `extract_words` near the axis, parse floats,
  associate each to nearest tick pixel. If the page is raster / labels unextractable → use
  the LLM-supplied `tick_values_*`. (Reading the *number* is the only genuinely-OCR part;
  positions and the fit are deterministic.)
- **Fit (`fit_axis`):** least squares `data = m·pixel + b` over (tick_pixel, tick_value).
  Compute `residual_rms` in data units and `r_squared`. **Also fit `log10(data)` vs pixel**;
  if the linear residual is high but the log residual is low, the axis is logarithmic (e.g.
  the 0.05→1.0 concentration sweep) — pick the better model and record it. `ok = residual_rms
  < threshold` where threshold is a fraction of median tick spacing (e.g. 2%); `not ok` =>
  flag the paper for manual review (mis-detected ticks, unexpected axis transform).

---

## 5. Validation strategy

### 5.1 Oracle paper (the diagnosed one): Swain & Otu 2011, Fig. 2

- **Ground truth:** the documented expected per-element marker count for Fig. 2 (the manual
  count). Encode it as a fixture `tests/fixtures/swain2011_fig2_expected.json`.
- **Primary assertion:** `per_group_counts` matches expected for **all 14 elements**, with
  special focus on the previously-failing set (La/Ce/Pr/Nd under-counted, the 2 over-counted)
  — these must now match (tolerance ±0; if ±1 is accepted anywhere, document why per element).
- **Calibration assertions:** x/y `residual_rms` below threshold; `model == "linear"` for the
  pH axis; the digitized endpoints reproduce the paper's stated text endpoints (Lu 0.9→39.96%,
  4.0→99.16%; Yb 0.9→29.43%, 4.0→99.51%) within ±0.1 pH / ±2 %E — reusing the existing
  `text_endpoint_cross_check` logic as a unit assertion.
- This test **must pass before the module is trusted on any new paper** (gate in CI).

### 5.2 Second paper (generality) — pick one that stresses *different* code paths

A second colour/linear figure proves little. Choose a paper that is **monochrome with
shape-coded series** and/or has a **log axis**, to exercise: the stroked-glyph assembly path
(§4.2), shape-based `group_by_identity` (§2), and log-axis detection (§4.4). Document its
expected per-series counts the same way. Acceptance = counts match + calibration `ok` +
correct `model` chosen for the axis. If we lack such a paper, that's a required input to
collect before claiming generality.

### 5.3 Unit tests (no PDF needed)

- `classify_marker_type`: hand-built filled/stroked/rect objects incl. edge cases.
- `_calibrate_eps`: synthetic bimodal fragment clouds; assert `eps` lands in the valley and
  is invariant to injected density changes (the regression test for the actual bug).
- `assemble_markers`: synthetic × and circle clusters at varying spacing; assert no
  under/over-merge across a 10× density sweep.
- `fit_axis`: synthetic linear ticks, synthetic log ticks, and a deliberately corrupt tick
  set that must trip `ok == False`.

### 5.4 Regression guard

Snapshot the full `CurveExtractionResult` (markers + calibration) for the oracle paper as a
JSON fixture; a CI diff fails on any drift.

---

## 6. Interface to the downstream LLM step

**Recommended wiring (option B — pre-pass, no LLM-written code):** `extraction/runner.py`
gains a deterministic pre-pass that calls `extract_curves(pdf_bytes, page_index)` **on our
side**, serialises the `CurveExtractionResult` to JSON, and injects it into the user turn
*alongside* the still-attached PDF. A new prompt version (**`extraction_v6`**) changes the
model's task from "digitize the figure" to:

> "You are given pre-computed marker data (`group_key → [{data_x, data_y, marker_type}]`)
> plus the PDF. Do **not** re-digitize. Read the legend to map each `group_key` to an
> element, apply the y-quantity conversion if the axis is log D / D, and emit the 26-column
> rows. Heed any `warnings` (a flagged group may be mis-counted — note it for review)."

This removes the code-execution sandbox cost for clustering entirely (the dominant cost in
the diagnosed runs — repeated image rendering + narrated cluster cycles), giving the ~4×
token reduction and the minutes→milliseconds speed-up on the geometry step.

**Fallback (option A):** for raster figures (`is_vector == False`) or when calibration is
`not ok`, fall back to the current vision/code-execution path. The pre-pass output's
`warnings` and `ok` flags decide automatically which path a paper takes.

What stays LLM work (unchanged): legend→element mapping, y-quantity conversion, the 26-column
canonical-string formatting, which-figure-is-primary, and paper-specific judgment (expected
non-monotonicity, per-figure legend differences).

---

## 7. Build milestones

1. **M1 — geometry core:** add `pdfplumber`; `detect.py` + `classify_marker_type` +
   `_calibrate_eps` + `assemble_markers`, with the §5.3 unit tests. No PDF I/O beyond loading.
2. **M2 — calibration:** `detect_ticks` / `read_tick_values` / `fit_axis` incl. log detection,
   with synthetic-tick unit tests.
3. **M3 — orchestrator + oracle:** `extract_curves`; the Swain 2011 Fig. 2 oracle test (§5.1).
   **Gate: must pass before M5.**
4. **M4 — generality:** second-paper fixture + tests (§5.2).
5. **M5 — pipeline integration:** `extraction_v6` prompt; `runner.py` pre-pass + automatic
   vector/raster routing; end-to-end on the oracle paper; compare token/time vs the v5.2 run.

---

## 8. Risks / honest limits

- **Raster figures:** zero drawing objects → `is_vector=False` → must fall back (M5 routing).
  The module's job there is to *detect and signal*, not to solve.
- **Flattened/anti-aliased vector PDFs** where `fill` flags are unreliable or markers are tiny
  embedded images — classification degrades; mitigated by the §4.3 warnings + fallback.
- **Genuinely coincident markers** in the densest zone: the tool will not invent points (nor
  should it) — but it must not under-count *cleanly separated* ones, which is the actual bug.
- **Legend-exclusion heuristic** counting legend markers as data → mitigated by the optional
  `legend_bbox` LLM seam.
- **Dependency:** `pdfplumber` pulls in `pdfminer.six`; acceptable. Avoiding `sklearn` keeps
  the footprint small; revisit only if numpy clustering proves too slow (>~10⁴ objects/page).
