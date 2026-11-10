# Extraction prompt changelog

Every extraction run records the exact `prompt_version` and `prompt_sha256` it
used (`prompt_runs` table), so the dataset stays reproducible across versions.
Old versions are never edited or deleted.

## extraction_v16 — the 2026-11 ten-paper audit of v15: values in the right column, every varied axis kept, nothing skipped (current pinned default)
Measured against the PDFs, v15's digitised numbers were accurate (typically
< 0.5 %E), but values landed in wrong columns, varied axes were lost and data
was skipped. Each finding became a short rule folded into the step it belongs
to (no new section):
- **Stripping / scrubbing** (Step 1): back-extraction percentages go in
  `Recovery %` with `Extract%` blank; `Treatment` is `Stripping` or
  `Scrubbing` as the paper names it. v15's Step 1 mapped "% recovery" to
  `Extract%` — one paper put 70 stripping rows there; a scrubbing table was
  labelled Stripping. A stripping D is stored as reported (v15's
  `%stripping = 100/(1+D)` aside is gone).
- **A varied axis is never blank** (Step 8): every sweep row carries its x;
  vol% of a neat extractant converts with stated densities and molar masses
  (D2EHPA ≈ 30.1, TBP ≈ 36.5, PC88A ≈ 31.0, Cyanex 272 ≈ 31.7 mM per vol%);
  an x no schema column holds (wt% of a mixture, a formulation ratio) means
  no rows and a reason in `figures` — 12 rows of a Cyanex 923 wt% sweep had
  identical conditions and different %E. v15's "Acid Solution conc. blank
  when acidity is varied" now says when pH is varied.
- **Element groups are not elements** (Step 1): no LREE/MREE/HREE/TREE rows
  (106 on one paper), but a single-element bar in a grouped chart is
  digitised (another paper skipped six charts with a Y bar each), and a
  shared legend series gives one row per named element.
- **Column definitions** (Step 8): `Acid Solution` is the aqueous medium the
  row's percentage refers to; `Co-extractant` is organic-phase only (aqueous
  complexants go in `Material Process`); a self-extracting ionic liquid is the
  `Extractant`; `Phase ratio (O/A)` is of the row's own contact;
  `Figure/Table` is where the numbers were taken from; a separation factor
  needs its pair and direction in `Material Process` ("SF Dy/Nd").
- **Derived %E only where it means %E** (Step 7): D → %E for a single batch
  contact only; successive contacts, counter-current circuits and
  mixer-settlers keep the reported quantity (a 4th-successive-contact D had
  become "Y 99.5 %E").
- **Scope** (Step 0): emulsion/supported liquid membranes, pertraction and
  other transport experiments get no rows (owner decision); batch kinetics
  stay in scope.
- **Completeness** (Steps 0, 5; `figures`): every numbered figure and table,
  and every numeric REE sentence, yields rows or a reason; every marker of a
  kinetics plot is a row (v15 took 6 of 14 time points on 71 of 75 series);
  hidden markers are digitised from their visible edge or listed as omitted;
  a drawn (0, 0) origin marker is not a row.
- **pH type inheritance** (Step 8), **overlaid PDFs** (Step 2: digitise only
  what the rendered page shows — a hidden "FOR PEER REVIEW" draft layer
  doubles the geometry), and **`text_endpoints.x_basis`** is the exact schema
  column name, `x_value` null when the sentence states no x (v15 invented ~10
  bases and once copied the D value into `x_value`).
QA changed alongside (validation/checks.py; one curve definition,
validation/curves.py, shared with the Analysis page) and the curve pre-pass
no longer trusts overlaid pages. Output shape is unchanged from v15. (Not yet
validated live — no `prompt_runs` record exists for this version.)

## extraction_v15 — nine columns after the lab's 26, so a row keeps its conditions; each row names its own figure
The 2026-10 ten-paper trial (v13, four papers re-run on v14) showed rows
losing information the lab's 26 columns have nowhere to put. The 26 stay
exactly as they are — names and order mirror the lab's spreadsheet — and nine
are appended after `Recovery %` (validation/schema.py `ADDED_COLUMNS`):
- `Diluent` — one trial paper is entirely about diluent effects.
- `Co-extractant`, `Co-extractant Conc. (mM)` — Banda 2012b's varied synergist
  concentration was folded into the extractant's name ("Cyanex 272 + TBP
  (0.5 M)"), inconsistent with papers whose varied concentration is numeric.
- `Phase ratio (O/A)` — v14 needs it to convert D to %E but stored it nowhere.
  For a solid sorbent it is m/V in g/mL, so `%E = 100·D/(D + A/O)` holds for
  both systems (Step 7 now writes R as A/O = 1/(O/A)).
- `Contact time (minute)` — v14 put a kinetics plot's shaking time in
  `Leaching time (minute)` (XAD-7 paper: 15–180 min), which is now reserved
  for a leaching step that dissolves a solid feed.
- `Distribution ratio (D)` — D as reported (log D → D), next to the %E derived
  from it; only the derived %E survived before. QA now recomputes %E from D
  and the phase ratio on every row that has all three.
- `Total REE feed (mM)` — Afonin 2024 varies the total feed of a mixed-REE
  solution (0.05–0.99 M), and v14's rule against copying totals onto each
  element left those rows without their x.
- `pH type` — `equilibrium` or `initial`; mixing them corrupts any curve comparison.
- `Figure/Table` — the row's source. Replaces v13–v14's parallel
  `row_figures` list, which was off by a few entries on the three longest
  tables of the trial (345, 265 and 84 rows), losing every row's provenance.
  A curve is now (paper, Figure/Table, element, series), which is also what
  stops QA pooling two figures of the same element.
- `text_endpoints` entries may name their `extractant`, which QA uses to pick
  the curve a statement is about.
`parse_output` still reads v7–v14 output (26 columns; the added ones load
null, and an aligned `row_figures` fills `Figure/Table`). Everything else is
unchanged from v14. (Not yet validated live — no `prompt_runs` record exists
for this version.)

## extraction_v14 — the errors the 2026-10-07 ten-paper audit found; no PDF library named
Checked against the PDFs' own vector geometry, v13's rows were exact on the
clean papers, but:
- **Phase ratio.** A resin paper's log D (mL/g) was converted with
  `100·D/(1+D)`, overstating low-pH %E up to fivefold (Wang 1998). Step 7 now
  converts every D / K_d with the experiment's R — V_aq/V_org liquid–liquid,
  V/m in mL/g for a solid sorbent, never 1 for a solid — and Step 1 asks for
  the volumes and masses. A solid-phase figure with no stated V/m gets no rows.
- **Rare earths only.** A Pb paper produced 84 rows; Step 1 now scopes rows to
  La–Lu, Y, Sc and other metals get none (QA's `element_scope` enforces it).
- **Feed totals.** v13's pitfall row told the model to put a total mM in the
  per-element column (P507: 0.21 M total copied onto 14 elements). Now: this
  element's concentration, computed from a composition only when its basis is
  stated, else blank.
- **Markers at their own x.** Values "read visually and anchored to the text"
  snapped to the caption's concentration list and missed 15 markers (Banda
  2012b). Step 5: one row per located marker, at its own x.
- **Precision.** Derived numbers to ≥ 3 significant figures (molar ratios had
  been rounded to 0.2, tripping QA's arithmetic check).
- **No PDF library named.** PyMuPDF left the sandbox; snippets use pdfplumber
  for vector objects and the toolkit's `pages.render` to render.

## extraction_v13 — say where each row came from and why any figure was skipped; raster steps use the toolkit's legend workflow
- **Provenance.** The OUTPUT CONTRACT gains two keys beside `text_endpoints`:
  `row_figures` (the source figure of each row, in row order) and `figures`
  (every numbered figure, whether it was digitised, and a reason when not).
  From the sibling ree-extraction-local pipeline, whose per-panel record of
  what it read and why it skipped a panel made its runs reviewable. The 26
  columns are unchanged; `extraction/parse_output.py` still accepts v7–v12
  output (the keys are optional there), drops a `row_figures` that doesn't
  line up with `rows` (QA notes it), and the review page lists the figures.
- **Raster steps use the toolkit's legend workflow.** Steps 3–5 point at the
  SANDBOX TOOLKIT block's `find_panels` → `find_swatches` / `match_markers`
  → `label_centres` / `fit_ticks` chain (ported from the local pipeline on
  2026-09-29) instead of the blob detector, which stays as the fallback for a
  legend without drawn swatches.
- **Step 2's snippet keeps the document alive.** `fitz.open(path)[i]` orphans
  the page under the sandbox's PyMuPDF 1.21, so the render failed.
- Everything else is unchanged from v12.
- (Not yet validated live — no `prompt_runs` record exists for this version.)

## extraction_v12 — digitise the markers the figure has, no more; state only what is true of the sandbox
- **No point-count target.** v5–v11 said "~10–20 points per element … that is the
  target" in five places. Quinn et al. 2015's series have 5–7 markers, and the
  extraction_v10 run returned 13 evenly spaced points per series sampled off the
  fitted lines (see `validation/checks.py` `sampled_from_line`, which now rejects
  that). The target is now the figure's own marker count, with an explicit rule
  never to sample points along a fitted or guide line; Step 10, rule 7 and the
  pitfalls table say the same, plus a new pitfalls row.
- **The model cannot see sandbox renders.** Step 2 said to "render the page and
  look at it"; renders are never shown to the model, and opening a PNG with the
  file viewer returns base64 into the transcript. Visual reading now points at the
  PDF document; a pitfalls row names the file-viewer trap; the estimate-page
  guidance no longer suggests re-rendering to check.
- **PyMuPDF, not pdfplumber.** Step 4's snippet and marker descriptions used
  pdfplumber's `page.curves` / `page.lines`, which does not import in the sandbox;
  they now use `page.get_drawings()`.
- **Raster render scale.** Step 2 says to render the pre-pass bbox at 300 dpi (the
  toolkit's scale) and not to re-render at another — the v11 run re-rendered at
  native resolution mid-figure. Step 3 notes corner ticks are not returned.
- **Counts are filled-marker only.** The deterministic section now says the
  authoritative counts cover filled-marker series and the legend may list more
  (Swain & Otu Fig. 2: 9 counted, 14 in the legend). The pre-pass block says the
  same.
- **Dead text removed:** the v4-era "What changed" preamble (history lives here)
  and the v8 `DIGITIZED CURVE DATA` subsection, now one line deferring to the
  block's own instructions — the block has never fired on a real paper
  (`calibrate.auto_ticks` reads no axis on either validation paper).
- Everything else is unchanged from v11. ~5% shorter.
- (Not yet validated live — no `prompt_runs` record exists for this version.)

## extraction_v11 — the sandbox toolkit replaces hand-written raster digitisers
- Companion to `extraction/sandbox_toolkit.py`: the repository's tested
  `curve_extractor` package now rides into the code-execution sandbox next to
  the PDF, described by a per-run "SANDBOX TOOLKIT" user-turn block (injected
  like the curve pre-pass block, never part of this file). Diagnosed on Quinn
  et al. 2015 under v9 (Batch, 2026-09-16): 51 code-execution iterations and
  still unfinished — ~18 of them probing the environment (pdfplumber's install
  is broken there; PyMuPDF works) or exploring page structure the pre-pass
  already knew, the rest writing a blob detector, text-row filter and
  calibrator from scratch across seven scripts. Every iteration re-reads the
  whole growing transcript, so the loop length is the cost.
- Step 2 no longer asks for a pdfplumber probe: the DETERMINISTIC CURVE
  ANALYSIS block classifies every figure page (raster pages now carry the
  figure's PDF bbox), and any probe uses PyMuPDF.
- Step 3 and Step 5 route raster figures through the toolkit's
  `find_frame` / `tick_pixels` / `fit_axis` / `detect_markers_in_image`
  instead of describing the algorithm for the model to re-implement.
- New OUTPUT CONTRACT section "Sandbox toolkit (when present)" with a budget
  of about three code runs per raster figure, and a rewritten "How you receive
  the paper" (no `pip install pdfplumber`, no directory listing to find the
  file). Two matching "Common pitfalls" rows. Everything else is unchanged
  from v10.
- (Not yet validated live — no `prompt_runs` record exists for this version.)

## extraction_v10 — in-plot text is not data; tangled raster series are omitted, not guessed
- Found on Quinn et al. 2015 (raster, monochrome): every Fig. 2 panel prints its
  title inside the frame at log D ≈ 0.7, and the model digitised those letters
  as the filled-square (Lu) series — thirteen rows at 83–85 %E in series whose
  real points sit at 1–5 %E. Step 5 now treats panel titles, annotations and any
  other in-plot lettering as regions to mask before digitising, exactly like the
  legend, with the monochrome-raster failure mode spelled out.
- Step 5 also stops the model apportioning a cluster of three or more
  overlapping monochrome series by guesswork (Quinn's Er/Y/Ho/Dy/Tb tangle
  produced inconsistent, partly invented rows): digitise the separable series,
  omit the rest.
- Step 6 adds the check the QA layer now also runs (`validation/checks.py`
  `off_curve`): fit each series as a straight line in log D vs pH and drop
  points an order of magnitude off it — before text end-point validation, so an
  artifact can't be "validated" by coincidence.
- Three matching "Common pitfalls" rows. Everything else is unchanged from v9.

## extraction_v9 — log-log concentration sweeps are data, not always derived
- Step 1's y-axis classification no longer skips every `log D` vs
  `log[extractant]` plot as a derived re-plot. Found on Swain & Otu 2011: its
  Fig. 5 holds concentration sweeps at equilibrium pH 1.25 and 2.50 that appear
  in **no other figure** — the old blanket skip rule silently dropped them, and
  only the pH 1.75 sweep (Fig. 4) made it into the dataset. v9 digitises the
  log-log series whose conditions no other figure reports, converting both axes
  (`[ext] = 10^x`, `D = 10^y` → `%E`), and still skips the series that merely
  duplicate an already-digitised sweep.
- Adds a matching "Common pitfalls" row. Everything else (the v8 DIGITIZED
  CURVE DATA shortcut, the v7 positional wire format, Steps 0–10) is unchanged.

## extraction_v8 — deterministic curve pre-pass becomes a digitizer
- `extraction/curve_prepass.py` no longer discards the calibrated (x, y) marker
  coordinates the vector path already computes (`curve_extractor.extractor`
  runs axis calibration for every vector page, but the pre-pass previously
  kept only per-series counts). Authoritative pages now also carry a
  `DIGITIZED CURVE DATA` JSON block (`{group_key: [[x, y], ...], ...}`,
  coordinates rounded to 4 significant figures) injected alongside the
  existing ground-truth count sentence.
- The prompt's "Deterministic curve analysis" section gains a new subsection
  telling the model to skip its own axis calibration and clustering (Steps
  3-5) for series covered by this block, going straight to legend-mapping
  (Step 4) and unit/experimental-constant conversions (Steps 7-8) — this
  targets the dominant cost driver on dense multi-figure papers: the model's
  own exploratory code-execution digitization work.
- Scoped to vector, filled-marker (coloured) series on already-authoritative
  pages only — the same trust gate the existing counts anchor uses. Monochrome
  (marker-shape) series are explicitly excluded; that assembly path isn't
  implemented (docs/curve_extractor_plan.md), so those series still receive
  full Steps 3-5 treatment regardless of what else on the page is pre-digitised.
  Estimate/raster pages are unaffected. *(Update 2026-08-27: stroked-glyph
  assembly now exists — `markers.assemble_stroked` — but is still kept out of
  this block and the authoritative gate until it has an oracle-paper
  validation; the prompt text is unchanged.)*
- No change to the QA layer: `deterministic_curve_count` still cross-checks
  row counts exactly as before — the new coordinate block is a prompt-side
  cost optimization, not a new ground-truth signal QA validates against.
- Core extraction behaviour (Steps 0–10, OUTPUT CONTRACT wire format) is
  otherwise unchanged from v7.
- A paired request-level change (context editing, `clear_tool_uses_20250919`
  in `extraction/anthropic_client.py`) was tried alongside this and reverted:
  clearing stale tool results mid-conversation breaks the prompt-cache prefix
  for everything downstream of the clear, forcing cache-write-priced tokens
  where this pipeline was getting cache-read-priced tokens. Measured on a live
  run as a 2-3x cost regression (cache_creation_input_tokens rose 7.7x), not a
  saving. Don't re-attempt it without a plan for the cache invalidation.

## extraction_v7 — compact positional output format
- The OUTPUT CONTRACT's `rows` shape changes from a list of objects (each row
  repeating all 26 verbose column-name keys, e.g.
  `"Rare Earth Elements (REY:La, Ce, Nd)"`) to a positional form: a top-level
  `columns` array (the 26 names, written once) plus `rows` as arrays of 26
  values, index-aligned to `columns`. A typical 90-row extraction was paying
  for ~26 keys × 90 rows of output tokens purely on key repetition; this cuts
  that to 26 keys total. `extraction/parse_output.py` accepts both shapes, so
  older `prompt_runs.raw_response` records still replay correctly.
- Core extraction behaviour (Steps 0–10) is unchanged from v6.
- Paired with two request-level changes in `extraction/anthropic_client.py`
  (not prompt changes, so no new prompt version needed for them): a cache
  breakpoint on the user turn so the PDF is reused (not rebilled) across the
  model's internal code-execution iterations, and a `task_budget` backstop
  bounding total spend per extraction call.

## extraction_v6 — deterministic curve pre-pass anchor
- A deterministic pass (`extraction/curve_prepass.py`, using the vector curve
  extractor) runs BEFORE the model call and injects a "DETERMINISTIC CURVE
  ANALYSIS" block into the user turn: authoritative per-series marker counts for
  clean single-panel vector figures (ground truth the model must match), softer
  estimates for multi-panel figures, and a raster-image flag. Directly attacks
  the under-digitisation failure mode by telling the model how many points each
  curve actually has (docs/curve_extractor_plan.md §6).
- The same counts feed a new `deterministic_curve_count` QA check (README §9).
- **Cost fix:** on an authoritative page, the model is told to treat the count as
  a stopping condition — one clustering pass, no trial-and-error tolerance
  guessing, no re-rendering the page to "visually re-confirm" a count that
  already matches, no narrated multi-attempt prose. This targets the diagnosed
  dominant cost driver directly (iterative narrated clustering + repeated image
  re-rendering), not just the under-counting. Estimate/raster pages keep the
  full visual-diligence path since they have no verified count to target.
- Core extraction behaviour (Steps 0–10) is unchanged from v5.2.
- (Not yet validated live — no `prompt_runs` record exists for this version;
  the one live attempt failed on insufficient API credits before producing
  output. This changelog entry was revised once, in place, before any run used
  it — once a run is recorded against it, future changes fork a new version.)

## extraction_v5.2 — code-execution digitization
- Flips the "How you receive the paper" note: the model is now also given the
  PDF inside a **code execution** sandbox (`anthropic_client.py`), so Steps 2–6's
  `pdfplumber`/`numpy` snippets are run for real (vector/raster detection, axis
  calibration, point digitization) instead of being read by eye off the
  rendered page. Vision is still used for what code can't tell you (legend
  colour/marker mapping, judging genuinely unresolvable overlap).
- Adds an explicit instruction to digitise every distinct experiment/figure in
  the paper (Step 0), not just the first one found — v5.1's first live run
  (Swain & Otu 2011) only processed one of several figures.
- Everything else is unchanged from v5.1.

## extraction_v5.1 — additive text endpoints
- Adds the **OUTPUT CONTRACT** section: the model returns a single JSON object
  with `rows` (the 26-column data) and `text_endpoints` (the paper's stated
  numeric claims), so the QA layer can cross-check digitized values against the
  paper's prose (README §6, §9).
- The core 26-column extraction behaviour is **unchanged** from v5.

## extraction_v5 — prior production prompt
- The iterated figure-reading rules (raster→vector→figure-first triage→per-element
  molar ratio→canonical field-value strings). Retained for reproducibility.
