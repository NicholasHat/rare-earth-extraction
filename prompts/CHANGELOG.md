# Extraction prompt changelog

Every extraction run records the exact `prompt_version` and `prompt_sha256` it
used (`prompt_runs` table), so the dataset stays reproducible across versions.
Old versions are never edited or deleted.

## extraction_v8 — deterministic curve pre-pass becomes a digitizer (current pinned default)
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
  Estimate/raster pages are unaffected.
- No change to the QA layer: `deterministic_curve_count` still cross-checks
  row counts exactly as before — the new coordinate block is a prompt-side
  cost optimization, not a new ground-truth signal QA validates against.
- Core extraction behaviour (Steps 0–10, OUTPUT CONTRACT wire format) is
  otherwise unchanged from v7.
- Paired with a request-level change in `extraction/anthropic_client.py` (not
  a prompt change, so no separate version bump for it): context editing
  (`clear_tool_uses_20250919`) clears stale code-execution tool results
  mid-loop, since the server-side tool loop otherwise resends the entire
  accumulated transcript on every internal iteration — the other diagnosed
  driver of the same cost problem.

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
