# Extraction prompt changelog

Every extraction run records the exact `prompt_version` and `prompt_sha256` it
used (`prompt_runs` table), so the dataset stays reproducible across versions.
Old versions are never edited or deleted.

## extraction_v5.2 — code-execution digitization (current pinned default)
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
