# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit web app that turns rare-earth-element (REE) solvent-extraction research papers (PDFs) into a queryable SQLite database. One writer uploads PDFs; an LLM extraction pipeline produces one row-per-data-point table against a fixed **26-column schema**; after automatic QA and human review, rows merge into a master DB that two read-only consumers (a calculator and an AI assistant) sit on top of.

The README.md is the original design document (a plan written before implementation) — it is authoritative on *intent* and section-numbered (`README §6` references appear throughout the code), but the code has moved past it in places (e.g. the pinned prompt is now `extraction_v6`, not `extraction_v5.1`). Trust the code for current behavior; use the README for rationale.

## Commands

```bash
# Run the app (multi-page: home = extraction, pages/ = DB / Calculator / Assistant)
streamlit run app.py

# Tests (pure/offline — no API calls, no network)
python -m pytest                    # full suite (~113 tests)
python -m pytest tests/test_solve.py            # one file
python -m pytest tests/test_checks.py -k monotonicity   # one test by name
```

There is no lint/format config and no `pytest.ini` — `conftest.py` just puts the repo root on `sys.path`. Dependencies: `pip install -r requirements.txt` (a `.venv/` already exists).

## Environment

Config is read once in `config.py` from `.env` at the repo root. Keys: `ANTHROPIC_API_KEY` (required for extraction/assistant, not for tests), `REQUIRE_PASSWORD`/`WRITE_PASSWORD` (write gate), `EXTRACTION_PROMPT_VERSION` (default `extraction_v6`), `EXTRACTION_MODEL` (`claude-opus-4-8`), `ASSISTANT_MODEL` (`claude-haiku-4-5-...`). Never hardcode paths or model IDs — add them to `config.py`.

## Architecture: three pillars over one DB

The master SQLite DB (`data/master.db`) is the hub. **Pillar A is the only writer; Pillars B and C are strictly read-only** and query through the `v_current_best` view, never the raw `extractions` table.

- **Pillar A — extraction** (`app.py` home page, `ingestion/`, `extraction/`, `validation/`, `database/`): upload → dedup → LLM extract → QA → human review → merge.
- **Pillar B — calculator** (`pages/2_Calculator.py`, `calculator/`): deterministic ppm↔mM / molar-ratio / concentration math, cross-checked against DB typical ranges.
- **Pillar C — assistant** (`pages/3_Lab_Assistant.py`, `assistant/`): Anthropic tool-calling agent that answers questions by querying the DB and calling the calculator — never by inventing numbers.

### Key invariants (violating these breaks the design)

- **Read/write split is enforced two ways.** Write actions call `auth.require_write_access()` first; read consumers open the DB via `connection.get_readonly_conn()` (SQLite `mode=ro`). The assistant's `query_database` tool has a *second* independent backstop, `assistant/sql_guard.py`, which rejects anything but a single SELECT/CTE over a table whitelist even if the model is jailbroken. When touching the assistant's DB access, keep both layers.
- **Prompt versioning = reproducibility.** The extraction prompt is a pinned, versioned file in `prompts/extraction_vN.md`, loaded only through `extraction/prompt_loader.py`, which records the version + sha256 on every run. Improving extraction means dropping in a new `prompts/` file and bumping `EXTRACTION_PROMPT_VERSION` — **no pipeline code changes**. Never hardcode prompt text elsewhere.
- **Coexistence, not replacement.** Re-extracting a paper under a new prompt version inserts a new `prompt_runs` row; it never overwrites old data. `v_current_best` (defined in `database/schema.sql`) resolves "latest approved prompt version per paper" at query time, so old and new prompt versions coexist without migration. Consumers must read `v_current_best`.
- **The DB is only written on human approval,** through the single atomic transaction in `database/merge.py::commit_extraction`. `extraction/runner.py` produces an `ExtractionResult` and touches nothing persistent. `review_log` is append-only and records no reviewer identity (single shared password ⇒ actor unknowable).

### Extraction flow (`extraction/runner.py::extract_paper`)

1. Load pinned prompt (`prompt_loader`).
2. **Deterministic curve pre-pass** (`extraction/curve_prepass.py` → `extraction/curve_extractor/`): counts figure markers from the PDF's own vector geometry, purely and with no API call, and injects the counts as a grounding "ground truth" block into the prompt + a QA cross-check anchor. It only marks a page *authoritative* when series counts are internally uniform (a clean single-panel figure); multi-panel and raster figures are downgraded to hints. A failure here degrades to "no anchor" and must never block extraction.
3. Anthropic API call (`extraction/anthropic_client.py`) → parse (`extraction/parse_output.py`).
4. **QA checks** (`validation/checks.py`): pure, unit-tested functions producing a `QAReport` of RED (merge-gated) / AMBER (warning) flags — row-count sanity, axis bounds, monotonicity, duplicates, vocabulary drift, and cross-checks against both the paper's stated `text_endpoints` and the deterministic marker counts.

The review UI (`app.py`) stages each extracted table as `data/staging/<sha>.xlsx` + a `.meta.json` sidecar so the review queue survives a Streamlit server restart.

### Layout notes

- `data/` (gitignored): `incoming/<sha256>.pdf` source PDFs, `staging/` pending review, `exports/` approved per-paper XLSX, `master.db`.
- `database/`: `connection.py` (rw + ro conns, applies schema), `*_repo.py` (per-table CRUD), `merge.py` (the one write path), `browse.py` (read views for the DB page). Schema lives in `database/schema.sql` (idempotent, applied on startup) — `migrations/` mirrors the initial cut.
- The 26 extraction columns have exact human-readable names *with units and punctuation* (e.g. `"Rare Earth Elements (REY:La, Ce, Nd)"`, `"Extractant Conc. (mM)"`); they are the DataFrame columns, the SQLite columns, and the XLSX headers. `validation/schema.py` is the single source of truth — coerce/validate through it rather than re-listing columns.
- `docs/curve_extractor_plan.md` documents the deterministic curve extractor's design and validated scope (vector path is trustworthy on clean single-panel figures; raster path remains an estimate).
