# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit web app that turns rare-earth-element (REE) solvent-extraction research papers (PDFs) into a queryable SQLite database. One writer uploads PDFs; an LLM extraction pipeline produces one row-per-data-point table against a fixed **26-column schema**; after automatic QA and human review, rows merge into a master DB that two read-only consumers (a calculator and an AI assistant) sit on top of.

Two design documents, referenced by section number throughout the code:

- `plan.md` is the original pre-implementation design plan — authoritative on *intent*. Code comments saying `README §6` etc. refer to **plan.md's** sections (the plan used to live in README.md; README.md is now a public-facing overview with no section numbers). The code has moved past the plan in places (e.g. the pinned prompt is now `extraction_v7`, not `extraction_v5.1`). Trust the code for current behavior; use the plan for rationale.
- `docs/curve_extractor_plan.md` is the deterministic curve extractor's design doc — comments saying `plan §4.2` etc. refer to *its* sections.

## Commands

```bash
# Run the app (multi-page: home = extraction, pages/ = DB / Calculator / Assistant)
streamlit run app.py

# Tests (pure/offline — no API calls, no network)
python -m pytest                    # full suite (~115 tests)
python -m pytest tests/test_solve.py            # one file
python -m pytest tests/test_checks.py -k monotonicity   # one test by name
```

There is no lint/format config and no `pytest.ini` — `conftest.py` just puts the repo root on `sys.path`. Dependencies: `pip install -r requirements.txt` (a `.venv/` already exists).

## Environment

Config is read once in `config.py` from `.env` at the repo root. Keys: `ANTHROPIC_API_KEY` (required for extraction/assistant, not for tests), `REQUIRE_PASSWORD`/`WRITE_PASSWORD` (write gate), `EXTRACTION_PROMPT_VERSION` (default `extraction_v7`), `EXTRACTION_MODEL` (`claude-opus-4-8`), `ASSISTANT_MODEL` (`claude-haiku-4-5-...`), `EXTRACTION_TASK_BUDGET_TOKENS` (default `100000` — a self-moderated backstop on total tokens per extraction call, not a hard cap). Never hardcode paths or model IDs — add them to `config.py`.

## Architecture: three pillars over one DB

The master SQLite DB (`data/master.db`) is the hub. **Pillar A is the only writer; Pillars B and C are strictly read-only** and query through the `v_current_best` view, never the raw `extractions` table.

- **Pillar A — extraction** (`app.py` home page, `ingestion/`, `extraction/`, `validation/`, `database/`): upload → dedup → LLM extract → QA → human review → merge.
- **Pillar B — calculator** (`pages/2_Calculator.py`, `calculator/`): deterministic ppm↔mM / molar-ratio / concentration math, cross-checked against DB typical ranges.
- **Pillar C — assistant** (`pages/3_Lab_Assistant.py`, `assistant/`): Anthropic tool-calling agent that answers questions by querying the DB and calling the calculator — never by inventing numbers.

### Key invariants (violating these breaks the design)

- **Read/write split is enforced two ways.** Write actions call `auth.require_write_access()` first; read consumers open the DB via `connection.get_readonly_conn()` (SQLite `mode=ro`). The assistant's `query_database` tool has a *second* independent backstop, `assistant/sql_guard.py`, which rejects anything but a single SELECT/CTE over a table whitelist even if the model is jailbroken. When touching the assistant's DB access, keep both layers.
- **Prompt versioning = reproducibility.** The extraction prompt is a pinned, versioned file in `prompts/extraction_vN.md`, loaded only through `extraction/prompt_loader.py`, which records the version + sha256 on every run. Improving extraction means dropping in a new `prompts/` file, adding a `prompts/CHANGELOG.md` entry, and bumping `EXTRACTION_PROMPT_VERSION` — **no pipeline code changes**. A version with a recorded `prompt_runs` row is immutable: never edit it, fork a new version. Never hardcode prompt text elsewhere.
- **Coexistence, not replacement.** Re-extracting a paper under a new prompt version inserts a new `prompt_runs` row; it never overwrites old data. `v_current_best` (defined in `database/schema.sql`) resolves "latest approved prompt version per paper" at query time, so old and new prompt versions coexist without migration. Consumers must read `v_current_best`.
- **The DB is only written on human approval,** through the single atomic transaction in `database/merge.py::commit_extraction`. `extraction/runner.py` produces an `ExtractionResult` and touches nothing persistent. `review_log` is append-only and records no reviewer identity (single shared password ⇒ actor unknowable).

### Extraction flow (`extraction/runner.py`)

Shared pre-pass + parse + QA logic (`runner._prepass`, `runner._postprocess`) backs two entry points: `extract_paper()` (synchronous, blocks until done) and `submit_batch()` / `batch_status()` / `collect_batch()` (the Anthropic Message Batches API — 50% cheaper, asynchronous; opt-in via a toggle in `app.py`, default off since code execution + Files API + task budgets inside a batched request is unverified against the live API — test on 1-2 papers before trusting it for a full run).

1. Load pinned prompt (`prompt_loader`).
2. **Deterministic curve pre-pass** (`extraction/curve_prepass.py` → `extraction/curve_extractor/`): counts figure markers from the PDF's own vector geometry, purely and with no API call, and injects the counts as a grounding "ground truth" block into the prompt + a QA cross-check anchor. It only marks a page *authoritative* when series counts are internally uniform (a clean single-panel figure); multi-panel and raster figures are downgraded to hints. A failure here degrades to "no anchor" and must never block extraction.
3. Anthropic API call (`extraction/anthropic_client.py`) → parse (`extraction/parse_output.py`). The client uploads the PDF once via the Files API and hands it to the model **two ways in one turn**: a `document` block (visual reading of legends/markers) and a `container_upload` for the **code-execution tool**, where the model runs pdfplumber/numpy itself for axis calibration and point digitization. Both the synchronous and Batch API paths build their request from the same `_message_kwargs()`, which sets: adaptive thinking; **two prompt-cache breakpoints** — one on the system prompt (1h TTL) and one on the last user-turn block after the PDF (5min TTL), so the PDF is reused rather than rebilled across the model's internal code-execution tool-loop iterations; and a `task_budget` (`output_config`, beta `task-budgets-2026-03-13`) as a self-moderated backstop on total spend. Token usage (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`) comes back on `ExtractResponse`/`ExtractionResult` and is persisted on `prompt_runs` — a high `cache_read_input_tokens` share confirms the cache breakpoint is working.
4. **QA checks** (`validation/checks.py`): pure, unit-tested functions producing a `QAReport` of RED (merge-gated) / AMBER (warning) flags — row-count sanity, axis bounds, monotonicity, duplicates, vocabulary drift, and cross-checks against both the paper's stated `text_endpoints` and the deterministic marker counts.

The review UI (`app.py`) stages each extracted table as `data/staging/<sha>.xlsx` + a `.meta.json` sidecar so the review queue survives a Streamlit server restart. A submitted-but-not-yet-collected Batch API job gets its own sidecar (`_batch_<id>.batch.json`) so it survives a restart too — check status from the "Batch API jobs" section to fold results into the normal staging queue once the batch ends.

### Layout notes

- `data/` (gitignored): `incoming/<sha256>.pdf` source PDFs, `staging/` pending review, `exports/` approved per-paper XLSX, `master.db`.
- `database/`: `connection.py` (rw + ro conns, applies schema), `*_repo.py` (per-table CRUD), `merge.py` (the one write path), `browse.py` (read views for the DB page). Schema lives in `database/schema.sql` (idempotent `CREATE ... IF NOT EXISTS`, applied on startup) — `migrations/` mirrors changes for the record but nothing executes those files directly. Because SQLite's `ALTER TABLE` has no `ADD COLUMN IF NOT EXISTS`, a column added to an existing table (e.g. `prompt_runs`'s usage-tracking columns, added via `connection._ensure_prompt_run_usage_columns`) needs a matching `PRAGMA table_info` + `ALTER TABLE` guard in `connection.py`, not just a `schema.sql` edit, or pre-existing DBs never pick it up — follow that pattern for the next column addition.
- The 26 extraction columns have exact human-readable names *with units and punctuation* (e.g. `"Rare Earth Elements (REY:La, Ce, Nd)"`, `"Extractant Conc. (mM)"`); they are the DataFrame columns, the SQLite columns, and the XLSX headers. `validation/schema.py` is the single source of truth — coerce/validate through it rather than re-listing columns. The *wire format* the model returns them in is separate and versioned in the prompt: `extraction_v7`+ uses a compact positional `{"columns": [...], "rows": [[...], ...]}` shape (cheaper — 26 keys once, not once per row) rather than v5.1–v6's one-object-per-row shape; `extraction/parse_output.py` accepts both.
- `docs/curve_extractor_plan.md` documents the deterministic curve extractor's design and validated scope (vector path is trustworthy on clean single-panel figures; raster path remains an estimate).
