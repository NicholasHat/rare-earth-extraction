# REE Extraction Dashboard

> Turn rare-earth-element solvent-extraction research papers into a structured, queryable database — with an LLM extraction pipeline that is reproducible, auditable, and human-reviewed before anything is trusted.

A full-stack data application that ingests scientific PDFs, digitizes the experimental data locked inside their figures into a fixed 26-column schema, and exposes it through a calculator and a natural-language assistant. Built in Python with Streamlit, SQLite, and the Anthropic API.

---

## The problem

Rare-earth solvent-extraction data lives inside thousands of journal papers — usually as *% Extraction vs. pH* or *log D vs. [extractant]* plots that a researcher can read but a computer cannot. Manually transcribing those curves is slow, error-prone, and unrepeatable. This project automates the extraction while treating the automation as *untrusted*: every number is versioned, QA-checked, and gated behind human review before it enters the dataset.

## What it does

- **Extract** — Upload one or more PDFs. A versioned prompt drives the Anthropic API to digitize each figure into a 26-column table (one row per data point per element), with a deterministic pre-pass that counts figure markers straight from the PDF's vector geometry to anchor the model against under-digitization.
- **Validate** — Every extraction runs through a suite of automatic QA checks (row-count sanity, axis-range bounds, monotonicity, duplicate detection, vocabulary drift, and cross-checks against the paper's own stated numeric claims). Failures are surfaced as red/amber flags and *gate* the merge.
- **Review & merge** — A human approves, edits, or rejects each result in an editable grid. Only on approval do rows enter the master database, through a single atomic transaction with a full audit log.
- **Calculate** — An open, no-login calculator does the ppm ↔ mM / molar-ratio / concentration bench math, sanity-checked against typical ranges in the accumulated data.
- **Ask** — A tool-calling AI assistant answers natural-language questions by querying the database and calling the calculator — never by inventing numbers.

## Architecture

The app is organized as **three pillars over one SQLite database**. Pillar A is the *only* writer; Pillars B and C are strictly read-only consumers.

```
                    ┌─────────────────────────────────────────────┐
   Upload PDF  ──►  │  A · EXTRACTION PIPELINE (write, gated)      │
                    │  ingest → dedup → LLM extract → QA → review  │
                    └───────────────────────┬─────────────────────┘
                                            │ merge (on human approval)
                                            ▼
                    ┌─────────────────────────────────────────────┐
                    │           MASTER DATABASE (SQLite)           │
                    │  papers · prompt_runs · extractions          │
                    │  text_endpoints · review_log                 │
                    │  → v_current_best  (latest approved version) │
                    └───────────┬────────────────────┬────────────┘
                                │ read-only          │ read-only
                    ┌───────────▼──────────┐ ┌───────▼──────────────┐
                    │ B · CALCULATOR       │ │ C · AI ASSISTANT     │
                    │ deterministic bench  │ │ tool-calling agent   │
                    │ math + DB sanity     │ │ over read-only SQL   │
                    └──────────────────────┘ └──────────────────────┘
```

**Data flow:** `upload → ingest/dedup → extract (versioned prompt) → validate/QA → human review → merge → {query, calculate}`.

## Engineering highlights

- **Reproducibility by construction.** The extraction prompt is a pinned, versioned file (`prompts/extraction_v6.md`) loaded through one seam; every run records the exact prompt version and its SHA-256. Improving extraction means dropping in a new prompt file and bumping one config value — no pipeline code changes.
- **Coexistence, not migration.** Re-extracting a paper under a new prompt version *adds* rows rather than replacing them. A SQL view (`v_current_best`) resolves "latest approved version per paper" at query time, so old and new results coexist with zero migration and full A/B comparability.
- **Defense in depth on the read/write boundary.** Writes pass a password gate; read consumers open the database in SQLite read-only mode; and the assistant's SQL tool has a *second, independent* guard that rejects anything but a single whitelisted `SELECT` — so even a jailbroken model cannot mutate data.
- **Deterministic where it counts.** Figure-marker counting and all unit conversions are pure, unit-tested Python, leaving the LLM only the genuinely visual work. QA checks are likewise pure functions — 113 tests run offline with no API calls.
- **Human-in-the-loop, audited.** Nothing is trusted automatically. The database is written only on explicit approval, through one atomic transaction, with an append-only review log and an explicit override path for merging past red flags.

## Tech stack

Python · Streamlit (multi-page UI) · SQLite · Anthropic API (Claude) · pandas · pdfplumber / pypdf · NumPy / SciPy (deterministic curve extraction) · pytest.

## Getting started

```bash
pip install -r requirements.txt

# Configure secrets/settings (copy and edit)
cp .env.example .env    # set ANTHROPIC_API_KEY; optional REQUIRE_PASSWORD / WRITE_PASSWORD

streamlit run app.py
```

The app opens on the extraction page; the sidebar links to the Database browser, Calculator, and Lab Assistant. The database and data directories are created automatically on first run.

## Testing

```bash
python -m pytest            # full suite, offline (no API calls, no network)
```

The suite covers unit conversions, the solve-for-the-blank calculator, QA checks, dedup, the merge transaction, the SQL guard, the deterministic curve pre-pass, and the assistant's tools.

## Repository layout

| Path | Responsibility |
|------|----------------|
| `app.py` | Extraction, review & merge UI (Pillar A) |
| `pages/` | Database browser, Calculator, Lab Assistant |
| `ingestion/` | PDF upload, hashing, DOI parsing, dedup |
| `extraction/` | Prompt loader, Anthropic client, output parsing, deterministic curve extractor |
| `validation/` | 26-column schema, automatic QA checks, controlled vocabularies |
| `database/` | Connections, per-table repositories, the atomic merge, `schema.sql` |
| `calculator/` | Pure REE unit-conversion and solve engine |
| `assistant/` | Tool-calling agent + independent read-only SQL guard |
| `prompts/` | Versioned extraction prompts + changelog |
| `tests/` | Offline test suite |

## Design document

The original end-to-end design — architecture rationale, database schema, per-pillar build plan, and QA strategy — is preserved in **[plan.md](plan.md)**. `CLAUDE.md` provides an orientation guide for working in the codebase.
