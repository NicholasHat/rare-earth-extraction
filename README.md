# REE Extraction Dashboard

![status](https://img.shields.io/badge/status-actively%20developing-brightgreen)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![tests](https://img.shields.io/badge/tests-215%20passing-brightgreen)

A web app that turns rare-earth-element (REE) solvent-extraction **research
papers into a structured, queryable database** — digitizing the data locked
inside their figures, then serving it through a bench calculator and a
natural-language assistant. The automation is treated as *untrusted*: every
number is versioned, QA-checked, and human-reviewed before it's trusted.

> **Status:** the full spine is working and tested — ingestion, a versioned
> LLM extraction pipeline with a deterministic figure pre-pass, an automatic
> QA suite, human review & merge, a calculator, and a tool-calling assistant.
> Current work is live validation on real papers and widening what the
> deterministic pre-pass can vouch for — see the [roadmap](#roadmap).

Built to solve a real problem for a UCI chemistry lab (thousands of papers,
none of them machine-readable) and, along the way, to work through the hard
parts of a trustworthy LLM data pipeline: reproducibility, provenance, QA
gating, and a strict read/write boundary.

## What it does

- 📄 **Digitizes figures into data** — upload research-paper PDFs; a versioned
  prompt drives Claude to turn each *% Extraction vs. pH* / *log D vs.
  [extractant]* plot into a fixed 26-column table, one row per data point per
  element
- 🔎 **Grounds the model against itself** — a deterministic pre-pass counts
  figure markers straight from the PDF's own vector geometry, calibrates the
  axes, and hands the model both the counts and the (x, y) coordinates as
  ground truth, directly attacking silent under-digitization and the cost of
  the model doing that work itself
- ✅ **Auto-QAs every extraction** — row-count sanity, axis-range bounds,
  monotonicity, duplicates, vocabulary drift, and cross-checks against the
  paper's own stated numbers; failures **gate the merge**
- 👤 **Keeps a human in the loop** — approve / edit / reject each result in an
  editable grid; data enters the database only on approval, via one atomic
  transaction with a full audit log
- 🧮 **Does the bench math** — an open calculator handles ppm ↔ mM /
  molar-ratio / concentration conversions, sanity-checked against the dataset,
  and predicts Extract% at a target pH by interpolating inside prior papers'
  measured sweeps (never extrapolating)
- 💬 **Answers questions in plain English** — a tool-calling assistant queries
  the database and calls the calculator, and is architecturally prevented from
  inventing numbers

## How it works

```
   Upload PDF/s
        │
   ┌────▼──────────────────────────────────────┐
   │  PILLAR A · Extraction pipeline (write)    │
   │  ingest → dedup → LLM extract → QA → review│
   └────┬──────────────────────────────────────┘
        │ merge  (only on human approval, one transaction)
   ┌────▼──────────────────────────────────────┐
   │  MASTER DATABASE (SQLite)                  │
   │  papers · prompt_runs · extractions        │
   │  text_endpoints · review_log               │
   │  → v_current_best (latest approved version)│
   └────┬───────────────────────┬───────────────┘
        │ read-only             │ read-only
   ┌────▼──────────┐      ┌──────▼───────────────┐
   │ PILLAR B      │      │ PILLAR C             │
   │ Calculator    │      │ AI assistant         │
   │ bench math +  │      │ tool-calling agent   │
   │ DB sanity     │      │ over read-only SQL   │
   └───────────────┘      └──────────────────────┘
```

The master database is the hub. **Pillar A is the only writer; B and C are
strictly read-only consumers** that query through a SQL view
(`v_current_best`), never the raw table. Adding a new capability to the
assistant is just writing a function and registering it as a tool.

## Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM | Anthropic API — Claude Sonnet 5 (extraction, with code execution + Files API; Batch API optional) · Claude Haiku (assistant, tool-calling) |
| UI | Streamlit (multi-page) |
| Database | SQLite (read-only mode for consumers) + a `v_current_best` view |
| PDF / figures | pdfplumber · pypdf · NumPy / SciPy / scikit-image (deterministic curve extraction) |
| Data | pandas · openpyxl |
| Tests | pytest (offline, no API calls) |

## Quickstart

```bash
# 1. Set up the project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY

# 2. Run
streamlit run app.py
```

The app opens on the extraction page; the sidebar links to the Database
browser, Calculator, and Lab Assistant. The SQLite database and data
directories are created automatically on first run.

## Engineering notes

A few decisions I made deliberately, and why:

- **Reproducibility by construction.** The extraction prompt is a pinned,
  versioned file loaded through a single seam (`prompt_loader`); every run
  records the exact prompt version and its SHA-256. Improving extraction means
  dropping in a new prompt file and bumping one config value — **no pipeline
  code changes**.
- **Coexistence, not migration.** Re-extracting a paper under a new prompt
  version *adds* rows rather than replacing them; a SQL view resolves "latest
  approved version per paper" at query time. Old and new results coexist with
  zero migration and full A/B comparability.
- **Defense in depth on the read/write boundary.** Writes pass a password gate;
  consumers open the DB in SQLite read-only mode; and the assistant's SQL tool
  has a *second, independent* guard that rejects anything but a single
  whitelisted `SELECT` — so even a jailbroken model **cannot mutate data**.
- **Deterministic where it counts.** Figure-marker counting and every unit
  conversion are pure, unit-tested Python, leaving the LLM only the genuinely
  visual work — and making the whole QA layer testable without spending API
  tokens.

## Tests

```bash
python -m pytest        # 215 tests, no live model or network required
```

The suite covers unit conversions, the solve-for-the-blank calculator and its
Extract% prediction, the QA checks, dedup, the atomic merge transaction and
version supersession, the read-only SQL guard, the deterministic curve
extractor and pre-pass, the extraction client's continuation and batch
handling, the review-queue persistence, the tracking sheet, and the
assistant's tools.

## Roadmap

**Working today**

- [x] SQLite schema, repositories, and the `v_current_best` coexistence view
- [x] Versioned, SHA-pinned extraction prompts with full run provenance
- [x] Extraction pipeline — single + batch upload, staging queue that survives restarts
- [x] Content-hash + DOI de-duplication before extraction
- [x] Full automatic QA suite with red/amber flags and merge gating
- [x] Deterministic vector-figure curve pre-pass as a grounding anchor (`extraction_v6`)
      and as a digitizer handing the model pre-calibrated coordinates (`extraction_v8`)
- [x] Human review / edit / merge with an append-only audit log, plus an on-demand
      "re-extract with QA feedback" middle path
- [x] Message Batches API path (50% cheaper) with transparent continuation of paused items
- [x] Per-run token/cost telemetry and prompt caching across the code-execution loop
- [x] Bench calculator with database sanity-checking and Extract% prediction (Pillar B)
- [x] Tool-calling AI assistant over a guarded read-only SQL tool (Pillar C)
- [x] Tracking-sheet view mirroring the lab's paper-tracking spreadsheet, with per-paper exports

**Building next**

- [ ] **Live validation** of the 2026-07/08 changes that have only unit tests so far —
      `extraction_v9`'s log-log sweep recovery, the panel-merge gate, QA-feedback
      re-extraction, and the scikit-image raster path — then tuning the QA
      tolerances (text-endpoint %E / pH thresholds) on the first dozen papers
- [ ] **Monochrome figures as ground truth** — stroked-glyph markers (×/+/✶) are
      now assembled but need an oracle paper before the pre-pass can vouch for them
- [ ] **Axis tick-label reading** success rate on real papers — the coordinate
      hand-off only fires when both axes calibrate
- [ ] **Reliable per-series counts on raster figures** — the vector path is
      ground truth today; the raster CV path is still a lower-confidence estimate
- [ ] Bulk / selective re-extraction workflow when a new prompt version ships
- [ ] Real per-user authentication (the current shared-password gate is a
      convenience gate, not real auth) for multi-user lab rollout

## Project documents

- **[plan.md](plan.md)** — the original end-to-end design: architecture
  rationale, database schema, per-pillar build plan, and QA strategy.
- **[CLAUDE.md](CLAUDE.md)** — an orientation guide for working in the codebase.
