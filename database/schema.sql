-- REE Extraction Dashboard — master schema (see README §5).
-- Idempotent: safe to run on every startup (CREATE ... IF NOT EXISTS).

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;   -- safer concurrent read during a write

-- papers: one row per source paper; the dedup + provenance anchor.
CREATE TABLE IF NOT EXISTS papers (
    paper_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_no      TEXT,
    doi               TEXT UNIQUE,          -- canonicalised, lowercase; dedup key #1
    title             TEXT,
    content_sha256    TEXT UNIQUE NOT NULL, -- hash of PDF bytes; dedup key #2
    original_filename TEXT,
    pdf_path          TEXT NOT NULL,        -- data/incoming/<sha256>.pdf; PDFs are persisted
    figure_type       TEXT,                 -- 'pct_E_vs_pH' | 'logD_vs_conc' | 'other'
    is_raster_figure  INTEGER,              -- 1/0/NULL; set by pdf_inspect
    uploaded_at       TEXT NOT NULL DEFAULT (datetime('now'))
    -- NOTE: no paper-level 'status'; review state lives per-version on prompt_runs.status.
);

-- prompt_runs: one row per extraction ATTEMPT; (paper_id, prompt_version) is the
-- coexistence key — re-running under a new version adds a row, never replaces.
CREATE TABLE IF NOT EXISTS prompt_runs (
    prompt_run_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id         INTEGER NOT NULL REFERENCES papers(paper_id),
    prompt_version   TEXT NOT NULL,         -- e.g. 'extraction_v5.1'
    prompt_sha256    TEXT NOT NULL,         -- hash of the prompt file actually used
    model            TEXT NOT NULL,         -- e.g. 'claude-opus-4-8'
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','approved','rejected')),
    run_timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
    reviewed_at      TEXT,                  -- when approved/rejected; NO reviewer identity
    n_rows_returned  INTEGER,
    qa_passed        INTEGER,               -- 1/0 overall QA verdict
    qa_report_json   TEXT,                  -- serialized QAReport (warnings, flags)
    raw_response     TEXT,                  -- full model output, for audit/replay

    -- Usage/cost telemetry (added post-launch — database.connection adds these
    -- columns to pre-existing DBs at startup; see _ensure_prompt_run_usage_columns).
    input_tokens                    INTEGER,
    output_tokens                   INTEGER,
    cache_creation_input_tokens     INTEGER,
    cache_read_input_tokens         INTEGER
);

-- one approved run per (paper, prompt_version).
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_approved_per_version
    ON prompt_runs(paper_id, prompt_version) WHERE status = 'approved';

-- extractions: THE 26-COLUMN SCHEMA, one row per digitized data point per element.
CREATE TABLE IF NOT EXISTS extractions (
    extraction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        INTEGER NOT NULL REFERENCES papers(paper_id),
    prompt_run_id   INTEGER NOT NULL REFERENCES prompt_runs(prompt_run_id),

    "Reference No."                        TEXT,
    "DOI"                                  TEXT,
    "Treatment"                            TEXT,
    "Sources"                              TEXT,
    "Material Process"                     TEXT,
    "Si (%)"                               REAL,
    "Al (%)"                               REAL,
    "Zn (%)"                               REAL,
    "Fe (%)"                               REAL,
    "Rare Earth Elements (REY:La, Ce, Nd)" TEXT,
    "RRE composition (ppm)"                REAL,
    "RRE composition (mM)"                 REAL,
    "Extractant"                           TEXT,
    "Extractant type"                      TEXT,
    "Extractant Conc. (mM)"                REAL,
    "Molar ratio of EX/REE"                REAL,
    "Extract%"                             REAL,
    "Extract Temperature (oC)"             REAL,
    "pH"                                   REAL,
    "Separation factor (SF%)"              REAL,
    "Acid Solution"                        TEXT,
    "Acid Solution conc. (M)"              REAL,
    "mixing method"                        TEXT,
    "Stripping Temperature (oC)"           REAL,
    "Leaching time (minute)"               REAL,
    "Recovery %"                           REAL,

    merged_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_extractions_run        ON extractions(prompt_run_id);
CREATE INDEX IF NOT EXISTS idx_extractions_paper      ON extractions(paper_id);
CREATE INDEX IF NOT EXISTS idx_extractions_element    ON extractions("Rare Earth Elements (REY:La, Ce, Nd)");
CREATE INDEX IF NOT EXISTS idx_extractions_extractant ON extractions("Extractant");
CREATE INDEX IF NOT EXISTS idx_extractions_ph         ON extractions("pH");

-- text_endpoints: the paper's STATED numeric claims, captured by extraction_v5.1.
-- A QA anchor (README §9), deliberately NOT part of the 26-column extractions table.
CREATE TABLE IF NOT EXISTS text_endpoints (
    endpoint_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        INTEGER NOT NULL REFERENCES papers(paper_id),
    prompt_run_id   INTEGER NOT NULL REFERENCES prompt_runs(prompt_run_id),
    element         TEXT NOT NULL,
    x_value         REAL,
    x_basis         TEXT,        -- 'pH' | 'extractant_conc_mM' | ...
    y_value         REAL,
    y_metric        TEXT,        -- 'Extract%' | 'logD' | 'Recovery %' | ...
    source_quote    TEXT
);
CREATE INDEX IF NOT EXISTS idx_text_endpoints_run ON text_endpoints(prompt_run_id);

-- review_log: append-only audit of every approve/edit/reject. NO reviewer identity
-- (single shared write password => actor is unknowable).
CREATE TABLE IF NOT EXISTS review_log (
    review_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id         INTEGER NOT NULL REFERENCES papers(paper_id),
    prompt_run_id    INTEGER NOT NULL REFERENCES prompt_runs(prompt_run_id),
    action           TEXT NOT NULL CHECK (action IN ('approve','edit','reject')),
    note             TEXT,
    edited_diff_json TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- v_current_best: the rows Pillars B & C read. For each paper, the extractions from
-- the MOST RECENT prompt_version that has an APPROVED run. Coexistence resolved here.
CREATE VIEW IF NOT EXISTS v_current_best AS
WITH best_run AS (
    SELECT pr.paper_id,
           pr.prompt_run_id,
           ROW_NUMBER() OVER (
               PARTITION BY pr.paper_id
               ORDER BY pr.prompt_version DESC, pr.reviewed_at DESC
           ) AS rn
    FROM prompt_runs pr
    WHERE pr.status = 'approved'
)
SELECT e.*
FROM extractions e
JOIN best_run b
  ON e.prompt_run_id = b.prompt_run_id
WHERE b.rn = 1;
