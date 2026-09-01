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
    uploaded_at       TEXT NOT NULL DEFAULT (datetime('now')),

    -- Tracking-sheet fields (reviewer-entered at approval; mirror the external
    -- "REE Paper Tracking" spreadsheet — not derivable from the extracted data.
    -- Added post-launch: database.connection adds these to pre-existing DBs at
    -- startup; see _ensure_added_columns).
    short_citation    TEXT,                 -- e.g. 'Swain & Otu'; also names the export file
    pub_year          TEXT,                 -- publication year, e.g. '2011'
    figures_used      TEXT,                 -- e.g. 'Fig. 2, Fig. 4'
    known_issues      TEXT,                 -- caveats spotted during review
    short_description TEXT                  -- one-paragraph summary of the paper/extraction
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
    -- columns to pre-existing DBs at startup; see _ensure_added_columns).
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
-- its MOST RECENTLY APPROVED run. Coexistence resolved here.
--
-- Ordering is by approval time, not by prompt_version: prompt_version is TEXT, so
-- sorting it puts 'extraction_v9' above 'extraction_v10' (and any dotted-version
-- arithmetic collides v5.1 with v5.10). Approval time needs no parsing, assumes
-- nothing about the version-string format, and lets a reviewer roll back a
-- regressed version by re-approving an older one. reviewed_at has second
-- resolution, so prompt_run_id breaks ties within the same second.
CREATE VIEW IF NOT EXISTS v_current_best AS
WITH best_run AS (
    SELECT pr.paper_id,
           pr.prompt_run_id,
           ROW_NUMBER() OVER (
               PARTITION BY pr.paper_id
               ORDER BY pr.reviewed_at DESC, pr.prompt_run_id DESC
           ) AS rn
    FROM prompt_runs pr
    WHERE pr.status = 'approved'
)
SELECT e.*
FROM extractions e
JOIN best_run b
  ON e.prompt_run_id = b.prompt_run_id
WHERE b.rn = 1;

-- v_paper_summary: one row per paper — the "have I already done this paper?"
-- dedup sheet. Aggregates the paper's CURRENT-BEST data (via v_current_best, so
-- it always reflects the same rows Pillars B & C see): which elements were
-- studied and how many rows each, which extractants/types, and the pH range.
-- LEFT JOINs from `papers` so an uploaded-but-not-yet-approved paper still
-- appears (with NULL data columns) — it too shouldn't be re-extracted.
-- GROUP_CONCAT(DISTINCT x) can't take a separator, so distinctness comes from
-- the dedup subqueries instead.
CREATE VIEW IF NOT EXISTS v_paper_summary AS
WITH per_element AS (
    SELECT paper_id,
           COALESCE("Rare Earth Elements (REY:La, Ce, Nd)", '?') AS element,
           COUNT(*) AS n_rows
    FROM v_current_best
    GROUP BY paper_id, element
),
element_rollup AS (
    SELECT paper_id,
           GROUP_CONCAT(element, ', ' ORDER BY element) AS elements,
           GROUP_CONCAT(element || ' (' || n_rows || ')', ', ' ORDER BY element)
               AS rows_per_element,
           COUNT(*) AS n_elements,
           SUM(n_rows) AS data_rows
    FROM per_element
    GROUP BY paper_id
),
extractant_rollup AS (
    SELECT paper_id, GROUP_CONCAT(x, ', ' ORDER BY x) AS extractants
    FROM (SELECT DISTINCT paper_id, "Extractant" AS x
          FROM v_current_best WHERE "Extractant" IS NOT NULL)
    GROUP BY paper_id
),
type_rollup AS (
    SELECT paper_id, GROUP_CONCAT(x, ', ' ORDER BY x) AS extractant_types
    FROM (SELECT DISTINCT paper_id, "Extractant type" AS x
          FROM v_current_best WHERE "Extractant type" IS NOT NULL)
    GROUP BY paper_id
),
paper_rollup AS (
    SELECT paper_id,
           MAX(prompt_run_id) AS prompt_run_id,   -- all current-best rows share one run
           MIN("pH") AS ph_min,
           MAX("pH") AS ph_max
    FROM v_current_best
    GROUP BY paper_id
)
SELECT p.paper_id,
       p.doi,
       p.title,
       p.reference_no,
       p.short_citation,
       p.pub_year,
       p.figures_used,
       p.known_issues,
       p.short_description,
       CASE WHEN pu.paper_id IS NOT NULL THEN 'Extracted' ELSE 'Uploaded' END
           AS status,
       DATE(pr.reviewed_at) AS date_processed,
       er.elements,
       er.rows_per_element,
       er.n_elements,
       er.data_rows,
       xr.extractants,
       tr.extractant_types,
       pu.ph_min,
       pu.ph_max,
       pr.prompt_version,
       pr.reviewed_at AS approved_at,
       p.uploaded_at
FROM papers p
LEFT JOIN element_rollup    er ON er.paper_id = p.paper_id
LEFT JOIN extractant_rollup xr ON xr.paper_id = p.paper_id
LEFT JOIN type_rollup       tr ON tr.paper_id = p.paper_id
LEFT JOIN paper_rollup      pu ON pu.paper_id = p.paper_id
LEFT JOIN prompt_runs       pr ON pr.prompt_run_id = pu.prompt_run_id
ORDER BY p.paper_id;
