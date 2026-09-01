-- Mirror of the v_paper_summary view added to schema.sql (2026-08-31).
-- Nothing executes this file; schema.sql's CREATE VIEW IF NOT EXISTS creates
-- the view on existing DBs at startup. Kept for the record, per the
-- migrations/ convention.

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
       er.elements,
       er.rows_per_element,
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
