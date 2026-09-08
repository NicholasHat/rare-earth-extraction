-- Mirror of the v_current_best redefinition made to schema.sql on 2026-08-10
-- (commit 24420f0), recorded late: it predates 0003/0004 but had no migration
-- file at the time. Nothing executes this file; database/connection.py drops
-- and recreates every view in connection._VIEWS from schema.sql at startup.
--
-- The view used to rank a paper's approved runs by prompt_version, which
-- sorts as TEXT: 'extraction_v9' > 'extraction_v10'. It now ranks by
-- reviewed_at (prompt_run_id breaks same-second ties), so version strings
-- need no parsing and re-approving an older version rolls a paper back.
-- See schema.sql for the authoritative definition.

DROP VIEW IF EXISTS v_current_best;
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
