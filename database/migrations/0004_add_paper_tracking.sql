-- Mirror of the paper tracking-sheet changes made to schema.sql (2026-09-01).
-- Nothing executes this file; database/connection.py applies both parts at
-- startup (_ensure_added_columns for the columns, the _VIEWS drop-and-recreate
-- for the view redefinition). Kept for the record, like the rest of migrations/.

-- 1. Reviewer-entered tracking fields on papers (mirror the external
--    "REE Paper Tracking" spreadsheet; not derivable from the extracted data).
ALTER TABLE papers ADD COLUMN short_citation    TEXT;
ALTER TABLE papers ADD COLUMN pub_year          TEXT;
ALTER TABLE papers ADD COLUMN figures_used      TEXT;
ALTER TABLE papers ADD COLUMN known_issues      TEXT;
ALTER TABLE papers ADD COLUMN short_description TEXT;

-- 2. v_paper_summary redefined to expose the tracking fields plus derived
--    status ('Extracted'/'Uploaded'), date_processed, and n_elements.
--    See schema.sql for the authoritative definition.
