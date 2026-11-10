-- Mirror of the 2026-11 schema.sql change for the open-licence merge gate.
-- Nothing executes this file; database/connection.py applies both parts at
-- startup (_ensure_added_columns for the columns, the _VIEWS drop-and-recreate
-- for the view). Kept for the record, like the rest of migrations/.

-- 1. Licence provenance on papers. database/merge.py refuses to merge a paper
--    without an open licence (config.OPEN_LICENSES). Papers merged before this
--    (Swain & Otu 2011) keep a NULL licence; their rows are untouched.
ALTER TABLE papers ADD COLUMN license          TEXT;
ALTER TABLE papers ADD COLUMN license_evidence TEXT;
ALTER TABLE papers ADD COLUMN license_url      TEXT;
ALTER TABLE papers ADD COLUMN license_origin   TEXT;

-- 2. v_paper_summary exposes p.license. See schema.sql for the definition.
