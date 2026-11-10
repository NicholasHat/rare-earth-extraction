-- Mirror of the 2026-11 schema.sql change for extraction_v15. Nothing executes
-- this file; database/connection.py adds the columns at startup
-- (_ensure_added_columns, from validation/schema.py ADDED_COLUMNS). Kept for
-- the record, like the rest of migrations/.

-- Nine extraction columns appended after the lab's 26. On a pre-existing DB
-- they land after merged_at; consumers read columns by name. Rows merged
-- before this stay null in them. v_current_best selects e.*, so it carries them.
ALTER TABLE extractions ADD COLUMN "Diluent"                  TEXT;
ALTER TABLE extractions ADD COLUMN "Co-extractant"            TEXT;
ALTER TABLE extractions ADD COLUMN "Co-extractant Conc. (mM)" REAL;
ALTER TABLE extractions ADD COLUMN "Phase ratio (O/A)"        REAL;
ALTER TABLE extractions ADD COLUMN "Contact time (minute)"    REAL;
ALTER TABLE extractions ADD COLUMN "Distribution ratio (D)"   REAL;
ALTER TABLE extractions ADD COLUMN "Total REE feed (mM)"      REAL;
ALTER TABLE extractions ADD COLUMN "pH type"                  TEXT;
ALTER TABLE extractions ADD COLUMN "Figure/Table"             TEXT;
