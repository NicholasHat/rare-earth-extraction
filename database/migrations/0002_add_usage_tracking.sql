-- Migration 0002 — usage/cost telemetry on prompt_runs.
--
-- The columns below are now part of prompt_runs' CREATE TABLE in ../schema.sql
-- (so new DBs get them for free). SQLite's ALTER TABLE has no ADD COLUMN IF
-- NOT EXISTS, so a DB created before this change gets these columns added in
-- Python instead — see database.connection._ensure_prompt_run_usage_columns,
-- run on every init_db() / get_conn(). This file is a record of the change,
-- not separately executed — see 0001_init.sql for the same note.

ALTER TABLE prompt_runs ADD COLUMN input_tokens INTEGER;
ALTER TABLE prompt_runs ADD COLUMN output_tokens INTEGER;
ALTER TABLE prompt_runs ADD COLUMN cache_creation_input_tokens INTEGER;
ALTER TABLE prompt_runs ADD COLUMN cache_read_input_tokens INTEGER;
