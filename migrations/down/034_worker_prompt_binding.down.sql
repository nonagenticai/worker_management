-- DOWN for migrations/034_worker_prompt_binding.sql. Applied BY HAND only
-- (psql -1 -f); the startup runner lists top-level migrations/*.sql and never
-- descends into this directory.
--
-- ⚠️ DESTRUCTIVE: drops every worker -> prompt binding. prompt_versions and its
-- ledger are untouched (034 never altered them). Set WM_PROMPT_SOURCE=legacy
-- FIRST: with the flag on and the table gone, every platform message pays a
-- failed resolver read (it falls back to the config prompt and counts the
-- error, but it is noise). 034's schema_migration_files row is removed so the
-- runner re-applies it on the next start.

BEGIN;

DROP TABLE IF EXISTS worker_prompt_binding;

DELETE FROM schema_migration_files WHERE filename = '034_worker_prompt_binding.sql';

COMMIT;
