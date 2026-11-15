-- DOWN for migrations/033_prompt_version_transitions.sql. Applied BY HAND only
-- (psql -1 -f); the startup runner lists top-level migrations/*.sql and never
-- descends into this directory.
--
-- ⚠️ DESTRUCTIVE: drops the transition ledger and every row in it. prompt_versions
-- is untouched (033 never altered it). The code that writes the ledger must be
-- rolled back FIRST, or every prompt write route fails on the missing table.
-- 033's schema_migration_files row is removed so the runner re-applies it (with
-- a fresh creation backfill) on the next start.
--
-- The append-only triggers have no opt-out, so dropping them first (below) is
-- what lets the DROP TABLE proceed without relying on any session setting.

BEGIN;

DO $$
BEGIN
    IF to_regclass('prompt_version_transitions') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS prompt_version_transitions_no_truncate_trg
            ON prompt_version_transitions;
        DROP TRIGGER IF EXISTS prompt_version_transitions_append_only_trg
            ON prompt_version_transitions;
    END IF;
END
$$;
DROP TABLE IF EXISTS prompt_version_transitions;
DROP FUNCTION IF EXISTS prompt_version_transitions_append_only();

DELETE FROM schema_migration_files WHERE filename = '033_prompt_version_transitions.sql';

COMMIT;
