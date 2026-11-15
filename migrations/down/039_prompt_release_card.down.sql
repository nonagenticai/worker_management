-- DOWN for migrations/039_prompt_release_card.sql. Applied BY HAND only
-- (psql -1 -f); the startup runner never descends into this directory.
--
-- Roll the code back to pre-039 FIRST (039-aware code writes actor_card_id).
-- ⚠️ DESTRUCTIVE: dropping the column loses which board card authorised each
-- release since 039. The ledger is append-only, but DROP COLUMN is DDL, not a
-- row UPDATE/DELETE, so the triggers do not fire. 039's schema_migration_files
-- row is removed so the runner re-applies it on the next start.

BEGIN;

ALTER TABLE prompt_version_transitions
    DROP CONSTRAINT IF EXISTS prompt_version_transitions_release_card_check;
ALTER TABLE prompt_version_transitions
    DROP CONSTRAINT IF EXISTS prompt_version_transitions_actor_card_id_format_check;
ALTER TABLE prompt_version_transitions
    DROP COLUMN IF EXISTS actor_card_id;

DELETE FROM schema_migration_files WHERE filename = '039_prompt_release_card.sql';

COMMIT;
