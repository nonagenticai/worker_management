-- DOWN for migrations/035_prompt_owner.sql. Applied BY HAND only (psql -1 -f);
-- the startup runner lists top-level migrations/*.sql and never descends into
-- this directory.
--
-- ⚠️ REMOVES THE DB GUARANTEE that a worker binding's tenant owns its prompt.
-- Roll the code back to pre-035 FIRST: 035-aware code writes
-- prompt_versions.owner_tenant_id and prompt_owner, and reads prompt_owner in
-- the resolver (which then falls back to the config prompt on every call and
-- counts wm_prompt_resolve_errors_total). Prompt versions and bindings
-- themselves are kept; only ownership is dropped. 035's schema_migration_files
-- row is removed so the runner re-applies it (and re-backfills every prompt to
-- the system tenant) on the next start.

BEGIN;

ALTER TABLE IF EXISTS worker_prompt_binding
    DROP CONSTRAINT IF EXISTS worker_prompt_binding_owner_fk;

DROP TRIGGER IF EXISTS prompt_versions_fill_owner ON prompt_versions;
DROP FUNCTION IF EXISTS prompt_versions_fill_owner();

DROP INDEX IF EXISTS prompt_versions_owner_idx;

ALTER TABLE IF EXISTS prompt_versions DROP CONSTRAINT IF EXISTS prompt_versions_owner_fk;
ALTER TABLE IF EXISTS prompt_versions DROP COLUMN IF EXISTS owner_tenant_id;

DROP TABLE IF EXISTS prompt_owner;

DELETE FROM schema_migration_files WHERE filename = '035_prompt_owner.sql';

COMMIT;
