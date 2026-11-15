-- DOWN for migrations/038_tenant_llm_config_audit_test_refused.sql. Applied BY
-- HAND only (psql -1 -f); the startup runner never descends into this directory.
--
-- Roll the code back to pre-038 FIRST (038-aware code writes ``test_refused``).
-- ⚠️ DESTRUCTIVE for those rows: the narrowed CHECK cannot be added while a
-- ``test_refused`` row exists, so they are deleted — the customer loses the
-- record of refused tests. 038's schema_migration_files row is removed so the
-- runner re-applies it on the next start.

BEGIN;

DELETE FROM tenant_llm_config_audit WHERE action = 'test_refused';

ALTER TABLE tenant_llm_config_audit
    DROP CONSTRAINT IF EXISTS tenant_llm_config_audit_action_chk;
ALTER TABLE tenant_llm_config_audit
    ADD CONSTRAINT tenant_llm_config_audit_action_chk
    CHECK (action IN ('set_key', 'clear_key', 'update', 'remove', 'test'));

DELETE FROM schema_migration_files
 WHERE filename = '038_tenant_llm_config_audit_test_refused.sql';

COMMIT;
