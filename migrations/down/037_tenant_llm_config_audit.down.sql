-- DOWN for migrations/037_tenant_llm_config_audit.sql. Applied BY HAND only
-- (psql -1 -f); the startup runner lists top-level migrations/*.sql and never
-- descends into this directory.
--
-- ⚠️ DESTRUCTIVE: drops every tenant's BYO-config audit history — the record
-- the customer is entitled to see. Roll the code back to pre-037 FIRST: 037-aware
-- code writes this table in the same transaction as every PUT/DELETE
-- /tenant-config/llm, so those requests FAIL while the table is absent. 037's
-- schema_migration_files row is removed so the runner re-applies it (empty) on
-- the next start.

BEGIN;

DROP TABLE IF EXISTS tenant_llm_config_audit;

DELETE FROM schema_migration_files WHERE filename = '037_tenant_llm_config_audit.sql';

COMMIT;
