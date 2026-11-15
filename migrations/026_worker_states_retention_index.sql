-- 026_worker_states_retention_index.sql
--
-- The index the retention sweep needs, which production does NOT have.
--
-- WHY THIS IS NOT ALREADY THERE
-- -----------------------------
-- `postgres_schema.sql` declares eight indexes on `worker_states`, including
-- `idx_worker_states_timestamp ON worker_states(timestamp DESC)`. That file is
-- the BOOTSTRAP baseline -- it is what `scripts/init_db.py` applies to a fresh
-- (CI) database. Production predates it and was built by the numbered
-- migrations only, so it has just the four indexes those migrations created.
-- Verified live on the read-only standby 2026-08-27:
--
--     select indexname from pg_indexes where tablename='worker_states';
--       idx_worker_states_tenant_subject
--       idx_worker_states_tenant_visibility
--       idx_worker_states_tenant_worker_id
--       ix_worker_states_tenant_name
--
-- No timestamp index. So the schema file and the live database disagree, and
-- reading the schema file would have produced a retention sweep that
-- sequential-scans a 3 GB table on every batch. This migration closes that
-- drift for the one index the sweep depends on.
--
-- (`idx_worker_states_tenant_worker_id`, which the sweep's per-worker
-- keep-guard scans, DOES exist live -- it came from migration 019.)
--
-- IDEMPOTENCY
-- -----------
-- `run_migrations()` (src/db/db_config.py) re-runs EVERY file on EVERY pod
-- boot and does not consult `schema_migrations`. `IF NOT EXISTS` makes the
-- second and every subsequent run a no-op.
--
-- LOCKING
-- -------
-- A plain `CREATE INDEX` takes SHARE, which blocks writers for the duration of
-- the build. `CREATE INDEX CONCURRENTLY` cannot be used here: the runner wraps
-- each file in a transaction and CONCURRENTLY is not allowed inside one.
--
-- The runner sets a session `lock_timeout` of 10s, which is a fine budget for
-- eleven small tables and a poor one for the busiest table in the service.
-- `SET LOCAL` raises it for this transaction only -- deliberately LOCAL, and
-- deliberately bounded: if the lock cannot be had in 90s the file fails, is
-- logged, and is retried on the next boot with the rest of the schema intact.
-- This is the same treatment migration 024 gives its `worker_states` step, and
-- 024's operation was a full table REWRITE under ACCESS EXCLUSIVE, so the
-- precedent is strictly heavier than this one.
--
-- Build cost is a single-column btree over 988,948 rows -- seconds, not
-- minutes, and it touches no TOASTed column.

BEGIN;

SET LOCAL lock_timeout = '90s';

CREATE INDEX IF NOT EXISTS idx_worker_states_timestamp
    ON worker_states (timestamp DESC);

INSERT INTO schema_migrations (version, description)
VALUES ('026', 'Add idx_worker_states_timestamp for the retention sweep')
ON CONFLICT (version) DO NOTHING;

COMMIT;
