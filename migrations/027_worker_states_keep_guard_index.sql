-- 027_worker_states_keep_guard_index.sql
--
-- The index the retention sweep's per-worker keep-guard actually needs.
--
-- WHAT 026 GOT RIGHT AND WHAT IT MISSED
-- -------------------------------------
-- 026 added `idx_worker_states_timestamp` for the sweep's age predicate. That
-- index is not the bottleneck and, measured, is not even usable for it: the
-- `timestamp < now() - 30 days` predicate matches ~84% of the table, so the
-- planner correctly declines an index scan and seq-scans instead.
--
-- The cost is in the keep-guard subquery, which asks "does this row have at
-- least `min_keep` NEWER siblings for the same (tenant_id, worker_id)". The
-- sweep relied on `idx_worker_states_tenant_worker_id` (tenant_id, worker_id),
-- created by migration 019. Two columns are not enough: the subquery filters
-- on `n.id > s.id` and orders by `n.id`, neither of which that index can
-- serve, so the plan on production 2026-08-27 was
--
--     Limit  (cost=119.86..119.89 rows=10 width=8)
--       ->  Sort  (cost=119.86..119.98 rows=45 width=8)
--             Sort Key: n.id
--             ->  Index Scan using idx_worker_states_tenant_worker_id ...
--                   Index Cond: ((tenant_id = ...) AND (worker_id = ...))
--                   Filter: (id > s.id)
--
-- i.e. read EVERY row of the pair, filter, then sort -- it cannot stop after
-- `min_keep`. That is fine for a small pair and ruinous for a large one:
-- measured, the busiest (tenant_id, worker_id) pair on production holds
-- **131,507** rows, and the top seven each hold >38,000.
--
-- With `id` as the third column the same subquery becomes an index-only scan
-- in `id` order that stops after `min_keep` entries, with no sort at all.
--
-- WHY NOT JUST EXTEND 019's INDEX
-- -------------------------------
-- `idx_worker_states_tenant_worker_id` still serves equality lookups on
-- (tenant_id, worker_id) elsewhere in the service, and dropping/recreating it
-- would leave a window with no index for those. A three-column index is a
-- superset for those lookups, so 019's can be retired later as a separate,
-- reversible step -- not inside the migration that the sweep depends on.
--
-- IDEMPOTENCY
-- -----------
-- `run_migrations()` (src/db/db_config.py) re-runs EVERY file on EVERY pod
-- boot and does not consult `schema_migrations`. `IF NOT EXISTS` makes the
-- second and every subsequent run a no-op.
--
-- LOCKING
-- -------
-- Same treatment as 026: a plain `CREATE INDEX` takes SHARE and blocks writers
-- for the build. `CREATE INDEX CONCURRENTLY` cannot be used -- the runner wraps
-- each file in a transaction and CONCURRENTLY is not allowed inside one. The
-- runner's session `lock_timeout` of 10s is too tight for the busiest table in
-- the service, so `SET LOCAL` raises it for this transaction only; if the lock
-- cannot be had in 90s the file fails, is logged, and is retried on the next
-- boot with the rest of the schema intact.
--
-- Build cost is a three-column btree over ~990k rows on small fixed-width
-- columns. It touches no TOASTed column.
--
-- ⚠️ THIS MIGRATION MAY NOT APPLY ON PRODUCTION, AND THAT IS TOLERATED
-- ---------------------------------------------------------------------
-- Do NOT read a green deploy as proof this index exists. Measured on the pod
-- that booted 2026-08-27 12:07 UTC, migration 026 -- which uses this exact
-- pattern on this exact table -- did not apply:
--
--     Applying migration: 026_worker_states_retention_index.sql
--     Migration 026_worker_states_retention_index.sql failed (continuing):
--       canceling statement due to lock timeout
--     Migrations: applied=18 failed=8
--
-- `idx_worker_states_timestamp` is genuinely ABSENT from production, verified
-- against `pg_indexes`. It burned the full 90s and gave up. **8 of the 26
-- migrations fail this way on EVERY pod boot** (004, 005, 010, 011, 016, 017,
-- 019, 026): lock contention on this database is chronic, not incidental.
-- `run_migrations()` is non-strict, so each failure is a logged WARNING and
-- startup continues.
--
-- Why: a plain `CREATE INDEX` takes SHARE, which conflicts with the
-- ROW EXCLUSIVE held by the constant INSERTs into `worker_states` (a row is
-- written on every agent message). An uncontended 90s window essentially
-- never arrives.
--
-- The sweep does NOT depend on this index for correctness or for the bulk of
-- its speedup. Measured with EXPLAIN against production, with prod's CURRENT
-- indexes and WITHOUT this migration, the new batch statement costs
-- **242,488** against the old form's **119,253,518** -- 492x cheaper purely
-- from bounding the candidate set on the primary key first. This index
-- removes the remaining per-loop subplan cost (~120,000 of that 242,488). So
-- if this file times out it is a logged no-op that costs an optimisation, not
-- a broken sweep.
--
-- The durable fix is `CREATE INDEX CONCURRENTLY` (SHARE UPDATE EXCLUSIVE --
-- does not block writers), which cannot be used until `run_migrations()` can
-- execute a file OUTSIDE its per-file transaction. That is a change to the
-- shared migration runner with its own blast radius and its own tests, and it
-- is deliberately NOT bundled here. Until then, an operator can create this
-- index by hand, once, and this file becomes a permanent no-op:
--
--     CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_worker_states_tenant_worker_id_id
--         ON worker_states (tenant_id, worker_id, id);
--
-- (A failed CONCURRENTLY leaves an INVALID index that `IF NOT EXISTS` will
-- then skip forever -- check `pg_index.indisvalid` and drop it before retrying.)

BEGIN;

SET LOCAL lock_timeout = '90s';

CREATE INDEX IF NOT EXISTS idx_worker_states_tenant_worker_id_id
    ON worker_states (tenant_id, worker_id, id);

INSERT INTO schema_migrations (version, description)
VALUES ('027', 'Add (tenant_id, worker_id, id) index so the retention keep-guard is bounded')
ON CONFLICT (version) DO NOTHING;

COMMIT;
