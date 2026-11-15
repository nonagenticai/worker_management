-- Migration: 017_backfill_reserved_worker_names — purge orphan rows whose
-- worker_name is a reserved JS/SQL literal that should never have been
-- accepted by validate_identifier() but slipped through a URL path-param
-- auto-create path (send_message / update_worker).
-- Date: 2026-05-13
--
-- Live evidence (master:5432 worker_management.workers, queried 2026-05-13):
--   tenant_id=00000000-0000-0000-0000-000000000001
--   worker_name='undefined'
--   created_at=2026-05-12 20:38:55
-- This row produced the /workers/undefined entry in fr's workers list that
-- PR #250 added a defensive frontend filter for (frontend/src/routes/workers/
-- +page.svelte:38-43). With this migration plus the path-param validator
-- tightening in src/api/workers/{message_endpoints,crud_endpoints}.py, the
-- write-path is closed and the existing orphan is removed.
--
-- The reserved set mirrors src/api/security_validation.py:560-589 verbatim so
-- the migration cannot drift from the validator. Forward-only: idempotent
-- (DELETE … WHERE lower(worker_name) IN …) and safe to re-run.
--
-- A live audit on 2026-05-13 confirmed only 'undefined' (1 row) matches the
-- reserved set; none of admin/root/system/etc are in the table.

BEGIN;

DELETE FROM workers
 WHERE lower(worker_name) IN (
    'null', 'undefined', 'none', 'nil', 'true', 'false',
    'admin', 'root', 'system', 'guest', 'anonymous',
    'select', 'insert', 'update', 'delete', 'drop', 'create',
    'exec', 'execute', 'eval', 'function', 'var', 'let', 'const'
 );

INSERT INTO schema_migrations (version, description)
VALUES ('017', 'Backfill: delete worker rows with reserved-literal worker_name')
ON CONFLICT (version) DO NOTHING;

COMMIT;
