-- Migration: Composite index on (tenant_id, created_at DESC) for sessions/list
-- Version: 015
-- Description: Backs the eval-set sampler's pull pattern on
--   GET /execution/sessions/list. The query is
--     SELECT * FROM graph_execution_sessions
--     WHERE tenant_id = $1 AND ...
--     ORDER BY created_at DESC LIMIT N OFFSET M
--   Existing indexes from migration 010 cover (tenant_id, subject_id) and
--   (tenant_id, visibility) but neither helps the ORDER BY created_at on
--   the bare tenant filter, so the planner falls back to a sort over the
--   full tenant slice. On busy tenants this becomes a Top-N heap sort
--   instead of an index range scan. The composite below makes the
--   refresh-from-wm fan-out cheap.
--
-- Idempotent: CREATE INDEX IF NOT EXISTS, schema_migrations row inserted
--   with ON CONFLICT (version) DO NOTHING.

BEGIN;

CREATE INDEX IF NOT EXISTS idx_graph_exec_tenant_created_at
    ON graph_execution_sessions (tenant_id, created_at DESC);

INSERT INTO schema_migrations (version, description) VALUES
    ('015', 'Composite (tenant_id, created_at DESC) index on graph_execution_sessions')
ON CONFLICT (version) DO NOTHING;

COMMIT;
