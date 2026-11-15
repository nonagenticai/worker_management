-- Migration: Add tenant_name column to all owned tables
-- Version: 011
-- Description: Adds tenant_name VARCHAR(255) to every table that carries
--   tenant_id, backfills from tenant_id, then makes it NOT NULL.
--   A covering index supports lookup-by-name queries.
--
-- Tables: workers, skills, workload, worker_states, graphs, graph_states,
--         graph_execution_sessions, node_execution_logs, execution_events, tools
--
-- Re-runnable (idempotent): ADD COLUMN IF NOT EXISTS + CREATE INDEX IF NOT EXISTS.

BEGIN;
DO $$
DECLARE
  t TEXT;
BEGIN
  FOR t IN SELECT unnest(ARRAY[
    'workers',
    'skills',
    'workload',
    'worker_states',
    'graphs',
    'graph_states',
    'graph_execution_sessions',
    'node_execution_logs',
    'execution_events',
    'tools'
  ])
  LOOP
    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS tenant_name VARCHAR(255)', t);
    EXECUTE format('UPDATE %I SET tenant_name = tenant_id WHERE tenant_name IS NULL', t);
    EXECUTE format('ALTER TABLE %I ALTER COLUMN tenant_name SET NOT NULL', t);
    EXECUTE format(
      'CREATE INDEX IF NOT EXISTS %I ON %I (tenant_name)',
      'ix_' || t || '_tenant_name',
      t
    );
  END LOOP;
END $$;
COMMIT;
