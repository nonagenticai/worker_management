-- Migration: Add correlation_id column to graph_execution_sessions
-- Version: 012
-- Description: Adds a UUID correlation_id propagated end-to-end from n8n /
--   fr-backend (X-Correlation-Id header) through worker_management and into
--   metering_service.cost_events.correlation_id. Used as the join key for
--   the metering_service.exec_quality materialized table populated by the
--   Loki → Postgres ETL (see metering_service alembic 0004).
--
-- Re-runnable (idempotent): ADD COLUMN IF NOT EXISTS + CREATE INDEX IF NOT EXISTS.

BEGIN;

ALTER TABLE graph_execution_sessions
    ADD COLUMN IF NOT EXISTS correlation_id UUID;

CREATE INDEX IF NOT EXISTS idx_ges_correlation
    ON graph_execution_sessions (correlation_id)
    WHERE correlation_id IS NOT NULL;

COMMIT;
