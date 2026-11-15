-- Migration: Add quality_score column to graph_execution_sessions
-- Version: 018
-- Description: Persist the per-session quality_score returned by the validator
--   worker (alongside validation_passed added in migration 009). Sourced from
--   the validator response dict at session-finalize; ranges 0.0000–1.0000.
--   Consumed by the metering_service ETL via the new
--   GET /api/graph/sessions/by-correlation/{cid} endpoint to populate
--   exec_quality.quality_score (currently NULL).
--
-- Re-runnable (idempotent): ADD COLUMN IF NOT EXISTS.

BEGIN;

ALTER TABLE graph_execution_sessions
    ADD COLUMN IF NOT EXISTS quality_score NUMERIC(5, 4);

INSERT INTO schema_migrations (version, description)
VALUES ('0.18.0', 'Add quality_score column to graph_execution_sessions')
ON CONFLICT (version) DO NOTHING;

COMMIT;
