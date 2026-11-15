-- Migration 009: Add inference telemetry columns to graph_execution_sessions
-- These nullable columns capture per-session inference telemetry populated at
-- session finalize. First-class columns (rather than JSONB inside variables)
-- so trainer/flywheel queries can use indexed column predicates, e.g.:
--   SELECT ... WHERE graph_id = $1 AND validation_passed = true
--            ORDER BY created_at DESC LIMIT N;

ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS chosen_provider TEXT;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS fallback_count INT;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS prompt_tokens INT;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS completion_tokens INT;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS validation_passed BOOLEAN;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS request_payload JSONB;
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS final_response JSONB;

-- Composite index for trainer pull queries (graph_id + created_at DESC).
CREATE INDEX IF NOT EXISTS graph_execution_sessions_graph_id_created_at_idx
    ON graph_execution_sessions (graph_id, created_at DESC);

INSERT INTO schema_migrations (version, description)
VALUES ('0.9.0', 'Add inference telemetry columns to graph_execution_sessions')
ON CONFLICT (version) DO NOTHING;
