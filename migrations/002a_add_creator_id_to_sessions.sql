-- Migration: Add creator_id to graph_execution_sessions
-- Version: 002a
-- Description: Adds creator_id column before 003 can reference it
-- MUST RUN BEFORE migration 003
--
-- Idempotent + forward-compatible: this whole migration is a no-op once
-- creator_id has been dropped service-wide (migration 019). The guard checks
-- the source column it backfills from (graphs.creator_id); if that is gone we
-- skip entirely so we neither re-add the column nor fail on the absent ref.
-- (DDL must go through EXECUTE inside PL/pgSQL.)

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'graphs' AND column_name = 'creator_id'
    ) THEN
        RAISE NOTICE 'migration 002a skipped: creator_id already removed (post-019)';
        RETURN;
    END IF;

    EXECUTE 'ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS creator_id VARCHAR(255)';

    -- Backfill from graphs table (session's graph has the creator)
    UPDATE graph_execution_sessions s
    SET creator_id = g.creator_id
    FROM graphs g
    WHERE s.graph_id = g.id
      AND s.creator_id IS NULL;

    -- Handle orphan sessions (no matching graph) - use 'system' as fallback
    UPDATE graph_execution_sessions
    SET creator_id = 'system'
    WHERE creator_id IS NULL;

    EXECUTE 'ALTER TABLE graph_execution_sessions ALTER COLUMN creator_id SET NOT NULL';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_graph_execution_sessions_creator_id '
            'ON graph_execution_sessions(creator_id)';

    RAISE NOTICE 'Migration 002a completed: creator_id added to graph_execution_sessions';
END $$;
