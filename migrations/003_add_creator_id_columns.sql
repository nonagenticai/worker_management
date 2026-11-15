-- Migration: Add creator_id columns to remaining tables
-- Version: 003
-- Description: Adds creator_id columns to execution_events, node_execution_logs, skills, and workload tables for multi-tenancy support
--
-- Idempotent + forward-compatible: this whole migration is a no-op once
-- creator_id has been dropped service-wide (migration 019). The guard checks
-- a backfill source (graph_execution_sessions.creator_id); if it is gone we skip
-- entirely so we neither re-add the columns nor fail on the absent reference.
-- (DDL must go through EXECUTE inside PL/pgSQL.)

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'graph_execution_sessions' AND column_name = 'creator_id'
    ) THEN
        RAISE NOTICE 'migration 003 skipped: creator_id already removed (post-019)';
        RETURN;
    END IF;

    -- ---- execution_events (backfill from graph_execution_sessions) ----
    EXECUTE 'ALTER TABLE execution_events ADD COLUMN IF NOT EXISTS creator_id VARCHAR(255)';
    UPDATE execution_events e
    SET creator_id = s.creator_id
    FROM graph_execution_sessions s
    WHERE e.session_id = s.session_id AND e.creator_id IS NULL;
    DELETE FROM execution_events WHERE creator_id IS NULL;
    EXECUTE 'ALTER TABLE execution_events ALTER COLUMN creator_id SET NOT NULL';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_execution_events_creator_id ON execution_events(creator_id)';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_execution_events_creator_session ON execution_events(creator_id, session_id)';

    -- ---- node_execution_logs (backfill from graph_execution_sessions) ----
    EXECUTE 'ALTER TABLE node_execution_logs ADD COLUMN IF NOT EXISTS creator_id VARCHAR(255)';
    UPDATE node_execution_logs n
    SET creator_id = s.creator_id
    FROM graph_execution_sessions s
    WHERE n.session_id = s.session_id AND n.creator_id IS NULL;
    DELETE FROM node_execution_logs WHERE creator_id IS NULL;
    EXECUTE 'ALTER TABLE node_execution_logs ALTER COLUMN creator_id SET NOT NULL';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_node_execution_logs_creator_id ON node_execution_logs(creator_id)';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_node_execution_logs_creator_session ON node_execution_logs(creator_id, session_id)';

    -- ---- skills (backfill from workers) ----
    EXECUTE 'ALTER TABLE skills ADD COLUMN IF NOT EXISTS creator_id VARCHAR(255)';
    UPDATE skills sk
    SET creator_id = w.creator_id
    FROM workers w
    WHERE sk.worker_id = w.worker_id AND sk.creator_id IS NULL;
    DELETE FROM skills WHERE creator_id IS NULL;
    EXECUTE 'ALTER TABLE skills ALTER COLUMN creator_id SET NOT NULL';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_skills_creator_id ON skills(creator_id)';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_skills_creator_worker ON skills(creator_id, worker_id)';
    EXECUTE 'ALTER TABLE skills DROP CONSTRAINT IF EXISTS unique_worker_skill';
    EXECUTE 'ALTER TABLE skills ADD CONSTRAINT unique_worker_skill UNIQUE (worker_id, skill_name)';

    -- ---- workload (backfill from workers) ----
    EXECUTE 'ALTER TABLE workload ADD COLUMN IF NOT EXISTS creator_id VARCHAR(255)';
    UPDATE workload wl
    SET creator_id = w.creator_id
    FROM workers w
    WHERE wl.worker_id = w.worker_id AND wl.creator_id IS NULL;
    DELETE FROM workload WHERE creator_id IS NULL;
    EXECUTE 'ALTER TABLE workload ALTER COLUMN creator_id SET NOT NULL';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_workload_creator_id ON workload(creator_id)';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_workload_creator_worker ON workload(creator_id, worker_id)';

    RAISE NOTICE 'Migration 003 completed: creator_id added to execution_events, node_execution_logs, skills, workload';
END $$;
