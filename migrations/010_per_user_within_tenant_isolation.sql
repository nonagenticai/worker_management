-- Migration: Per-user-within-tenant isolation
-- Version: 010
-- Description: Adds (tenant_id, subject_id, visibility) to all owned tables.
-- Backfills tenant_id := subject_id := creator_id, visibility := 'tenant'
-- so existing rows preserve the legacy single-axis behavior.
-- Mirrors procedure_management commit 7f372e0 (Phase 1 pilot).
--
-- Tables (FK topo-order):
--   roots:       workers, graphs, tools
--   workers->:   skills, workload, worker_states
--   graphs->:    graph_states, graph_execution_sessions
--   sessions->:  node_execution_logs, execution_events
--
-- Re-runnable: every change is guarded with IF NOT EXISTS / DO blocks.

-- ============================================================================
-- Helper macros via DO blocks for idempotent ALTER TABLE
-- ============================================================================

-- ============================================================================
-- 1. workers (root)
-- ============================================================================

ALTER TABLE workers ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE workers ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE workers ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'workers' AND column_name = 'creator_id') THEN
        UPDATE workers SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE workers SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE workers ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE workers ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'workers_visibility_check'
    ) THEN
        ALTER TABLE workers
            ADD CONSTRAINT workers_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_workers_tenant_subject ON workers(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_workers_tenant_visibility ON workers(tenant_id, visibility);

-- ============================================================================
-- 2. graphs (root)
-- ============================================================================

ALTER TABLE graphs ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE graphs ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE graphs ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'graphs' AND column_name = 'creator_id') THEN
        UPDATE graphs SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE graphs SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE graphs ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE graphs ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'graphs_visibility_check'
    ) THEN
        ALTER TABLE graphs
            ADD CONSTRAINT graphs_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_graphs_tenant_subject ON graphs(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_graphs_tenant_visibility ON graphs(tenant_id, visibility);

-- ============================================================================
-- 3. tools (root)
-- ============================================================================

ALTER TABLE tools ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE tools ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE tools ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'tools' AND column_name = 'creator_id') THEN
        UPDATE tools SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE tools SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE tools ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE tools ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'tools_visibility_check'
    ) THEN
        ALTER TABLE tools
            ADD CONSTRAINT tools_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_tools_tenant_subject ON tools(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_tools_tenant_visibility ON tools(tenant_id, visibility);

-- ============================================================================
-- 4. skills (workers child)
-- ============================================================================

ALTER TABLE skills ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE skills ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE skills ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'skills' AND column_name = 'creator_id') THEN
        UPDATE skills SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE skills SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE skills ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE skills ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'skills_visibility_check'
    ) THEN
        ALTER TABLE skills
            ADD CONSTRAINT skills_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_skills_tenant_subject ON skills(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_skills_tenant_visibility ON skills(tenant_id, visibility);

-- ============================================================================
-- 5. workload (workers child)
-- ============================================================================

ALTER TABLE workload ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE workload ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE workload ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'workload' AND column_name = 'creator_id') THEN
        UPDATE workload SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE workload SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE workload ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE workload ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'workload_visibility_check'
    ) THEN
        ALTER TABLE workload
            ADD CONSTRAINT workload_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_workload_tenant_subject ON workload(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_workload_tenant_visibility ON workload(tenant_id, visibility);

-- ============================================================================
-- 6. worker_states (workers child)
-- ============================================================================

ALTER TABLE worker_states ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE worker_states ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE worker_states ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'worker_states' AND column_name = 'creator_id') THEN
        UPDATE worker_states SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE worker_states SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE worker_states ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE worker_states ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'worker_states_visibility_check'
    ) THEN
        ALTER TABLE worker_states
            ADD CONSTRAINT worker_states_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_worker_states_tenant_subject ON worker_states(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_worker_states_tenant_visibility ON worker_states(tenant_id, visibility);

-- ============================================================================
-- 7. graph_states (graphs child)
-- ============================================================================

ALTER TABLE graph_states ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE graph_states ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE graph_states ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'graph_states' AND column_name = 'creator_id') THEN
        UPDATE graph_states SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE graph_states SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE graph_states ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE graph_states ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'graph_states_visibility_check'
    ) THEN
        ALTER TABLE graph_states
            ADD CONSTRAINT graph_states_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_graph_states_tenant_subject ON graph_states(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_graph_states_tenant_visibility ON graph_states(tenant_id, visibility);

-- ============================================================================
-- 8. graph_execution_sessions (graphs child)
-- ============================================================================

ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE graph_execution_sessions ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'graph_execution_sessions' AND column_name = 'creator_id') THEN
        UPDATE graph_execution_sessions SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE graph_execution_sessions SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE graph_execution_sessions ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE graph_execution_sessions ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'graph_execution_sessions_visibility_check'
    ) THEN
        ALTER TABLE graph_execution_sessions
            ADD CONSTRAINT graph_execution_sessions_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_graph_exec_tenant_subject ON graph_execution_sessions(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_graph_exec_tenant_visibility ON graph_execution_sessions(tenant_id, visibility);

-- ============================================================================
-- 9. node_execution_logs (sessions child)
-- ============================================================================

ALTER TABLE node_execution_logs ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE node_execution_logs ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE node_execution_logs ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'node_execution_logs' AND column_name = 'creator_id') THEN
        UPDATE node_execution_logs SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE node_execution_logs SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE node_execution_logs ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE node_execution_logs ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'node_execution_logs_visibility_check'
    ) THEN
        ALTER TABLE node_execution_logs
            ADD CONSTRAINT node_execution_logs_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_node_exec_tenant_subject ON node_execution_logs(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_node_exec_tenant_visibility ON node_execution_logs(tenant_id, visibility);

-- ============================================================================
-- 10. execution_events (sessions child)
-- ============================================================================

ALTER TABLE execution_events ADD COLUMN IF NOT EXISTS tenant_id   VARCHAR(255);
ALTER TABLE execution_events ADD COLUMN IF NOT EXISTS subject_id  VARCHAR(255);
ALTER TABLE execution_events ADD COLUMN IF NOT EXISTS visibility  VARCHAR(16) NOT NULL DEFAULT 'tenant';

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'execution_events' AND column_name = 'creator_id') THEN
        UPDATE execution_events SET tenant_id  = creator_id WHERE tenant_id  IS NULL;
        UPDATE execution_events SET subject_id = creator_id WHERE subject_id IS NULL;
    END IF;
END $$;

ALTER TABLE execution_events ALTER COLUMN tenant_id  SET NOT NULL;
ALTER TABLE execution_events ALTER COLUMN subject_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'execution_events_visibility_check'
    ) THEN
        ALTER TABLE execution_events
            ADD CONSTRAINT execution_events_visibility_check
            CHECK (visibility IN ('tenant', 'user'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_execution_events_tenant_subject ON execution_events(tenant_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_execution_events_tenant_visibility ON execution_events(tenant_id, visibility);

-- ============================================================================
-- creator_id stays as a redundant backfilled column for rollback safety.
-- A follow-up migration may DROP it once Phase 3 has been live in prod and no
-- consumers still reference it. For now keep both axes populated during writes.
-- ============================================================================

INSERT INTO schema_migrations (version, description)
VALUES ('010', 'Per-user-within-tenant isolation (tenant_id, subject_id, visibility)')
ON CONFLICT (version) DO NOTHING;
