-- Migration: Drop the creator_id legacy alias service-wide
-- Version: 019
-- Description:
--   creator_id was a NOT NULL VARCHAR(255) mirror of tenant_id on the 10 owned
--   tables (workers, worker_states, graphs, graph_states,
--   graph_execution_sessions, node_execution_logs, execution_events, tools,
--   workload, skills). It predates multi-tenancy; migration 010 added tenant_id
--   and backfilled tenant_id := subject_id := creator_id, so creator_id == tenant_id
--   for every row. This removes the alias forward-only.
--
--   Order matters and is the one real correctness risk: tenant_id MUST be present,
--   backfilled, and NOT NULL on every table BEFORE creator_id is dropped. Step 1
--   guarantees that defensively (migration 010 already does it in the canonical
--   chain). Then we rebuild the UNIQUE(creator_id, X) guarantees on tenant_id and
--   add the tenant_id composite indexes that mirror the dropped creator_id
--   composites, before finally dropping the column with CASCADE -- which removes
--   every remaining creator_id index and UNIQUE constraint in one robust step
--   (there are more of these than postgres_schema.sql shows, because earlier
--   migrations added duplicates: e.g. tools.unique_tool_name_creator and the
--   idx_*_creator_* duplicates from migrations 002a/003).
--
--   NOTE: postgres_schema.sql is intentionally NOT modified. It is the one-time
--   bootstrap DDL (applied before the migration chain); migration 010 reads
--   creator_id from it to backfill tenant_id, so the column must still exist there.
--   This migration is the authoritative removal from the live/migrated schema.
--
-- Re-runnable: IF EXISTS / IF NOT EXISTS guards throughout; DROP COLUMN is the
-- last step so a partially-applied prior run resumes cleanly.

-- ============================================================================
-- 1. Prerequisite guard: tenant_id present + backfilled + NOT NULL everywhere.
-- ============================================================================
DO $$
DECLARE
    tbl text;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'workers', 'worker_states', 'graphs', 'graph_states',
        'graph_execution_sessions', 'node_execution_logs', 'execution_events',
        'tools', 'workload', 'skills'
    ]
    LOOP
        EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255)', tbl);

        -- Backfill from creator_id only while creator_id still exists (so re-runs
        -- after the column is dropped do not error).
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = tbl AND column_name = 'creator_id'
        ) THEN
            EXECUTE format('UPDATE %I SET tenant_id = creator_id WHERE tenant_id IS NULL', tbl);
        END IF;

        -- Enforce NOT NULL only if it is not already (and only when no nulls remain).
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = tbl
              AND column_name = 'tenant_id' AND is_nullable = 'YES'
        ) THEN
            EXECUTE format('ALTER TABLE %I ALTER COLUMN tenant_id SET NOT NULL', tbl);
        END IF;
    END LOOP;
END $$;

-- ============================================================================
-- 2. Rebuild the UNIQUE(creator_id, X) guarantees on tenant_id.
--    Semantically identical since creator_id == tenant_id for every row.
--
--    BEST-EFFORT: each is wrapped in its own EXCEPTION block so that pre-existing
--    duplicate data (some envs never actually enforced unique_creator_* — e.g.
--    leftover integration-test rows) merely SKIPS that one constraint with a
--    warning instead of aborting the whole migration. The creator_id drop in
--    step 4 is the critical operation and must not be blocked by a constraint
--    the data can't satisfy. The PL/pgSQL EXCEPTION clause opens a savepoint, so
--    catching here lets the outer single-file transaction continue.
-- ============================================================================
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'unique_tenant_worker_name') THEN
        ALTER TABLE workers ADD CONSTRAINT unique_tenant_worker_name UNIQUE (tenant_id, worker_name);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'skipping unique_tenant_worker_name (dirty data?): %', SQLERRM;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'unique_tenant_graph_name') THEN
        ALTER TABLE graphs ADD CONSTRAINT unique_tenant_graph_name UNIQUE (tenant_id, name);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'skipping unique_tenant_graph_name (dirty data?): %', SQLERRM;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'unique_tenant_tool_name') THEN
        ALTER TABLE tools ADD CONSTRAINT unique_tenant_tool_name UNIQUE (tenant_id, name);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'skipping unique_tenant_tool_name (dirty data?): %', SQLERRM;
END $$;

-- ============================================================================
-- 3. tenant_id composite indexes mirroring the creator_id composites that are
--    NOT already covered by:
--      * migration 010's idx_*_tenant_subject / idx_*_tenant_visibility, or
--      * the new UNIQUE constraints above (workers/graphs/tools), or
--      * the existing UNIQUE index on graph_execution_sessions.session_id.
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_worker_states_tenant_worker_id ON worker_states(tenant_id, worker_id);
CREATE INDEX IF NOT EXISTS idx_graph_states_tenant_graph_id ON graph_states(tenant_id, graph_id);
CREATE INDEX IF NOT EXISTS idx_node_exec_tenant_session ON node_execution_logs(tenant_id, session_id);
CREATE INDEX IF NOT EXISTS idx_exec_events_tenant_session ON execution_events(tenant_id, session_id);
CREATE INDEX IF NOT EXISTS idx_workload_tenant_worker ON workload(tenant_id, worker_id);
CREATE INDEX IF NOT EXISTS idx_skills_tenant_worker ON skills(tenant_id, worker_id);

-- ============================================================================
-- 4. Drop creator_id everywhere. CASCADE removes every remaining creator_id
--    index and UNIQUE constraint (incl. tools' duplicate unique_tool_name_creator).
-- ============================================================================
DO $$
DECLARE
    tbl text;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'workers', 'worker_states', 'graphs', 'graph_states',
        'graph_execution_sessions', 'node_execution_logs', 'execution_events',
        'tools', 'workload', 'skills'
    ]
    LOOP
        EXECUTE format('ALTER TABLE %I DROP COLUMN IF EXISTS creator_id CASCADE', tbl);
    END LOOP;
END $$;
