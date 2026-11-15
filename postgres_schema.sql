-- ============================================================================
-- PostgreSQL Schema for Worker Management System
-- Database: worker_management
-- Version: 1.0.0
-- ============================================================================
--
-- ############################################################################
-- ##  WARNING: THIS FILE IS *NOT* A FAITHFUL REPRESENTATION OF PRODUCTION.  ##
-- ############################################################################
--
-- Measured against live prod (your PostgreSQL host) 2026-08-07:
--
--   * Prod has 15 base tables; this file declares 12. Undeclared but live:
--     `tenant_llm_config` (migration 021), `tenant_modality_config` (022),
--     and `agents` -- an orphan created by nothing in this repo, 0 rows.
--   * 12 of 15 live tables have NO PRIMARY KEY, and the live schema has ZERO
--     foreign keys -- including `fk_graph_exec` declared below, which does not
--     exist in prod. Only the three migration-created tables have PKs.
--   * Every bootstrap-era table uses naive `timestamp without time zone` where
--     this file declares TIMESTAMPTZ; `graph_execution_sessions.session_id`
--     and `.graph_id` are `text` in prod where this file declares UUID.
--   * `schema_migrations` is not a reliable "applied" ledger: 4 files record a
--     semver-shaped key instead of their ordinal (004->'1.0.4', 008->'0.8.0',
--     009->'0.9.0', 018->'0.18.0') and 8 of 23 files record nothing at all.
--
-- Full measurements, evidence and the recommended remediation sequence:
--     docs/SCHEMA_DRIFT_2026-08-07.md
--
-- What this file IS: the fresh-install DDL, and what CI applies to its
-- ephemeral postgres:18 sidecar via `scripts/init_db.py`. The `pg`-marked test
-- suite therefore validates against the schema DECLARED HERE, not against a
-- replica of production. Do not read a green CI run as proof that prod's
-- schema is correct.
--
-- Do NOT "fix" the drift by editing this file by hand. The first step is a
-- `pg_dump -s` of prod and a diff -- see the doc above.
--
-- ---------------------------------------------------------------------------
-- 2026-08-10: that diff has now been DONE. Measured prod (15 base tables, 236
-- columns) against this file applied to an ephemeral postgres:18 (12 tables,
-- 149 columns); row counts cross-checked against independent count(*) queries
-- so nothing was silently truncated. Full result:
--
--   56 column-TYPE divergences, in 7 classes (file -> prod):
--        19 timestamptz -> timestamp     14 varchar -> text
--        10 uuid        -> text           9 int8    -> int4
--         2 jsonb       -> text[]         1 skill_level enum -> text
--         1 float4      -> numeric
--    1 nullability divergence  (workers.db_path: file NULL-able, prod NOT NULL)
--   20 default divergences     (18 are fallout of the type classes above)
--   46 file-declared INDEXES absent from prod (incl. every *_pkey)
--   24 file-declared CONSTRAINTS absent from prod: 11 PK, 7 FK, 4 UNIQUE, 2 CHECK
--      -- prod has ZERO foreign keys, on any table
--    1 prod column undeclared here (graphs.execution_config)
--
-- ONLY the 2 jsonb->text[] rows are reconciled in this file (tools.tags,
-- skills.domains). They are the sole class where prod is provably RIGHT and
-- this file is the stale artifact -- proven by migrations/002_fix_tool_tags.sql,
-- by the tool_registry repo binding the same column as an array, and by a live
-- reproduction of the write failure. Everything else is prod having drifted
-- FROM this file (it was bootstrapped by the old auto-CREATE-TABLE code, never
-- from this DDL), so rewriting those lines to match prod would CODIFY THE
-- DAMAGE. Reconciling them is a migration project with an outage window and an
-- owner decision, not a drive-by edit -- it is deliberately NOT done here.
-- ---------------------------------------------------------------------------
-- 2026-08-13 AMENDMENT -- one narrow exception, on evidence that did not exist
-- when the paragraph above was written. It is amended here rather than
-- contradicted silently.
--
-- `worker_id` JOINS the "prod is provably RIGHT" class alongside tools.tags and
-- skills.domains. It is NOT decay. The #55 investigation (merged #171,
-- 2026-08-12) established that the registry fallback deliberately keys worker
-- state by worker NAME when it has no id -- that is the mechanism behind the
-- split history #171 fixed, and it is current, intended behaviour.
--
-- Measured read-only against prod 2026-08-13:
--     worker_states rows                                      857,075
--       ...whose worker_id is NOT UUID-shaped                  16,224
--       ...with no matching workers row                        16,505
--     foreign keys anywhere in the database                          0
--
-- So `UUID` is not an aspiration this column has drifted from -- it is a type
-- that could never have held 16,224 rows the normal code path wrote. And
-- `fk_worker` could not be added even in principle: ADD CONSTRAINT aborts on
-- the 16,505 unparented rows, and once added it would reject writes from live
-- workers using the name-keyed fallback.
--
-- Ratified, therefore, and ONLY for this class:
--   * workers / worker_states / workload / skills `.worker_id`: UUID -> TEXT
--   * `fk_worker` on worker_states: REMOVED (see the note at that table)
--
-- The alternative end-state -- stop the application writing names into
-- worker_id, repair the 16,505 rows, then constrain -- remains open and is a
-- real project (measure -> repair -> constrain; see planka-private #39 for the
-- shape). Choosing it later means re-typing these columns, which is why the
-- evidence is recorded rather than just the edit.
--
-- Everything else in the diff above is UNTOUCHED and the paragraph above still
-- governs it: the graph-side UUID columns, the other 6 declared FKs, the
-- missing PKs and the timestamptz class are all still "do not rewrite to match
-- prod". Full evidence: docs/SCHEMA_DRIFT_2026-08-07.md (addendum 2026-08-13).
-- ---------------------------------------------------------------------------
-- ############################################################################
--
-- This is the canonical fresh-install DDL and reflects the *current* schema
-- end-state: per-tenant ownership via (tenant_id, subject_id, visibility).
-- The legacy single-axis ownership alias was retired service-wide by migration
-- 019; this base schema is tenant-native, so migrations 010 (adds the tenant
-- axes) and 019 (drops the legacy alias) run as idempotent no-ops on a fresh
-- install. The denormalized ``tenant_name`` column added by migration 011 is
-- likewise baked into every owned table here (NOT NULL, with an ``ix_*_tenant_name``
-- index), so migration 011 is also a no-op on a fresh install.
-- ============================================================================

-- Drop existing objects if re-running (use with caution in production)
-- DROP TABLE IF EXISTS schema_migrations CASCADE;
-- DROP TABLE IF EXISTS skills CASCADE;
-- DROP TABLE IF EXISTS workload CASCADE;
-- DROP TABLE IF EXISTS tools CASCADE;
-- DROP TABLE IF EXISTS execution_events CASCADE;
-- DROP TABLE IF EXISTS node_execution_logs CASCADE;
-- DROP TABLE IF EXISTS graph_execution_sessions CASCADE;
-- DROP TABLE IF EXISTS graph_states CASCADE;
-- DROP TABLE IF EXISTS graphs CASCADE;
-- DROP TABLE IF EXISTS worker_states CASCADE;
-- DROP TABLE IF EXISTS workers CASCADE;
-- DROP TYPE IF EXISTS skill_level;

-- ============================================================================
-- TYPES
-- ============================================================================

CREATE TYPE skill_level AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');

-- ============================================================================
-- TABLES
-- ============================================================================

-- Workers Registry Table
CREATE TABLE workers (
    id BIGSERIAL PRIMARY KEY,
    worker_id TEXT NOT NULL UNIQUE,
    worker_name VARCHAR(255) NOT NULL,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    db_path TEXT,  -- Deprecated for PostgreSQL workers, kept for compatibility
    is_agent BOOLEAN NOT NULL DEFAULT false,
    status VARCHAR(50) NOT NULL DEFAULT 'created',
    operational_status VARCHAR(50) DEFAULT 'offline',
    config JSONB,
    output_schema JSONB,
    output_validator TEXT,
    -- config_revision (migration 016): client-stamped config revision (max 64
    -- chars enforced at the API layer). NULL = unstamped / predates the column.
    config_revision TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_tenant_worker_name UNIQUE (tenant_id, worker_name),
    CONSTRAINT workers_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Worker States History Table
CREATE TABLE worker_states (
    id BIGSERIAL PRIMARY KEY,
    worker_id TEXT NOT NULL,
    worker_name VARCHAR(255) NOT NULL,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    is_agent BOOLEAN NOT NULL DEFAULT false,
    state_type VARCHAR(50) NOT NULL,
    worker_state JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    CONSTRAINT unique_worker_state UNIQUE (worker_id, timestamp),
    -- fk_worker REMOVED 2026-08-13: worker_states is an opaque-keyed history
    -- stream, not a child of workers. 16,505 live rows have no matching
    -- workers row and the name-keyed registry fallback keeps creating more.
    -- See the 2026-08-13 amendment in this file's header before restoring it.
    CONSTRAINT worker_states_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Graphs Definition Table
CREATE TABLE graphs (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    description TEXT NOT NULL DEFAULT '',
    nodes JSONB NOT NULL,
    edges JSONB NOT NULL,
    variables JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_tenant_graph_name UNIQUE (tenant_id, name),
    CONSTRAINT graphs_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Graph States History Table
CREATE TABLE graph_states (
    id BIGSERIAL PRIMARY KEY,
    graph_id UUID NOT NULL,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    state_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_graph FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE,
    CONSTRAINT graph_states_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Graph Execution Sessions Table
CREATE TABLE graph_execution_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL,
    graph_id UUID NOT NULL,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    execution_mode VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'initialized',
    variables JSONB,
    node_outputs JSONB,
    worker_assignments JSONB,
    node_errors JSONB,
    completed_nodes JSONB,
    failed_nodes JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    -- Inference telemetry (Phase 1 inference-graph, migration 009).
    -- All nullable; populated at session finalize, read by trainer/flywheel.
    chosen_provider TEXT,
    fallback_count INT,
    prompt_tokens INT,
    completion_tokens INT,
    validation_passed BOOLEAN,
    request_payload JSONB,
    final_response JSONB,
    -- correlation_id (migration 012): UUID propagated end-to-end (X-Correlation-Id)
    -- and used as the join key into metering_service.cost_events.
    correlation_id UUID,
    -- quality_score (migration 018): per-session validator score, 0.0000–1.0000.
    quality_score NUMERIC(5, 4),
    CONSTRAINT fk_graph_exec FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE,
    CONSTRAINT graph_execution_sessions_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Node Execution Logs Table
CREATE TABLE node_execution_logs (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    node_id VARCHAR(255) NOT NULL,
    worker_name VARCHAR(255) NOT NULL,
    task_template TEXT NOT NULL,
    resolved_task TEXT NOT NULL,
    output TEXT,
    error_message TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    retry_count INTEGER DEFAULT 0,
    timeout_seconds INTEGER DEFAULT 300,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_session FOREIGN KEY (session_id) REFERENCES graph_execution_sessions(session_id) ON DELETE CASCADE,
    CONSTRAINT node_execution_logs_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Execution Events Table
CREATE TABLE execution_events (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    event_type VARCHAR(50) NOT NULL,
    node_id VARCHAR(255),
    event_data JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_event_session FOREIGN KEY (session_id) REFERENCES graph_execution_sessions(session_id) ON DELETE CASCADE,
    CONSTRAINT execution_events_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Tools Registry Table
CREATE TABLE tools (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    description TEXT NOT NULL,
    python_code TEXT NOT NULL,
    input_schema JSONB,
    output_schema JSONB,
    -- text[], NOT jsonb. Production has been `text[] NOT NULL DEFAULT
    -- '{}'::text[]` since the column was migrated off JSON; this line was never
    -- updated, so `migrations/002_fix_tool_tags.sql` raised `malformed array
    -- literal: "[]"` against the real DB on every pod boot. The other prod DB
    -- that owns this table (`tool_registry`, repo nonagenticai/tool_registry,
    -- `src/db/postgres_db.py:105-117`) binds it as an array too.
    tags TEXT[] NOT NULL DEFAULT '{}'::TEXT[],
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_valid BOOLEAN NOT NULL DEFAULT true,
    validation_error TEXT,
    CONSTRAINT unique_tenant_tool_name UNIQUE (tenant_id, name),
    CONSTRAINT tools_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Workload Management Table
CREATE TABLE workload (
    id BIGSERIAL PRIMARY KEY,
    worker_id TEXT NOT NULL UNIQUE,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    current_tasks INTEGER DEFAULT 0 CHECK (current_tasks >= 0),
    capacity INTEGER DEFAULT 5 CHECK (capacity > 0 AND current_tasks <= capacity),
    availability REAL,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_workload_worker FOREIGN KEY (worker_id) REFERENCES workers(worker_id) ON DELETE CASCADE,
    CONSTRAINT workload_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Worker Skills Table
CREATE TABLE skills (
    id BIGSERIAL PRIMARY KEY,
    worker_id TEXT NOT NULL,
    tenant_id  uuid         NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    visibility VARCHAR(16)  NOT NULL DEFAULT 'tenant',
    skill_name VARCHAR(255) NOT NULL,
    skill_level skill_level NOT NULL,
    -- text[], NOT jsonb — same reason as tools.tags above. Production is
    -- `text[] NOT NULL DEFAULT '{}'::text[]`. Declaring jsonb here is what let
    -- `src/db/skills_db.py` ship a write path that CI accepted and production
    -- rejects with `column "domains" is of type text[] but expression is of
    -- type jsonb`. Correcting this line makes CI able to catch that.
    domains TEXT[] NOT NULL DEFAULT '{}'::TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_skill_worker FOREIGN KEY (worker_id) REFERENCES workers(worker_id) ON DELETE CASCADE,
    CONSTRAINT unique_worker_skill UNIQUE (worker_id, skill_name),
    CONSTRAINT skills_visibility_check CHECK (visibility IN ('tenant', 'user'))
);

-- Schema Migrations Table
CREATE TABLE schema_migrations (
    id BIGSERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Workflow Routing Pointer Table (migration 014): (card_type, list) ->
-- active_workflow_id pointer for blue/green canary dispatch, with a
-- prior_workflow_id retained for rollback during the observation window.
CREATE TABLE workflow_routing_pointer (
    card_type                 text        NOT NULL,
    list                      text        NOT NULL,
    active_workflow_id        uuid        NOT NULL,
    prior_workflow_id         uuid        NULL,
    promoted_at               timestamptz NOT NULL DEFAULT now(),
    observation_window_until  timestamptz NOT NULL,
    rollback_thresholds       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (card_type, list)
);

-- ============================================================================
-- INDEXES
-- ============================================================================
--
-- Tenant-ownership indexes (idx_*_tenant_subject / idx_*_tenant_visibility and
-- the tenant composites) match the names migrations 010 and 019 create with
-- IF NOT EXISTS, so those migrations skip them on a fresh, schema-bootstrapped
-- database.

-- Workers indexes
CREATE INDEX idx_workers_worker_id ON workers(worker_id);
CREATE INDEX idx_workers_worker_name ON workers(worker_name);
CREATE INDEX idx_workers_tenant_subject ON workers(tenant_id, subject_id);
CREATE INDEX idx_workers_tenant_visibility ON workers(tenant_id, visibility);
CREATE INDEX idx_workers_status ON workers(status);
CREATE INDEX idx_workers_operational_status ON workers(operational_status);
CREATE INDEX idx_workers_config ON workers USING GIN(config);
-- config_revision diff-across-tenants lookups (migration 016); partial.
CREATE INDEX workers_tenant_revision_idx
    ON workers (tenant_id, config_revision)
    WHERE config_revision IS NOT NULL;

-- Worker States indexes
CREATE INDEX idx_worker_states_worker_id ON worker_states(worker_id);
CREATE INDEX idx_worker_states_tenant_subject ON worker_states(tenant_id, subject_id);
CREATE INDEX idx_worker_states_tenant_visibility ON worker_states(tenant_id, visibility);
CREATE INDEX idx_worker_states_tenant_worker_id ON worker_states(tenant_id, worker_id);
-- Third column `id` is load-bearing, not cosmetic: the retention sweep's
-- per-worker keep-guard filters `id > s.id` and orders by `id`, which the
-- two-column index above cannot serve (see migrations/027).
CREATE INDEX idx_worker_states_tenant_worker_id_id ON worker_states(tenant_id, worker_id, id);
CREATE INDEX idx_worker_states_timestamp ON worker_states(timestamp DESC);
CREATE INDEX idx_worker_states_state_type ON worker_states(state_type);
CREATE INDEX idx_worker_states_worker_state ON worker_states USING GIN(worker_state);

-- Graphs indexes
CREATE INDEX idx_graphs_name ON graphs(name);
CREATE INDEX idx_graphs_tenant_subject ON graphs(tenant_id, subject_id);
CREATE INDEX idx_graphs_tenant_visibility ON graphs(tenant_id, visibility);
CREATE INDEX idx_graphs_nodes ON graphs USING GIN(nodes);
CREATE INDEX idx_graphs_edges ON graphs USING GIN(edges);

-- Graph States indexes
CREATE INDEX idx_graph_states_graph_id ON graph_states(graph_id);
CREATE INDEX idx_graph_states_tenant_subject ON graph_states(tenant_id, subject_id);
CREATE INDEX idx_graph_states_tenant_visibility ON graph_states(tenant_id, visibility);
CREATE INDEX idx_graph_states_tenant_graph_id ON graph_states(tenant_id, graph_id);
CREATE INDEX idx_graph_states_created_at ON graph_states(created_at DESC);

-- Graph Execution Sessions indexes
CREATE INDEX idx_graph_exec_session_id ON graph_execution_sessions(session_id);
CREATE INDEX idx_graph_exec_graph_id ON graph_execution_sessions(graph_id);
CREATE INDEX idx_graph_exec_tenant_subject ON graph_execution_sessions(tenant_id, subject_id);
CREATE INDEX idx_graph_exec_tenant_visibility ON graph_execution_sessions(tenant_id, visibility);
CREATE INDEX idx_graph_exec_status ON graph_execution_sessions(status);
-- Composite index for trainer/flywheel pull queries (migration 009).
CREATE INDEX graph_execution_sessions_graph_id_created_at_idx
    ON graph_execution_sessions (graph_id, created_at DESC);
-- correlation_id join-key lookups (migration 012); partial since most rows
-- predate correlation propagation.
CREATE INDEX idx_ges_correlation
    ON graph_execution_sessions (correlation_id)
    WHERE correlation_id IS NOT NULL;
-- Eval-set sampler pull pattern: tenant filter + ORDER BY created_at DESC
-- (migration 015).
CREATE INDEX idx_graph_exec_tenant_created_at
    ON graph_execution_sessions (tenant_id, created_at DESC);

-- Node Execution Logs indexes
CREATE INDEX idx_node_exec_session_id ON node_execution_logs(session_id);
CREATE INDEX idx_node_exec_node_id ON node_execution_logs(node_id);
CREATE INDEX idx_node_exec_worker ON node_execution_logs(worker_name);
CREATE INDEX idx_node_exec_tenant_subject ON node_execution_logs(tenant_id, subject_id);
CREATE INDEX idx_node_exec_tenant_visibility ON node_execution_logs(tenant_id, visibility);
CREATE INDEX idx_node_exec_tenant_session ON node_execution_logs(tenant_id, session_id);

-- Execution Events indexes
CREATE INDEX idx_exec_events_session_id ON execution_events(session_id);
CREATE INDEX idx_exec_events_timestamp ON execution_events(timestamp DESC);
CREATE INDEX idx_exec_events_event_type ON execution_events(event_type);
CREATE INDEX idx_execution_events_tenant_subject ON execution_events(tenant_id, subject_id);
CREATE INDEX idx_execution_events_tenant_visibility ON execution_events(tenant_id, visibility);
CREATE INDEX idx_exec_events_tenant_session ON execution_events(tenant_id, session_id);

-- Tools indexes
CREATE INDEX idx_tools_name ON tools(name);
CREATE INDEX idx_tools_tenant_subject ON tools(tenant_id, subject_id);
CREATE INDEX idx_tools_tenant_visibility ON tools(tenant_id, visibility);
CREATE INDEX idx_tools_tags ON tools USING GIN(tags);
CREATE INDEX idx_tools_valid ON tools(is_valid);
CREATE INDEX idx_tools_version ON tools(version);

-- Workload indexes
CREATE INDEX idx_workload_worker_id ON workload(worker_id);
CREATE INDEX idx_workload_availability ON workload(availability);
CREATE INDEX idx_workload_tenant_subject ON workload(tenant_id, subject_id);
CREATE INDEX idx_workload_tenant_visibility ON workload(tenant_id, visibility);
CREATE INDEX idx_workload_tenant_worker ON workload(tenant_id, worker_id);

-- Skills indexes
CREATE INDEX idx_skills_worker_id ON skills(worker_id);
CREATE INDEX idx_skills_skill_name ON skills(skill_name);
CREATE INDEX idx_skills_domains ON skills USING GIN(domains);
CREATE INDEX idx_skills_tenant_subject ON skills(tenant_id, subject_id);
CREATE INDEX idx_skills_tenant_visibility ON skills(tenant_id, visibility);
CREATE INDEX idx_skills_tenant_worker ON skills(tenant_id, worker_id);

-- Tenant-name lookup indexes (migration 011). Names match the
-- ``ix_<table>_tenant_name`` that migration 011 creates with IF NOT EXISTS,
-- so it skips them on a fresh, schema-bootstrapped database.
CREATE INDEX ix_workers_tenant_name ON workers(tenant_name);
CREATE INDEX ix_skills_tenant_name ON skills(tenant_name);
CREATE INDEX ix_workload_tenant_name ON workload(tenant_name);
CREATE INDEX ix_worker_states_tenant_name ON worker_states(tenant_name);
CREATE INDEX ix_graphs_tenant_name ON graphs(tenant_name);
CREATE INDEX ix_graph_states_tenant_name ON graph_states(tenant_name);
CREATE INDEX ix_graph_execution_sessions_tenant_name ON graph_execution_sessions(tenant_name);
CREATE INDEX ix_node_execution_logs_tenant_name ON node_execution_logs(tenant_name);
CREATE INDEX ix_execution_events_tenant_name ON execution_events(tenant_name);
CREATE INDEX ix_tools_tenant_name ON tools(tenant_name);

-- Workflow routing pointer indexes (migration 014).
CREATE INDEX workflow_routing_pointer_active
    ON workflow_routing_pointer (active_workflow_id);
-- Partial index narrows the rollback-monitor scan to rows still inside an
-- observation window (i.e. with a prior version available to roll back to).
CREATE INDEX workflow_routing_pointer_observation
    ON workflow_routing_pointer (observation_window_until)
    WHERE prior_workflow_id IS NOT NULL;

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update_updated_at_column trigger to all relevant tables
CREATE TRIGGER update_workers_updated_at
    BEFORE UPDATE ON workers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_graph_states_updated_at
    BEFORE UPDATE ON graph_states
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_graphs_updated_at
    BEFORE UPDATE ON graphs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tools_updated_at
    BEFORE UPDATE ON tools
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_skills_updated_at
    BEFORE UPDATE ON skills
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_graph_execution_sessions_updated_at
    BEFORE UPDATE ON graph_execution_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert initial migration record
INSERT INTO schema_migrations (version, description)
VALUES ('1.0.0', 'Initial PostgreSQL schema for worker management system');

-- ============================================================================
-- GRANT PERMISSIONS (Run after creating user)
-- ============================================================================

-- Instructions for database administrator:
-- 1. First create user (if not already created):
--    CREATE USER worker_mgmt_user WITH PASSWORD 'your_secure_password_here';
--
-- 2. Grant database permissions:
--    GRANT ALL PRIVILEGES ON DATABASE worker_management TO worker_mgmt_user;
--    GRANT ALL ON SCHEMA public TO worker_mgmt_user;
--
-- 3. Grant table permissions:
--    GRANT ALL ON ALL TABLES IN SCHEMA public TO worker_mgmt_user;
--    GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO worker_mgmt_user;
--
-- 4. Grant default privileges for future objects:
--    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO worker_mgmt_user;
--    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO worker_mgmt_user;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Verify all tables were created
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;

-- Verify all indexes were created
-- SELECT indexname FROM pg_indexes WHERE schemaname = 'public' ORDER BY indexname;

-- Check schema migration
-- SELECT * FROM schema_migrations;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
