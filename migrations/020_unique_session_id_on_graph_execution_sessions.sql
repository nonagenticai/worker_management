-- Migration: Ensure a single-column UNIQUE index on graph_execution_sessions.session_id
-- Version: 020
-- Description:
--   ``upsert_finalized_session`` (src/db/graph_execution_db.py) finalizes every
--   graph execution with ``INSERT ... ON CONFLICT (session_id) DO UPDATE``.
--   ``postgres_schema.sql`` declares ``session_id UUID UNIQUE`` so FRESH/CI DBs
--   have the matching unique constraint and the upsert works (proven by
--   tests/integration/test_inference_graph_end_to_end.py +
--   test_quality_score_graph_persist.py).
--
--   But the LIVE (migration-built) ``graph_execution_sessions`` table had NO
--   single-column unique on ``session_id``: it was created by the early
--   (pre-postgres_schema.sql) migration chain, and migration 019's
--   ``DROP COLUMN creator_id CASCADE`` removed the creator_id-composite uniques.
--   019 step 3 deliberately skipped re-adding a session_id index, on the
--   *incorrect* assumption that "the existing UNIQUE index on
--   graph_execution_sessions.session_id" was still present. On migrated DBs it
--   was not -> ON CONFLICT (session_id) raised "there is no unique or exclusion
--   constraint matching the ON CONFLICT specification" -> finalize failed and
--   was swallowed (logged WARNING) -> graph_execution_sessions stayed EMPTY,
--   so the exec_quality signal (quality_score) never persisted.
--
--   This adds the missing unique index so finalize can upsert. A unique INDEX
--   (not a table CONSTRAINT) is used because it supports ``CREATE ... IF NOT
--   EXISTS`` and equally satisfies ON CONFLICT (session_id) inference.
--
-- Re-runnable (idempotent): the DO-block adds the index only when no
-- single-column unique on session_id already exists, so it is a no-op on fresh
-- DBs (which already have ``session_id UUID UNIQUE``) and on any DB where a
-- prior run / manual hotfix already created it.

BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indrelid
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY (i.indkey)
        WHERE c.relname = 'graph_execution_sessions'
          AND i.indisunique
          AND array_length(i.indkey, 1) = 1
          AND a.attname = 'session_id'
    ) THEN
        CREATE UNIQUE INDEX uq_graph_execution_sessions_session_id
            ON graph_execution_sessions (session_id);
    END IF;
END $$;

INSERT INTO schema_migrations (version, description)
VALUES ('020', 'Ensure single-column UNIQUE index on graph_execution_sessions.session_id')
ON CONFLICT (version) DO NOTHING;

COMMIT;
