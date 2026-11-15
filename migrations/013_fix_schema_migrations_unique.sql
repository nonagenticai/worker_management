-- Migration: Ensure schema_migrations.version has a UNIQUE constraint
-- Version: 013
-- Description: The production worker_management database was bootstrapped from
--   an older `postgres_schema.sql` that did not enforce UNIQUE on
--   `schema_migrations.version`. As a result every prior migration that ends
--   with `INSERT INTO schema_migrations ... ON CONFLICT (version) DO NOTHING`
--   (006, 007, 008, 009, 010, plus 004's variant) raises:
--     ERROR: there is no unique or exclusion constraint matching the
--            ON CONFLICT specification
--   The migration runner (`psql -f $f || echo`) shrugs the error off, the
--   schema work in those files DID get applied, but the bookkeeping row for
--   each migration is never recorded. This file fixes the bookkeeping table
--   so future re-runs of 004/006/007/008/009/010 succeed cleanly.
--
-- Idempotent: deduplicates first, then adds the constraint via ALTER TABLE
--   ... ADD CONSTRAINT IF NOT EXISTS-equivalent guarded by pg_constraint check.

BEGIN;

-- 1. Deduplicate any duplicate (version) rows that might exist from prior
--    half-applied migrations. Keep the lowest id.
DELETE FROM schema_migrations a
USING schema_migrations b
WHERE a.version = b.version
  AND a.id > b.id;

-- 2. Add the UNIQUE constraint if missing.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint con
        JOIN pg_class cl ON cl.oid = con.conrelid
        WHERE cl.relname = 'schema_migrations'
          AND con.contype = 'u'
          AND pg_get_constraintdef(con.oid) ILIKE '%(version)%'
    ) THEN
        ALTER TABLE schema_migrations
            ADD CONSTRAINT schema_migrations_version_key UNIQUE (version);
    END IF;
END $$;

-- 3. Now that the unique constraint exists, backfill the bookkeeping rows for
--    migrations that already ran (their schema work is present, but the INSERT
--    failed before the constraint existed). All ON CONFLICT (version) DO NOTHING
--    so this is safe to re-run.
INSERT INTO schema_migrations (version, description) VALUES
    ('1.0.4', 'Add operational_status column to workers table'),
    ('006',   'Convert TEXT columns to JSONB for psycopg compatibility'),
    ('007',   'Convert worker_states.worker_state from TEXT to JSONB'),
    ('0.8.0', 'Add description and variables columns to graphs table'),
    ('0.9.0', 'Add inference telemetry columns to graph_execution_sessions'),
    ('010',   'Per-user-within-tenant isolation (tenant_id, subject_id, visibility)'),
    ('013',   'Ensure schema_migrations.version UNIQUE + backfill bookkeeping')
ON CONFLICT (version) DO NOTHING;

COMMIT;
