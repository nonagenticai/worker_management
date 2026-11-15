-- Migration 008: Add description and variables columns to graphs table
-- These columns are referenced by graph_state_db.py but missing from the production schema.

ALTER TABLE graphs ADD COLUMN IF NOT EXISTS description TEXT NOT NULL DEFAULT '';
ALTER TABLE graphs ADD COLUMN IF NOT EXISTS variables JSONB NOT NULL DEFAULT '{}';

INSERT INTO schema_migrations (version, description)
VALUES ('0.8.0', 'Add description and variables columns to graphs table')
ON CONFLICT (version) DO NOTHING;
