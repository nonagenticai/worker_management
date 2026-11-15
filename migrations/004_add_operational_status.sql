-- Migration: Add operational_status column to workers table
-- Date: 2025-01-09
-- Description: Adds missing operational_status column

ALTER TABLE workers ADD COLUMN IF NOT EXISTS operational_status VARCHAR(50) DEFAULT 'offline';

-- Add index if not exists
CREATE INDEX IF NOT EXISTS idx_workers_operational_status ON workers(operational_status);

-- Insert migration record
INSERT INTO schema_migrations (version, description)
VALUES ('1.0.4', 'Add operational_status column to workers table')
ON CONFLICT (version) DO NOTHING;
