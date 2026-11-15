-- Migration: Add output_schema and output_validator columns to workers table
-- These columns support structured output validation for agent workers

ALTER TABLE workers
ADD COLUMN IF NOT EXISTS output_schema JSONB DEFAULT NULL,
ADD COLUMN IF NOT EXISTS output_validator TEXT DEFAULT NULL;

-- Add comment for documentation
COMMENT ON COLUMN workers.output_schema IS 'JSON Schema for validating agent output';
COMMENT ON COLUMN workers.output_validator IS 'Python validator class name for output validation';
