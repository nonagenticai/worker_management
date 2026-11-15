-- Convert TEXT columns to JSONB for psycopg compatibility
-- Migration: 006_convert_to_jsonb
--
-- psycopg expects dict objects for JSONB columns, not JSON strings.
-- The production DB was created with TEXT columns but the code now passes dicts directly.

-- Workers table: config and output_schema
DO $$
BEGIN
    -- Check if config is TEXT and convert to JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'workers' AND column_name = 'config' AND data_type = 'text'
    ) THEN
        ALTER TABLE workers
        ALTER COLUMN config TYPE JSONB USING config::jsonb;
        RAISE NOTICE 'Converted workers.config to JSONB';
    END IF;

    -- Check if output_schema is TEXT and convert to JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'workers' AND column_name = 'output_schema' AND data_type = 'text'
    ) THEN
        ALTER TABLE workers
        ALTER COLUMN output_schema TYPE JSONB USING output_schema::jsonb;
        RAISE NOTICE 'Converted workers.output_schema to JSONB';
    END IF;

    -- Check if worker_states.worker_state is TEXT and convert to JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'worker_states' AND column_name = 'worker_state' AND data_type = 'text'
    ) THEN
        ALTER TABLE worker_states
        ALTER COLUMN worker_state TYPE JSONB USING worker_state::jsonb;
        RAISE NOTICE 'Converted worker_states.worker_state to JSONB';
    END IF;
END $$;

-- Record migration
INSERT INTO schema_migrations (version, description)
VALUES ('006', 'Convert TEXT columns to JSONB for psycopg compatibility')
ON CONFLICT (version) DO NOTHING;
