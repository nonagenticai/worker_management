-- Convert worker_states.worker_state from TEXT to JSONB
-- Migration: 007_convert_worker_state_to_jsonb
--
-- The worker_states table has worker_state as TEXT but the code expects dict objects.
-- This breaks GET /workers/{name} which tries to access worker_state["status"]

DO $$
BEGIN
    -- Check if worker_states.worker_state is TEXT and convert to JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'worker_states' AND column_name = 'worker_state' AND data_type = 'text'
    ) THEN
        ALTER TABLE worker_states
        ALTER COLUMN worker_state TYPE JSONB USING worker_state::jsonb;
        RAISE NOTICE 'Converted worker_states.worker_state to JSONB';
    ELSE
        RAISE NOTICE 'worker_states.worker_state is already JSONB or does not exist';
    END IF;
END $$;

-- Record migration
INSERT INTO schema_migrations (version, description)
VALUES ('007', 'Convert worker_states.worker_state from TEXT to JSONB')
ON CONFLICT (version) DO NOTHING;
