-- DOWN for migrations/036_workflow_routing_release.sql. Applied BY HAND only
-- (psql -1 -f); the startup runner lists top-level migrations/*.sql and never
-- descends into this directory.
--
-- Restores 014's shape: PK (card_type, list), uuid ids, NOT NULL
-- observation_window_until, both 014 indexes, and none of 036's columns or
-- CHECKs.
--
-- ⚠️ DESTRUCTIVE: deletes EVERY pointer row first. 036's rows are keyed by
-- asset_key and carry 16-char n8n ids; neither fits 014 (no (card_type, list)
-- key, and an n8n id does not cast to uuid). Set AA_WORKFLOW_CANARY_ROUTABLE=false
-- FIRST and deploy the pre-036 image: the 036 code reads asset_key and 500s
-- on 014's shape (the resolver then falls back to the env map, which is the
-- same dispatch as with no pointer at all). 036's schema_migration_files row is
-- removed so the runner re-applies it on the next start.

BEGIN;

DELETE FROM workflow_routing_pointer;

DROP INDEX IF EXISTS workflow_routing_pointer_active;

ALTER TABLE workflow_routing_pointer
    DROP CONSTRAINT IF EXISTS wrp_workflow_id_shape,
    DROP CONSTRAINT IF EXISTS wrp_canary_all_or_none,
    DROP CONSTRAINT IF EXISTS wrp_canary_differs,
    DROP CONSTRAINT IF EXISTS workflow_routing_pointer_pkey;

ALTER TABLE workflow_routing_pointer
    DROP COLUMN IF EXISTS asset_key,
    DROP COLUMN IF EXISTS canary_workflow_id,
    DROP COLUMN IF EXISTS canary_pct,
    DROP COLUMN IF EXISTS canary_id,
    DROP COLUMN IF EXISTS released_by_card_id,
    DROP COLUMN IF EXISTS released_by_tenant_id,
    DROP COLUMN IF EXISTS trace_id,
    DROP COLUMN IF EXISTS promoted_trace_id,
    DROP COLUMN IF EXISTS released_at,
    ALTER COLUMN active_workflow_id TYPE uuid USING active_workflow_id::uuid,
    ALTER COLUMN prior_workflow_id  TYPE uuid USING prior_workflow_id::uuid,
    ALTER COLUMN card_type SET NOT NULL,
    ALTER COLUMN list SET NOT NULL,
    ALTER COLUMN observation_window_until SET NOT NULL,
    ADD CONSTRAINT workflow_routing_pointer_pkey PRIMARY KEY (card_type, list);

CREATE INDEX IF NOT EXISTS workflow_routing_pointer_active
    ON workflow_routing_pointer (active_workflow_id);

CREATE INDEX IF NOT EXISTS workflow_routing_pointer_observation
    ON workflow_routing_pointer (observation_window_until)
    WHERE prior_workflow_id IS NOT NULL;

DELETE FROM schema_migration_files WHERE filename = '036_workflow_routing_release.sql';

COMMIT;
