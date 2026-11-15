-- Migration: 036_workflow_routing_release — the pointer becomes a per-ASSET
-- workflow release (active + one armed canary), keyed by the dispatch key.
-- Date: 2026-10-01
--
-- WHY (G-15 DESIGN D1/D3, approved by phase-10 REVIEW 2026-10-01).
-- Dispatch resolves workflows BY NAME (AA_ACTIVITY / AA_PATTERN / AA_CELL env
-- maps), so the pointer must answer "which workflow id runs for asset X", not
-- "for (card_type, list)". New PK ``asset_key`` = '<map>:<name>', e.g.
-- 'activity:AA Activity - Reasoning' or 'cell:doing'. card_type / list stay as
-- NULLABLE legacy columns.
--
-- Ids become TEXT: an n8n workflow id is a 16-char nanoid (measured 10-01 on
-- worker4: 364 of 364 workflow_entity ids have length 16), never a uuid — the
-- 014 uuid columns could not hold a single real workflow id.
--
-- ⛔ FAIL CLOSED ON DATA. The uuid -> text change is not data-preserving in
-- meaning (no uuid is a valid n8n id), so the DO block refuses the whole file
-- when ANY row exists. Measured 10-01: 0 rows. Re-verify 0 rows on worker4
-- (BEGIN READ ONLY, direct — never through pgbouncer) immediately before merge.
--
-- No clock (phase-10 decisions 5/7): observation_window_until becomes NULLABLE
-- and the observation index is dropped, not rebuilt. canary_pct is capped at
-- 50 — a canary is never the majority arm.
--
-- RE-RUNNABLE on an EMPTY table (tests/integration/pg_guard.apply_migrations
-- forgets the ledger and re-runs the whole chain strict): every statement is
-- IF [NOT] EXISTS or drop-then-add. With rows present a re-run RAISEs, which is
-- the intended refusal.
--
-- DOWN: migrations/down/036_workflow_routing_release.down.sql (by hand only).

BEGIN;

DO $$
BEGIN
  IF (SELECT count(*) FROM workflow_routing_pointer) > 0 THEN
    RAISE EXCEPTION '036: workflow_routing_pointer is not empty; refusing the id type change';
  END IF;
END $$;

DROP INDEX IF EXISTS workflow_routing_pointer_active;
DROP INDEX IF EXISTS workflow_routing_pointer_observation;

ALTER TABLE workflow_routing_pointer DROP CONSTRAINT IF EXISTS workflow_routing_pointer_pkey;

ALTER TABLE workflow_routing_pointer
    ALTER COLUMN card_type DROP NOT NULL,
    ALTER COLUMN list DROP NOT NULL,
    ALTER COLUMN active_workflow_id TYPE text USING active_workflow_id::text,
    ALTER COLUMN prior_workflow_id  TYPE text USING prior_workflow_id::text,
    ALTER COLUMN observation_window_until DROP NOT NULL,          -- decisions 5/7: no clock
    ADD COLUMN IF NOT EXISTS asset_key text,
    ADD COLUMN IF NOT EXISTS canary_workflow_id text,
    ADD COLUMN IF NOT EXISTS canary_pct smallint,
    ADD COLUMN IF NOT EXISTS canary_id text,
    ADD COLUMN IF NOT EXISTS released_by_card_id text,
    ADD COLUMN IF NOT EXISTS released_by_tenant_id text,
    ADD COLUMN IF NOT EXISTS trace_id text,                       -- decision 4: X-AA-Trace-Id of the release call
    ADD COLUMN IF NOT EXISTS promoted_trace_id text,
    ADD COLUMN IF NOT EXISTS released_at timestamptz;

ALTER TABLE workflow_routing_pointer
    DROP CONSTRAINT IF EXISTS wrp_workflow_id_shape,
    DROP CONSTRAINT IF EXISTS wrp_canary_all_or_none,
    DROP CONSTRAINT IF EXISTS wrp_canary_differs;

ALTER TABLE workflow_routing_pointer
    ALTER COLUMN asset_key SET NOT NULL,
    ADD CONSTRAINT workflow_routing_pointer_pkey PRIMARY KEY (asset_key),
    ADD CONSTRAINT wrp_workflow_id_shape CHECK (
        active_workflow_id ~ '^[A-Za-z0-9]{16}$'
        AND (prior_workflow_id IS NULL OR prior_workflow_id ~ '^[A-Za-z0-9]{16}$')
        AND (canary_workflow_id IS NULL OR canary_workflow_id ~ '^[A-Za-z0-9]{16}$')),
    ADD CONSTRAINT wrp_canary_all_or_none CHECK (
        (canary_workflow_id IS NULL AND canary_pct IS NULL AND canary_id IS NULL
            AND released_by_card_id IS NULL)
     OR (canary_workflow_id IS NOT NULL AND canary_pct BETWEEN 1 AND 50
            AND canary_id IS NOT NULL AND released_by_card_id IS NOT NULL)),
    ADD CONSTRAINT wrp_canary_differs CHECK (
        canary_workflow_id IS NULL OR canary_workflow_id <> active_workflow_id);

CREATE INDEX IF NOT EXISTS workflow_routing_pointer_active
    ON workflow_routing_pointer (active_workflow_id);

COMMIT;
