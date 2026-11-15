-- Migration: Workflow routing pointer (blue/green canary)
-- Version: 014
-- Description: Adds the (card_type, list) → active_workflow_id pointer table
--   used by n8n entry workflows to dispatch to the correct workflow version,
--   plus a prior_workflow_id for rollback during the GATE-2 observation
--   window. Spec: Improvement Plan 4 / continuous-improvement-design §10.2.
--
--   Lookup model:
--     n8n entry workflow gets (card_type, list) from kanban event,
--     calls GET /api/workflow-routing/active?card_type=&list= and
--     dispatches to active_workflow_id. After promote(), prior_workflow_id
--     stays populated until observation_window_until passes (then a monitor
--     workflow calls /clear-prior to lock the new version in) or a regression
--     trips the rollback monitor (which swaps the two ids).
--
-- Idempotent: CREATE TABLE / CREATE INDEX use IF NOT EXISTS, schema_migrations
--   row inserted with ON CONFLICT (version) DO NOTHING (matches the pattern
--   established in 013_fix_schema_migrations_unique.sql).

BEGIN;

CREATE TABLE IF NOT EXISTS workflow_routing_pointer (
    card_type                 text        NOT NULL,
    list                      text        NOT NULL,
    active_workflow_id        uuid        NOT NULL,
    prior_workflow_id         uuid        NULL,
    promoted_at               timestamptz NOT NULL DEFAULT now(),
    observation_window_until  timestamptz NOT NULL,
    rollback_thresholds       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (card_type, list)
);

CREATE INDEX IF NOT EXISTS workflow_routing_pointer_active
    ON workflow_routing_pointer (active_workflow_id);

-- Partial index narrows the rollback-monitor scan to rows still inside an
-- observation window (i.e. with a prior version available to roll back to).
CREATE INDEX IF NOT EXISTS workflow_routing_pointer_observation
    ON workflow_routing_pointer (observation_window_until)
    WHERE prior_workflow_id IS NOT NULL;

INSERT INTO schema_migrations (version, description) VALUES
    ('014', 'Workflow routing pointer (blue/green canary) for (card_type, list)')
ON CONFLICT (version) DO NOTHING;

COMMIT;
