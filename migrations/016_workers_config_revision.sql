-- Migration: 016_workers_config_revision — persist client-stamped config revision
-- Date: 2026-05-04
--
-- Until now `config_revision` was only client-side intent: seed_fr_agent.py
-- printed it (line 177) and fr/backend/src/agent/wm-bootstrap.ts used it as
-- a local cache key (line 113). Neither side actually shipped the value to
-- WM, and WorkerCreateRequest's `extra="ignore"` would have silently dropped
-- it anyway. Operators had no way to ask "what revision is fr-agent-default
-- at?" — making the "lockstep TS/Python constants" rule unenforceable in
-- prod.
--
-- This migration adds a top-level `config_revision` column (free text, max
-- 64 chars enforced at the API layer) plus a partial index so operators can
-- diff revisions across tenants without a full scan.
--
-- Forward-only / additive. NULL means "row predates this column or the
-- caller didn't stamp one"; both are valid. No backfill — there is nothing
-- on disk to recover, the constants were never persisted.

BEGIN;

ALTER TABLE workers
    ADD COLUMN IF NOT EXISTS config_revision TEXT;

CREATE INDEX IF NOT EXISTS workers_tenant_revision_idx
    ON workers (tenant_id, config_revision)
 WHERE config_revision IS NOT NULL;

COMMIT;
