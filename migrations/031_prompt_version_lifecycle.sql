-- Migration: 031_prompt_version_lifecycle — DRAFT/CANARY/PUBLISHED/ARCHIVED/DISCARDED
-- Date: 2026-09-28
--
-- WHY. Migration 030 gave the ledger two statuses, `active` / `retired`, with a
-- partial unique index allowing ONE active version per prompt. That cannot
-- express the lifecycle the conveyor needs (P4 contract; twin of
-- procedure_management D4 R9/R10): a version under canary, a version that was
-- live and has been superseded, and one that was abandoned are all "retired".
--
-- MAPPING (binding, P4 contract §1):
--   active                                → PUBLISHED
--   retired AND activated_at IS NULL      → DRAFT     (never live)
--   retired AND activated_at IS NOT NULL  → ARCHIVED  (was live once)
-- Measured on prod 2026-09-28: 41 rows, all `retired`, 0 with activated_at ⇒
-- all 41 become DRAFT. Nothing is live today, so nothing a caller resolves moves.
--
-- `prompt_versions_one_active_idx` is DROPPED: several PUBLISHED versions may
-- coexist and the highest one wins (resolver, behind
-- PROMPTS_READ_REQUIRES_PUBLISHED). The deprecated `POST …/activate` keeps
-- today's "one live" semantics by archiving the other PUBLISHED versions itself.
--
-- PRECONDITION. Refuse when any prompt_id has more than one `active` row. The
-- 030 index makes that impossible while it exists; the check is for a database
-- where someone dropped it by hand — mapping two actives to two PUBLISHED would
-- silently pick a winner by version number instead of by who activated last.
--
-- ROLLOUT SAFETY. The CHECK added below is TRANSITIONAL (it still admits the
-- legacy spellings an old pod writes) — see the comment on it, and the
-- follow-up 032 that tightens it. If this file FAILS under the production
-- runner (strict=False logs and starts the pod anyway), the new code detects
-- the 030 CHECK at startup and its prompt write routes answer 503
-- (``repository.probe_lifecycle_schema``; /health ``prompt_lifecycle_schema``)
-- instead of 500ing on a CheckViolation.
--
-- IDEMPOTENT, as the startup runner requires: the CHECK is dropped before the
-- UPDATE and re-added after it, the UPDATE only touches legacy values, and a
-- re-run on the end state is a no-op.
--
-- DOWN: migrations/down/031_prompt_version_lifecycle.down.sql (NOT in this
-- directory's top level, so the runner never applies it).

BEGIN;

DO $$
DECLARE
    dup TEXT;
BEGIN
    SELECT prompt_id INTO dup
      FROM prompt_versions
     WHERE status = 'active'
     GROUP BY prompt_id
    HAVING count(*) > 1
     LIMIT 1;
    IF dup IS NOT NULL THEN
        RAISE EXCEPTION
            '031_prompt_version_lifecycle refused: prompt_id % has more than one active version; resolve it by hand first',
            dup;
    END IF;
END
$$;

ALTER TABLE prompt_versions DROP CONSTRAINT IF EXISTS prompt_versions_status_check;

DROP INDEX IF EXISTS prompt_versions_one_active_idx;

UPDATE prompt_versions
   SET status = CASE
           WHEN status = 'active' THEN 'PUBLISHED'
           WHEN activated_at IS NULL THEN 'DRAFT'
           ELSE 'ARCHIVED'
       END
 WHERE status IN ('active', 'retired');

ALTER TABLE prompt_versions ALTER COLUMN status SET DEFAULT 'DRAFT';

-- TRANSITIONAL CHECK: the lifecycle set PLUS the legacy 030 spellings.
-- During the rolling deploy a pre-031 pod is still serving next to the new
-- one, and it WRITES 'retired' (every PUT) and 'active'/'retired' (activate).
-- A CHECK without them would turn every one of those into a 500 for the
-- length of the rollout. The new code never writes a legacy value and reads
-- one through the same mapping as the UPDATE above
-- (src/api/prompts/repository.py ``effective_status``), so a row an old pod
-- writes in the window reads correctly.
--
-- FOLLOW-UP (required, once no pre-031 pod can be running):
-- migrations/032_prompt_version_lifecycle_tighten.sql re-runs the mapping
-- UPDATE above for any leftover legacy rows and replaces this CHECK with the
-- lifecycle set only ('DRAFT','CANARY','PUBLISHED','ARCHIVED','DISCARDED').
ALTER TABLE prompt_versions
    ADD CONSTRAINT prompt_versions_status_check
    CHECK (status IN ('DRAFT', 'CANARY', 'PUBLISHED', 'ARCHIVED', 'DISCARDED',
                      'active', 'retired'));

-- The resolver's query: highest PUBLISHED version of one prompt.
CREATE INDEX IF NOT EXISTS prompt_versions_status_idx
    ON prompt_versions (prompt_id, status, version DESC);

COMMIT;
