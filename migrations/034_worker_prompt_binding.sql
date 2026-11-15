-- Migration: 034_worker_prompt_binding — which prompt a platform worker RUNS
-- Date: 2026-10-01
--
-- WHY. A worker's runtime system prompt is read from
-- workers.config->>'system_prompt' (src/api/workers/_globals.py), and nothing
-- resolved a PUBLISHED prompt_versions row at runtime — so a prompt release
-- changed nothing that ran. Worse, n8n "ensure" nodes keep REWRITING
-- config.system_prompt, a second writer that silently undoes any hand-edit.
--
-- This table is the missing link: one row says "worker W of tenant T runs the
-- highest PUBLISHED version of prompt P". It is read by
-- src/api/workers/prompt_source.py, ONLY when WM_PROMPT_SOURCE=published AND the
-- tenant is a platform tenant (src/api/tenant_constants.py). A customer tenant
-- is never resolved, even if a row exists for it.
--
-- Rows are written by scripts/adopt_live_worker_prompts.py (bound_by
-- 'adopt_live_worker_prompts'), which first publishes the worker's LIVE config
-- text as v1 of its prompt — so flipping the flag runs the same bytes as before.
--
-- NO FK to prompt_versions: its PK is (prompt_id, version) and a binding names a
-- prompt, not a version. Integrity is checked in code: a binding whose prompt
-- has no PUBLISHED version resolves to nothing and the worker keeps its legacy
-- config prompt (fail-safe, never a 500).
--
-- IDEMPOTENT: IF NOT EXISTS everywhere.
--
-- DOWN: migrations/down/034_worker_prompt_binding.down.sql (never run by the
-- runner).

BEGIN;

CREATE TABLE IF NOT EXISTS worker_prompt_binding (
    tenant_id   TEXT        NOT NULL,
    worker_name TEXT        NOT NULL,
    prompt_id   TEXT        NOT NULL,
    bound_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    bound_by    TEXT        NOT NULL,
    PRIMARY KEY (tenant_id, worker_name)
);

CREATE INDEX IF NOT EXISTS worker_prompt_binding_prompt_id_idx
    ON worker_prompt_binding (prompt_id);

COMMIT;
