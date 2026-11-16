-- Migration: 025_purge_local_preset_paid_tier_rows — delete the two
-- tenant_llm_config rows that said `provider=local` and billed api.deepseek.com.
-- Date: 2026-08-26
--
-- ⭐ WHAT THIS COST, measured on prod 2026-08-26. Tenant
-- aaaaaaaa-0000-4000-8000-000000000001 (platform-cp-1) spent **$140.31 lifetime
-- over 46,102 requests** on paid DeepSeek. Nobody ever named DeepSeek: every
-- metered row reads model='tier-medium' -> served_model='deepseek-v4-pro'.
-- The whole spend traces to two rows inserted 2026-07-19 16:11 (billing starts
-- 15:45 the same day):
--
--     role='tier:medium'   provider='local'   model='tier-medium'
--     role='tier:high'     provider='local'   model='tier-medium'
--
-- `tier-medium` names a work SIZE, not a vendor. The live routing configmap
-- (gitops clusters/strato/services/inference-router/providers-configmap.yaml)
-- owns that alias solely on the paid `deepseek` provider, so rows that say
-- `local` egressed to https://api.deepseek.com/v1 for thirteen months of
-- KAG/karma extraction traffic. The repo's own billing table agrees:
-- `_BILLING_PROVIDER_OVERRIDES["local"]["tier-medium"] == "deepseek"`
-- (src/worker/agents/provider_catalog.py).
--
-- WHY DELETE RATHER THAN REWRITE
-- ------------------------------
-- Absence is the intended safe state, not a second-best. With no `tier:medium`
-- / `tier:high` row, resolve_llm_config falls back to the `__default__` row and
-- then to seeded system routing (inference-router -> pgx-local), which is
-- exactly where this tenant's work belongs. Rewriting the rows to
-- model='local-llm' would encode a NEW per-tier decision nobody made, and would
-- re-assert the tier rows as deliberate configuration. Deleting them removes
-- the decision instead of replacing it.
--
-- WHAT THIS DELIBERATELY DOES NOT TOUCH
-- -------------------------------------
-- * `fr-agent-default` (`tier-flash` -> deepseek-v4-flash, $37.70 lifetime) is
--   a deliberate 2026-07-19 decision and is explicitly KEPT. It is safe from
--   this file by construction: it is a row in `workers`, written by
--   scripts/seed_fr_agent.py, and has no `tenant_llm_config` row at all.
-- * Any other tenant, any other role, and this tenant's `__default__` row.
-- * Rows written from 2026-08-26 onward — see the created_at bound below.
--
-- SAFETY OF THE PREDICATE
-- -----------------------
-- Every clause is required, and each one independently narrows the blast radius
-- to the two known rows:
--   * tenant_id  — this tenant only.
--   * role       — the two tier roles only; `__default__` survives untouched.
--   * provider   — only a row that CLAIMS local. A tenant that has since made a
--                  deliberate `deepseek` choice keeps it.
--   * model      — only the work-size alias. `deepseek-v4-pro` names its own
--                  destination and is a consented choice; it is left alone,
--                  matching _local_safe_model's scoping rule (2026-08-16).
--   * created_at — the offending rows are from 2026-07-19. Bounding at
--                  2026-08-26 matters because run_migrations() re-runs EVERY
--                  file on EVERY pod boot: without it, a future operator who
--                  deliberately re-creates such a row would have it silently
--                  deleted at the next restart. The write-path guard added in
--                  the same change (assert_local_preset_stays_on_hardware,
--                  src/worker/agents/provider_catalog.py) is what prevents
--                  recurrence; this file only repairs history.
--
-- Idempotent and forward-only: re-running deletes nothing more, which
-- tests/integration/test_migration_chain.py (whole chain, twice, strict)
-- requires.

BEGIN;

DELETE FROM tenant_llm_config
 WHERE tenant_id = 'aaaaaaaa-0000-4000-8000-000000000001'::uuid
   AND role IN ('tier:medium', 'tier:high')
   AND provider = 'local'
   AND model = 'tier-medium'
   AND created_at < TIMESTAMPTZ '2026-08-26 00:00:00+00';

INSERT INTO schema_migrations (version, description)
VALUES ('025', 'Purge the two local-preset tier rows that billed paid DeepSeek')
ON CONFLICT (version) DO NOTHING;

COMMIT;
