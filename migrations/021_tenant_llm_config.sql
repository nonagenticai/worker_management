-- Migration: 021_tenant_llm_config — per-tenant BYO LLM provider config
-- Date: 2026-06-19
--
-- Phase 1 of "Bring Your Own LLM key/provider" (BYO). Today every worker is
-- seeded with model_provider='inference_router' (or 'mlx'), routing all LLM
-- traffic through inference-router -> local pgx-local vLLM. Commercial tenants
-- must instead run against THEIR OWN provider + key so they never touch owner
-- GPU hardware. This table is the single per-tenant source of truth that the
-- run-time resolution seam overlays onto every worker's config — so fr / kag /
-- coding-ai / karma all inherit it with no per-app change.
--
-- Keyed (tenant_id, role): Phase 1 only ever writes role='__default__' (one
-- tenant-wide default model). Phase 2 inserts per-role rows (kag_query_expand,
-- eval_judge, …) WITHOUT a migration redo — purely additive.
--
-- `provider` is TEXT (not an enum) so adding a provider never needs ALTER TYPE.
-- `api_key_encrypted` is AES-256-GCM ciphertext (src/security/secret_box.py);
-- plaintext is never stored and never returned by the API.
--
-- Forward-only / additive. tenant_id is VARCHAR(255) to match the existing
-- worker_management isolation columns (workers.tenant_id etc.), NOT UUID.

BEGIN;

CREATE TABLE IF NOT EXISTS tenant_llm_config (
    tenant_id          VARCHAR(255) NOT NULL,
    role               TEXT         NOT NULL DEFAULT '__default__',
    provider           TEXT         NOT NULL,
    model              TEXT         NOT NULL DEFAULT '',
    base_url           TEXT,
    temperature        REAL,
    max_tokens         INTEGER,
    params             JSONB        NOT NULL DEFAULT '{}'::jsonb,
    api_key_encrypted  TEXT,
    cost_source        TEXT         NOT NULL DEFAULT 'worker-management',
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT tenant_llm_config_pkey PRIMARY KEY (tenant_id, role)
);

CREATE INDEX IF NOT EXISTS idx_tenant_llm_config_tenant
    ON tenant_llm_config (tenant_id);

INSERT INTO schema_migrations (version, description)
VALUES ('021', 'Per-tenant BYO LLM provider config (tenant_llm_config)')
ON CONFLICT (version) DO NOTHING;

COMMIT;
