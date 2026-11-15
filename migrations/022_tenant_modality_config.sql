-- Migration: 022_tenant_modality_config — per-tenant BYO provider config for
-- the non-LLM modalities (tts / asr / image / embedding).
-- Date: 2026-07-07
--
-- Slice 3 of "Bring Your Own" (BYO). tenant_llm_config (021) covers LLM workers
-- that funnel through worker_management. The modality backends (tts_backend,
-- asr_backend, image_gen_backend, mlx_backend) bypass WM entirely, so they need
-- their own per-tenant provider config. Rather than a table per service, this is
-- ONE table with a `modality` dimension; WM stays the single config authority
-- that holds WM_SECRET_KEY and decrypts keys at /tenant-config/modality/resolve
-- (the modality backends never hold the master key — per-service-secrets stays
-- intact).
--
-- Keyed (tenant_id, modality, role): mirrors tenant_llm_config's role model
-- ('__default__' + optional 'tier:<x>'), with `modality` added. Purely additive
-- rows per modality — no migration redo to add another modality or tier.
--
-- `provider` is TEXT (not an enum) so adding a provider never needs ALTER TYPE.
-- `api_key_encrypted` is AES-256-GCM ciphertext (src/security/secret_box.py);
-- plaintext is never stored and never returned by the API. tenant_id is
-- VARCHAR(255) to match the existing worker_management isolation columns.

BEGIN;

CREATE TABLE IF NOT EXISTS tenant_modality_config (
    tenant_id          VARCHAR(255) NOT NULL,
    modality           TEXT         NOT NULL,
    role               TEXT         NOT NULL DEFAULT '__default__',
    provider           TEXT         NOT NULL,
    model              TEXT         NOT NULL DEFAULT '',
    base_url           TEXT,
    params             JSONB        NOT NULL DEFAULT '{}'::jsonb,
    api_key_encrypted  TEXT,
    cost_source        TEXT         NOT NULL DEFAULT 'worker-management',
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT tenant_modality_config_pkey PRIMARY KEY (tenant_id, modality, role)
);

CREATE INDEX IF NOT EXISTS idx_tenant_modality_config_tenant
    ON tenant_modality_config (tenant_id);

INSERT INTO schema_migrations (version, description)
VALUES ('022', 'Per-tenant BYO modality (tts/asr/image/embedding) provider config')
ON CONFLICT (version) DO NOTHING;

COMMIT;
