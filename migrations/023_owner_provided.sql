-- Migration: 023_owner_provided — mark a stored BYO key as owner-provided
-- (managed BYO) rather than tenant-supplied, on both per-tenant key tables.
-- Date: 2026-07-08
--
-- P3 of the two-axis commercial model ("managed BYO"). Until now a per-tenant
-- external provider key was byte-identical whether the TENANT supplied it (they
-- pay the provider; the platform takes only a fee) or the platform OWNER
-- supplied it on the tenant's behalf as part of a paid offering (the platform
-- pays the provider and bills the tenant marked-up). This column records that
-- provenance so the resolve seam + cost emit can distinguish the two: an
-- owner-provided line is emitted as a normal marked-up cost event
-- (usage_extra.owner_provided="true", no byo flag) instead of a force-zeroed
-- BYO line.
--
-- Purely additive; default false preserves every existing row's current
-- tenant-BYO behaviour. Applied to both tenant_llm_config (021) and
-- tenant_modality_config (022).

BEGIN;

ALTER TABLE tenant_llm_config
    ADD COLUMN IF NOT EXISTS owner_provided BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE tenant_modality_config
    ADD COLUMN IF NOT EXISTS owner_provided BOOLEAN NOT NULL DEFAULT false;

INSERT INTO schema_migrations (version, description)
VALUES ('023', 'Mark stored BYO key as owner-provided (managed BYO) on both per-tenant key tables')
ON CONFLICT (version) DO NOTHING;

COMMIT;
