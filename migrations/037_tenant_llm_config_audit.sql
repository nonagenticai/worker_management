-- Migration: 037_tenant_llm_config_audit — the CUSTOMER-visible history of who
-- set, cleared, changed, removed or tested a tenant's BYO LLM config.
-- Date: 2026-10-02
--
-- WHY (owner ruling 2026-10-02). The owner (portal super_admin, via fr) may set,
-- test and remove a CUSTOMER tenant's BYO key. When that happens the customer
-- must be able to SEE it. The INFO ``owner_action`` log line is operator-only;
-- this table is what ``GET /tenant-config/llm/audit`` serves back to the tenant.
-- Every caller is recorded (platform_operator, workspace_admin, service), not
-- only the owner, so the customer sees the full history rather than only the
-- entries someone decided were interesting.
--
-- ⛔ WHAT IS NEVER STORED HERE: the plaintext key, any hash of the plaintext
-- key, and the actor's ``sub``. ``key_id`` is 12 hex of SHA-256 of the STORED
-- (encrypted) value — ``_key_id`` in src/api/tenant_config/_router.py — or the
-- literal ``none``; the CHECK below refuses any other shape, so a key pasted
-- into the column by a future bug is refused by the database, not by review.
-- The actor ``sub`` stays in the operator INFO line only.
--
-- Written in the SAME transaction as the tenant_llm_config change it records
-- (PUT/DELETE), so a config change without its audit row cannot commit.
--
-- tenant_id is ``uuid`` like tenant_llm_config since 024. Additive; re-runnable
-- (IF NOT EXISTS). DOWN: migrations/down/037_tenant_llm_config_audit.down.sql.

BEGIN;

CREATE TABLE IF NOT EXISTS tenant_llm_config_audit (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id   uuid        NOT NULL,
    role        TEXT        NOT NULL,
    action      TEXT        NOT NULL
        CONSTRAINT tenant_llm_config_audit_action_chk
        CHECK (action IN ('set_key', 'clear_key', 'update', 'remove', 'test')),
    actor_kind  TEXT        NOT NULL
        CONSTRAINT tenant_llm_config_audit_actor_kind_chk
        CHECK (actor_kind IN ('platform_operator', 'workspace_admin', 'service')),
    provider    TEXT,
    key_id      TEXT        NOT NULL
        CONSTRAINT tenant_llm_config_audit_key_id_chk
        CHECK (key_id = 'none' OR key_id ~ '^[0-9a-f]{12}$'),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The only read: one tenant's entries, newest first.
CREATE INDEX IF NOT EXISTS idx_tenant_llm_config_audit_tenant
    ON tenant_llm_config_audit (tenant_id, created_at DESC, id DESC);

COMMIT;
