-- Migration: 035_prompt_owner — every prompt has ONE owning tenant, enforced by the DB
-- Date: 2026-10-01
--
-- WHY. Strict tenant separation (coordinator ruling 2026-10-01, per-tenant
-- blast radius): a worker in tenant A must NEVER resolve its prompt through a
-- prompt owned by tenant B. Before this migration ``prompt_versions`` had no
-- tenant at all, and the adoption job deduped identical text ACROSS tenants
-- into one shared prompt_id — so a release made for one tenant's worker
-- changed what another tenant's worker ran. Ownership was a naming convention
-- (``<worker>.t<tag>``). It is now an explicit column with a DB guarantee.
--
-- WHAT.
--   * ``prompt_owner(prompt_id PK, owner_tenant_id)`` — one owner per prompt,
--     with UNIQUE (owner_tenant_id, prompt_id) as the composite FK target.
--   * ``prompt_versions.owner_tenant_id`` NOT NULL, FK (owner_tenant_id,
--     prompt_id) -> prompt_owner: every version of a prompt carries the SAME
--     owner (the prompt's), so a version cannot be filed under another tenant.
--   * ``worker_prompt_binding`` FK (tenant_id, prompt_id) -> prompt_owner
--     (owner_tenant_id, prompt_id). THIS is the guarantee: a binding whose
--     tenant is not the prompt's owner cannot be inserted. The resolver
--     (src/api/workers/prompt_source.py) re-checks it anyway (defense in depth).
--
-- COLUMN TYPES. ``worker_prompt_binding.tenant_id`` is TEXT (034), so
-- ``owner_tenant_id`` is TEXT too and the composite FK needs no cast and no
-- generated column. Writers store the canonical lowercase uuid text
-- (``workers.tenant_id::text`` in the adoption job; the lowercased caller
-- tenant in the repository), the same form the resolver looks bindings up by.
--
-- BACKFILL. Measured live 2026-10-01: all 60 existing prompt_versions rows were
-- written by the system (``actor_kind=migration`` and the
-- ``claude-asset-bootstrap`` SA), so every existing prompt is owned by the
-- system tenant ``aaaaaaaa-0000-4000-8000-000000000001`` (60/60 mapped, 0
-- unmapped). ``worker_prompt_binding`` had 0 rows, so its FK validates nothing
-- old. A DO block refuses the migration if any version is left without an owner.
--
-- ROLLOUT. A pre-035 pod's PUT does not set ``owner_tenant_id``. The BEFORE
-- INSERT trigger fills it from ``prompt_owner`` when the prompt already has an
-- owner (the only value the FK would accept anyway — it never INVENTS one); a
-- pre-035 PUT of a brand-new prompt has no owner row and is refused
-- (NOT NULL). New code always sets the column explicitly.
--
-- IDEMPOTENT: IF NOT EXISTS / ON CONFLICT DO NOTHING / guarded constraint adds.
-- The CI chain re-runs every migration on a database that already holds rows.
--
-- DOWN: migrations/down/035_prompt_owner.down.sql (never run by the runner).

BEGIN;

CREATE TABLE IF NOT EXISTS prompt_owner (
    prompt_id       TEXT        PRIMARY KEY,
    owner_tenant_id TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prompt_owner_owner_prompt_key UNIQUE (owner_tenant_id, prompt_id)
);

-- Every existing prompt is a system prompt (see BACKFILL above).
INSERT INTO prompt_owner (prompt_id, owner_tenant_id)
SELECT DISTINCT prompt_id, 'aaaaaaaa-0000-4000-8000-000000000001'
FROM prompt_versions
ON CONFLICT DO NOTHING;

ALTER TABLE prompt_versions ADD COLUMN IF NOT EXISTS owner_tenant_id TEXT;

UPDATE prompt_versions pv
   SET owner_tenant_id = o.owner_tenant_id
  FROM prompt_owner o
 WHERE o.prompt_id = pv.prompt_id
   AND pv.owner_tenant_id IS NULL;

DO $$
DECLARE
    orphan TEXT;
BEGIN
    SELECT pv.prompt_id INTO orphan
      FROM prompt_versions pv
      LEFT JOIN prompt_owner o ON o.prompt_id = pv.prompt_id
     WHERE o.prompt_id IS NULL OR pv.owner_tenant_id IS NULL
     LIMIT 1;
    IF orphan IS NOT NULL THEN
        RAISE EXCEPTION
            '035_prompt_owner refused: prompt_id % has a version without an owner; assign one in prompt_owner by hand first',
            orphan;
    END IF;
END
$$;

-- Rollout bridge (see ROLLOUT above): fill a missing owner from prompt_owner.
CREATE OR REPLACE FUNCTION prompt_versions_fill_owner() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.owner_tenant_id IS NULL THEN
        SELECT o.owner_tenant_id INTO NEW.owner_tenant_id
          FROM prompt_owner o
         WHERE o.prompt_id = NEW.prompt_id;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS prompt_versions_fill_owner ON prompt_versions;
CREATE TRIGGER prompt_versions_fill_owner
    BEFORE INSERT ON prompt_versions
    FOR EACH ROW EXECUTE FUNCTION prompt_versions_fill_owner();

ALTER TABLE prompt_versions ALTER COLUMN owner_tenant_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'prompt_versions_owner_fk'
           AND conrelid = 'prompt_versions'::regclass
    ) THEN
        ALTER TABLE prompt_versions
            ADD CONSTRAINT prompt_versions_owner_fk
            FOREIGN KEY (owner_tenant_id, prompt_id)
            REFERENCES prompt_owner (owner_tenant_id, prompt_id);
    END IF;
    -- THE guarantee: binding.tenant_id == the prompt's owner.
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'worker_prompt_binding_owner_fk'
           AND conrelid = 'worker_prompt_binding'::regclass
    ) THEN
        ALTER TABLE worker_prompt_binding
            ADD CONSTRAINT worker_prompt_binding_owner_fk
            FOREIGN KEY (tenant_id, prompt_id)
            REFERENCES prompt_owner (owner_tenant_id, prompt_id);
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS prompt_versions_owner_idx
    ON prompt_versions (owner_tenant_id, prompt_id);

COMMIT;
