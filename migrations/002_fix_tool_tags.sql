-- Backfill NULL tool tags to the empty value before asserting NOT NULL.

-- ⚠️ DO NOT "fix" '{}' back to '[]'. It was '[]' from this file's creation
-- until 2026-08-09 and had therefore NEVER ONCE SUCCEEDED in production.
--
-- tools.tags has drifted apart between environments:
--   * production            -> text[]  NOT NULL DEFAULT '{}'::text[]
--   * postgres_schema.sql   -> JSONB   NOT NULL   (what CI/test bootstraps)
-- The column was originally JSON and was migrated to text[] in production;
-- this line was never updated to match, so against the real database it
-- raised `ERROR: malformed array literal: "[]"` on EVERY pod boot ('[' is
-- not a valid array literal). That is why `schema_migrations` contains only
-- 006 and 007 from the 00x range -- 002 never recorded itself, so it re-ran
-- and re-failed forever.
--
-- Postgres parses the literal against the column type at PLAN time, so this
-- failed even though the NOT NULL column means the WHERE matches ZERO rows.
-- A zero-row UPDATE is not a no-op if the literal will not cast.
--
-- '{}' is the ONLY literal that parses under BOTH live types:
--   '{}'::text[] -> empty array   |  '[]'::text[] -> ERROR
--   '{}'::jsonb  -> empty object  |  '[]'::jsonb  -> empty array
-- The empty-object/empty-array difference on the JSONB side is inert: the
-- column is NOT NULL in both schemas, so this statement updates zero rows.
-- It exists only to protect a legacy database that still has NULLs.
UPDATE tools SET tags = '{}' WHERE tags IS NULL;

-- Ensure the tags column is NOT NULL
-- This is redundant but ensures consistency
ALTER TABLE tools ALTER COLUMN tags SET NOT NULL;

-- Tool-name uniqueness was historically scoped on (name, creator_id).
--
-- Idempotent + forward-compatible: this block is a no-op once creator_id has
-- been dropped service-wide (migration 019). The guard checks for the column
-- it references; if creator_id is gone we skip entirely, so we neither fail on
-- the absent reference nor re-add a constraint that 019's CASCADE removes. The
-- canonical guarantee post-019 is unique_tenant_tool_name(tenant_id, name),
-- created by the base schema / migration 019.
-- (DDL must go through EXECUTE inside PL/pgSQL.)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tools' AND column_name = 'creator_id'
    ) THEN
        RAISE NOTICE 'migration 002 skipped tool unique rebuild: creator_id already removed (post-019)';
        RETURN;
    END IF;

    EXECUTE 'ALTER TABLE tools DROP CONSTRAINT IF EXISTS unique_tool_name';
    EXECUTE 'ALTER TABLE tools ADD CONSTRAINT unique_tool_name_creator UNIQUE (name, creator_id)';
END $$;
