-- DOWN for migrations/031_prompt_version_lifecycle.sql. Applied BY HAND only
-- (psql -1 -f); the startup runner lists top-level migrations/*.sql and never
-- descends into this directory.
--
-- Mapping (P4 contract §1):
--   the HIGHEST PUBLISHED version per prompt_id → active
--   everything else (DRAFT/CANARY/other PUBLISHED/ARCHIVED/DISCARDED) → retired
-- then the 030 CHECK, default and one-active partial index are restored, and
-- 031's ledger row is removed so the runner re-applies it on the next start.
--
-- ⚠️ LOSSY by construction — the 030 schema has two states. Re-applying 031
-- maps retired rows back by activated_at, so a CANARY or DISCARDED version that
-- was never PUBLISHED returns as DRAFT, and a superseded PUBLISHED version
-- (activated_at set) returns as ARCHIVED.

BEGIN;

ALTER TABLE prompt_versions DROP CONSTRAINT IF EXISTS prompt_versions_status_check;

DROP INDEX IF EXISTS prompt_versions_status_idx;

-- One statement: the subquery reads the pre-UPDATE snapshot, so the "highest
-- PUBLISHED" is decided before any row is rewritten.
-- A legacy 'active' written by a pre-031 pod during the transitional window
-- counts as PUBLISHED here, so the one-active index below cannot collide.
UPDATE prompt_versions p
   SET status = CASE
           WHEN p.status IN ('PUBLISHED', 'active')
            AND p.version = (
                SELECT max(q.version) FROM prompt_versions q
                 WHERE q.prompt_id = p.prompt_id AND q.status IN ('PUBLISHED', 'active')
            )
           THEN 'active'
           ELSE 'retired'
       END
 WHERE p.status IN ('DRAFT', 'CANARY', 'PUBLISHED', 'ARCHIVED', 'DISCARDED', 'active', 'retired');

ALTER TABLE prompt_versions ALTER COLUMN status SET DEFAULT 'retired';

ALTER TABLE prompt_versions
    ADD CONSTRAINT prompt_versions_status_check CHECK (status IN ('active', 'retired'));

CREATE UNIQUE INDEX IF NOT EXISTS prompt_versions_one_active_idx
    ON prompt_versions (prompt_id)
 WHERE status = 'active';

DELETE FROM schema_migration_files WHERE filename = '031_prompt_version_lifecycle.sql';

COMMIT;
