-- Migration: 030_prompt_versions — the prompt version ledger (conveyor A5 / S2)
-- Date: 2026-09-23
--
-- WHY. A worker call today cannot be attributed to a prompt. `workers.
-- config_revision` (migration 016) is free text, `swap_worker_prompt` writes a
-- canary NAME into it, and the improvement-loop ETL does `parseInt(config_
-- revision)` — which is `NaN` for every value that is not a bare integer, so it
-- falls back to asset_version 1. Measured on prod 2026-09-22: 2 of 793 workers
-- carry a `config_revision` at all and NEITHER is a `<prompt_id>@<version>`
-- stamp, and all 7,033 `exec_quality` rows are `legacy_default`. The stamping
-- path (n8n_custom_nodes #1603, roadmap A2) is MERGED and has produced ZERO
-- real observations, because the server half — a place to publish a version and
-- a response that says which version actually ran — does not exist.
--
-- This file is that server half's storage. Specified by
-- `docs/redesign/design/improvement-design.md` §2.3 in `n8n_custom_nodes`
-- (roadmap §10 item A5).
--
-- THE TWO DIGESTS, and why one is not enough.
--
--   sha256                -- over the whole version payload (question, system,
--                            template, output_schema, worker_names,
--                            purpose_tags). This is the IDENTITY of the version:
--                            the publisher's "same sha ⇒ no write" and the
--                            immutability refusal both key on it. It must cover
--                            `template` and `output_schema`, or a template-only
--                            edit would read as "unchanged" and silently never
--                            publish.
--   system_prompt_sha256  -- over the `system` text ALONE, exactly the bytes a
--                            worker's `system_prompt` is set to. This is the
--                            JOIN KEY with what a live call reports:
--                            `MessageResponse.metadata.system_prompt_sha256` is
--                            the digest of the prompt the agent instance
--                            actually ran with. A consumer can therefore ask
--                            "did this call really use prompt_id@version?" and
--                            get a yes/no rather than a stamp's assertion.
--
-- A single digest cannot do both jobs: one covering the whole payload never
-- equals the digest of a live prompt body, and one covering only `system`
-- cannot detect a template change. `git_sha` is deliberately NOT in either
-- digest — it is provenance, so re-publishing byte-identical content from a
-- later commit is `unchanged`, not a conflict.
--
-- IMMUTABILITY. `(prompt_id, version)` is the primary key and the API refuses
-- (409) any PUT that would change the `sha256` of an existing row. You bump the
-- version; you never rewrite one. That is what makes replay — reading an OLD
-- version to reproduce a past call — mean anything. There is no DELETE route.
--
-- STATUS. A row is born `retired` (published and readable, but not the one a
-- caller gets from `GET /prompts/{id}`) and becomes `active` only through an
-- explicit `POST …/activate`. Publishing is not activating — the two steps are
-- separate on purpose (memory: publishing-to-the-registry-is-arming-production).
-- `prompt_versions_one_active_idx` makes "at most one active version per
-- prompt" an invariant the database enforces, not a convention the writer
-- remembers.
--
-- NO TENANT COLUMN. Prompts are fleet assets sourced from git
-- (`AA_IMPLEMENTATION/infer/**.json`), not tenant data — the same shape as
-- `workflow_routing_pointer` (migration 014), which is also global. Tenant
-- scoping here would mean 60+ identical copies of one git file.
--
-- Forward-only / additive. Creates one new table; touches nothing existing.
-- Idempotent (`IF NOT EXISTS` throughout), as the startup runner requires.

BEGIN;

CREATE TABLE IF NOT EXISTS prompt_versions (
    -- Registry key from AA_IMPLEMENTATION/infer/<purpose>.json. API-validated
    -- to [A-Za-z0-9][A-Za-z0-9._-]{0,127}; kept as text here so the ledger
    -- never refuses a row the API already accepted.
    prompt_id            TEXT        NOT NULL,
    version              INTEGER     NOT NULL,
    -- Identity of the whole version payload (see header). char(64) = hex sha256.
    sha256               CHAR(64)    NOT NULL,
    -- Digest of the `system` text alone — the join key with a live call.
    system_prompt_sha256 CHAR(64)    NOT NULL,
    -- Short human description; the retrieval shortlist shows this.
    question             TEXT,
    -- The system prompt body itself. The publisher converges
    -- workers.config.system_prompt to exactly these bytes.
    system               TEXT        NOT NULL,
    template             TEXT,
    output_schema        JSONB,
    worker_names         TEXT[]      NOT NULL DEFAULT '{}',
    purpose_tags         TEXT[]      NOT NULL DEFAULT '{}',
    -- Commit that FIRST published this (id, version). Not re-written on an
    -- idempotent re-publish, because that would mutate a published row.
    git_sha              TEXT,
    status               TEXT        NOT NULL DEFAULT 'retired',
    published_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_at         TIMESTAMPTZ,
    PRIMARY KEY (prompt_id, version),
    CONSTRAINT prompt_versions_status_check CHECK (status IN ('active', 'retired')),
    CONSTRAINT prompt_versions_version_positive CHECK (version > 0)
);

-- At most ONE active version per prompt. The activate route retires the
-- previous one inside the same transaction; this index is what makes a bug in
-- that route an error instead of two live "active" prompts.
CREATE UNIQUE INDEX IF NOT EXISTS prompt_versions_one_active_idx
    ON prompt_versions (prompt_id)
 WHERE status = 'active';

-- Free-text search over question + tags + the prompt body, per
-- improvement-design §2.3 ("plus a tsvector over question, system and tags").
--
-- A STORED generated column rather than a trigger: it cannot drift out of sync
-- with the row the way a trigger can when someone writes around it. Weights let
-- `question` and the tags outrank an incidental word deep inside a 4 kB prompt
-- body.
--
-- ⚠️ EVERY function in a generated expression must be IMMUTABLE, and the
-- obvious spelling is NOT. `array_to_string(purpose_tags, ' ')` is STABLE
-- (`pg_proc.provolatile = 's'`, because element output functions may be), so
-- `to_tsvector('english', array_to_string(...))` fails the whole migration with
-- "generation expression is not immutable". `array_to_tsvector(text[])` IS
-- immutable and is the right tool anyway: it makes each tag one exact lexeme
-- instead of stemming it. Likewise `to_tsvector` is immutable ONLY in its
-- two-argument `(regconfig, text)` form — dropping the 'english' literal makes
-- it stable and breaks this the same way.
ALTER TABLE prompt_versions
    ADD COLUMN IF NOT EXISTS search_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(question, '')), 'A')
        || setweight(array_to_tsvector(purpose_tags), 'B')
        || setweight(to_tsvector('english', coalesce(prompt_id, '')), 'B')
        || setweight(to_tsvector('english', coalesce(system, '')), 'C')
    ) STORED;

CREATE INDEX IF NOT EXISTS prompt_versions_search_idx
    ON prompt_versions USING GIN (search_tsv);

-- `?tag=` filtering on the search route.
CREATE INDEX IF NOT EXISTS prompt_versions_tags_idx
    ON prompt_versions USING GIN (purpose_tags);

-- "which prompt versions name this worker" — the publisher's convergence step
-- and any audit of a live worker's stamp.
CREATE INDEX IF NOT EXISTS prompt_versions_worker_names_idx
    ON prompt_versions USING GIN (worker_names);

COMMIT;
