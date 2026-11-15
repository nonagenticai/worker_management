-- Migration: 033_prompt_version_transitions — append-only lifecycle ledger
-- Date: 2026-09-28
--
-- NUMBERING. 032 is RESERVED for 032_prompt_version_lifecycle_tighten.sql (named
-- by 031's header and by src/api/prompts/repository.py): it lands once no
-- pre-031 pod can be running. This file takes 033 and leaves 032 free. The
-- runner applies files in filename order and skips recorded ones, so a 032
-- landing AFTER 033 is still applied on the next start.
--
-- WHY. prompt_versions holds only the CURRENT status of a version. Who moved it,
-- from what, when and why is lost on every UPDATE — so an audit cannot tell a
-- version promoted through CANARY from one a Claude session created PUBLISHED
-- (the sanctioned `initial_create`, P4 owner ruling 2026-09-28), and the
-- improvement statistics cannot separate the two either. This table is that
-- record: one row per status change, written in the SAME transaction as the
-- change (src/api/prompts/repository.py), by every write path:
--   PUT create           NULL      -> DRAFT
--   POST …/status        matrix move, or DRAFT -> PUBLISHED with reason
--                        'initial_create'
--   POST …/activate      target -> PUBLISHED, and each other PUBLISHED -> ARCHIVED
--
-- actor_sub / actor_kind come from the VERIFIED auth principal only
-- (src/api/prompts/actor.py), never from a request header or body:
--   service_account  — azp of a signature-verified X-Caller-Authorization SA JWT
--   licence_subject  — subject_id returned by licence introspection
--   unverified       — no verified principal (tokenless / bypass); actor_sub NULL
--   migration        — the backfill below
--
-- APPEND-ONLY, enforced: a row trigger refuses every UPDATE and every DELETE,
-- and a statement trigger refuses every TRUNCATE (row triggers never see a
-- TRUNCATE, including one cascaded from `TRUNCATE prompt_versions CASCADE`).
-- There is NO opt-out: no GUC, no session flag. Any client can set a custom
-- GUC, so a GUC opt-out is a bypass for every client. The only way past the
-- triggers is `ALTER TABLE … DISABLE TRIGGER USER`, which needs table
-- ownership — a power that can already DROP the table, so it adds no bypass
-- (test teardown uses exactly that, inside its own transaction). The FK is ON
-- DELETE RESTRICT, so a prompt_versions row with history cannot vanish from
-- under its ledger either.
--
-- BACKFILL. Every existing prompt_versions row gets its creation row
-- (NULL -> DRAFT at published_at, actor_kind 'migration'). That fact is true of
-- every row ever written (030 inserted 'retired'+never activated = DRAFT; 031+
-- inserts DRAFT). Later moves of pre-033 rows are NOT reconstructed — nothing
-- recorded them, and inventing them would be the fabrication this table exists
-- to prevent. Prod 2026-09-28: 41 rows, all DRAFT.
--
-- IDEMPOTENT: IF NOT EXISTS everywhere, CREATE OR REPLACE for the function
-- (so a re-run also replaces an earlier body, e.g. one that had an opt-out),
-- DROP TRIGGER IF EXISTS before CREATE, and the backfill skips rows that already
-- have a creation row.
--
-- DOWN: migrations/down/033_prompt_version_transitions.down.sql (never run by
-- the runner).

BEGIN;

CREATE TABLE IF NOT EXISTS prompt_version_transitions (
    id          BIGSERIAL PRIMARY KEY,
    prompt_id   TEXT        NOT NULL,
    version     INTEGER     NOT NULL,
    from_status TEXT,
    to_status   TEXT        NOT NULL,
    reason      TEXT,
    actor_sub   TEXT,
    actor_kind  TEXT        NOT NULL,
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prompt_version_transitions_version_fk
        FOREIGN KEY (prompt_id, version)
        REFERENCES prompt_versions (prompt_id, version) ON DELETE RESTRICT,
    CONSTRAINT prompt_version_transitions_from_status_check
        CHECK (from_status IS NULL
               OR from_status IN ('DRAFT', 'CANARY', 'PUBLISHED', 'ARCHIVED', 'DISCARDED')),
    CONSTRAINT prompt_version_transitions_to_status_check
        CHECK (to_status IN ('DRAFT', 'CANARY', 'PUBLISHED', 'ARCHIVED', 'DISCARDED')),
    CONSTRAINT prompt_version_transitions_actor_kind_check
        CHECK (actor_kind IN ('service_account', 'licence_subject', 'unverified', 'migration')),
    CONSTRAINT prompt_version_transitions_reason_len_check
        CHECK (reason IS NULL OR char_length(reason) <= 200),
    CONSTRAINT prompt_version_transitions_actor_sub_len_check
        CHECK (actor_sub IS NULL OR char_length(actor_sub) <= 255)
);

CREATE INDEX IF NOT EXISTS prompt_version_transitions_version_at_idx
    ON prompt_version_transitions (prompt_id, version, at);

CREATE OR REPLACE FUNCTION prompt_version_transitions_append_only()
RETURNS trigger
LANGUAGE plpgsql
AS $fn$
BEGIN
    -- No opt-out (see header): every UPDATE, DELETE and TRUNCATE is refused.
    RAISE EXCEPTION 'prompt_version_transitions is append-only (% refused)', TG_OP
        USING ERRCODE = 'insufficient_privilege';
END
$fn$;

DROP TRIGGER IF EXISTS prompt_version_transitions_append_only_trg ON prompt_version_transitions;
CREATE TRIGGER prompt_version_transitions_append_only_trg
    BEFORE UPDATE OR DELETE ON prompt_version_transitions
    FOR EACH ROW EXECUTE FUNCTION prompt_version_transitions_append_only();

DROP TRIGGER IF EXISTS prompt_version_transitions_no_truncate_trg ON prompt_version_transitions;
CREATE TRIGGER prompt_version_transitions_no_truncate_trg
    BEFORE TRUNCATE ON prompt_version_transitions
    FOR EACH STATEMENT EXECUTE FUNCTION prompt_version_transitions_append_only();

INSERT INTO prompt_version_transitions
    (prompt_id, version, from_status, to_status, reason, actor_sub, actor_kind, at)
SELECT p.prompt_id, p.version, NULL, 'DRAFT', 'backfill_033_creation', NULL, 'migration',
       p.published_at
  FROM prompt_versions p
 WHERE NOT EXISTS (
        SELECT 1 FROM prompt_version_transitions t
         WHERE t.prompt_id = p.prompt_id AND t.version = p.version
           AND t.from_status IS NULL
       );

COMMIT;
