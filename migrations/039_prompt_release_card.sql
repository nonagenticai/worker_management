-- Migration: 039_prompt_release_card — every prompt release carries its board card
-- Date: 2026-10-03
--
-- NUMBERING. 032 stays RESERVED for 032_prompt_version_lifecycle_tighten.sql
-- (see 033's header). 039 is the next free number: origin/main ends at 038 and
-- no open PR or remote branch adds a migration (checked 2026-10-03).
--
-- WHY (coordinator ruling, REALIGN R7, 2026-10-03). Every prompt release goes
-- through the board: a transition INTO CANARY or PUBLISHED must name the Planka
-- card that authorised it. The ledger (033) recorded who and why, not which
-- card; this column is that record, and the CHECK below makes it impossible to
-- append a release row without one.
--
--   actor_card_id  — the Planka card id (a 19-digit decimal string), supplied
--                    by the caller of POST …/status or POST …/activate. The
--                    route refuses a release without one (422
--                    prompt_release_card_required) before any write; this
--                    CHECK is the backstop for any other writer.
--
-- EXEMPTION: actor_kind = 'migration' only. Those rows are written in-process
-- by this service's own one-time jobs (the 033 backfill,
-- scripts/adopt_live_worker_prompts.py's ``adopt_live``), never by an HTTP
-- caller — src/api/prompts/actor.py cannot produce 'migration' from a request.
-- There is no card for "adopt the text that is already live", so those rows
-- keep actor_card_id NULL rather than a fabricated sentinel; actor_kind and
-- reason already say what they are.
--
-- NOT VALID on the release CHECK: rows written before this migration (activate
-- / initial_create releases) have no card and are history — the ledger is
-- append-only, so they cannot be "fixed", and inventing a card would be the
-- fabrication the ledger exists to prevent. NOT VALID still enforces the CHECK
-- on every new row. ⛔ Never VALIDATE it.
--
-- IDEMPOTENT: ADD COLUMN IF NOT EXISTS; DROP CONSTRAINT IF EXISTS + ADD.
-- DOWN: migrations/down/039_prompt_release_card.down.sql (never run by the runner).

BEGIN;

ALTER TABLE prompt_version_transitions
    ADD COLUMN IF NOT EXISTS actor_card_id TEXT;

ALTER TABLE prompt_version_transitions
    DROP CONSTRAINT IF EXISTS prompt_version_transitions_actor_card_id_format_check;
ALTER TABLE prompt_version_transitions
    ADD CONSTRAINT prompt_version_transitions_actor_card_id_format_check
    CHECK (actor_card_id IS NULL OR actor_card_id ~ '^[0-9]{19}$');

ALTER TABLE prompt_version_transitions
    DROP CONSTRAINT IF EXISTS prompt_version_transitions_release_card_check;
ALTER TABLE prompt_version_transitions
    ADD CONSTRAINT prompt_version_transitions_release_card_check
    CHECK (to_status NOT IN ('CANARY', 'PUBLISHED')
           OR actor_card_id IS NOT NULL
           OR actor_kind = 'migration')
    NOT VALID;

COMMIT;
