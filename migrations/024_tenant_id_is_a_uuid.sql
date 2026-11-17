-- 024_tenant_id_is_a_uuid.sql
--
-- `tenant_id` becomes a real Postgres `uuid` column on all 12 tenant-scoped
-- tables. `tenant_name` is deliberately left untouched and keeps the human
-- label -- see step 2 for why it needs no repair.
--
-- WHY THIS IS A DEFECT AND NOT A SHAPE PREFERENCE
-- ----------------------------------------------
-- Measured on production 2026-08-25: `worker_states` holds 952,817 rows, of
-- which 93,842 carry a tenant_id that is not a UUID at all -- 16 distinct
-- literals, dominated by the bare string `n8n-dev` (88,092 rows). A VARCHAR
-- column cannot tell `n8n-dev` from a uuid, so those rows are invisible to
-- every `WHERE tenant_id = <uuid-from-the-JWT>` the service issues. This is
-- the largest single concentration of the defect in the fleet: 187,920 of the
-- 318,643 non-uuid tenant rows measured across all 47 databases are here.
--
-- WHAT THIS MIGRATION DELIBERATELY DOES NOT DO
-- --------------------------------------------
-- It does NOT move any row between tenants. Every legacy literal is mapped
-- 1:1 to a deterministic UUIDv5 of its own label, so the isolation boundary
-- that existed before this migration exists after it, unchanged -- the values
-- merely stop being untyped.
--
-- That restraint is deliberate, and it is the whole reason this migration is
-- safe to run unattended. The tempting move is to fold `n8n-dev` into an
-- existing registered tenant. There are two candidates and they are in open
-- conflict:
--
--   aaaaaaaa-0000-4000-8000-000000000001  -- what Keycloak's portal realm
--       actually mints as the `tenant_id` claim for n8n-dev
--       (gitops-production clusters/strato/services/keycloak/
--        realm-patch-script-configmap.yaml:111-113), and this deployment's
--       own DEFAULT_TENANT_ID. 553,550 rows here already.
--   aaaaaaaa-0000-4000-8000-0000000000a5  -- the row literally named
--       `n8n-dev` in `licenses.public.tenants`. 64,822 rows here already.
--
-- The cluster has already written the collision down, in that same configmap
-- at lines 146-169: the licences-side `n8n-dev` "is UNREACHABLE, because no
-- JWT ever carries that uuid", and it ends with an explicit
-- "do NOT 'fix' this by renumbering", because the values are pinned into live
-- Keycloak user attributes and changing one moves real users between tenants.
--
-- On top of that, MEASURED 2026-08-25, neither merge is even clean: folding
-- `n8n-dev` into the first candidate collides on 67 rows of
-- `unique_tenant_worker_name`, into the second on 6. A merge therefore needs a
-- human to choose which duplicate worker survives. That is a tenant-ownership
-- decision with an irreversible edge, and it is not a type migration's to make.
--
-- So: the crosswalk below is the single place where such a merge would ever be
-- expressed. To fold a legacy label into a real tenant later, UPDATE its one
-- row in `tenant_id_crosswalk`, resolve the duplicate worker names, and re-run
-- the same `UPDATE ... FROM` this file already contains. Nothing else changes.
--
-- WHY UUIDv5 AND NOT UUIDv7
-- -------------------------
-- A v7 encodes a mint time, and there is no honest mint time for a value that
-- was already sitting in the table. v5 is a pure function of the label, so it
-- is reproducible and reversible, and -- the load-bearing part -- ANY OTHER
-- SERVICE deriving an id for the same legacy label lands on the same uuid.
-- The namespace below is byte-for-byte the one requirements_management used in
-- its own flip (2026_08_24_0042), so the fleet stays joinable instead of
-- fragmenting one label into one uuid per service. A genuine v7 can only come
-- from registering the tenant for real; that is an owner decision, not this
-- file's.
--
-- IDEMPOTENCY IS NOT OPTIONAL HERE
-- -------------------------------
-- `run_migrations()` (src/db/db_config.py:381) re-runs EVERY file on EVERY pod
-- boot -- `schema_migrations` is bookkeeping, not a gate ("the source of truth
-- ... is the database itself", db_config.py:348-349). CI proves it:
-- tests/integration/test_migration_chain.py runs the whole chain twice under
-- strict=True and asserts zero failures both times. Every step below is
-- therefore guarded on the CURRENT column type, read from information_schema,
-- so a second run is a no-op rather than an error.
--
-- AND THE RUNNER STEPS OVER FAILURES
-- ----------------------------------
-- In production `WORKER_MGMT_MIGRATIONS_STRICT` is unset, so a failing file is
-- logged and skipped (db_config.py:433-437). That is deliberate -- it is what
-- stops a lock-wait convoy from consuming a boot (the 2026-08-08 incident,
-- db_config.py:466-484) -- so it is not being changed. It does mean "the flip
-- happened" and "the flip silently did not happen" would otherwise look
-- identical in a green boot. The final DO block raises with the offending
-- table names so a stall is a loud, repeated, actionable log line.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Deliberately NOT called `tenant_id`: this is an operational lookup that must
-- stay readable across tenants, and a column named `tenant_id` would enrol the
-- table in every tenant sweep -- including this migration's own.
CREATE TABLE IF NOT EXISTS tenant_id_crosswalk (
    legacy_value VARCHAR(255)             PRIMARY KEY,
    tenant_uuid  uuid                     NOT NULL,
    source       VARCHAR(32)              NOT NULL,
    -- The registered tenant this legacy label NAMES, when it names one. NOT
    -- applied: folding the rows into it is a lossy, per-row human decision
    -- (measured 2026-08-25: of the 67 colliding `(tenant_id, worker_name)`
    -- pairs only 17 are effectively identical; the other 50 carry different
    -- system_prompt / model / model_provider / output_schema, and
    -- `principle_consistency`'s two sides declare incompatible output schemas).
    -- It is recorded so the decision has its answer in one place: to merge,
    -- `UPDATE tenant_id_crosswalk SET tenant_uuid = merge_target WHERE ...`,
    -- resolve the duplicate worker names, and re-run this file.
    merge_target uuid,
    note         TEXT,
    created_at   TIMESTAMPTZ              NOT NULL DEFAULT now()
);

DO $tenant_uuid_flip$
DECLARE
    -- Same namespace as requirements_management's 2026_08_24_0042. Never change
    -- it: it is what makes two services agree on the uuid for one legacy label.
    ns              CONSTANT uuid := '6ba7b811-9dad-11d1-80b4-00c04fd430c8';

    -- The 12 tenant-scoped tables. `tenant_id_crosswalk` is absent on purpose.
    all_tables      CONSTANT text[] := ARRAY[
        'workers', 'worker_states', 'graphs', 'graph_states',
        'graph_execution_sessions', 'node_execution_logs', 'execution_events',
        'tools', 'workload', 'skills',
        'tenant_llm_config', 'tenant_modality_config'
    ];

    pending         text[] := ARRAY[]::text[];   -- still VARCHAR, this run
    tbl             text;
    remapped        bigint;
    collisions      bigint;
    conflict        record;
    still_text      text[];
BEGIN
    ------------------------------------------------------------------
    -- 0. Which tables still need the flip?
    --    Read the live type: on a re-run (or a fresh DB built from
    --    postgres_schema.sql, which ships `uuid` from this commit on) the
    --    answer is "none" and the whole block falls through.
    ------------------------------------------------------------------
    FOREACH tbl IN ARRAY all_tables LOOP
        -- Qualified with current_schema(): a bare to_regclass() resolves through
        -- the WHOLE search_path, while the type check below and the postcondition
        -- are both pinned to current_schema(). With search_path = 'app, public'
        -- and the tables in public, the unqualified form found them, the pinned
        -- queries did not, and the file reported 'nothing to do' while leaving
        -- every column VARCHAR -- with the postcondition, whose whole job is to
        -- make that loud, silenced by the same mismatch. Found by an adversarial
        -- agent; latent today (nothing sets a search_path) and wrong regardless.
        IF to_regclass(current_schema() || '.' || quote_ident(tbl)) IS NULL THEN
            CONTINUE;                                   -- table not created yet
        END IF;
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name   = tbl
              AND column_name  = 'tenant_id'
              AND data_type   <> 'uuid'
        ) THEN
            pending := pending || tbl;
        END IF;
    END LOOP;

    IF cardinality(pending) = 0 THEN
        RAISE NOTICE '024: tenant_id is already uuid on every table; nothing to do';
        RETURN;
    END IF;

    ------------------------------------------------------------------
    -- 1. Record the mapping for every legacy value BEFORE touching a
    --    single row, so the repair is auditable and reversible even if a
    --    later step fails and the transaction is retried on the next boot.
    ------------------------------------------------------------------
    FOREACH tbl IN ARRAY pending LOOP
        -- ⛔ CLASSIFY BY WHETHER IT CASTS, NOT BY A REGEX. An earlier draft used
        --    ^[0-9a-fA-F]{8}-...{12}$ and two adversarial reproductions killed it:
        --
        --    * `AAAAAAAA-D000-...` PASSES that regex, so it was treated as "already
        --      a uuid" and left alone -- and then `::uuid` normalised it to
        --      lowercase during the ALTER, colliding with the lowercase row under
        --      `unique_tenant_worker_name`. The ALTER died inside the index rebuild,
        --      and because prod is non-strict the file was stepped over on EVERY
        --      boot: the flip would never have happened, quietly.
        --    * `aaaaaaaad0007000...` (no dashes) and `{aaaaaaaa-...}` (braces) FAIL
        --      the regex but cast fine -- so they were handed a fabricated v5 and
        --      SPLIT OFF from the tenant they are literally spelling. That is the
        --      exact property this migration's header promises it preserves.
        --
        --    `pg_input_is_valid` (PG16+; prod is 18.4, CI is postgres:18) is the
        --    honest predicate: it answers the only question that matters, which is
        --    what the ALTER's own cast will do.
        EXECUTE format(
            'INSERT INTO tenant_id_crosswalk (legacy_value, tenant_uuid, source, note)
             SELECT DISTINCT t.tenant_id,
                    uuid_generate_v5($1, t.tenant_id),
                    ''derived'',
                    ''no registered tenant resolves this value; deterministic UUIDv5 ''
                    ''of the label, 1:1 so the isolation boundary is unchanged''
             FROM %I t
             WHERE t.tenant_id IS NOT NULL
               AND NOT pg_input_is_valid(t.tenant_id, ''uuid'')
             ON CONFLICT (legacy_value) DO NOTHING', tbl)
        USING ns;
    END LOOP;

    -- Record, for the labels the portal realm actually mints, which registered
    -- tenant they name. Copied from
    -- gitops-production/.../keycloak/realm-patch-script-configmap.yaml:111-146.
    UPDATE tenant_id_crosswalk c
       SET merge_target = m.uuid_value
      FROM (VALUES
              ('n8n-dev',       'aaaaaaaa-0000-4000-8000-000000000001'::uuid),
              ('george-dev',    'aaaaaaaa-0000-4000-8000-000000000003'::uuid),
              ('fr-test',       'aaaaaaaa-0000-4000-8000-000000000099'::uuid),
              ('cdmsfirst',     'aaaaaaaa-0000-4000-8000-000000000100'::uuid),
              ('ahtapol',       'aaaaaaaa-0000-4000-8000-000000000101'::uuid),
              ('ahtapol-probe', 'aaaaaaaa-0000-4000-8000-000000000102'::uuid)
           ) AS m(label, uuid_value)
     WHERE c.legacy_value = m.label AND c.merge_target IS DISTINCT FROM m.uuid_value;

    ------------------------------------------------------------------
    -- 1b. Pre-flight: would the cast itself create a duplicate?
    --
    --     A value that is a VALID uuid written non-canonically -- `AAAAAAAA-…`,
    --     `aaaaaaaad000…` with no dashes, `{aaaaaaaa-…}` -- is normalised by
    --     `::uuid`. If another row already holds the canonical spelling under
    --     the same tenant-scoped unique key, the ALTER dies INSIDE the index
    --     rebuild with `could not create unique index`, which names neither the
    --     rows nor what to do. Production runs the migration runner non-strict,
    --     so that error is logged and stepped over on EVERY boot: the flip would
    --     silently never happen. An adversarial agent reproduced exactly that.
    --
    --     This does NOT change which uuid any row ends up with -- the ALTER's
    --     own cast does the normalising either way. It changes the FAILURE from
    --     an opaque index error into a named, actionable one, raised before a
    --     single table has been altered. Measured 2026-08-25: production has
    --     zero non-canonical values on all 12 tables, so this is a guard, not a
    --     repair.
    --
    --     The keys are read from `pg_constraint` rather than hardcoded, so a
    --     unique constraint added later is covered without editing this file.
    ------------------------------------------------------------------
    FOR conflict IN
        SELECT c.conrelid::regclass::text AS tbl,
               c.conname                  AS constraint_name,
               string_agg(quote_ident(a.attname), ', ' ORDER BY k.ord) AS cols
          FROM pg_constraint c
          JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
          JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
         WHERE c.contype IN ('u', 'p')
           AND c.connamespace = current_schema()::regnamespace
           AND c.conrelid::regclass::text = ANY(pending)
           AND EXISTS (SELECT 1 FROM pg_attribute a2
                        WHERE a2.attrelid = c.conrelid
                          AND a2.attnum = ANY(c.conkey)
                          AND a2.attname = 'tenant_id')
         GROUP BY c.conrelid, c.conname
    LOOP
        EXECUTE format(
            'SELECT count(*) FROM (SELECT 1 FROM %I
               WHERE pg_input_is_valid(tenant_id, ''uuid'')
               GROUP BY tenant_id::uuid, %s
              HAVING count(*) > 1) z',
            conflict.tbl,
            replace(conflict.cols, 'tenant_id, ', ''))
        INTO collisions;

        IF collisions > 0 THEN
            RAISE EXCEPTION
                '024 cannot flip %: % row group(s) hold ONE uuid under two spellings '
                'and would violate %(%). Canonicalise them first '
                '(UPDATE % SET tenant_id = tenant_id::uuid::text WHERE '
                'tenant_id <> tenant_id::uuid::text), decide which duplicate survives, '
                'then re-run.',
                conflict.tbl, collisions, conflict.constraint_name, conflict.cols,
                conflict.tbl;
        END IF;
    END LOOP;

    ------------------------------------------------------------------
    -- 2. Remap `tenant_id`, and DO NOT touch `tenant_name`.
    --
    --    An earlier draft of this file also "restored" the legacy label into
    --    `tenant_name` with
    --        SET tenant_name = CASE WHEN t.tenant_name = t.tenant_id
    --                               THEN x.legacy_value ELSE t.tenant_name END
    --    A mutation test killed it: deleting that clause changed no row, in any
    --    case. It cannot -- the THEN arm only fires when `tenant_name` already
    --    EQUALS `tenant_id`, and it then assigns the value it already holds,
    --    while a row whose `tenant_id` is already a uuid never matches the join
    --    at all. It was code that read as though it did something.
    --
    --    The label survives anyway, and this is why: migration 011 backfilled
    --    `tenant_name = tenant_id`, and measured 2026-08-25 it is STILL a
    --    byte-identical copy on all 952,808 rows (same=952808, differ=0). So
    --    every row whose tenant_id is a legacy label already carries that same
    --    label in `tenant_name`; flipping `tenant_id` to a uuid leaves the
    --    readable name exactly where it was. `n8n-dev` is not lost -- it stops
    --    being an identifier and becomes only a name, which is what 011 built
    --    the column for. The test asserts that postcondition directly.
    --
    --    The write paths that used to persist `tenant_name or tenant_id` are
    --    fixed in the same change (src/api/tenant_constants.py::tenant_label);
    --    without that, new rows would refill the label column with uuids and
    --    this property would decay.
    ------------------------------------------------------------------
    FOREACH tbl IN ARRAY pending LOOP
        EXECUTE format(
            'UPDATE %I t
                SET tenant_id = x.tenant_uuid::text
               FROM tenant_id_crosswalk x
              WHERE t.tenant_id = x.legacy_value', tbl);

        GET DIAGNOSTICS remapped = ROW_COUNT;
        IF remapped > 0 THEN
            RAISE NOTICE '024: remapped % row(s) in %', remapped, tbl;
        END IF;
    END LOOP;

    ------------------------------------------------------------------
    -- 3. The type flip.
    --
    --    There is NO row level security in this repo (verified: zero
    --    `CREATE POLICY` / `ROW LEVEL SECURITY` across all migrations,
    --    postgres_schema.sql and src/), so unlike the sibling services there
    --    are no policies to drop and recreate -- a policy referencing the
    --    column would otherwise make ALTER TYPE fail outright. Indexes and
    --    unique constraints survive: Postgres rebuilds them.
    --
    --    `worker_states` is 2.8 GB / 953k rows, so its ALTER is a full table
    --    rewrite under ACCESS EXCLUSIVE. run_migrations() sets a session
    --    lock_timeout of 10s to stop a boot being consumed by a lock convoy;
    --    that is a fine budget for eleven small tables and a poor one for
    --    acquiring an exclusive lock on the busiest table in the service while
    --    the outgoing pod is still serving it. SET LOCAL raises it for this
    --    transaction only -- deliberately LOCAL, and deliberately not an
    --    unbounded wait: if it cannot be had in 90s the file fails, is logged,
    --    and is retried on the next boot with the rest of the schema intact.
    ------------------------------------------------------------------
    SET LOCAL lock_timeout = '90s';

    FOREACH tbl IN ARRAY pending LOOP
        EXECUTE format(
            'ALTER TABLE %I ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid', tbl);
        RAISE NOTICE '024: % .tenant_id is now uuid', tbl;
    END LOOP;

    ------------------------------------------------------------------
    -- 4. Postcondition. The runner is non-strict in production, so without
    --    this a failed flip is indistinguishable from a successful one in a
    --    green boot. Naming the survivors makes a stall loud and actionable.
    ------------------------------------------------------------------
    SELECT array_agg(table_name ORDER BY table_name) INTO still_text
      FROM information_schema.columns
     WHERE table_schema = current_schema()
       AND column_name  = 'tenant_id'
       AND data_type   <> 'uuid'
       AND table_name   = ANY(all_tables);

    IF still_text IS NOT NULL THEN
        RAISE EXCEPTION
            '024 postcondition failed: tenant_id is still not uuid on %', still_text;
    END IF;
END
$tenant_uuid_flip$;
