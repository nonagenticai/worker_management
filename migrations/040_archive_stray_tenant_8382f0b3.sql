-- Migration: 040_archive_stray_tenant_8382f0b3 — archive, then delete, the rows
-- keyed by tenant_id = 8382f0b3-c082-4f97-8d47-120ce5a18e05, which is a licences
-- SUBJECT (a user), not a tenant.
-- Date: 2026-10-14. Ruling: George, 2026-10-14 — archive then delete, same
-- treatment as nonagenticai_mcp_server migration 0003 (itself after
-- requirements_management 2026_10_02_0048). Nothing is destroyed.
--
-- WHERE THE ROWS CAME FROM
-- ------------------------
-- An integration run (Mar–Apr 2026) authenticated as that subject while the
-- resolvers preferred `subject_id` over `tenant_id` (auth_middleware.py fixed
-- 2026-04-27; src/auth/tenant.py's fallback fixed in the commit before this
-- one). The workers were created 2026-03-10 .. 2026-04-23. On 2026-09-23/24 a
-- fleet-wide "updated" sweep re-stamped every existing worker, these included,
-- which is why the stray `worker_states` rows (one `updated` row per stray
-- worker) carry September timestamps: they are side effects of the old rows,
-- not new data under this id.
--
-- Measured read-only on worker4 (`BEGIN READ ONLY`, direct, 2026-10-14):
--
--     workers         73 of   637 rows   (created 2026-03-10 .. 2026-04-23)
--     graphs          38 of    42 rows   (created 2026-03-10 .. 2026-03-28)
--     worker_states   73 of  5,446 rows  (every one belongs to a stray worker)
--     every other tenant-keyed column (execution_events, graph_execution_sessions,
--     graph_states, node_execution_logs, skills, tools, workload,
--     tenant_llm_config[_audit], tenant_modality_config, worker_prompt_binding,
--     prompt_owner/prompt_versions.owner_tenant_id,
--     workflow_routing_pointer.released_by_tenant_id, tenant_id_crosswalk): 0
--     foreign keys INTO workers/graphs/worker_states in prod: NONE (the only 3
--     FKs in the database are among the prompt tables), no RLS, no triggers on
--     the three tables.
--
-- ⛔ THE MATCH IS `tenant_id` AND NOTHING ELSE
-- --------------------------------------------
-- The same subject legitimately owns rows in 12 real tenants (prod: 185
-- workers, 1,353 worker_states with subject_id = 8382f0b3... and another
-- tenant_id). `subject_id` is an actor column; matching on it, or on
-- created_by, would delete real tenants' data. The test seeds exactly that
-- negative control.
--
-- ⚠️ FRESH SCHEMA ≠ PROD SCHEMA. postgres_schema.sql declares ON DELETE CASCADE
-- FKs into graphs and workers (graph_states, graph_execution_sessions ->
-- node_execution_logs / execution_events, workload, skills) that prod does not
-- have. So the target set is not a literal list: it is the three tables PLUS
-- every table reachable from them by foreign key, discovered at run time, and
-- every table in it is matched on its own `tenant_id`. In prod that closure is
-- just the three tables.
--
-- WHAT IT DOES (one transaction — the runner's; 10s lock_timeout via SET LOCAL)
-- -----------------------------------------------------------------------------
--  1. Discover the FK closure; refuse a closure table outside this schema, one
--     with no uuid `tenant_id`, one with FORCE ROW LEVEL SECURITY (its count
--     could be blind) or with a user DELETE/TRUNCATE trigger.
--  2. LOCK every target IN SHARE ROW EXCLUSIVE MODE so counts cannot move.
--  3. Refuse if any FK child row of ANOTHER tenant references a stray parent
--     (a cascade would destroy someone else's row unarchived).
--  4. Refuse if any OTHER tenant-keyed column (`*tenant_id`, `*tenant_uuid`)
--     outside the target set holds the stray id: that is stray data this file
--     was not written for — re-measure and extend, do not guess.
--  5. Zero stray rows (CI, fresh installs, already cleaned): log, change
--     nothing, create no schema.
--  6. CREATE SCHEMA IF NOT EXISTS archive (refuse if it exists and is not ours);
--     refuse if a copy already exists; copy each table's stray rows to
--     archive.<table>_8382f0b3 and assert copied == measured.
--  7. DELETE deepest FK level first, so a cascade has nothing left to do;
--     assert deleted == archived per table.
--  8. Assert no stray row remains and every non-stray count is unchanged.
--  9. Close the archive.
--
-- ROLE ANALYSIS — WHY "UNREADABLE BY THE APP ROLE" IS A GUARD, NOT A BOUNDARY
-- ---------------------------------------------------------------------------
-- Measured 2026-10-14: there is NO owner/app split here. `worker_mgmt_user` is
-- the DATABASE_URL role, runs this file, owns the database, the `public` schema
-- and all 22 tables; it is not superuser and has no role memberships; there
-- are no default ACLs. So `archive` is owned by the app role, exactly as in
-- mcp 0003. This file REVOKEs ALL on the schema and its tables from PUBLIC and
-- from CURRENT_USER, and verifies the migrating role then lacks USAGE: the
-- app's own connection gets 42501 on any read. An owner can always re-grant
-- itself; a real boundary needs the copies owned by a role the app is not a
-- member of — a separate, reviewed DBA step.
--
-- WHEN IT RUNS
-- ------------
-- At API pod startup (lifespan -> run_migrations, src/api/worker_management_api.py),
-- over DATABASE_URL (pgbouncer, transaction pooling), NON-strict (no
-- WORKER_MGMT_MIGRATIONS_STRICT in the Deployment). Deployment
-- workers/worker-management: replicas 1, maxSurge 1, maxUnavailable 0, no
-- initContainer. A refusal above therefore ROLLS BACK, is logged as
-- "Migration 040_... failed (continuing)", is not recorded in the ledger, and
-- is retried on the next boot. A concurrent second pod waits on the LOCK, then
-- sees zero stray rows and no-ops.
--
-- Ledger: one-time by design. After it applies, the only way back here is a
-- forgotten ledger row; with the copies present a re-run either no-ops (no
-- stray rows) or refuses (stray rows + existing copy) — never a second,
-- silent delete.
--
-- DOWN: migrations/down/040_archive_stray_tenant_8382f0b3.down.sql (by hand).

BEGIN;

DO $mig040$
DECLARE
    v_stray   constant uuid := '8382f0b3-c082-4f97-8d47-120ce5a18e05';
    v_suffix  constant text := '_8382f0b3';
    v_schema  constant text := current_schema();
    v_roots   constant text[] := ARRAY['workers', 'graphs', 'worker_states'];
    v_targets text[];        -- relnames, deepest FK level first (= delete order)
    v_bad     text;
    v_cond    text;
    v_n       bigint;
    v_m       bigint;
    v_total   bigint := 0;
    v_stray_n jsonb := '{}';
    v_other_n jsonb := '{}';
    v_super   boolean;
    t         text;
    r         record;
BEGIN
    FOREACH t IN ARRAY v_roots LOOP
        IF to_regclass(format('%I.%I', v_schema, t)) IS NULL THEN
            RAISE EXCEPTION '040: table %.% does not exist; refusing', v_schema, t;
        END IF;
    END LOOP;

    -- 1. FK closure of the roots (children of children, ...), deepest first.
    WITH RECURSIVE closure(rel, depth) AS (
        SELECT to_regclass(format('%I.%I', v_schema, x))::oid, 0 FROM unnest(v_roots) AS x
        UNION ALL
        SELECT k.conrelid, c.depth + 1
          FROM pg_constraint k
          JOIN closure c ON k.confrelid = c.rel
         WHERE k.contype = 'f' AND k.conrelid <> k.confrelid AND c.depth < 8
    )
    SELECT array_agg(s.relname ORDER BY s.d DESC, s.relname),
           string_agg(s.relname, ', ') FILTER (WHERE s.nspname <> v_schema OR s.d >= 8)
      INTO v_targets, v_bad
      FROM (SELECT cl.relname::text AS relname, n.nspname::text AS nspname, max(c.depth) AS d
              FROM closure c
              JOIN pg_class cl ON cl.oid = c.rel
              JOIN pg_namespace n ON n.oid = cl.relnamespace
             GROUP BY 1, 2) s;
    IF v_bad IS NOT NULL THEN
        RAISE EXCEPTION '040: FK closure reaches % outside schema % (or is cyclic/too deep); refusing', v_bad, v_schema;
    END IF;

    SELECT string_agg(cl.relname, ', ') INTO v_bad
      FROM pg_class cl
     WHERE cl.relnamespace = v_schema::regnamespace
       AND cl.relname = ANY (v_targets)
       AND (cl.relforcerowsecurity
            OR NOT EXISTS (SELECT 1 FROM pg_attribute a
                            WHERE a.attrelid = cl.oid AND a.attname = 'tenant_id'
                              AND a.atttypid = 'uuid'::regtype AND NOT a.attisdropped)
            OR EXISTS (SELECT 1 FROM pg_trigger g
                        WHERE g.tgrelid = cl.oid AND NOT g.tgisinternal
                          AND (g.tgtype & (8 | 32)) <> 0));  -- DELETE | TRUNCATE
    IF v_bad IS NOT NULL THEN
        RAISE EXCEPTION '040: target table(s) % lack a uuid tenant_id, force RLS, or carry a DELETE/TRUNCATE trigger; refusing', v_bad;
    END IF;

    -- 2. Freeze the targets.
    EXECUTE 'LOCK TABLE '
        || (SELECT string_agg(format('%I.%I', v_schema, x), ', ') FROM unnest(v_targets) AS x)
        || ' IN SHARE ROW EXCLUSIVE MODE';

    -- 3. No other tenant's row may hang off a stray parent.
    FOR r IN
        SELECT k.oid, k.conrelid::regclass AS child, k.confrelid::regclass AS parent
          FROM pg_constraint k
         WHERE k.contype = 'f'
           AND k.confrelid IN (SELECT to_regclass(format('%I.%I', v_schema, x))::oid
                                 FROM unnest(v_targets) AS x)
    LOOP
        SELECT string_agg(format('ch.%I = p.%I', ca.attname, pa.attname), ' AND ')
          INTO v_cond
          FROM pg_constraint k
          CROSS JOIN LATERAL unnest(k.conkey, k.confkey) AS u(ck, pk)
          JOIN pg_attribute ca ON ca.attrelid = k.conrelid AND ca.attnum = u.ck
          JOIN pg_attribute pa ON pa.attrelid = k.confrelid AND pa.attnum = u.pk
         WHERE k.oid = r.oid;
        EXECUTE format(
            'SELECT count(*) FROM %s ch WHERE ch.tenant_id IS DISTINCT FROM %L::uuid '
            'AND EXISTS (SELECT 1 FROM %s p WHERE p.tenant_id = %L::uuid AND %s)',
            r.child, v_stray, r.parent, v_stray, v_cond)
          INTO v_n;
        IF v_n > 0 THEN
            RAISE EXCEPTION '040: % row(s) of another tenant in % reference stray rows of %; deleting would cascade into them; refusing', v_n, r.child, r.parent;
        END IF;
    END LOOP;

    -- 4. Stray id anywhere else is data this file was not written for.
    FOR r IN
        SELECT c.relname::text AS tbl, a.attname::text AS col
          FROM pg_attribute a
          JOIN pg_class c ON c.oid = a.attrelid
         WHERE c.relnamespace = v_schema::regnamespace
           AND c.relkind IN ('r', 'p')
           AND a.attnum > 0 AND NOT a.attisdropped
           AND a.attname ~ '(^|_)tenant_(id|uuid)$'
           AND NOT (c.relname = ANY (v_targets) AND a.attname = 'tenant_id')
    LOOP
        EXECUTE format('SELECT count(*) FROM %I.%I WHERE %I::text = %L',
                       v_schema, r.tbl, r.col, v_stray::text)
          INTO v_n;
        IF v_n > 0 THEN
            RAISE EXCEPTION '040: %.% holds % row(s) for %, outside the archived set; re-measure and extend this migration; refusing', r.tbl, r.col, v_n, v_stray;
        END IF;
    END LOOP;

    -- Census: (stray, other) per target.
    FOREACH t IN ARRAY v_targets LOOP
        EXECUTE format('SELECT count(*) FILTER (WHERE tenant_id = %L::uuid), '
                       'count(*) FILTER (WHERE tenant_id IS DISTINCT FROM %L::uuid) FROM %I.%I',
                       v_stray, v_stray, v_schema, t)
          INTO v_n, v_m;
        v_stray_n := v_stray_n || jsonb_build_object(t, v_n);
        v_other_n := v_other_n || jsonb_build_object(t, v_m);
        v_total := v_total + v_n;
    END LOOP;
    RAISE NOTICE '040 census before: stray=% other=%', v_stray_n, v_other_n;

    -- 5. Nothing to do.
    IF v_total = 0 THEN
        RAISE NOTICE '040: no rows for %; nothing archived', v_stray;
        RETURN;
    END IF;

    -- 6. Archive.
    CREATE SCHEMA IF NOT EXISTS archive;
    IF NOT (SELECT pg_has_role(current_user, nspowner, 'USAGE')
              FROM pg_namespace WHERE nspname = 'archive') THEN
        RAISE EXCEPTION '040: schema archive exists but is not owned by %; refusing', current_user;
    END IF;
    GRANT USAGE, CREATE ON SCHEMA archive TO CURRENT_USER;

    FOREACH t IN ARRAY v_targets LOOP
        CONTINUE WHEN (v_stray_n ->> t)::bigint = 0;
        IF to_regclass(format('archive.%I', t || v_suffix)) IS NOT NULL THEN
            RAISE EXCEPTION '040: archive.% already exists; refusing to mix archives',
                t || v_suffix;
        END IF;
        EXECUTE format('CREATE TABLE archive.%I AS SELECT * FROM %I.%I WHERE tenant_id = %L::uuid',
                       t || v_suffix, v_schema, t, v_stray);
        EXECUTE format('SELECT count(*) FROM archive.%I', t || v_suffix) INTO v_n;
        IF v_n <> (v_stray_n ->> t)::bigint THEN
            RAISE EXCEPTION '040: %: measured % stray rows, archived %', t, v_stray_n ->> t, v_n;
        END IF;
    END LOOP;

    -- 7. Delete, deepest FK level first.
    FOREACH t IN ARRAY v_targets LOOP
        EXECUTE format('DELETE FROM %I.%I WHERE tenant_id = %L::uuid', v_schema, t, v_stray);
        GET DIAGNOSTICS v_n = ROW_COUNT;
        IF v_n <> (v_stray_n ->> t)::bigint THEN
            RAISE EXCEPTION '040: %: archived % rows but deleted %', t, v_stray_n ->> t, v_n;
        END IF;
        IF v_n > 0 THEN
            RAISE NOTICE '040: archived+deleted % rows of % -> archive.%', v_n, t, t || v_suffix;
        END IF;
    END LOOP;

    -- 8. Nothing stray left, nothing else moved.
    FOREACH t IN ARRAY v_targets LOOP
        EXECUTE format('SELECT count(*) FILTER (WHERE tenant_id = %L::uuid), '
                       'count(*) FILTER (WHERE tenant_id IS DISTINCT FROM %L::uuid) FROM %I.%I',
                       v_stray, v_stray, v_schema, t)
          INTO v_n, v_m;
        IF v_n <> 0 THEN
            RAISE EXCEPTION '040: % stray rows remain in %', v_n, t;
        END IF;
        IF v_m <> (v_other_n ->> t)::bigint THEN
            RAISE EXCEPTION '040: %: non-stray rows changed % -> %', t, v_other_n ->> t, v_m;
        END IF;
    END LOOP;

    -- 9. Close the archive to PUBLIC and to the app/owner role itself.
    REVOKE ALL ON ALL TABLES IN SCHEMA archive FROM PUBLIC;
    REVOKE ALL ON ALL TABLES IN SCHEMA archive FROM CURRENT_USER;
    REVOKE ALL ON SCHEMA archive FROM PUBLIC;
    REVOKE ALL ON SCHEMA archive FROM CURRENT_USER;
    SELECT rolsuper INTO v_super FROM pg_roles WHERE rolname = current_user;
    IF NOT v_super AND has_schema_privilege(current_user, 'archive', 'USAGE') THEN
        RAISE EXCEPTION '040: % can still use schema archive after REVOKE; refusing', current_user;
    END IF;
END
$mig040$;

INSERT INTO schema_migrations (version, description)
VALUES ('040', 'Archive then delete rows keyed by licence subject 8382f0b3 as tenant_id')
ON CONFLICT (version) DO NOTHING;

COMMIT;
