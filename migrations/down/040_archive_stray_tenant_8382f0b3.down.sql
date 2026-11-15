-- DOWN for migrations/040_archive_stray_tenant_8382f0b3.sql. Applied BY HAND only
-- (psql -1 -f, as the owner role); the startup runner never descends into this
-- directory.
--
-- Re-grants the owner its privileges on `archive`, restores every
-- archive.<table>_8382f0b3 copy into its table (shallowest FK level first, so
-- parents exist before children; original ids kept), drops the copies, drops
-- the schema if it is then empty, and removes 040's ledger rows.
--
-- ⚠️ Restoring re-creates rows keyed by a licence SUBJECT as tenant_id — the
-- state George ruled (2026-10-14) should not exist. Use it only to recover
-- from a bad 040, and re-apply 040 afterwards.

BEGIN;

DO $down040$
DECLARE
    v_suffix constant text := '_8382f0b3';
    v_schema constant text := current_schema();
    v_n      bigint;
    v_m      bigint;
    v_cols   text;
    r        record;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'archive') THEN
        RAISE NOTICE '040 down: schema archive absent; nothing to restore';
        RETURN;
    END IF;
    GRANT USAGE, CREATE ON SCHEMA archive TO CURRENT_USER;
    GRANT ALL ON ALL TABLES IN SCHEMA archive TO CURRENT_USER;

    -- Shallowest first: a table that is an FK child of another copy comes later.
    FOR r IN
        SELECT c.relname::text AS copy, left(c.relname, length(c.relname) - length(v_suffix)) AS tbl,
               (SELECT count(*) FROM pg_constraint k
                 WHERE k.contype = 'f'
                   AND k.conrelid = to_regclass(format('%I.%I', v_schema,
                                   left(c.relname, length(c.relname) - length(v_suffix))))) AS fks
          FROM pg_class c
         WHERE c.relnamespace = 'archive'::regnamespace AND c.relkind = 'r'
           AND c.relname LIKE '%' || v_suffix
         ORDER BY 3, 1
    LOOP
        SELECT string_agg(format('%I', a.attname), ', ' ORDER BY a.attnum) INTO v_cols
          FROM pg_attribute a
         WHERE a.attrelid = format('archive.%I', r.copy)::regclass AND a.attnum > 0 AND NOT a.attisdropped;
        EXECUTE format('SELECT count(*) FROM archive.%I', r.copy) INTO v_n;
        EXECUTE format('INSERT INTO %I.%I (%s) SELECT %s FROM archive.%I',
                       v_schema, r.tbl, v_cols, v_cols, r.copy);
        GET DIAGNOSTICS v_m = ROW_COUNT;
        IF v_m <> v_n THEN
            RAISE EXCEPTION '040 down: %: archived %, restored %', r.tbl, v_n, v_m;
        END IF;
        EXECUTE format('DROP TABLE archive.%I', r.copy);
        RAISE NOTICE '040 down: restored % rows into %', v_m, r.tbl;
    END LOOP;

    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relnamespace = 'archive'::regnamespace) THEN
        DROP SCHEMA archive;
    ELSE
        REVOKE ALL ON ALL TABLES IN SCHEMA archive FROM CURRENT_USER;
        REVOKE ALL ON SCHEMA archive FROM CURRENT_USER;
    END IF;
END
$down040$;

DELETE FROM schema_migration_files WHERE filename = '040_archive_stray_tenant_8382f0b3.sql';
DELETE FROM schema_migrations WHERE version = '040';

COMMIT;
