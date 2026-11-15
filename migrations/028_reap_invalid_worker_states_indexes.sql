-- 028_reap_invalid_worker_states_indexes.sql
--
-- migrate:always
--
-- Closes the one trap that `CREATE INDEX CONCURRENTLY` introduces, and that
-- nothing else in this runner closes.
--
-- THE TRAP, stated by 027's own header before it had bitten anyone:
--
--     "(A failed CONCURRENTLY leaves an INVALID index that `IF NOT EXISTS`
--      will then skip forever -- check `pg_index.indisvalid` and drop it
--      before retrying.)"
--
-- It has now bitten someone. On 2026-08-27 a hand-run `CREATE INDEX
-- CONCURRENTLY` on worker4 deadlocked against the application's own boot-time
-- 027 and left an INVALID 21 MB `idx_worker_states_timestamp` behind.
--
-- Measured against postgres:18 on 2026-08-28, on an index forced to
-- `indisvalid = false`:
--
--     CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_invalid ...
--       -> reports success, statusmessage "CREATE INDEX"
--       -> pg_index.indisvalid is STILL false afterwards
--
-- So the object is invisible to every query plan -- the sweep is exactly as
-- slow as it was with no index at all -- while the migration that was written
-- to create it reports success and is then recorded in `schema_migration_files`
-- permanently. That is the shape this service has been bitten by repeatedly: a
-- green signal for an operation whose RESULT never happened.
--
-- WHY `migrate:always`. This is a control loop, not a one-time change. As an
-- ordinary migration it would be recorded on its first boot and never reap
-- anything again, which is worse than not existing -- it would read like
-- protection while providing none. Its steady-state cost is one catalog SELECT
-- against `pg_index` against a handful of rows. It takes ACCESS EXCLUSIVE only
-- in the rare boot where it actually has an invalid index to drop, and
-- dropping an index no plan can use is instant.
--
-- SCOPE: reaping only, deliberately.
--
-- This file does NOT rebuild what it reaps. Rebuilding needs CONCURRENTLY
-- (a plain `CREATE INDEX` takes SHARE and blocks the constant INSERTs into
-- `worker_states` -- the original incident), and CONCURRENTLY cannot run in
-- the transaction this file needs for its `DO` block. After a reap, rebuild
-- deliberately: drop the file's row from `schema_migration_files` so 026/027
-- re-run, or build it by hand with CONCURRENTLY off the boot path. Reaping on
-- its own is already strictly correct -- an INVALID index is unusable by every
-- planner, is still maintained on write, and blocks its own recreation by name
-- forever.

DO $reap$
DECLARE
    dead record;
BEGIN
    FOR dead IN
        SELECT c.relname AS idxname
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_class t ON t.oid = i.indrelid
        WHERE t.relname = 'worker_states'
          AND NOT i.indisvalid
    LOOP
        RAISE WARNING
            'reaping invalid index % on worker_states: an interrupted CREATE '
            'INDEX CONCURRENTLY left it unusable, and IF NOT EXISTS would skip '
            'it forever. Rebuild it deliberately -- see 028''s header.',
            dead.idxname;
        EXECUTE format('DROP INDEX IF EXISTS %I', dead.idxname);
    END LOOP;
END
$reap$;
