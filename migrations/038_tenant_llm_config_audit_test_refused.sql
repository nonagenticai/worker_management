-- Migration: 038_tenant_llm_config_audit_test_refused — a refused key test is
-- recorded as ``test_refused``, not as ``test``.
-- Date: 2026-10-02
--
-- WHY (WM-4). ``POST /tenant-config/llm/test`` wrote its customer-visible audit
-- row BEFORE the entitlement / key-usability verdict, so a test that was refused
-- (withheld or undecryptable key, refused endpoint — no live call to the vendor)
-- showed up in the customer's history as "Key tested". The row is now written
-- after the verdict: ``test`` only when a live test is allowed to run,
-- ``test_refused`` otherwise. Still fail-closed (the row is written before any
-- live call, and a failed write refuses the test with 503).
--
-- Widens the action CHECK only; existing rows are untouched. Re-runnable (DROP
-- IF EXISTS + ADD). DOWN: migrations/down/038_tenant_llm_config_audit_test_refused.down.sql.

BEGIN;

ALTER TABLE tenant_llm_config_audit
    DROP CONSTRAINT IF EXISTS tenant_llm_config_audit_action_chk;
ALTER TABLE tenant_llm_config_audit
    ADD CONSTRAINT tenant_llm_config_audit_action_chk
    CHECK (action IN ('set_key', 'clear_key', 'update', 'remove', 'test', 'test_refused'));

COMMIT;
