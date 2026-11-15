"""The suite must not depend on which tests were collected, or on the operator's shell.

Two isolation defects were measured on 2026-08-29/30 and fixed in tests/conftest.py.
Both were invisible to CI, which is why they survived:

1. ``worker_management_api`` reads ``ENABLE_AUTH`` ONCE, at import, to decide whether
   to attach ``AuthenticationMiddleware``. ~18 test modules each set it to "false" at
   their own import time, so whether the middleware existed depended on WHICH tests
   were collected. ``tests/test_security/test_data_isolation.py`` never set it and was
   **13 failed / 4 passed with 401s standalone**, green in the full run.

2. The lifespan starts a background sweeper that DELETEs from ``worker_states``. Unit
   tests enter ``TestClient(app)``, which runs that lifespan, so with ``DATABASE_URL``
   exported the unit pass pointed a live deleter at the real database. Running the unit
   suite and ``-m pg`` in one shell failed 3 otherwise-clean integration tests.

These assert the PROPERTY, not the env var, so they still bite if someone reaches the
same broken state by another route.
"""

from __future__ import annotations


def test_auth_middleware_is_not_attached_in_tests():
    """Every test in this repo assumes it can address a tenant by a bare string."""
    from src.api.worker_management_api import app

    attached = [getattr(m.cls, "__name__", str(m.cls)) for m in app.user_middleware]
    assert "AuthenticationMiddleware" not in attached, (
        "AuthenticationMiddleware is attached during tests. It is decided ONCE at "
        f"import from ENABLE_AUTH, so this makes results collection-order dependent. "
        f"Attached: {attached}"
    )


def test_the_retention_sweeper_is_disabled_in_tests():
    """A test suite must never run a background deleter against a real database."""
    from src.services.worker_states_retention import start_retention_task

    assert start_retention_task() is None, (
        "the worker_states retention sweeper started during tests — with DATABASE_URL "
        "exported it DELETEs from the real database and corrupts the -m pg pass"
    )


def test_the_audit_retention_prune_is_disabled_in_tests():
    """WM-5: the tenant_llm_config_audit prune is a background DELETE too."""
    from src.services.tenant_llm_audit_retention import start_audit_retention_task

    assert start_audit_retention_task() is None


def test_the_prompt_drift_refresher_is_disabled_in_tests():
    """The lifespan's drift refresher (#314) is read-only, but it moves the shared
    wm_prompt_drift_* series under tests/test_prompt_drift.py, which measures them."""
    from src.api.workers.prompt_drift import start_drift_task

    assert start_drift_task() is None, (
        "the prompt-binding drift refresher started during tests — set "
        "WM_PROMPT_DRIFT_INTERVAL_S=0 in tests/conftest.py"
    )
