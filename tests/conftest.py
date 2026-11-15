"""Shared test configuration: environment pinned before any src import, plus
autouse fixtures that reset settings, pin the platform tenant set and stub DNS."""

# Set test environment BEFORE any imports
import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only-32chars")
# Settings.model_post_init fail-fasts in production when required infra fields
# are empty. Tests import src modules that call get_settings() at import time,
# so pin environment=test before any src import reaches that validator.
os.environ.setdefault("ENVIRONMENT", "test")
# Unit tests run without a connection pool, so the run_migrations() call in
# the lifespan would raise. Tests that hit a
# real Postgres (tests/integration/) override this env var to allow the
# canonical path to run.
os.environ.setdefault("WORKER_MGMT_SKIP_MIGRATIONS", "1")
# `worker_management_api` reads ENABLE_AUTH ONCE, at import, to decide whether to
# `app.add_middleware(AuthenticationMiddleware, ...)`. ~18 test modules each set
# this to "false" at their own import time, so whether the middleware exists
# depended on WHICH tests were collected. tests/test_security/test_data_isolation.py
# never set it and is 13 failed / 4 passed with 401s when run STANDALONE, green in
# the full suite — so CI could not see it. Pinned here, once, before any src import.
# Measured: every existing reference in tests/ sets it to "false"; none wants it on.
os.environ.setdefault("ENABLE_AUTH", "false")
# The lifespan starts a background retention sweeper that DELETEs from
# worker_states. Unit tests enter `TestClient(app)`, which runs that lifespan, so
# with DATABASE_URL exported the unit pass points a live deleter at the real
# database — measured 2026-08-29: running the unit suite and `-m pg` in one shell
# failed 3 tests/integration/test_inference_graph_end_to_end.py tests that are
# clean in isolation. A test suite must not run a destructive sweeper.
os.environ.setdefault("WORKER_STATE_RETENTION_ENABLED", "false")
# Same for the tenant_llm_config_audit prune (WM-5): a background DELETE.
os.environ.setdefault("TENANT_LLM_AUDIT_RETENTION_ENABLED", "false")
# Same reason, read-only this time: the lifespan also starts the prompt-binding
# drift refresher (#314). Under TestClient it would hit a mocked/absent pool and
# move the shared wm_prompt_drift_* series under tests that measure deltas on
# them. tests/test_prompt_drift.py drives refresh_once() directly instead.
os.environ.setdefault("WM_PROMPT_DRIFT_INTERVAL_S", "0")

import pytest

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    """Clear the cached Settings before each test.

    Tests frequently mutate os.environ via monkeypatch / patch.dict to
    exercise config-loading behavior. Since ``get_settings`` is
    ``lru_cache``-memoized, those mutations would otherwise be ignored.
    """
    from src.settings import reset_settings_cache

    reset_settings_cache()


# The commercial-platform set is defined ONCE, in gitops
# (``clusters/strato/services/workers/deployment.yaml`` env
# ``COMMERCIAL_PLATFORM_TENANT_IDS``); the code default is EMPTY (fail closed).
# Tests run against the estate's value so the platform/customer split they pin
# is the production one. A test that needs another set overrides the env.
TEST_COMMERCIAL_PLATFORM_TENANT_IDS = (
    "0195f48b-d000-7000-8000-000000000001,"
    "0195f48b-d000-7000-8000-000000000002,"
    "0195f48b-d000-7000-8000-000000000003,"
    "0195f48b-d000-7000-8000-000000000004,"
    "0195f48b-d000-7000-8000-000000000005,"
    "019de842-c92a-7e37-8fdd-eb0d8c5396f9,"
    "00000000-0000-0000-0000-000000000001"
)


@pytest.fixture(autouse=True)
def _commercial_platform_tenant_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test sees the gitops commercial-platform set (see above)."""
    monkeypatch.delenv("WORKER_MANAGEMENT_COMMERCIAL_PLATFORM_TENANT_IDS", raising=False)
    monkeypatch.setenv("COMMERCIAL_PLATFORM_TENANT_IDS", TEST_COMMERCIAL_PLATFORM_TENANT_IDS)


# A routable, GLOBAL address (TEST-NET ranges are not ``is_global``, so they
# would read as owner infra). Never contacted: every test that sends traffic
# mocks the transport.
STUB_PUBLIC_IP = "93.184.215.14"


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tenant BYO endpoint checks resolve hostnames (validator 3 D3); tests never
    touch real DNS. Every name resolves to one public address unless a test
    re-stubs ``_resolve_host_ips`` to pin a specific resolution."""
    from src.security import request_overrides

    request_overrides._dns_cache.clear()
    monkeypatch.setattr(request_overrides, "_resolve_host_ips", lambda _host: (STUB_PUBLIC_IP,))


def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.

    Registers custom markers for categorizing tests:
    - unit: Unit tests
    - integration: Integration tests
    - security: Security-focused tests
    - slow: Tests that take longer to run
    """
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "security: Security-focused tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
