"""fr#1385 — the commercial-platform set is read from ONE env name only.

gitops CI guards ``COMMERCIAL_PLATFORM_TENANT_IDS`` (UUIDs, identical
everywhere). A ``WORKER_MANAGEMENT_``-prefixed alias used to be accepted as the
FIRST ``AliasChoices`` entry, so a stray prefixed copy — even an EMPTY one —
silently overrode the guarded name. Settings has no ``env_prefix``; the field's
``validation_alias`` is the single unprefixed name.
"""

from __future__ import annotations

import pytest

from src.settings import Settings, get_settings, reset_settings_cache

GUARDED = "COMMERCIAL_PLATFORM_TENANT_IDS"
PREFIXED = "WORKER_MANAGEMENT_COMMERCIAL_PLATFORM_TENANT_IDS"
T1 = "aaaaaaaa-0000-4000-8000-000000000001"
T2 = "aaaaaaaa-0000-4000-8000-000000000002"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(GUARDED, raising=False)
    monkeypatch.delenv(PREFIXED, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


def _platform_set() -> frozenset[str]:
    from src.api.tenant_constants import commercial_platform_tenant_ids

    reset_settings_cache()
    return commercial_platform_tenant_ids()


def test_unprefixed_name_is_parsed(monkeypatch):
    monkeypatch.setenv(GUARDED, f"{T1},{T2}")
    assert Settings().commercial_platform_tenant_ids == f"{T1},{T2}"
    assert _platform_set() == frozenset({T1, T2})


def test_prefixed_name_alone_is_ignored(monkeypatch):
    monkeypatch.setenv(PREFIXED, T1)
    assert Settings().commercial_platform_tenant_ids == ""
    assert _platform_set() == frozenset()


@pytest.mark.parametrize("prefixed_value", ["", T2])
def test_unprefixed_wins_when_both_are_set(monkeypatch, prefixed_value):
    """An EMPTY prefixed copy used to win (and empty the set) — the bug."""
    monkeypatch.setenv(PREFIXED, prefixed_value)
    monkeypatch.setenv(GUARDED, T1)
    assert get_settings().commercial_platform_tenant_ids == T1
    assert _platform_set() == frozenset({T1})
