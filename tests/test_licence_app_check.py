"""Tenant×app licence check (George 2026-10-15): a licence here must be a ``workers`` one.

Mirrors inference-router's ``EXPECTED_APP_KEY`` / ``wrong_app`` check, shipped
OBSERVE-first behind ``WM_LICENCE_APP_MODE`` (observe | enforce, default observe):

- a ``workers`` licence is ``tenant_app_ok`` and served in both modes;
- a ``coding`` licence without delegation is ``wrong_app``: served under observe,
  401 ``wrong_app`` under enforce;
- a delegating subject using ``X-Effective-Tenant-Id`` is ``delegated`` and served
  in BOTH modes (fr's licence is a ``coding`` licence; removal is a later step);
- an introspection with no ``app`` is ``no_app``;
- an unknown flag value refuses startup (Settings validation), never parses to observe.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

import src.api.auth_middleware as mw
from src.api.auth_middleware import AuthenticationMiddleware
from src.settings import Settings, reset_settings_cache
from src.telemetry.attribution_metrics import AUTH_REFUSED_TOTAL, LICENCE_APP_TOTAL

LICENCE_TENANT = "aaaaaaaa-0000-4000-8000-0000000000aa"
RUN_TENANT = "aaaaaaaa-0000-4000-8000-0000000000bb"
SUBJECT = "01a0dd02-0000-7000-8000-000000000001"  # NOT delegating
DELEGATOR = "019ffe9f-0000-7000-8000-00000000f00d"
TRUSTED_PEER = ("10.42.1.7", 54321)
MODE_ENV = "WM_LICENCE_APP_MODE"
PATH = "/api/probe"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(MODE_ENV, raising=False)
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", DELEGATOR)
    mw._licence_app_log_seen.clear()
    reset_settings_cache()
    yield
    mw._licence_app_log_seen.clear()
    reset_settings_cache()


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get(PATH)
    async def probe(request: Request) -> dict[str, object]:
        return {"tenant_id": request.state.tenant_id}

    return application


def _intro(app_key: str | None, subject: str = SUBJECT) -> dict[str, Any]:
    result: dict[str, Any] = {
        "active": True,
        "tenant_id": LICENCE_TENANT,
        "subject_id": subject,
        "license_id": "lic-0001",
        "byo_mode": "local_allowed",
    }
    if app_key is not None:
        # licences main.py TokenIntrospectResponse.app = App.to_dict(): {id, key, name, ...}
        result["app"] = {"id": "app-1", "key": app_key, "name": app_key.title()}
    return result


def _call(app: FastAPI, intro: dict[str, Any], headers: dict[str, str] | None = None):
    with (
        patch.object(mw, "_introspect_license", AsyncMock(return_value=intro)),
        patch.object(
            mw, "_lookup_tenant_byo_mode", new_callable=AsyncMock, return_value="local_allowed"
        ),
        patch.object(mw, "get_metrics_client", return_value=MagicMock()),
        TestClient(app, client=TRUSTED_PEER, raise_server_exceptions=False) as client,
    ):
        resp = client.get(PATH, headers={"Authorization": "Bearer lic-a", **(headers or {})})
    return resp.status_code, resp.json()


def _count(app_label: str, outcome: str, path: str = PATH) -> float:
    return LICENCE_APP_TOTAL.labels(app=app_label, path=path, outcome=outcome)._value.get()


def _refused(reason: str = "wrong_app") -> float:
    return AUTH_REFUSED_TOTAL.labels(reason=reason)._value.get()


def _mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv(MODE_ENV, mode)
    reset_settings_cache()


class TestWorkersLicence:
    @pytest.mark.parametrize("mode", ["observe", "enforce"])
    def test_workers_licence_is_ok_in_both_modes(self, app, monkeypatch, mode):
        _mode(monkeypatch, mode)
        before = _count("workers", "tenant_app_ok")
        status, body = _call(app, _intro("workers"))
        assert status == 200
        assert body == {"tenant_id": LICENCE_TENANT}
        assert _count("workers", "tenant_app_ok") == before + 1


class TestWrongApp:
    def test_default_mode_is_observe(self):
        assert Settings().wm_licence_app_mode == "observe"

    def test_coding_licence_is_counted_wrong_app_and_served_in_observe(self, app, caplog) -> None:
        before = _count("coding", "wrong_app")
        refused_before = _refused()
        with caplog.at_level(logging.WARNING, logger="src.api.auth_middleware"):
            status, _ = _call(app, _intro("coding"))
            _call(app, _intro("coding"))  # deduped: no second line
        assert status == 200
        assert _count("coding", "wrong_app") == before + 2
        assert _refused() == refused_before
        lines = [r.getMessage() for r in caplog.records if "licence_app" in r.getMessage()]
        assert len(lines) == 1
        assert f"subject={SUBJECT}" in lines[0]
        assert "app=coding" in lines[0]
        assert "action=SERVED" in lines[0]

    def test_coding_licence_is_401_wrong_app_in_enforce(self, app, monkeypatch) -> None:
        _mode(monkeypatch, "enforce")
        # Refused before routing: the path label is the unmatched constant.
        before = _count("coding", "wrong_app", path="<unmatched>")
        refused_before = _refused()
        status, body = _call(app, _intro("coding"))
        assert status == 401
        assert body["error"]["code"] == "wrong_app"
        assert body["error"]["expected_app"] == "workers"
        assert body["error"]["app"] == "coding"
        assert _count("coding", "wrong_app", path="<unmatched>") == before + 1
        assert _refused() == refused_before + 1


class TestDelegated:
    @pytest.mark.parametrize("mode", ["observe", "enforce"])
    def test_delegated_coding_licence_is_unaffected(self, app, monkeypatch, mode) -> None:
        _mode(monkeypatch, mode)
        before = _count("coding", "delegated")
        refused_before = _refused()
        status, body = _call(
            app,
            _intro("coding", subject=DELEGATOR),
            headers={"X-Effective-Tenant-Id": RUN_TENANT},
        )
        assert status == 200
        assert body == {"tenant_id": RUN_TENANT}
        assert _count("coding", "delegated") == before + 1
        assert _refused() == refused_before

    def test_delegating_subject_without_effective_tenant_is_judged_on_its_app(
        self, app, monkeypatch
    ) -> None:
        _mode(monkeypatch, "enforce")
        status, _ = _call(app, _intro("coding", subject=DELEGATOR))
        assert status == 401


class TestNoApp:
    def test_missing_app_is_counted_no_app_and_served_in_observe(self, app) -> None:
        before = _count("none", "no_app")
        status, _ = _call(app, _intro(None))
        assert status == 200
        assert _count("none", "no_app") == before + 1

    def test_missing_app_is_401_in_enforce(self, app, monkeypatch) -> None:
        _mode(monkeypatch, "enforce")
        status, body = _call(app, _intro(None))
        assert status == 401
        assert body["error"]["code"] == "wrong_app"
        assert body["error"]["app"] is None

    def test_non_key_shaped_app_is_labelled_other(self, app) -> None:
        before = _count("other", "wrong_app")
        _call(app, _intro("Not A Key!"))
        assert _count("other", "wrong_app") == before + 1


class TestFlagParsing:
    @pytest.mark.parametrize("raw", ["Enforce", "on", "true", "", "observe "])
    def test_unknown_flag_value_refuses_startup(self, monkeypatch, raw) -> None:
        monkeypatch.setenv(MODE_ENV, raw)
        with pytest.raises(ValidationError, match=r"wm_licence_app_mode|WM_LICENCE_APP_MODE"):
            Settings()

    @pytest.mark.parametrize("raw", ["observe", "enforce"])
    def test_known_values_parse(self, monkeypatch, raw) -> None:
        monkeypatch.setenv(MODE_ENV, raw)
        assert Settings().wm_licence_app_mode == raw


def _licence_app_total() -> float:
    return sum(
        s.value
        for m in LICENCE_APP_TOTAL.collect()
        for s in m.samples
        if s.name == "wm_licence_app_total"
    )


class TestTokenlessPath:
    """The tokenless trusted-network branch presents no licence, so it has no app:
    the app check must neither count nor refuse it, in either mode."""

    @pytest.mark.parametrize("mode", ["observe", "enforce"])
    def test_tokenless_trusted_request_is_not_counted_or_refused(
        self, app, monkeypatch, mode
    ) -> None:
        _mode(monkeypatch, mode)
        before = _licence_app_total()
        refused_before = _refused()
        spy = AsyncMock()
        with (
            patch.object(mw, "_introspect_license", spy),
            patch.object(
                mw,
                "_lookup_tenant_byo_mode",
                new_callable=AsyncMock,
                return_value="local_allowed",
            ),
            patch.object(mw, "get_metrics_client", return_value=MagicMock()),
            TestClient(app, client=TRUSTED_PEER, raise_server_exceptions=False) as client,
        ):
            resp = client.get(PATH, headers={"X-Tenant-Id": LICENCE_TENANT})
        assert resp.status_code == 200
        assert resp.json() == {"tenant_id": LICENCE_TENANT}
        spy.assert_not_awaited()
        assert _licence_app_total() == before
        assert _refused() == refused_before
