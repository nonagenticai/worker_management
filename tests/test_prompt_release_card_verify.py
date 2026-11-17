"""worker_management#336 — a prompt release's card must EXIST and be a PLATFORM card.

The REAL ``release_card.verify_release_card`` runs through the real router;
only the network (an ``httpx.MockTransport`` standing in for fr) and the
repository are replaced. "Nothing written" against real PG is proved in
tests/integration/test_prompt_transitions_ledger.py (TestReleaseCardLookup).

Producer contract: fr ``backend/src/routes/internal-card-platform-owner.ts`` @
``214cac74`` — 200 ``{card_id, tenant_id, platform: true}`` (:161), every
refusal 404 ``{"error":"card_not_found"}`` (:60), outage 503
``card_lookup_unavailable`` (:155), unconfigured 503 ``card_lookup_unconfigured``
(:133).

MUTATIONS (each turns a test here red):
* treat a 5xx as "proceed" ⇒ ``TestMapping.test_unverifiable_*[500/503]``;
* skip WM's own tenant re-check ⇒ ``test_200_for_a_non_platform_tenant_is_422``;
* no TTL on the cache ⇒ ``TestCache.test_a_cached_positive_expires_after_the_ttl``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import httpx2 as httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.prompts import release_card, repository
from src.api.prompts.routes import router
from src.auth import get_tenant_id

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

PLATFORM_TENANT = "aaaaaaaa-0000-4000-8000-000000000001"
AA_CONVEYOR_TENANT = "bbbbbbbb-0000-4000-8000-0000000000a5"
CUSTOMER_TENANT = "aaaaaaaa-0000-4000-8000-000000000100"
# In COMMERCIAL_PLATFORM_TENANT_IDS only — must NOT count as platform here.
SYSTEM_TENANT = "00000000-0000-0000-0000-000000000001"
CARD = "1234567890123456789"
OTHER_CARD = "9876543210987654321"
TOKEN = "ab" * 32  # 64 hex, like the live FR_WM_CARD_LOOKUP_TOKEN
BASE = "http://fr-backend.fr.svc.cluster.local:8888"
STATUS = "/prompts/kw/versions/1/status"
ACTIVATE = "/prompts/kw/versions/1/activate"


def _err(code: str) -> dict[str, Any]:
    return {"detail": {"error": code}}


UNKNOWN = _err("prompt_release_card_unknown")
NOT_PLATFORM = _err("prompt_release_card_not_platform")
UNVERIFIABLE = _err("prompt_release_card_unverifiable")


def _ok(tenant: str = PLATFORM_TENANT) -> Callable[[httpx.Request], httpx.Response]:
    def answer(request: httpx.Request) -> httpx.Response:
        card = request.url.path.split("/")[-2]
        return httpx.Response(200, json={"card_id": card, "tenant_id": tenant, "platform": True})

    return answer


class FakeFr:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.answer: Callable[[httpx.Request], httpx.Response] = _ok()

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.answer(request)

    @property
    def calls(self) -> int:
        return len(self.requests)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Iterator[Clock]:
    c = Clock()
    release_card.set_clock_for_tests(c)
    yield c
    release_card.set_clock_for_tests(None)


@pytest.fixture
def env(monkeypatch):
    from src.settings import reset_settings_cache

    monkeypatch.setenv("FR_CARD_LOOKUP_URL", BASE)
    monkeypatch.setenv("FR_WM_CARD_LOOKUP_TOKEN", TOKEN)
    reset_settings_cache()
    yield monkeypatch
    monkeypatch.undo()
    reset_settings_cache()


@pytest.fixture
def fr(env, clock) -> Iterator[FakeFr]:
    fake = FakeFr()
    release_card.clear_cache()
    release_card.set_transport_for_tests(httpx.MockTransport(fake.handle))
    yield fake
    release_card.set_transport_for_tests(None)
    release_card.clear_cache()


@pytest.fixture
def roles(monkeypatch):
    from src.auth import caller_identity

    def _set(*granted: str) -> None:
        monkeypatch.setattr(caller_identity, "get_caller_roles_from_context", lambda: granted)

    _set("service-internal")
    return _set


@pytest.fixture
def repo():
    with (
        patch.object(repository, "transition", new_callable=AsyncMock) as transition,
        patch.object(repository, "activate", new_callable=AsyncMock) as activate,
        patch.object(repository, "lifecycle_schema_state", return_value="ok"),
    ):
        transition.return_value = (
            {
                "prompt_id": "kw",
                "version": 1,
                "status": "CANARY",
                "config_revision": "kw@1",
                "activated_at": None,
            },
            "DRAFT",
        )
        activate.return_value = (
            {"prompt_id": "kw", "version": 1, "status": "PUBLISHED", "config_revision": "kw@1"},
            None,
            True,
        )
        yield {"transition": transition, "activate": activate}


@pytest.fixture
def client(repo, roles, fr) -> Iterator[TestClient]:
    app = FastAPI()
    app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
    app.include_router(router)

    @asynccontextmanager
    async def _noop(_a):
        yield

    app.router.lifespan_context = _noop
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _release(client: TestClient, route: str, card: str | None = CARD, status: str = "CANARY"):
    if route == "activate":
        return client.post(ACTIVATE, json={"actor_card_id": card})
    return client.post(STATUS, json={"status": status, "actor_card_id": card})


def _written(repo) -> bool:
    return repo["transition"].await_count + repo["activate"].await_count > 0


def _metric(outcome: str) -> float:
    return release_card.VERIFY_TOTAL.labels(outcome=outcome)._value.get()


def _timeout(_r: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout("connect timed out")


def _read_timeout(_r: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("read timed out")


def _refused(_r: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("[Errno 111] Connection refused")


@pytest.mark.parametrize("route", ["status", "activate"])
class TestMapping:
    def test_200_platform_proceeds(self, client, repo, fr, route):
        resp = _release(client, route)
        assert resp.status_code == 200, resp.text
        assert fr.calls == 1
        assert _written(repo)

    def test_200_for_the_aa_conveyor_platform_tenant_proceeds(self, client, repo, fr, route):
        fr.answer = _ok(AA_CONVEYOR_TENANT)
        assert _release(client, route).status_code == 200

    @pytest.mark.parametrize("tenant", [CUSTOMER_TENANT, SYSTEM_TENANT])
    def test_200_for_a_non_platform_tenant_is_422(self, client, repo, fr, route, tenant):
        fr.answer = _ok(tenant)
        resp = _release(client, route)
        assert resp.status_code == 422, resp.text
        assert resp.json() == NOT_PLATFORM
        assert not _written(repo)

    @pytest.mark.parametrize(
        "body", [{"error": "card_not_found"}, {"anything": 1}, None], ids=["fr", "other", "empty"]
    )
    def test_404_is_422_unknown(self, client, repo, fr, route, body):
        fr.answer = lambda _r: httpx.Response(404, json=body) if body else httpx.Response(404)
        resp = _release(client, route)
        assert resp.status_code == 422, resp.text
        assert resp.json() == UNKNOWN
        assert not _written(repo)

    @pytest.mark.parametrize(
        "answer",
        [
            lambda _r: httpx.Response(503, json={"error": "card_lookup_unavailable"}),
            lambda _r: httpx.Response(503, json={"error": "card_lookup_unconfigured"}),
            lambda _r: httpx.Response(500),
            lambda _r: httpx.Response(502),
            lambda _r: httpx.Response(401, json={"error": "Unauthorized"}),
            lambda _r: httpx.Response(403),
            lambda _r: httpx.Response(204),
            lambda _r: httpx.Response(302, headers={"Location": "/"}),
            lambda _r: httpx.Response(200, content=b"{not json"),
            lambda _r: httpx.Response(200, json=[1, 2]),
            lambda _r: httpx.Response(200, json={"card_id": CARD, "tenant_id": PLATFORM_TENANT}),
            lambda _r: httpx.Response(
                200, json={"card_id": CARD, "tenant_id": PLATFORM_TENANT, "platform": "true"}
            ),
            lambda _r: httpx.Response(
                200, json={"card_id": OTHER_CARD, "tenant_id": PLATFORM_TENANT, "platform": True}
            ),
            lambda _r: httpx.Response(200, json={"card_id": CARD, "platform": True}),
            _timeout,
            _read_timeout,
            _refused,
        ],
        ids=[
            "503-unavailable",
            "503-unconfigured",
            "500",
            "502",
            "401",
            "403",
            "204",
            "302",
            "malformed-json",
            "not-an-object",
            "no-platform",
            "platform-not-bool",
            "other-card",
            "no-tenant",
            "connect-timeout",
            "read-timeout",
            "connection-refused",
        ],
    )
    def test_unverifiable_is_503_and_writes_nothing(self, client, repo, fr, route, answer):
        fr.answer = answer
        resp = _release(client, route)
        assert resp.status_code == 503, resp.text
        assert resp.json() == UNVERIFIABLE
        assert resp.headers.get("retry-after") == "30"
        assert fr.calls == 1, "no retry"
        assert not _written(repo)

    @pytest.mark.parametrize("missing", ["FR_CARD_LOOKUP_URL", "FR_WM_CARD_LOOKUP_TOKEN"])
    def test_missing_env_is_503_with_no_call(self, client, repo, fr, env, route, missing):
        from src.settings import reset_settings_cache

        env.delenv(missing)
        reset_settings_cache()
        resp = _release(client, route)
        assert resp.status_code == 503, resp.text
        assert resp.json() == UNVERIFIABLE
        assert fr.calls == 0
        assert not _written(repo)


class TestRequest:
    def test_the_url_and_the_token_header(self, client, fr):
        assert _release(client, "status").status_code == 200
        (req,) = fr.requests
        assert req.method == "GET"
        assert str(req.url) == f"{BASE}/api/internal/cards/{CARD}/platform-owner"
        assert req.headers["X-Internal-Token"] == TOKEN

    def test_a_trailing_slash_on_the_base_is_tolerated(self, client, fr, env):
        from src.settings import reset_settings_cache

        env.setenv("FR_CARD_LOOKUP_URL", BASE + "/")
        reset_settings_cache()
        assert _release(client, "status").status_code == 200
        assert str(fr.requests[0].url) == f"{BASE}/api/internal/cards/{CARD}/platform-owner"

    def test_the_timeout_is_3s_total_1s_connect(self):
        assert httpx.Timeout(3.0, connect=1.0) == release_card.TIMEOUT

    @pytest.mark.parametrize(
        "answer",
        [_ok(), _ok(CUSTOMER_TENANT), lambda _r: httpx.Response(404), _refused, _timeout],
        ids=["ok", "not-platform", "unknown", "refused", "timeout"],
    )
    def test_the_token_is_never_logged(self, client, fr, caplog, answer):
        fr.answer = answer
        with caplog.at_level(logging.DEBUG):
            _release(client, "status")
        assert caplog.records, "the outcome is logged"
        assert TOKEN not in caplog.text
        assert CARD in caplog.text

    def test_the_metric_counts_each_outcome(self, client, fr, clock):
        before = {
            o: _metric(o) for o in ("ok", "cache_hit", "unknown", "not_platform", "unverifiable")
        }
        _release(client, "status")  # ok
        _release(client, "status")  # cache_hit
        fr.answer = lambda _r: httpx.Response(404)
        _release(client, "status", card=OTHER_CARD)  # unknown
        fr.answer = _ok(CUSTOMER_TENANT)
        _release(client, "status", card=OTHER_CARD)  # not_platform
        fr.answer = _refused
        _release(client, "status", card=OTHER_CARD)  # unverifiable
        after = {o: _metric(o) for o in before}
        assert {o: after[o] - before[o] for o in before} == dict.fromkeys(before, 1.0)


class TestOrdering:
    def test_the_role_gate_403_precedes_the_lookup(self, client, repo, fr, roles):
        roles()  # no prompt role
        resp = _release(client, "status")
        assert resp.status_code == 403, resp.text
        assert fr.calls == 0
        assert not _written(repo)

    def test_activate_role_gate_403_precedes_the_lookup(self, client, repo, fr, roles):
        roles("prompt-asset-bootstrap")  # not the elevated role activate needs
        assert _release(client, "activate").status_code == 403
        assert fr.calls == 0

    @pytest.mark.parametrize("route", ["status", "activate"])
    @pytest.mark.parametrize("card", [None, "123", "12345678901234567890", "abcdefghijklmnopqrs"])
    def test_the_format_422_precedes_the_lookup(self, client, repo, fr, route, card):
        resp = _release(client, route, card=card)
        assert resp.status_code == 422, resp.text
        assert resp.json() == _err("prompt_release_card_required")
        assert fr.calls == 0
        assert not _written(repo)

    @pytest.mark.parametrize("target", ["ARCHIVED", "DISCARDED"])
    def test_moves_out_never_call_fr(self, client, repo, fr, target):
        fr.answer = _refused  # would 503 if it were called
        resp = client.post(STATUS, json={"status": target})
        assert resp.status_code == 200, resp.text
        assert fr.calls == 0

    def test_published_is_checked_too(self, client, repo, fr):
        fr.answer = lambda _r: httpx.Response(404)
        resp = _release(client, "status", status="PUBLISHED")
        assert resp.status_code == 422
        assert resp.json() == UNKNOWN


class TestCache:
    def test_a_cache_hit_avoids_a_second_call(self, client, repo, fr):
        assert _release(client, "status").status_code == 200
        assert _release(client, "activate").status_code == 200
        assert fr.calls == 1

    def test_the_cache_is_keyed_by_card(self, client, fr):
        _release(client, "status")
        _release(client, "status", card=OTHER_CARD)
        assert fr.calls == 2

    def test_a_cached_positive_expires_after_the_ttl(self, client, fr, clock):
        """The documented 60 s window: a card that turned non-platform inside
        it is still accepted; after the TTL the next release re-asks and is refused."""
        assert _release(client, "status").status_code == 200
        fr.answer = _ok(CUSTOMER_TENANT)  # the card moved to a customer board
        clock.now += release_card.CACHE_TTL_SECONDS - 1
        assert _release(client, "status").status_code == 200, "inside the window: cached"
        assert fr.calls == 1
        clock.now += 1  # exactly the TTL
        resp = _release(client, "status")
        assert resp.status_code == 422, resp.text
        assert resp.json() == NOT_PLATFORM
        assert fr.calls == 2

    def test_the_ttl_is_60_seconds(self):
        assert release_card.CACHE_TTL_SECONDS == 60.0

    @pytest.mark.parametrize(
        "answer",
        [
            lambda _r: httpx.Response(404),
            _ok(CUSTOMER_TENANT),
            _refused,
            lambda _r: httpx.Response(500),
        ],
        ids=["unknown", "not-platform", "refused", "500"],
    )
    def test_refusals_are_never_cached(self, client, fr, answer):
        fr.answer = answer
        _release(client, "status")
        fr.answer = _ok()
        assert _release(client, "status").status_code == 200
        assert fr.calls == 2
