"""HTTP surface of ``/prompts*`` with the repository mocked.

The DB behaviour (immutability, the single-active invariant, search ranking) is
proved against a real PostgreSQL 18 in
``tests/integration/test_prompt_versions_api.py``. What is proved HERE is the
part that has nothing to do with Postgres and everything to do with getting the
route wrong: path validation, status codes, the digests being computed
SERVER-side from the body, and — the one a reviewer cannot see by reading —
that ``/prompts/search`` is not swallowed by ``/prompts/{prompt_id}``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.api.prompts import repository
from src.api.prompts.hashing import sha256_text, version_payload_sha256
from src.api.prompts.routes import router
from src.auth import get_tenant_id

# …0001 is in PLATFORM_TENANT_IDS; …0100 (cdmsfirst) is a real customer.
PLATFORM_TENANT = "0195f48b-d000-7000-8000-000000000001"
CUSTOMER_TENANT = "0195f48b-d000-7000-8000-000000000100"
# In COMMERCIAL_PLATFORM_TENANT_IDS only, never in the prompt-write set.
SYSTEM_TENANT = "00000000-0000-0000-0000-000000000001"
SYSTEM = "You are a keyword picker."
# A Planka card id: REALIGN R7 requires one on every move into CANARY/PUBLISHED.
CARD = "1234567890123456789"
CARD_BODY: dict[str, Any] = {"actor_card_id": CARD}
BODY: dict[str, Any] = {
    "system": SYSTEM,
    "question": "Pick keywords",
    "template": "Card: {{card}}",
    "worker_names": ["worker-decide-keywords"],
    "purpose_tags": ["keywords"],
    "git_sha": "abc123",
}
EXPECTED_PAYLOAD_SHA = version_payload_sha256(
    question=BODY["question"],
    system=SYSTEM,
    template=BODY["template"],
    output_schema=None,
    worker_names=BODY["worker_names"],
    purpose_tags=BODY["purpose_tags"],
)


def _row(**overrides):
    row = {
        "prompt_id": "kw",
        "version": 1,
        "sha256": EXPECTED_PAYLOAD_SHA,
        "system_prompt_sha256": sha256_text(SYSTEM),
        "question": BODY["question"],
        "system": SYSTEM,
        "template": BODY["template"],
        "output_schema": None,
        "worker_names": BODY["worker_names"],
        "purpose_tags": BODY["purpose_tags"],
        "git_sha": "abc123",
        "status": "DRAFT",
        "published_at": "2026-09-23T00:00:00+00:00",
        "activated_at": None,
        "config_revision": "kw@1",
    }
    row.update(overrides)
    return row


@pytest.fixture
def repo():
    with (
        patch.object(repository, "get_version", new_callable=AsyncMock) as get_version,
        patch.object(repository, "get_active", new_callable=AsyncMock) as get_active,
        patch.object(repository, "search", new_callable=AsyncMock) as search,
        patch.object(repository, "upsert_version", new_callable=AsyncMock) as upsert,
        patch.object(repository, "activate", new_callable=AsyncMock) as activate,
        patch.object(repository, "transition", new_callable=AsyncMock) as transition,
        patch.object(repository, "get_transitions", new_callable=AsyncMock) as get_transitions,
    ):
        get_version.return_value = _row()
        get_active.return_value = _row(status="PUBLISHED")
        search.return_value = []
        upsert.return_value = (_row(), True)
        activate.return_value = (_row(status="PUBLISHED"), None, True)
        transition.return_value = (_row(status="CANARY"), "DRAFT")
        get_transitions.return_value = []
        yield {
            "get_transitions": get_transitions,
            "transition": transition,
            "get_version": get_version,
            "get_active": get_active,
            "search": search,
            "upsert": upsert,
            "activate": activate,
        }


@pytest.fixture
def roles(monkeypatch):
    """Set the verified caller roles ``require_role`` reads for this test.

    Patched at the module ``require_role`` resolves it from (a module global,
    visible to the TestClient's worker thread — a ContextVar would not be).
    """
    from src.auth import caller_identity

    def _set(*granted: str) -> None:
        monkeypatch.setattr(caller_identity, "get_caller_roles_from_context", lambda: granted)

    _set()  # default: no roles, as for every live caller today
    return _set


@pytest.fixture
def customer(client):
    """Make the caller a NON-platform (customer) tenant for this test."""
    client.app.dependency_overrides[get_tenant_id] = lambda: CUSTOMER_TENANT  # type: ignore[attr-defined]


@pytest.fixture
def admin(roles):
    roles("service-internal")


@pytest.fixture
def schema_state():
    """Set the cached 031 probe result for one test; restore afterwards."""
    before = repository.lifecycle_schema_state()
    yield repository.set_lifecycle_schema_state
    repository.set_lifecycle_schema_state(before)


@pytest.fixture
def card_verify():
    """worker_management#336's fr lookup, stubbed to "platform card".

    The lookup itself is proved in tests/test_prompt_release_card_verify.py;
    every class here tests what a release does once the card is confirmed.
    """
    from src.api.prompts import release_card

    with patch.object(release_card, "verify_release_card", new_callable=AsyncMock) as verify:
        yield verify


@pytest.fixture
def client(repo, card_verify):
    app = FastAPI()
    # A PLATFORM tenant by default: every prompt route, read or write, is
    # platform-only (TestPlatformGate), and every other class here tests what a
    # call does once admitted. ``customer`` switches to a non-platform tenant.
    app.dependency_overrides[get_tenant_id] = lambda: PLATFORM_TENANT
    app.include_router(router)

    @asynccontextmanager
    async def _noop(_a):
        yield

    app.router.lifespan_context = _noop
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


OTHER_PLATFORM_TENANT = "0195f48b-d000-7000-8000-000000000003"  # george-dev


class TestOwnership:
    """Migration 035: a prompt has ONE owning tenant. The repository enforces it
    (proved on real PG in tests/integration/test_prompt_versions_api.py,
    TestPromptOwnership); here: the route hands it the CALLER's tenant and the
    caller's privilege, and maps a refusal to the fixed 403."""

    def test_put_passes_the_callers_tenant_as_owner_unprivileged(self, client, repo, roles):
        assert client.put("/prompts/kw/versions/1", json=BODY).status_code == 200
        kwargs = repo["upsert"].await_args.kwargs
        assert kwargs["owner_tenant_id"] == PLATFORM_TENANT
        assert kwargs["privileged"] is False

    def test_a_platform_tenant_writing_another_tenants_prompt_is_403(self, client, repo, roles):
        client.app.dependency_overrides[get_tenant_id] = lambda: OTHER_PLATFORM_TENANT  # type: ignore[attr-defined]
        repo["upsert"].side_effect = repository.PromptOwnerMismatch(
            "kw", PLATFORM_TENANT, OTHER_PLATFORM_TENANT
        )
        resp = client.put("/prompts/kw/versions/2", json=BODY)
        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": {"error": "prompt_owner_mismatch"}}
        # The refusal names neither tenant.
        assert PLATFORM_TENANT not in resp.text
        assert OTHER_PLATFORM_TENANT not in resp.text
        kwargs = repo["upsert"].await_args.kwargs
        assert (kwargs["owner_tenant_id"], kwargs["privileged"]) == (OTHER_PLATFORM_TENANT, False)

    def test_a_status_move_on_another_tenants_prompt_is_403(self, client, repo, roles):
        client.app.dependency_overrides[get_tenant_id] = lambda: OTHER_PLATFORM_TENANT  # type: ignore[attr-defined]
        repo["transition"].side_effect = repository.PromptOwnerMismatch(
            "kw", PLATFORM_TENANT, OTHER_PLATFORM_TENANT
        )
        resp = client.post("/prompts/kw/versions/1/status", json={"status": "ARCHIVED"})
        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": {"error": "prompt_owner_mismatch"}}
        kwargs = repo["transition"].await_args.kwargs
        assert (kwargs["caller_tenant_id"], kwargs["privileged"]) == (OTHER_PLATFORM_TENANT, False)

    @pytest.mark.parametrize("role", ["service-internal", "prompt-asset-bootstrap"])
    def test_a_prompt_role_is_privileged(self, client, repo, roles, role):
        roles(role)
        assert client.put("/prompts/kw/versions/1", json=BODY).status_code == 200
        assert repo["upsert"].await_args.kwargs["privileged"] is True

    def test_a_missing_owner_table_is_503_naming_035(self, client, repo):
        import psycopg

        repo["upsert"].side_effect = psycopg.errors.UndefinedTable(
            'relation "prompt_owner" does not exist'
        )
        resp = client.put("/prompts/kw/versions/1", json=BODY)
        assert resp.status_code == 503
        assert "prompt_owner_schema_missing" in resp.text


class TestRouteOrder:
    def test_search_is_not_served_as_a_prompt_named_search(self, client, repo):
        """⚠️ Registration order is the ONLY thing separating these two routes.
        Declare ``/{prompt_id}`` first and this request silently becomes a
        lookup of a prompt literally called 'search'."""
        resp = client.get("/prompts/search", params={"q": "keywords"})
        assert resp.status_code == 200
        repo["search"].assert_awaited_once()
        repo["get_active"].assert_not_awaited()


class TestPublish:
    def test_server_computes_both_digests_from_the_body(self, client, repo):
        resp = client.put("/prompts/kw/versions/1", json=BODY)
        assert resp.status_code == 200, resp.text
        kwargs = repo["upsert"].await_args.kwargs
        assert kwargs["sha256"] == EXPECTED_PAYLOAD_SHA
        assert kwargs["system_prompt_sha256"] == sha256_text(SYSTEM)

    def test_a_client_supplied_sha_is_never_stored_as_given(self, client, repo):
        """It is an assertion to check, not an input to trust — otherwise a
        publisher could label content with any digest it liked."""
        lie = "0" * 64
        resp = client.put("/prompts/kw/versions/1", json={**BODY, "expected_sha256": lie})
        assert resp.status_code == 422
        repo["upsert"].assert_not_awaited()

    def test_a_matching_expected_sha_passes_through(self, client, repo):
        resp = client.put(
            "/prompts/kw/versions/1", json={**BODY, "expected_sha256": EXPECTED_PAYLOAD_SHA}
        )
        assert resp.status_code == 200
        assert repo["upsert"].await_args.kwargs["sha256"] == EXPECTED_PAYLOAD_SHA

    def test_written_true_when_inserted(self, client, repo):
        body = client.put("/prompts/kw/versions/1", json=BODY).json()
        assert body["written"] is True
        assert body["unchanged"] is False
        assert body["config_revision"] == "kw@1"

    def test_written_false_when_already_published_identically(self, client, repo):
        repo["upsert"].return_value = (_row(), False)
        body = client.put("/prompts/kw/versions/1", json=BODY).json()
        assert body["written"] is False
        assert body["unchanged"] is True

    def test_a_rewrite_is_409_not_a_silent_update(self, client, repo):
        repo["upsert"].side_effect = repository.PromptVersionConflict("kw", 1, "aa" * 32, "bb" * 32)
        resp = client.put("/prompts/kw/versions/1", json=BODY)
        assert resp.status_code == 409
        assert "Publish a new version instead" in resp.text

    def test_unknown_body_fields_are_refused(self, client):
        """extra='forbid': a typo'd field name must not be silently dropped —
        that is how a template edit ships as 'unchanged'."""
        assert (
            client.put("/prompts/kw/versions/1", json={**BODY, "sytsem": "oops"}).status_code == 422
        )

    def test_empty_system_is_refused(self, client):
        assert client.put("/prompts/kw/versions/1", json={**BODY, "system": ""}).status_code == 422

    def test_version_zero_is_not_a_version(self, client):
        assert client.put("/prompts/kw/versions/0", json=BODY).status_code == 422


class TestReads:
    def test_get_active_404s_when_nothing_is_activated(self, client, repo):
        repo["get_active"].return_value = None
        resp = client.get("/prompts/kw")
        assert resp.status_code == 404
        assert "versions/{version}" in resp.text or "/versions/" in resp.text

    def test_get_version_returns_the_row_with_its_revision_stamp(self, client):
        body = client.get("/prompts/kw/versions/1").json()
        assert body["config_revision"] == "kw@1"
        assert body["system_prompt_sha256"] == sha256_text(SYSTEM)

    def test_get_version_404(self, client, repo):
        repo["get_version"].return_value = None
        assert client.get("/prompts/kw/versions/9").status_code == 404

    @pytest.mark.parametrize("bad", ["../etc", "has space", "-leading", "a" * 200])
    def test_a_malformed_prompt_id_is_refused(self, client, bad, repo):
        resp = client.get(f"/prompts/{bad}")
        assert resp.status_code in (400, 404), resp.text
        if resp.status_code == 400:
            repo["get_active"].assert_not_awaited()

    def test_search_rejects_an_unknown_status(self, client, repo):
        assert client.get("/prompts/search", params={"status": "nonsense"}).status_code == 422
        repo["search"].assert_not_awaited()

    def test_search_defaults_to_active_only(self, client, repo):
        """The pre-031 default ``active`` now means PUBLISHED — what is live."""
        client.get("/prompts/search")
        assert repo["search"].await_args.kwargs["status"] == ["PUBLISHED"]

    def test_search_legacy_retired_means_everything_not_published(self, client, repo):
        client.get("/prompts/search", params={"status": "retired"})
        assert sorted(repo["search"].await_args.kwargs["status"]) == [
            "ARCHIVED",
            "CANARY",
            "DISCARDED",
            "DRAFT",
        ]

    @pytest.mark.parametrize("given", ["CANARY", "canary"])
    def test_search_filters_on_a_lifecycle_status(self, client, repo, given):
        assert client.get("/prompts/search", params={"status": given}).status_code == 200
        assert repo["search"].await_args.kwargs["status"] == ["CANARY"]

    def test_search_any_passes_no_status_filter(self, client, repo):
        client.get("/prompts/search", params={"status": "any"})
        assert repo["search"].await_args.kwargs["status"] is None

    def test_search_without_filters_is_todays_call_and_body(self, client, repo):
        """Omitted ``prompt_id``/``worker_name`` reach the repository as today's
        exact kwargs; the only new response key is the ``prompt_id`` echo."""
        repo["search"].return_value = [
            {
                "prompt_id": "kw",
                "version": 1,
                "sha256": "a" * 64,
                "system_prompt_sha256": "b" * 64,
                "status": "PUBLISHED",
                "config_revision": "kw@1",
                "rank": 0.0,
            }
        ]
        body = client.get("/prompts/search", params={"status": "any"}).json()
        assert set(repo["search"].await_args.kwargs) == {"q", "tag", "status", "limit"}
        assert set(body) == {"query", "tag", "status", "prompt_id", "count", "hits"}
        assert body["prompt_id"] is None
        assert body["hits"][0]["prompt_id"] == "kw"

    def test_search_echoes_the_applied_prompt_id(self, client, repo):
        body = client.get("/prompts/search", params={"prompt_id": "kw", "status": "any"}).json()
        assert body["prompt_id"] == "kw"

    def test_search_passes_worker_name_with_status_and_prompt_id(self, client, repo):
        client.get(
            "/prompts/search",
            params={"worker_name": "worker-a", "status": "PUBLISHED", "prompt_id": "kw"},
        )
        kwargs = repo["search"].await_args.kwargs
        assert (kwargs["worker_name"], kwargs["status"], kwargs["prompt_id"]) == (
            "worker-a",
            ["PUBLISHED"],
            "kw",
        )

    def test_search_worker_name_is_passed_verbatim_not_rewritten(self, client, repo):
        evil = "x' OR '1'='1"
        assert client.get("/prompts/search", params={"worker_name": evil}).status_code == 200
        assert repo["search"].await_args.kwargs["worker_name"] == evil

    def test_search_worker_name_is_length_capped(self, client, repo):
        assert client.get("/prompts/search", params={"worker_name": "w" * 256}).status_code == 422
        repo["search"].assert_not_awaited()

    def test_search_passes_prompt_id_with_status(self, client, repo):
        client.get("/prompts/search", params={"prompt_id": "kw", "status": "PUBLISHED"})
        kwargs = repo["search"].await_args.kwargs
        assert (kwargs["prompt_id"], kwargs["status"]) == ("kw", ["PUBLISHED"])

    def test_search_refuses_a_malformed_prompt_id(self, client, repo):
        assert client.get("/prompts/search", params={"prompt_id": "../x"}).status_code == 400
        repo["search"].assert_not_awaited()

    def test_search_limit_is_capped(self, client):
        assert client.get("/prompts/search", params={"limit": 5000}).status_code == 422


@pytest.mark.usefixtures("admin")
class TestActivate:
    def test_activate_reports_what_it_replaced(self, client, repo):
        repo["activate"].return_value = (_row(version=2, status="PUBLISHED"), 1, True)
        body = client.post("/prompts/kw/versions/2/activate", json=CARD_BODY).json()
        assert body["status"] == "PUBLISHED"
        assert body["previous_active_version"] == 1
        assert body["changed"] is True

    def test_activating_an_unpublished_version_is_404(self, client, repo):
        repo["activate"].return_value = (None, None, False)
        resp = client.post("/prompts/kw/versions/7/activate", json=CARD_BODY)
        assert resp.status_code == 404
        assert "publish it before activating" in resp.text

    def test_reactivating_the_same_version_reports_changed_false(self, client, repo):
        repo["activate"].return_value = (_row(status="PUBLISHED"), None, False)
        assert (
            client.post("/prompts/kw/versions/1/activate", json=CARD_BODY).json()["changed"]
            is False
        )

    def test_activating_a_discarded_version_is_409(self, client, repo):
        repo["activate"].side_effect = repository.InvalidStatusTransition(
            "kw", 1, "DISCARDED", "PUBLISHED"
        )
        assert client.post("/prompts/kw/versions/1/activate", json=CARD_BODY).status_code == 409


@pytest.fixture
def flag(monkeypatch):
    """Set PROMPTS_READ_REQUIRES_PUBLISHED for one test (settings are lru-cached)."""
    from src.settings import reset_settings_cache

    def _set(on: bool) -> None:
        monkeypatch.setenv("PROMPTS_READ_REQUIRES_PUBLISHED", "true" if on else "false")
        reset_settings_cache()

    yield _set
    monkeypatch.delenv("PROMPTS_READ_REQUIRES_PUBLISHED", raising=False)
    reset_settings_cache()


class TestUnpinnedReadFlag:
    """The route hands the flag to the resolver; the resolver's two orders are
    proved against real Postgres in the integration module."""

    def test_flag_off_asks_for_todays_resolution(self, client, repo, flag):
        flag(False)
        assert client.get("/prompts/kw").status_code == 200
        assert repo["get_active"].await_args.kwargs == {"highest_published": False}

    def test_flag_on_asks_for_the_highest_published(self, client, repo, flag):
        flag(True)
        assert client.get("/prompts/kw").status_code == 200
        assert repo["get_active"].await_args.kwargs == {"highest_published": True}


@pytest.mark.usefixtures("admin")
class TestStatusRoute:
    def test_a_permitted_move_reports_both_ends(self, client, repo):
        resp = client.post(
            "/prompts/kw/versions/1/status", json={"actor_card_id": CARD, "status": "CANARY"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "CANARY"
        assert body["previous_status"] == "DRAFT"
        assert body["config_revision"] == "kw@1"
        assert repo["transition"].await_args.args == ("kw", 1, "CANARY")

    def test_a_refused_move_is_409(self, client, repo):
        repo["transition"].side_effect = repository.InvalidStatusTransition(
            "kw", 1, "DRAFT", "PUBLISHED"
        )
        resp = client.post(
            "/prompts/kw/versions/1/status", json={"actor_card_id": CARD, "status": "PUBLISHED"}
        )
        assert resp.status_code == 409
        assert "DRAFT -> PUBLISHED" in resp.text

    def test_an_unpublished_version_is_404(self, client, repo):
        repo["transition"].return_value = (None, None)
        resp = client.post(
            "/prompts/kw/versions/9/status", json={"actor_card_id": CARD, "status": "CANARY"}
        )
        assert resp.status_code == 404

    @pytest.mark.parametrize("bad", ["active", "retired", "canary", "LIVE"])
    def test_an_unknown_status_is_422_and_never_reaches_the_ledger(self, client, repo, bad):
        resp = client.post("/prompts/kw/versions/1/status", json={"status": bad})
        assert resp.status_code == 422
        repo["transition"].assert_not_awaited()


class TestTransitionMatrix:
    """The matrix itself, exhaustively: every (from, to) pair of the five
    statuses, 25 cells, against the contract's list."""

    ALLOWED: ClassVar = {
        ("DRAFT", "CANARY"),
        ("DRAFT", "DISCARDED"),
        ("CANARY", "PUBLISHED"),
        ("CANARY", "DISCARDED"),
        ("PUBLISHED", "ARCHIVED"),
        ("PUBLISHED", "DISCARDED"),
        ("ARCHIVED", "PUBLISHED"),
        ("ARCHIVED", "DISCARDED"),
    }

    @pytest.mark.parametrize("src", repository.STATUSES)
    @pytest.mark.parametrize("dst", repository.STATUSES)
    def test_cell(self, src, dst):
        assert (dst in repository.TRANSITIONS[src]) == ((src, dst) in self.ALLOWED)


class TestFlagOffErrorTextsAreMains:
    """Flag off = main's responses byte-for-byte (only the status VALUES change)."""

    MAIN_404 = (
        "No ACTIVE version for prompt 'kw'. Published-but-not-activated versions are "
        "readable at /prompts/kw/versions/{version}."
    )

    def test_unpinned_404_flag_off_is_mains_exact_text(self, client, repo, flag):
        flag(False)
        repo["get_active"].return_value = None
        resp = client.get("/prompts/kw")
        assert resp.status_code == 404
        assert resp.json()["detail"] == self.MAIN_404

    def test_unpinned_404_flag_on_uses_the_lifecycle_wording(self, client, repo, flag):
        flag(True)
        repo["get_active"].return_value = None
        msg = client.get("/prompts/kw").json()["detail"]
        assert msg.startswith("No PUBLISHED version for prompt 'kw'.")

    def test_search_422_flag_off_is_mains_exact_text(self, client, repo, flag):
        flag(False)
        resp = client.get("/prompts/search", params={"status": "bogus"})
        assert resp.status_code == 422
        assert resp.json()["detail"] == "status must be one of active, retired, any"

    def test_search_422_flag_on_names_the_lifecycle_statuses(self, client, repo, flag):
        flag(True)
        msg = client.get("/prompts/search", params={"status": "bogus"}).json()["detail"]
        assert "published" in msg and "canary" in msg


class TestElevatedAuth:
    """DISCARDED (terminal) and the deprecated activate need ``service-internal``
    from a VERIFIED SA JWT; every other move keeps plain licence auth."""

    def test_discard_without_the_role_is_403_and_writes_nothing(self, client, repo, roles):
        resp = client.post("/prompts/kw/versions/1/status", json={"status": "DISCARDED"})
        assert resp.status_code == 403
        assert "requires_role:service-internal" in resp.text
        repo["transition"].assert_not_awaited()

    def test_a_licence_holders_other_roles_do_not_count(self, client, repo, roles):
        roles("license-admin", "onboarding-internal")
        resp = client.post("/prompts/kw/versions/1/status", json={"status": "DISCARDED"})
        assert resp.status_code == 403
        repo["transition"].assert_not_awaited()

    def test_discard_with_the_role_proceeds(self, client, repo, roles):
        roles("service-internal")
        repo["transition"].return_value = (_row(status="DISCARDED"), "DRAFT")
        resp = client.post("/prompts/kw/versions/1/status", json={"status": "DISCARDED"})
        assert resp.status_code == 200, resp.text
        assert repo["transition"].await_args.args == ("kw", 1, "DISCARDED")

    @pytest.mark.parametrize("target", ["ARCHIVED", "DRAFT"])
    def test_other_moves_need_no_role(self, client, repo, roles, target):
        resp = client.post(
            "/prompts/kw/versions/1/status", json={"status": target, "actor_card_id": CARD}
        )
        assert resp.status_code == 200, resp.text
        repo["transition"].assert_awaited_once()

    def test_activate_without_the_role_is_403_and_writes_nothing(self, client, repo, roles):
        resp = client.post("/prompts/kw/versions/1/activate", json=CARD_BODY)
        assert resp.status_code == 403
        repo["activate"].assert_not_awaited()

    def test_activate_with_the_role_proceeds(self, client, repo, roles):
        roles("service-internal")
        assert client.post("/prompts/kw/versions/1/activate", json=CARD_BODY).status_code == 200
        repo["activate"].assert_awaited_once()

    def test_publish_needs_no_role(self, client, repo, roles):
        assert client.put("/prompts/kw/versions/1", json=BODY).status_code == 200


class TestPlatformGate:
    """Prompts are SYSTEM assets (no tenant column): a customer licence can
    neither read nor write them (strict tenant separation, owner ruling
    2026-09-30), unless it carries a verified prompt SA role. Refused BEFORE
    the repository is reached."""

    FORBIDDEN: ClassVar[dict[str, Any]] = {"detail": {"error": "platform_prompt_forbidden"}}
    READS: ClassVar[list[str]] = [
        "/prompts/kw",
        "/prompts/kw/versions/1",
        "/prompts/search",
        "/prompts/search?status=any",
        "/prompts/kw/versions/1/transitions",
    ]

    def _assert_untouched(self, repo):
        repo["upsert"].assert_not_awaited()
        repo["transition"].assert_not_awaited()
        repo["activate"].assert_not_awaited()

    def test_customer_put_is_403_and_writes_nothing(self, client, repo, roles, customer):
        resp = client.put("/prompts/kw/versions/1", json=BODY)
        assert resp.status_code == 403, resp.text
        assert resp.json() == self.FORBIDDEN
        self._assert_untouched(repo)

    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED", "ARCHIVED", "DISCARDED"])
    def test_customer_status_move_is_403_and_writes_nothing(
        self, client, repo, roles, customer, target
    ):
        resp = client.post(
            "/prompts/kw/versions/1/status", json={"status": target, "actor_card_id": CARD}
        )
        assert resp.status_code == 403, resp.text
        assert resp.json() == self.FORBIDDEN
        self._assert_untouched(repo)

    def test_customer_initial_create_without_a_role_is_403(self, client, repo, roles, customer):
        body = {"actor_card_id": CARD, "status": "PUBLISHED", "reason": "initial_create"}
        resp = client.post("/prompts/kw/versions/1/status", json=body)
        assert resp.status_code == 403, resp.text
        assert resp.json() == self.FORBIDDEN
        self._assert_untouched(repo)

    def test_customer_activate_is_403_and_writes_nothing(self, client, repo, roles, customer):
        resp = client.post("/prompts/kw/versions/1/activate", json=CARD_BODY)
        assert resp.status_code == 403, resp.text
        assert resp.json() == self.FORBIDDEN
        self._assert_untouched(repo)

    def test_a_customers_unrelated_roles_do_not_count(self, client, repo, roles, customer):
        roles("license-admin", "onboarding-internal")
        assert client.put("/prompts/kw/versions/1", json=BODY).status_code == 403
        self._assert_untouched(repo)

    def test_system_tenant_put_is_403_and_writes_nothing(self, client, repo, roles):
        """The ``system`` tenant ``00000000-…-0001`` is commercial-LLM platform
        (inference-router parity) but NOT prompt-write platform: fr's list does
        not contain it (validator 5 T1, scenario B).

        MUTATION: gate ``_require_platform_caller`` on
        ``is_commercial_platform_tenant`` ⇒ the system tenant writes ⇒ red.
        """
        client.app.dependency_overrides[get_tenant_id] = lambda: SYSTEM_TENANT
        resp = client.put("/prompts/kw/versions/1", json=BODY)
        assert resp.status_code == 403, resp.text
        assert resp.json() == self.FORBIDDEN
        self._assert_untouched(repo)

    def test_platform_tenant_writes_as_before(self, client, repo, roles):
        assert client.put("/prompts/kw/versions/1", json=BODY).status_code == 200
        resp = client.post("/prompts/kw/versions/1/status", json={"status": "ARCHIVED"})
        assert resp.status_code == 200, resp.text
        repo["upsert"].assert_awaited_once()
        repo["transition"].assert_awaited_once()

    @pytest.mark.parametrize(
        "tenant",
        [
            "0195f48b-d000-7000-8000-000000000003",
            "019de842-c92a-7e37-8fdd-eb0d8c5396f9",
            "019DE842-C92A-7E37-8FDD-EB0D8C5396F9",
        ],
    )
    def test_every_platform_spelling_is_admitted(self, client, repo, roles, tenant):
        client.app.dependency_overrides[get_tenant_id] = lambda: tenant
        assert client.put("/prompts/kw/versions/1", json=BODY).status_code == 200

    def test_a_platform_tenant_still_needs_the_admin_role_to_activate(self, client, repo, roles):
        """The gate is layered UNDER the existing role gates, never instead."""
        resp = client.post("/prompts/kw/versions/1/activate", json=CARD_BODY)
        assert resp.status_code == 403
        assert "requires_role:service-internal" in resp.text
        repo["activate"].assert_not_awaited()

    def test_customer_with_bootstrap_role_can_still_initial_create(
        self, client, repo, roles, customer
    ):
        roles("prompt-asset-bootstrap")
        repo["transition"].return_value = (_row(status="PUBLISHED"), "DRAFT")
        body = {"actor_card_id": CARD, "status": "PUBLISHED", "reason": "initial_create"}
        resp = client.post("/prompts/kw/versions/1/status", json=body)
        assert resp.status_code == 200, resp.text
        assert repo["transition"].await_args.kwargs["reason"] == "initial_create"

    def test_customer_with_bootstrap_role_still_cannot_activate(
        self, client, repo, roles, customer
    ):
        roles("prompt-asset-bootstrap")
        resp = client.post("/prompts/kw/versions/1/activate", json=CARD_BODY)
        assert resp.status_code == 403
        assert "requires_role:service-internal" in resp.text
        repo["activate"].assert_not_awaited()

    def test_customer_with_admin_role_can_activate(self, client, repo, roles, customer):
        roles("service-internal")
        assert client.post("/prompts/kw/versions/1/activate", json=CARD_BODY).status_code == 200
        repo["activate"].assert_awaited_once()

    @pytest.mark.parametrize("path", READS)
    def test_customer_read_is_403_and_reads_nothing(self, client, repo, roles, customer, path):
        """Prompt text is system data: a customer never sees it, on any GET."""
        resp = client.get(path)
        assert resp.status_code == 403, resp.text
        assert resp.json() == self.FORBIDDEN
        for name in ("get_active", "get_version", "search", "get_transitions"):
            repo[name].assert_not_awaited()

    def test_a_customers_unrelated_roles_do_not_admit_a_read(self, client, repo, roles, customer):
        roles("license-admin", "onboarding-internal")
        assert client.get("/prompts/kw/versions/1").status_code == 403
        repo["get_version"].assert_not_awaited()

    @pytest.mark.parametrize("path", READS)
    def test_platform_tenant_reads(self, client, repo, roles, path):
        assert client.get(path).status_code == 200

    @pytest.mark.parametrize("role", ["prompt-asset-bootstrap", "service-internal"])
    @pytest.mark.parametrize("path", READS)
    def test_customer_with_a_prompt_role_can_read(self, client, repo, roles, customer, role, path):
        roles(role)
        assert client.get(path).status_code == 200


class TestPlatformTenantIds:
    def test_exactly_the_six_ids_mirrored_from_fr(self):
        from src.api.tenant_constants import PLATFORM_TENANT_IDS

        assert {
            "0195f48b-d000-7000-8000-000000000001",
            "0195f48b-d000-7000-8000-000000000002",
            "0195f48b-d000-7000-8000-000000000003",
            "0195f48b-d000-7000-8000-000000000004",
            "0195f48b-d000-7000-8000-000000000005",
            "019de842-c92a-7e37-8fdd-eb0d8c5396f9",
        } == PLATFORM_TENANT_IDS

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "n8n-dev",
            "0195f48b-d000-7000-8000-000000000099",
            "0195f48b-d000-7000-8000-000000000100",
            "00000000-0000-0000-0000-000000000000",
        ],
    )
    def test_names_customers_and_nil_are_not_platform(self, value):
        from src.api.tenant_constants import is_platform_tenant

        assert is_platform_tenant(value) is False


@pytest.mark.usefixtures("admin")
class TestMissing031Is503:
    """A failed 031 under the non-strict runner leaves the 030 CHECK: writes 503."""

    def test_every_write_route_is_503_and_reaches_no_ledger(self, client, repo, schema_state):
        schema_state(repository.SCHEMA_MISSING_031)
        put = client.put("/prompts/kw/versions/1", json=BODY)
        status = client.post(
            "/prompts/kw/versions/1/status", json={"actor_card_id": CARD, "status": "CANARY"}
        )
        act = client.post("/prompts/kw/versions/1/activate", json=CARD_BODY)
        for resp in (put, status, act):
            assert resp.status_code == 503, resp.text
            assert "prompt_lifecycle_schema_missing" in resp.text
        repo["upsert"].assert_not_awaited()
        repo["transition"].assert_not_awaited()
        repo["activate"].assert_not_awaited()

    def test_reads_are_unaffected(self, client, repo, schema_state):
        schema_state(repository.SCHEMA_MISSING_031)
        assert client.get("/prompts/kw").status_code == 200
        assert client.get("/prompts/kw/versions/1").status_code == 200
        assert client.get("/prompts/search").status_code == 200

    @pytest.mark.parametrize("state", [repository.SCHEMA_READY, repository.SCHEMA_UNKNOWN])
    def test_ready_or_unprobed_writes_proceed(self, client, repo, schema_state, state):
        schema_state(state)
        assert client.put("/prompts/kw/versions/1", json=BODY).status_code == 200

    def test_a_check_violation_is_503_not_500(self, client, repo, schema_state):
        """Unprobed pod (probe failed at startup) on a 030 schema: same 503."""
        import psycopg

        schema_state(repository.SCHEMA_UNKNOWN)
        repo["upsert"].side_effect = psycopg.errors.CheckViolation("prompt_versions_status_check")
        resp = client.put("/prompts/kw/versions/1", json=BODY)
        assert resp.status_code == 503, resp.text


class TestLegacyStatusMapping:
    """A row an OLD pod writes during the rollout reads as its lifecycle status."""

    @pytest.mark.parametrize(
        ("stored", "activated_at", "expected"),
        [
            ("active", "2026-09-28T00:00:00+00:00", "PUBLISHED"),
            ("active", None, "PUBLISHED"),
            ("retired", None, "DRAFT"),
            ("retired", "2026-09-28T00:00:00+00:00", "ARCHIVED"),
            ("CANARY", None, "CANARY"),
            ("DISCARDED", "2026-09-28T00:00:00+00:00", "DISCARDED"),
        ],
    )
    def test_effective_status(self, stored, activated_at, expected):
        assert repository.effective_status(stored, activated_at) == expected


# ---------------------------------------------------------------------------
# Migration 033: transition ledger, actor, initial_create
# ---------------------------------------------------------------------------


class TestInitialCreateRoute:
    """The HTTP half of ``reason=initial_create``; the history guard itself is
    proved against Postgres in tests/integration/test_prompt_transitions_ledger.py."""

    def test_without_the_role_is_403_and_never_reaches_the_repository(self, client, repo, roles):
        resp = client.post(
            "/prompts/kw/versions/1/status",
            json={"actor_card_id": CARD, "status": "PUBLISHED", "reason": "initial_create"},
            headers={"X-Caller-Roles": "service-internal"},
        )
        assert resp.status_code == 403
        assert "requires_role:service-internal" in resp.text
        repo["transition"].assert_not_awaited()

    def test_with_the_role_the_reason_and_actor_reach_the_repository(self, client, repo, roles):
        roles("service-internal")
        repo["transition"].return_value = (_row(status="PUBLISHED"), "DRAFT")
        resp = client.post(
            "/prompts/kw/versions/1/status",
            json={"actor_card_id": CARD, "status": "PUBLISHED", "reason": "initial_create"},
        )
        assert resp.status_code == 200, resp.text
        call = repo["transition"].await_args
        assert call.args == ("kw", 1, "PUBLISHED")
        assert call.kwargs["reason"] == "initial_create"
        assert call.kwargs["actor"].kind in {"service_account", "licence_subject", "unverified"}

    @pytest.mark.parametrize(
        "reason",
        [
            repository.INITIAL_CREATE_HISTORY,
            repository.INITIAL_CREATE_NOT_DRAFT,
            repository.INITIAL_CREATE_NOT_PUBLISH,
        ],
    )
    def test_a_refusal_is_409_with_a_top_level_body(self, client, repo, roles, reason):
        roles("service-internal")
        repo["transition"].side_effect = repository.InitialCreateNotAllowed("kw", 1, reason)
        resp = client.post(
            "/prompts/kw/versions/1/status",
            json={"actor_card_id": CARD, "status": "PUBLISHED", "reason": "initial_create"},
        )
        assert resp.status_code == 409
        assert resp.json() == {
            "error": "initial_create_not_allowed",
            "prompt_id": "kw",
            "version": 1,
            "reason": reason,
        }

    def test_an_ordinary_reason_needs_no_role(self, client, repo, roles):
        resp = client.post(
            "/prompts/kw/versions/1/status", json={"status": "ARCHIVED", "reason": "card:42"}
        )
        assert resp.status_code == 200, resp.text
        assert repo["transition"].await_args.kwargs["reason"] == "card:42"

    def test_a_reason_over_200_chars_is_422(self, client, repo):
        resp = client.post(
            "/prompts/kw/versions/1/status",
            json={"actor_card_id": CARD, "status": "CANARY", "reason": "r" * 201},
        )
        assert resp.status_code == 422
        repo["transition"].assert_not_awaited()

    @pytest.mark.parametrize(
        "reason",
        [
            "activate",
            "activate_superseded",
            "backfill_033_creation",
            "backfill_later",
            "migration",
            "",
            "card 42",
            "Card",
            "a/b",
            "x\n",
        ],
    )
    def test_a_reserved_or_malformed_reason_is_422(self, client, repo, reason):
        resp = client.post(
            "/prompts/kw/versions/1/status",
            json={"actor_card_id": CARD, "status": "CANARY", "reason": reason},
        )
        assert resp.status_code == 422, resp.text
        repo["transition"].assert_not_awaited()

    @pytest.mark.parametrize("reason", ["card:42", "fix-typo.v2", "r" * 200])
    def test_a_token_reason_is_accepted(self, client, repo, reason):
        resp = client.post(
            "/prompts/kw/versions/1/status", json={"status": "ARCHIVED", "reason": reason}
        )
        assert resp.status_code == 200, resp.text
        assert repo["transition"].await_args.kwargs["reason"] == reason

    def test_every_system_written_reason_is_reserved(self):
        """Drift guard: the reserved set covers what the code and 033 actually write."""
        from pathlib import Path

        from src.api.prompts import models

        mig = (
            Path(__file__).resolve().parents[1]
            / "migrations"
            / "033_prompt_version_transitions.sql"
        ).read_text()
        assert "'backfill_033_creation'" in mig
        written = {
            repository.REASON_ACTIVATE,
            repository.REASON_ACTIVATE_SUPERSEDED,
            repository.REASON_ADOPT_LIVE,
            "backfill_033_creation",
            "migration",
        }
        assert written <= models.RESERVED_TRANSITION_REASONS
        assert repository.INITIAL_CREATE not in models.RESERVED_TRANSITION_REASONS


class TestTransitionsRoute:
    def test_returns_the_rows(self, client, repo):
        repo["get_transitions"].return_value = [
            {
                "id": 1,
                "prompt_id": "kw",
                "version": 1,
                "from_status": None,
                "to_status": "DRAFT",
                "reason": None,
                "actor_sub": "sa",
                "actor_kind": "service_account",
                "at": "2026-09-28T00:00:00+00:00",
            }
        ]
        resp = client.get("/prompts/kw/versions/1/transitions")
        assert resp.status_code == 200, resp.text
        assert resp.json()["count"] == 1
        assert resp.json()["transitions"][0]["actor_sub"] == "sa"

    def test_absent_version_is_404(self, client, repo):
        repo["get_transitions"].return_value = None
        assert client.get("/prompts/kw/versions/9/transitions").status_code == 404

    def test_a_missing_ledger_is_503_not_500(self, client, repo):
        import psycopg

        repo["get_transitions"].side_effect = psycopg.errors.UndefinedTable(
            "prompt_version_transitions"
        )
        resp = client.get("/prompts/kw/versions/1/transitions")
        assert resp.status_code == 503, resp.text
        assert "prompt_transition_ledger_missing" in resp.text


@pytest.mark.usefixtures("admin")
class TestMissing033Is503:
    def test_every_write_route_is_503_on_a_missing_ledger(self, client, repo):
        import psycopg

        err = psycopg.errors.UndefinedTable("prompt_version_transitions")
        for key in ("upsert", "transition", "activate"):
            repo[key].side_effect = err
        put = client.put("/prompts/kw/versions/1", json=BODY)
        status = client.post(
            "/prompts/kw/versions/1/status", json={"actor_card_id": CARD, "status": "CANARY"}
        )
        act = client.post("/prompts/kw/versions/1/activate", json=CARD_BODY)
        for resp in (put, status, act):
            assert resp.status_code == 503, resp.text
            assert "prompt_transition_ledger_missing" in resp.text


class TestResolveActor:
    """The actor is the VERIFIED principal — never a header, never request.state.subject_id."""

    @staticmethod
    def _request(headers=None, **state):
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            "state": dict(state),
        }
        return Request(scope)

    SPOOF: ClassVar[dict[str, str]] = {
        "X-Subject-Id": "spoof",
        "X-Actor-Id": "spoof",
        "X-Caller-Client-Id": "spoof",
    }

    def test_a_verified_service_account_wins(self, monkeypatch):
        from src.api.prompts import actor

        monkeypatch.setattr(actor, "get_caller_client_id_from_context", lambda: "n8n-publisher")
        req = self._request(
            self.SPOOF, identity_proven=True, license={"subject_id": "lic"}, subject_id="spoof"
        )
        assert actor.resolve_actor(req) == actor.Actor("n8n-publisher", "service_account")

    def test_the_introspected_subject_not_the_forwarded_one(self, monkeypatch):
        from src.api.prompts import actor

        monkeypatch.setattr(actor, "get_caller_client_id_from_context", lambda: "")
        req = self._request(
            self.SPOOF, identity_proven=True, license={"subject_id": "lic"}, subject_id="spoof"
        )
        assert actor.resolve_actor(req) == actor.Actor("lic", "licence_subject")

    def test_an_unproven_request_is_unverified_whatever_it_claims(self, monkeypatch):
        from src.api.prompts import actor

        monkeypatch.setattr(actor, "get_caller_client_id_from_context", lambda: "")
        req = self._request(
            self.SPOOF,
            identity_proven=False,
            license={"subject_id": "lic"},
            subject_id="spoof",
            user={"sub": "spoof"},
        )
        assert actor.resolve_actor(req) == actor.Actor(None, "unverified")


class _CaptureConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.calls.append((sql, args))
        return []


def _captured_search_sql(**kwargs: Any) -> tuple[str, tuple[Any, ...]]:
    import asyncio

    conn = _CaptureConn()

    @asynccontextmanager
    async def _conn():
        yield conn

    with patch.object(repository, "get_pg_connection", _conn):
        asyncio.run(repository.search(q=None, tag=None, status=None, limit=20, **kwargs))
    (call,) = conn.calls
    return call


class TestSearchSql:
    """The SQL itself, captured: what a mocked repository cannot show."""

    NO_FILTER_SQL = (
        "SELECT prompt_id, version, sha256, system_prompt_sha256, question, purpose_tags, "
        "worker_names, status, published_at, activated_at, 0::float4 AS rank "
        "FROM prompt_versions WHERE TRUE "
        "ORDER BY rank DESC, prompt_id ASC, version DESC LIMIT $1"
    )

    def test_no_filter_query_is_unchanged(self):
        assert _captured_search_sql() == (self.NO_FILTER_SQL, (20,))

    def test_worker_name_is_a_bound_element_match(self):
        evil = "x' OR '1'='1"
        sql, args = _captured_search_sql(worker_name=evil)
        assert "WHERE TRUE AND $1 = ANY(worker_names) ORDER BY" in sql
        assert evil not in sql
        assert args == (evil, 20)


class TestPublishRoleGate:
    """U6 step 4: a move INTO CANARY or PUBLISHED needs a prompt role.

    With WM_PROMPT_SOURCE=published the highest PUBLISHED version of a bound
    prompt is a live worker's system prompt; a platform licence alone (the
    router gate) must not be able to put one there.
    """

    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
    def test_a_platform_licence_without_a_role_is_403_and_writes_nothing(
        self, client, repo, roles, target
    ):
        resp = client.post(
            "/prompts/kw/versions/1/status", json={"status": target, "actor_card_id": CARD}
        )
        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": {"error": "prompt_publish_role_required"}}
        repo["transition"].assert_not_awaited()

    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
    def test_unrelated_roles_do_not_count(self, client, repo, roles, target):
        roles("license-admin", "onboarding-internal")
        resp = client.post(
            "/prompts/kw/versions/1/status", json={"status": target, "actor_card_id": CARD}
        )
        assert resp.status_code == 403, resp.text
        repo["transition"].assert_not_awaited()

    @pytest.mark.parametrize("role", ["service-internal", "prompt-asset-bootstrap"])
    @pytest.mark.parametrize(("target", "previous"), [("CANARY", "DRAFT"), ("PUBLISHED", "CANARY")])
    def test_a_prompt_role_holder_moves(self, client, repo, roles, role, target, previous):
        roles(role)
        repo["transition"].return_value = (_row(status=target), previous)
        resp = client.post(
            "/prompts/kw/versions/1/status", json={"status": target, "actor_card_id": CARD}
        )
        assert resp.status_code == 200, resp.text
        assert repo["transition"].await_args.args == ("kw", 1, target)

    def test_archiving_still_needs_only_the_platform_gate(self, client, repo, roles):
        repo["transition"].return_value = (_row(status="ARCHIVED"), "PUBLISHED")
        resp = client.post("/prompts/kw/versions/1/status", json={"status": "ARCHIVED"})
        assert resp.status_code == 200, resp.text


class TestReleaseCard:
    """REALIGN R7 (coordinator ruling 2026-10-03): every move INTO CANARY or
    PUBLISHED names the Planka card authorising it — 19 ASCII digits — or is
    422 ``prompt_release_card_required`` before the repository is reached.
    Checked AFTER the role gates: a role-less caller still gets its 403.

    MUTATION: drop ``_require_release_card`` from either route ⇒ the
    missing/malformed cases reach the repository ⇒ red. Stop passing
    ``actor_card_id`` to the repository ⇒ the "valid" cases are red.
    """

    REQUIRED: ClassVar[dict[str, Any]] = {"detail": {"error": "prompt_release_card_required"}}
    MALFORMED: ClassVar[list[Any]] = [
        "",
        "123456789012345678",  # 18 digits
        "12345678901234567890",  # 20 digits
        "12345678901234567a9",  # pragma: allowlist secret
        "abcdefghijklmnopqrs",
        " 1234567890123456789",
        "1234567890123456789\n",
        "\uff11" * 19,  # FULLWIDTH DIGIT ONE x19: \d would admit it
        "-123456789012345678",
    ]

    def _status(self, client, target: str, **extra: Any):
        return client.post("/prompts/kw/versions/1/status", json={"status": target, **extra})

    @pytest.mark.usefixtures("admin")
    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
    def test_a_release_without_a_card_is_422_and_writes_nothing(self, client, repo, target):
        resp = self._status(client, target)
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED
        repo["transition"].assert_not_awaited()

    @pytest.mark.usefixtures("admin")
    def test_a_null_card_is_422(self, client, repo):
        resp = self._status(client, "CANARY", actor_card_id=None)
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED
        repo["transition"].assert_not_awaited()

    @pytest.mark.usefixtures("admin")
    def test_initial_create_without_a_card_is_422(self, client, repo):
        resp = self._status(client, "PUBLISHED", reason="initial_create")
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED
        repo["transition"].assert_not_awaited()

    @pytest.mark.usefixtures("admin")
    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
    @pytest.mark.parametrize("bad", MALFORMED)
    def test_a_malformed_card_is_422_and_writes_nothing(self, client, repo, target, bad):
        resp = self._status(client, target, actor_card_id=bad)
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED
        repo["transition"].assert_not_awaited()

    @pytest.mark.usefixtures("admin")
    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
    def test_a_valid_card_reaches_the_ledger(self, client, repo, target):
        repo["transition"].return_value = (_row(status=target), "DRAFT")
        resp = self._status(client, target, actor_card_id=CARD)
        assert resp.status_code == 200, resp.text
        assert repo["transition"].await_args.kwargs["actor_card_id"] == CARD

    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
    def test_a_role_less_caller_still_gets_403_first(self, client, repo, roles, target):
        """No card AND no role: the role refusal wins (403, not 422)."""
        resp = self._status(client, target)
        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": {"error": "prompt_publish_role_required"}}
        repo["transition"].assert_not_awaited()

    def test_a_role_less_activate_still_gets_403_first(self, client, repo, roles):
        resp = client.post("/prompts/kw/versions/1/activate")
        assert resp.status_code == 403, resp.text
        repo["activate"].assert_not_awaited()

    @pytest.mark.parametrize("target", ["ARCHIVED", "DRAFT"])
    def test_moves_out_need_no_card(self, client, repo, roles, target):
        resp = self._status(client, target)
        assert resp.status_code == 200, resp.text
        assert repo["transition"].await_args.kwargs["actor_card_id"] is None

    @pytest.mark.usefixtures("admin")
    def test_discard_needs_no_card(self, client, repo):
        repo["transition"].return_value = (_row(status="DISCARDED"), "DRAFT")
        assert self._status(client, "DISCARDED").status_code == 200

    @pytest.mark.usefixtures("admin")
    def test_activate_without_a_body_is_422_and_writes_nothing(self, client, repo):
        resp = client.post("/prompts/kw/versions/1/activate")
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED
        repo["activate"].assert_not_awaited()

    @pytest.mark.usefixtures("admin")
    @pytest.mark.parametrize("bad", MALFORMED)
    def test_activate_with_a_malformed_card_is_422(self, client, repo, bad):
        resp = client.post("/prompts/kw/versions/1/activate", json={"actor_card_id": bad})
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED
        repo["activate"].assert_not_awaited()

    @pytest.mark.usefixtures("admin")
    def test_activate_passes_the_card_to_the_ledger(self, client, repo):
        resp = client.post("/prompts/kw/versions/1/activate", json=CARD_BODY)
        assert resp.status_code == 200, resp.text
        assert repo["activate"].await_args.kwargs["actor_card_id"] == CARD

    def test_the_ledger_listing_returns_the_card(self, client, repo):
        repo["get_transitions"].return_value = [
            {
                "id": 2,
                "prompt_id": "kw",
                "version": 1,
                "from_status": "DRAFT",
                "to_status": "CANARY",
                "reason": None,
                "actor_sub": "sa",
                "actor_kind": "service_account",
                "actor_card_id": CARD,
                "at": "2026-10-03T00:00:00+00:00",
            }
        ]
        resp = client.get("/prompts/kw/versions/1/transitions")
        assert resp.status_code == 200, resp.text
        assert resp.json()["transitions"][0]["actor_card_id"] == CARD

    def test_a_missing_039_column_is_503_not_500(self, client, repo):
        import psycopg

        repo["get_transitions"].side_effect = psycopg.errors.UndefinedColumn(
            'column "actor_card_id" does not exist'
        )
        resp = client.get("/prompts/kw/versions/1/transitions")
        assert resp.status_code == 503, resp.text
        assert "prompt_release_card_schema_missing" in resp.text

    @staticmethod
    def _check_violation(constraint: str):
        """A CheckViolation whose ``diag.constraint_name`` is ``constraint``
        (psycopg's ``diag`` is read-only, built from a server result)."""
        import psycopg

        class _Named(psycopg.errors.CheckViolation):
            @property
            def diag(self):  # type: ignore[override]
                return SimpleNamespace(constraint_name=constraint)

        return _Named(f'violates check constraint "{constraint}"')

    @pytest.mark.usefixtures("admin")
    def test_the_card_backstop_violation_is_422(self, client, repo):
        repo["transition"].side_effect = self._check_violation(repository.RELEASE_CARD_CONSTRAINT)
        resp = self._status(client, "CANARY", actor_card_id=CARD)
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED

    @pytest.mark.usefixtures("admin")
    @pytest.mark.parametrize("route", ["status", "activate"])
    def test_another_check_violation_is_not_the_card_422(self, client, repo, route):
        """Only 039's constraint maps to the card 422; any other CHECK (031's
        status CHECK on a pre-031 schema) keeps the lifecycle 503.

        MUTATION: map every CheckViolation to the card 422 ⇒ red.
        """
        exc = self._check_violation("prompt_versions_status_check")
        if route == "status":
            repo["transition"].side_effect = exc
            resp = self._status(client, "CANARY", actor_card_id=CARD)
        else:
            repo["activate"].side_effect = exc
            resp = client.post("/prompts/kw/versions/1/activate", json=CARD_BODY)
        assert resp.status_code == 503, resp.text
        assert "prompt_lifecycle_schema_missing" in resp.text
        assert "prompt_release_card_required" not in resp.text

    def test_the_openapi_documents_the_card(self, client):
        schemas = client.app.openapi()["components"]["schemas"]
        for name in ("PromptStatusChange", "PromptActivateRequest", "PromptTransition"):
            assert "actor_card_id" in schemas[name]["properties"], name
        assert schemas["PromptStatusChange"]["properties"]["actor_card_id"]["pattern"] == (
            "^[0-9]{19}$"
        )
