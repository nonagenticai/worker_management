"""Migration 033 — the prompt transition ledger and the sanctioned ``initial_create``.

Against a real PostgreSQL 18, through the real router and repository:

* every write path appends exactly one ledger row per status change, in the
  same transaction, with the actor taken from the VERIFIED principal
  (SA ``azp``, else the introspected licence subject) — never from a header;
* ``POST …/status {"status":"PUBLISHED","reason":"initial_create"}`` publishes
  a DRAFT of a prompt with no PUBLISHED/ARCHIVED history, needs
  ``service-internal``, and is refused 409 (top-level body) on history found in
  the rows OR in the ledger;
* a plain DRAFT -> PUBLISHED is still the matrix's 409;
* the ledger is append-only.

Run locally::

    export DATABASE_URL="postgresql+psycopg://user@127.0.0.1:PORT/worker_management_ci"
    export POSTGRES_SSLMODE=disable
    uv run python scripts/init_db.py
    uv run pytest tests/integration/test_prompt_transitions_ledger.py -m pg -v --no-cov
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import psycopg
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.auth import get_tenant_id
from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool

from .pg_guard import (
    PG_CONNECT_ERRORS,
    apply_migrations,
    pg_conninfo,
    require_database_url,
    unreachable_pg,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

pytestmark = pytest.mark.pg

# How long the gated history check waits for the other initial_create.
_RACE_WINDOW = 2.0

SA = "n8n-prompt-publisher"
# A Planka card id: REALIGN R7 requires one on every move into CANARY/PUBLISHED.
CARD = "1234567890123456789"
CARD_BODY: dict[str, Any] = {"actor_card_id": CARD}
PLATFORM_TENANT = "aaaaaaaa-0000-4000-8000-000000000001"
LICENCE_SUBJECT = "aaaaaaaa-0000-7000-8000-00000000abcd"
SPOOF = "spoofed-actor"
# Every header an attacker could use to name an actor. None may reach the ledger.
SPOOF_HEADERS = {
    "X-Subject-Id": SPOOF,
    "X-Actor-Id": SPOOF,
    "X-Caller-Client-Id": SPOOF,
    "X-Caller-Roles": "service-internal",
    "X-Tenant-Name": SPOOF,
}


def _body(n: int = 1) -> dict[str, Any]:
    return {
        "system": f"You are a keyword picker, revision {n}.",
        "question": "Which keywords?",
        "worker_names": ["worker-decide-keywords"],
        "purpose_tags": ["keywords"],
    }


@pytest_asyncio.fixture(scope="module")
async def pg_pool() -> AsyncIterator[None]:
    require_database_url()
    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    await apply_migrations()
    try:
        yield
    finally:
        await close_pg_pool()


async def _purge(pid: str) -> None:
    """Delete a test prompt and its ledger rows.

    The append-only triggers have NO opt-out. Teardown disables them for its
    own transaction only (``DISABLE TRIGGER USER`` is transactional, so a
    failure rolls the disable back too). That needs table OWNERSHIP, which
    already implies the power to drop the table: no bypass is added.
    """
    async with get_pg_connection() as conn, conn.transaction():
        await conn.execute("ALTER TABLE prompt_version_transitions DISABLE TRIGGER USER")
        await conn.execute("DELETE FROM prompt_version_transitions WHERE prompt_id = $1", pid)
        await conn.execute("ALTER TABLE prompt_version_transitions ENABLE TRIGGER USER")
        await conn.execute("DELETE FROM prompt_versions WHERE prompt_id = $1", pid)
        await conn.execute("DELETE FROM prompt_owner WHERE prompt_id = $1", pid)


@pytest_asyncio.fixture
async def prompt_id(pg_pool) -> AsyncIterator[str]:
    pid = f"test.ledger-{uuid.uuid4().hex[:12]}"
    try:
        yield pid
    finally:
        await _purge(pid)


class Principal:
    """What the (simulated) auth middleware verified for the next request."""

    def __init__(self) -> None:
        self.sa: str = SA
        self.roles: tuple[str, ...] = ("service-internal",)
        self.licence_subject: str | None = None

    def service_account(self, name: str = SA, *, admin: bool = True) -> None:
        self.sa, self.licence_subject = name, None
        self.roles = ("service-internal",) if admin else ()

    def licence(self, subject: str = LICENCE_SUBJECT) -> None:
        self.sa, self.roles, self.licence_subject = "", (), subject

    def nobody(self) -> None:
        self.sa, self.roles, self.licence_subject = "", (), None


@pytest.fixture
def principal() -> Principal:
    return Principal()


class FakeFr:
    """fr's card lookup behind an httpx MockTransport (worker_management#336).

    The REAL ``release_card.verify_release_card`` runs; only the network is
    replaced. Default answer: the card is a platform card (fr 200).
    """

    def __init__(self) -> None:
        self.calls = 0
        self.answer: Callable[[httpx.Request], httpx.Response] = self.platform

    @staticmethod
    def platform(request: httpx.Request) -> httpx.Response:
        card = request.url.path.split("/")[-2]
        return httpx.Response(
            200, json={"card_id": card, "tenant_id": PLATFORM_TENANT, "platform": True}
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self.answer(request)


@pytest.fixture
def fr(monkeypatch) -> Iterator[FakeFr]:
    from src.api.prompts import release_card
    from src.settings import reset_settings_cache

    fake = FakeFr()
    monkeypatch.setenv("FR_CARD_LOOKUP_URL", "http://fr.test:8888")
    monkeypatch.setenv("FR_WM_CARD_LOOKUP_TOKEN", "t" * 64)
    reset_settings_cache()
    release_card.clear_cache()
    release_card.set_transport_for_tests(httpx.MockTransport(fake.handle))
    try:
        yield fake
    finally:
        release_card.set_transport_for_tests(None)
        release_card.clear_cache()
        monkeypatch.undo()
        reset_settings_cache()


@pytest.fixture
def client(pg_pool, monkeypatch, principal: Principal, fr: FakeFr) -> Iterator[TestClient]:
    from src.api.prompts import actor
    from src.api.prompts.routes import router
    from src.auth import caller_identity

    # The verified SA identity and its roles: the two ContextVar readers are
    # module globals, patched here (a ContextVar would not reach the
    # TestClient's worker thread).
    monkeypatch.setattr(caller_identity, "get_caller_roles_from_context", lambda: principal.roles)
    monkeypatch.setattr(actor, "get_caller_client_id_from_context", lambda: principal.sa)

    app = FastAPI()
    # A PLATFORM tenant: every prompt route, reads included, is platform-only (or
    # needs a prompt SA role), and the licence/nobody principals below carry no
    # role. The gate itself is proved in tests/test_prompt_routes_unit.py
    # (TestPlatformGate).
    app.dependency_overrides[get_tenant_id] = lambda: "aaaaaaaa-0000-4000-8000-000000000001"

    @app.middleware("http")
    async def _stamp(request: Request, call_next):
        # What AuthenticationMiddleware._stamp_identity leaves on the token
        # branch. On the delegation branch it also overwrites
        # request.state.subject_id from X-Subject-Id — mirrored here so a test
        # can prove the ledger does NOT read that slot.
        if principal.licence_subject is not None:
            request.state.identity_proven = True
            request.state.license = {"active": True, "subject_id": principal.licence_subject}
        request.state.subject_id = request.headers.get("X-Subject-Id")
        request.state.user = {"sub": request.headers.get("X-Tenant-Name")}
        return await call_next(request)

    app.include_router(router)

    @asynccontextmanager
    async def _noop(_a):
        yield

    app.router.lifespan_context = _noop
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _ledger(client: TestClient, pid: str, v: int) -> list[dict[str, Any]]:
    resp = client.get(f"/prompts/{pid}/versions/{v}/transitions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == len(body["transitions"])
    return body["transitions"]


def _moves(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [(r["from_status"], r["to_status"], r["reason"]) for r in rows]


def _put(client: TestClient, pid: str, v: int, **kw: Any) -> None:
    resp = client.put(f"/prompts/{pid}/versions/{v}", json=_body(v), **kw)
    assert resp.status_code == 200, resp.text


def _status(
    client: TestClient,
    pid: str,
    v: int,
    status: str,
    reason: str | None = None,
    card: str | None = None,
    **kw,
):
    """POST …/status. A CANARY/PUBLISHED target carries :data:`CARD` unless
    ``card`` overrides it (REALIGN R7); other targets carry none."""
    body: dict[str, Any] = {"status": status}
    if card is not None:
        body["actor_card_id"] = card
    elif status in ("CANARY", "PUBLISHED"):
        body["actor_card_id"] = CARD
    if reason is not None:
        body["reason"] = reason
    return client.post(f"/prompts/{pid}/versions/{v}/status", json=body, **kw)


# ---------------------------------------------------------------------------
# Every write path writes exactly one row per status change
# ---------------------------------------------------------------------------


class TestEveryPathWritesTheLedger:
    def test_put_create_writes_null_to_draft_and_a_reput_writes_nothing(
        self, client, prompt_id, principal
    ):
        _put(client, prompt_id, 1)
        _put(client, prompt_id, 1)  # idempotent re-PUT: no change, no row
        rows = _ledger(client, prompt_id, 1)
        assert _moves(rows) == [(None, "DRAFT", None)]
        assert (rows[0]["actor_sub"], rows[0]["actor_kind"]) == (SA, "service_account")

    def test_status_moves_record_reason_and_the_licence_subject(self, client, prompt_id, principal):
        _put(client, prompt_id, 1)
        # CANARY / PUBLISHED need a prompt role (U6 step 4): the SA moves in,
        # a licence subject may still move a version OUT (ARCHIVED).
        assert _status(client, prompt_id, 1, "CANARY", "canary:card-42").status_code == 200
        assert _status(client, prompt_id, 1, "PUBLISHED").status_code == 200
        principal.licence()
        assert _status(client, prompt_id, 1, "ARCHIVED", "superseded").status_code == 200
        principal.service_account()
        assert _status(client, prompt_id, 1, "DISCARDED", "bad_prompt").status_code == 200
        rows = _ledger(client, prompt_id, 1)
        assert _moves(rows) == [
            (None, "DRAFT", None),
            ("DRAFT", "CANARY", "canary:card-42"),
            ("CANARY", "PUBLISHED", None),
            ("PUBLISHED", "ARCHIVED", "superseded"),
            ("ARCHIVED", "DISCARDED", "bad_prompt"),
        ]
        kinds = [(r["actor_kind"], r["actor_sub"]) for r in rows]
        assert kinds[1:3] == [("service_account", SA)] * 2
        assert kinds[3] == ("licence_subject", LICENCE_SUBJECT)
        assert kinds[4] == ("service_account", SA)

    def test_a_refused_move_writes_nothing(self, client, prompt_id):
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, "ARCHIVED").status_code == 409
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    def test_activate_records_the_publish_and_every_archive_it_causes(self, client, prompt_id):
        for v in (1, 2, 3):
            _put(client, prompt_id, v)
        assert (
            client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
            == 200
        )
        assert (
            client.post(f"/prompts/{prompt_id}/versions/3/activate", json=CARD_BODY).status_code
            == 200
        )
        # re-activating the live one moves nothing and records nothing
        assert (
            client.post(f"/prompts/{prompt_id}/versions/3/activate", json=CARD_BODY).json()[
                "changed"
            ]
            is False
        )
        assert _moves(_ledger(client, prompt_id, 1)) == [
            (None, "DRAFT", None),
            ("DRAFT", "PUBLISHED", "activate"),
            ("PUBLISHED", "ARCHIVED", "activate_superseded"),
        ]
        assert _moves(_ledger(client, prompt_id, 2)) == [(None, "DRAFT", None)]
        rows3 = _ledger(client, prompt_id, 3)
        assert _moves(rows3) == [(None, "DRAFT", None), ("DRAFT", "PUBLISHED", "activate")]
        assert {r["actor_sub"] for r in rows3} == {SA}

    def test_reason_over_200_chars_is_422_and_writes_nothing(self, client, prompt_id):
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, "CANARY", "x" * 201).status_code == 422
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    @pytest.mark.parametrize(
        "reason",
        [
            "activate",
            "activate_superseded",
            "backfill_033_creation",
            "backfill_034_anything",
            "migration",
            "",
            "initial create",
            "INITIAL_CREATE",
            "card 42",
        ],
    )
    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
    def test_reserved_or_malformed_reason_is_422_and_writes_nothing(
        self, client, prompt_id, reason, target
    ):
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, target, reason).status_code == 422
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    def test_transitions_of_an_absent_version_is_404(self, client, prompt_id):
        assert client.get(f"/prompts/{prompt_id}/versions/7/transitions").status_code == 404


# ---------------------------------------------------------------------------
# The actor is the verified principal — never a header
# ---------------------------------------------------------------------------


class TestActorIsNeverAHeader:
    def test_headers_cannot_name_the_actor_of_a_service_account_call(
        self, client, prompt_id, principal
    ):
        _put(client, prompt_id, 1, headers=SPOOF_HEADERS)
        rows = _ledger(client, prompt_id, 1)
        assert (rows[0]["actor_sub"], rows[0]["actor_kind"]) == (SA, "service_account")

    def test_x_subject_id_does_not_replace_the_introspected_subject(
        self, client, prompt_id, principal
    ):
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, "CANARY").status_code == 200
        assert _status(client, prompt_id, 1, "PUBLISHED").status_code == 200
        principal.licence()
        # A spoofed X-Caller-Roles header does not satisfy the publish-role gate.
        refused = _status(client, prompt_id, 1, "PUBLISHED", headers=SPOOF_HEADERS)
        assert refused.status_code == 403, refused.text
        assert _status(client, prompt_id, 1, "ARCHIVED", headers=SPOOF_HEADERS).status_code == 200
        row = _ledger(client, prompt_id, 1)[-1]
        assert (row["actor_sub"], row["actor_kind"]) == (LICENCE_SUBJECT, "licence_subject")

    def test_an_unverified_caller_is_recorded_as_nobody_whatever_it_claims(
        self, client, prompt_id, principal
    ):
        principal.nobody()
        _put(client, prompt_id, 1, headers=SPOOF_HEADERS)
        row = _ledger(client, prompt_id, 1)[0]
        assert (row["actor_sub"], row["actor_kind"]) == (None, "unverified")

    def test_the_body_cannot_carry_an_actor(self, client, prompt_id):
        _put(client, prompt_id, 1)
        resp = client.post(
            f"/prompts/{prompt_id}/versions/1/status",
            json={"status": "CANARY", "actor_sub": SPOOF},
        )
        assert resp.status_code == 422
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]


# ---------------------------------------------------------------------------
# initial_create
# ---------------------------------------------------------------------------


def _refused(resp, pid: str, version: int, reason: str = "prompt_has_publish_history") -> None:
    """The FULL 409 body: exact key set AND values.

    Same top-level shape as the other pin errors (``src/api/workers/prompt_pin.py``:
    ``{"error", "prompt_id", "version"}``) plus ``reason``.
    """
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert set(body) == {"error", "prompt_id", "version", "reason"}, body
    assert body == {
        "error": "initial_create_not_allowed",
        "prompt_id": pid,
        "version": version,
        "reason": reason,
    }


def _raw(query: str, *args: Any) -> None:
    """Out-of-band SQL on its own libpq connection (``%s`` placeholders), autocommit."""
    with psycopg.connect(pg_conninfo(), autocommit=True) as conn:
        conn.execute(query, args)


@pytest.fixture
def flag(monkeypatch):
    from src.settings import reset_settings_cache

    def _set(on: bool) -> None:
        monkeypatch.setenv("PROMPTS_READ_REQUIRES_PUBLISHED", "true" if on else "false")
        reset_settings_cache()

    yield _set
    monkeypatch.delenv("PROMPTS_READ_REQUIRES_PUBLISHED", raising=False)
    reset_settings_cache()


class TestInitialCreate:
    @pytest.mark.parametrize("flag_on", [False, True])
    def test_succeeds_on_a_fresh_prompt_under_either_flag(self, client, prompt_id, flag, flag_on):
        flag(flag_on)
        _put(client, prompt_id, 1)
        resp = _status(client, prompt_id, 1, "PUBLISHED", "initial_create")
        assert resp.status_code == 200, resp.text
        assert (resp.json()["previous_status"], resp.json()["status"]) == ("DRAFT", "PUBLISHED")
        assert resp.json()["activated_at"] is not None
        rows = _ledger(client, prompt_id, 1)
        assert _moves(rows) == [(None, "DRAFT", None), ("DRAFT", "PUBLISHED", "initial_create")]
        assert (rows[1]["actor_sub"], rows[1]["actor_kind"]) == (SA, "service_account")
        live = client.get(f"/prompts/{prompt_id}")
        assert live.status_code == 200
        assert live.json()["version"] == 1

    def test_refused_when_the_prompt_has_published_history(self, client, prompt_id):
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, "PUBLISHED", "initial_create").status_code == 200
        _put(client, prompt_id, 2)
        _refused(_status(client, prompt_id, 2, "PUBLISHED", "initial_create"), prompt_id, 2)
        assert client.get(f"/prompts/{prompt_id}/versions/2").json()["status"] == "DRAFT"
        assert _moves(_ledger(client, prompt_id, 2)) == [(None, "DRAFT", None)]

    def test_refused_when_the_only_history_is_archived(self, client, prompt_id):
        for v in (1, 2, 3):
            _put(client, prompt_id, v)
        assert (
            client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
            == 200
        )
        assert (
            client.post(f"/prompts/{prompt_id}/versions/2/activate", json=CARD_BODY).status_code
            == 200
        )
        assert _status(client, prompt_id, 2, "DISCARDED").status_code == 200
        statuses = {
            v: client.get(f"/prompts/{prompt_id}/versions/{v}").json()["status"] for v in (1, 2, 3)
        }
        assert statuses == {1: "ARCHIVED", 2: "DISCARDED", 3: "DRAFT"}  # nothing PUBLISHED
        _refused(_status(client, prompt_id, 3, "PUBLISHED", "initial_create"), prompt_id, 3)

    def test_history_only_in_the_rows_is_enough(self, client, prompt_id):
        """Pre-ledger history: a row ARCHIVED with no ledger row saying so."""
        _put(client, prompt_id, 1)
        _put(client, prompt_id, 2)
        _raw(
            "UPDATE prompt_versions SET status = 'ARCHIVED' WHERE prompt_id = %s AND version = 1",
            prompt_id,
        )
        assert all(r["to_status"] == "DRAFT" for r in _ledger(client, prompt_id, 1))
        _refused(_status(client, prompt_id, 2, "PUBLISHED", "initial_create"), prompt_id, 2)

    def test_history_only_in_the_ledger_is_enough(self, client, prompt_id):
        """The rows no longer show it (reset by hand); the ledger still does."""
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, "PUBLISHED", "initial_create").status_code == 200
        _raw(
            "UPDATE prompt_versions SET status = 'DRAFT', activated_at = NULL WHERE prompt_id = %s",
            prompt_id,
        )
        _put(client, prompt_id, 2)
        _refused(_status(client, prompt_id, 2, "PUBLISHED", "initial_create"), prompt_id, 2)

    def test_without_service_internal_is_403_and_writes_nothing(self, client, prompt_id, principal):
        _put(client, prompt_id, 1)
        principal.service_account(admin=False)
        resp = _status(client, prompt_id, 1, "PUBLISHED", "initial_create", headers=SPOOF_HEADERS)
        assert resp.status_code == 403
        principal.licence()
        assert _status(client, prompt_id, 1, "PUBLISHED", "initial_create").status_code == 403
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["status"] == "DRAFT"
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    def test_the_bootstrap_sa_initial_creates_and_the_ledger_names_it(
        self, client, prompt_id, principal
    ):
        """ncn#1780: ``claude-asset-bootstrap`` carries ONLY ``prompt-asset-bootstrap``."""
        principal.sa, principal.roles = "claude-asset-bootstrap", ("prompt-asset-bootstrap",)
        _put(client, prompt_id, 1)
        resp = _status(client, prompt_id, 1, "PUBLISHED", "initial_create")
        assert resp.status_code == 200, resp.text
        # It cannot discard or activate what it created.
        assert _status(client, prompt_id, 1, "DISCARDED").status_code == 403
        assert (
            client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
            == 403
        )
        rows = _ledger(client, prompt_id, 1)
        assert _moves(rows) == [(None, "DRAFT", None), ("DRAFT", "PUBLISHED", "initial_create")]
        assert {(r["actor_sub"], r["actor_kind"]) for r in rows} == {
            ("claude-asset-bootstrap", "service_account")
        }

    @pytest.mark.parametrize("reason", [None, "promote", "initial_create_v2"])
    def test_any_other_draft_to_published_is_still_the_matrix_409(self, client, prompt_id, reason):
        _put(client, prompt_id, 1)
        resp = _status(client, prompt_id, 1, "PUBLISHED", reason)
        assert resp.status_code == 409
        assert "DRAFT -> PUBLISHED" in resp.text
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    def test_initial_create_from_a_non_draft_is_refused(self, client, prompt_id):
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, "CANARY").status_code == 200
        _refused(
            _status(client, prompt_id, 1, "PUBLISHED", "initial_create"),
            prompt_id,
            1,
            "version_is_not_draft",
        )

    def test_initial_create_targets_published_only(self, client, prompt_id):
        _put(client, prompt_id, 1)
        _refused(
            _status(client, prompt_id, 1, "CANARY", "initial_create"),
            prompt_id,
            1,
            "initial_create_targets_published_only",
        )


# ---------------------------------------------------------------------------
# Concurrency: two initial_creates of ONE prompt serialise
# ---------------------------------------------------------------------------


async def _draft(pid: str, v: int) -> None:
    from src.api.prompts import repository
    from src.api.prompts.actor import Actor

    await repository.upsert_version(
        prompt_id=pid,
        version=v,
        sha256=f"{v:064x}",
        system_prompt_sha256=f"{v:064x}",
        question="Which keywords?",
        system=f"You are a keyword picker, revision {v}.",
        template=None,
        output_schema=None,
        worker_names=["worker-decide-keywords"],
        purpose_tags=["keywords"],
        git_sha=None,
        actor=Actor(sub=SA, kind="service_account"),
        owner_tenant_id="aaaaaaaa-0000-4000-8000-000000000001",
    )


class TestInitialCreateConcurrency:
    @pytest.mark.asyncio
    async def test_two_concurrent_initial_creates_one_wins_one_is_409(self, prompt_id, monkeypatch):
        """Two DRAFTs of ONE prompt, both initial_created at once, on two pool
        connections. Exactly one may publish; the other must see its history.

        The history check is held open until BOTH calls have reached it (or
        ``_RACE_WINDOW`` passes). Without the ``FOR UPDATE`` on every row of the
        prompt, both calls reach it together, both see no history, and both
        publish — deterministically, not flakily. With the lock, the second call
        blocks on ``FOR UPDATE`` until the first commits, reaches the check alone
        after the window, and sees the first call's PUBLISHED row.
        """
        from src.api.prompts import repository
        from src.api.prompts.actor import Actor

        await _draft(prompt_id, 1)
        await _draft(prompt_id, 2)

        real = repository._has_publish_history
        arrived: list[int] = []
        both_in = asyncio.Event()

        async def gated(conn: Any, pid: str) -> bool:
            arrived.append(1)
            if len(arrived) >= 2:
                both_in.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(both_in.wait(), _RACE_WINDOW)
            return await real(conn, pid)

        monkeypatch.setattr(repository, "_has_publish_history", gated)
        actor = Actor(sub=SA, kind="service_account")

        async def create(v: int) -> str:
            try:
                row, prev = await repository.transition(
                    prompt_id,
                    v,
                    "PUBLISHED",
                    actor=actor,
                    reason="initial_create",
                    actor_card_id=CARD,
                )
            except repository.InitialCreateNotAllowed as exc:
                assert exc.reason == repository.INITIAL_CREATE_HISTORY
                return "409"
            assert (prev, row["status"] if row else None) == ("DRAFT", "PUBLISHED")
            return "200"

        results = await asyncio.gather(create(1), create(2))
        assert sorted(results) == ["200", "409"], results
        async with get_pg_connection() as conn:
            rows = await conn.fetch(
                "SELECT to_status FROM prompt_version_transitions "
                "WHERE prompt_id = $1 AND to_status = 'PUBLISHED'",
                prompt_id,
            )
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Append-only
# ---------------------------------------------------------------------------


class TestAppendOnly:
    @pytest.mark.parametrize(
        "stmt",
        [
            "UPDATE prompt_version_transitions SET reason = 'rewritten' WHERE prompt_id = %s",
            "DELETE FROM prompt_version_transitions WHERE prompt_id = %s",
        ],
    )
    def test_update_and_delete_are_refused(self, client, prompt_id, stmt):
        _put(client, prompt_id, 1)
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="append-only"):
            _raw(stmt, prompt_id)
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    @pytest.mark.parametrize(
        "stmt",
        [
            "TRUNCATE prompt_version_transitions",
            "TRUNCATE prompt_versions CASCADE",
        ],
    )
    def test_truncate_is_refused(self, client, prompt_id, stmt):
        _put(client, prompt_id, 1)
        # Own transaction, always rolled back: were the guard missing, the
        # TRUNCATE must not survive to wipe the shared test database.
        with psycopg.connect(pg_conninfo()) as conn:
            try:
                with pytest.raises(psycopg.errors.InsufficientPrivilege, match="append-only"):
                    conn.execute(stmt)
            finally:
                conn.rollback()
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    @pytest.mark.parametrize(
        "stmt",
        [
            "DELETE FROM prompt_version_transitions WHERE prompt_id = %s",
            "TRUNCATE prompt_version_transitions",
            "UPDATE prompt_version_transitions SET reason = 'x' WHERE prompt_id = %s",
        ],
    )
    @pytest.mark.parametrize(
        "set_guc",
        [
            "SET LOCAL prompt_version_transitions.allow_delete = 'on'",
            "SET SESSION prompt_version_transitions.allow_delete = 'on'",
        ],
    )
    def test_the_retired_allow_delete_guc_opens_nothing(self, client, prompt_id, stmt, set_guc):
        """The GUC opt-out is GONE: any client could set it. Setting it must
        change nothing, whether SET LOCAL or a session SET."""
        _put(client, prompt_id, 1)
        with psycopg.connect(pg_conninfo()) as conn:
            try:
                conn.execute(set_guc)
                with pytest.raises(psycopg.errors.InsufficientPrivilege, match="append-only"):
                    conn.execute(stmt, (prompt_id,) if "%s" in stmt else None)
            finally:
                conn.rollback()
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    def test_a_version_with_history_cannot_be_deleted(self, client, prompt_id):
        _put(client, prompt_id, 1)
        with pytest.raises(psycopg.errors.RestrictViolation):
            _raw("DELETE FROM prompt_versions WHERE prompt_id = %s", prompt_id)


# ---------------------------------------------------------------------------
# REALIGN R7 (migration 039): every release names its board card
# ---------------------------------------------------------------------------


def _cards(rows: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    return [(r["to_status"], r["actor_card_id"]) for r in rows]


class TestReleaseCard:
    """Against real rows: a move into CANARY/PUBLISHED needs ``actor_card_id``
    (422 ``prompt_release_card_required`` and NOTHING written otherwise), the
    card lands on the ledger row and is returned by ``…/transitions``, and
    migration 039's CHECK refuses a card-less release row from any non-migration
    writer.

    MUTATIONS: drop ``_require_release_card`` from the route ⇒ the 422 cases
    still 422 here, but from the DB backstop — the route's own check is proved
    red in tests/test_prompt_routes_unit.py::TestReleaseCard; drop the 039 CHECK
    ⇒ ``test_the_database_refuses_a_card_less_release`` is red; persist nothing
    (``_record_transition`` ignoring ``actor_card_id``) ⇒ every ``_cards`` assert
    with a CARD is red.
    """

    REQUIRED: ClassVar[dict[str, Any]] = {"detail": {"error": "prompt_release_card_required"}}

    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
    def test_missing_card_is_422_and_writes_nothing(self, client, prompt_id, target):
        _put(client, prompt_id, 1)
        if target == "PUBLISHED":
            assert _status(client, prompt_id, 1, "CANARY").status_code == 200
        before = _ledger(client, prompt_id, 1)
        resp = client.post(f"/prompts/{prompt_id}/versions/1/status", json={"status": target})
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED
        assert _ledger(client, prompt_id, 1) == before
        expected = "CANARY" if target == "PUBLISHED" else "DRAFT"
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["status"] == expected

    @pytest.mark.parametrize(
        "bad",
        ["123456789012345678", "12345678901234567ab", ""],  # pragma: allowlist secret
    )
    def test_malformed_card_is_422_and_writes_nothing(self, client, prompt_id, bad):
        _put(client, prompt_id, 1)
        resp = _status(client, prompt_id, 1, "CANARY", card=bad)
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    def test_initial_create_without_a_card_is_422(self, client, prompt_id):
        _put(client, prompt_id, 1)
        resp = client.post(
            f"/prompts/{prompt_id}/versions/1/status",
            json={"status": "PUBLISHED", "reason": "initial_create"},
        )
        assert resp.status_code == 422, resp.text
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    def test_a_valid_card_is_on_the_ledger_row(self, client, prompt_id, principal):
        other = "9876543210987654321"
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, "CANARY").status_code == 200
        assert _status(client, prompt_id, 1, "PUBLISHED", card=other).status_code == 200
        # ARCHIVED needs none — a licence subject (no prompt role) may retire.
        principal.licence()
        assert _status(client, prompt_id, 1, "ARCHIVED").status_code == 200
        assert _cards(_ledger(client, prompt_id, 1)) == [
            ("DRAFT", None),
            ("CANARY", CARD),
            ("PUBLISHED", other),
            ("ARCHIVED", None),
        ]

    def test_a_card_on_a_non_release_move_is_not_recorded(self, client, prompt_id):
        """``transition`` records the card only on a move INTO CANARY/PUBLISHED:
        a card sent with ARCHIVED or DISCARDED is accepted and dropped, so the
        ledger never attributes a retirement to a release card.

        MUTATION: pass ``actor_card_id`` unconditionally to
        ``_record_transition`` ⇒ the ARCHIVED/DISCARDED rows carry ``other`` ⇒ red.
        """
        other = "9876543210987654321"
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, "CANARY").status_code == 200
        assert _status(client, prompt_id, 1, "PUBLISHED").status_code == 200
        assert _status(client, prompt_id, 1, "ARCHIVED", card=other).status_code == 200
        assert _status(client, prompt_id, 1, "DISCARDED", card=other).status_code == 200
        assert _cards(_ledger(client, prompt_id, 1)) == [
            ("DRAFT", None),
            ("CANARY", CARD),
            ("PUBLISHED", CARD),
            ("ARCHIVED", None),
            ("DISCARDED", None),
        ]

    def test_a_role_less_caller_gets_403_not_422(self, client, prompt_id, principal):
        _put(client, prompt_id, 1)
        principal.licence()
        resp = client.post(f"/prompts/{prompt_id}/versions/1/status", json={"status": "CANARY"})
        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": {"error": "prompt_publish_role_required"}}

    def test_activate_without_a_card_is_422_and_writes_nothing(self, client, prompt_id):
        _put(client, prompt_id, 1)
        resp = client.post(f"/prompts/{prompt_id}/versions/1/activate")
        assert resp.status_code == 422, resp.text
        assert resp.json() == self.REQUIRED
        assert _moves(_ledger(client, prompt_id, 1)) == [(None, "DRAFT", None)]

    def test_activate_records_its_card_on_every_row(self, client, prompt_id):
        other = "9876543210987654321"
        _put(client, prompt_id, 1)
        _put(client, prompt_id, 2)
        assert (
            client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
            == 200
        )
        resp = client.post(
            f"/prompts/{prompt_id}/versions/2/activate", json={"actor_card_id": other}
        )
        assert resp.status_code == 200, resp.text
        assert _cards(_ledger(client, prompt_id, 1)) == [
            ("DRAFT", None),
            ("PUBLISHED", CARD),
            ("ARCHIVED", other),  # the superseded archive belongs to v2's release
        ]
        assert _cards(_ledger(client, prompt_id, 2)) == [("DRAFT", None), ("PUBLISHED", other)]

    @pytest.mark.asyncio
    async def test_the_adoption_actor_publishes_without_a_card(self, prompt_id):
        """The decided exemption: ``actor_kind=migration`` + ``adopt_live`` (the
        in-process adoption job) has no card to name; its row keeps NULL."""
        from src.api.prompts import repository
        from src.api.prompts.actor import Actor

        await _draft(prompt_id, 1)
        row, prev = await repository.transition(
            prompt_id,
            1,
            repository.PUBLISHED,
            actor=Actor(sub="adopt_live_worker_prompts", kind="migration"),
            reason=repository.REASON_ADOPT_LIVE,
        )
        assert (prev, row["status"] if row else None) == ("DRAFT", "PUBLISHED")
        rows = await repository.get_transitions(prompt_id, 1)
        assert rows is not None
        assert [(r["to_status"], r["actor_kind"], r["actor_card_id"]) for r in rows] == [
            ("DRAFT", "service_account", None),
            ("PUBLISHED", "migration", None),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target", ["CANARY", "PUBLISHED"])
    async def test_the_database_refuses_a_card_less_release(self, prompt_id, target):
        """The 039 backstop: any non-migration writer that bypasses the route."""
        from src.api.prompts import repository
        from src.api.prompts.actor import Actor

        await _draft(prompt_id, 1)
        reason = "initial_create" if target == "PUBLISHED" else None
        with pytest.raises(psycopg.errors.CheckViolation) as exc:
            await repository.transition(
                prompt_id,
                1,
                target,
                actor=Actor(sub=SA, kind="service_account"),
                reason=reason,
            )
        assert exc.value.diag.constraint_name == repository.RELEASE_CARD_CONSTRAINT
        rows = await repository.get_transitions(prompt_id, 1)
        assert rows is not None
        assert [r["to_status"] for r in rows] == ["DRAFT"]

    def test_the_database_refuses_a_malformed_card(self, client, prompt_id):
        _put(client, prompt_id, 1)
        with pytest.raises(psycopg.errors.CheckViolation, match="actor_card_id_format"):
            _raw(
                "INSERT INTO prompt_version_transitions (prompt_id, version, from_status, "
                "to_status, actor_kind, actor_card_id) VALUES (%s, 1, 'DRAFT', 'CANARY', "
                "'service_account', '12345')",
                prompt_id,
            )


# ---------------------------------------------------------------------------
# worker_management#336: the card must exist and be a platform card
# ---------------------------------------------------------------------------


def _version_status(client: TestClient, pid: str, v: int) -> str:
    resp = client.get(f"/prompts/{pid}/versions/{v}")
    assert resp.status_code == 200, resp.text
    return resp.json()["status"]


def _timeout(_request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("slow fr")


def _refused_conn(_request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused")


class TestReleaseCardLookup:
    """Real PG: a refused lookup writes NOTHING — no status change, no ledger row."""

    UNVERIFIABLE: ClassVar[dict[str, Any]] = {
        "detail": {"error": "prompt_release_card_unverifiable"}
    }

    def test_a_platform_card_releases_and_is_recorded(self, client, prompt_id, fr):
        _put(client, prompt_id, 1)
        assert _status(client, prompt_id, 1, "CANARY").status_code == 200
        assert fr.calls == 1
        assert _version_status(client, prompt_id, 1) == "CANARY"
        assert _ledger(client, prompt_id, 1)[-1]["actor_card_id"] == CARD

    @pytest.mark.parametrize(
        ("answer", "status", "error"),
        [
            (
                lambda _r: httpx.Response(404, json={"error": "card_not_found"}),
                422,
                "prompt_release_card_unknown",
            ),
            (
                lambda _r: httpx.Response(
                    200,
                    json={
                        "card_id": CARD,
                        "tenant_id": "aaaaaaaa-0000-4000-8000-000000000100",
                        "platform": True,
                    },
                ),
                422,
                "prompt_release_card_not_platform",
            ),
            (
                lambda _r: httpx.Response(503, json={"error": "card_lookup_unavailable"}),
                503,
                "prompt_release_card_unverifiable",
            ),
            (lambda _r: httpx.Response(500), 503, "prompt_release_card_unverifiable"),
            (lambda _r: httpx.Response(401), 503, "prompt_release_card_unverifiable"),
            (
                lambda _r: httpx.Response(200, content=b"{not json"),
                503,
                "prompt_release_card_unverifiable",
            ),
            (_timeout, 503, "prompt_release_card_unverifiable"),
            (_refused_conn, 503, "prompt_release_card_unverifiable"),
        ],
        ids=["404", "not-platform", "503", "500", "401", "malformed", "timeout", "refused"],
    )
    @pytest.mark.parametrize("route", ["status", "activate"])
    def test_a_refused_lookup_writes_nothing(
        self, client, prompt_id, fr, answer, status, error, route
    ):
        _put(client, prompt_id, 1)
        before = _ledger(client, prompt_id, 1)
        fr.answer = answer
        if route == "status":
            resp = _status(client, prompt_id, 1, "CANARY")
        else:
            resp = client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY)
        assert resp.status_code == status, resp.text
        assert resp.json() == {"detail": {"error": error}}
        assert fr.calls == 1
        assert _version_status(client, prompt_id, 1) == "DRAFT"
        assert _ledger(client, prompt_id, 1) == before

    def test_missing_env_is_503_and_writes_nothing(self, client, prompt_id, fr, monkeypatch):
        from src.settings import reset_settings_cache

        _put(client, prompt_id, 1)
        before = _ledger(client, prompt_id, 1)
        monkeypatch.delenv("FR_WM_CARD_LOOKUP_TOKEN")
        reset_settings_cache()
        resp = _status(client, prompt_id, 1, "CANARY")
        assert resp.status_code == 503, resp.text
        assert resp.json() == self.UNVERIFIABLE
        assert fr.calls == 0
        assert _version_status(client, prompt_id, 1) == "DRAFT"
        assert _ledger(client, prompt_id, 1) == before
