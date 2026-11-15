"""``prompt_versions`` + ``/prompts*`` against a real PostgreSQL 18.

Proves the half of roadmap item A5 that only a database can prove: the
migration applies, the ledger is append-only, exactly one version can be
active, and search ranks. The metadata echo — the other half of the gate — is
proved without a database in ``tests/test_message_prompt_attribution.py``.

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql+psycopg://user:pw@127.0.0.1:PORT/worker_management_ci"  # pragma: allowlist secret
    export POSTGRES_SSLMODE=disable
    uv run python scripts/init_db.py
    uv run pytest tests/integration/test_prompt_versions_api.py -m pg -v --no-cov
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar

import psycopg
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.prompts.hashing import sha256_text, version_payload_sha256
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
    from collections.abc import AsyncIterator, Iterator

# The DDL tests below are async; the route tests drive a sync TestClient, so
# `asyncio` is applied per-test rather than to the whole module.
pytestmark = pytest.mark.pg

SYSTEM_V1 = "You are a keyword picker. Answer with single words."
SYSTEM_TENANT = "0195f48b-d000-7000-8000-000000000001"
# A Planka card id: REALIGN R7 requires one on every move into CANARY/PUBLISHED.
CARD = "1234567890123456789"
CARD_BODY: dict[str, Any] = {"actor_card_id": CARD}
OTHER_TENANT = "0195f48b-d000-7000-8000-000000000003"
SYSTEM_V2 = "You are a keyword picker. Answer with single lowercase words."


@pytest.fixture(autouse=True)
def _card_lookup_confirms(monkeypatch):
    """worker_management#336's fr card lookup, stubbed to "platform card".

    The real lookup is proved against real PG in
    test_prompt_transitions_ledger.py (TestReleaseCardLookup); here every card
    is confirmed so these tests exercise the version store.
    """
    from unittest.mock import AsyncMock

    from src.api.prompts import release_card

    monkeypatch.setattr(release_card, "verify_release_card", AsyncMock(return_value=None))


@pytest_asyncio.fixture(scope="module")
async def pg_pool() -> AsyncIterator[None]:
    require_database_url()
    os.environ.setdefault("POSTGRES_SSLMODE", "disable")
    try:
        await init_pg_pool()
    except PG_CONNECT_ERRORS as exc:  # pragma: no cover - environment-dependent
        unreachable_pg(exc)
    # The table is created by migrations/030, not by postgres_schema.sql, and
    # CI bootstraps the sidecar from the schema file alone. Applying the chain
    # here is what makes this module self-contained rather than dependent on
    # test_migration_chain.py having run first in the same job.
    await apply_migrations()
    try:
        yield
    finally:
        await close_pg_pool()


@pytest_asyncio.fixture
async def prompt_id(pg_pool) -> AsyncIterator[str]:
    """A unique prompt id per test, deleted afterwards.

    The ledger is fleet-global (no tenant column — prompts are git assets), so
    isolation has to come from the key, not from a tenant scope.
    """
    pid = f"test.kw-{uuid.uuid4().hex[:12]}"
    # Migration 035: owned by the system tenant (the client's tenant) up front,
    # so the raw-SQL inserts below (which, like a pre-035 pod, set no owner)
    # get it from the fill trigger. A route PUT by the owner is unaffected.
    async with get_pg_connection() as conn:
        await conn.execute(
            "INSERT INTO prompt_owner (prompt_id, owner_tenant_id) VALUES ($1, $2)",
            pid,
            SYSTEM_TENANT,
        )
    try:
        yield pid
    finally:
        await _purge_prompt(pid)


async def _purge_prompt(pid: str) -> None:
    """Delete a test prompt AND its ledger rows (migration 033).

    The ledger is append-only with NO opt-out, and its FK is ON DELETE RESTRICT,
    so the ledger goes first, with its user triggers disabled for this
    transaction only (``DISABLE TRIGGER USER`` is transactional). That needs
    table ownership, which already implies the power to drop the table.
    """
    async with get_pg_connection() as conn, conn.transaction():
        await conn.execute("ALTER TABLE prompt_version_transitions DISABLE TRIGGER USER")
        await conn.execute("DELETE FROM prompt_version_transitions WHERE prompt_id = $1", pid)
        await conn.execute("ALTER TABLE prompt_version_transitions ENABLE TRIGGER USER")
        await conn.execute("DELETE FROM prompt_versions WHERE prompt_id = $1", pid)
        await conn.execute("DELETE FROM prompt_owner WHERE prompt_id = $1", pid)


@pytest.fixture
def client(pg_pool, monkeypatch) -> Iterator[TestClient]:
    from src.api.prompts.routes import router
    from src.auth import caller_identity

    # DISCARD and activate need the elevated ``service-internal`` role; the
    # authorization itself is proved in tests/test_prompt_routes_unit.py
    # (TestElevatedAuth). Here the caller holds it, so these tests exercise
    # the ledger.
    monkeypatch.setattr(
        caller_identity, "get_caller_roles_from_context", lambda: ("service-internal",)
    )

    app = FastAPI()
    # A PLATFORM tenant: every prompt route, read or write, is platform-only
    # (routes.py PLATFORM GATE).
    app.dependency_overrides[get_tenant_id] = lambda: "0195f48b-d000-7000-8000-000000000001"
    app.include_router(router)

    @asynccontextmanager
    async def _noop(_a):
        yield

    app.router.lifespan_context = _noop
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _body(system: str = SYSTEM_V1, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "system": system,
        "question": "Which keywords describe this card?",
        "template": "Card: {{card}}",
        "output_schema": {"type": "object", "properties": {"keywords": {"type": "array"}}},
        "worker_names": ["worker-decide-keywords"],
        "purpose_tags": ["keywords", "retrieval"],
        "git_sha": "deadbeef",
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# The migration itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_creates_the_ledger_with_its_invariants(pg_pool):
    async with get_pg_connection() as conn:
        cols = await conn.fetch(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name = 'prompt_versions' ORDER BY column_name"
        )
        names = {c["column_name"] for c in cols}
        assert {
            "prompt_id",
            "version",
            "sha256",
            "system_prompt_sha256",
            "question",
            "system",
            "template",
            "output_schema",
            "worker_names",
            "purpose_tags",
            "git_sha",
            "status",
            "published_at",
            "activated_at",
            "search_tsv",
        } <= names

        pk = await conn.fetch(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = 'prompt_versions'::regclass AND i.indisprimary"
        )
        assert {r["attname"] for r in pk} == {"prompt_id", "version"}

        indexes = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'prompt_versions'"
        )
        idx = {r["indexname"] for r in indexes}
        # 031 dropped the one-active index: several PUBLISHED may coexist.
        assert "prompt_versions_one_active_idx" not in idx
        assert "prompt_versions_status_idx" in idx
        assert "prompt_versions_search_idx" in idx


@pytest.mark.asyncio
async def test_the_status_check_constraint_is_real(pg_pool, prompt_id):
    """A CHECK nobody tests is a CHECK that may not exist."""
    with pytest.raises(Exception) as exc:
        async with get_pg_connection() as conn:
            await conn.execute(
                "INSERT INTO prompt_versions (prompt_id, version, sha256, "
                "system_prompt_sha256, system, status) VALUES ($1, 1, $2, $3, $4, 'nonsense')",
                prompt_id,
                "a" * 64,
                "b" * 64,
                SYSTEM_V1,
            )
    assert "prompt_versions_status_check" in str(exc.value)


@pytest.mark.asyncio
async def test_legacy_statuses_are_admitted_transitionally_and_the_default_is_draft(
    pg_pool, prompt_id
):
    """031's CHECK is TRANSITIONAL: a pre-031 pod still running mid-rollout
    writes 'active'/'retired' and must not 500. A row inserted without a status
    is born DRAFT. (032 tightens the CHECK once the rollout is done.)"""
    async with get_pg_connection() as conn:
        await conn.execute(
            "INSERT INTO prompt_versions (prompt_id, version, sha256, "
            "system_prompt_sha256, system) VALUES ($1, 1, $2, $3, $4)",
            prompt_id,
            "a" * 64,
            sha256_text(SYSTEM_V1),
            SYSTEM_V1,
        )
        row = await conn.fetchrow(
            "SELECT status FROM prompt_versions WHERE prompt_id = $1", prompt_id
        )
    assert row["status"] == "DRAFT"
    for legacy in ("active", "retired"):
        async with get_pg_connection() as conn:
            await conn.execute(
                "UPDATE prompt_versions SET status = $2 WHERE prompt_id = $1",
                prompt_id,
                legacy,
            )


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------


class TestPublishAndRead:
    def test_publish_then_read_back_the_exact_digests(self, client, prompt_id):
        resp = client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        assert resp.status_code == 200, resp.text
        written = resp.json()
        assert written["written"] is True
        assert written["status"] == "DRAFT", "a PUT publishes; it must not activate"
        assert written["system_prompt_sha256"] == sha256_text(SYSTEM_V1)
        assert written["sha256"] == version_payload_sha256(
            question=_body()["question"],
            system=SYSTEM_V1,
            template=_body()["template"],
            output_schema=_body()["output_schema"],
            worker_names=_body()["worker_names"],
            purpose_tags=_body()["purpose_tags"],
        )

        got = client.get(f"/prompts/{prompt_id}/versions/1").json()
        assert got["system"] == SYSTEM_V1
        assert got["worker_names"] == ["worker-decide-keywords"]
        assert got["purpose_tags"] == ["keywords", "retrieval"]
        assert got["output_schema"] == _body()["output_schema"]
        assert got["config_revision"] == f"{prompt_id}@1"

    def test_republishing_identical_content_writes_nothing(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        again = client.put(f"/prompts/{prompt_id}/versions/1", json=_body()).json()
        assert again["written"] is False
        assert again["unchanged"] is True

    def test_a_different_git_sha_alone_is_still_unchanged(self, client, prompt_id):
        """git_sha is provenance, not content. Re-publishing the same bytes
        from a later commit must not look like a rewrite."""
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        again = client.put(
            f"/prompts/{prompt_id}/versions/1", json=_body(git_sha="cafebabe")
        ).json()
        assert again["unchanged"] is True
        # …and the ORIGINAL provenance survives: a published row is not edited.
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["git_sha"] == "deadbeef"

    def test_rewriting_a_published_version_is_refused(self, client, prompt_id):
        """⛔ The invariant the whole ledger rests on. If a version could be
        rewritten, no past attribution could ever be checked again."""
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        resp = client.put(f"/prompts/{prompt_id}/versions/1", json=_body(system=SYSTEM_V2))
        assert resp.status_code == 409, resp.text
        # The stored row is untouched — the refusal is not a partial write.
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["system"] == SYSTEM_V1

    def test_a_template_only_edit_is_also_refused(self, client, prompt_id):
        """Mutation control on the payload digest: a digest that covered only
        the system text would call this 'unchanged' and silently drop it."""
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        resp = client.put(
            f"/prompts/{prompt_id}/versions/1", json=_body(template="Card: {{other}}")
        )
        assert resp.status_code == 409

    def test_a_new_version_is_how_content_changes(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        resp = client.put(f"/prompts/{prompt_id}/versions/2", json=_body(system=SYSTEM_V2))
        assert resp.status_code == 200
        assert resp.json()["written"] is True
        assert resp.json()["system_prompt_sha256"] == sha256_text(SYSTEM_V2)
        # v1 is still readable — replay needs it.
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["system"] == SYSTEM_V1


class TestActivate:
    def test_publishing_is_not_activating(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        assert client.get(f"/prompts/{prompt_id}").status_code == 404

    def test_activate_swaps_the_single_active_version(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        client.put(f"/prompts/{prompt_id}/versions/2", json=_body(system=SYSTEM_V2))
        assert (
            client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
            == 200
        )

        second = client.post(f"/prompts/{prompt_id}/versions/2/activate", json=CARD_BODY).json()
        assert second["previous_active_version"] == 1
        assert second["changed"] is True

        active = client.get(f"/prompts/{prompt_id}").json()
        assert active["version"] == 2
        assert active["system"] == SYSTEM_V2
        assert active["config_revision"] == f"{prompt_id}@2"
        # The superseded version is ARCHIVED (it was live), not DRAFT.
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["status"] == "ARCHIVED"

    def test_rollback_to_a_lower_version_is_permitted(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        client.put(f"/prompts/{prompt_id}/versions/2", json=_body(system=SYSTEM_V2))
        client.post(f"/prompts/{prompt_id}/versions/2/activate", json=CARD_BODY)
        rolled = client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).json()
        assert rolled["previous_active_version"] == 2
        assert client.get(f"/prompts/{prompt_id}").json()["version"] == 1

    def test_activating_an_unpublished_version_is_404(self, client, prompt_id):
        assert (
            client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
            == 404
        )

    def test_activate_archives_every_other_published_version(self, client, prompt_id):
        """The deprecated alias keeps "exactly one live" even after the status
        route has made several versions PUBLISHED."""
        for v, system in ((1, SYSTEM_V1), (2, SYSTEM_V2), (3, SYSTEM_V1 + " v3")):
            client.put(f"/prompts/{prompt_id}/versions/{v}", json=_body(system=system))
        for v in (1, 2):
            _walk(client, prompt_id, v, "CANARY", "PUBLISHED")
        resp = client.post(f"/prompts/{prompt_id}/versions/3/activate", json=CARD_BODY).json()
        assert resp["previous_active_version"] == 2
        statuses = [
            client.get(f"/prompts/{prompt_id}/versions/{v}").json()["status"] for v in (1, 2, 3)
        ]
        assert statuses == ["ARCHIVED", "ARCHIVED", "PUBLISHED"]

    def test_activating_a_discarded_version_is_409(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        _walk(client, prompt_id, 1, "DISCARDED")
        assert (
            client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
            == 409
        )
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["status"] == "DISCARDED"


def _walk(client, prompt_id: str, version: int, *statuses: str) -> None:
    for status in statuses:
        resp = client.post(
            f"/prompts/{prompt_id}/versions/{version}/status",
            json={"status": status, "actor_card_id": CARD},
        )
        assert resp.status_code == 200, f"-> {status}: {resp.text}"


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


class TestLifecycle:
    """The transition matrix against the real ledger (P4 §2)."""

    STATUSES = ("DRAFT", "CANARY", "PUBLISHED", "ARCHIVED", "DISCARDED")
    ALLOWED: ClassVar = {
        "DRAFT": {"CANARY", "DISCARDED"},
        "CANARY": {"PUBLISHED", "DISCARDED"},
        "PUBLISHED": {"ARCHIVED", "DISCARDED"},
        "ARCHIVED": {"PUBLISHED", "DISCARDED"},
        "DISCARDED": set(),
    }
    # How to reach each from-status from a fresh DRAFT, via permitted moves only.
    REACH: ClassVar = {
        "DRAFT": (),
        "CANARY": ("CANARY",),
        "PUBLISHED": ("CANARY", "PUBLISHED"),
        "ARCHIVED": ("CANARY", "PUBLISHED", "ARCHIVED"),
        "DISCARDED": ("DISCARDED",),
    }

    @pytest.mark.parametrize("src", STATUSES)
    def test_every_cell_of_the_matrix(self, client, prompt_id, src):
        """25 cells: each from-status against each target, on a real row."""
        for n, dst in enumerate(self.STATUSES, start=1):
            client.put(f"/prompts/{prompt_id}/versions/{n}", json=_body(system=f"{src}->{dst}"))
            _walk(client, prompt_id, n, *self.REACH[src])
            resp = client.post(
                f"/prompts/{prompt_id}/versions/{n}/status",
                json={"status": dst, "actor_card_id": CARD},
            )
            after = client.get(f"/prompts/{prompt_id}/versions/{n}").json()["status"]
            if dst in self.ALLOWED[src]:
                assert resp.status_code == 200, f"{src}->{dst}: {resp.text}"
                assert resp.json()["previous_status"] == src
                assert after == dst
            else:
                assert resp.status_code == 409, f"{src}->{dst} should be refused: {resp.text}"
                assert after == src, f"a refused {src}->{dst} must write nothing"

    def test_unknown_version_is_404(self, client, prompt_id):
        resp = client.post(
            f"/prompts/{prompt_id}/versions/1/status",
            json={"status": "CANARY", "actor_card_id": CARD},
        )
        assert resp.status_code == 404

    def test_publishing_stamps_activated_at(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["activated_at"] is None
        _walk(client, prompt_id, 1, "CANARY")
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["activated_at"] is None
        _walk(client, prompt_id, 1, "PUBLISHED")
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["activated_at"] is not None


class TestResolver:
    """Pinned reads return any status; the unpinned read resolves per the flag."""

    def _three(self, client, prompt_id):
        # v3 published FIRST, then v1 — so "most recently activated" (flag off)
        # and "highest" (flag on) disagree. v2 sits in CANARY and must never be
        # resolved unpinned.
        for v in (1, 2, 3):
            client.put(f"/prompts/{prompt_id}/versions/{v}", json=_body(system=f"v{v}"))
        _walk(client, prompt_id, 3, "CANARY", "PUBLISHED")
        _walk(client, prompt_id, 1, "CANARY", "PUBLISHED")
        _walk(client, prompt_id, 2, "CANARY")

    def test_flag_on_resolves_the_highest_published(self, client, prompt_id, flag):
        flag(True)
        self._three(client, prompt_id)
        got = client.get(f"/prompts/{prompt_id}").json()
        assert (got["version"], got["status"]) == (3, "PUBLISHED")

    def test_flag_off_resolves_todays_live_version(self, client, prompt_id, flag):
        flag(False)
        self._three(client, prompt_id)
        assert client.get(f"/prompts/{prompt_id}").json()["version"] == 1

    @pytest.mark.parametrize("on", [True, False])
    def test_nothing_published_is_404_either_way(self, client, prompt_id, flag, on):
        flag(on)
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        _walk(client, prompt_id, 1, "CANARY")
        assert client.get(f"/prompts/{prompt_id}").status_code == 404

    def test_a_pinned_read_returns_any_status(self, client, prompt_id, flag):
        flag(True)
        for v, path in enumerate(
            ((), ("CANARY",), ("CANARY", "PUBLISHED", "ARCHIVED"), ("DISCARDED",)), start=1
        ):
            client.put(f"/prompts/{prompt_id}/versions/{v}", json=_body(system=f"v{v}"))
            _walk(client, prompt_id, v, *path)
        got = {
            v: client.get(f"/prompts/{prompt_id}/versions/{v}").json()["status"]
            for v in (1, 2, 3, 4)
        }
        assert got == {1: "DRAFT", 2: "CANARY", 3: "ARCHIVED", 4: "DISCARDED"}

    def test_search_status_filter(self, client, prompt_id):
        self._three(client, prompt_id)
        canary = client.get(
            "/prompts/search", params={"q": "keywords", "status": "CANARY", "limit": 100}
        ).json()
        mine = [h["version"] for h in canary["hits"] if h["prompt_id"] == prompt_id]
        assert mine == [2]


class TestSearch:
    def test_finds_by_question_text_and_ranks(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY)

        hits = client.get("/prompts/search", params={"q": "keywords card", "limit": 50}).json()
        ids = [h["prompt_id"] for h in hits["hits"]]
        assert prompt_id in ids
        hit = next(h for h in hits["hits"] if h["prompt_id"] == prompt_id)
        assert hit["rank"] > 0
        assert hit["config_revision"] == f"{prompt_id}@1"
        assert hit["system_prompt_sha256"] == sha256_text(SYSTEM_V1)
        assert "system" not in hit, "search hits must not carry prompt bodies"

    def test_a_query_that_matches_nothing_returns_nothing(self, client, prompt_id):
        """Negative control: without this, the test above could be passing on
        an unfiltered listing rather than on the index."""
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY)
        hits = client.get(
            "/prompts/search", params={"q": "xyzzy-no-such-token", "limit": 50}
        ).json()
        assert prompt_id not in [h["prompt_id"] for h in hits["hits"]]

    def test_tag_filter(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY)
        assert prompt_id in [
            h["prompt_id"]
            for h in client.get("/prompts/search", params={"tag": "keywords", "limit": 50}).json()[
                "hits"
            ]
        ]
        assert prompt_id not in [
            h["prompt_id"]
            for h in client.get(
                "/prompts/search", params={"tag": "no-such-tag", "limit": 50}
            ).json()["hits"]
        ]

    def test_default_status_hides_a_published_but_unactivated_version(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        default = client.get("/prompts/search", params={"q": "keywords", "limit": 50}).json()
        assert prompt_id not in [h["prompt_id"] for h in default["hits"]]
        any_status = client.get(
            "/prompts/search", params={"q": "keywords", "status": "any", "limit": 50}
        ).json()
        assert prompt_id in [h["prompt_id"] for h in any_status["hits"]]


class TestSearchByPromptId:
    """``?prompt_id=`` is the only way to prove absence past ``limit`` rows."""

    def test_filter_across_150_rows_returns_only_that_prompts_versions(self, client, prompt_id):
        filler = f"test.filler-{uuid.uuid4().hex[:8]}"
        fillers = [f"{filler}-{n:03d}" for n in range(147)]
        try:
            # 147 other prompts that sort BEFORE the target ("test.f" < "test.k").
            for pid in fillers:
                assert client.put(f"/prompts/{pid}/versions/1", json=_body()).status_code == 200
            for v in (1, 2, 3):
                body = _body(system=f"target v{v}")
                assert (
                    client.put(f"/prompts/{prompt_id}/versions/{v}", json=body).status_code == 200
                )
            assert (
                client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
                == 200
            )

            # Control: the unfiltered 100-row page cannot see the target at all.
            page = client.get("/prompts/search", params={"status": "any", "limit": 100}).json()
            assert page["count"] == 100
            assert prompt_id not in {h["prompt_id"] for h in page["hits"]}

            hits = client.get(
                "/prompts/search", params={"prompt_id": prompt_id, "status": "any", "limit": 100}
            ).json()["hits"]
            assert [(h["prompt_id"], h["version"]) for h in hits] == [
                (prompt_id, 3),
                (prompt_id, 2),
                (prompt_id, 1),
            ]

            # Combinable with status.
            published = client.get(
                "/prompts/search", params={"prompt_id": prompt_id, "status": "PUBLISHED"}
            ).json()["hits"]
            assert [(h["prompt_id"], h["version"]) for h in published] == [(prompt_id, 1)]

            # Absence is provable.
            absent = client.get(
                "/prompts/search", params={"prompt_id": f"{prompt_id}-nope", "status": "any"}
            ).json()
            assert absent["count"] == 0
        finally:
            _purge_many(fillers)


class TestSearchByWorkerName:
    """``?worker_name=`` is an exact ELEMENT match on ``worker_names``."""

    def test_filter_across_150_rows_returns_only_rows_containing_the_name(self, client):
        tag = uuid.uuid4().hex[:8]
        worker = f"worker-target-{tag}"
        pids = [f"test.wn-{tag}-{n:03d}" for n in range(150)]
        # Every 7th row carries the name — alone, or among several others.
        # Every other row carries only lookalikes, so a LIKE / prefix / substring
        # match would return them and an exact element match must not.
        expected: set[str] = set()
        published: set[str] = set()
        try:
            for n, pid in enumerate(pids):
                if n % 7 == 0:
                    names = [worker] if n % 14 == 0 else ["worker-other", worker, "worker-z"]
                    expected.add(pid)
                else:
                    names = [f"{worker}-x", worker[:-1], f"x{worker}", "worker-other"]
                body = _body(system=f"wn {n}", worker_names=names)
                assert client.put(f"/prompts/{pid}/versions/1", json=body).status_code == 200
                if n % 3 == 0:
                    assert (
                        client.post(
                            f"/prompts/{pid}/versions/1/activate", json=CARD_BODY
                        ).status_code
                        == 200
                    )
                    if pid in expected:
                        published.add(pid)
            assert len(expected) == 22
            assert 0 < len(published) < len(expected)

            resp = client.get(
                "/prompts/search", params={"worker_name": worker, "status": "any", "limit": 100}
            ).json()
            assert {h["prompt_id"] for h in resp["hits"]} == expected
            assert resp["count"] == len(expected)
            assert all(worker in h["worker_names"] for h in resp["hits"])
            # Rows holding the name among several others are included.
            assert any(len(h["worker_names"]) == 3 for h in resp["hits"])

            # Combinable with status.
            pub = client.get(
                "/prompts/search",
                params={"worker_name": worker, "status": "PUBLISHED", "limit": 100},
            ).json()["hits"]
            assert {h["prompt_id"] for h in pub} == published
            assert all(h["status"] == "PUBLISHED" for h in pub)

            # Combinable with prompt_id, and limit still applies.
            one = sorted(expected)[0]
            both = client.get(
                "/prompts/search",
                params={"worker_name": worker, "prompt_id": one, "status": "any"},
            ).json()
            assert [h["prompt_id"] for h in both["hits"]] == [one]
            assert both["prompt_id"] == one
            capped = client.get(
                "/prompts/search", params={"worker_name": worker, "status": "any", "limit": 5}
            ).json()
            assert capped["count"] == 5

            # Control: without the filter the lookalike rows ARE listed, so the
            # filter above is what removed them.
            unfiltered = client.get(
                "/prompts/search", params={"q": "wn", "status": "any", "limit": 100}
            ).json()["hits"]
            assert {h["prompt_id"] for h in unfiltered if h["prompt_id"] in pids} - expected
        finally:
            _purge_many(pids)

    def test_every_row_carries_its_prompt_id(self, client, prompt_id):
        for v in (1, 2):
            put = client.put(f"/prompts/{prompt_id}/versions/{v}", json=_body(system=f"p{v}"))
            assert put.status_code == 200
        body = client.get("/prompts/search", params={"status": "any", "limit": 100}).json()
        assert body["hits"]
        assert all(isinstance(h.get("prompt_id"), str) and h["prompt_id"] for h in body["hits"])
        assert body["prompt_id"] is None
        filtered = client.get(
            "/prompts/search", params={"prompt_id": prompt_id, "status": "any"}
        ).json()
        assert filtered["prompt_id"] == prompt_id
        assert {h["prompt_id"] for h in filtered["hits"]} == {prompt_id}

    @pytest.mark.parametrize(
        "evil",
        [
            "x' OR '1'='1",
            "worker'); DROP TABLE prompt_versions; --",
            "%",
            "worker-%",
            "{worker-decide-keywords}",
            "worker-decide-keywords' OR TRUE --",
        ],
    )
    def test_injection_shaped_names_are_bound_not_interpolated(self, client, prompt_id, evil):
        assert client.put(f"/prompts/{prompt_id}/versions/1", json=_body()).status_code == 200
        resp = client.get(
            "/prompts/search", params={"worker_name": evil, "status": "any", "limit": 100}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["count"] == 0
        # The table survived and still answers.
        ok = client.get("/prompts/search", params={"prompt_id": prompt_id, "status": "any"}).json()
        assert ok["count"] == 1

    def test_a_name_holding_sql_metacharacters_matches_itself_exactly(self, client, prompt_id):
        name = "o'brien\"; DROP TABLE x; -- %"
        body = _body(worker_names=["plain", name])
        assert client.put(f"/prompts/{prompt_id}/versions/1", json=body).status_code == 200
        hits = client.get(
            "/prompts/search", params={"worker_name": name, "status": "any", "limit": 100}
        ).json()["hits"]
        assert [h["prompt_id"] for h in hits] == [prompt_id]


def _purge_many(pids: list[str]) -> None:
    """Sync twin of :func:`_purge_prompt` for a batch (the TestClient owns the loop)."""
    with psycopg.connect(pg_conninfo()) as conn, conn.transaction():
        conn.execute("ALTER TABLE prompt_version_transitions DISABLE TRIGGER USER")
        conn.execute("DELETE FROM prompt_version_transitions WHERE prompt_id = ANY(%s)", (pids,))
        conn.execute("ALTER TABLE prompt_version_transitions ENABLE TRIGGER USER")
        conn.execute("DELETE FROM prompt_versions WHERE prompt_id = ANY(%s)", (pids,))
        conn.execute("DELETE FROM prompt_owner WHERE prompt_id = ANY(%s)", (pids,))


# ---------------------------------------------------------------------------
# Rollout safety, both directions
# ---------------------------------------------------------------------------


async def _old_pod_put(prompt_id: str, version: int, system: str) -> None:
    """Exactly what main's (pre-031) repository.upsert_version inserts."""
    async with get_pg_connection() as conn:
        await conn.execute(
            "INSERT INTO prompt_versions (prompt_id, version, sha256, system_prompt_sha256, "
            "system, worker_names, purpose_tags, status) "
            "VALUES ($1, $2, $3, $4, $5, $6::text[], $7::text[], 'retired')",
            prompt_id,
            version,
            ("%02d" % version) * 32,
            sha256_text(system),
            system,
            [],
            [],
        )


async def _old_pod_activate(prompt_id: str, version: int) -> None:
    """Exactly what main's (pre-031) repository.activate writes."""
    async with get_pg_connection() as conn, conn.transaction():
        await conn.execute(
            "UPDATE prompt_versions SET status = 'retired' "
            "WHERE prompt_id = $1 AND status = 'active' AND version <> $2",
            prompt_id,
            version,
        )
        await conn.execute(
            "UPDATE prompt_versions SET status = 'active', activated_at = now() "
            "WHERE prompt_id = $1 AND version = $2",
            prompt_id,
            version,
        )


class TestOldPodWritesAfter031:
    """A pre-031 pod still serving during the rolling deploy writes the 030
    spellings. They must be ACCEPTED (transitional CHECK) and READ by the new
    code as their lifecycle status, with the same mapping 031 applied."""

    @pytest.mark.asyncio
    async def test_old_style_rows_are_written_and_read_as_lifecycle_statuses(
        self, client, prompt_id
    ):
        await _old_pod_put(prompt_id, 1, SYSTEM_V1)  # retired, never live -> DRAFT
        await _old_pod_put(prompt_id, 2, SYSTEM_V2)
        await _old_pod_activate(prompt_id, 1)  # v1 active
        await _old_pod_activate(prompt_id, 2)  # v1 retired+activated -> ARCHIVED, v2 active
        await _old_pod_put(prompt_id, 3, SYSTEM_V1 + " v3")  # retired, never live -> DRAFT
        async with get_pg_connection() as conn:
            raw = {
                r["version"]: r["status"]
                for r in await conn.fetch(
                    "SELECT version, status FROM prompt_versions WHERE prompt_id = $1", prompt_id
                )
            }
        assert raw == {1: "retired", 2: "active", 3: "retired"}, "old-pod writes must land"

        got = {v: client.get(f"/prompts/{prompt_id}/versions/{v}").json()["status"] for v in raw}
        assert got == {1: "ARCHIVED", 2: "PUBLISHED", 3: "DRAFT"}
        live = client.get(f"/prompts/{prompt_id}")
        assert live.status_code == 200 and live.json()["version"] == 2
        assert live.json()["status"] == "PUBLISHED"

        def _search(status: str) -> dict[int, str]:
            hits = client.get("/prompts/search", params={"status": status, "limit": 100}).json()[
                "hits"
            ]
            return {h["version"]: h["status"] for h in hits if h["prompt_id"] == prompt_id}

        assert _search("PUBLISHED") == {2: "PUBLISHED"}
        assert _search("active") == {2: "PUBLISHED"}
        assert _search("ARCHIVED") == {1: "ARCHIVED"}
        assert _search("DRAFT") == {3: "DRAFT"}
        assert _search("retired") == {1: "ARCHIVED", 3: "DRAFT"}

    @pytest.mark.asyncio
    async def test_new_code_transitions_and_activates_old_style_rows(self, client, prompt_id):
        await _old_pod_put(prompt_id, 1, SYSTEM_V1)
        await _old_pod_put(prompt_id, 2, SYSTEM_V2)
        await _old_pod_activate(prompt_id, 1)
        # Legacy 'retired' (never live) is DRAFT: DRAFT -> CANARY is permitted,
        # DRAFT -> PUBLISHED is not.
        bad = client.post(
            f"/prompts/{prompt_id}/versions/2/status",
            json={"status": "PUBLISHED", "actor_card_id": CARD},
        )
        assert bad.status_code == 409 and "DRAFT -> PUBLISHED" in bad.text
        ok = client.post(
            f"/prompts/{prompt_id}/versions/2/status",
            json={"status": "CANARY", "actor_card_id": CARD},
        )
        assert ok.status_code == 200 and ok.json()["previous_status"] == "DRAFT"
        # activate v2 archives the legacy 'active' v1.
        act = client.post(f"/prompts/{prompt_id}/versions/2/activate", json=CARD_BODY).json()
        assert act["previous_active_version"] == 1
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["status"] == "ARCHIVED"
        assert client.get(f"/prompts/{prompt_id}").json()["version"] == 2


class TestNewCodeAgainstAPre031Schema:
    """031 failed under the non-strict prod runner: the table still carries the
    030 CHECK. The startup probe must say so, so the write routes 503 instead
    of 500ing on a CheckViolation."""

    @pytest.mark.asyncio
    async def test_probe_reads_missing_031_on_the_030_check_and_ready_after_031(self, pg_pool):
        import re
        from pathlib import Path

        import psycopg

        from src.api.prompts import repository

        from .pg_guard import pg_conninfo

        migrations = Path(__file__).resolve().parents[2] / "migrations"

        def _body_of(name: str) -> str:
            sql = (migrations / name).read_text()
            return re.sub(r"^\s*(BEGIN|COMMIT)\s*;\s*$", "", sql, flags=re.M | re.I)

        before = repository.lifecycle_schema_state()
        schema = f"p4_pre031_{uuid.uuid4().hex[:8]}"
        # DDL on a raw connection: the migration files are not pool-wrapper SQL.
        raw = psycopg.connect(pg_conninfo(), autocommit=True)
        try:
            with raw.transaction():
                raw.execute(f"CREATE SCHEMA {schema}")
                raw.execute(f"SET LOCAL search_path = {schema}")
                raw.execute(_body_of("030_prompt_versions.sql"))
            assert (
                await repository.probe_lifecycle_schema(f"{schema}.prompt_versions")
                == repository.SCHEMA_MISSING_031
            )
            assert repository.lifecycle_schema_state() == repository.SCHEMA_MISSING_031
            # And the 030 CHECK really does refuse what new code writes.
            with pytest.raises(psycopg.errors.CheckViolation):
                raw.execute(
                    f"INSERT INTO {schema}.prompt_versions (prompt_id, version, sha256, "
                    "system_prompt_sha256, system, status) VALUES "
                    "('x', 1, %s, %s, 's', 'DRAFT')",
                    ("a" * 64, "b" * 64),
                )
            # An absent table is not ready either.
            assert (
                await repository.probe_lifecycle_schema(f"{schema}.no_such_table")
                == repository.SCHEMA_MISSING_031
            )
            # Negative control: apply 031 to the same table and the probe flips.
            with raw.transaction():
                raw.execute(f"SET LOCAL search_path = {schema}")
                raw.execute(_body_of("031_prompt_version_lifecycle.sql"))
            assert (
                await repository.probe_lifecycle_schema(f"{schema}.prompt_versions")
                == repository.SCHEMA_READY
            )
            # The live (public) ledger this module migrated is ready.
            assert await repository.probe_lifecycle_schema() == repository.SCHEMA_READY
        finally:
            raw.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            raw.close()
            repository.set_lifecycle_schema_state(before)


# ---------------------------------------------------------------------------
# Ownership (migration 035): one owning tenant per prompt, enforced on writes
# ---------------------------------------------------------------------------


@pytest.fixture
def tenant_client(pg_pool, monkeypatch):
    """A client factory: ``(tenant, *roles)`` -> TestClient on the real repository."""
    from src.api.prompts.routes import router
    from src.auth import caller_identity

    state: dict[str, Any] = {"tenant": SYSTEM_TENANT, "roles": ()}
    monkeypatch.setattr(caller_identity, "get_caller_roles_from_context", lambda: state["roles"])
    app = FastAPI()
    app.dependency_overrides[get_tenant_id] = lambda: state["tenant"]
    app.include_router(router)

    @asynccontextmanager
    async def _noop(_a):
        yield

    app.router.lifespan_context = _noop
    with TestClient(app, raise_server_exceptions=False) as c:

        def _as(tenant: str, *roles: str) -> TestClient:
            state["tenant"], state["roles"] = tenant, roles
            return c

        yield _as


def _owner_of(pid: str) -> str | None:
    with psycopg.connect(pg_conninfo()) as conn:
        row = conn.execute("SELECT owner_tenant_id FROM prompt_owner WHERE prompt_id = %s", (pid,))
        found = row.fetchone()
    return None if found is None else found[0]


def _version_count(pid: str) -> int:
    with psycopg.connect(pg_conninfo()) as conn:
        return conn.execute(
            "SELECT count(*) FROM prompt_versions WHERE prompt_id = %s", (pid,)
        ).fetchone()[0]


class TestPromptOwnership:
    def test_a_new_prompt_is_owned_by_the_creating_tenant(self, tenant_client):
        pid = f"test.owner-{uuid.uuid4().hex[:12]}"
        try:
            resp = tenant_client(OTHER_TENANT).put(f"/prompts/{pid}/versions/1", json=_body())
            assert resp.status_code == 200, resp.text
            assert _owner_of(pid) == OTHER_TENANT
            # And its next version, by the same tenant, is accepted.
            resp = tenant_client(OTHER_TENANT).put(
                f"/prompts/{pid}/versions/2", json=_body(SYSTEM_V2)
            )
            assert resp.status_code == 200, resp.text
        finally:
            _purge_many([pid])

    def test_another_platform_tenant_cannot_write_or_move_it(self, tenant_client, prompt_id):
        """``prompt_id`` is owned by the system tenant (fixture). A different
        PLATFORM tenant, holding no prompt role, is refused on PUT and status."""
        assert (
            tenant_client(SYSTEM_TENANT)
            .put(f"/prompts/{prompt_id}/versions/1", json=_body())
            .status_code
            == 200
        )
        other = tenant_client(OTHER_TENANT)
        put = other.put(f"/prompts/{prompt_id}/versions/2", json=_body(SYSTEM_V2))
        assert put.status_code == 403, put.text
        assert put.json() == {"detail": {"error": "prompt_owner_mismatch"}}
        # Even an idempotent re-PUT of the existing version is refused.
        assert other.put(f"/prompts/{prompt_id}/versions/1", json=_body()).status_code == 403
        assert _version_count(prompt_id) == 1
        assert _owner_of(prompt_id) == SYSTEM_TENANT

    def test_the_role_gate_refuses_a_roleless_move_into_canary(self, tenant_client, prompt_id):
        """U6 step 4: on CANARY/PUBLISHED the role gate fires BEFORE ownership,
        so a roleless other tenant sees ``prompt_publish_role_required``. This
        pins the gate only; ownership is pinned by the ARCHIVED test below."""
        assert (
            tenant_client(SYSTEM_TENANT)
            .put(f"/prompts/{prompt_id}/versions/1", json=_body())
            .status_code
            == 200
        )
        move = tenant_client(OTHER_TENANT).post(
            f"/prompts/{prompt_id}/versions/1/status",
            json={"status": "CANARY", "actor_card_id": CARD},
        )
        assert move.status_code == 403, move.text
        assert move.json() == {"detail": {"error": "prompt_publish_role_required"}}
        got = tenant_client(SYSTEM_TENANT).get(f"/prompts/{prompt_id}/versions/1").json()
        assert got["status"] == "DRAFT"

    def test_another_platform_tenant_cannot_archive_a_published_version(
        self, tenant_client, prompt_id
    ):
        """ARCHIVED is not role-gated, so ownership is its ONLY barrier: a
        roleless platform tenant B moving tenant A's PUBLISHED version to
        ARCHIVED is 403 ``prompt_owner_mismatch`` and A's row is unchanged."""
        owner = tenant_client(SYSTEM_TENANT, "service-internal")
        assert owner.put(f"/prompts/{prompt_id}/versions/1", json=_body()).status_code == 200
        _walk(owner, prompt_id, 1, "CANARY", "PUBLISHED")
        before = owner.get(f"/prompts/{prompt_id}/versions/1").json()
        assert before["status"] == "PUBLISHED"
        with psycopg.connect(pg_conninfo()) as conn:
            ledger_before = conn.execute(
                "SELECT count(*) FROM prompt_version_transitions WHERE prompt_id = %s",
                (prompt_id,),
            ).fetchone()[0]

        move = tenant_client(OTHER_TENANT).post(
            f"/prompts/{prompt_id}/versions/1/status",
            json={"status": "ARCHIVED", "actor_card_id": CARD},
        )
        assert move.status_code == 403, move.text
        assert move.json() == {"detail": {"error": "prompt_owner_mismatch"}}

        after = tenant_client(SYSTEM_TENANT).get(f"/prompts/{prompt_id}/versions/1").json()
        assert after == before
        assert _owner_of(prompt_id) == SYSTEM_TENANT
        with psycopg.connect(pg_conninfo()) as conn:
            ledger_after = conn.execute(
                "SELECT count(*) FROM prompt_version_transitions WHERE prompt_id = %s",
                (prompt_id,),
            ).fetchone()[0]
        assert ledger_after == ledger_before
        # Positive control: the OWNER, also roleless, may archive it.
        ok = tenant_client(SYSTEM_TENANT).post(
            f"/prompts/{prompt_id}/versions/1/status",
            json={"status": "ARCHIVED", "actor_card_id": CARD},
        )
        assert ok.status_code == 200, ok.text

    def test_the_bootstrap_sa_may_write_any_prompt_and_never_changes_its_owner(
        self, tenant_client, prompt_id
    ):
        boot = tenant_client(OTHER_TENANT, "prompt-asset-bootstrap")
        resp = boot.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        assert resp.status_code == 200, resp.text
        assert _owner_of(prompt_id) == SYSTEM_TENANT
        with psycopg.connect(pg_conninfo()) as conn:
            owner = conn.execute(
                "SELECT owner_tenant_id FROM prompt_versions WHERE prompt_id = %s", (prompt_id,)
            ).fetchone()[0]
        assert owner == SYSTEM_TENANT


def _published_count(prompt_id: str, bound: str) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(
        "wm_prompt_published_bound_total", {"prompt_id": prompt_id, "bound": bound}
    )
    return value or 0.0


async def _bind(prompt_id: str) -> None:
    async with get_pg_connection() as conn:
        await conn.execute(
            "INSERT INTO worker_prompt_binding (tenant_id, worker_name, prompt_id, bound_by) "
            "VALUES ($1, $2, $3, 'test')",
            SYSTEM_TENANT,
            f"w-{prompt_id}",
            prompt_id,
        )


async def _unbind(prompt_id: str) -> None:
    async with get_pg_connection() as conn:
        await conn.execute("DELETE FROM worker_prompt_binding WHERE prompt_id = $1", prompt_id)


class TestPublishedCounter:
    """``wm_prompt_published_bound_total{prompt_id, bound}`` (U6 step 4, F1):
    one increment per version moved INTO PUBLISHED, on every path, labelled by
    whether a worker_prompt_binding references the prompt at that moment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bound", [True, False])
    async def test_status_route_counts_canary_to_published(self, client, prompt_id, bound):
        label, other = ("true", "false") if bound else ("false", "true")
        if bound:
            await _bind(prompt_id)
        try:
            client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
            before, before_other = (
                _published_count(prompt_id, label),
                _published_count(prompt_id, other),
            )
            _walk(client, prompt_id, 1, "CANARY")
            assert _published_count(prompt_id, label) == before  # CANARY is not a publish
            _walk(client, prompt_id, 1, "PUBLISHED")
            assert _published_count(prompt_id, label) == before + 1
            # ARCHIVED -> PUBLISHED is a publish too; ARCHIVED alone is not.
            _walk(client, prompt_id, 1, "ARCHIVED")
            assert _published_count(prompt_id, label) == before + 1
            _walk(client, prompt_id, 1, "PUBLISHED")
            assert _published_count(prompt_id, label) == before + 2
            assert _published_count(prompt_id, other) == before_other
        finally:
            await _unbind(prompt_id)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bound", [True, False])
    async def test_activate_counts_a_new_publish_only(self, client, prompt_id, bound):
        label = "true" if bound else "false"
        if bound:
            await _bind(prompt_id)
        try:
            client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
            before = _published_count(prompt_id, label)
            assert (
                client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
                == 200
            )
            assert _published_count(prompt_id, label) == before + 1
            # Re-activating an already-PUBLISHED version moves nothing into PUBLISHED.
            assert (
                client.post(f"/prompts/{prompt_id}/versions/1/activate", json=CARD_BODY).status_code
                == 200
            )
            assert _published_count(prompt_id, label) == before + 1
        finally:
            await _unbind(prompt_id)

    def test_a_refused_publish_counts_nothing(self, client, prompt_id):
        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        before = _published_count(prompt_id, "false")
        bad = client.post(
            f"/prompts/{prompt_id}/versions/1/status",
            json={"status": "PUBLISHED", "actor_card_id": CARD},
        )
        assert bad.status_code == 409  # DRAFT -> PUBLISHED is outside the matrix
        assert _published_count(prompt_id, "false") == before


class TestPublishRoleGateOnRealRows:
    """F2: a platform licence WITHOUT a prompt role cannot move a real row into
    CANARY or PUBLISHED; nothing is written. Authorization is unit-tested in
    tests/test_prompt_routes_unit.py (TestPublishRoleGate); here the row and
    the ledger are checked to be untouched."""

    @pytest.mark.asyncio
    async def test_no_role_moves_nothing(self, client, prompt_id, monkeypatch):
        from src.auth import caller_identity

        client.put(f"/prompts/{prompt_id}/versions/1", json=_body())
        _walk(client, prompt_id, 1, "CANARY")
        monkeypatch.setattr(caller_identity, "get_caller_roles_from_context", lambda: ())
        before = _published_count(prompt_id, "false")
        resp = client.post(
            f"/prompts/{prompt_id}/versions/1/status",
            json={"status": "PUBLISHED", "actor_card_id": CARD},
        )
        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": {"error": "prompt_publish_role_required"}}
        assert client.get(f"/prompts/{prompt_id}/versions/1").json()["status"] == "CANARY"
        assert _published_count(prompt_id, "false") == before
