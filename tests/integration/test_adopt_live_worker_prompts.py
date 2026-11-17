"""``scripts/adopt_live_worker_prompts.py`` + the ``adopt_live`` repository path, on real PG 18.

Every fixture row uses a per-test ``worker-adopttest-<hex>-`` prefix, and the
job is run with ``name_like`` scoped to it, so counts are exact whatever else
the database holds.

Run locally in the shape CI uses::

    export DATABASE_URL="postgresql+psycopg://user:pw@127.0.0.1:PORT/worker_management_ci"  # pragma: allowlist secret
    export POSTGRES_SSLMODE=disable
    uv run pytest tests/integration/test_adopt_live_worker_prompts.py -m pg -v --no-cov
"""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from scripts import adopt_live_worker_prompts as job
from src.api.prompts import repository
from src.api.prompts.actor import Actor
from src.api.prompts.hashing import sha256_text, version_payload_sha256
from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool

from .pg_guard import PG_CONNECT_ERRORS, apply_migrations, require_database_url, unreachable_pg

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.pg

P1 = "aaaaaaaa-0000-4000-8000-000000000001"  # platform (n8n-dev)
P3 = "aaaaaaaa-0000-4000-8000-000000000003"  # platform (george-dev)
CUSTOMER = "aaaaaaaa-0000-4000-8000-0000000000a7"  # cdmsfirst (in the synthetic label map)
TEST_ACTOR = Actor(sub="test_adopt_live", kind="migration")


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


@pytest_asyncio.fixture
async def prefix(pg_pool) -> AsyncIterator[str]:
    """A unique ``worker-adopttest-<hex>-`` name prefix; every row under it is purged."""
    pfx = f"worker-adopttest-{uuid.uuid4().hex[:10]}-"
    try:
        yield pfx
    finally:
        await _purge(pfx)


async def _purge(pfx: str) -> None:
    like = pfx + "%"
    async with get_pg_connection() as conn, conn.transaction():
        await conn.execute("DELETE FROM worker_prompt_binding WHERE worker_name LIKE $1", like)
        await conn.execute("ALTER TABLE prompt_version_transitions DISABLE TRIGGER USER")
        await conn.execute("DELETE FROM prompt_version_transitions WHERE prompt_id LIKE $1", like)
        await conn.execute("ALTER TABLE prompt_version_transitions ENABLE TRIGGER USER")
        await conn.execute("DELETE FROM prompt_versions WHERE prompt_id LIKE $1", like)
        await conn.execute("DELETE FROM prompt_owner WHERE prompt_id LIKE $1", like)
        await conn.execute("DELETE FROM workers WHERE worker_name LIKE $1", like)


async def _worker(tenant_id: str, worker_name: str, system: str | None) -> None:
    config = None if system is None else json.dumps({"system_prompt": system})
    async with get_pg_connection() as conn:
        await conn.execute(
            "INSERT INTO workers (worker_id, worker_name, tenant_id, tenant_name, subject_id, "
            "visibility, is_agent, status, config) "
            "VALUES ($1, $2, $3::uuid, $4, $4, 'tenant', false, 'active', $5::jsonb)",
            f"wid-{uuid.uuid4().hex}",
            worker_name,
            tenant_id,
            tenant_id[:36],
            config,
        )


async def _counts(like: str) -> dict[str, int]:
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            "SELECT "
            "(SELECT count(*) FROM prompt_versions WHERE prompt_id LIKE $1) AS pv, "
            "(SELECT count(*) FROM prompt_version_transitions WHERE prompt_id LIKE $1) AS tr, "
            "(SELECT count(*) FROM worker_prompt_binding WHERE worker_name LIKE $1) AS wb, "
            "(SELECT count(*) FROM workers WHERE worker_name LIKE $1) AS w, "
            "(SELECT md5(string_agg(worker_id || coalesce(config::text, ''), ',' "
            "  ORDER BY worker_id)) FROM workers WHERE worker_name LIKE $1) AS wh",
            like,
        )
    return dict(row)


async def _bindings(like: str) -> dict[tuple[str, str], str]:
    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            "SELECT tenant_id, worker_name, prompt_id, bound_by FROM worker_prompt_binding "
            "WHERE worker_name LIKE $1",
            like,
        )
    for r in rows:
        assert r["bound_by"] == "adopt_live_worker_prompts"
    return {(str(r["tenant_id"]), r["worker_name"]): r["prompt_id"] for r in rows}


async def _owners(like: str) -> dict[str, str]:
    """prompt_id -> owner_tenant_id (migration 035) for every prompt under ``like``."""
    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            "SELECT prompt_id, owner_tenant_id FROM prompt_owner WHERE prompt_id LIKE $1", like
        )
    return {r["prompt_id"]: str(r["owner_tenant_id"]) for r in rows}


async def _seed_mixed(pfx: str) -> dict[str, Any]:
    """Platform (P1, P3), non-platform (random uuid) and customer rows."""
    legacy = str(uuid.uuid4())
    shared, split, empty = pfx + "shared", pfx + "split", pfx + "empty"
    await _worker(P1, shared, "You are shared.")
    await _worker(P3, shared, "You are shared.")
    await _worker(P1, split, "You are split, tenant one.")
    await _worker(P3, split, "You are split, tenant three.")
    await _worker(P1, empty, "   ")
    await _worker(P3, empty, None)
    await _worker(legacy, shared, "legacy text")
    await _worker(CUSTOMER, shared, "customer text")
    return {"legacy": legacy, "shared": shared, "split": split, "empty": empty}


# ---------------------------------------------------------------------------
# (a) dry run writes nothing and counts correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_writes_nothing_and_counts(prefix):
    names = await _seed_mixed(prefix)
    like = prefix + "%"
    before = await _counts(like)

    report = await job.adopt(apply=False, name_like=like)

    assert await _counts(like) == before
    # 035: nothing is shared across tenants, identical text or not.
    # P1: shared + split would_create (bare names), empty empty_prompt.
    # P3: shared + split would_create (its OWN .t<tag> prompts), empty.
    pt = report.per_tenant
    assert (pt[P1]["would_create"], pt[P1]["would_bind"], pt[P1]["empty_prompt"]) == (2, 0, 1)
    assert (pt[P3]["would_create"], pt[P3]["would_bind"], pt[P3]["empty_prompt"]) == (2, 0, 1)
    assert pt[names["legacy"]]["skipped_non_platform"] == 1
    assert pt[CUSTOMER]["skipped_customer"] == 1
    totals = report.totals()
    assert totals["would_create"] == 4
    assert totals["would_bind"] == 0
    assert totals["already_bound"] == 0
    assert totals["conflict"] == 0
    assert totals["skipped_non_platform"] == 1
    assert totals["skipped_customer"] == 1
    last = json.loads(job.render(report).splitlines()[-1])
    assert last["mode"] == "dry_run"
    assert last["totals"] == totals


# ---------------------------------------------------------------------------
# (b) apply creates v1 PUBLISHED, ledger adopt_live / migration, binding
# (d) every tenant gets its OWN prompt, owned by itself (035) — identical text too
# (e) customer and non-platform rows untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_publishes_live_text_ledgers_and_binds(prefix):
    names = await _seed_mixed(prefix)
    like = prefix + "%"
    workers_before = (await _counts(like))["wh"]

    report = await job.adopt(apply=True, name_like=like)
    assert report.conflicts == []

    shared, split = names["shared"], names["split"]
    # P1 (the system tenant) keeps the bare name; P3 shares P1's 8-char prefix
    # "aaaaaaaa", so its tag is widened (a bare [:8] would collide).
    assert job.tenant_tag(P3) == "aaaaaaaa-0003"
    shared_p3, split_p3 = f"{shared}.taaaaaaaa-0003", f"{split}.taaaaaaaa-0003"
    bindings = await _bindings(like)
    assert bindings[(P1, shared)] == shared
    assert bindings[(P3, shared)] == shared_p3
    assert bindings[(P1, split)] == split
    assert bindings[(P3, split)] == split_p3
    assert await _owners(like) == {shared: P1, split: P1, shared_p3: P3, split_p3: P3}
    assert (P1, names["empty"]) not in bindings
    assert (P3, names["empty"]) not in bindings
    # (e): no binding for customer / non-platform, workers rows unchanged.
    assert {t for t, _ in bindings} == {P1, P3}
    assert (await _counts(like))["wh"] == workers_before

    v1 = await repository.get_version(shared, 1)
    assert v1 is not None
    assert v1["status"] == repository.PUBLISHED
    assert v1["system"] == "You are shared."
    assert v1["system_prompt_sha256"] == sha256_text("You are shared.")

    ledger = await repository.get_transitions(shared, 1)
    assert ledger is not None
    assert [(r["from_status"], r["to_status"]) for r in ledger] == [
        (None, "DRAFT"),
        ("DRAFT", "PUBLISHED"),
    ]
    publish = ledger[-1]
    assert publish["reason"] == "adopt_live"
    assert publish["actor_kind"] == "migration"
    assert publish["actor_sub"] == "adopt_live_worker_prompts"


@pytest.mark.asyncio
async def test_identical_text_in_two_tenants_gets_two_prompts_each_owned_by_its_tenant(prefix):
    """Strict tenant separation (035): byte-identical live text in two platform
    tenants is NOT deduped into one prompt — a release for one tenant's worker
    must never change what the other's runs. Replaces the old "identical text
    shares one prompt_id" test, which encoded the rejected behaviour."""
    # The AA tenant's 8-char prefix is unique, so it keeps the bare [:8] tag.
    aa = "bbbbbbbb-0000-4000-8000-0000000000a5"
    same = prefix + "same"
    await _worker(P1, same, "Same text.")
    await _worker(P3, same, "Same text.")
    await _worker(aa, same, "Same text.")
    like = prefix + "%"

    dry = await job.adopt(apply=False, name_like=like)
    assert dry.totals()["would_create"] == 3
    assert dry.totals()["would_bind"] == 0

    report = await job.adopt(apply=True, name_like=like)
    assert report.conflicts == []
    assert report.totals()["would_create"] == 3
    assert report.totals()["would_bind"] == 0

    id_p1, id_p3, id_aa = same, f"{same}.taaaaaaaa-0003", f"{same}.tbbbbbbbb"
    assert await _bindings(like) == {(P1, same): id_p1, (P3, same): id_p3, (aa, same): id_aa}
    assert await _owners(like) == {id_p1: P1, id_p3: P3, id_aa: aa}
    for pid in (id_p1, id_p3, id_aa):
        v1 = await repository.get_version(pid, 1)
        assert v1["status"] == repository.PUBLISHED
        assert v1["system"] == "Same text."


@pytest.mark.asyncio
async def test_a_prompt_owned_by_another_tenant_is_a_conflict_and_untouched(prefix):
    """The would-be id already exists, owned by ANOTHER tenant: nothing written."""
    name = prefix + "squat"
    taken = f"{name}.taaaaaaaa-0003"
    await _draft(taken, "P1 wrote this.", owner=P1)
    await _worker(P3, name, "The LIVE text.")
    like = prefix + "%"

    dry = await job.adopt(apply=False, name_like=like)
    assert dry.totals()["conflict"] == 1
    report = await job.adopt(apply=True, name_like=like)

    assert report.totals()["conflict"] == 1
    assert "owned by another tenant" in report.conflicts[0]["detail"]
    assert (await _counts(like))["pv"] == 1
    assert await _bindings(like) == {}
    assert await _owners(like) == {taken: P1}


# ---------------------------------------------------------------------------
# (c) a re-run is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerun_is_a_noop(prefix):
    await _seed_mixed(prefix)
    like = prefix + "%"
    await job.adopt(apply=True, name_like=like)
    after_first = await _counts(like)

    report = await job.adopt(apply=True, name_like=like)

    assert await _counts(like) == after_first
    totals = report.totals()
    assert totals["already_bound"] == 4
    assert totals["would_create"] == totals["would_bind"] == totals["conflict"] == 0


# ---------------------------------------------------------------------------
# (f) existing versions: publish history -> conflict; never-published DRAFTs ->
#     adopt_as_next (live text becomes the NEXT version; the DRAFTs stay as-is)
# ---------------------------------------------------------------------------

GIT_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}
GIT_SHA = "a" * 40  # a fake git sha; low entropy so detect-secrets does not flag test data


async def _draft(
    prompt_id: str,
    system: str,
    *,
    question: str | None = "publisher draft",
    output_schema: dict[str, Any] | None = None,
    git_sha: str | None = None,
    owner: str = P1,
) -> None:
    payload = {
        "question": question,
        "system": system,
        "template": None,
        "output_schema": output_schema,
        "worker_names": [],
        "purpose_tags": [],
    }
    await repository.upsert_version(
        prompt_id=prompt_id,
        version=1,
        sha256=version_payload_sha256(**payload),
        system_prompt_sha256=sha256_text(system),
        git_sha=git_sha,
        actor=TEST_ACTOR,
        owner_tenant_id=owner,
        **payload,
    )


async def _git_draft(prompt_id: str, system: str) -> None:
    """A prompt-publisher-shaped v1 DRAFT: git-sourced question, output_schema, git_sha."""
    await _draft(
        prompt_id,
        system,
        question="A git-sourced question.",
        output_schema=GIT_SCHEMA,
        git_sha=GIT_SHA,
    )


async def _assert_adopted_as_v2(
    name: str,
    like: str,
    before_v1: dict[str, Any],
    ledger_v1: list[dict[str, Any]] | None,
    *,
    total_versions: int = 2,
) -> None:
    """v1 untouched (every field + its ledger); v2 PUBLISHED from the LIVE text only."""
    assert await repository.get_version(name, 1) == before_v1
    assert before_v1["status"] == repository.DRAFT
    assert await repository.get_transitions(name, 1) == ledger_v1
    v2 = await repository.get_version(name, 2)
    assert v2 is not None
    assert v2["status"] == repository.PUBLISHED
    assert v2["system"] == "The LIVE text."
    assert v2["system_prompt_sha256"] == sha256_text("The LIVE text.")
    # No git-sourced field is published.
    assert v2["question"] is None
    assert v2["output_schema"] is None
    assert v2["template"] is None
    assert v2["git_sha"] is None
    ledger = await repository.get_transitions(name, 2)
    assert [(r["from_status"], r["to_status"]) for r in ledger] == [
        (None, "DRAFT"),
        ("DRAFT", "PUBLISHED"),
    ]
    assert ledger[-1]["reason"] == "adopt_live"
    assert ledger[-1]["actor_kind"] == "migration"
    assert ledger[-1]["actor_sub"] == "adopt_live_worker_prompts"
    assert (await _counts(like))["pv"] == total_versions


@pytest.mark.asyncio
async def test_prompt_with_publish_history_is_a_conflict_and_untouched(prefix):
    name = prefix + "taken"
    await _draft(name, "An OLDER git copy.")
    await repository.transition(
        name, 1, repository.PUBLISHED, actor=TEST_ACTOR, reason=repository.INITIAL_CREATE
    )
    await _worker(P1, name, "The LIVE text.")
    before = await repository.get_version(name, 1)
    ledger_before = await repository.get_transitions(name, 1)
    like = prefix + "%"

    dry = await job.adopt(apply=False, name_like=like)
    assert dry.totals()["conflict"] == 1
    assert dry.totals()["adopt_as_next"] == 0

    report = await job.adopt(apply=True, name_like=like)

    assert report.totals()["conflict"] == 1
    assert report.totals()["adopt_as_next"] == 0
    assert report.conflicts[0]["prompt_id"] == name
    assert await repository.get_version(name, 1) == before
    assert before["status"] == repository.PUBLISHED
    assert await repository.get_transitions(name, 1) == ledger_before
    assert (await _counts(like))["pv"] == 1
    assert await _bindings(like) == {}


@pytest.mark.asyncio
async def test_same_text_publisher_draft_stays_untouched_and_live_is_adopted_as_next(prefix):
    """A same-sha git DRAFT is NOT published: its git fields must never become published."""
    name = prefix + "draftsame"
    await _git_draft(name, "The LIVE text.")
    await _worker(P1, name, "The LIVE text.")
    await _worker(P3, name, "The LIVE text.")
    like = prefix + "%"
    before = await repository.get_version(name, 1)
    ledger_before = await repository.get_transitions(name, 1)
    assert before["question"] == "A git-sourced question."
    assert before["output_schema"] == GIT_SCHEMA
    assert before["git_sha"] == GIT_SHA

    dry = await job.adopt(apply=False, name_like=like)
    # P1 (owner of the bare-name draft) adopts as v2; P3 gets its OWN prompt.
    assert dry.totals()["adopt_as_next"] == 1
    assert dry.totals()["would_create"] == 1
    assert dry.totals()["would_bind"] == 0
    assert dry.totals()["conflict"] == 0
    assert json.loads(job.render(dry).splitlines()[-1])["totals"]["adopt_as_next"] == 1
    assert (await _counts(like))["pv"] == 1

    report = await job.adopt(apply=True, name_like=like)

    assert report.conflicts == []
    assert report.totals()["adopt_as_next"] == 1
    assert report.totals()["would_bind"] == 0
    assert report.totals()["would_create"] == 1
    # v1 draft + v2 adopted (P1) + P3's own v1.
    await _assert_adopted_as_v2(name, like, before, ledger_before, total_versions=3)
    p3_id = f"{name}.taaaaaaaa-0003"
    assert await _bindings(like) == {(P1, name): name, (P3, name): p3_id}
    assert await _owners(like) == {name: P1, p3_id: P3}

    # Re-run: already bound, nothing written.
    after = await _counts(like)
    rerun = await job.adopt(apply=True, name_like=like)
    assert rerun.totals()["already_bound"] == 2
    assert await _counts(like) == after


@pytest.mark.asyncio
async def test_different_sha_publisher_draft_stays_untouched_and_live_is_adopted_as_next(prefix):
    """An OLDER git DRAFT is left alone; the LIVE text wins as v2."""
    name = prefix + "draftdiff"
    await _git_draft(name, "A publisher DRAFT that is not the live text.")
    await _worker(P1, name, "The LIVE text.")
    like = prefix + "%"
    before = await repository.get_version(name, 1)
    ledger_before = await repository.get_transitions(name, 1)

    report = await job.adopt(apply=True, name_like=like)

    assert report.conflicts == []
    assert report.totals()["adopt_as_next"] == 1
    assert report.totals()["conflict"] == 0
    await _assert_adopted_as_v2(name, like, before, ledger_before)
    assert await _bindings(like) == {(P1, name): name}


# ---------------------------------------------------------------------------
# before migration 034: worker_prompt_binding absent
# (simulated by patching the existence probe; dropping the real table would
# race other tests sharing the CI database)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_without_the_binding_table_reports_it_and_does_not_crash(monkeypatch, prefix):
    name = prefix + "nobind"
    await _worker(P1, name, "The LIVE text.")
    monkeypatch.setattr(job, "binding_table_exists", AsyncMock(return_value=False))

    report = await job.adopt(apply=False, name_like=prefix + "%")

    assert report.binding_table_absent is True
    assert report.totals()["would_create"] == 1
    assert json.loads(job.render(report).splitlines()[-1])["binding_table_absent"] is True
    with pytest.raises(job.BindingTableAbsent):
        await job.adopt(apply=True, name_like=prefix + "%")


@pytest.mark.asyncio
async def test_apply_without_the_binding_table_is_refused(monkeypatch, prefix):
    monkeypatch.setenv(job.CONFIRM_ENV, "yes")
    monkeypatch.setattr(job, "binding_table_exists", AsyncMock(return_value=False))
    # The module-scoped pool stays open: amain must not init/close it here.
    monkeypatch.setattr(job, "init_pg_pool", AsyncMock())
    monkeypatch.setattr(job, "close_pg_pool", AsyncMock())
    adopt = AsyncMock()
    monkeypatch.setattr(job, "adopt", adopt)
    assert await job.amain(["--apply"]) == 2
    adopt.assert_not_awaited()


# ---------------------------------------------------------------------------
# --apply confirmation gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_without_confirmation_is_refused(monkeypatch, prefix):
    monkeypatch.delenv(job.CONFIRM_ENV, raising=False)
    assert await job.amain(["--apply"]) == 2


# ---------------------------------------------------------------------------
# repository: the adopt_live reason (INITIAL_CREATE_REASONS) and conn=
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adopt_live_publishes_a_draft_without_history_on_the_callers_conn(prefix):
    pid = prefix + "repo-ok"
    await _draft(pid, "v1 text")
    async with get_pg_connection() as conn, conn.transaction():
        row, from_status = await repository.transition(
            pid, 1, repository.PUBLISHED, actor=TEST_ACTOR, reason="adopt_live", conn=conn
        )
    assert from_status == repository.DRAFT
    assert row["status"] == repository.PUBLISHED
    ledger = await repository.get_transitions(pid, 1)
    assert ledger[-1]["reason"] == "adopt_live"
    assert (ledger[-1]["from_status"], ledger[-1]["to_status"]) == ("DRAFT", "PUBLISHED")


@pytest.mark.asyncio
async def test_adopt_live_is_refused_with_publish_history_like_initial_create(prefix):
    pid = prefix + "repo-hist"
    await _draft(pid, "v1 text")
    await repository.transition(
        pid, 1, repository.PUBLISHED, actor=TEST_ACTOR, reason=repository.INITIAL_CREATE
    )
    payload = {
        "question": None,
        "system": "v2 text",
        "template": None,
        "output_schema": None,
        "worker_names": [],
        "purpose_tags": [],
    }
    await repository.upsert_version(
        prompt_id=pid,
        version=2,
        sha256=version_payload_sha256(**payload),
        system_prompt_sha256=sha256_text("v2 text"),
        git_sha=None,
        actor=TEST_ACTOR,
        owner_tenant_id=P1,
        **payload,
    )
    for reason in (repository.INITIAL_CREATE, "adopt_live"):
        with pytest.raises(repository.InitialCreateNotAllowed) as exc:
            await repository.transition(
                pid, 2, repository.PUBLISHED, actor=TEST_ACTOR, reason=reason
            )
        assert exc.value.reason == repository.INITIAL_CREATE_HISTORY
    assert (await repository.get_version(pid, 2))["status"] == repository.DRAFT


@pytest.mark.asyncio
async def test_a_refusal_on_the_callers_conn_rolls_back_with_the_callers_transaction(prefix):
    """conn= opens no transaction: the upsert is undone when the caller's tx aborts."""
    pid = prefix + "repo-rb"
    payload = {
        "question": None,
        "system": "rb text",
        "template": None,
        "output_schema": None,
        "worker_names": [],
        "purpose_tags": [],
    }

    class _Abort(Exception):
        pass

    with pytest.raises(_Abort):
        async with get_pg_connection() as conn, conn.transaction():
            await repository.upsert_version(
                prompt_id=pid,
                version=1,
                sha256=version_payload_sha256(**payload),
                system_prompt_sha256=sha256_text("rb text"),
                git_sha=None,
                actor=TEST_ACTOR,
                owner_tenant_id=P1,
                conn=conn,
                **payload,
            )
            raise _Abort
    assert await repository.get_version(pid, 1) is None


def _adopted_count(prompt_id: str, bound: str) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(
        "wm_prompt_published_bound_total", {"prompt_id": prompt_id, "bound": bound}
    )
    return value or 0.0


@pytest.mark.asyncio
async def test_adoption_counts_a_publish_only_after_its_commit(monkeypatch, prefix):
    """``wm_prompt_published_bound_total``: the adoption counts after its
    per-worker transaction commits, always ``bound="false"``; a worker rolled
    back by ``_Skip("already_bound")`` AFTER the publish counts nothing."""
    name = prefix + "count"
    await _worker(P1, name, "You are counted.")
    like = prefix + "%"
    real_apply_one = job._apply_one

    async def _publish_then_lose_the_race(conn, row, prompt_id, system_sha):
        await real_apply_one(conn, row, prompt_id, system_sha)  # publish + bind
        raise job._Skip("already_bound", "bound concurrently")

    monkeypatch.setattr(job, "_apply_one", _publish_then_lose_the_race)
    f0, t0 = _adopted_count(name, "false"), _adopted_count(name, "true")
    rolled = await job.adopt(apply=True, name_like=like)
    assert rolled.totals()["already_bound"] == 1
    assert await repository.get_version(name, 1) is None  # the publish rolled back
    assert _adopted_count(name, "false") == f0

    monkeypatch.setattr(job, "_apply_one", real_apply_one)
    report = await job.adopt(apply=True, name_like=like)
    assert report.conflicts == []
    assert (await repository.get_version(name, 1))["status"] == repository.PUBLISHED
    assert _adopted_count(name, "false") == f0 + 1
    assert _adopted_count(name, "true") == t0
