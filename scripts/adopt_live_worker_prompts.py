"""Adopt each platform worker's LIVE system prompt as a PUBLISHED prompt_versions row.

Owner rule: NO SEEDING of system assets. A worker's prompt lives in Postgres
(``workers.config->>'system_prompt'``) and is continuously improved there; a git
copy may be OLDER. So the version published here is the text read from the
database at run time — this script never opens a prompt file.

Report-first:

    # default: dry run, writes NOTHING, prints per-tenant counts
    uv run python scripts/adopt_live_worker_prompts.py

    # apply: a separate, explicit invocation, refused without the confirmation
    ADOPT_CONFIRM=yes uv run python scripts/adopt_live_worker_prompts.py --apply

Scope: ``workers`` rows ``WHERE worker_name LIKE 'worker-%'``. Only tenants in
``PLATFORM_TENANT_IDS`` are ever written. Every other row is only COUNTED:

* ``skipped_customer``      — a realm tenant (``TENANT_UUIDS``) that is not platform;
* ``skipped_non_platform``  — anything else (legacy / unregistered tenant ids).

Per platform row, in order:

* a ``worker_prompt_binding`` exists           -> ``already_bound`` (skip)
* live ``system_prompt`` is NULL / blank      -> ``empty_prompt`` (skip)
* ``prompt_id`` (see below) fails validation   -> ``invalid_prompt_id`` (skip)
* no version of ``prompt_id`` exists           -> ``would_create``: insert v1
  DRAFT, DRAFT -> PUBLISHED with reason ``adopt_live`` (initial-create
  semantics, actor ``migration``/``adopt_live_worker_prompts``), bind
* the prompt exists and is owned by ANOTHER tenant -> ``conflict`` (035)
* the prompt HAS publish history (any version ever PUBLISHED/ARCHIVED):
  exactly one PUBLISHED version whose system sha equals the live sha
  -> ``would_bind`` (bind only); anything else -> ``conflict``
* versions exist, NONE ever published, all DRAFT/DISCARDED (typically
  prompt-publisher's v1 DRAFT) -> ``adopt_as_next``: insert version
  ``max(version)+1`` built ONLY from the live text, DRAFT -> PUBLISHED with
  reason ``adopt_live``, bind. Every existing row is left exactly as it was.
* anything else (e.g. a never-published CANARY) -> ``conflict``: reported,
  NOTHING written

Why ``adopt_as_next`` and not "publish the existing DRAFT in place": the
publisher DRAFTs are built from git and carry git-sourced fields (``question``,
``output_schema``, ``git_sha``) even when their ``system`` sha equals the live
text. Publishing one would make GET /prompts/{id} serve git content as a
PUBLISHED system asset, and every later version would inherit it — seeding
through the back door. Only the LIVE text may become published, so the new
version carries ``system`` from ``workers.config`` and NULL/empty everything
else. And the existing drafts belong to someone else (the publisher, a human):
this job never rewrites, transitions or discards them, whatever their sha.

Concurrency: the insert races cleanly (``ON CONFLICT DO NOTHING`` -> conflict),
and the DRAFT -> PUBLISHED step re-checks "no publish history" under the
repository's ``FOR UPDATE`` lock on every row of the prompt, so a concurrent run
that published first turns this one into a conflict and its transaction rolls
back.

Before migration 034 (``worker_prompt_binding`` absent) the dry run still runs:
it reports ``binding_table_absent=true`` and treats every row as unbound.
``--apply`` refuses (exit 2) until the table exists. Before migration 035
(``prompt_owner`` absent) the dry run skips the owner check; ``--apply``
refuses until 035 is applied.

OWNERSHIP (migration 035, strict tenant separation). Every prompt this job
creates is OWNED by the worker row's tenant (``prompt_owner``), passed
explicitly to ``repository.upsert_version``; the binding is written for the
same tenant, and the DB's composite FK refuses a binding whose tenant is not the
prompt's owner. A prompt is therefore NEVER shared across tenants, even when
two tenants' live text is byte-identical: each gets its own prompt, owned by
itself. An existing prompt owned by another tenant is a ``conflict``.

``prompt_id`` naming (just NAMING — ownership is the column):

* the system tenant ``aaaaaaaa-…0001`` keeps the bare ``worker_name`` (its
  existing prompt-publisher drafts, all owned by …0001 since 035's backfill);
* every other platform tenant uses ``f"{worker_name}.t{tenant_tag(tenant_id)}"``
  — ``tenant_id[:8]``, widened to ``<8>-<last 4>`` where platform tenants share
  the 8-char prefix (see :func:`tenant_tag`; the bare ``[:8]`` collides for
  ``aaaaaaaa-…0001..0005``).

The dry run's "an earlier row of this run would create it" memo is keyed by
``(tenant, prompt_id)``: it never lets one tenant's planned prompt satisfy
another tenant's row.

One transaction per worker. Idempotent: a re-run reports ``already_bound``.
The last stdout line is a JSON object with the totals, for the deploy report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Imports must be after path manipulation
from src.api.prompts import metrics as prompt_metrics
from src.api.prompts import repository
from src.api.prompts.actor import Actor
from src.api.prompts.hashing import sha256_text, version_payload_sha256
from src.api.prompts.models import validate_prompt_id
from src.api.tenant_constants import (
    is_platform_tenant,
    platform_tenant_ids,
    system_tenant_id,
    tenant_uuids,
)
from src.db.db_config import close_pg_pool, get_pg_connection, init_pg_pool

JOB_NAME = "adopt_live_worker_prompts"
ACTOR = Actor(sub=JOB_NAME, kind="migration")
CONFIRM_ENV = "ADOPT_CONFIRM"
DEFAULT_NAME_LIKE = "worker-%"
FIRST_VERSION = 1

COUNTERS = (
    "would_create",
    "would_bind",
    "adopt_as_next",
    "already_bound",
    "empty_prompt",
    "invalid_prompt_id",
    "conflict",
    "skipped_non_platform",
    "skipped_customer",
)

# ⛔ NO MODULE-LEVEL DERIVATION OF THE TENANT SETS. They are environment-supplied
# now (``PLATFORM_TENANT_IDS`` / ``TENANT_UUIDS`` / ``SYSTEM_TENANT_ID``), and a
# module-level read would either bake a stale copy (import happened before the
# operator's env applied) or refuse to import at all. ``_validate_tenant_config``
# runs from ``main()``, where an unconfigured deployment is a real error.


def _customer_tenant_ids() -> frozenset[str]:
    """Realm tenants that are NOT platform: customers. Never written, only counted."""
    return frozenset(tenant_uuids().values()) - platform_tenant_ids()


def _validate_tenant_config() -> None:
    """Refuse an unconfigured or ambiguous tenant config before touching a row."""
    platform = platform_tenant_ids()
    if not platform:
        raise SystemExit(
            "PLATFORM_TENANT_IDS is unset (fail-closed default) — nothing to adopt. "
            "Export the estate value on the Deployment before running this job."
        )
    system = system_tenant_id()
    if system not in platform:
        raise SystemExit(
            f"SYSTEM_TENANT_ID={system!r} is not in PLATFORM_TENANT_IDS — the system tenant "
            "must be a platform tenant (its prompts keep the bare worker name)."
        )
    tags = {tenant_tag(t) for t in platform}
    if len(tags) != len(platform):
        raise SystemExit(
            "tenant_tag() is not unique across the configured platform set — two tenants "
            f"would collide on one prompt_id suffix: {sorted(tags)}"
        )


def tenant_tag(tenant_id: str) -> str:
    """The per-tenant prompt_id suffix: ``tenant_id[:8]``, widened when ambiguous.

    ⚠️ The contract's bare ``tenant_id[:8]`` COLLIDES: five of the six platform
    tenants are ``aaaaaaaa-0000-4000-8000-00000000000N`` and share ``aaaaaaaa``,
    so two tenants with different text would get the SAME id and the second
    would be refused as a conflict. Where the 8-char prefix is shared by another
    platform tenant the last 4 hex digits are appended (``aaaaaaaa-0003``);
    the AA tenant keeps ``aaaaaaaa``. Uniqueness is asserted at import.
    """
    tid = tenant_id.strip().lower()
    head = tid[:8]
    if sum(1 for p in platform_tenant_ids() if p[:8] == head) > 1:
        return f"{head}-{tid[-4:]}"
    return head


@dataclass
class WorkerRow:
    tenant_id: str
    worker_name: str
    system: str | None

    @property
    def has_prompt(self) -> bool:
        return bool(self.system and self.system.strip())


@dataclass
class Report:
    apply: bool
    per_tenant: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
    )
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    adopted: list[dict[str, Any]] = field(default_factory=list)
    binding_table_absent: bool = False

    def count(self, tenant_id: str, key: str) -> None:
        self.per_tenant[tenant_id][key] += 1

    def totals(self) -> dict[str, int]:
        out = dict.fromkeys(COUNTERS, 0)
        for counts in self.per_tenant.values():
            for k, v in counts.items():
                out[k] += v
        return out


class BindingTableAbsent(RuntimeError):
    """``worker_prompt_binding`` does not exist (migration 034 not applied)."""


class _Skip(Exception):
    """Roll back this worker's transaction and count it under ``counter``."""

    def __init__(self, counter: str, detail: str) -> None:
        super().__init__(detail)
        self.counter = counter
        self.detail = detail


def tenant_class(tenant_id: str) -> str:
    tid = tenant_id.strip().lower()
    if is_platform_tenant(tid):
        return "platform"
    if tid in _customer_tenant_ids():
        return "customer"
    return "non_platform"


def prompt_id_for(tenant_id: str, worker_name: str) -> str:
    """The prompt a platform worker's live text is adopted as (NAMING only)."""
    if tenant_id.strip().lower() == system_tenant_id():
        return worker_name
    return f"{worker_name}.t{tenant_tag(tenant_id)}"


def plan_prompt_ids(rows: list[WorkerRow]) -> dict[tuple[str, str], str]:
    """``(tenant_id, worker_name) -> prompt_id`` for every platform row with a prompt.

    Never shared across tenants: identical text in two tenants still yields two
    prompts, each owned by its own tenant (migration 035).
    """
    return {
        (r.tenant_id, r.worker_name): prompt_id_for(r.tenant_id, r.worker_name)
        for r in rows
        if tenant_class(r.tenant_id) == "platform" and r.has_prompt
    }


async def load_rows(name_like: str) -> list[WorkerRow]:
    async with get_pg_connection() as conn:
        rows = await conn.fetch(
            "SELECT tenant_id::text AS tenant_id, worker_name, "
            "config->>'system_prompt' AS system "
            "FROM workers WHERE worker_name LIKE $1 ORDER BY worker_name, tenant_id",
            name_like,
        )
    return [
        WorkerRow(tenant_id=str(r["tenant_id"]), worker_name=r["worker_name"], system=r["system"])
        for r in rows
    ]


async def binding_table_exists() -> bool:
    """Whether ``worker_prompt_binding`` (migration 034) exists on this database."""
    async with get_pg_connection() as conn:
        found = await conn.fetchval("SELECT to_regclass('worker_prompt_binding') IS NOT NULL")
    return bool(found)


async def owner_table_exists() -> bool:
    """Whether ``prompt_owner`` (migration 035) exists on this database."""
    async with get_pg_connection() as conn:
        found = await conn.fetchval("SELECT to_regclass('prompt_owner') IS NOT NULL")
    return bool(found)


async def _binding(conn: Any, row: WorkerRow) -> str | None:
    found = await conn.fetchrow(
        "SELECT prompt_id FROM worker_prompt_binding WHERE tenant_id = $1 AND worker_name = $2",
        row.tenant_id,
        row.worker_name,
    )
    return None if found is None else str(found["prompt_id"])


async def _classify_version(
    conn: Any,
    tenant_id: str,
    prompt_id: str,
    system_sha: str,
    planned: dict[tuple[str, str], str],
    *,
    check_owner: bool = True,
) -> tuple[str, int]:
    """``(action, version)``: ``would_create`` | ``would_bind`` | ``adopt_as_next``.

    ``version`` is the version the action writes (or binds to); raises
    :class:`_Skip` for everything else. See the module docstring for the rules.
    ``planned`` (dry run only) is keyed ``(tenant, prompt_id)``: a prompt another
    tenant's row would create is NOT this tenant's.
    """
    key = (tenant_id, prompt_id)
    if key in planned:  # dry run: an earlier row of THIS tenant would publish it
        if planned[key] == system_sha:
            return "would_bind", 0  # version not yet known in a dry run; unused there
        raise _Skip("conflict", f"planned version of {prompt_id} has a different system sha")
    owner = (
        await conn.fetchval(
            "SELECT owner_tenant_id FROM prompt_owner WHERE prompt_id = $1", prompt_id
        )
        if check_owner
        else None
    )
    if owner is not None and str(owner).strip().lower() != tenant_id.strip().lower():
        raise _Skip("conflict", f"{prompt_id} is owned by another tenant ({owner})")
    versions = await conn.fetch(
        "SELECT version, system_prompt_sha256, status, activated_at FROM prompt_versions "
        "WHERE prompt_id = $1 ORDER BY version",
        prompt_id,
    )
    if not versions:
        return "would_create", FIRST_VERSION
    statuses = {
        int(v["version"]): repository.effective_status(v["status"], v["activated_at"])
        for v in versions
    }
    if await repository._has_publish_history(conn, prompt_id):
        published = [v for v in versions if statuses[int(v["version"])] == repository.PUBLISHED]
        if len(published) == 1 and str(published[0]["system_prompt_sha256"]).strip() == system_sha:
            return "would_bind", int(published[0]["version"])
        raise _Skip(
            "conflict",
            f"{prompt_id} has publish history and no single PUBLISHED version with the live text",
        )
    blocking = sorted(
        f"{n}:{st}"
        for n, st in statuses.items()
        if st not in (repository.DRAFT, repository.DISCARDED)
    )
    if blocking:
        raise _Skip("conflict", f"{prompt_id} has unpublished non-DRAFT versions {blocking}")
    # Never-published DRAFT/DISCARDED rows (prompt-publisher's git DRAFTs): left
    # untouched; the live text becomes the NEXT version.
    return "adopt_as_next", max(statuses) + 1


async def _publish_live(
    conn: Any, row: WorkerRow, prompt_id: str, version: int, system_sha: str
) -> None:
    """Insert ``(prompt_id, version)`` from the LIVE text only, then DRAFT -> PUBLISHED.

    ``system`` is the live ``workers.config`` text; ``question``/``template``/
    ``output_schema``/``git_sha`` are NULL and ``purpose_tags`` empty, because no
    git-sourced field may become a published system asset. ``worker_names`` is
    the worker the text was read from (provenance, not content).
    """
    payload = {
        "question": None,
        "system": row.system or "",
        "template": None,
        "output_schema": None,
        "worker_names": [row.worker_name],
        "purpose_tags": [],
    }
    _, written = await repository.upsert_version(
        prompt_id=prompt_id,
        version=version,
        sha256=version_payload_sha256(**payload),
        system_prompt_sha256=system_sha,
        git_sha=None,
        actor=ACTOR,
        owner_tenant_id=row.tenant_id,
        conn=conn,
        **payload,
    )
    if not written:
        raise _Skip("conflict", f"{prompt_id}@{version} appeared concurrently")
    # adopt_live takes the initial-create branch: it locks every row of the
    # prompt FOR UPDATE and refuses (InitialCreateNotAllowed) if any version was
    # published meanwhile — the caller then rolls this whole worker back.
    published, _ = await repository.transition(
        prompt_id,
        version,
        repository.PUBLISHED,
        actor=ACTOR,
        reason=repository.REASON_ADOPT_LIVE,
        conn=conn,
    )
    if published is None or published["status"] != repository.PUBLISHED:
        raise _Skip("conflict", f"{prompt_id}@{version} did not reach PUBLISHED")


async def _apply_one(conn: Any, row: WorkerRow, prompt_id: str, system_sha: str) -> str:
    """Create/publish/bind on ``conn`` (the caller's open transaction)."""
    action, version = await _classify_version(conn, row.tenant_id, prompt_id, system_sha, {})
    if action in ("would_create", "adopt_as_next"):
        await _publish_live(conn, row, prompt_id, version, system_sha)
    bound = await conn.fetchrow(
        "INSERT INTO worker_prompt_binding (tenant_id, worker_name, prompt_id, bound_by) "
        "VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING RETURNING prompt_id",
        row.tenant_id,
        row.worker_name,
        prompt_id,
        JOB_NAME,
    )
    if bound is None:
        raise _Skip("already_bound", "bound concurrently")
    return action


async def adopt(*, apply: bool, name_like: str = DEFAULT_NAME_LIKE) -> Report:
    """Run the adoption over an initialised pool. Dry run unless ``apply``."""
    report = Report(apply=apply)
    has_bindings = await binding_table_exists()
    if not has_bindings:
        if apply:
            raise BindingTableAbsent(
                "worker_prompt_binding does not exist: apply migration 034 first"
            )
        # Dry run before 034: report it and treat every row as unbound.
        report.binding_table_absent = True
    has_owner = await owner_table_exists()
    if not has_owner and apply:
        raise BindingTableAbsent("prompt_owner does not exist: apply migration 035 first")
    rows = await load_rows(name_like)
    ids = plan_prompt_ids(rows)
    # dry run only: (tenant, prompt_id) -> system sha that tenant's row would create
    planned: dict[tuple[str, str], str] = {}

    for row in rows:
        cls = tenant_class(row.tenant_id)
        if cls != "platform":
            report.count(row.tenant_id, f"skipped_{cls}")
            continue
        if has_bindings:
            async with get_pg_connection() as conn:
                if await _binding(conn, row) is not None:
                    report.count(row.tenant_id, "already_bound")
                    continue
        if not row.has_prompt:
            report.count(row.tenant_id, "empty_prompt")
            continue
        prompt_id = ids[(row.tenant_id, row.worker_name)]
        try:
            validate_prompt_id(prompt_id)
        except ValueError as exc:
            report.count(row.tenant_id, "invalid_prompt_id")
            report.conflicts.append(_detail(row, prompt_id, "invalid_prompt_id", str(exc)))
            continue
        system_sha = sha256_text(row.system or "")
        try:
            if apply:
                async with get_pg_connection() as conn, conn.transaction():
                    if await _binding(conn, row) is not None:
                        raise _Skip("already_bound", "bound since the scan")
                    action = await _apply_one(conn, row, prompt_id, system_sha)
                # Counted only here, AFTER the transaction committed: a _Skip
                # (e.g. "already_bound") raised inside it rolls the publish back
                # and must not count. bound="false" always: the publish precedes
                # the binding insert, and the text published IS the worker's live
                # text, so no worker's prompt changes. NB this is a separate CLI
                # process, so the increment lands in ITS registry, which nothing
                # scrapes (see src/api/prompts/metrics.py).
                if action in ("would_create", "adopt_as_next"):
                    prompt_metrics.count_published(prompt_id, bound=False)
                report.adopted.append(_detail(row, prompt_id, action, system_sha))
            else:
                async with get_pg_connection() as conn:
                    action, _ = await _classify_version(
                        conn, row.tenant_id, prompt_id, system_sha, planned, check_owner=has_owner
                    )
                if action in ("would_create", "adopt_as_next"):
                    # A later row of the SAME tenant and prompt_id would find it PUBLISHED.
                    planned[(row.tenant_id, prompt_id)] = system_sha
        except _Skip as skip:
            report.count(row.tenant_id, skip.counter)
            if skip.counter == "conflict":
                report.conflicts.append(_detail(row, prompt_id, "conflict", skip.detail))
            continue
        except (
            repository.InitialCreateNotAllowed,
            repository.PromptVersionConflict,
            repository.PromptOwnerMismatch,
        ) as exc:
            report.count(row.tenant_id, "conflict")
            report.conflicts.append(_detail(row, prompt_id, "conflict", str(exc)))
            continue
        report.count(row.tenant_id, action)
    return report


def _detail(row: WorkerRow, prompt_id: str, kind: str, detail: str) -> dict[str, Any]:
    return {
        "tenant_id": row.tenant_id,
        "worker_name": row.worker_name,
        "prompt_id": prompt_id,
        "kind": kind,
        "detail": detail,
    }


def render(report: Report) -> str:
    cols = ("tenant_id", "class", *COUNTERS)
    lines = [
        f"mode: {'APPLY' if report.apply else 'DRY-RUN (nothing written)'}",
        *(
            [
                "binding_table_absent: worker_prompt_binding missing (034 not applied); "
                "every row treated as unbound"
            ]
            if report.binding_table_absent
            else []
        ),
        "\t".join(cols),
    ]
    for tid in sorted(report.per_tenant):
        counts = report.per_tenant[tid]
        lines.append("\t".join([tid, tenant_class(tid), *(str(counts[c]) for c in COUNTERS)]))
    totals = report.totals()
    lines.append("\t".join(["TOTAL", "", *(str(totals[c]) for c in COUNTERS)]))
    for c in report.conflicts:
        lines.append(
            f"CONFLICT {c['tenant_id']} {c['worker_name']} -> {c['prompt_id']}: {c['detail']}"
        )
    lines.append(
        json.dumps(
            {
                "job": JOB_NAME,
                "mode": "apply" if report.apply else "dry_run",
                "totals": totals,
                "conflicts": len(report.conflicts),
                "adopted": len(report.adopted),
                "binding_table_absent": report.binding_table_absent,
            },
            sort_keys=True,
        )
    )
    return "\n".join(lines)


async def amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report only (default)")
    mode.add_argument(
        "--apply", action="store_true", help=f"write; requires {CONFIRM_ENV}=yes in the env"
    )
    args = parser.parse_args(argv)
    _validate_tenant_config()
    if args.apply and os.environ.get(CONFIRM_ENV) != "yes":
        print(
            f"refusing --apply: set {CONFIRM_ENV}=yes to confirm (run the dry run first)",
            file=sys.stderr,
        )
        return 2
    await init_pg_pool()
    try:
        if args.apply and not (await binding_table_exists() and await owner_table_exists()):
            print(
                "refusing --apply: table worker_prompt_binding or prompt_owner does not "
                "exist (migration 034 / 035 not applied); the dry run still works",
                file=sys.stderr,
            )
            return 2
        report = await adopt(apply=bool(args.apply))
    finally:
        await close_pg_pool()
    print(render(report))
    # Conflicts are a REPORT, not a failure: they are listed above and counted
    # in the JSON line; nothing was written for them.
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
