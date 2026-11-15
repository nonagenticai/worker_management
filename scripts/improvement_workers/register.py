"""Register improvement-loop meta-workers in worker_management.

Two loops are registered from this single script:

* Plan 5 (workflow improvement loop) -- 5 workers under
  ``scripts/improvement_workers/workers/workflow_loop.py``, owned by the
  ``__system_workflow__`` internal tenant.
* Plan 6 (prompt improvement loop) -- 3 workers under
  ``scripts/improvement_workers/workers/prompt_loop.py``, owned by the
  ``__system_prompt__`` internal tenant.

Each worker carries a placeholder ``system_prompt`` and a default
``output_schema``. Plan 6 itself rewrites prompts in-place via
``PUT /workers/{name}`` once GATE-1 approvals land -- per memory
``no-prompt-registry-wm-is-the-store``, WM is the prompt store and PUT is
reflected on the next call.

Per memory ``upstream-license-token-pattern``, the license token is sent as
``X-Tenant-Id: Bearer <token>``; the system-workflow internal tenant UUID
``00000000-0000-4000-8000-000000000001`` (from
``fr/backend/src/middleware/internal-tenant-only.ts``) is sent in
``X-Internal-Tenant-Id`` for any WM hook that pins the worker row to the
internal workspace. WM's RFC1918 trusted-network bypass accepts in-cluster
traffic without a license token; pass ``--license-token`` for off-cluster
seeding.

Idempotency mirrors ``karma_service/src/wm_bootstrap.py``: GET first, then
POST on 404, PUT on 200 + ``--force`` (per memory
``no-prompt-registry-wm-is-the-store``: PUT is reflected on the next call,
no client-side TTL). ``--force-prompt`` also overwrites the live
``system_prompt`` from the seed; default behavior preserves it.

⛔ WHICH TENANT A WRITE LANDS ON IS NOT DECIDED BY THIS FILE, AND UNTIL
2026-09-14 THIS FILE COULD NOT TELL YOU WHICH ONE IT HIT.
``SYSTEM_WORKFLOW_TENANT_ID`` below is sent only as ``X-Internal-Tenant-Id``;
it does NOT select the row's tenant. The tenant comes from WM's auth
middleware: with a licence token, from the token; without one, from WM's own
``DEFAULT_TENANT_ID`` on the trusted-network branch. So a run against a
deployment whose ``DEFAULT_TENANT_ID`` is the seed tenant writes to the SEED
tenant while reporting plain success -- which is exactly what happened on
2026-09-13, when a ``--force --force-prompt`` reseed believed to have repaired
production landed a 4106-char prompt on
``0195f48b-d000-7000-8000-000000000001`` and left the live propose worker on
``019de842-c92a-7e37-8fdd-eb0d8c5396f9`` running with
``system_prompt = NULL`` for a further day, 17 days after it first went null.

⭐ ``--tenant <uuid>`` IS THE OTHER HALF, ADDED 2026-09-15. Reading the tenant
back made the wrong tenant VISIBLE; it did not make the right one reachable,
so ``workers/improvement-worker-reconcile`` -- a ``--dry-run --expect-tenant``
checker -- failed every run since it was created: it correctly reported that
tokenless calls land on ``…-000000000001`` and correctly refused to proceed.
``--tenant`` sends a bare ``X-Tenant-Id: <uuid>``, which WM's tokenless
trusted-network branch already accepts
(``auth_middleware.dispatch`` -> ``_trusted_network_tenant`` ->
``tenant_constants.resolve_tenant_id``). No licence token and no new Secret
are needed.

⛔ "IT WIDENS NOTHING" IS TRUE OF THE MIDDLEWARE AND FALSE OF THIS SCRIPT.
Both halves have to be said, because only the second one can hurt you.

* WM's authority is UNCHANGED. Any caller admitted by ``TRUSTED_CIDRS`` could
  already assert any tenant on that header; this script simply declined to.
  Nothing was granted, no boundary moved. Live-probed from inside the WM pod:
  no header -> ``…-000000000001``; bare ``019de842-…`` -> production.
* THIS SCRIPT'S REACH GOES FROM ONE TENANT TO ALL. Pre-fix, a tokenless run
  could touch ``DEFAULT_TENANT_ID`` and nothing else -- reaching any other
  tenant required a licence token somebody had to obtain. Post-fix,
  ``--force --force-prompt --tenant X --expect-tenant X`` overwrites every
  ``system_prompt`` on whatever X is typed. ``--dry-run`` is the only thing
  standing between a mistyped ``--tenant`` and a mass prompt overwrite on a
  live tenant, and the gitops manifest sanctions a human running exactly that
  ``--force --force-prompt`` shape.

⚠️ ``--expect-tenant`` DOES NOT PROTECT YOU AGAINST A TYPO IN ``--tenant``.
When both are set to the same value the read-back confirms our own assertion:
the run matches ITSELF, and a value that is wrong in both flags is wrong
consistently. What the guard still catches is a deployment that IGNORES the
header (older WM, or the assertion refused) and resolves to
``DEFAULT_TENANT_ID``; an armed ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT``
that 401s and aborts undeterminable; and a token run, whose tenant still comes
from the token rather than the flag. None of those is a typo check.

⭐ SO WHAT DO YOU DO INSTEAD, when you pass ``--tenant`` and intend to write?

1. Run it ``--dry-run`` FIRST, with the same ``--tenant``, and read the
   ``asserted tenant:`` and ``target tenant:`` lines. That is the only
   independent look at the uuid you typed.
2. Confirm the uuid against the licences registry rather than against your own
   command line -- ``psql -d licenses -c "select name from tenants where
   id='<uuid>'"`` on worker4 -- and check the NAME that comes back is the
   tenant you meant. ⛔ Not against ``tenant_constants.py``, whose crosswalk is
   measurably stale; see :func:`asserted_tenant_uuid`.
3. Only then re-run with ``--force``. Treat ``--expect-tenant`` as a check that
   the SERVER agreed with you, never as a check that YOU were right.

⛔ ``--expect-tenant`` is deliberately NOT allowed to imply ``--tenant``: that
would make ``resolve_target_tenant`` an echo of a header we sent, which is the
one thing the probe comment above forbids.

⛔ PASS THE UUID, NEVER THE NAME. ``resolve_tenant_id`` would accept
``n8n-dev``, but ``src/api/tenant_constants.py`` maps that name to the SEED
tenant ``0195f48b-…0001`` while the authoritative licences registry has
``n8n-dev`` as ``019de842-…`` -- the very collision behind this whole failure.
``asserted_tenant_uuid`` refuses a name for that reason.

Three things exist here so that cannot recur silently:

* ``resolve_target_tenant`` READS the tenant back from WM (a tenant-scoped
  row carries its own ``tenant_id``) instead of echoing a flag, and every
  run -- ``--dry-run`` included -- prints it BEFORE any write.
* ``--expect-tenant <uuid>`` aborts non-zero before the first write when the
  resolved tenant is not the one the operator named, and aborts when the
  tenant cannot be determined at all. Undeterminable never proceeds to a
  write.
* Every PUT/POST is read back on the SAME resolved tenant and the stored
  ``system_prompt`` / ``output_schema`` compared with what was sent, reported
  per worker as ``VERIFIED`` / ``MISMATCH``; any mismatch exits non-zero.

And a registered worker whose live ``system_prompt`` is NULL/empty is a hard
failure rather than a ``skipped-exists``: nothing else in the estate
reconciles those rows -- no CI, no Flux, no cron -- so a silent skip is how
one sat prompt-less for 17 days behind green checks.

Usage::

    uv run python scripts/improvement_workers/register.py --dry-run
    uv run python scripts/improvement_workers/register.py
    uv run python scripts/improvement_workers/register.py --force
    uv run python scripts/improvement_workers/register.py --force --force-prompt
    uv run python scripts/improvement_workers/register.py --force --force-prompt \
        --expect-tenant 019de842-c92a-7e37-8fdd-eb0d8c5396f9

    # In-cluster, tokenless, aimed at PRODUCTION (what the reconcile CronJob
    # must run -- it currently omits --tenant and therefore always fails):
    uv run python scripts/improvement_workers/register.py --dry-run \
        --tenant 019de842-c92a-7e37-8fdd-eb0d8c5396f9 \
        --expect-tenant 019de842-c92a-7e37-8fdd-eb0d8c5396f9
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import httpx

# Make ``scripts/`` importable when invoked via ``uv run python scripts/improvement_workers/register.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.improvement_workers.workers.asset_authoring import (
    AGENT_CONFIGS as ASSET_AUTHORING_CONFIGS,
)
from scripts.improvement_workers.workers.asset_loop import (
    AGENT_CONFIGS as ASSET_LOOP_CONFIGS,
)
from scripts.improvement_workers.workers.procedure_loop import (
    AGENT_CONFIGS as PROCEDURE_LOOP_CONFIGS,
)
from scripts.improvement_workers.workers.prompt_loop import (
    AGENT_CONFIGS as PROMPT_LOOP_CONFIGS,
)
from scripts.improvement_workers.workers.workflow_loop import (
    AGENT_CONFIGS as WORKFLOW_LOOP_CONFIGS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger("register_improvement_workers")


@runtime_checkable
class WorkerConfig(Protocol):
    """Structural protocol shared by every loop module's WorkerConfig dataclass.

    Each loop module (``workflow_loop``, ``prompt_loop``) defines its own
    frozen dataclass with these attributes; the registration script accepts
    any value that satisfies this protocol so new loops can be added without
    touching ``register.py`` (just import their ``AGENT_CONFIGS`` and merge
    into ``ALL_AGENT_CONFIGS`` below).
    """

    worker_name: str
    description: str
    system_prompt: str
    temperature: float
    model_provider: str
    model: str

    def output_schema(self) -> dict[str, Any]: ...


# Merged registry: every loop module contributes its workers here. Adding a
# new loop is a 2-line change -- import its AGENT_CONFIGS and merge.
AGENT_CONFIGS: dict[str, WorkerConfig] = {
    **WORKFLOW_LOOP_CONFIGS,
    **PROMPT_LOOP_CONFIGS,
    **ASSET_AUTHORING_CONFIGS,
    **PROCEDURE_LOOP_CONFIGS,
    **ASSET_LOOP_CONFIGS,
}

DEFAULT_WM_URL = "http://worker-management.workers.svc.cluster.local:8887"
DEFAULT_INFERENCE_ROUTER_URL = "http://inference-router.inference-router.svc.cluster.local:8150/api"

# UUID of the ``__system_workflow__`` internal tenant -- mirrors the value
# enforced in fr/backend/src/middleware/internal-tenant-only.ts. Workers seeded
# by this script are workflow-improvement-only; never reuse this UUID for
# real-tenant traffic.
SYSTEM_WORKFLOW_TENANT_ID = "00000000-0000-4000-8000-000000000001"

# ⛔ The probe that answers "which tenant would a write from THIS invocation
# land on?". It has to be a READ of something the server scoped for us, never
# an echo of a header we sent: the whole defect being fixed here is that the
# headers this script sends do not decide the answer. ``GET /workers_states``
# is scoped by ``scope_clause(tenant_id, subject_id)``
# (src/repositories/visibility.py) -- literally ``tenant_id = $1`` -- and its
# rows carry their own ``tenant_id`` (src/db/workers_state_db.py::_row_to_dict),
# so the value comes back from the database under the tenant WM resolved.
#
# ``GET /workers/{name}`` cannot be used for this: ``WorkerResponse``
# (src/api/workers/models.py:681) has no ``tenant_id`` field and is
# ``extra="ignore"``, so the registry row's tenant is stripped before the wire.
TENANT_PROBE_PATH = "/workers_states"
# Small on purpose: one row is enough to name the tenant, a handful survives a
# backfilled row whose tenant_id is NULL without turning this into a real read.
TENANT_PROBE_LIMIT = 5

SeedAction = Literal[
    "created",
    "skipped-exists",
    "updated",
    "updated-preserved-prompt",
    "would-create",
    "would-update",
    "would-update-preserve-prompt",
    "would-skip",
    "exists-empty-prompt",
    "would-fail-empty-prompt",
]


class RegistrationError(RuntimeError):
    """A failure that must exit non-zero rather than print a success report.

    Carries the partial ``actions`` map so the operator still sees what the
    run did before it gave up -- a bare traceback would hide it.
    """

    def __init__(self, message: str, *, actions: dict[str, SeedAction] | None = None) -> None:
        super().__init__(message)
        self.actions: dict[str, SeedAction] = dict(actions or {})


class TenantResolutionError(RegistrationError):
    """WM would not tell us which tenant a write lands on. Never proceed to a write."""


class TenantGuardError(RegistrationError):
    """``--expect-tenant`` named a tenant that is not the one WM resolved."""


class WriteVerificationError(RegistrationError):
    """A PUT/POST read back differently than it was sent on the same tenant."""

    def __init__(self, verdicts: dict[str, str], *, actions: dict[str, SeedAction]) -> None:
        detail = "; ".join(f"{name}: {verdict}" for name, verdict in sorted(verdicts.items()))
        super().__init__(
            f"{len(verdicts)} worker(s) failed post-write read-back on the resolved "
            f"tenant -- {detail}",
            actions=actions,
        )
        self.verdicts = dict(verdicts)


class EmptyLivePromptError(RegistrationError):
    """A registered worker has a NULL/empty ``system_prompt``.

    This is a hard failure and NOT a ``skipped-exists``. Nothing reconciles
    these rows -- no CI, no Flux, no cron -- so reporting the skip is how
    ``improvement_workflow_development`` ran prompt-less for 17 days behind
    green checks.
    """

    def __init__(self, workers: list[str], *, actions: dict[str, SeedAction]) -> None:
        super().__init__(
            f"{len(workers)} registered worker(s) have a NULL/empty system_prompt and were "
            f"NOT repaired: {', '.join(sorted(workers))}. Re-run with --force --force-prompt "
            f"(and --expect-tenant) to reseed them; a run without --force cannot fix this "
            f"and must not report it as a skip.",
            actions=actions,
        )
        self.workers = sorted(workers)


def canonical_tenant(value: str) -> str:
    """Canonical spelling of a tenant id, so two spellings of one uuid compare equal.

    A non-uuid value is returned stripped rather than rejected: WM's
    trusted-network branch resolves tenant NAMES too
    (``auth_middleware._trusted_network_tenant``), and refusing one here would
    turn a usable guard into a reason to skip the guard.
    """
    stripped = (value or "").strip()
    try:
        return str(uuid.UUID(stripped))
    except ValueError, AttributeError, TypeError:
        return stripped


def _probe_refusal_hint(status_code: int, license_token: str, assert_tenant: str) -> str:
    """Name the ONE configuration that turns a bare tenant assertion into a 401.

    ⛔ WHY THIS IS NOT COSMETIC. A bare ``X-Tenant-Id`` is served today because
    ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT`` is default-False and is
    unset on the live Deployment (verified 2026-09-15). The flag exists to be
    turned ON per-caller, and the day it is, THIS caller is refused 401 --
    ``auth_middleware._audit_unauthenticated_tenant_assertion`` returns a 401
    for any tokenless trusted-network request that asserts a tenant.

    Without this hint the operator sees ``GET /workers_states returned HTTP
    401`` and has no way to tell an armed gate from a network fault or a
    mis-typed URL. The failure stays fail-closed either way -- a
    ``TenantResolutionError`` never proceeds to a write -- but it must be
    LEGIBLE, because the remedy (obtain a licence token for this tenant) is not
    guessable from the status code.

    ⭐ Note the asymmetry that makes asserting the header the SAFER option:
    with the gate armed and NO header, the audit returns ``None`` (it counts
    only actual assertions) and the run silently lands on the deployment's
    ``DEFAULT_TENANT_ID``. Asserting converts that silent wrong-tenant write
    into a loud refusal.
    """
    if status_code not in (401, 403):
        return ""
    if license_token:
        return (
            ". The licence token was refused -- it is expired, revoked, or issued for a "
            "different tenant."
        )
    if assert_tenant:
        return (
            f". A bare X-Tenant-Id: {assert_tenant} was sent with no licence token, and "
            f"worker_management refused it. The expected cause is "
            f"WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT=true on the "
            f"worker-management Deployment, which 401s any tokenless trusted-network "
            f"caller that asserts a tenant. Once that gate is armed this script needs a "
            f"real licence token for {assert_tenant} passed as --license-token; a bare "
            f"header can no longer work, and must not be replaced by dropping --tenant, "
            f"which would silently write the deployment's DEFAULT_TENANT_ID instead. "
            f"Other causes: the caller is outside TRUSTED_CIDRS, or the tenant is unknown "
            f"to this deployment."
        )
    return (
        ". No licence token and no --tenant were supplied, so this request carried no "
        "tenant credential at all."
    )


def asserted_tenant_uuid(value: str) -> str:
    """Validate the value this script will ASSERT in a bare ``X-Tenant-Id``.

    Stricter than :func:`canonical_tenant` on purpose, and stricter than WM's
    own ``resolve_tenant_id`` on purpose: a canonical 8-4-4-4-12 uuid and
    nothing else.

    ⛔ A TENANT **NAME** IS REFUSED HERE, AND THAT REFUSAL IS THE POINT.
    WM would accept one -- ``resolve_tenant_id`` maps names through
    ``src/api/tenant_constants.py::TENANT_UUIDS`` -- but that map is the ROOT
    CAUSE of the failure this function is part of fixing. It says

        "n8n-dev" -> 0195f48b-d000-7000-8000-000000000001

    while the authoritative licences ``tenants`` registry has ``n8n-dev`` as
    ``019de842-c92a-7e37-8fdd-eb0d8c5396f9`` and ``…-000000000001`` as
    ``n8n-dev-portal-realm``. So the SAME STRING resolves to two different
    tenants depending on which registry answers, and sending the name would
    silently re-enter the exact bug: aiming at production and landing on the
    seed tenant while reporting success.

    Measured on worker4 2026-09-15, and these two rows are the whole basis for
    the paragraph above -- nothing here is carried over from another file::

        $ psql -d licenses -At -F'|' -c "select id, name from tenants where ..."
        0195f48b-d000-7000-8000-000000000001|n8n-dev-portal-realm
        019de842-c92a-7e37-8fdd-eb0d8c5396f9|n8n-dev

    ⚠️ DO NOT RE-COPY THE CROSSWALK IN ``tenant_constants.py``'s DOCSTRING. An
    earlier draft of this docstring did, and asserted that the licences side
    calls ``…-000000000001`` ``platform-cp-1``, sourced to a direct worker4
    reading it never had. Measured: ``platform-cp-1`` is not in that table at
    all, and that crosswalk is stale on 2 of its 3 rows (``…0002`` is
    ``platform-core``, ``…0003`` is ``aa-agents``, not ``platform-cp-3``). The
    collision it describes is real; its names are not. Re-query, do not quote.

    Passing the uuid is not a workaround for that collision -- it is the only
    spelling that means one thing in both registries.

    Raises:
        ValueError: not a canonical uuid (a name, a nil uuid, a braced/urn/
            undashed spelling, or empty). Never falls back to a default: an
            unusable value must stop the run, not redirect it.
    """
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("--tenant was given an empty value; pass a canonical tenant uuid")
    try:
        parsed = uuid.UUID(candidate)
    except ValueError as exc:
        raise ValueError(
            f"--tenant {candidate!r} is not a uuid. Pass the tenant UUID, NEVER the name: "
            f"tenant NAMES resolve through src/api/tenant_constants.py, whose 'n8n-dev' is "
            f"the SEED tenant 0195f48b-d000-7000-8000-000000000001 and NOT the licences "
            f"registry's n8n-dev (019de842-c92a-7e37-8fdd-eb0d8c5396f9). Production is "
            f"019de842-c92a-7e37-8fdd-eb0d8c5396f9."
        ) from exc
    if parsed.int == 0:
        raise ValueError(
            "--tenant is the nil uuid, which is registered as 'default-anon' and is not a "
            "tenant on any inbound path (see tenant_constants.NIL_TENANT_UUID)"
        )
    if candidate.lower() != str(parsed):
        # Same rule WM applies at its own boundary: a tenant id that changes
        # shape between the header and the log is a debugging trap.
        raise ValueError(f"--tenant {candidate!r} is not canonical 8-4-4-4-12; pass {parsed!s}")
    return str(parsed)


def resolve_target_tenant(
    *,
    wm_url: str,
    license_token: str,
    assert_tenant: str = "",
    client: httpx.Client | None = None,
) -> str:
    """Read back from WM the tenant a write from this invocation would land on.

    This is a real read, not a restatement of what we sent: the returned value
    is the ``tenant_id`` stamped on rows the server selected under the tenant
    its auth middleware resolved for these exact headers. Sending a different
    ``--license-token`` (or none) changes the answer, which is the point.

    Raises:
        TenantResolutionError: the tenant could not be determined. The caller
            must NOT proceed to a write -- "undeterminable" is the state the
            2026-09-13 reseed was in when it reported success.
    """
    if not wm_url:
        raise ValueError("wm_url is required")

    base = wm_url.rstrip("/")
    headers = _build_headers(license_token, assert_tenant)
    owns_client = client is None
    http = client if client is not None else httpx.Client(timeout=30.0)
    try:
        resp = http.get(
            f"{base}{TENANT_PROBE_PATH}",
            params={"limit": TENANT_PROBE_LIMIT},
            headers=headers,
        )
        if resp.status_code != 200:
            raise TenantResolutionError(
                f"cannot determine target tenant: GET {TENANT_PROBE_PATH} returned HTTP "
                f"{resp.status_code}{_probe_refusal_hint(resp.status_code, license_token, assert_tenant)}"
            )
        try:
            body = resp.json()
        except ValueError as exc:
            raise TenantResolutionError(
                f"cannot determine target tenant: GET {TENANT_PROBE_PATH} returned a non-JSON body"
            ) from exc
    finally:
        if owns_client:
            http.close()

    if not isinstance(body, list):
        raise TenantResolutionError(
            f"cannot determine target tenant: GET {TENANT_PROBE_PATH} returned "
            f"{type(body).__name__}, expected a list of tenant-scoped rows"
        )

    seen = {
        canonical_tenant(str(row["tenant_id"]))
        for row in body
        if isinstance(row, dict) and row.get("tenant_id")
    }
    if not seen:
        raise TenantResolutionError(
            "cannot determine target tenant: worker_management returned no tenant-scoped "
            "row carrying a tenant_id. Pass --license-token so the tenant comes from the "
            "token, or seed from a context that has state on the target tenant; do NOT "
            "write blind."
        )
    if len(seen) > 1:
        raise TenantResolutionError(
            f"cannot determine target tenant: tenant-scoped read returned {len(seen)} "
            f"distinct tenant_ids ({', '.join(sorted(seen))}); refusing to guess"
        )
    return next(iter(seen))


def build_payload(
    config: WorkerConfig,
    *,
    base_url_override: str,
) -> dict[str, Any]:
    """Translate a ``WorkerConfig`` into a ``WorkerCreateRequest`` body.

    ``output_schema`` is loaded from disk so callers do not have to pass it
    per-call (per memory ``wm-as-sole-llm-gateway``, callers MAY still
    override per-call when WM is invoked).

    ``max_tokens`` is included only when the config sets it -- workflow_loop's
    WorkerConfig does not declare it; prompt_loop's does (8192 / 16384).
    """
    payload: dict[str, Any] = {
        "worker_name": config.worker_name,
        "agent_role": f"improvement loop -- {config.worker_name}",
        "agent_type": "pydantic_ai",
        "model_provider": config.model_provider,
        "model": config.model,
        "temperature": config.temperature,
        "system_prompt": config.system_prompt,
        "description": config.description,
        "base_url_override": base_url_override,
        "output_schema": config.output_schema(),
    }
    max_tokens = getattr(config, "max_tokens", None)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _build_headers(license_token: str, assert_tenant: str = "") -> dict[str, str]:
    """Headers for every call this script makes.

    ``X-Tenant-Id`` carries ONE of two things and never both:

    * ``Bearer <token>`` when a licence token is supplied -- WM introspects it
      and the tenant comes from the token (per memory
      ``upstream-license-token-pattern``).
    * a bare canonical tenant uuid when ``assert_tenant`` is supplied and there
      is no token. WM's tokenless trusted-network branch accepts exactly this:
      ``auth_middleware.dispatch`` calls ``_trusted_network_tenant(x_tenant_id)``,
      which hands the raw header to ``tenant_constants.resolve_tenant_id`` and
      returns the canonical uuid. This is a documented existing path, not a new
      one -- an in-cluster caller could already assert a tenant here; until now
      this script simply declined to, and so inherited the DEPLOYMENT's
      ``DEFAULT_TENANT_ID`` instead.
    * neither, when both are empty -- unchanged behaviour, and the reason this
      fix exists: WM then falls back to its own ``DEFAULT_TENANT_ID``, which on
      the live Deployment is the SEED tenant
      ``0195f48b-d000-7000-8000-000000000001``, not production.

    Raises:
        ValueError: both were supplied. They write the same header, so one
            would have to be dropped, and a silently dropped tenant assertion
            is precisely the failure mode this whole file exists to prevent.
            The caller states which credential it means.
    """
    headers: dict[str, str] = {
        "X-Internal-Tenant-Id": SYSTEM_WORKFLOW_TENANT_ID,
    }
    if license_token and assert_tenant:
        raise ValueError(
            "--tenant and --license-token are mutually exclusive: both are carried in "
            "X-Tenant-Id, so honouring one means silently ignoring the other. With a "
            "licence token the tenant comes from the token; drop --tenant, and use "
            "--expect-tenant to assert which tenant you believe that token resolves to."
        )
    if license_token:
        # Per memory upstream-license-token-pattern.
        headers["X-Tenant-Id"] = f"Bearer {license_token}"
    elif assert_tenant:
        headers["X-Tenant-Id"] = assert_tenant
    return headers


def _live_prompt(get_resp: httpx.Response) -> str:
    """The ``system_prompt`` currently stored on an existing worker, or ``""``.

    ``""`` covers all three of "absent", "null" and "blank" on purpose: every
    one of them means the running system has no prompt for this worker, and
    the caller must treat them identically.
    """
    try:
        live_body = get_resp.json()
    except ValueError:
        return ""
    if not isinstance(live_body, dict):
        return ""
    raw_prompt = live_body.get("system_prompt")
    return raw_prompt if isinstance(raw_prompt, str) else ""


def _verify_write(
    http: httpx.Client,
    *,
    base: str,
    name: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> str:
    """Re-GET ``name`` and compare the stored prompt/schema with what was sent.

    The re-GET carries the SAME headers as the write, so it resolves to the
    same tenant: reseeding one row and then reading a different one is
    precisely how the 2026-09-13 reseed was reported done while being undone.

    Returns ``"VERIFIED"`` or a ``"MISMATCH: ..."`` string; it never raises, so
    one bad worker cannot stop the remaining workers from being reported.
    """
    resp = http.get(f"{base}/workers/{name}", headers=headers)
    if resp.status_code != 200:
        return f"MISMATCH: read-back GET returned HTTP {resp.status_code}"
    try:
        body = resp.json()
    except ValueError:
        return "MISMATCH: read-back GET returned a non-JSON body"
    if not isinstance(body, dict):
        return "MISMATCH: read-back GET returned a non-object body"

    differing = [
        field
        for field in ("system_prompt", "output_schema")
        if body.get(field) != payload.get(field)
    ]
    if differing:
        return f"MISMATCH: stored {', '.join(differing)} differs from what was sent"
    return "VERIFIED"


def _verify_and_record(
    http: httpx.Client,
    *,
    base: str,
    name: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    verdicts: dict[str, str],
    verify_writes: bool,
) -> None:
    """Read ``name`` back after a write and record the verdict; no-op when off."""
    if not verify_writes:
        return
    verdicts[name] = _verify_write(http, base=base, name=name, payload=payload, headers=headers)
    logger.info("worker %s read-back: %s", name, verdicts[name])


def _empty_prompt_action(name: str, *, dry_run: bool) -> SeedAction:
    """Label a registered-but-prompt-less worker, loudly.

    ``logger.error``, not ``info``: this is the condition that ran undetected
    for 17 days, and the only thing that ever looks at it is this line.
    """
    logger.error(
        "worker %s is registered in WM with a NULL/empty system_prompt -- this is a "
        "FAILURE, not a skip; re-run with --force --force-prompt (and --expect-tenant) "
        "to reseed it",
        name,
    )
    return "would-fail-empty-prompt" if dry_run else "exists-empty-prompt"


def _raise_on_failure(
    *,
    verdicts: dict[str, str],
    prompt_less: list[str],
    actions: dict[str, SeedAction],
) -> None:
    """Raise after the whole run, so one invocation names every affected worker.

    Verification first: a write that did not read back from where it was
    written is the more dangerous of the two failures, and is the one this
    file exists to make impossible to miss.
    """
    failed = {name: verdict for name, verdict in verdicts.items() if verdict != "VERIFIED"}
    if failed:
        raise WriteVerificationError(failed, actions=actions)
    if prompt_less:
        raise EmptyLivePromptError(prompt_less, actions=actions)


def register_workers(
    *,
    wm_url: str,
    license_token: str,
    assert_tenant: str = "",
    base_url_override: str,
    force: bool,
    dry_run: bool,
    force_prompt: bool = False,
    verify_writes: bool = False,
    configs: Mapping[str, WorkerConfig] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, SeedAction]:
    """Idempotently seed every workflow-loop worker into WM.

    Returns a map of ``worker_name -> action``. Network errors are NOT
    swallowed: a failed seed must surface clearly so callers can fail fast
    rather than serve traffic with auto-created fallback workers.

    When ``force=True`` and ``force_prompt=False`` (default), the existing
    worker's ``system_prompt`` is preserved across the PUT (GET-then-merge),
    so operator edits made via ``PUT /workers/{name}`` are not clobbered by
    re-runs (per memory ``no-prompt-registry-wm-is-the-store``: WM is the
    prompt store). All other fields (``output_schema``, ``model``,
    ``temperature``, ``model_provider``, ``description``,
    ``base_url_override``, ``agent_role``, ``agent_type``) are taken from
    the seed and are seed-authoritative.

    Set ``force_prompt=True`` to opt in to overwriting ``system_prompt``
    from the seed too -- only do this when the seed prompt is the new
    intended value.

    ``verify_writes=True`` re-GETs every worker this call actually wrote, on
    the same headers (therefore the same tenant), and compares the stored
    ``system_prompt`` / ``output_schema`` with what was sent, logging a
    per-worker ``VERIFIED`` / ``MISMATCH`` and raising
    :class:`WriteVerificationError` at the end if any differed. ``main()``
    always passes it; it defaults off only so a caller that supplies its own
    ``client`` is not forced to answer read-back GETs it did not ask for.

    An existing worker whose live ``system_prompt`` is NULL/empty raises
    :class:`EmptyLivePromptError` at the end of the run rather than being
    reported as ``skipped-exists``, on the ``force=False`` path where this
    call cannot repair it.

    Raises:
        EmptyLivePromptError: a registered worker has no live system_prompt.
        WriteVerificationError: a write did not read back as sent.
    """
    if not wm_url:
        raise ValueError("wm_url is required")
    if not base_url_override:
        raise ValueError("base_url_override is required")

    base = wm_url.rstrip("/")
    headers = _build_headers(license_token, assert_tenant)
    items = configs if configs is not None else AGENT_CONFIGS

    owns_client = client is None
    http = client if client is not None else httpx.Client(timeout=30.0)
    actions: dict[str, SeedAction] = {}
    verdicts: dict[str, str] = {}
    prompt_less: list[str] = []
    try:
        for name, config in items.items():
            payload = build_payload(config, base_url_override=base_url_override)

            get_resp = http.get(f"{base}/workers/{name}", headers=headers)
            if get_resp.status_code == 200:
                if not force:
                    # ⛔ NOT A SKIP WHEN THE ROW HAS NO PROMPT. "Registered"
                    # and "usable" are different facts, and only the first one
                    # was ever checked here. A worker whose live system_prompt
                    # is NULL/empty is a worker the running system calls with
                    # no instructions at all; nothing else reconciles it, so
                    # the only moment anyone would notice is this one.
                    if _live_prompt(get_resp).strip() == "":
                        actions[name] = _empty_prompt_action(name, dry_run=dry_run)
                        prompt_less.append(name)
                        continue
                    actions[name] = "skipped-exists" if not dry_run else "would-skip"
                    logger.info("worker %s already exists in WM; skipping", name)
                    continue

                # GET-then-merge protection: by default we preserve the
                # live system_prompt so operator edits via PUT survive
                # re-runs. Only --force-prompt opts in to overwriting.
                live_prompt = "" if force_prompt else _live_prompt(get_resp)

                preserve_prompt = (not force_prompt) and live_prompt.strip() != ""
                if preserve_prompt:
                    payload = {**payload, "system_prompt": live_prompt}

                if dry_run:
                    if preserve_prompt:
                        actions[name] = "would-update-preserve-prompt"
                        logger.info(
                            "worker %s would be UPDATED via PUT (preserving system_prompt)",
                            name,
                        )
                    else:
                        actions[name] = "would-update"
                        logger.info(
                            "worker %s would be UPDATED via PUT (overwriting system_prompt)",
                            name,
                        )
                    continue
                put_resp = http.put(f"{base}/workers/{name}", json=payload, headers=headers)
                put_resp.raise_for_status()
                if preserve_prompt:
                    actions[name] = "updated-preserved-prompt"
                    logger.info("worker %s updated in WM (system_prompt preserved)", name)
                else:
                    actions[name] = "updated"
                    logger.info("worker %s updated in WM (system_prompt overwritten)", name)
                _verify_and_record(
                    http,
                    base=base,
                    name=name,
                    payload=payload,
                    headers=headers,
                    verdicts=verdicts,
                    verify_writes=verify_writes,
                )
                continue

            if get_resp.status_code != 404:
                # Surface 401/403/5xx loudly -- do not silently fall through to POST.
                get_resp.raise_for_status()

            if dry_run:
                actions[name] = "would-create"
                logger.info("worker %s would be CREATED via POST", name)
                continue
            post_resp = http.post(f"{base}/workers", json=payload, headers=headers)
            post_resp.raise_for_status()
            actions[name] = "created"
            logger.info("worker %s created in WM", name)
            _verify_and_record(
                http,
                base=base,
                name=name,
                payload=payload,
                headers=headers,
                verdicts=verdicts,
                verify_writes=verify_writes,
            )
    finally:
        if owns_client:
            http.close()

    _raise_on_failure(verdicts=verdicts, prompt_less=prompt_less, actions=actions)
    return actions


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without calling worker_management.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing records via PUT (otherwise leave them alone).",
    )
    p.add_argument(
        "--force-prompt",
        action="store_true",
        help=(
            "Also overwrite system_prompt from the seed when --force is set. "
            "Default behavior preserves the live system_prompt so operator "
            "edits made via PUT /workers/{name} survive re-runs (per memory "
            "no-prompt-registry-wm-is-the-store)."
        ),
    )
    p.add_argument(
        "--expect-tenant",
        default="",
        metavar="UUID",
        help=(
            "Abort non-zero BEFORE any write unless the tenant worker_management "
            "actually resolves for these credentials equals this value. The tenant is "
            "read back from WM, not echoed from this flag; if it cannot be determined "
            "the run aborts rather than writing blind. Deliberately has NO environment "
            "default -- an env var could arm or disarm the guard without appearing in "
            "the command anybody later reads."
        ),
    )
    p.add_argument(
        "--tenant",
        default="",
        metavar="UUID",
        help=(
            "Tenant to ASSERT, as a bare X-Tenant-Id header, on a tokenless in-cluster "
            "run. Without it worker_management falls back to its own DEFAULT_TENANT_ID, "
            "which on the live Deployment is the SEED tenant, not production. "
            "⛔ A UUID, never a tenant name -- 'n8n-dev' means two different tenants in "
            "the two registries. Mutually exclusive with --license-token. Deliberately "
            "has NO environment default: an env var could silently change WHICH TENANT "
            "IS WRITTEN without appearing in the command anybody later reads."
        ),
    )
    p.add_argument(
        "--wm-url",
        default=os.getenv("WORKER_MANAGEMENT_URL", DEFAULT_WM_URL),
        help="worker_management base URL (default: $WORKER_MANAGEMENT_URL or in-cluster svc).",
    )
    p.add_argument(
        "--license-token",
        default=os.getenv("WORKER_MANAGEMENT_LICENSE_TOKEN", ""),
        help=(
            "License token used in X-Tenant-Id: Bearer <token>. Empty relies on "
            "WM's RFC1918 trusted-network bypass (in-cluster only)."
        ),
    )
    p.add_argument(
        "--inference-router-url",
        default=os.getenv("INFERENCE_ROUTER_URL", DEFAULT_INFERENCE_ROUTER_URL),
        help="URL stored as base_url_override on each worker record.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    # Validate BEFORE the banner and before any request: an unusable --tenant
    # must stop the run, never be silently dropped back to DEFAULT_TENANT_ID.
    try:
        assert_tenant = asserted_tenant_uuid(args.tenant) if args.tenant else ""
        if assert_tenant and args.license_token:
            # Surfaces _build_headers' own refusal here, before the banner
            # claims a tenant the headers would not carry.
            _build_headers(args.license_token, assert_tenant)
    except ValueError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Registering improvement workers at {args.wm_url}")
    print(f"  inference-router: {args.inference_router_url}")
    print(f"  force: {args.force}, force_prompt: {args.force_prompt}, dry_run: {args.dry_run}")
    print(f"  expect_tenant: {args.expect_tenant or '(not set)'}")
    # ⚠️ Print BOTH, and never conflate them. `asserted tenant` is an INPUT --
    # the tenant these headers claim. `target tenant` below is a READ-BACK.
    # They are separate lines because a reader who sees only one cannot tell an
    # honoured assertion from an ignored one.
    print(
        f"  asserted tenant: {assert_tenant or '(none -- WM will use its own DEFAULT_TENANT_ID)'}"
    )
    print(
        "  NOTE: Plan 5 ships placeholder system_prompts -- Plan 6 will improve them in-place via "
        "PUT /workers/{name}. Per memory wm-as-sole-llm-gateway, callers may still pass "
        "output_schema per-call to override the registered default."
    )

    client = httpx.Client(timeout=30.0)
    try:
        # ⛔ RESOLVE AND PRINT THE TENANT BEFORE ANYTHING ELSE. Every line below
        # this point is about a specific tenant, and until 2026-09-14 the output
        # of this script never said which one.
        try:
            tenant = resolve_target_tenant(
                wm_url=args.wm_url,
                license_token=args.license_token,
                assert_tenant=assert_tenant,
                client=client,
            )
        except TenantResolutionError as exc:
            tenant = None
            print("  target tenant: UNRESOLVED")
            # Fail-closed: undeterminable never proceeds to a write. A --dry-run
            # with no guard requested writes nothing, so it is allowed to carry
            # on and show the operator what it WOULD do -- that is the run you
            # make to diagnose this very failure. Asking for --expect-tenant and
            # being unable to evaluate it is itself a refusal.
            if args.expect_tenant or not args.dry_run:
                print(f"FAILED: {exc}", file=sys.stderr)
                return 1
            print(f"  WARNING: {exc}")
        else:
            print(f"  target tenant: {tenant}  (read back from worker_management)")

        if args.expect_tenant:
            expected = canonical_tenant(args.expect_tenant)
            if tenant != expected:
                print(
                    f"FAILED: --expect-tenant {expected} but worker_management resolves "
                    f"these credentials to {tenant}. No write was attempted.",
                    file=sys.stderr,
                )
                return 1
            print(f"  --expect-tenant OK: {expected}")
        print()

        actions = register_workers(
            wm_url=args.wm_url,
            license_token=args.license_token,
            assert_tenant=assert_tenant,
            base_url_override=args.inference_router_url,
            force=args.force,
            force_prompt=args.force_prompt,
            dry_run=args.dry_run,
            # Not optional and not a flag: a reseed nobody read back is the
            # failure this script is being fixed for.
            verify_writes=True,
            client=client,
        )
    except RegistrationError as exc:
        if exc.actions:
            print(json.dumps(exc.actions, indent=2, sort_keys=True))
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print(json.dumps(actions, indent=2, sort_keys=True))
    # ⛔ Say "no write" when there was no write. "VERIFIED" printed after a run
    # that wrote nothing is the same class of claim as the 2026-09-13 "reseed
    # verified": true of something, but not of the thing the reader will take
    # it for.
    wrote = {"created", "updated", "updated-preserved-prompt"}
    written = sum(1 for action in actions.values() if action in wrote)
    target = tenant or "UNRESOLVED"
    if written:
        print(f"{written} write(s) read back VERIFIED on tenant {target}")
    else:
        print(f"No write was issued; nothing to verify on tenant {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
