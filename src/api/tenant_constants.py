"""Canonical tenant UUID constants, and the boundary that enforces them.

`tenant_id` is a real Postgres ``uuid`` column on all 12 tenant-scoped tables
(migration ``024_tenant_id_is_a_uuid.sql``). A human label such as ``n8n-dev``
is a `tenant_name`, never a `tenant_id`, and a request carrying one must have it
RESOLVED here before it reaches the database — otherwise every statement of that
request dies with ``invalid input syntax for type uuid``.

WHERE THESE VALUES COME FROM
----------------------------
The portal realm writes the ``tenant_id`` user attribute in Keycloak, and the
map below mirrors that job's own table verbatim:
``gitops-production/clusters/strato/services/keycloak/realm-patch-script-configmap.yaml:111-146``.
That is the authority: `license-authz` resolves entitlement on the JWT's
``tenant_id`` claim, so whatever the realm mints IS the tenant id.

⚠️ THESE LABELS DO NOT MATCH THE ``licenses`` DB, AND THAT IS KNOWN.
Migration 014 on the licences side seeded platform tenants onto the SAME uuid7
values, so the label here and the name in ``licenses.tenants`` disagree.

⚠️ CORRECTED 2026-09-16 — the table that stood here was wrong on its own terms
and named two tenants that do not exist. It claimed ``…0001`` is licences-side
``platform-cp-1`` and ``…0003`` is ``platform-cp-3``. Measured on worker4
(``psql -d licenses -Atc "select id, name from tenants"``),
``select count(*) from tenants where name like 'platform-cp%'`` returns **0** —
neither name is in that registry at all. Re-derived, the real rows are::

    uuid                                    | licences name         | this map
    …-000000000001                          | n8n-dev-portal-realm  | n8n-dev
    …-000000000003                          | aa-agents             | george-dev
    …-000000000099                          | fr-test-user          | fr-test
    …-000000000100                          | cdmsfirst             | cdmsfirst   (agree)
    (2026-10-07: both re-keyed to 01a1117c-d9b0-72fc-aeda-a9d1187f2a36 —
     licenses 047; the …-100 row above is the dated 09-16 measurement)
    …-000000000101                          | ahtapol               | ahtapol     (agree)
    …-000000000102                          | <absent>              | ahtapol-probe
    019de842-c92a-7e37-8fdd-eb0d8c5396f9    | n8n-dev               | —
    019efaa0-512b-733e-9e98-b1ac82a6f7ee    | fr-test               | —

⛔ THE ROW THAT MATTERS IS NOT THE MISMATCH, IT IS THE COLLISION. Two of the
labels below — ``n8n-dev`` and ``fr-test`` — are ALSO live tenant names in the
licences registry, pointing at a DIFFERENT uuid. So the same string resolves to
two different real tenants depending on which registry answers, and neither
answer is detectably wrong at the point of use. That is why
:func:`resolve_asserted_tenant_id` refuses a name outright on an unauthenticated
inbound header: there is no "correct" resolution to pick.

The consequence is documented in that same configmap (lines 146-169): a licence
issued to the licences-side tenant literally named ``n8n-dev``
(``019de842-c92a-7e37-8fdd-eb0d8c5396f9``) is UNREACHABLE, because no JWT ever
carries that uuid.

⛔ Do NOT "fix" this by renumbering. These values are pinned into live Keycloak
user attributes; changing one moves real users between tenants and drains a
different quota pool. The same ⛔ is on the gitops side.

CORRECTED 2026-08-25 — the previous version of this docstring cited
gitops-production commits ``9606ca9b`` and ``65d727b6``. **Neither SHA exists in
that repository** (checked across all 26,201 commits and all refs). The real
commit is ``4865bee6`` (2026-04-29 11:01), "fix(keycloak/portal): tenant_id is
uuid7 always; tenant_name carries text label", which introduced this exact map
27 minutes before this file was created. Also removed: ``gerard-dev``
(``…-000000000002``), which the realm dropped 2026-08-12 because gerard no
longer has cluster access. Its suffix stays retired and is deliberately NOT
remapped here — a request naming it now gets a clean 400 rather than access.
"""

from __future__ import annotations

import functools
import logging
import uuid

# Mapping from tenant_name → the uuid7 tenant_id the portal realm actually mints.
# Mirrors realm-patch-script-configmap.yaml. Extend by adding a tenant THERE
# first; never recycle a suffix.
TENANT_UUIDS: dict[str, str] = {
    "n8n-dev": "0195f48b-d000-7000-8000-000000000001",
    # "gerard-dev" (…-000000000002) REMOVED from the realm 2026-08-12. Suffix
    # retired, not recycled. See the module docstring.
    "george-dev": "0195f48b-d000-7000-8000-000000000003",
    "fr-test": "0195f48b-d000-7000-8000-000000000099",
    # Re-keyed 2026-10-07 (owner ruling, unit B2) off the reserved slot
    # …-000000000100, together with licenses 047 and the realm's tenant_id
    # attributes. Suffix …-100 is retired, not recycled.
    "cdmsfirst": "01a1117c-d9b0-72fc-aeda-a9d1187f2a36",
    "ahtapol": "0195f48b-d000-7000-8000-000000000101",
    "ahtapol-probe": "0195f48b-d000-7000-8000-000000000102",
}

# Default tenant_id used by the trusted-network auth bypass and Swagger UI.
DEFAULT_TENANT_UUID: str = TENANT_UUIDS["n8n-dev"]

# The all-zeros UUID. NOT a tenant, on any inbound path.
#
# ⛔ WHY IT NEEDS ITS OWN NAME AND ITS OWN CHECK. It is a syntactically valid
# UUID, so ``uuid.UUID()`` accepts it and every shape gate in this service passed
# it straight through into ``request.state.tenant_id`` — a caller on the pod
# network could simply ASSERT it in ``X-Tenant-Id``. It is also *registered* in
# ``licenses.tenants`` as ``default-anon``, so it survives a dirty-row audit
# while collapsing every unattributed caller into one bucket; an anonymous
# principal was once granted admin on exactly this tenant. Seven sibling services
# already reject it by value at their inbound boundary
# (``w2v_orchestrator/src/auth/run_tenant.py``,
# ``video_production_service/src/api/tenant.py`` and five more); this is that same
# rule, in the one place WM resolves a tenant.
#
# Checking ``.int == 0`` rather than the string covers the undashed, braced and
# ``urn:uuid:`` spellings for free — though ``resolve_tenant_id`` refuses those
# shapes anyway.
#
# ⚠️ NOT to be confused with ``…-000000000001`` (``n8n-dev``), which is a real
# tenant, nor with ``billing-service``'s deliberate registration of this value as
# ``default-anon``/UNATTRIBUTED — that service is the ledger of record for spend
# nobody can be charged for, and is out of scope here by design.
NIL_TENANT_UUID: str = str(uuid.UUID(int=0))

# The namespace migration 024 uses to derive a stable uuid for a legacy label
# that no registry resolves. It is byte-for-byte the namespace
# requirements_management used in its own flip, so two services deriving an id
# for the same label land on the same uuid and the fleet stays joinable.
# Never change it.
TENANT_NAMESPACE: uuid.UUID = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")

# Synthetic tenant for the auth-bypass test path. Derived with the same function
# migration 024 uses, from the label this path used to write verbatim
# ("test_user"), so if such a row ever escapes into a real database the
# crosswalk explains where it came from instead of it being an unattributable
# uuid. It is NOT a registered tenant and must never appear in production.
BYPASS_TENANT_UUID: str = str(uuid.uuid5(TENANT_NAMESPACE, "test_user"))


# The PLATFORM tenants: ours, not a customer's. System data (worker prompts
# first) is writable only from here or by a verified elevated SA role.
#
# ⛔ MIRRORS fr ``PLATFORM_TENANT_IDS``
# (``backend/src/middleware/internal-tenant-only.ts:83``; re-exported by
# ``routes/assets/_lib/system-union.ts``): its
# INTERNAL_TENANT_IDS ``…0001/2/4/5`` + EXECUTION ``…0003`` + the AA conveyor
# tenant ``019de842…``. The two lists MUST change together — a tenant platform
# in one service and customer in the other is exactly the leak this closes.
# Drift is caught by tests/test_platform_tenant_ids_drift.py (#339) against a
# pinned copy of fr's set in tests/fixtures/fr_platform_tenant_ids.json.
#
# Owner ruling 2026-09-30: system data is never visible to, nor writable by,
# customers. ``…0002`` (gerard-dev) is retired from TENANT_UUIDS above but is
# kept here because fr keeps it; it resolves nowhere, so it grants nothing.
PLATFORM_TENANT_IDS: frozenset[str] = frozenset(
    {
        "0195f48b-d000-7000-8000-000000000001",
        "0195f48b-d000-7000-8000-000000000002",
        "0195f48b-d000-7000-8000-000000000003",
        "0195f48b-d000-7000-8000-000000000004",
        "0195f48b-d000-7000-8000-000000000005",
        "019de842-c92a-7e37-8fdd-eb0d8c5396f9",
    }
)


# The system tenant: owner of every system-written prompt (U6, migration 035).
SYSTEM_TENANT_ID: str = "0195f48b-d000-7000-8000-000000000001"


def is_platform_tenant(tenant_id: str | None) -> bool:
    """True only for a canonical uuid in :data:`PLATFORM_TENANT_IDS`.

    Compared case-insensitively (Postgres compares uuids by value); a NAME is
    never matched — ``n8n-dev`` resolves to different uuids in different
    registries (module docstring), so only the uuid is an answer.
    """
    if not tenant_id:
        return False
    return tenant_id.strip().lower() in PLATFORM_TENANT_IDS


# The COMMERCIAL-PLATFORM tenants: the set that decides customer vs platform on
# the LLM / BYO / provider-routing surface ONLY. Platform tenants get the full
# provider catalog, any inference-router model and an honoured inbound
# ``X-Provider`` pin, and — ONLY when the request also PROVES the tenant
# (``src.auth.tenant_proof.owner_commercial_keys_permitted``) — the owner's
# gitops-registered commercial keys (owner ruling 2026-10-03: "allow all gitops
# registered commercial apis for platform workspaces only"). EVERY OTHER TENANT
# is a customer: inference-router ``local-llm`` only, plus the commercial
# providers it has its OWN usable BYO key for (own, or managed BYO set by the
# owner for that one tenant). Owner ruling 2026-10-01.
#
# ⛔ DEFINED ONCE, IN GITOPS — NOT HERE (fr#1385, 2026-10-03). The set used to
# be a frozenset literal in this file (``PLATFORM_TENANT_IDS | {system}``). It
# is now read from ``COMMERCIAL_PLATFORM_TENANT_IDS`` (``Settings.
# commercial_platform_tenant_ids``), set on the Deployment in
# ``gitops-production/clusters/strato/services/workers/deployment.yaml``. An
# unset/empty value means NO platform tenant (fail closed). Rollout order is
# therefore WM FIRST, with the env unset (every tenant is a customer: local-llm
# + own BYO only — restrictive but safe), THEN the gitops PR that sets it.
#
# ⛔ WHY THIS IS A SEPARATE SET FROM :data:`PLATFORM_TENANT_IDS`. The two answer
# different questions and must not be merged:
#
# * :data:`PLATFORM_TENANT_IDS` gates SYSTEM-DATA writes (worker prompts) and
#   MIRRORS fr's list verbatim. Widening it grants prompt-write rights and
#   breaks the fr mirror contract. It is never edited from here.
# * This set gates which tenant may use the OWNER'S commercial LLM keys. The
#   gitops value is today ``PLATFORM_TENANT_IDS`` plus the ``system`` tenant
#   ``00000000-0000-0000-0000-000000000001`` (the owner's own internal calls),
#   matching inference-router's ``INFERENCE_ROUTER_PLATFORM_TENANT_IDS``
#   default. A tenant platform here and customer in IR (or vice versa) would
#   either strand our own calls or hand a customer the owner's keys.
#
# ⛔ AN ALLOWLIST OF CANONICAL UUIDS. A tenant NAME, the nil uuid or any
# non-canonical spelling in the env value is DROPPED (and logged), never
# resolved: two labels collide with different licences-side tenants (module
# docstring). A tenant missing from it is a customer — the safe failure.
#
# ⚠️ Membership alone is NOT proof. On the tokenless trusted-network branch the
# tenant is caller-asserted and an absent header defaults to
# ``DEFAULT_TENANT_UUID`` (``…-000000000001``, ``n8n-dev``, in this set). That is
# why the owner-key gate also requires the request to have PROVEN the tenant.
SYSTEM_TENANT_UUID: str = "00000000-0000-0000-0000-000000000001"

_commercial_log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _parse_commercial_platform_tenant_ids(raw: str) -> frozenset[str]:
    """Canonical uuids from the CSV ``raw``; anything else is dropped + logged."""
    ids: set[str] = set()
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        try:
            parsed = uuid.UUID(candidate)
        except ValueError:
            parsed = None
        if parsed is None or parsed.int == 0 or candidate.lower() != str(parsed):
            _commercial_log.error(
                "commercial_platform_tenant_ids_entry_dropped entry=%r -- only canonical, "
                "non-nil tenant uuids are honoured (never a name)",
                candidate,
            )
            continue
        ids.add(str(parsed))
    return frozenset(ids)


def commercial_platform_tenant_ids() -> frozenset[str]:
    """The gitops-defined commercial-platform set (``COMMERCIAL_PLATFORM_TENANT_IDS``).

    Read through ``get_settings()`` on every call, so it follows the pod's
    environment; the parse is cached per raw string.
    """
    from src.settings import get_settings

    return _parse_commercial_platform_tenant_ids(get_settings().commercial_platform_tenant_ids)


def is_commercial_platform_tenant(tenant_id: object) -> bool:
    """True only for a tenant in :func:`commercial_platform_tenant_ids`.

    Use THIS (never :func:`is_platform_tenant`) wherever the question is
    "may this tenant use the owner's providers / any IR model". ``None``,
    empty, a non-string, the nil uuid, a name, or any non-canonical spelling is
    NOT platform: an unknown or absent tenant fails closed to the customer rules.

    ⛔ This is MEMBERSHIP only. Whether the owner's commercial KEYS may be used
    also needs the tenant to be proven by the request:
    ``src.auth.tenant_proof.owner_commercial_keys_permitted``.
    """
    if not isinstance(tenant_id, str) or not tenant_id:
        return False
    return tenant_id.strip().lower() in commercial_platform_tenant_ids()


def derive_tenant_uuid(label: str) -> str:
    """The deterministic uuid migration 024 assigns to an unregistered label."""
    return str(uuid.uuid5(TENANT_NAMESPACE, label))


def resolve_tenant_id(value: str) -> str | None:
    """Resolve an inbound tenant identifier to a canonical uuid string.

    Returns ``None`` when the caller supplied something that is neither a UUID
    nor a tenant this deployment knows — the caller turns that into a clean
    ``400``.

    ⛔ WHY THIS IS FAIL-CLOSED AND NOT FALL-BACK-TO-DEFAULT. Measured
    2026-08-25, `worker_states` held 93,842 rows whose tenant_id was not a UUID
    at all — 16 distinct literals such as ``sales``, ``auto``,
    ``__system_workflow__``, ``smoke-plan-7`` and bare snowflake integers. Every
    one of them entered through the trusted-network branch of
    ``auth_middleware``, which took ``X-Tenant-Id`` verbatim and unvalidated.
    Silently substituting the default here would keep that pipe open while
    hiding it: a typo'd tenant would quietly read and write the DEFAULT
    tenant's data, which is worse than the untyped column this replaces.

    A UUID is accepted whatever its case and returned lowercase, because
    Postgres compares uuids by value and would otherwise let ``ABC…`` and
    ``abc…`` look like two tenants in logs and metrics.

    ⛔ THE NIL UUID IS REFUSED BY VALUE — see :data:`NIL_TENANT_UUID`. It is the
    one unusable value that a *shape* check cannot catch, because it parses
    perfectly. This function is the single funnel for all three inbound
    resolution sites in ``auth_middleware`` (the tokenless ``X-Tenant-Id``
    header, the ``X-Effective-Tenant-Id`` delegation header, and the introspected
    licence claim), so rejecting it here closes the inbound boundary once rather
    than three times — and makes a fourth site impossible to add unguarded.
    """
    candidate = value.strip()
    if not candidate:
        return None
    if candidate in TENANT_UUIDS:
        return TENANT_UUIDS[candidate]
    try:
        parsed = uuid.UUID(candidate)
    except ValueError:
        return None
    if parsed.int == 0:
        return None
    # Reject the shorthand forms uuid.UUID() also accepts (urn:, braces, no
    # dashes): they round-trip to a different string than the caller sent, and a
    # tenant id that changes shape between the header and the log is a debugging
    # trap. Only canonical 8-4-4-4-12 is honoured.
    if candidate.lower() != str(parsed):
        return None
    return str(parsed)


def resolve_asserted_tenant_id(value: str) -> str | None:
    """Resolve a tenant a caller ASSERTED with no credential. Names refused.

    Same contract as :func:`resolve_tenant_id` — canonical uuid string, or
    ``None`` for the caller to turn into a clean ``400`` — with exactly one
    value class removed: a tenant **NAME**.

    ⛔ WHY A NAME IS REFUSED HERE AND ACCEPTED THERE. The two are not
    inconsistent; they answer different questions. ``resolve_tenant_id`` is also
    the funnel for values this deployment ITSELF supplies (``DEFAULT_TENANT_ID``
    and ``WM_MAINTENANCE_TENANT_ID``, both operator-set), where a name is a
    convenience and the operator picked which registry they meant. This function
    is the funnel for a value a REMOTE, UNAUTHENTICATED caller put on a header,
    where nobody picked anything.

    ⛔ AND FOR THAT CALLER A NAME HAS NO SINGLE ANSWER. Measured on worker4
    2026-09-16 against ``licenses.tenants``::

        name        this module's TENANT_UUIDS      licences registry
        n8n-dev     0195f48b-…-000000000001        019de842-c92a-7e37-8fdd-eb0d8c5396f9
        fr-test     0195f48b-…-000000000099        019efaa0-512b-733e-9e98-b1ac82a6f7ee

    and ``0195f48b-…-000000000001`` is registered as ``n8n-dev-portal-realm`` —
    a real, different, SEED tenant. So ``X-Tenant-Id: n8n-dev`` from a caller
    meaning production resolved to the seed tenant, returned 200, and wrote
    there: not an error, not a 404, a plausible wrong answer. Resolving the name
    "correctly" is not an available option because the two registries disagree
    about what correct is; refusing is the only choice that cannot silently
    misattribute.

    ⛔ DO NOT "FIX" THIS BY POINTING THE MAP AT THE LICENCES UUIDS. Those values
    are pinned into live Keycloak user attributes (see the module docstring);
    changing one moves real users between tenants. The map is right for what it
    is — a mirror of what the realm mints — and wrong only as an answer to an
    inbound header.

    A uuid is unambiguous in both registries, so it is the one spelling this
    boundary accepts. Every other rule (nil uuid by value, non-canonical
    braced / ``urn:`` / undashed spellings, case-folding, empty) is delegated to
    :func:`resolve_tenant_id` rather than restated, so the two boundaries cannot
    drift apart on anything except the name question.
    """
    candidate = value.strip()
    if candidate in TENANT_UUIDS:
        return None
    return resolve_tenant_id(candidate)


# Reverse of TENANT_UUIDS, built once. Two labels can never share a uuid (the
# realm forbids recycling a suffix), so this inversion is total.
_NAMES_BY_UUID: dict[str, str] = {v: k for k, v in TENANT_UUIDS.items()}


def tenant_label(tenant_id: str, tenant_name: str | None) -> str:
    """The human label to store in ``tenant_name`` alongside ``tenant_id``.

    ⛔ WHY THIS EXISTS. Eleven write paths in ``src/db/`` used to persist
    ``tenant_name or tenant_id``. ``tenant_name`` is NOT NULL, so whenever the
    caller had no label the tenant's UUID was written into the *label* column —
    which is exactly why, measured 2026-08-25, ``tenant_name`` was a
    byte-identical copy of ``tenant_id`` on all 952,808 rows of ``worker_states``
    (same=952808, differ=0). Migration 011 added that column to carry the human
    label; the fallback quietly ensured it never did.

    Migration 024 does NOT rewrite that column, and does not need to: 011
    backfilled it FROM ``tenant_id``, so every dirty row already carries its own
    legacy label there, and flipping ``tenant_id`` to a uuid leaves the label
    where it was. What 024 cannot do is stop the column filling up with uuids
    again — that is this helper's job, and without it the property would decay
    from the first write after the migration.

    An explicit label always wins. Otherwise a uuid this deployment knows is
    rendered as its name, and an unknown one falls back to the uuid string —
    still not a label, but there is nothing better to say, and it is now the
    exception rather than the rule.
    """
    if tenant_name:
        return tenant_name
    return _NAMES_BY_UUID.get(tenant_id, tenant_id)


def maintenance_tenant_id() -> str:
    """The tenant a maintenance script operates on.

    ⛔ WHY THIS EXISTS. Three scripts — ``cleanup_agents.py``,
    ``cleanup_schemathesis_workers.py`` and ``setup_graph_execution.py`` — passed
    ``tenant_id="system"`` to the registry, at seven call sites. There is no
    ``system`` tenant: measured against production 2026-08-25, the complete set
    of non-uuid ``tenant_id`` literals is 16 values and ``system`` is not among
    them, and nothing in ``src/`` has ever written it. Every one of those sweeps
    matched zero rows, inside an ``except Exception``, and the script exited 0
    having deleted nothing — a cleanup that reported success while doing
    nothing, for as long as it has existed.

    Once ``tenant_id`` is a ``uuid`` column the same call is a hard
    ``InvalidTextRepresentation``, still swallowed, and additionally retried
    three times with backoff by ``with_retry`` before being swallowed. Loud
    would have been better; silent is what it was.

    ``WM_MAINTENANCE_TENANT_ID`` overrides, and accepts a tenant NAME as well as
    a uuid. A maintenance script is not a request boundary, so an unusable value
    raises here rather than returning a refusal — the operator sees it before
    anything is deleted.
    """
    import os

    raw = os.getenv("WM_MAINTENANCE_TENANT_ID", DEFAULT_TENANT_UUID)
    resolved = resolve_tenant_id(raw)
    if resolved is None:
        raise ValueError(
            f"WM_MAINTENANCE_TENANT_ID={raw!r} is neither a UUID nor a known tenant name "
            f"({sorted(TENANT_UUIDS)}). Refusing to sweep an unresolvable tenant."
        )
    return resolved
