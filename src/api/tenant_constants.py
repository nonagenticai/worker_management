"""Canonical tenant UUID handling, and the boundary that enforces it.

`tenant_id` is a real Postgres ``uuid`` column on all 12 tenant-scoped tables
(migration ``024_tenant_id_is_a_uuid.sql``). A human label such as ``<n8n-dev>``
is a `tenant_name`, never a `tenant_id`, and a request carrying one must have it
RESOLVED here before it reaches the database — otherwise every statement of that
request dies with ``invalid input syntax for type uuid``.

WHERE THESE VALUES COME FROM
----------------------------
⛔ NOT FROM THIS FILE. The real tenant UUIDs — the label map, the platform set,
the release set, the system tenant and the default tenant — are supplied by the
environment (``Settings.tenant_uuids`` / ``platform_tenant_ids`` /
``release_tenant_ids`` / ``system_tenant_id`` / ``default_tenant_id``, set on the
Deployment in gitops). They used to be literals here; this repository is PUBLIC
and a literal publishes the tenant map of the production estate.

The portal realm writes the ``tenant_id`` user attribute in Keycloak, and the
gitops value mirrors that job's own table verbatim
(``realm-patch-script-configmap.yaml``). That is the authority:
`license-authz` resolves entitlement on the JWT's ``tenant_id`` claim, so
whatever the realm mints IS the tenant id.

Every accessor below FAILS CLOSED on an unset value: no label resolves, no
tenant is platform, and the boundary returns ``None`` (a clean 400 at the
caller) rather than guessing.

⚠️ THE LABELS DO NOT MATCH THE ``licenses`` DB, AND THAT IS KNOWN.
Migration 014 on the licences side seeded platform tenants onto the SAME uuid7
values, so a label here and the name in ``licenses.tenants`` can disagree.

⚠️ MEASURED 2026-09-16 on the licences registry: the labels in this service and
the names in ``licenses.tenants`` are NOT a 1:1 map — two labels point at a
uuid that the licences registry names differently, one label has no licences row
at all, and two licences-side tenants have no label here::

    this module's label  | licences registry name
    ---------------------+-------------------------
    <n8n-dev>            | <a different uuid again>
    <george-dev>         | <aa-agents>
    <fr-test>            | <fr-test-user>
    <cdmsfirst>          | <cdmsfirst>          (agree)
    <ahtapol>            | <ahtapol>            (agree)
    <ahtapol-probe>      | <absent>

⛔ THE ROW THAT MATTERS IS NOT THE MISMATCH, IT IS THE COLLISION. Two of the
labels above are ALSO live tenant names in the licences registry, pointing at a
DIFFERENT uuid. So the same string resolves to two different real tenants
depending on which registry answers, and neither answer is detectably wrong at
the point of use. That is why :func:`resolve_asserted_tenant_id` refuses a name
outright on an unauthenticated inbound header: there is no "correct" resolution
to pick.

The consequence is documented in that same configmap: a licence issued to the
licences-side tenant whose name collides with a label here is UNREACHABLE,
because no JWT ever carries that uuid.

⛔ Do NOT "fix" this by renumbering. These values are pinned into live Keycloak
user attributes; changing one moves real users between tenants and drains a
different quota pool. The same ⛔ is on the gitops side.
"""

from __future__ import annotations

import functools
import logging
import uuid

from src.settings import parse_tenant_id_csv, parse_tenant_label_map

# The all-zeros UUID. NOT a tenant, on any inbound path.
#
# ⛔ WHY IT NEEDS ITS OWN NAME AND ITS OWN CHECK. It is a syntactically valid
# UUID, so ``uuid.UUID()`` accepts it and every shape gate in this service passed
# it straight through into ``request.state.tenant_id`` — a caller on the pod
# network could simply ASSERT it in ``X-Tenant-Id``. It is also *registered* in
# ``licenses.tenants`` as ``default-anon``, so it survives a dirty-row audit
# while collapsing every unattributed caller into one bucket; an anonymous
# principal was once granted admin on exactly this tenant. Seven sibling services
# already reject it by value at their inbound boundary; this is that same rule,
# in the one place WM resolves a tenant.
#
# Checking ``.int == 0`` rather than the string covers the undashed, braced and
# ``urn:uuid:`` spellings for free — though ``resolve_tenant_id`` refuses those
# shapes anyway.
#
# ⚠️ NOT to be confused with the system tenant (a real tenant), nor with
# ``billing-service``'s deliberate registration of this value as
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

# The ``system`` tenant of the gitops COMMERCIAL set: ``00000000-…-0001``. A
# reserved synthetic id, not a provisioned workspace, kept here (and in gitops)
# so the two spellings cannot drift. It is deliberately NOT a member of
# ``platform_tenant_ids()``.
SYSTEM_TENANT_UUID: str = "00000000-0000-0000-0000-000000000001"

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment-backed accessors. Each is read through ``get_settings()`` on every
# call, so it follows the pod's environment; the parse is cached per raw string.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _cached_tenant_uuids(raw: str) -> dict[str, str]:
    return parse_tenant_label_map(raw, env_name="TENANT_UUIDS")


@functools.lru_cache(maxsize=8)
def _cached_tenant_ids(raw: str, env_name: str) -> frozenset[str]:
    return parse_tenant_id_csv(raw, env_name=env_name)


def tenant_uuids() -> dict[str, str]:
    """The ``label → tenant uuid`` map this deployment knows (gitops ``TENANT_UUIDS``).

    Empty when unset: no label resolves, and every inbound header carrying one
    is refused rather than guessed.
    """
    from src.settings import get_settings

    return _cached_tenant_uuids(get_settings().tenant_uuids)


def default_tenant_uuid() -> str:
    """The tenant the trusted-network path uses when a caller sends no header.

    ``DEFAULT_TENANT_ID`` may be a label from :func:`tenant_uuids` or a uuid.
    Empty when unset — that path then refuses instead of picking a tenant
    nobody chose.
    """
    from src.settings import get_settings

    raw = get_settings().default_tenant_id.strip()
    if not raw:
        return ""
    known = tenant_uuids()
    return known.get(raw, raw.lower())


def platform_tenant_ids() -> frozenset[str]:
    """The PLATFORM tenants (gitops ``PLATFORM_TENANT_IDS``); empty when unset.

    System data (worker prompts first) is writable only from here or by a
    verified elevated SA role.

    ⛔ MIRRORS fr's ``PLATFORM_TENANT_IDS`` (``internal-tenant-only.ts``). The two
    lists MUST change together — a tenant platform in one service and customer in
    the other is exactly the leak this closes. Drift is caught by
    ``tests/test_platform_tenant_ids_drift.py``: what used to be a comparison of
    two source literals is now a comparison of the two DEPLOYED values, which is
    the comparison that actually matters.
    """
    from src.settings import get_settings

    return _cached_tenant_ids(get_settings().platform_tenant_ids, "PLATFORM_TENANT_IDS")


def release_tenant_ids() -> frozenset[str]:
    """Tenants allowed to release workflows (gitops ``RELEASE_TENANT_IDS``).

    Always a subset of :func:`platform_tenant_ids` — enforced at startup in
    ``Settings._validate_tenant_sets``. Empty when unset: nobody may release.
    """
    from src.settings import get_settings

    return _cached_tenant_ids(get_settings().release_tenant_ids, "RELEASE_TENANT_IDS")


def system_tenant_id() -> str:
    """The SYSTEM tenant — owner of every system-written prompt (migration 035).

    Empty when unset; the prompt surfaces that need it refuse.
    """
    from src.settings import get_settings

    return get_settings().system_tenant_id.strip().lower()


def is_platform_tenant(tenant_id: str | None) -> bool:
    """True only for a canonical uuid in :func:`platform_tenant_ids`.

    Compared case-insensitively (Postgres compares uuids by value); a NAME is
    never matched — the same label resolves to different uuids in different
    registries (module docstring), so only the uuid is an answer.
    """
    if not tenant_id:
        return False
    return tenant_id.strip().lower() in platform_tenant_ids()


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
# ⛔ DEFINED ONCE, IN GITOPS — NOT HERE (fr#1385, 2026-10-03). Read from
# ``COMMERCIAL_PLATFORM_TENANT_IDS`` (``Settings.commercial_platform_tenant_ids``),
# set on the Deployment in gitops. An unset/empty value means NO platform tenant
# (fail closed). Rollout order is therefore WM FIRST, with the env unset (every
# tenant is a customer: local-llm + own BYO only — restrictive but safe), THEN
# the gitops PR that sets it.
#
# ⛔ WHY THIS IS A SEPARATE SET FROM :func:`platform_tenant_ids`. The two answer
# different questions and must not be merged:
#
# * :func:`platform_tenant_ids` gates SYSTEM-DATA writes (worker prompts) and
#   MIRRORS fr's list verbatim. Widening it grants prompt-write rights and
#   breaks the fr mirror contract. It is never edited from here.
# * This set gates which tenant may use the OWNER'S commercial LLM keys. The
#   gitops value is today the platform set plus the ``system`` tenant
#   (``SYSTEM_TENANT_UUID``), matching inference-router's
#   ``INFERENCE_ROUTER_PLATFORM_TENANT_IDS`` default. A tenant platform here and
#   customer in IR (or vice versa) would either strand our own calls or hand a
#   customer the owner's keys.
#
# ⛔ AN ALLOWLIST OF CANONICAL UUIDS. A tenant NAME, the nil uuid or any
# non-canonical spelling in the env value is DROPPED (and logged), never
# resolved: two labels collide with different licences-side tenants (module
# docstring). A tenant missing from it is a customer — the safe failure.
#
# ⚠️ Membership alone is NOT proof. On the tokenless trusted-network branch the
# tenant is caller-asserted and an absent header defaults to
# :func:`default_tenant_uuid`. That is why the owner-key gate also requires the
# request to have PROVEN the tenant.


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
            _log.error(
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
    known = tenant_uuids()
    if candidate in known:
        return known[candidate]
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

    ⛔ AND FOR THAT CALLER A NAME HAS NO SINGLE ANSWER. The same label resolves
    to one uuid in this module's map and to a DIFFERENT one in
    ``licenses.tenants`` (module docstring). So ``X-Tenant-Id: <a label>`` from a
    caller meaning production resolved to the seed tenant, returned 200, and
    wrote there: not an error, not a 404, a plausible wrong answer. Resolving the
    name "correctly" is not an available option because the two registries
    disagree about what correct is; refusing is the only choice that cannot
    silently misattribute.

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
    if candidate in tenant_uuids():
        return None
    return resolve_tenant_id(candidate)


@functools.lru_cache(maxsize=8)
def _names_by_uuid(raw: str) -> dict[str, str]:
    """Reverse of the label map.

    Two labels can never share a uuid (the realm forbids recycling a suffix), so
    this inversion is total — and it is parsed from the same raw string
    :func:`tenant_uuids` used, so it cannot go stale against it.
    """
    return {value: label for label, value in _cached_tenant_uuids(raw).items()}


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
    from src.settings import get_settings

    return _names_by_uuid(get_settings().tenant_uuids).get(tenant_id, tenant_id)


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
    anything is deleted. With neither set, this RAISES rather than picking a
    tenant: a sweep that silently chose a default is the exact bug above.
    """
    import os

    raw = os.getenv("WM_MAINTENANCE_TENANT_ID") or default_tenant_uuid()
    resolved = resolve_tenant_id(raw) if raw else None
    if resolved is None:
        raise ValueError(
            f"no maintenance tenant: WM_MAINTENANCE_TENANT_ID={raw!r} (and "
            f"DEFAULT_TENANT_ID) is neither a UUID nor a known tenant name "
            f"({sorted(tenant_uuids())}). Refusing to sweep an unresolvable tenant."
        )
    return resolved


# The system tenant as a module-level alias is GONE — read :func:`system_tenant_id`
# instead. It is config-supplied now, so a captured constant would go stale the
# moment the Deployment env changed.
