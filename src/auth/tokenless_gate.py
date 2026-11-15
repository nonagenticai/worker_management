"""Single choke point for the tokenless-tenant-assertion request surface.

``WorkerManagementTokenlessTenantSurfaceGrew`` (gitops-production
``clusters/strato/services/prometheus/rules-c.yaml``) originally named 8
routes that serve a caller admitted purely on ``TRUSTED_CIDRS`` membership,
presenting no licence token, that names its own tenant in a bare
``X-Tenant-Id`` header (``src/api/auth_middleware.py``'s trusted-network
branch; see ``src/settings.py:79-117`` for why the governing flag,
``require_authenticated_tenant``, defaults OFF).

⚠️ ROUND 16 (billing-licensing-estate-repair lane A3): live measurement
2026-09-17 showed all 16 series for those original 8 routes at 0 over 6h,
while the sibling ``worker_management_unauthenticated_tenant_assertion_total``
counter showed the busiest tokenless-served route in that window was
``/workers/{worker_name}/message`` (697 served) — at that moment, a route
this gate had not yet been wired to. This module now wires every
tenant-reading route on the tokenless-eligible surface, not just the
original 8: worker messaging, tenant-config LLM GET/PUT/DELETE, tenant-config
modality CRUD + resolve (count-only, see its own docstring below), the 5
typed worker-creation convenience endpoints, graph execution, graph
management, and graph-sessions correlation. The current closed set is
``TOKENLESS_TENANT_REQUEST_ROUTES`` in ``src/telemetry/agent_metrics.py`` —
read it, do not assume "8" from this docstring's history. Measured
2026-09-18: 50 routes across ``src/api/`` carry
``dependencies=[Depends(tokenless_tenant_gate(...))]``,
``/workers/{worker_name}/message`` included — reproduce by grepping
``src/api/`` for that ``dependencies=`` line, one hit per wired route,
rather than trusting this number as it ages.

⚠️ PR #276 AUDIT FOLLOW-UP (same round): the sweep above still missed 13
more ``Depends(get_tenant_context)`` routes entirely — the rest of
``src/api/workers/crud_endpoints.py`` (create/list/list-types/update/
delete-all/delete/prompt-swap) and ``state_endpoints.py`` (all-states get
+ delete, latest-state, per-worker-states delete, instance cleanup), plus
``src/api/worker_states.py``'s ``save_workers_state``. These are now wired
too — see ``TOKENLESS_TENANT_REQUEST_ROUTES`` for the 13 new labels. A
second, exhaustive grep for ``get_tenant_context``/``request.state.tenant_id``/
``resolve_tenant_id`` across ``src/api/`` found exactly two more
tenant-identity readers that are deliberately NOT part of this gate:

* ``POST /chat/completions`` (``src/api/providers/_chat_router.py``) reads
  ``request.state.tenant_id`` directly (no ``Depends(get_tenant_context)``)
  purely to attribute a BYO cost event — the handler is documented
  stateless ("No worker registry, no DB writes — purely request/response"),
  so tenant identity never scopes a read or write here the way it does on
  every route above. Not in ``PUBLIC_ENDPOINTS`` either, so the middleware's
  own flag-gated refusal still applies to it like any other route; this
  per-route gate is additionally for the closed ``get_tenant_context``
  surface only, and this route was never part of it.
* ``src/api/workflow_routing/routes.py`` uses a DIFFERENT dependency,
  ``Depends(get_tenant_id)`` (license-token auth path, not the tokenless
  trusted-network path this gate measures), and already carries its own
  purpose-built telemetry (``WORKFLOW_ROUTING_AUTH_PATH_*``) plus an
  extensive in-file warning that its missing role-check is a KNOWN,
  deliberately-sequenced gap tracked as worker_management#230 — touching it
  here would be a second, uncoordinated fix to a problem already being
  worked in a specific order. Left alone on purpose.

That flag is ALREADY enforced once, centrally, in
``AuthenticationMiddleware._audit_unauthenticated_tenant_assertion`` — every
non-public request is refused there, before routing, when the flag is on. So
why a SECOND gate here, wired per-route? Two reasons, and neither is
redundancy:

1. **Measurability.** The middleware's own counter
   (``worker_management_unauthenticated_tenant_assertion_total``) is keyed on
   the matched route TEMPLATE and is, by its own docstring, sized for "which
   caller do I need to migrate", not "which of the named routes on THIS
   surface is serving tokenless traffic right now". This module's counter
   (``worker_management_tokenless_tenant_request_total``) answers that
   narrower question with a closed, hand-picked label set — see
   ``src/telemetry/agent_metrics.py``'s ``TOKENLESS_TENANT_REQUEST_ROUTES``.
2. **Defence in depth for ONE specific regression — NOT for bypasses in
   general.** This gate runs only when tenant provenance is ``asserted`` (see
   ``_enforce_tokenless_tenant_gate`` below); it returns silently for
   ``unknown`` provenance — ``test_unknown_provenance_is_not_this_surface``
   pins that on purpose. A future bypass branch, a broadened
   ``PUBLIC_ENDPOINTS``, or a reordering of the middleware's checks all leave
   provenance ``unknown`` on the request they let through, so THIS gate
   cannot catch any of them. Proven, not theoretical: with the flag on,
   adding ``/providers`` to ``PUBLIC_ENDPOINTS`` in memory made a
   tenant-header request return 200 and this gate never ran. What it DOES
   still back up is narrower: a broken refusal inside
   ``AuthenticationMiddleware._audit_unauthenticated_tenant_assertion``
   itself, for a request whose provenance is ALREADY ``asserted`` — nothing
   else. ``tests/`` proves this narrow case with a mutation control: remove a
   route's ``Depends(...)`` here and a wiring test goes red, independent of
   whether the middleware still catches it.

Usage — add to the route decorator's ``dependencies=`` (NOT a function
parameter; this returns nothing an endpoint needs):

    @router.get("/llm/catalog", dependencies=[Depends(tokenless_tenant_gate(
        TOKENLESS_ROUTE_TENANT_CONFIG_LLM_CATALOG
    ))])

For ``GET /providers/{name}/models``, whose route label depends on the path
parameter, use ``tokenless_tenant_gate_provider_models`` instead — see its
docstring for why that one route gets a name-derived label while none of the
other 7 do.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from src.api.identity_context import (
    TENANT_PROVENANCE_ASSERTED,
    get_tenant_byo_lookup_outcome_from_context,
    get_tenant_provenance_from_context,
)
from src.settings import get_settings
from src.telemetry.agent_metrics import (
    TOKENLESS_ROUTE_GRAPH_EXECUTION_EXECUTE,
    TOKENLESS_ROUTE_GRAPH_EXECUTION_EXECUTE_STEPWISE,
    TOKENLESS_ROUTE_GRAPH_EXECUTION_SESSION_STATUS,
    TOKENLESS_ROUTE_GRAPH_EXECUTION_SESSIONS_LIST,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_CREATE,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_ALL,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_ALL_STATES,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_STATES,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_GET,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_LIST,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_STATE_HISTORY,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_STRUCTURE,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_UPDATE,
    TOKENLESS_ROUTE_GRAPH_MANAGEMENT_VALIDATE,
    TOKENLESS_ROUTE_GRAPH_SESSIONS_CORRELATION,
    TOKENLESS_ROUTE_PROVIDERS_LIST,
    TOKENLESS_ROUTE_PROVIDERS_MODELS_INFERENCE_ROUTER,
    TOKENLESS_ROUTE_PROVIDERS_MODELS_OTHER,
    TOKENLESS_ROUTE_PROVIDERS_MODELS_VLLM,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_ALL,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_CATALOG,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_DELETE,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_GET,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_PUT,
    TOKENLESS_ROUTE_TENANT_CONFIG_LLM_TEST,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_ALL,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_CATALOG,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_DELETE,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_GET,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_PUT,
    TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_RESOLVE,
    TOKENLESS_ROUTE_WORKERS_ALL_STATES,
    TOKENLESS_ROUTE_WORKERS_ALL_STATES_DELETE,
    TOKENLESS_ROUTE_WORKERS_CREATE,
    TOKENLESS_ROUTE_WORKERS_CREATE_CLAUDE_CODE,
    TOKENLESS_ROUTE_WORKERS_CREATE_CODEX,
    TOKENLESS_ROUTE_WORKERS_CREATE_PYDANTIC_AI,
    TOKENLESS_ROUTE_WORKERS_CREATE_REMOTE_CLAUDE,
    TOKENLESS_ROUTE_WORKERS_CREATE_REMOTE_CODEX,
    TOKENLESS_ROUTE_WORKERS_DELETE,
    TOKENLESS_ROUTE_WORKERS_DELETE_ALL,
    TOKENLESS_ROUTE_WORKERS_INSTANCE_CLEANUP,
    TOKENLESS_ROUTE_WORKERS_LIST,
    TOKENLESS_ROUTE_WORKERS_MESSAGE,
    TOKENLESS_ROUTE_WORKERS_PROMPT_SWAP,
    TOKENLESS_ROUTE_WORKERS_STATE_LATEST,
    TOKENLESS_ROUTE_WORKERS_STATES_CREATE,
    TOKENLESS_ROUTE_WORKERS_STATES_DELETE,
    TOKENLESS_ROUTE_WORKERS_STATES_LIST,
    TOKENLESS_ROUTE_WORKERS_TYPES_LIST,
    TOKENLESS_ROUTE_WORKERS_UPDATE,
    TOKENLESS_ROUTE_WORKERS_WORKER_STATES,
    TOKENLESS_ROUTE_WORKERS_WORKER_STATUS,
    TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN,
    TOKENLESS_TENANT_REQUEST_OUTCOME_REFUSED,
    TOKENLESS_TENANT_REQUEST_OUTCOME_SERVED,
    TOKENLESS_TENANT_REQUEST_ROUTES,
    get_metrics_client,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

logger = logging.getLogger(__name__)

# Re-exported so a router file needs only this module's import, not
# agent_metrics's too. Kept as the SAME objects (not copies) — see
# agent_metrics.py for why re-declaring a closed set is the mistake, not the
# fix.
ROUTE_TENANT_CONFIG_LLM_CATALOG = TOKENLESS_ROUTE_TENANT_CONFIG_LLM_CATALOG
ROUTE_TENANT_CONFIG_LLM_ALL = TOKENLESS_ROUTE_TENANT_CONFIG_LLM_ALL
ROUTE_PROVIDERS_LIST = TOKENLESS_ROUTE_PROVIDERS_LIST
ROUTE_WORKERS_WORKER_STATUS = TOKENLESS_ROUTE_WORKERS_WORKER_STATUS
ROUTE_WORKERS_WORKER_STATES = TOKENLESS_ROUTE_WORKERS_WORKER_STATES
ROUTE_WORKERS_STATES_LIST = TOKENLESS_ROUTE_WORKERS_STATES_LIST

# --- Round 16 re-exports ----------------------------------------------------
ROUTE_WORKERS_MESSAGE = TOKENLESS_ROUTE_WORKERS_MESSAGE
ROUTE_TENANT_CONFIG_LLM_GET = TOKENLESS_ROUTE_TENANT_CONFIG_LLM_GET
ROUTE_TENANT_CONFIG_LLM_PUT = TOKENLESS_ROUTE_TENANT_CONFIG_LLM_PUT
ROUTE_TENANT_CONFIG_LLM_DELETE = TOKENLESS_ROUTE_TENANT_CONFIG_LLM_DELETE
ROUTE_TENANT_CONFIG_LLM_TEST = TOKENLESS_ROUTE_TENANT_CONFIG_LLM_TEST
ROUTE_TENANT_CONFIG_MODALITY_GET = TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_GET
ROUTE_TENANT_CONFIG_MODALITY_ALL = TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_ALL
ROUTE_TENANT_CONFIG_MODALITY_CATALOG = TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_CATALOG
ROUTE_TENANT_CONFIG_MODALITY_PUT = TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_PUT
ROUTE_TENANT_CONFIG_MODALITY_DELETE = TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_DELETE
ROUTE_TENANT_CONFIG_MODALITY_RESOLVE = TOKENLESS_ROUTE_TENANT_CONFIG_MODALITY_RESOLVE
ROUTE_WORKERS_CREATE_PYDANTIC_AI = TOKENLESS_ROUTE_WORKERS_CREATE_PYDANTIC_AI
ROUTE_WORKERS_CREATE_CLAUDE_CODE = TOKENLESS_ROUTE_WORKERS_CREATE_CLAUDE_CODE
ROUTE_WORKERS_CREATE_CODEX = TOKENLESS_ROUTE_WORKERS_CREATE_CODEX
ROUTE_WORKERS_CREATE_REMOTE_CLAUDE = TOKENLESS_ROUTE_WORKERS_CREATE_REMOTE_CLAUDE
ROUTE_WORKERS_CREATE_REMOTE_CODEX = TOKENLESS_ROUTE_WORKERS_CREATE_REMOTE_CODEX
ROUTE_GRAPH_EXECUTION_EXECUTE = TOKENLESS_ROUTE_GRAPH_EXECUTION_EXECUTE
ROUTE_GRAPH_EXECUTION_EXECUTE_STEPWISE = TOKENLESS_ROUTE_GRAPH_EXECUTION_EXECUTE_STEPWISE
ROUTE_GRAPH_EXECUTION_SESSION_STATUS = TOKENLESS_ROUTE_GRAPH_EXECUTION_SESSION_STATUS
ROUTE_GRAPH_EXECUTION_SESSIONS_LIST = TOKENLESS_ROUTE_GRAPH_EXECUTION_SESSIONS_LIST
ROUTE_GRAPH_MANAGEMENT_CREATE = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_CREATE
ROUTE_GRAPH_MANAGEMENT_LIST = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_LIST
ROUTE_GRAPH_MANAGEMENT_DELETE_ALL_STATES = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_ALL_STATES
ROUTE_GRAPH_MANAGEMENT_DELETE_ALL = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_ALL
ROUTE_GRAPH_MANAGEMENT_GET = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_GET
ROUTE_GRAPH_MANAGEMENT_UPDATE = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_UPDATE
ROUTE_GRAPH_MANAGEMENT_DELETE_STATES = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE_STATES
ROUTE_GRAPH_MANAGEMENT_DELETE = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_DELETE
ROUTE_GRAPH_MANAGEMENT_STRUCTURE = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_STRUCTURE
ROUTE_GRAPH_MANAGEMENT_STATE_HISTORY = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_STATE_HISTORY
ROUTE_GRAPH_MANAGEMENT_VALIDATE = TOKENLESS_ROUTE_GRAPH_MANAGEMENT_VALIDATE
ROUTE_GRAPH_SESSIONS_CORRELATION = TOKENLESS_ROUTE_GRAPH_SESSIONS_CORRELATION

# --- Round 16, PR #276 audit follow-up --------------------------------------
ROUTE_WORKERS_CREATE = TOKENLESS_ROUTE_WORKERS_CREATE
ROUTE_WORKERS_LIST = TOKENLESS_ROUTE_WORKERS_LIST
ROUTE_WORKERS_TYPES_LIST = TOKENLESS_ROUTE_WORKERS_TYPES_LIST
ROUTE_WORKERS_UPDATE = TOKENLESS_ROUTE_WORKERS_UPDATE
ROUTE_WORKERS_DELETE_ALL = TOKENLESS_ROUTE_WORKERS_DELETE_ALL
ROUTE_WORKERS_DELETE = TOKENLESS_ROUTE_WORKERS_DELETE
ROUTE_WORKERS_PROMPT_SWAP = TOKENLESS_ROUTE_WORKERS_PROMPT_SWAP
ROUTE_WORKERS_ALL_STATES = TOKENLESS_ROUTE_WORKERS_ALL_STATES
ROUTE_WORKERS_ALL_STATES_DELETE = TOKENLESS_ROUTE_WORKERS_ALL_STATES_DELETE
ROUTE_WORKERS_STATE_LATEST = TOKENLESS_ROUTE_WORKERS_STATE_LATEST
ROUTE_WORKERS_STATES_DELETE = TOKENLESS_ROUTE_WORKERS_STATES_DELETE
ROUTE_WORKERS_INSTANCE_CLEANUP = TOKENLESS_ROUTE_WORKERS_INSTANCE_CLEANUP
ROUTE_WORKERS_STATES_CREATE = TOKENLESS_ROUTE_WORKERS_STATES_CREATE

TOKENLESS_TENANT_ROUTES = TOKENLESS_TENANT_REQUEST_ROUTES


async def _enforce_tokenless_tenant_gate(request: Request, route: str) -> None:
    """The one function every wired route runs through. Do not duplicate this.

    Keyed on tenant PROVENANCE — HOW ``request.state.tenant_id`` came to be —
    never on the tenant id itself:

    * ``asserted`` (``TENANT_PROVENANCE_ASSERTED``): the tokenless
      trusted-network branch, and the caller NAMED a tenant in a bare
      ``X-Tenant-Id`` header (``auth_middleware.py``, the ``set_identity_context``
      call in that branch). This is the surface. Counted ``served``/
      ``grace_open`` on every call when the flag is off. The ``refused`` 401
      below IS reachable by calling this function directly (unit tests do,
      e.g. ``test_flag_on_refuses_401_and_counts_refused``), but NOT by live
      traffic: with the flag on, ``AuthenticationMiddleware`` already refuses
      an asserted tenant before ``call_next`` runs, so the router — and this
      dependency with it — never executes. On live traffic this counter's
      ``refused`` outcome is therefore dead code: it reads 0 on every route
      forever and is not a measurement of anything. See the module docstring
      for the one regression this gate genuinely still catches.
    * Anything else — ``license``/``delegated`` (token branch), ``bypass``,
      ``unknown``, and above all ``default`` — is NOT this surface. Returns
      without touching the counter.

    ⛔ WHY ``default`` MUST BE EXCLUDED, AND WHY IT IS NOT GIVEN ITS OWN
    OUTCOME. Before this fix (PR #273's audit) this keyed on
    ``request.state.tenant_id`` being truthy. The same trusted-network branch sets that to ``DEFAULT_TENANT_ID``
    when NO header is sent, so a request that asserted NOTHING was (a) counted
    as ``served``/``grace_open`` for the default tenant and (b) with the flag
    on, 401'd HERE after the middleware had already decided to SERVE it —
    ``_audit_unauthenticated_tenant_assertion`` returns ``None`` for an absent
    header before it ever reads the flag. Live, absent-header is the MAJORITY
    class on these routes, so the counter mostly measured the wrong thing and
    the flag would have broken callers the middleware deliberately admits.

    The sibling counter this alert is built on
    (``worker_management_unauthenticated_tenant_assertion_total``) counts
    ASSERTED tenants only: its emitter leaves the absent-header case
    deliberately uncounted "so the counter means one thing only". This counter
    matches that exactly — the ``default`` case is left uncounted rather than
    given a ``default_tenant`` outcome, so a ratio of the two is a ratio of
    the same population.
    """
    if get_tenant_provenance_from_context() != TENANT_PROVENANCE_ASSERTED:
        return
    tenant_id = getattr(request.state, "tenant_id", None)

    settings = get_settings()
    metrics = get_metrics_client()
    enforced = settings.require_authenticated_tenant

    if enforced:
        metrics.record_tokenless_tenant_request(route, TOKENLESS_TENANT_REQUEST_OUTCOME_REFUSED)
        logger.warning(
            "tokenless_tenant_request route=%s tenant=%s outcome=refused "
            "(require_authenticated_tenant is enabled)",
            route,
            tenant_id,
        )
        raise HTTPException(
            status_code=401,
            detail=(
                f"{route} requires a verified licence token when "
                "WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT is enabled; "
                "send `X-Tenant-Id: Bearer <token>`."
            ),
        )

    # Not refused. Distinguish a genuinely-served request from one that also
    # rode a fail-open BYO posture this same request — see
    # ``get_tenant_byo_lookup_outcome_from_context``'s docstring for why the
    # resolved ``byo_mode`` value alone cannot tell these apart.
    if get_tenant_byo_lookup_outcome_from_context() == TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN:
        metrics.record_tokenless_tenant_request(route, TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN)
        return

    metrics.record_tokenless_tenant_request(route, TOKENLESS_TENANT_REQUEST_OUTCOME_SERVED)


def tokenless_tenant_gate(route: str) -> Callable[[Request], Coroutine[Any, Any, None]]:
    """Dependency factory: bind one FIXED route label from the closed set.

    ``route`` must be one of ``TOKENLESS_TENANT_REQUEST_ROUTES``
    (``src/telemetry/agent_metrics.py``); nothing here validates that, on
    purpose — the counter's own pre-materialisation loop is the thing that
    would visibly disagree (a route label used here but absent from that
    loop shows up unseeded, not silently accepted).
    """

    async def _gate(request: Request) -> None:
        await _enforce_tokenless_tenant_gate(request, route)

    return _gate


async def tokenless_tenant_gate_provider_models(request: Request, name: str) -> None:
    """The one gate whose label is derived from a path parameter.

    ``GET /providers/{name}/models`` is a single FastAPI route serving many
    provider names, only two of which
    (``WorkerManagementTokenlessTenantSurfaceGrew``'s
    ``/providers/inference_router/models`` and ``/providers/vllm/models``)
    are among the 8 this alert named. Any OTHER provider name reaching this
    same route still needs a label — collapsing it into one of the two named
    ones would misattribute it, and minting a label per provider name would
    repeat the exact unbounded-cardinality mistake
    ``auth_middleware._route_shape`` was introduced to fix for the sibling
    counter (see ``tenant_config/_router.py``'s docstring). So a third,
    bounded bucket: ``providers_models_other``. ``name`` here is validated
    FastAPI path-parameter binding on this route (same parameter the endpoint
    itself receives), not a raw path segment.
    """
    if name == "inference_router":
        route = TOKENLESS_ROUTE_PROVIDERS_MODELS_INFERENCE_ROUTER
    elif name == "vllm":
        route = TOKENLESS_ROUTE_PROVIDERS_MODELS_VLLM
    else:
        route = TOKENLESS_ROUTE_PROVIDERS_MODELS_OTHER
    await _enforce_tokenless_tenant_gate(request, route)


__all__ = [
    "ROUTE_GRAPH_EXECUTION_EXECUTE",
    "ROUTE_GRAPH_EXECUTION_EXECUTE_STEPWISE",
    "ROUTE_GRAPH_EXECUTION_SESSIONS_LIST",
    "ROUTE_GRAPH_EXECUTION_SESSION_STATUS",
    "ROUTE_GRAPH_MANAGEMENT_CREATE",
    "ROUTE_GRAPH_MANAGEMENT_DELETE",
    "ROUTE_GRAPH_MANAGEMENT_DELETE_ALL",
    "ROUTE_GRAPH_MANAGEMENT_DELETE_ALL_STATES",
    "ROUTE_GRAPH_MANAGEMENT_DELETE_STATES",
    "ROUTE_GRAPH_MANAGEMENT_GET",
    "ROUTE_GRAPH_MANAGEMENT_LIST",
    "ROUTE_GRAPH_MANAGEMENT_STATE_HISTORY",
    "ROUTE_GRAPH_MANAGEMENT_STRUCTURE",
    "ROUTE_GRAPH_MANAGEMENT_UPDATE",
    "ROUTE_GRAPH_MANAGEMENT_VALIDATE",
    "ROUTE_GRAPH_SESSIONS_CORRELATION",
    "ROUTE_PROVIDERS_LIST",
    "ROUTE_TENANT_CONFIG_LLM_ALL",
    "ROUTE_TENANT_CONFIG_LLM_CATALOG",
    "ROUTE_TENANT_CONFIG_LLM_DELETE",
    "ROUTE_TENANT_CONFIG_LLM_GET",
    "ROUTE_TENANT_CONFIG_LLM_PUT",
    "ROUTE_TENANT_CONFIG_MODALITY_ALL",
    "ROUTE_TENANT_CONFIG_MODALITY_CATALOG",
    "ROUTE_TENANT_CONFIG_MODALITY_DELETE",
    "ROUTE_TENANT_CONFIG_MODALITY_GET",
    "ROUTE_TENANT_CONFIG_MODALITY_PUT",
    "ROUTE_TENANT_CONFIG_MODALITY_RESOLVE",
    "ROUTE_WORKERS_ALL_STATES",
    "ROUTE_WORKERS_ALL_STATES_DELETE",
    "ROUTE_WORKERS_CREATE",
    "ROUTE_WORKERS_CREATE_CLAUDE_CODE",
    "ROUTE_WORKERS_CREATE_CODEX",
    "ROUTE_WORKERS_CREATE_PYDANTIC_AI",
    "ROUTE_WORKERS_CREATE_REMOTE_CLAUDE",
    "ROUTE_WORKERS_CREATE_REMOTE_CODEX",
    "ROUTE_WORKERS_DELETE",
    "ROUTE_WORKERS_DELETE_ALL",
    "ROUTE_WORKERS_INSTANCE_CLEANUP",
    "ROUTE_WORKERS_LIST",
    "ROUTE_WORKERS_MESSAGE",
    "ROUTE_WORKERS_PROMPT_SWAP",
    "ROUTE_WORKERS_STATES_CREATE",
    "ROUTE_WORKERS_STATES_DELETE",
    "ROUTE_WORKERS_STATES_LIST",
    "ROUTE_WORKERS_STATE_LATEST",
    "ROUTE_WORKERS_TYPES_LIST",
    "ROUTE_WORKERS_UPDATE",
    "ROUTE_WORKERS_WORKER_STATES",
    "ROUTE_WORKERS_WORKER_STATUS",
    "TOKENLESS_TENANT_ROUTES",
    "tokenless_tenant_gate",
    "tokenless_tenant_gate_provider_models",
]
