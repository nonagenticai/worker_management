"""Tests for src/auth/tokenless_gate.py — the single choke point wiring every
tenant-reading route on the tokenless surface through one dependency.

Originally the 8 routes measured by
``WorkerManagementTokenlessTenantSurfaceGrew``. ROUND 16
(billing-licensing-estate-repair lane A3): live measurement 2026-09-17 showed
all 16 series for those 8 at 0 for 6h while the sibling
``worker_management_unauthenticated_tenant_assertion_total`` counter showed
697 served requests on ``/workers/{worker_name}/message`` in the same
window — at that moment, a route this gate had not yet been wired to. This
file's wiring list was widened to match every tenant-reading route the gate
now covers (50 routes measured 2026-09-18, this one included); see
``src/auth/tokenless_gate.py``'s module docstring and
``src/telemetry/agent_metrics.py``'s ``TOKENLESS_TENANT_REQUEST_ROUTES`` for
the authoritative current set.

Three layers, on purpose:

1. ``TestWiring`` — a STATIC, DB-free, network-free introspection of the real
   FastAPI route table proving every route in ``TOKENLESS_TENANT_REQUEST_ROUTES``
   is actually wired to the gate. This is the MUTATION CONTROL: delete a
   route's ``dependencies=[Depends(tokenless_tenant_gate(...))]`` in its
   router file and ``test_all_wired_routes_are_wired`` goes red — independent
   of whether ``AuthenticationMiddleware`` would still separately refuse the
   request. Measured live for ``workers_message``, round 16: removing that
   route's ``dependencies=`` entry turns this test RED (``AssertionError:
   send_message (/workers/{worker_name}/message) is not wired to
   tokenless_tenant_gate('workers_message')``); restoring it turns the same
   test, and ``test_every_pre_seeded_route_constant_is_actually_used_somewhere``
   alongside it, GREEN again.
2. ``TestEnforceCore`` — direct unit tests of ``_enforce_tokenless_tenant_gate``,
   the one function every wired route runs through: the provenance key
   (``asserted`` only — ``default``/``license``/``unknown`` are not this
   surface), the counter's three outcomes, and the 401.
3. ``TestProviderModelsLabel`` — the one route whose label is derived from a
   path parameter, bounded to 3 values.
4. ``TestEndToEnd`` — real requests through the actual
   ``AuthenticationMiddleware`` + the real ``/providers`` router, proving the
   wiring produces a genuine 200 + a genuine counter increment, not just a
   satisfied mock — AND that an absent-header (default-tenant) request is
   neither counted nor 401'd, in both flag states. The latter two are the
   MUTATION CONTROL for the provenance key: revert the gate to keying on
   ``request.state.tenant_id`` and both go red (flag off: counted as
   ``grace_open``; flag on: 401 from the gate on a request the middleware
   served).
"""

from __future__ import annotations

import types
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import src.auth.tokenless_gate as tokenless_gate_module
from src.api.auth_middleware import AuthenticationMiddleware
from src.api.identity_context import (
    TENANT_PROVENANCE_ASSERTED,
    TENANT_PROVENANCE_DEFAULT,
    TENANT_PROVENANCE_LICENSE,
    TENANT_PROVENANCE_UNKNOWN,
    set_identity_context,
    set_tenant_byo_lookup_outcome,
)
from src.api.providers._router import router as providers_router
from src.api.router import api_router
from src.api.tenant_constants import default_tenant_uuid
from src.auth.tokenless_gate import (
    ROUTE_PROVIDERS_LIST,
    _enforce_tokenless_tenant_gate,
    tokenless_tenant_gate,
    tokenless_tenant_gate_provider_models,
)
from src.settings import get_settings, reset_settings_cache
from src.telemetry.agent_metrics import (
    TOKENLESS_ROUTE_PROVIDERS_MODELS_INFERENCE_ROUTER,
    TOKENLESS_ROUTE_PROVIDERS_MODELS_OTHER,
    TOKENLESS_ROUTE_PROVIDERS_MODELS_VLLM,
    TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN,
    TOKENLESS_TENANT_REQUEST_OUTCOME_REFUSED,
    TOKENLESS_TENANT_REQUEST_OUTCOME_SERVED,
    TOKENLESS_TENANT_REQUEST_ROUTES,
)

ENV_VAR = "WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT"
ASSERTED_TENANT = "aaaaaaaa-0000-4000-8000-000000000003"


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("REQUIRE_AUTHENTICATED_TENANT", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture(autouse=True)
def _clean_identity_context():
    """The gate keys on the tenant-provenance ContextVar, which the real
    middleware sets per request. Unit tests set it by hand; reset it to
    ``unknown`` afterwards so one test's ``asserted`` cannot leak into the
    next one's "not this surface" assertion."""
    yield
    set_identity_context(None, None, None, tenant_provenance=TENANT_PROVENANCE_UNKNOWN)


def _fake_request(
    *,
    tenant_id: str | None = ASSERTED_TENANT,
    provenance: str = TENANT_PROVENANCE_ASSERTED,
) -> Any:
    """A minimal duck-typed Request plus the ContextVar the real middleware
    would have set alongside it: `_enforce_tokenless_tenant_gate` reads only
    `.state.tenant_id` (for its log line) and the provenance ContextVar (for
    its decision), so nothing heavier is needed (same convention as
    `tests/test_auth/test_caller_identity.py`'s `_FAKE_REQUEST`).
    """
    set_identity_context(tenant_id, None, None, tenant_provenance=provenance)
    state = types.SimpleNamespace(tenant_id=tenant_id)
    return cast("Any", types.SimpleNamespace(state=state))


# ═══════════════════════════════════════════════════════════════════════════
# 1. WIRING — the mutation control.
# ═══════════════════════════════════════════════════════════════════════════


def _unwrap_gate_route_label(call: Any) -> str | None:
    """Recover the route label a `tokenless_tenant_gate(route)` closure was
    built with, without executing it."""
    if getattr(call, "__name__", "") == "_gate" and call.__closure__:
        for cell in call.__closure__:
            try:
                value = cell.cell_contents
            except ValueError:  # pragma: no cover - empty cell, not our case
                continue
            if isinstance(value, str):
                return value
    return None


def _collect_gate_wiring() -> dict[str, dict[str, Any]]:
    """Walk the REAL app router tree and record, per target function, which
    tokenless-gate label (if any) is wired to it.

    Matches by `endpoint.__name__` rather than reconstructed path strings:
    this FastAPI version (0.137) defers `include_router` into
    `fastapi.routing._IncludedRouter` wrappers whose accumulated prefix is
    not trivial to reconstruct, but the leaf `APIRoute` objects — reached via
    `.original_router.routes` — are unaffected and carry the real `.path`,
    which this function also records for a human reading a failure.
    """
    target_fns = {
        "get_llm_catalog": "tenant_config_llm_catalog",
        "list_llm_configs": "tenant_config_llm_all",
        "list_providers": "providers_list",
        "list_provider_models": None,  # path-derived; see TestProviderModelsLabel
        "get_worker_status": "workers_worker_status",
        "get_worker_state_history": "workers_worker_states",
        "get_workers_states": "workers_states_list",
        # --- Round 16 ---
        "send_message": "workers_message",
        "get_llm_config": "tenant_config_llm_get",
        "put_llm_config": "tenant_config_llm_put",
        "delete_llm_config": "tenant_config_llm_delete",
        "get_modality_config": "tenant_config_modality_get",
        "list_modality_configs": "tenant_config_modality_all",
        "get_modality_catalog": "tenant_config_modality_catalog",
        "put_modality_config": "tenant_config_modality_put",
        "delete_modality_config": "tenant_config_modality_delete",
        "resolve_modality_config": "tenant_config_modality_resolve",
        "create_pydantic_ai_worker": "workers_create_pydantic_ai",
        "create_claude_code_worker": "workers_create_claude_code",
        "create_codex_worker": "workers_create_codex",
        "create_remote_claude_worker": "workers_create_remote_claude",
        "create_remote_codex_worker": "workers_create_remote_codex",
        "execute_graph": "graph_execution_execute",
        "execute_graph_stepwise": "graph_execution_execute_stepwise",
        "get_session_status": "graph_execution_session_status",
        "list_execution_sessions": "graph_execution_sessions_list",
        "create_graph_endpoint": "graph_management_create",
        "list_graphs_endpoint": "graph_management_list",
        "delete_all_graph_states_endpoint": "graph_management_delete_all_states",
        "delete_all_graphs_endpoint": "graph_management_delete_all",
        "get_graph_endpoint": "graph_management_get",
        "update_graph_endpoint": "graph_management_update",
        "delete_graph_states_endpoint": "graph_management_delete_states",
        "delete_graph_endpoint": "graph_management_delete",
        "get_graph_structure_endpoint": "graph_management_structure",
        "get_graph_state_history_endpoint": "graph_management_state_history",
        "validate_graph_endpoint": "graph_management_validate",
        "get_sessions_by_correlation_id": "graph_sessions_correlation",
        # --- Round 16, PR #276 audit follow-up: 13 more tenant-reading routes
        # on `crud_endpoints.py` / `state_endpoints.py` / `worker_states.py`
        # a first pass missed entirely (see `agent_metrics.py`'s comment
        # beside these labels).
        "create_worker_endpoint": "workers_create",
        "list_workers": "workers_list",
        "list_worker_types": "workers_types_list",
        "update_worker": "workers_update",
        "delete_all_workers": "workers_delete_all",
        "delete_worker": "workers_delete",
        "swap_worker_prompt": "workers_prompt_swap",
        "get_all_worker_states": "workers_all_states",
        "delete_all_worker_states": "workers_all_states_delete",
        "get_latest_worker_state": "workers_state_latest",
        "delete_worker_states": "workers_states_delete",
        "cleanup_worker_instance": "workers_instance_cleanup",
        "save_workers_state": "workers_states_create",
        # Owner ruling 2026-10-01: the BYO key test, gated like the write routes.
        "test_llm_config": "tenant_config_llm_test",
        # Owner ruling 2026-10-02: the customer-visible BYO-config audit.
        # Deliberately NOT wired: it REFUSES every tokenless caller, so it is
        # not tokenless surface and must not be counted on it.
        "get_llm_config_audit": None,
        # Deliberately NOT wired — no `ctx`/`get_tenant_context` dependency at
        # all, same exemption class as `/health`/`/healthz`/`/ready`. Listed
        # here (mapped to None, like the path-derived route) so a future
        # accidental wiring attempt is visible in this dict rather than
        # silently absent from both this test and TOKENLESS_TENANT_REQUEST_ROUTES.
        "execution_health_check": None,
    }
    found: dict[str, dict[str, Any]] = {}

    def walk(obj: Any) -> None:
        routes = getattr(obj, "routes", None)
        if routes is None:
            original = getattr(obj, "original_router", None)
            if original is not None:
                walk(original)
            return
        for route in routes:
            if isinstance(route, APIRoute):
                fn_name = getattr(route.endpoint, "__name__", None)
                if fn_name in target_fns:
                    labels = [
                        label
                        for dep in route.dependant.dependencies
                        if (label := _unwrap_gate_route_label(dep.call)) is not None
                    ]
                    uses_provider_models_gate = any(
                        dep.call is tokenless_tenant_gate_provider_models
                        for dep in route.dependant.dependencies
                    )
                    found[fn_name] = {
                        "path": route.path,
                        "methods": route.methods,
                        "gate_labels": labels,
                        "uses_provider_models_gate": uses_provider_models_gate,
                    }
            else:
                walk(route)

    walk(api_router)
    return found


class TestWiring:
    """The mutation control: delete a route's `dependencies=[Depends(...)]`
    wiring in its router file and this goes red.
    """

    def test_all_wired_routes_are_wired(self) -> None:
        """Was `test_all_eight_routes_are_wired` — renamed round 16 since the
        set is no longer 8. Covers every route in `target_fns` above that
        maps to a real label (the two `None` entries — the path-derived
        provider-models route and the intentionally-unwired health check —
        are asserted separately below)."""
        wiring = _collect_gate_wiring()

        expected_simple = {
            "get_llm_catalog": "tenant_config_llm_catalog",
            "list_llm_configs": "tenant_config_llm_all",
            "list_providers": "providers_list",
            "get_worker_status": "workers_worker_status",
            "get_worker_state_history": "workers_worker_states",
            "get_workers_states": "workers_states_list",
            "send_message": "workers_message",
            "get_llm_config": "tenant_config_llm_get",
            "put_llm_config": "tenant_config_llm_put",
            "delete_llm_config": "tenant_config_llm_delete",
            "get_modality_config": "tenant_config_modality_get",
            "list_modality_configs": "tenant_config_modality_all",
            "get_modality_catalog": "tenant_config_modality_catalog",
            "put_modality_config": "tenant_config_modality_put",
            "delete_modality_config": "tenant_config_modality_delete",
            "resolve_modality_config": "tenant_config_modality_resolve",
            "create_pydantic_ai_worker": "workers_create_pydantic_ai",
            "create_claude_code_worker": "workers_create_claude_code",
            "create_codex_worker": "workers_create_codex",
            "create_remote_claude_worker": "workers_create_remote_claude",
            "create_remote_codex_worker": "workers_create_remote_codex",
            "execute_graph": "graph_execution_execute",
            "execute_graph_stepwise": "graph_execution_execute_stepwise",
            "get_session_status": "graph_execution_session_status",
            "list_execution_sessions": "graph_execution_sessions_list",
            "create_graph_endpoint": "graph_management_create",
            "list_graphs_endpoint": "graph_management_list",
            "delete_all_graph_states_endpoint": "graph_management_delete_all_states",
            "delete_all_graphs_endpoint": "graph_management_delete_all",
            "get_graph_endpoint": "graph_management_get",
            "update_graph_endpoint": "graph_management_update",
            "delete_graph_states_endpoint": "graph_management_delete_states",
            "delete_graph_endpoint": "graph_management_delete",
            "get_graph_structure_endpoint": "graph_management_structure",
            "get_graph_state_history_endpoint": "graph_management_state_history",
            "validate_graph_endpoint": "graph_management_validate",
            "get_sessions_by_correlation_id": "graph_sessions_correlation",
            "create_worker_endpoint": "workers_create",
            "list_workers": "workers_list",
            "list_worker_types": "workers_types_list",
            "update_worker": "workers_update",
            "delete_all_workers": "workers_delete_all",
            "delete_worker": "workers_delete",
            "swap_worker_prompt": "workers_prompt_swap",
            "get_all_worker_states": "workers_all_states",
            "delete_all_worker_states": "workers_all_states_delete",
            "get_latest_worker_state": "workers_state_latest",
            "delete_worker_states": "workers_states_delete",
            "cleanup_worker_instance": "workers_instance_cleanup",
            "save_workers_state": "workers_states_create",
        }
        for fn_name, expected_label in expected_simple.items():
            entry = wiring.get(fn_name)
            assert entry is not None, f"route function {fn_name} not found in the app at all"
            assert expected_label in entry["gate_labels"], (
                f"{fn_name} ({entry['path']}) is not wired to "
                f"tokenless_tenant_gate({expected_label!r}) — the tokenless "
                "tenant surface just re-opened invisibly on this route"
            )

        # /providers/{name}/models: the 8th alert bullet's two routes share
        # one FastAPI route, gated by tokenless_tenant_gate_provider_models.
        provider_models_entry = wiring.get("list_provider_models")
        assert provider_models_entry is not None
        assert provider_models_entry["uses_provider_models_gate"] is True, (
            "GET /providers/{name}/models lost tokenless_tenant_gate_provider_models — "
            "both /providers/inference_router/models and /providers/vllm/models "
            "just re-opened invisibly"
        )

    def test_execution_health_check_is_deliberately_not_wired(self) -> None:
        """`GET /execution/health` takes no `ctx`/`get_tenant_context`
        dependency at all — it reads no tenant identity, same exemption class
        as `/health`/`/healthz`/`/ready` — so it must never be wired to the
        gate. Pinned so a future accidental `dependencies=[...]` addition
        there is caught, not just a future removal."""
        wiring = _collect_gate_wiring()
        entry = wiring.get("execution_health_check")
        assert entry is not None, "execution_health_check route not found in the app at all"
        assert entry["gate_labels"] == [], (
            "GET /execution/health is now wired to the tokenless gate, but it reads no "
            "tenant identity — this route should stay unwired (see the module docstring "
            "note next to TOKENLESS_ROUTE_GRAPH_EXECUTION_* in agent_metrics.py)"
        )

    def test_llm_audit_is_deliberately_not_wired(self) -> None:
        """``GET /tenant-config/llm/audit`` refuses tokenless callers (401), so
        it is NOT on this surface. The gate is a route-level dependency and runs
        before the refusal, so wiring it would count every refused request as
        ``served``. Pinned so a re-wiring is caught."""
        wiring = _collect_gate_wiring()
        entry = wiring.get("get_llm_config_audit")
        assert entry is not None, "get_llm_config_audit route not found in the app at all"
        assert entry["gate_labels"] == []
        assert "tenant_config_llm_audit" not in TOKENLESS_TENANT_REQUEST_ROUTES

    def test_every_pre_seeded_route_constant_is_actually_used_somewhere(self) -> None:
        """The inverse mistake: a label pre-seeded in agent_metrics.py that no
        route actually emits would sit at zero forever and read as "never
        happened" for a route that was never wired in the first place."""
        wiring = _collect_gate_wiring()
        wired_labels = {label for entry in wiring.values() for label in entry["gate_labels"]} | {
            TOKENLESS_ROUTE_PROVIDERS_MODELS_INFERENCE_ROUTER,
            TOKENLESS_ROUTE_PROVIDERS_MODELS_VLLM,
        }
        assert set(TOKENLESS_TENANT_REQUEST_ROUTES) <= wired_labels


# ═══════════════════════════════════════════════════════════════════════════
# 2. _enforce_tokenless_tenant_gate — direct unit tests.
# ═══════════════════════════════════════════════════════════════════════════

asyncio_test = pytest.mark.asyncio


class TestEnforceCore:
    ROUTE = "workers_states_list"

    @asyncio_test
    async def test_factory_binds_the_given_route_label(self) -> None:
        """`tokenless_tenant_gate(route)` itself, independent of any router."""
        gate = tokenless_tenant_gate(self.ROUTE)
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            await gate(_fake_request())
        metrics.record_tokenless_tenant_request.assert_called_once_with(
            self.ROUTE, TOKENLESS_TENANT_REQUEST_OUTCOME_SERVED
        )

    @asyncio_test
    async def test_proven_identity_is_not_this_surface(self) -> None:
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            await _enforce_tokenless_tenant_gate(
                _fake_request(tenant_id=ASSERTED_TENANT, provenance=TENANT_PROVENANCE_LICENSE),
                self.ROUTE,
            )
        metrics.record_tokenless_tenant_request.assert_not_called()

    @asyncio_test
    async def test_unknown_provenance_is_not_this_surface(self) -> None:
        """Nobody stated a provenance (no middleware ran, or a future branch
        forgot to) — the safe direction is silence, never a `served`."""
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            await _enforce_tokenless_tenant_gate(
                _fake_request(tenant_id=None, provenance=TENANT_PROVENANCE_UNKNOWN), self.ROUTE
            )
        metrics.record_tokenless_tenant_request.assert_not_called()

    @asyncio_test
    @pytest.mark.parametrize("flag", ["false", "true"])
    async def test_default_tenant_provenance_is_not_this_surface(
        self, monkeypatch: pytest.MonkeyPatch, flag: str
    ) -> None:
        """Absent X-Tenant-Id: the middleware lands the request on
        default_tenant_uuid() with provenance `default`. `request.state.tenant_id`
        is TRUTHY here — that is exactly the value the gate used to key on —
        yet the caller asserted nothing, so: not counted (matching the
        sibling counter, which skips the absent-header case) and, with the
        flag ON, not 401'd either (the middleware serves it; the gate must
        not contradict the layer that decides)."""
        monkeypatch.setenv(ENV_VAR, flag)
        reset_settings_cache()
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            await _enforce_tokenless_tenant_gate(
                _fake_request(
                    tenant_id=default_tenant_uuid(), provenance=TENANT_PROVENANCE_DEFAULT
                ),
                self.ROUTE,
            )
        metrics.record_tokenless_tenant_request.assert_not_called()

    @asyncio_test
    async def test_flag_off_serves_and_counts(self) -> None:
        assert get_settings().require_authenticated_tenant is False
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            await _enforce_tokenless_tenant_gate(_fake_request(), self.ROUTE)
        metrics.record_tokenless_tenant_request.assert_called_once_with(
            self.ROUTE, TOKENLESS_TENANT_REQUEST_OUTCOME_SERVED
        )

    @asyncio_test
    async def test_flag_on_refuses_401_and_counts_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_VAR, "true")
        reset_settings_cache()
        assert get_settings().require_authenticated_tenant is True

        metrics = MagicMock()
        from fastapi import HTTPException

        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            with pytest.raises(HTTPException) as exc_info:
                await _enforce_tokenless_tenant_gate(_fake_request(), self.ROUTE)

        assert exc_info.value.status_code == 401
        metrics.record_tokenless_tenant_request.assert_called_once_with(
            self.ROUTE, TOKENLESS_TENANT_REQUEST_OUTCOME_REFUSED
        )

    @asyncio_test
    async def test_grace_open_byo_lookup_is_a_distinct_outcome_not_refused(self) -> None:
        """Flag stays OFF; the request is served — but this same request's
        byo_mode lookup fell open inside the licences-unresolved grace
        window. That must count as 'grace_open', not silently fold into
        'served'."""
        set_tenant_byo_lookup_outcome(TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN)
        metrics = MagicMock()
        try:
            with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
                await _enforce_tokenless_tenant_gate(_fake_request(), self.ROUTE)
        finally:
            set_tenant_byo_lookup_outcome("")  # don't leak into the next test

        metrics.record_tokenless_tenant_request.assert_called_once_with(
            self.ROUTE, TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN
        )

    @asyncio_test
    async def test_grace_open_is_never_reported_when_enforced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal is a refusal even if this request's byo lookup also
        happened to grace-open — the flag governs the outcome label, not the
        lookup. (In production these two never actually coincide, since the
        middleware's own refusal runs BEFORE the byo_mode lookup — this pins
        the dependency's OWN precedence regardless.)"""
        monkeypatch.setenv(ENV_VAR, "true")
        reset_settings_cache()
        set_tenant_byo_lookup_outcome(TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN)
        metrics = MagicMock()
        from fastapi import HTTPException

        try:
            with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
                with pytest.raises(HTTPException):
                    await _enforce_tokenless_tenant_gate(_fake_request(), self.ROUTE)
        finally:
            set_tenant_byo_lookup_outcome("")

        metrics.record_tokenless_tenant_request.assert_called_once_with(
            self.ROUTE, TOKENLESS_TENANT_REQUEST_OUTCOME_REFUSED
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. GET /providers/{name}/models — path-derived label, bounded to 3 values.
# ═══════════════════════════════════════════════════════════════════════════


class TestProviderModelsLabel:
    @asyncio_test
    @pytest.mark.parametrize(
        ("name", "expected_route"),
        [
            ("inference_router", TOKENLESS_ROUTE_PROVIDERS_MODELS_INFERENCE_ROUTER),
            ("vllm", TOKENLESS_ROUTE_PROVIDERS_MODELS_VLLM),
            ("openai", TOKENLESS_ROUTE_PROVIDERS_MODELS_OTHER),
            ("anything-unregistered", TOKENLESS_ROUTE_PROVIDERS_MODELS_OTHER),
        ],
    )
    async def test_name_maps_to_a_bounded_label(self, name: str, expected_route: str) -> None:
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            await tokenless_tenant_gate_provider_models(_fake_request(), name)
        metrics.record_tokenless_tenant_request.assert_called_once_with(
            expected_route, TOKENLESS_TENANT_REQUEST_OUTCOME_SERVED
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. End-to-end through the REAL middleware + the REAL /providers router.
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """`/providers` needs no DB and no outbound call, so it is the one route
    of the 8 cheap enough for a full-stack TestClient run."""

    @pytest.fixture
    def app(self) -> FastAPI:
        application = FastAPI()
        application.add_middleware(AuthenticationMiddleware, bypass_auth=False)
        application.include_router(providers_router)
        return application

    def test_flag_off_tokenless_get_providers_is_200_and_counted(self, app: FastAPI) -> None:
        """Full real stack: middleware -> ContextVar -> gate -> handler.

        Test settings carry no ``license_introspect_url``, so
        ``_lookup_tenant_byo_mode`` genuinely transport-fails and this pod's
        first sighting genuinely falls open inside the grace window — the
        SAME real code path production hits during a licences blip. That
        makes this the strongest possible proof the ContextVar plumbing
        works end to end: outcome is ``grace_open``, not a plain ``served``,
        and it must be asserted as such rather than papered over.
        """
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            with TestClient(
                app, client=("10.42.1.7", 54321), raise_server_exceptions=False
            ) as client:
                resp = client.get("/providers", headers={"X-Tenant-Id": ASSERTED_TENANT})

        assert resp.status_code == 200
        metrics.record_tokenless_tenant_request.assert_called_once_with(
            ROUTE_PROVIDERS_LIST, TOKENLESS_TENANT_REQUEST_OUTCOME_GRACE_OPEN
        )

    def test_flag_off_absent_header_default_tenant_is_200_and_not_counted(
        self, app: FastAPI
    ) -> None:
        """The auditor's repro, flag OFF: no X-Tenant-Id, trusted peer. The
        middleware lands it on default_tenant_uuid() with provenance `default`
        (and its own byo lookup grace-opens, exactly as in the asserted case
        above — so if the gate keyed on `state.tenant_id` this would be
        counted `grace_open` for a tenant nobody asserted). It must be served
        and NOT counted at all, matching the sibling counter's absent-header
        semantics.

        MUTATION CONTROL: revert the gate's key to
        `getattr(request.state, "tenant_id", None)` and this goes red on the
        `assert_not_called`."""
        assert get_settings().require_authenticated_tenant is False
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            with TestClient(
                app, client=("10.42.1.7", 54321), raise_server_exceptions=False
            ) as client:
                resp = client.get("/providers")

        assert resp.status_code == 200
        metrics.record_tokenless_tenant_request.assert_not_called()

    def test_flag_on_absent_header_default_tenant_is_still_200_not_401(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The auditor's repro, flag ON: the middleware's refusal returns
        None for an absent header BEFORE it reads the flag, so it SERVES
        this request. The gate must not then 401 it — a route-level gate
        that refuses what the deciding layer admitted is a contradiction,
        not defence in depth. Still uncounted.

        MUTATION CONTROL: same revert as above; this goes red with a 401
        whose detail names `providers_list`."""
        monkeypatch.setenv(ENV_VAR, "true")
        reset_settings_cache()
        assert get_settings().require_authenticated_tenant is True
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            with TestClient(
                app, client=("10.42.1.7", 54321), raise_server_exceptions=False
            ) as client:
                resp = client.get("/providers")

        assert resp.status_code == 200, resp.text
        metrics.record_tokenless_tenant_request.assert_not_called()

    def test_a_proven_token_caller_is_never_counted_on_this_surface(self, app: FastAPI) -> None:
        """A request that never reaches the tokenless branch at all (no
        token, not on a trusted network) is a 401 from the MIDDLEWARE before
        routing — proving this surface's counter stays silent for traffic
        that was never tokenless in the first place."""
        metrics = MagicMock()
        with patch.object(tokenless_gate_module, "get_metrics_client", return_value=metrics):
            with TestClient(
                app, client=("8.8.8.8", 54321), raise_server_exceptions=False
            ) as client:
                resp = client.get("/providers")

        assert resp.status_code == 401
        metrics.record_tokenless_tenant_request.assert_not_called()
