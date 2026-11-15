"""The licence/subject omission on the tokenless branch now has a number.

⛔ WHAT THIS PINS

``tests/test_unauthenticated_tenant_assertion.py`` covers the TENANT half of
the tokenless trusted-network branch: a caller inside ``TRUSTED_CIDRS`` naming
its own tenant in a bare header. That half already warns and increments.

This file is about the half that was still silent. On the same branch,
``request.state.license_id`` and ``request.state.subject_id`` are set to
``None`` unconditionally — no log line, no counter, no trace. Every cost event
produced downstream of a tokenless service-to-service caller is therefore
structurally licence-less and subject-less. Billing's own cost flusher records
that as the mechanism behind a historically ~53% NULL-licence rate, which
means the fact was only ever visible three services away, as a NULL rate on
someone else's table, with no way to attribute it to a caller.

The properties asserted here:

  * The omission is counted, per field, on every tokenless request.
  * It is attributed to the one identifier that was actually PROVEN (the
    verified JWT ``azp``), falling back to ``"<unverified>"`` — which is itself
    the finding, and the hardest class to migrate.
  * The two tokenless populations are distinguishable: the caller that
    asserted its own tenant (which arming the tenant gate WOULD 401) and the
    caller that sent no header at all (which it would not). Those are
    different migration problems and must not share a series.
  * ⛔ OBSERVE-ONLY. Nothing that is served today starts failing. The counter
    cannot refuse, and a broken metrics backend cannot either.
  * The tenant gate is NOT armed by this change — still default False.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import src.api.auth_middleware as auth_middleware_module
from src.api.auth_middleware import AuthenticationMiddleware
from src.settings import Settings, get_settings, reset_settings_cache

ENV_VAR = "WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT"

# Spelled correctly, proves nothing — the input class this branch accepts.
ASSERTED_TENANT = "0195f48b-d000-7000-8000-000000000003"

ASSERTED = auth_middleware_module._OMISSION_REASON_ASSERTED_TENANT
DEFAULTED = auth_middleware_module._OMISSION_REASON_DEFAULT_TENANT


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    """Clear the module-global warning dedup cache + the settings cache."""
    auth_middleware_module._assertion_log_seen.clear()
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("REQUIRE_AUTHENTICATED_TENANT", raising=False)
    reset_settings_cache()
    yield
    auth_middleware_module._assertion_log_seen.clear()
    reset_settings_cache()


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, object]:
        return {
            "tenant_id": request.state.tenant_id,
            "license_id": request.state.license_id,
            "subject_id": request.state.subject_id,
        }

    return application


def _call(app: FastAPI, headers: dict[str, str]) -> tuple[int, dict]:
    """Hit the app from a REAL peer inside TRUSTED_CIDRS.

    Not ``client=("testclient", …)``: that sentinel was removed as an auth
    bypass, so a real caller reaches this branch the same way — by address.
    """
    with TestClient(app, client=("10.42.1.7", 54321), raise_server_exceptions=False) as client:
        resp = client.get("/api/probe", headers=headers)
    body = (
        resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    )
    return resp.status_code, body


def _omissions(metrics: MagicMock) -> list[tuple[str, dict]]:
    """(field, labels) for every omission recorded, in call order."""
    return [(call.args[0], call.kwargs) for call in metrics.record_identity_omission.call_args_list]


# ---------------------------------------------------------------------------
# The omission is counted, per field, and attributed.
# ---------------------------------------------------------------------------


def test_the_tokenless_branch_counts_both_omitted_fields(app: FastAPI) -> None:
    """⛔ THE LOAD-BEARING ASSERTION IN THIS FILE.

    Before this, these two omissions produced no signal of any kind. Both
    fields are emitted, not a single "identity missing" tick, so a future
    branch that drops only one of them stays visible instead of averaging into
    the other.
    """
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        status, body = _call(app, {"X-Tenant-Id": ASSERTED_TENANT})

    assert status == 200
    # The omission this counts is real, asserted at the response, not assumed.
    assert body["license_id"] is None
    assert body["subject_id"] is None

    assert [field for field, _ in _omissions(metrics)] == ["license_id", "subject_id"]


def test_the_omission_is_attributed_to_the_route_and_the_caller(app: FastAPI) -> None:
    """A counter with nobody to go and fix is not evidence.

    ``<unverified>`` is the honest answer for a caller that presented no
    verified JWT, and it is the class that must be attributed by route before
    anyone can migrate it.
    """
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        _call(app, {"X-Tenant-Id": ASSERTED_TENANT})

    for _field, labels in _omissions(metrics):
        assert labels["path"] == "/api/probe"
        assert labels["caller"] == "<unverified>"
        assert labels["reason"] == ASSERTED


def test_a_verified_caller_is_named_rather_than_lumped_into_unverified(app: FastAPI) -> None:
    """The proven ``azp`` is the label — that is what makes the list actionable."""
    metrics = MagicMock()
    with (
        patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics),
        patch.object(
            auth_middleware_module,
            "get_caller_client_id_from_context",
            return_value="kag-service",
        ),
    ):
        _call(app, {"X-Tenant-Id": ASSERTED_TENANT})

    assert {labels["caller"] for _f, labels in _omissions(metrics)} == {"kag-service"}


@pytest.mark.parametrize(
    "segment",
    [
        "0195f48b-d000-7000-8000-000000000003",
        "12345",
        # The case the old character heuristic let through RAW — and this
        # counter carries FOUR labels, so the blow-up was multiplicative.
        "zz-probe-nonexistent-worker",
    ],
    ids=["uuid", "numeric", "alphabetic"],
)
def test_a_parameterised_segment_is_collapsed_to_the_route_template(segment: str) -> None:
    """A raw path carries worker names; the label must be the route, not the row."""
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.post("/workers/{worker_name}/message")
    async def _message(request: Request, worker_name: str) -> dict[str, str]:
        return {"ok": worker_name}

    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        with TestClient(
            application, client=("10.42.1.7", 54321), raise_server_exceptions=False
        ) as client:
            resp = client.post(
                f"/workers/{segment}/message",
                headers={"X-Tenant-Id": ASSERTED_TENANT},
            )

    assert resp.status_code == 200, "the request must actually route, or this is vacuous"
    assert {labels["path"] for _f, labels in _omissions(metrics)} == {
        "/workers/{worker_name}/message"
    }
    assert segment not in repr(metrics.record_identity_omission.call_args_list)


# ---------------------------------------------------------------------------
# The two tokenless populations stay distinguishable.
# ---------------------------------------------------------------------------


def test_the_no_header_caller_is_counted_under_a_different_reason(app: FastAPI) -> None:
    """⛔ This is the case the SIBLING counter deliberately does not count.

    A caller that sent no X-Tenant-Id asserted nothing, so it is (correctly)
    absent from the tenant-assertion counter — but its cost events are just as
    licence-less. If both populations shared one reason here, arming the tenant
    gate would look like it should drive this counter to zero, and it would
    not: the gate only fires when a header is present.
    """
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        status, body = _call(app, {})

    assert status == 200
    assert body["tenant_id"] == auth_middleware_module.DEFAULT_TENANT_ID
    assert body["license_id"] is None

    recorded = _omissions(metrics)
    assert [field for field, _ in recorded] == ["license_id", "subject_id"]
    assert {labels["reason"] for _f, labels in recorded} == {DEFAULTED}
    # The sibling counter stays silent here; that asymmetry is the point.
    metrics.record_unauthenticated_tenant_assertion.assert_not_called()


def test_the_two_reasons_are_distinct_values() -> None:
    """A shared constant would silently merge the two migration problems."""
    assert ASSERTED != DEFAULTED


# ---------------------------------------------------------------------------
# Cardinality guard.
# ---------------------------------------------------------------------------


def test_the_counter_carries_no_tenant_and_no_peer_in_its_labels(app: FastAPI) -> None:
    """⛔ Cardinality guard.

    Tenant and peer are caller-controlled on exactly this path, so putting
    either in a label lets the untrusted input this counter watches blow up the
    time series it is watched with.
    """
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        _call(app, {"X-Tenant-Id": ASSERTED_TENANT})

    for field, labels in _omissions(metrics):
        rendered = repr(field) + repr(labels)
        assert ASSERTED_TENANT not in rendered
        assert "10.42.1.7" not in rendered
        assert set(labels) == {"reason", "caller", "path"}


def test_the_caller_label_is_never_the_attacker_supplied_header(app: FastAPI) -> None:
    """The raw x-caller-client-id header is caller-chosen on this branch.

    Labelling it would let any peer inside TRUSTED_CIDRS both blow up the
    cardinality and stamp an innocent service as the owner of its omissions.
    Only the VERIFIED azp is used.
    """
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        _call(
            app,
            {"X-Tenant-Id": ASSERTED_TENANT, "X-Caller-Client-Id": "spoofed-innocent-service"},
        )

    for _field, labels in _omissions(metrics):
        assert labels["caller"] == "<unverified>"


# ---------------------------------------------------------------------------
# ⛔ Observe-only.
# ---------------------------------------------------------------------------


def test_a_broken_metrics_backend_still_serves_the_request(app: FastAPI) -> None:
    """⛔ Adding an observation must not add a failure mode.

    If the counter could propagate, this change would turn every tokenless
    request into a 500 the moment the OTLP collector misbehaved — breaking the
    ten-plus live workloads on this branch in order to watch them.
    """
    metrics = MagicMock()
    metrics.record_identity_omission.side_effect = RuntimeError("otlp exploded")
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        status, body = _call(app, {"X-Tenant-Id": ASSERTED_TENANT})

    assert status == 200
    assert body["tenant_id"] == ASSERTED_TENANT


def test_the_metrics_client_itself_being_unavailable_is_swallowed() -> None:
    """Even ACQUIRING the client must not escape this auditor.

    Asserted against the method directly rather than through a request, and
    that is not a convenience: on this branch the pre-existing
    ``_audit_unauthenticated_tenant_assertion`` calls ``get_metrics_client()``
    unguarded at the line above ours, so an end-to-end version of this test
    would 500 in the SIBLING and pass/fail for a reason that has nothing to do
    with the code under test. (That pre-existing gap is untouched here — it is
    not a regression from this change, and widening scope to "fix" it would
    itself be a behaviour change on a path this PR promised not to alter.)
    """
    request = MagicMock()
    request.url.path = "/api/probe"
    middleware = AuthenticationMiddleware(app=MagicMock(), bypass_auth=False)
    with patch.object(
        auth_middleware_module,
        "get_metrics_client",
        side_effect=RuntimeError("no metrics client"),
    ):
        # Must not raise.
        assert middleware._audit_identity_omission(request, ASSERTED_TENANT) is None


def test_the_counter_is_not_sampled(app: FastAPI) -> None:
    """Rate is read from the counter; the WARNING beside it is deduplicated.

    If this were ever moved inside that dedup guard the rate would silently
    become one-per-window while the hole stayed wide open.
    """
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        for _ in range(3):
            _call(app, {"X-Tenant-Id": ASSERTED_TENANT})

    # 3 requests x 2 fields.
    assert metrics.record_identity_omission.call_count == 6


# ---------------------------------------------------------------------------
# A proven identity is not an omission.
# ---------------------------------------------------------------------------


def test_a_licence_token_request_records_no_omission(app: FastAPI) -> None:
    """The token branch HAS an introspected identity — nothing was omitted.

    If this counted, the metric would measure traffic rather than the defect,
    and could never reach the zero that says the branch is safe to close.
    """
    metrics = MagicMock()
    with (
        patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value={
                "active": True,
                "tenant_id": "11111111-1111-4111-8111-111111111111",
                "subject_id": "user-uuid-1",
                "license_id": "lic-1",
            },
        ),
    ):
        status, body = _call(app, {"X-Tenant-Id": "Bearer a-real-licence-token"})

    assert status == 200
    assert body["tenant_id"] == "11111111-1111-4111-8111-111111111111"
    metrics.record_identity_omission.assert_not_called()


# ---------------------------------------------------------------------------
# ⛔ This change does NOT arm the tenant gate.
# ---------------------------------------------------------------------------


def test_the_tenant_gate_remains_default_false() -> None:
    """⛔ Instrument now, arm later — and NOT in the same change.

    Arming the gate before the registry is reshaped would 401 exactly the
    mis-modelled service callers the counters above exist to find. The counter
    added here is what tells us when it is safe; it is not permission to do it.
    """
    assert get_settings().require_authenticated_tenant is False
    assert Settings().require_authenticated_tenant is False


def test_the_omission_counter_has_no_enforcement_path() -> None:
    """The auditor returns nothing — there is no branch by which it can refuse.

    Its sibling returns an optional 401 because it is an enforcement point.
    This one is structurally incapable of being one, and that asymmetry is
    deliberate.
    """
    request = MagicMock()
    request.url.path = "/api/probe"
    middleware = AuthenticationMiddleware(app=MagicMock(), bypass_auth=False)
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        assert middleware._audit_identity_omission(request, ASSERTED_TENANT) is None
