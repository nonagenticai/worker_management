"""A BILLING RECORD MUST SAY HOW ITS TENANT WAS ESTABLISHED.

⛔ THE DEFECT, stated precisely. ``emit_byo_cost_event_task`` attributes a BYO
cost event to ``get_tenant_id_from_context()`` and refuses only when that value
is falsy. On the delegation branch of ``AuthenticationMiddleware`` that
ContextVar holds the ``X-Effective-Tenant-Id`` value — a header
``_effective_tenant`` validates for SPELLING ONLY (``resolve_tenant_id``: is it
a uuid, or a tenant name this deployment knows). The middleware KNOWS the
difference; it stamps ``request.state.tenant_proven = not delegated`` two
statements earlier. It then calls ``set_identity_context(...)``
unconditionally, and that function has no provenance parameter — so the one
fact that distinguishes a proven owner from an adopted one never reaches the
emitter, and the billed row cannot be told apart from a licensed one.

⛔ WHAT IS *NOT* THE DEFECT. Attributing to the ADOPTED tenant is INTENDED, and
refusing to emit would be the wrong fix twice over: it would drop real revenue,
and it would replace one lying field with its inverse. The middleware nulls
``license_id`` on this branch precisely so customer spend does not land on the
service account's entitlement — the design wants the customer billed. The
defect is that the attribution is UNVERIFIED and SILENT: ``AUTH_DELEGATING_SUBJECTS``
is a flat allowlist of subjects permitted to delegate to ANY tenant, and
nothing in this service ever checks that svc-kag may act for tenant X. Until
the event says so, the size of that population is unmeasurable in production.

These tests pin the DISTINGUISHABILITY, not a refusal.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.auth_middleware import AuthenticationMiddleware
from src.telemetry import cost_events

# A configured delegator. Real shape: the live set is svc-kag-karma / svc-kag /
# svc-requirements / svc-fr-backend, four subjects that may name ANY tenant.
SA_SUBJECT = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
SA_TENANT = "22222222-2222-4222-8222-222222222222"
# The tenant the service account ADOPTS. Nothing verifies it may.
ADOPTED_TENANT = "44444444-4444-4444-8444-444444444444"
# A tenant established by the caller's OWN licence token.
OWN_TENANT = "11111111-1111-4111-8111-111111111111"

# ⚠️ Hoisted out of the assertions deliberately. detect-secrets' Secret Keyword
# plugin matches on the KEY NAME, and the cost event has a field whose name ends
# in `_key`; asserting a plain string against it inline put an ordinary service
# name directly beside that name, and the plugin scored it as a credential. The
# field name is production's and must not change, so the ADJACENCY is what goes.
#
# ⛔ Do not restore the inline form, and do not quote it in a comment either:
# the scanner reads comments too. An earlier version of this very note spelled
# out the offending pattern as an example and re-tripped the gate on itself.
# Renaming beats baselining — a baseline entry silently widens what the scanner
# agrees to ignore next time.
THIS_SERVICE = "worker-management"


def _app(sink: list) -> FastAPI:
    """An app whose one route emits a BYO cost event, behind the REAL middleware.

    The route is deliberately reached through ``AuthenticationMiddleware``
    rather than by calling ``set_identity_context`` by hand: the claim under
    test is about what the middleware propagates, and a hand-set ContextVar
    would pin the emitter against a fixture instead of against the auth branch
    that actually runs in production.
    """
    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @app.get("/api/workers")
    async def emit_route():
        cost_events.emit_byo_cost_event_task(
            provider="anthropic",
            model="claude-opus-4-8",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1234,
            model_provider="anthropic",
            worker_name="karma_extractor",
        )
        # Drain the fire-and-forget task inside the request's own event loop.
        await asyncio.gather(*list(cost_events._background_tasks))
        return {"emitted": len(sink)}

    return app


def _fake_redis(sink: list) -> AsyncMock:
    client = AsyncMock()

    async def _xadd(stream, fields, **kwargs):
        sink.append(dict(fields))

    client.xadd.side_effect = _xadd
    return client


def _drive(monkeypatch, headers: dict, introspect: dict) -> list:
    """Run one request end-to-end and return the cost-event fields XADD'd."""
    sink: list = []
    monkeypatch.setattr(
        cost_events,
        "get_settings",
        lambda: SimpleNamespace(
            metering_redis_url="redis://metering:6379/0",
            metering_costs_stream="improvement-loop:costs",
        ),
    )
    monkeypatch.setattr(cost_events, "_get_redis", lambda: _fake_redis(sink))
    client = TestClient(_app(sink), raise_server_exceptions=False)
    with (
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect,
        ),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        resp = client.get("/api/workers", headers=headers)
    assert resp.status_code == 200, resp.text
    return sink


def _delegated(monkeypatch) -> dict:
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    sink = _drive(
        monkeypatch,
        headers={
            "X-Tenant-Id": "Bearer sa-token",
            "X-Effective-Tenant-Id": ADOPTED_TENANT,
            "X-Subject-Id": "end-user-7",
        },
        introspect={
            "active": True,
            "tenant_id": SA_TENANT,
            "subject_id": SA_SUBJECT,
            "byo_mode": "platform_only",
        },
    )
    assert len(sink) == 1, "the delegated cost event must still be emitted"
    return sink[0]


def _proven(monkeypatch) -> dict:
    monkeypatch.delenv("AUTH_DELEGATING_SUBJECTS", raising=False)
    sink = _drive(
        monkeypatch,
        headers={"X-Tenant-Id": "Bearer real-token"},
        introspect={
            "active": True,
            "tenant_id": OWN_TENANT,
            "subject_id": "user-uuid-1",
            "license_id": "lic-1",
        },
    )
    assert len(sink) == 1, "the licensed cost event must still be emitted"
    return sink[0]


# ---------------------------------------------------------------------------
# THE RED TEST
# ---------------------------------------------------------------------------


def test_an_adopted_tenant_is_distinguishable_from_a_proven_one(monkeypatch):
    """⭐ THE DEFECT. Two cost events, two different provenances, one record.

    Pre-fix both events carry an IDENTICAL ``usage_extra`` (``{"byo":"true"}``)
    and an identical field set apart from ``tenant_id`` — which is exactly the
    value in question, so it cannot be its own evidence. A downstream
    reconciler looking at ``billing_cost_events`` has no column, no flag and no
    JSON key that says "this tenant was taken from a header nobody verified".
    """
    delegated = _delegated(monkeypatch)
    proven = _proven(monkeypatch)

    d_ux = json.loads(delegated["usage_extra"])
    p_ux = json.loads(proven["usage_extra"])

    assert "tenant_provenance" in d_ux, (
        "a delegated cost event carries no provenance marker: "
        f"usage_extra={d_ux!r} — indistinguishable from a licensed row"
    )
    assert d_ux["tenant_provenance"] == "delegated"
    assert p_ux["tenant_provenance"] == "license"
    assert d_ux["tenant_provenance"] != p_ux["tenant_provenance"]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"X-Tenant-Id": OWN_TENANT}, "asserted"),
        ({}, "default"),
    ],
)
def test_a_tokenless_tenant_is_marked_asserted_or_default(monkeypatch, headers, expected):
    """The trusted-network branch names its own tenant and is unproven too.

    Same shape of claim as delegation, different owner and different fix
    ("present a token" vs "present a token FOR THAT TENANT"), so it must not
    collapse into ``delegated``. And a bill landing on DEFAULT_TENANT_ID because
    nobody named a tenant at all is a worse finding than one on a tenant a
    caller chose, so ``default`` is kept apart from ``asserted`` too.
    """
    monkeypatch.delenv("AUTH_DELEGATING_SUBJECTS", raising=False)
    sink: list = []
    monkeypatch.setattr(
        cost_events,
        "get_settings",
        lambda: SimpleNamespace(
            metering_redis_url="redis://metering:6379/0",
            metering_costs_stream="improvement-loop:costs",
        ),
    )
    monkeypatch.setattr(cost_events, "_get_redis", lambda: _fake_redis(sink))
    client = TestClient(_app(sink), raise_server_exceptions=False)
    with (
        patch("src.api.auth_middleware._is_trusted_network", return_value=True),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        resp = client.get("/api/workers", headers=headers)
    assert resp.status_code == 200, resp.text
    assert len(sink) == 1
    assert json.loads(sink[0]["usage_extra"])["tenant_provenance"] == expected


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS — stated BEFORE the fix was written
# ---------------------------------------------------------------------------


def test_a_proven_tenants_event_is_otherwise_byte_identical(monkeypatch):
    """NEGATIVE CONTROL 1: nothing billing prices on may move.

    Every pre-existing field keeps its exact value, and ``usage_extra`` keeps
    the ``byo`` flag metering's rollup engine keys ``cost_daily.byo`` off. The
    only difference a licensed row may show is the new provenance key reading
    ``license``.
    """
    proven = _proven(monkeypatch)
    assert proven["tenant_id"] == OWN_TENANT
    assert proven["license_id"] == "lic-1"
    assert proven["subject_id"] == "user-uuid-1"
    assert proven["service_key"] == THIS_SERVICE
    assert proven["cost_source"] == THIS_SERVICE
    assert proven["provider"] == "anthropic"
    assert proven["served_model"] == "claude-opus-4-8"
    assert proven["input_tokens"] == "1000"
    assert proven["output_tokens"] == "500"
    assert proven["worker_name"] == "karma_extractor"
    ux = json.loads(proven["usage_extra"])
    assert ux["byo"] == "true"
    assert "owner_provided" not in ux
    assert set(ux) == {"byo", "funding", "tenant_provenance"}
    assert ux["funding"] == "tenant_byo"


def test_the_delegated_event_is_still_emitted_and_still_bills_the_adopted_tenant(
    monkeypatch,
):
    """NEGATIVE CONTROL 2: no legitimate event stops being emitted.

    ⛔ The fix must NOT refuse. Delegation is a supported path serving four live
    service accounts; dropping its events is lost revenue and merely moves the
    lie. The adopted tenant is still billed, the licence is still NULL (so
    billing's ``resolve_license`` backstop can find one the adopted tenant
    actually owns), and the ``byo`` flag is intact.
    """
    delegated = _delegated(monkeypatch)
    assert delegated["tenant_id"] == ADOPTED_TENANT
    assert delegated["license_id"] == ""
    assert delegated["subject_id"] == "end-user-7"
    assert delegated["cost_source"] == THIS_SERVICE
    assert delegated["input_tokens"] == "1000"
    assert json.loads(delegated["usage_extra"])["byo"] == "true"


def test_provenance_values_are_a_closed_set_and_carry_no_caller_input(monkeypatch):
    """NEGATIVE CONTROL 3: the new value is a LABEL, so it must be bounded.

    A tenant uuid, a subject or any caller-supplied string here would be the
    unbounded-cardinality hole every other closed set in ``agent_metrics``
    exists to avoid.
    """
    from src.api.identity_context import TENANT_PROVENANCES

    observed = {
        json.loads(_delegated(monkeypatch)["usage_extra"])["tenant_provenance"],
        json.loads(_proven(monkeypatch)["usage_extra"])["tenant_provenance"],
    }
    assert observed <= set(TENANT_PROVENANCES)
    for value in TENANT_PROVENANCES:
        assert ADOPTED_TENANT not in value
        assert SA_SUBJECT not in value
    assert len(TENANT_PROVENANCES) <= 8


# ---------------------------------------------------------------------------
# THE CONTEXTVAR ITSELF — the pin for a guard whose violating input is
# otherwise only reachable through the middleware.
# ---------------------------------------------------------------------------


def test_provenance_defaults_to_unknown_never_to_license():
    """A caller that does not state provenance must not be READ as licensed.

    The direction of the default is the whole safety property: an emit site
    with no request context, a rollback to a peer that predates the parameter,
    or a future auth branch that forgets to pass it must all land on
    ``unknown`` — visible and reconcilable — and never on ``license``.
    """
    from src.api.identity_context import (
        TENANT_PROVENANCE_LICENSE,
        TENANT_PROVENANCE_UNKNOWN,
        get_tenant_provenance_from_context,
        set_identity_context,
    )

    set_identity_context("tenant-x", "lic-x", "sub-x")
    assert get_tenant_provenance_from_context() == TENANT_PROVENANCE_UNKNOWN
    assert get_tenant_provenance_from_context() != TENANT_PROVENANCE_LICENSE


def test_the_contextvar_itself_defaults_to_unknown_in_a_context_nobody_stamped():
    """⛔ ADDED AFTER A SURVIVING MUTANT. The test above does NOT cover this.

    ``set_identity_context`` always writes the var, so its keyword default is
    the only thing that test exercises — flipping the ``ContextVar(...)``
    default to ``license`` left the whole file green. Two different defaults,
    two different failure modes, and only one of them was pinned.

    This one is reached by any context in which ``set_identity_context`` never
    ran: a background task, a worker started outside a request, a thread with a
    fresh context. It is not reachable from the emit path TODAY (no auth
    branch ⇒ no tenant ⇒ the ``no_tenant`` drop fires first), which is exactly
    why nothing was holding it — an unreachable guard is the one a refactor
    deletes without a single test going red, and the day a second writer of the
    tenant var appears, this default is what decides whether an unstamped
    billing record reads as proven.
    """
    import contextvars

    from src.api.identity_context import (
        TENANT_PROVENANCE_LICENSE,
        TENANT_PROVENANCE_UNKNOWN,
        get_tenant_provenance_from_context,
    )

    observed = contextvars.Context().run(get_tenant_provenance_from_context)
    assert observed == TENANT_PROVENANCE_UNKNOWN
    assert observed != TENANT_PROVENANCE_LICENSE


@pytest.mark.asyncio
async def test_an_out_of_set_provenance_is_coerced_to_unknown_not_written_through(
    monkeypatch,
):
    """⛔ THE PIN FOR A GUARD NOTHING ELSE REACHES.

    No production call site can pass an out-of-set value today: all four are
    module constants. That is exactly why the coercion needed its own test —
    a guard whose violating input is unreachable is the guard a refactor
    deletes without a single test going red, and this one is load-bearing
    twice over: the value becomes a Prometheus LABEL (unbounded cardinality if
    it can be anything) and a billing FIELD (a reconciler that greps for
    ``!= 'license'`` would miss a row saying ``licence``/``LICENSE``/``licensed``).

    Coerced to ``unknown``, never dropped and never raised: this is the billing
    hot path, and refusing the emit would trade a mislabelled event for a lost
    one.
    """
    from src.api.identity_context import (
        TENANT_PROVENANCE_UNKNOWN,
        get_tenant_provenance_from_context,
        set_identity_context,
    )

    set_identity_context("t", "l", "s", tenant_provenance="LICENSE")
    assert get_tenant_provenance_from_context() == TENANT_PROVENANCE_UNKNOWN
    set_identity_context("t", "l", "s", tenant_provenance="")
    assert get_tenant_provenance_from_context() == TENANT_PROVENANCE_UNKNOWN

    sink: list = []
    monkeypatch.setattr(cost_events, "_get_redis", lambda: _fake_redis(sink))
    await cost_events._emit(
        stream="s",
        tenant_id="t",
        license_id="",
        subject_id="",
        provider="anthropic",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        tenant_provenance=f"tenant-{ADOPTED_TENANT}",
    )
    assert json.loads(sink[0]["usage_extra"])["tenant_provenance"] == (TENANT_PROVENANCE_UNKNOWN)


def test_the_metric_label_can_never_carry_an_out_of_set_value(monkeypatch):
    """The counter's label is bounded by the SAME closed set, or it is unbounded.

    Pinned separately from the billing field: they are two different coercions
    in two different modules, and a single test covering one would let the
    other drift open.
    """
    from src.api.identity_context import TENANT_PROVENANCE_UNKNOWN
    from src.telemetry.agent_metrics import get_metrics_client

    seen: list = []

    class _Counter:
        def add(self, amount, attributes=None):
            seen.append(attributes)

    client = get_metrics_client()
    monkeypatch.setattr(client, "enabled", True)
    monkeypatch.setattr(client, "meter", object())
    monkeypatch.setitem(client._counters, "metering_event_tenant_provenance", _Counter())

    client.record_metering_event_tenant_provenance(ADOPTED_TENANT)
    client.record_metering_event_tenant_provenance("license")

    assert seen == [
        {"provenance": TENANT_PROVENANCE_UNKNOWN},
        {"provenance": "license"},
    ]


@pytest.mark.asyncio
async def test_emit_stamps_the_provenance_it_is_given(monkeypatch):
    """``_emit`` writes the provenance it was handed, and defaults it safely."""
    from src.api.identity_context import TENANT_PROVENANCE_UNKNOWN

    sink: list = []
    monkeypatch.setattr(cost_events, "_get_redis", lambda: _fake_redis(sink))
    await cost_events._emit(
        stream="improvement-loop:costs",
        tenant_id="tenant-1",
        license_id="",
        subject_id="",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
    )
    assert json.loads(sink[0]["usage_extra"])["tenant_provenance"] == (TENANT_PROVENANCE_UNKNOWN)
