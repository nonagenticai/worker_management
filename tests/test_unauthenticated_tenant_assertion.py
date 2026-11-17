"""The tokenless trusted-network branch is observed now, and refusable later.

⛔ WHAT THIS PINS

``tests/test_tenant_boundary_rejects_unknown_tenants.py`` closed the SYNTACTIC
half of this hole: a tenant that cannot exist is now a 400 instead of a row.
This file is about the half that is still open — a caller inside
``TRUSTED_CIDRS`` presenting NO licence token can still NAME ITS OWN TENANT in
a bare ``X-Tenant-Id`` header, and any canonical uuid is accepted. There is no
allowlist, no subject check and no licence introspection on that branch, and
the resulting identity is stamped with a NULL licence.

Measured in billing 2026-08-31: worker-management emits cost events for 9
distinct tenant_ids across 27 worker types on this path, two of which exist in
NO licences registry. The Istio-admitted blast radius includes ``n8n``, which
runs user-authored workflows.

So the shape is inference-router #125's: **observe now, refuse later, default
open.** The three properties that must all hold:

  * OFF (the default): behaviour is byte-for-byte today's — served, same
    tenant — but the WARNING and the counter fire anyway. A flag whose "off"
    state also silenced the observation would leave us with no list of callers
    to migrate, i.e. no way to ever turn it on.
  * ON: the same request is a 401.
  * A request that actually proved its tenant (licence token, or the
    delegation path) is untouched in BOTH states. If turning the gate on could
    break an authenticated caller, it would never be turned on.

And the safety property, asserted explicitly because it is the one that
protects production: with the env var UNSET the default is ``False``. Ten-plus
live workloads are on this path; a default-closed flag 401s them on the next
pod roll.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import src.api.auth_middleware as auth_middleware_module
from src.api.auth_middleware import AuthenticationMiddleware
from src.api.tenant_constants import default_tenant_uuid
from src.settings import Settings, get_settings, reset_settings_cache

ENV_VAR = "WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT"

# A canonical uuid that is spelled correctly and proves nothing — exactly the
# input class the middleware accepts today.
ASSERTED_TENANT = "aaaaaaaa-0000-4000-8000-000000000003"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    """Clear the WARNING dedup cache + the settings cache around every test.

    The dedup cache is module-global and keyed on (tenant, peer, shape,
    caller), so without this a second test in the same window would observe
    "no warning" and pass for the wrong reason.
    """
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


def _assertion_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Only the assertion WARNINGs.

    The tokenless branch also logs a byo_mode-lookup warning (no licences URL
    in tests), so selecting by position rather than content would assert
    against the wrong line — and would keep passing if this line vanished.
    """
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "unauthenticated tenant assertion" in r.getMessage()
    ]


def _assertion_warning(caplog: pytest.LogCaptureFixture) -> str:
    msgs = _assertion_warnings(caplog)
    assert msgs, "the branch was taken silently — nothing to migrate callers from"
    return msgs[-1]


def _call(app: FastAPI, headers: dict[str, str]) -> tuple[int, dict]:
    """Hit the app from a REAL peer inside TRUSTED_CIDRS.

    Not ``client=("testclient", …)``: that sentinel was removed as an auth
    bypass (see tests/test_testclient_peer_is_not_an_auth_bypass.py), so a real
    caller reaches this branch the same way — by its source address.
    """
    with TestClient(app, client=("10.42.1.7", 54321), raise_server_exceptions=False) as client:
        resp = client.get("/api/probe", headers=headers)
    body = (
        resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    )
    return resp.status_code, body


# ---------------------------------------------------------------------------
# (d) The safety property: the default is False.
# ---------------------------------------------------------------------------


def test_the_default_is_false_when_the_env_var_is_unset() -> None:
    """⛔ THE LOAD-BEARING ASSERTION IN THIS FILE.

    Ten-plus live workloads depend on the tokenless branch right now. If this
    ever defaults True, every one of them 401s on the next pod roll — a
    self-inflicted outage shipped by a one-character diff. Asserted against a
    freshly constructed Settings with the variable absent, not against a
    fixture that could be masking it.
    """
    assert get_settings().require_authenticated_tenant is False
    assert Settings().require_authenticated_tenant is False


@pytest.mark.parametrize(
    "env_name",
    [ENV_VAR, "REQUIRE_AUTHENTICATED_TENANT"],
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("1", True), ("false", False), ("0", False)],
)
def test_the_flag_binds_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, env_name: str, raw: str, expected: bool
) -> None:
    """Both accepted env names bind, so the manifest can name either."""
    monkeypatch.setenv(env_name, raw)
    reset_settings_cache()
    assert get_settings().require_authenticated_tenant is expected


# ---------------------------------------------------------------------------
# (a) Flag OFF — served as today, warning fires, counter increments.
# ---------------------------------------------------------------------------


def test_flag_off_serves_the_asserted_tenant_exactly_as_today(app: FastAPI) -> None:
    """Byte-for-byte today's behaviour: 200, the asserted tenant, NULL licence."""
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        status, body = _call(app, {"X-Tenant-Id": ASSERTED_TENANT})

    assert status == 200
    assert body["tenant_id"] == ASSERTED_TENANT
    # The NULL licence is the whole problem, stated as an assertion: this
    # identity is stamped with nothing that proves it.
    assert body["license_id"] is None
    assert body["subject_id"] is None


def test_flag_off_still_warns_and_names_the_peer(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """OFF is NOT "off" — the observation is the point of shipping the flag open.

    The line has to be actionable on its own: without the peer and the tenant
    an operator sees a counter go up with nobody to go and fix.
    """
    caplog.set_level(logging.WARNING, logger="src.api.auth_middleware")
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        status, _ = _call(
            app,
            {
                "X-Tenant-Id": ASSERTED_TENANT,
                "User-Agent": "kag-orchestrator/1.2.3",
                "X-Correlation-Id": "run-7",
            },
        )
    assert status == 200

    # Selected by content, not by position: the tokenless branch also emits a
    # byo_mode-lookup warning afterwards, so ``records[-1]`` is not this line.
    msg = _assertion_warning(caplog)
    assert ASSERTED_TENANT in msg  # the asserted tenant
    assert "10.42.1.7" in msg  # the peer
    assert "/api/probe" in msg  # the path
    assert "kag-orchestrator/1.2.3" in msg  # caller-identifying header
    assert "run-7" in msg
    assert "SERVED" in msg
    # It must say what has to happen before the gate can close.
    assert ENV_VAR in msg
    assert "Bearer" in msg


def test_flag_off_increments_the_counter_with_enforced_false(app: FastAPI) -> None:
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        status, _ = _call(app, {"X-Tenant-Id": ASSERTED_TENANT})
    assert status == 200
    metrics.record_unauthenticated_tenant_assertion.assert_called_once_with(
        "/api/probe", enforced=False
    )


def test_the_counter_carries_no_tenant_and_no_peer_in_its_labels(app: FastAPI) -> None:
    """⛔ Cardinality guard.

    Tenant and peer are caller-controlled on exactly this path, so putting
    either in a label lets the untrusted input this counter watches blow up the
    time series it is watched with. They live in the (deduplicated) log line.
    """
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        _call(app, {"X-Tenant-Id": ASSERTED_TENANT})
    args, kwargs = metrics.record_unauthenticated_tenant_assertion.call_args
    rendered = repr(args) + repr(kwargs)
    assert ASSERTED_TENANT not in rendered
    assert "10.42.1.7" not in rendered
    assert set(kwargs) == {"enforced"}


@pytest.mark.parametrize(
    "segment",
    [
        "aaaaaaaa-0000-4000-8000-000000000003",
        "12345",
        # ⛔ THE CASE THIS TEST USED TO MISS. The two above are the only ones the
        # old character heuristic collapsed; an alphabetic-after-stripping name
        # went into the label RAW, and ``POST /workers/{worker_name}/message``
        # AUTO-CREATES whatever name it is handed. Asserting only the two that
        # already worked proved the half that was not broken.
        "zz-probe-nonexistent-worker",
    ],
    ids=["uuid", "numeric", "alphabetic"],
)
def test_a_parameterised_segment_is_collapsed_to_the_route_template(segment: str) -> None:
    """A raw path carries worker names; the label must be the route, not the row.

    Driven through a real ASGI app rather than by calling a shaping function
    directly: the label now comes from ``scope["route"]``, which only the router
    can populate, so a test that computed a shape from a string in isolation
    could not tell a working extractor from a broken one. The route registered
    here is also spelled the way this app actually mounts it — the previous
    version of this test asserted about ``/api/workers/.../messages``, a prefix
    this service does not mount with a plural segment that does not exist.
    """
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.post("/workers/{worker_name}/message")
    async def _message(worker_name: str) -> dict[str, str]:
        return {"worker_name": worker_name}

    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        with TestClient(
            application, client=("10.42.1.7", 54321), raise_server_exceptions=False
        ) as client:
            resp = client.post(
                f"/workers/{segment}/message", headers={"X-Tenant-Id": ASSERTED_TENANT}
            )

    assert resp.status_code == 200, "the request must actually route, or this is vacuous"
    metrics.record_unauthenticated_tenant_assertion.assert_called_once_with(
        "/workers/{worker_name}/message", enforced=False
    )
    assert segment not in repr(metrics.record_unauthenticated_tenant_assertion.call_args_list)


def test_a_static_route_is_labelled_with_itself() -> None:
    """The fix must not collapse every route into one bucket."""
    metrics = MagicMock()
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def _probe() -> dict[str, bool]:
        return {"ok": True}

    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        with TestClient(
            application, client=("10.42.1.7", 54321), raise_server_exceptions=False
        ) as client:
            client.get("/api/probe", headers={"X-Tenant-Id": ASSERTED_TENANT})

    metrics.record_unauthenticated_tenant_assertion.assert_called_once_with(
        "/api/probe", enforced=False
    )


def test_no_header_is_not_an_assertion_and_is_not_counted(app: FastAPI) -> None:
    """The absent-header default tenant is this deployment's config, not a claim.

    Counted separately (i.e. not at all here) so the counter means exactly one
    thing and cannot be read as "10 workloads assert tenants" when half of them
    sent no header.
    """
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        status, body = _call(app, {})
    assert status == 200
    assert body["tenant_id"] == default_tenant_uuid()
    metrics.record_unauthenticated_tenant_assertion.assert_not_called()


# ---------------------------------------------------------------------------
# (b) Flag ON — the same request is a 401.
# ---------------------------------------------------------------------------


def test_flag_on_refuses_the_asserted_tenant_with_401(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, "true")
    reset_settings_cache()
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        status, body = _call(app, {"X-Tenant-Id": ASSERTED_TENANT})

    assert status == 401
    assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
    # 401, not 400: the value is well-formed, the caller's AUTHORITY is what is
    # missing, and presenting a token is a retry that would succeed.
    #
    # ⚠️ ``<unmatched>``, not ``/api/probe``, and that is a DELIBERATE trade, not
    # a slip. The ``path`` label is now the MATCHED ROUTE TEMPLATE read out of
    # ``scope["route"]`` — the only source that is bounded by the route table
    # rather than by a guess about the caller's spelling (see
    # tests/test_metric_path_is_the_route_template.py). A refusal returns before
    # the router ever runs, so this request HAS no template. The alternative was
    # to keep deriving a shape from the raw path with a character heuristic,
    # which is what let ``/workers/<any-alphabetic-name>/message`` mint a
    # permanent series named after a caller-chosen string.
    #
    # THE TRADE IS NOT ONE-DIRECTIONAL, and an earlier version of this comment
    # got it wrong in both directions:
    #  * UNENFORCED (the default, and where all live traffic is) STRICTLY
    #    IMPROVES. The old heuristic collapsed distinct routes together — two
    #    routes whose first segments were id-ish both became ``/{id}`` — and
    #    split one route across every caller-chosen name. Both are fixed.
    #  * ENFORCED loses the per-route breakdown of the COUNTER, and only of the
    #    counter. It was measured that the WARNING was losing it too — the dedup
    #    key contained the (now constant) shape, so three refusals to /alpha,
    #    /bravo and /charlie produced ONE line naming only /alpha. That was a
    #    real regression and is fixed at the source: on the unmatched leg the
    #    raw path joins the dedup key, so each refusal target is named once per
    #    window again (test_each_refused_route_is_still_named_in_the_log below).
    #    Do not "simplify" that 5th key component away.
    metrics.record_unauthenticated_tenant_assertion.assert_called_once_with(
        auth_middleware_module._UNMATCHED_PATH, enforced=True
    )


def test_flag_on_says_in_the_log_that_it_refused(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ENV_VAR, "true")
    reset_settings_cache()
    caplog.set_level(logging.WARNING, logger="src.api.auth_middleware")
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        _call(app, {"X-Tenant-Id": ASSERTED_TENANT})
    assert "REFUSED (401)" in _assertion_warning(caplog)


def test_each_refused_route_is_still_named_in_the_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """⛔ ENFORCED MODE MUST NOT COLLAPSE ALL REFUSALS INTO ONE LOG LINE.

    A refusal returns before the router runs, so its metric shape is the
    constant ``<unmatched>``. The WARNING's dedup key contains that shape, so
    without the raw path as a 5th component the FIRST refused route consumes the
    key and every other target is silently deduplicated away — measured on an
    earlier revision of this branch: three refusals, ONE warning, only /alpha
    named. The counter's per-route breakdown is genuinely gone in enforced mode;
    the log's must not be, because the log is then the only attribution left.
    """
    monkeypatch.setenv(ENV_VAR, "true")
    reset_settings_cache()
    caplog.set_level(logging.WARNING, logger="src.api.auth_middleware")

    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        with TestClient(
            application, client=("10.42.1.7", 54321), raise_server_exceptions=False
        ) as client:
            for target in ("/alpha", "/bravo", "/charlie"):
                assert (
                    client.get(target, headers={"X-Tenant-Id": ASSERTED_TENANT}).status_code == 401
                )

    warnings = _assertion_warnings(caplog)
    assert len(warnings) == 3, "distinct refusal targets were deduplicated into one line"
    for target in ("/alpha", "/bravo", "/charlie"):
        assert any(f"path={target} " in w for w in warnings), f"{target} was never named"


def test_the_matched_leg_keeps_its_tight_dedup(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """⛔ The other half: the raw path must NOT widen the key when a route matched.

    The unenforced hot path is used by 10+ workloads on every request. If the
    raw path joined the key there, ``/workers/<name>/message`` would emit one
    WARNING per worker name — reproducing, in the log, the exact unbounded
    fan-out this whole change removed from the metric.
    """
    caplog.set_level(logging.WARNING, logger="src.api.auth_middleware")
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        for _ in range(3):
            _call(app, {"X-Tenant-Id": ASSERTED_TENANT})
    assert len(_assertion_warnings(caplog)) == 1


def test_flag_on_leaves_the_no_header_default_path_alone(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is scoped to the ASSERTION, not to the whole tokenless branch.

    Widening it silently to the no-header case would be a second, unannounced
    behaviour change riding on the same flag.
    """
    monkeypatch.setenv(ENV_VAR, "true")
    reset_settings_cache()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        status, body = _call(app, {})
    assert status == 200
    assert body["tenant_id"] == default_tenant_uuid()


# ---------------------------------------------------------------------------
# (c) Regression guard: a proven identity is untouched in BOTH states.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enforced", [False, True])
def test_a_licence_token_request_is_unaffected_in_both_states(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, enforced: bool
) -> None:
    """The token branch never reaches the gate — it proved its tenant.

    If turning the flag on could break an authenticated caller, nobody would
    ever turn it on, and the observation above would be permanent instead of
    temporary.
    """
    if enforced:
        monkeypatch.setenv(ENV_VAR, "true")
    reset_settings_cache()
    introspect_result = {
        "active": True,
        "tenant_id": "11111111-1111-4111-8111-111111111111",
        "subject_id": "user-uuid-1",
    }
    metrics = MagicMock()
    with (
        patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ),
    ):
        status, body = _call(app, {"X-Tenant-Id": "Bearer a-real-licence-token"})

    assert status == 200
    assert body["tenant_id"] == "11111111-1111-4111-8111-111111111111"
    metrics.record_unauthenticated_tenant_assertion.assert_not_called()


@pytest.mark.parametrize("enforced", [False, True])
def test_the_delegation_path_is_unaffected_in_both_states(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, enforced: bool
) -> None:
    """A delegating service account still re-scopes to the run's tenant.

    The delegation branch gates on the INTROSPECTED subject, which is the
    thing the tokenless branch lacks. This change must not touch it.
    """
    sa_subject = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", sa_subject)
    if enforced:
        monkeypatch.setenv(ENV_VAR, "true")
    reset_settings_cache()
    introspect_result = {
        "active": True,
        "tenant_id": "22222222-2222-4222-8222-222222222222",
        "subject_id": sa_subject,
        "byo_mode": "platform_only",
    }
    metrics = MagicMock()
    with (
        patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspect_result,
        ),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
    ):
        status, body = _call(
            app,
            {
                "X-Tenant-Id": "Bearer sa-token",
                "X-Effective-Tenant-Id": "44444444-4444-4444-8444-444444444444",
                "X-Subject-Id": "end-user-7",
            },
        )

    assert status == 200
    assert body["tenant_id"] == "44444444-4444-4444-8444-444444444444"
    metrics.record_unauthenticated_tenant_assertion.assert_not_called()


# ---------------------------------------------------------------------------
# The counter is never sampled, even though the log is.
# ---------------------------------------------------------------------------


def test_the_log_is_deduplicated_but_the_counter_is_not(app: FastAPI) -> None:
    """⛔ Rate is read from the counter, never from log volume.

    The WARNING is deduplicated per (tenant, peer, route, caller) so a path
    10+ workloads use every request cannot become the loudest thing this
    service logs. If someone ever "simplifies" by counting inside that guard,
    the rate silently becomes one-per-window and every dashboard built on it
    reads near-zero while the hole is wide open.
    """
    metrics = MagicMock()
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=metrics):
        for _ in range(3):
            _call(app, {"X-Tenant-Id": ASSERTED_TENANT})
    assert metrics.record_unauthenticated_tenant_assertion.call_count == 3


def test_a_second_distinct_peer_is_still_named(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """Dedup must not hide a NEW caller behind an already-seen one."""
    caplog.set_level(logging.WARNING, logger="src.api.auth_middleware")
    with patch.object(auth_middleware_module, "get_metrics_client", return_value=MagicMock()):
        with TestClient(app, client=("10.42.1.7", 1), raise_server_exceptions=False) as c1:
            c1.get("/api/probe", headers={"X-Tenant-Id": ASSERTED_TENANT})
        with TestClient(app, client=("10.42.9.9", 1), raise_server_exceptions=False) as c2:
            c2.get("/api/probe", headers={"X-Tenant-Id": ASSERTED_TENANT})

    peers = {
        peer
        for msg in _assertion_warnings(caplog)
        for peer in ("10.42.1.7", "10.42.9.9")
        if peer in msg
    }
    assert peers == {"10.42.1.7", "10.42.9.9"}
