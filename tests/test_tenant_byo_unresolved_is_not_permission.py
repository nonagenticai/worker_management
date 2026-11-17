"""An unresolvable BYO posture is not the same answer as "platform inference is allowed".

``_lookup_tenant_byo_mode`` asks the licences service whether a tenant must
bring its own LLM key. It used to treat EVERY non-200 — 404, 401, 5xx, timeout,
connection error — as ``local_allowed``, the most permissive value in the enum,
which routes the run onto owner GPU with nothing billed to anyone. "I could not
ask" and "licences says yes" were literally the same return value.

⛔ THE HAZARD IS CROSS-SERVICE, which is why this suite exists on THIS side.
licences answers the ``?app=`` branch 200 with a FABRICATED ``byo_required``
when its own DB lookup faults, and its new ``on_unresolved`` parameter defaults
to that same fail-closed behaviour — chosen deliberately BECAUSE this caller
fails open on anything else, so defaulting the route to 503 would have turned a
fail-closed guess over there into an ALLOW over here. These tests pin the other
half of that argument: this service now decides the unresolved case explicitly,
so the licences default can stand on its own merits.

Each test drives ONE outcome of the ladder in ``_unresolved_byo_mode`` and
asserts the decision AND the counter together. Either alone is insufficient: an
uncounted decision is the invisible fail-open this fix removes, and a counted
one whose value is wrong is worse than no fix at all.

The label values are imported from ``agent_metrics`` rather than retyped, so a
rename that desynchronises the emitter from the pre-materialised label set
breaks here rather than silently creating an eighth, unmaterialised series.
"""

import logging
import os
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import httpx2 as httpx
import pytest

# Must precede the import of the module under test (see tests/test_agent_metrics.py).
os.environ["OTEL_SDK_DISABLED"] = "true"

from src.api import auth_middleware
from src.telemetry.agent_metrics import (
    TENANT_BYO_LOOKUP_FAIL_CLOSED,
    TENANT_BYO_LOOKUP_FAILURE_NON_200,
    TENANT_BYO_LOOKUP_FAILURE_NONE,
    TENANT_BYO_LOOKUP_FAILURE_TRANSPORT,
    TENANT_BYO_LOOKUP_FAILURES,
    TENANT_BYO_LOOKUP_GRACE_OPEN,
    TENANT_BYO_LOOKUP_OUTCOMES,
    TENANT_BYO_LOOKUP_PAIRS,
    TENANT_BYO_LOOKUP_RESOLVED,
    TENANT_BYO_LOOKUP_STALE,
)

TENANT = "019ffe9f-0000-7000-8000-000000000001"


class _FakeResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _patch_client(monkeypatch, *, status_code=200, payload=None, raise_exc=None):
    @asynccontextmanager
    async def _fake_client(*_a, **_k):
        class _C:
            async def get(self, _url, params=None):
                if raise_exc:
                    raise raise_exc
                return _FakeResp(status_code, payload or {})

        yield _C()

    monkeypatch.setattr(auth_middleware.httpx, "AsyncClient", _fake_client)


@pytest.fixture(autouse=True)
def _clear_caches():
    """All three, before AND after. See the same fixture in the sibling suite."""
    caches = (
        auth_middleware._tenant_byo_cache,
        auth_middleware._tenant_byo_stale_cache,
        auth_middleware._tenant_byo_unresolved_since,
    )
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


@pytest.fixture
def metrics(monkeypatch):
    """Intercept the metrics client at the lookup's own call site."""
    client = MagicMock()
    monkeypatch.setattr(auth_middleware, "get_metrics_client", lambda: client)
    return client


def _booked(metrics) -> list[tuple[str, str]]:
    """Every (outcome, failure) this lookup booked, in order."""
    return [
        (call.args[0], call.kwargs["failure"])
        for call in metrics.record_tenant_byo_mode_lookup.call_args_list
    ]


def _sole(metrics) -> tuple[str, str]:
    booked = _booked(metrics)
    assert len(booked) == 1, f"expected exactly one booked outcome, got {booked}"
    return booked[0]


# ---------------------------------------------------------------------------
# The closed label sets — the control for every assertion below
# ---------------------------------------------------------------------------


def test_the_label_values_under_test_are_the_prematerialised_set():
    """CONTROL. If this fails, the assertions below name series that do not exist.

    A label with no pre-materialised zero reports an absence, and absence reads
    identically to health — which is the failure mode this whole counter exists
    to avoid.
    """
    assert set(TENANT_BYO_LOOKUP_OUTCOMES) == {
        TENANT_BYO_LOOKUP_RESOLVED,
        TENANT_BYO_LOOKUP_STALE,
        TENANT_BYO_LOOKUP_GRACE_OPEN,
        TENANT_BYO_LOOKUP_FAIL_CLOSED,
    }
    assert set(TENANT_BYO_LOOKUP_FAILURES) == {
        TENANT_BYO_LOOKUP_FAILURE_NONE,
        TENANT_BYO_LOOKUP_FAILURE_NON_200,
        TENANT_BYO_LOOKUP_FAILURE_TRANSPORT,
    }
    # Every pair this suite asserts must be one of the seeded ones.
    for pair in (
        (TENANT_BYO_LOOKUP_RESOLVED, TENANT_BYO_LOOKUP_FAILURE_NONE),
        (TENANT_BYO_LOOKUP_STALE, TENANT_BYO_LOOKUP_FAILURE_NON_200),
        (TENANT_BYO_LOOKUP_GRACE_OPEN, TENANT_BYO_LOOKUP_FAILURE_TRANSPORT),
        (TENANT_BYO_LOOKUP_FAIL_CLOSED, TENANT_BYO_LOOKUP_FAILURE_NON_200),
    ):
        assert pair in TENANT_BYO_LOOKUP_PAIRS


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL — the healthy path must be untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_healthy_200_is_unchanged_and_books_the_denominator(monkeypatch, metrics, caplog):
    """⭐ THE CONTROL. Everything else here is about failure; this is the 90%.

    A 200 must still return licences' own value, still populate the 5-minute
    positive cache, and still log nothing at WARNING — 71 of the 79 measured
    production lookups were this path, and a warning on each would bury the
    eight that matter. The only addition is the ``resolved`` series, which is
    the DENOMINATOR: without it "3 unresolved" is a number with no scale.
    """
    _patch_client(monkeypatch, payload={"byo_mode": "byo_required"})
    with caplog.at_level(logging.WARNING, logger=auth_middleware.logger.name):
        assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "byo_required"

    assert _sole(metrics) == (TENANT_BYO_LOOKUP_RESOLVED, TENANT_BYO_LOOKUP_FAILURE_NONE)
    warned = [r for r in caplog.records if r.name == auth_middleware.logger.name]
    assert not warned, f"a healthy lookup warned: {[r.getMessage() for r in warned]}"
    assert auth_middleware._tenant_byo_cache[(TENANT, None)] == "byo_required"


@pytest.mark.asyncio
async def test_a_cache_hit_books_nothing(monkeypatch, metrics):
    """The cache short-circuit books nothing: it is not a lookup, it is a hit.

    Counting it would inflate the denominator against which the unresolved rate
    is read, making the fail-open rate look smaller the more caching helps.
    """
    _patch_client(monkeypatch, payload={"byo_mode": "platform_only"})
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT, "coding") == "platform_only"
    _patch_client(monkeypatch, raise_exc=RuntimeError("must not be called"))
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT, "coding") == "platform_only"
    assert _booked(metrics) == [(TENANT_BYO_LOOKUP_RESOLVED, TENANT_BYO_LOOKUP_FAILURE_NONE)]


# ---------------------------------------------------------------------------
# One test per unresolved outcome: 5xx / timeout / connection error / 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_5xx_is_unresolved_not_permission(monkeypatch, metrics):
    """A 503 from licences is an infrastructure fault, and it is labelled as a non-200.

    ⚠️ This is the exact shape the licences ``on_unresolved`` parameter can opt
    into. It must never be read here as "this tenant may use platform
    inference"; the cold tenant below still proceeds, but on the counted,
    bounded grace arm rather than on a fabricated entitlement.
    """
    _patch_client(monkeypatch, status_code=503, payload={})
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "local_allowed"
    assert _sole(metrics) == (TENANT_BYO_LOOKUP_GRACE_OPEN, TENANT_BYO_LOOKUP_FAILURE_NON_200)


@pytest.mark.asyncio
async def test_a_timeout_is_unresolved_and_labelled_transport(monkeypatch, metrics):
    """A timeout never reaches the status check at all — different arm, same ladder."""
    _patch_client(monkeypatch, raise_exc=httpx.TimeoutException("read timed out"))
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "local_allowed"
    assert _sole(metrics) == (TENANT_BYO_LOOKUP_GRACE_OPEN, TENANT_BYO_LOOKUP_FAILURE_TRANSPORT)


@pytest.mark.asyncio
async def test_a_connection_error_is_unresolved_and_labelled_transport(monkeypatch, metrics):
    """licences unreachable (pod gone, DNS, NetworkPolicy) — transport, not non_200.

    The two failure values are not decoration: a transport fault heals on its
    own and the remedy is to fix licences, while the measured non-200 population
    is a caller sending a selector that is not a tenant, which never heals and
    whose remedy is to fix the caller. One series could not tell them apart.
    """
    _patch_client(monkeypatch, raise_exc=httpx.ConnectError("connection refused"))
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "local_allowed"
    assert _sole(metrics) == (TENANT_BYO_LOOKUP_GRACE_OPEN, TENANT_BYO_LOOKUP_FAILURE_TRANSPORT)


@pytest.mark.asyncio
async def test_a_404_is_unresolved_and_labelled_non_200(monkeypatch, metrics, caplog):
    """The measured case: 8 of 79 prod lookups, ~10% of S2S runs unenforced.

    The tenant id stays in the log because a COUNT cannot answer "which id
    failed", and that id is the whole investigation.
    """
    _patch_client(monkeypatch, status_code=404, payload={"detail": "Tenant not found"})
    with caplog.at_level(logging.WARNING, logger=auth_middleware.logger.name):
        assert (
            await auth_middleware._lookup_tenant_byo_mode("1769980520185202330", "coding")
            == "local_allowed"
        )
    assert _sole(metrics) == (TENANT_BYO_LOOKUP_GRACE_OPEN, TENANT_BYO_LOOKUP_FAILURE_NON_200)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "1769980520185202330" in joined
    assert "404" in joined


# ---------------------------------------------------------------------------
# ⭐ THE TEST THAT FAILS IF THE UNRESOLVED CASE SILENTLY REVERTS TO PROCEEDING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_byo_required_tenant_is_not_downgraded_by_a_licences_fault(monkeypatch, metrics):
    """⭐ THE REGRESSION GUARD. Delete the stale arm and this test goes red.

    Before the fix, a tenant licences had JUST told us was ``byo_required``
    became ``local_allowed`` the moment the next lookup hit a 5xx — the run then
    rode owner GPU, billed to nobody, and the only trace was a log line. The
    positive cache is cleared here to simulate its 5-minute expiry while the
    15-minute last-known-good window is still warm, which is the overwhelmingly
    common shape of a licences blip.

    ⛔ ``local_allowed`` here is not a "slightly weaker" answer. It is the most
    permissive value in the enum and it is the ONLY value that silently grants
    platform inference, so a revert to it is the whole defect returning.
    """
    _patch_client(monkeypatch, payload={"byo_mode": "byo_required"})
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "byo_required"

    # The 5-minute positive cache expires; the last-known-good window has not.
    auth_middleware._tenant_byo_cache.clear()
    _patch_client(monkeypatch, status_code=500, payload={})

    mode = await auth_middleware._lookup_tenant_byo_mode(TENANT)
    assert mode == "byo_required", (
        "a licences fault downgraded a byo_required tenant to local_allowed — "
        "the run would proceed on platform inference, unbilled and unenforced"
    )
    assert _booked(metrics)[-1] == (TENANT_BYO_LOOKUP_STALE, TENANT_BYO_LOOKUP_FAILURE_NON_200)


@pytest.mark.asyncio
async def test_a_local_allowed_tenant_is_not_denied_by_a_licences_fault(monkeypatch, metrics):
    """The other direction, and it matters just as much.

    Last-known-good is the tenant's OWN posture, not a policy. A permissive
    tenant must keep running through a blip; if this test could only pass by
    denying, the fix would have traded a silent entitlement leak for an outage.
    """
    _patch_client(monkeypatch, payload={"byo_mode": "local_allowed"})
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "local_allowed"
    auth_middleware._tenant_byo_cache.clear()
    _patch_client(monkeypatch, raise_exc=httpx.ConnectError("down"))
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "local_allowed"
    assert _booked(metrics)[-1] == (TENANT_BYO_LOOKUP_STALE, TENANT_BYO_LOOKUP_FAILURE_TRANSPORT)


@pytest.mark.asyncio
async def test_an_unresolved_decision_is_never_written_to_the_five_minute_cache(
    monkeypatch, metrics
):
    """⛔ A fabricated posture must not outlive the outage that produced it.

    This is exactly the cost that makes the licences side's own fabricated
    ``byo_required`` expensive: worker_management caches for 300s, so one blip
    became up to five minutes of hard denial. Repeating that here would multiply
    it. Only an authoritative 200 may populate the positive cache.
    """
    _patch_client(monkeypatch, status_code=502, payload={})
    await auth_middleware._lookup_tenant_byo_mode(TENANT)
    assert (TENANT, None) not in auth_middleware._tenant_byo_cache

    monkeypatch.setattr(auth_middleware, "_TENANT_BYO_UNRESOLVED_GRACE", 0.0)
    await auth_middleware._lookup_tenant_byo_mode(TENANT)
    assert (TENANT, None) not in auth_middleware._tenant_byo_cache


# ---------------------------------------------------------------------------
# The bounded grace window, and its terminal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_cold_tenant_past_the_grace_window_fails_closed(monkeypatch, metrics, caplog):
    """The terminal arm: unresolved, nothing known, and no longer a blip.

    A grace of 0 is the documented "deny as soon as there is no last-known-good"
    setting, and it is how this arm is reached deterministically without
    sleeping. What it costs is stated at the constant: a 402 whose text talks
    about configuring a provider, which is not why the run was refused. The log
    line says so explicitly, which is the only reason that is tolerable.
    """
    monkeypatch.setattr(auth_middleware, "_TENANT_BYO_UNRESOLVED_GRACE", 0.0)
    _patch_client(monkeypatch, status_code=404, payload={})
    with caplog.at_level(logging.WARNING, logger=auth_middleware.logger.name):
        assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "byo_required"
    assert _sole(metrics) == (TENANT_BYO_LOOKUP_FAIL_CLOSED, TENANT_BYO_LOOKUP_FAILURE_NON_200)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "FAILING CLOSED" in joined


@pytest.mark.asyncio
async def test_a_permanently_unresolvable_selector_settles_instead_of_flapping(
    monkeypatch, metrics
):
    """Re-pinning the first sighting is what stops a 402/200 oscillation.

    Without the re-pin the since-entry evicts at its own TTL and the next
    failure opens a FRESH window, so a selector that will never resolve would
    alternate between denied and allowed forever — the worst of both decisions,
    and impossible to alert on. Same fix, same reason, as the token_not_found
    grace on the introspection path.
    """
    monkeypatch.setattr(auth_middleware, "_TENANT_BYO_UNRESOLVED_GRACE", 0.0)
    _patch_client(monkeypatch, status_code=404, payload={})
    for _ in range(5):
        auth_middleware._tenant_byo_cache.clear()
        assert await auth_middleware._lookup_tenant_byo_mode("not-a-tenant") == "byo_required"
    assert {outcome for outcome, _ in _booked(metrics)} == {TENANT_BYO_LOOKUP_FAIL_CLOSED}


@pytest.mark.asyncio
async def test_a_recovery_clears_the_window_so_it_measures_a_continuous_outage(
    monkeypatch, metrics
):
    """The window must measure a CONTINUOUS run of failures, not a lifetime total.

    A tenant that fails, recovers, and fails again hours later is having two
    blips, not one long outage. If the first sighting survived the recovery, the
    second blip would be denied on a timer the recovery should have reset.
    """
    _patch_client(monkeypatch, status_code=503, payload={})
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "local_allowed"
    assert (TENANT, None) in auth_middleware._tenant_byo_unresolved_since

    _patch_client(monkeypatch, payload={"byo_mode": "local_allowed"})
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "local_allowed"
    assert (TENANT, None) not in auth_middleware._tenant_byo_unresolved_since


# ---------------------------------------------------------------------------
# No branch may decide silently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_lookup_books_exactly_one_series(monkeypatch, metrics):
    """A decision this counter cannot see is the defect being fixed, restored.

    ⛔ Exactly ONE, not "at least one": a branch that books twice makes the
    unresolved RATE — the number the whole instrument exists to produce, and the
    number anyone tightening the grace window will read — wrong in the safe-
    looking direction.
    """
    cases = (
        {"payload": {"byo_mode": "byo_required"}},
        {"status_code": 404, "payload": {}},
        {"status_code": 500, "payload": {}},
        {"raise_exc": httpx.TimeoutException("t")},
        {"raise_exc": httpx.ConnectError("c")},
    )
    for index, case in enumerate(cases):
        metrics.record_tenant_byo_mode_lookup.reset_mock()
        for cache in (
            auth_middleware._tenant_byo_cache,
            auth_middleware._tenant_byo_stale_cache,
            auth_middleware._tenant_byo_unresolved_since,
        ):
            cache.clear()
        _patch_client(monkeypatch, **case)
        await auth_middleware._lookup_tenant_byo_mode(f"tenant-{index}")
        outcome, failure = _sole(metrics)
        assert outcome in TENANT_BYO_LOOKUP_OUTCOMES
        assert failure in TENANT_BYO_LOOKUP_FAILURES


@pytest.mark.asyncio
async def test_a_broken_metrics_backend_cannot_change_the_decision(monkeypatch):
    """⛔ The instrument must never be able to deny — or allow — a run.

    Ordering at the call site is what guarantees it: the posture is chosen and
    returned independently of the counter. A metrics client that raises on every
    call must leave both the permissive and the restrictive arm exactly as they
    were.
    """
    exploding = MagicMock()
    exploding.record_tenant_byo_mode_lookup.side_effect = RuntimeError("otlp down")
    monkeypatch.setattr(auth_middleware, "get_metrics_client", lambda: exploding)

    _patch_client(monkeypatch, payload={"byo_mode": "platform_only"})
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "platform_only"

    auth_middleware._tenant_byo_cache.clear()
    auth_middleware._tenant_byo_stale_cache.clear()
    monkeypatch.setattr(auth_middleware, "_TENANT_BYO_UNRESOLVED_GRACE", 0.0)
    _patch_client(monkeypatch, status_code=503, payload={})
    assert await auth_middleware._lookup_tenant_byo_mode(TENANT) == "byo_required"


# ---------------------------------------------------------------------------
# The pre-materialised zeros
# ---------------------------------------------------------------------------


def _client_with_fake_meter():
    from src.telemetry.agent_metrics import AgentMetricsClient

    client = AgentMetricsClient.__new__(AgentMetricsClient)
    client.enabled = True
    client._counters = {}
    client._histograms = {}
    client._up_down_counters = {}
    client.meter = MagicMock()
    client.meter.create_counter.side_effect = lambda **_kw: MagicMock()
    client.meter.create_histogram.side_effect = lambda **_kw: MagicMock()
    client.meter.create_up_down_counter.side_effect = lambda **_kw: MagicMock()
    client._setup_metrics()
    return client


def test_the_series_exist_at_zero_before_any_lookup_happens():
    """⭐ An ABSENT series is indistinguishable from a broken exporter.

    This estate has several counters that read "healthy" only because they were
    never born. Every reachable (outcome, failure) pair is materialised with
    ``add(0, ...)`` at setup, so "no fail-opens" reads differently from "nothing
    is scraping this".
    """
    counter = _client_with_fake_meter()._counters["tenant_byo_mode_lookup"]
    zeroed = {
        (call.args[1]["outcome"], call.args[1]["failure"])
        for call in counter.add.call_args_list
        if call.args[0] == 0
    }
    assert zeroed == set(TENANT_BYO_LOOKUP_PAIRS), (
        f"missing pre-materialised series: {sorted(set(TENANT_BYO_LOOKUP_PAIRS) - zeroed)}"
    )
    # ...and nothing has been COUNTED yet: startup must not fabricate a finding.
    assert not [call for call in counter.add.call_args_list if call.args[0] != 0]


def test_only_the_reachable_pairs_are_seeded():
    """⛔ Seeding the unreachable 5 would publish answers to questions never asked.

    ``resolved`` can only carry ``failure=none`` and the three unresolved
    outcomes can never carry it. A seeded zero on ``(fail_closed, none)`` would
    look like a measured "we have never denied for no reason", which is not a
    measurement — it is a series that cannot move.
    """
    assert len(TENANT_BYO_LOOKUP_PAIRS) == 7
    unreachable = {
        # ``resolved`` means nothing failed, so it can carry no failure kind...
        (TENANT_BYO_LOOKUP_RESOLVED, TENANT_BYO_LOOKUP_FAILURE_NON_200),
        (TENANT_BYO_LOOKUP_RESOLVED, TENANT_BYO_LOOKUP_FAILURE_TRANSPORT),
        # ...and the three unresolved outcomes are only ever reached BECAUSE
        # something failed, so none of them can carry ``none``.
        (TENANT_BYO_LOOKUP_STALE, TENANT_BYO_LOOKUP_FAILURE_NONE),
        (TENANT_BYO_LOOKUP_GRACE_OPEN, TENANT_BYO_LOOKUP_FAILURE_NONE),
        (TENANT_BYO_LOOKUP_FAIL_CLOSED, TENANT_BYO_LOOKUP_FAILURE_NONE),
    }
    assert len(unreachable) + len(TENANT_BYO_LOOKUP_PAIRS) == 12, "the cross product is 4x3"
    assert not unreachable & set(TENANT_BYO_LOOKUP_PAIRS)
