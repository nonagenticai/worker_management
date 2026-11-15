"""The Prometheus scrape endpoint exists, is unauthenticated, and is not empty.

⛔ WHY THIS FILE EXISTS. Measured 2026-09-16 against the running pod: ``/metrics``
→ 404, ``/api/metrics`` → 404, and the pod's own OpenAPI listed 47 paths with
zero metrics route. Every counter in ``agent_metrics.py`` — including
``worker_management_unauthenticated_tenant_assertion_total``, the instrument
whose reading decides whether ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT``
may be armed — existed in source and produced nothing observable.

The two properties under test are the two that were actually broken, and each
maps to a way the fix could silently regress:

1. The listener answers 200 with no credential. It is a bare WSGI server on its
   own port, so the licence gate cannot reach it — but if someone ever "fixes"
   this by moving /metrics onto the FastAPI app, the auth middleware WILL start
   running on it and Prometheus will get 401s. This test fails in that world.
2. The arming counter's family is PRESENT at zero. An OTel counter emits no
   series until its first ``add()``, so without pre-materialisation "absent"
   and "zero" are the same scrape — and "it is zero" is precisely the reading
   that would arm the flag.
"""

from __future__ import annotations

import os
import socket
import urllib.request

import pytest
from fastapi import FastAPI

ARMING_COUNTER = "worker_management_unauthenticated_tenant_assertion_total"


def _free_port() -> int:
    """A port the kernel just confirmed was bindable."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def metrics_url() -> str:
    """Start the real listener on an ephemeral port and return its URL.

    Uses the production ``start_metrics_server`` rather than a hand-rolled
    server: the point is to exercise the wiring that ships, including the fact
    that it serves prometheus_client's DEFAULT registry — the same registry the
    ``PrometheusMetricReader`` in ``agent_metrics`` attaches itself to.
    """
    import src.telemetry.agent_metrics as agent_metrics_module
    from src.telemetry.metrics_server import start_metrics_server, stop_metrics_server

    # ⚠️ tests/test_agent_metrics.py sets OTEL_SDK_DISABLED=true at MODULE
    # IMPORT time (line 13), i.e. during collection, before any test runs. So in
    # a multi-file run this process inherits "telemetry off" and the singleton
    # would be a disabled client with no instruments — the endpoint would serve
    # only python_* runtime metrics and this file would fail for a reason that
    # has nothing to do with the code under test. Pin the env to the shipped
    # default and build a FRESH client, so what is asserted is production
    # behaviour rather than collection order.
    previous = os.environ.pop("OTEL_SDK_DISABLED", None)
    previous_client = agent_metrics_module._metrics_client
    agent_metrics_module._metrics_client = None

    # Force the MeterProvider (and with it the reader registration + the
    # pre-materialised zero series) to exist before the first scrape, exactly as
    # the app's lifespan does.
    agent_metrics_module.get_metrics_client()

    app = FastAPI()
    port = _free_port()
    start_metrics_server(app, port)
    assert getattr(app.state, "metrics_httpd", None) is not None, (
        "start_metrics_server did not bind — it is fail-soft, so a bind failure "
        "is a WARNING and no listener, which is exactly the invisible-in-prod "
        "state this endpoint exists to end."
    )
    yield f"http://127.0.0.1:{port}/metrics"

    stop_metrics_server(app)
    agent_metrics_module._metrics_client = previous_client
    if previous is not None:
        os.environ["OTEL_SDK_DISABLED"] = previous


def test_metrics_endpoint_returns_200_without_auth(metrics_url: str) -> None:
    """A scrape with NO Authorization header and NO licence token gets 200.

    Prometheus scrapes anonymously. If this ever needs a credential the target
    goes down, and a down target looks identical to a service emitting nothing —
    which is the failure this whole change is undoing.
    """
    with urllib.request.urlopen(metrics_url, timeout=5) as resp:  # noqa: S310
        assert resp.status == 200
        body = resp.read().decode()

    assert body, "scrape returned 200 with an empty body"
    # Sanity: it is really Prometheus text exposition, not an error page that
    # happened to be 200.
    assert "# HELP" in body or "# TYPE" in body


def test_arming_counter_family_is_present_at_zero(metrics_url: str) -> None:
    """The arming counter is on the scrape BEFORE it has ever been incremented.

    ⛔ THIS IS THE LOAD-BEARING ASSERTION. Without the pre-materialised zero
    series, this counter is absent from the scrape until the first tokenless
    tenant assertion — so an operator reading it to decide whether to arm
    ``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT`` cannot distinguish "no
    caller does this, safe to arm" from "the instrument was never deployed".
    """
    with urllib.request.urlopen(metrics_url, timeout=5) as resp:  # noqa: S310
        body = resp.read().decode()

    assert ARMING_COUNTER in body, (
        f"{ARMING_COUNTER} is absent from the scrape. An absent counter and a "
        f"zero counter are not the same fact, and only one of them is safe to "
        f"arm a licence gate on.\n--- scrape ---\n{body[:4000]}"
    )

    # Both enforced legs are booked, and both are booked at ZERO — a test that
    # only checked presence would still pass if a stray increment leaked in.
    samples = [
        line
        for line in body.splitlines()
        if line.startswith(ARMING_COUNTER) and not line.startswith("#")
    ]
    assert samples, f"{ARMING_COUNTER} has HELP/TYPE but no series:\n{body[:2000]}"

    for leg in ('enforced="true"', 'enforced="false"'):
        matching = [s for s in samples if leg in s]
        assert matching, f"no pre-materialised series for {leg}; got {samples}"
        for sample in matching:
            value = float(sample.rsplit(" ", 1)[1])
            assert value == 0.0, f"expected a zero series, got: {sample}"
