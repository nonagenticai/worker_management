"""A DEDICATED plaintext Prometheus listener, on its own port.

⛔ WHY THIS EXISTS AND WHY ``/metrics`` IS **NOT** ON THE API PORT.

Measured 2026-09-16 against the running pod (``worker-management-…`` in ns
``workers``): ``/metrics`` → 404, ``/api/metrics`` → 404, and the pod's own
OpenAPI listed 47 paths with **zero** metrics route. Every counter in
``agent_metrics.py`` — including
``worker_management_unauthenticated_tenant_assertion_total``, the instrument
that is supposed to make the blast radius of
``WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT`` measurable BEFORE it is
armed — existed in source and was invisible in production. A gate that cannot
be measured cannot be armed; this module closes that.

The obvious wiring — mount ``/metrics`` on the FastAPI app and relax the API
port so a non-ambient Prometheus can scrape it — is a security regression, and
``inference-router``'s ``src/observability/metrics_server.py`` carries the
measurement that ruled it out. The relax (PeerAuthentication + the
``allow-prometheus-scrape`` NetworkPolicy ``ipBlock``) is PORT-scoped, not
path-scoped: relaxing ``:8887`` would expose the whole Worker Management API —
which auto-creates workers on ``POST /workers/{worker_name}/message`` — to every
pod in the cluster CIDR. So this is the shape ``inference-router`` (8150→9150),
``coding-security`` (8693→9693) and ``w2v_orchestrator`` already use: a second,
bare WSGI server on a metrics-only port, serving the same ``prometheus_client``
REGISTRY, with the API port left STRICT and closed.

⛔ THE AUTH EXEMPTION IS STRUCTURAL, NOT A LIST ENTRY. This listener is a
``prometheus_client`` WSGI server — it is not the ASGI app, so
``AuthenticationMiddleware`` / the licence gate / the rate limiter never run on
it. ``/metrics`` was already present in ``auth_middleware.PUBLIC_ENDPOINTS`` and
in ``rate_limiting``'s exempt tuple; those stay (they cost nothing and make the
intent legible) but they are NOT what exempts this endpoint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus_client import start_http_server

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def start_metrics_server(app: FastAPI, port: int) -> None:
    """Bind the metrics-only listener and stash it on ``app.state``.

    Fail-soft on purpose: a metrics listener must never be the reason the
    service does not start. The realistic bind failure is a test harness or a
    sidecar already holding the port, and an unscraped service is strictly
    better than a dead one. The WARNING is the signal — pair it with an
    ``absent()`` alert in gitops, because a silent listener and a quiet service
    look identical in Prometheus.
    """
    try:
        httpd, _thread = start_http_server(port)
    except OSError as exc:
        logger.warning(
            "Metrics server failed to bind on port %s — this pod will export no series: %s",
            port,
            exc,
        )
        return
    app.state.metrics_httpd = httpd
    logger.info("Prometheus metrics listener started on port %s", port)


def stop_metrics_server(app: FastAPI) -> None:
    """Shut the listener down if one was started."""
    httpd = getattr(app.state, "metrics_httpd", None)
    if httpd is not None:
        httpd.shutdown()
