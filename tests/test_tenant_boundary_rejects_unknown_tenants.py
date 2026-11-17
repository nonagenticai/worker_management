"""The tokenless trusted-network branch resolves tenants instead of trusting them.

⛔ WHAT THIS PINS, AND WHY IT MATTERS MORE THAN IT LOOKS

Until 2026-08-25 ``auth_middleware.dispatch`` did, on the tokenless
trusted-network path:

    tenant_id = x_tenant_id.strip() if x_tenant_id else default_tenant_uuid()

— i.e. whatever a caller inside the pod network put in ``X-Tenant-Id`` was
adopted as the request's tenant and written to the database verbatim. The
default ``TRUSTED_CIDRS`` covers ``10.0.0.0/8``, which is the whole cluster
network, so this was reachable from any pod.

That is how, measured against production on 2026-08-25, ``worker_states`` came
to hold **93,842** rows whose ``tenant_id`` is not a UUID at all, across 16
distinct literals — ``n8n-dev``, ``sales``, ``auto``, ``__system_workflow__``,
``smoke-plan-7``, ``proof-tier-diff-20260624`` and bare snowflake integers.

``tenant_id`` is now a real Postgres ``uuid`` column (migration
``024_tenant_id_is_a_uuid.sql``), so an unresolved label would no longer be
stored — it would blow up every statement of the request with ``invalid input
syntax for type uuid``, i.e. a 500 blaming the database for a value the caller
chose. The boundary therefore has to reject it, and reject it as a 400.

The two halves that must BOTH hold:
  * a canonical UUID works — and is the ONLY thing that does;
  * anything else is refused, and specifically is NOT silently replaced by
    ``default_tenant_uuid()``. A silent fallback would keep the pipe open while
    hiding it, and would quietly serve one tenant another tenant's data on a
    typo.

⚠️ AMENDED 2026-09-16 — the first half used to read "a NAME this deployment
knows still works — resolved, never stored". Names are now refused too. A name
is not merely unstored, it is UNDECIDABLE: ``n8n-dev`` resolves through
``tenant_uuids()`` to the seed tenant ``n8n-dev-portal-realm`` while the licences
registry has ``n8n-dev`` as a different live tenant, so "resolved" was resolving
to the wrong one. See ``tests/test_tokenless_tenant_name_is_refused.py``.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.auth_middleware import AuthenticationMiddleware
from src.api.tenant_constants import default_tenant_uuid, tenant_uuids


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, str]:
        return {"tenant_id": request.state.tenant_id}

    return application


def _get(app: FastAPI, tenant_header: str | None) -> tuple[int, dict]:
    headers = {"X-Tenant-Id": tenant_header} if tenant_header is not None else {}
    # A real peer inside TRUSTED_CIDRS — the tokenless branch is a pure
    # network-membership decision, so this is how a real caller reaches it.
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        resp = client.get("/api/probe", headers=headers)
    body = (
        resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    )
    return resp.status_code, body


@pytest.mark.parametrize("name", sorted(tenant_uuids()))
def test_a_known_tenant_name_is_refused(app: FastAPI, name: str) -> None:
    """⚠️ INVERTED 2026-09-16. This test used to assert a NAME is ACCEPTED.

    It was not a wrong test for the contract it was written against — the 2026-08-25
    change it pins deliberately kept names working while closing the verbatim-adoption
    hole. It became wrong when the name half was measured: ``n8n-dev`` resolves here to
    ``aaaaaaaa-…0001``, which the licences registry calls ``n8n-dev-portal-realm`` (the
    SEED tenant), while the tenant actually named ``n8n-dev`` is
    ``bbbbbbbb-0000-4000-8000-0000000000a5``. ``fr-test`` collides the same way.

    So the assertion below is not a relaxation of this file's thesis, it is its
    completion: a value that resolves to a *different real tenant* than the caller
    named is the same silent-misattribution failure this file exists to prevent, and
    the old assertion was the one place it was written down as intended behaviour.

    See ``tests/test_tokenless_tenant_name_is_refused.py`` for the full reasoning and
    the measured registry rows.
    """
    status, body = _get(app, name)
    assert status == 400
    # The load-bearing half: refusing must not hand back a tenant anyway.
    assert body.get("tenant_id") is None
    assert body.get("tenant_id") != tenant_uuids()[name]


def test_a_uuid_passes_through_case_insensitively(app: FastAPI) -> None:
    status, body = _get(app, "AAAAAAAA-0000-4000-8000-000000000003")
    assert status == 200
    assert body["tenant_id"] == "aaaaaaaa-0000-4000-8000-000000000003"


def test_absent_header_uses_the_default_tenant(app: FastAPI) -> None:
    """Unchanged behaviour, asserted so the change above is scoped to the header."""
    status, body = _get(app, None)
    assert status == 200
    assert body["tenant_id"] == default_tenant_uuid()


@pytest.mark.parametrize(
    "bogus",
    [
        # The real literals measured in production, i.e. exactly what used to
        # get through and become rows.
        "sales",
        "auto",
        "__system_workflow__",
        "smoke-plan-7",
        "1769980520185202330",
        # gerard-dev was REMOVED from the portal realm 2026-08-12; naming a
        # retired tenant must not grant anything.
        "gerard-dev",
        # Shapes uuid.UUID() would otherwise accept but that round-trip to a
        # different string than the caller sent.
        "urn:uuid:aaaaaaaa-0000-4000-8000-000000000003",
        "{aaaaaaaa-0000-4000-8000-000000000003}",
        "aaaaaaaad00070008000000000000003",
        # Injection-ish and empty shapes.
        "' OR 1=1 --",
        "..",
        "   ",
    ],
)
def test_an_unresolvable_tenant_is_a_clean_400(app: FastAPI, bogus: str) -> None:
    status, body = _get(app, bogus)
    assert status == 400, f"{bogus!r} was admitted with {status}"
    assert body["error"]["code"] == "INVALID_TENANT"


def test_a_rejection_names_the_header_and_the_value_in_the_log(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """A refusal nobody can attribute is an outage waiting for a support ticket.

    This is a behaviour CHANGE: these values were adopted verbatim before
    2026-08-25, so any caller still sending one starts getting 400s the moment
    this deploys. The operator needs the header and the value, or the first real
    diagnosis happens somewhere else entirely.
    """
    with caplog.at_level(logging.WARNING, logger="src.api.auth_middleware"):
        status, _ = _get(app, "typo-tenant")

    assert status == 400
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "the rejection produced no WARNING at all"
    message = warnings[-1].getMessage()
    assert "typo-tenant" in message, message
    assert "X-Tenant-Id" in message, message


def test_the_delegation_header_rejection_is_also_logged(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """Both mouths of the pipe, not just the one I closed first."""
    with caplog.at_level(logging.WARNING, logger="src.api.auth_middleware"):
        _get(app, "sales")
    assert any("sales" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("bogus", ["sales", "auto", "typo-tenant"])
def test_a_rejected_tenant_does_not_fall_back_to_the_default(app: FastAPI, bogus: str) -> None:
    """The non-vacuity control for the test above.

    A 400 is only the right outcome if the alternative was NOT "quietly serve
    the default tenant". If this ever starts returning 200 with
    ``default_tenant_uuid()``, a caller's typo silently reads and writes another
    tenant's data — the exact failure the 400 exists to prevent, and one that
    a status-code-only assertion would not catch.
    """
    status, body = _get(app, bogus)
    assert status != 200
    assert body.get("tenant_id") != default_tenant_uuid()
