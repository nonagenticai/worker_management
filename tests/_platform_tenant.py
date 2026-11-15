"""Run a test as a PLATFORM tenant (owner ruling 2026-10-01).

Since the ruling, the full provider catalog is for platform tenants only
(``tenant_constants.commercial_platform_tenant_ids()``, gitops
``COMMERCIAL_PLATFORM_TENANT_IDS`` — ``tests/conftest.py`` sets the estate value);
an absent or unknown tenant fails closed to the customer rules. Since 2026-10-03
(fr#1385) the owner's commercial env keys additionally need the request to have
PROVEN the tenant (``src.auth.tenant_proof``). Tests that exercise the platform
behaviour must therefore SAY they run as a (proven) platform tenant, rather than
inherit it from an absent tenant.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Iterator

PLATFORM_TENANT = "0195f48b-d000-7000-8000-000000000001"  # n8n-dev, a platform tenant
CUSTOMER_TENANT_A = "0195f48b-d000-7000-8000-0000000000aa"
CUSTOMER_TENANT_B = "0195f48b-d000-7000-8000-0000000000bb"


@contextlib.contextmanager
def platform_tenant_context(
    tenant_id: str = PLATFORM_TENANT, *, provenance: str = "license"
) -> Iterator[None]:
    """Set the request-identity tenant (and how it was proven) ModelFactory reads.

    ``provenance="license"`` (default) models a licence-introspected request —
    the tenant is PROVEN, so a platform tenant may use the owner's keys. Pass
    ``"asserted"`` / ``"default"`` to model an unproven header.
    """
    from src.api.identity_context import _tenant_id_var, _tenant_provenance_var

    token = _tenant_id_var.set(tenant_id)
    ptoken = _tenant_provenance_var.set(provenance)
    try:
        yield
    finally:
        _tenant_provenance_var.reset(ptoken)
        _tenant_id_var.reset(token)


@contextlib.contextmanager
def caller_tenant(tenant_id: str | None, *, proven: bool = True) -> Iterator[None]:
    """Make the /providers + /chat routers see ``tenant_id`` as the caller.

    ``proven=True`` (default) also makes the request count as having PROVEN that
    tenant (a licence-introspected caller): only the proof step is stubbed, the
    gitops-set MEMBERSHIP check stays real. ``proven=False`` models an asserted
    ``X-Tenant-Id`` / delegation header — the owner's keys must then be refused.
    """
    from src.auth import tenant_proof

    def _proof(t: str) -> str | None:
        same = bool(tenant_id) and t.strip().lower() == str(tenant_id).strip().lower()
        return None if (proven and same) else "provenance_asserted"

    with (
        patch("src.api.providers._router.caller_tenant_id", return_value=tenant_id),
        patch.object(tenant_proof, "tenant_proof_refusal", side_effect=_proof),
    ):
        yield


@contextlib.contextmanager
def proven_tenant(tenant_id: str | None, *, proven: bool = True) -> Iterator[None]:
    """Make THIS request count as having PROVEN ``tenant_id`` (or, ``proven=False``, not).

    Stubs only the proof step (``tenant_proof.tenant_proof_refusal``) — the
    gitops-set MEMBERSHIP check stays real — for code paths that read the tenant
    from their own context (``/workers/{name}/message``, the worker build,
    graph nodes) rather than from the providers router's ``caller_tenant_id``.
    """
    from src.auth import tenant_proof

    def _proof(t: str) -> str | None:
        same = bool(tenant_id) and t.strip().lower() == str(tenant_id).strip().lower()
        return None if (proven and same) else "provenance_asserted"

    with patch.object(tenant_proof, "tenant_proof_refusal", side_effect=_proof):
        yield
