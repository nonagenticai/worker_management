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

# ---------------------------------------------------------------------------
# SYNTHETIC TENANTS — the only tenant uuids this suite is allowed to name.
#
# ⛔ THE PRODUCTION TENANT IDS DO NOT BELONG IN THIS REPOSITORY (it is public),
# and they are no longer in it: every value below is fabricated in the
# ``aaaaaaaa-0000-4000-8000-`` namespace, which is reserved for nothing and
# resolves nowhere. The real values arrive through the environment
# (``tests/conftest.py`` sets the same synthetic values, so unit tests exercise
# the machinery without ever spelling a real tenant).
#
# ``tests/test_no_production_tenant_literals.py`` reds if a production uuid
# reappears anywhere under ``src/`` or ``tests/``.
# ---------------------------------------------------------------------------
SYSTEM_TENANT = "aaaaaaaa-0000-4000-8000-000000000001"  # system + default + "n8n-dev"
EXECUTION_TENANT = "aaaaaaaa-0000-4000-8000-000000000003"
KATA_TENANT = "aaaaaaaa-0000-4000-8000-000000000006"  # release tenant beside system
AA_CONVEYOR_TENANT = "bbbbbbbb-0000-4000-8000-0000000000a5"  # own 8-char prefix
V4_CONVEYOR_TENANT = "cccccccc-0000-4000-8000-0000000000a6"  # own 8-char prefix
FR_TEST_TENANT = "aaaaaaaa-0000-4000-8000-000000000099"

PLATFORM_TENANT = SYSTEM_TENANT  # the default "a platform tenant" for a test
PLATFORM_TENANTS = frozenset(
    {SYSTEM_TENANT, EXECUTION_TENANT, KATA_TENANT, AA_CONVEYOR_TENANT, V4_CONVEYOR_TENANT}
)
RELEASE_TENANTS = frozenset({SYSTEM_TENANT, KATA_TENANT})
CUSTOMER_TENANT_A = "aaaaaaaa-0000-4000-8000-0000000000aa"
CUSTOMER_TENANT_B = "aaaaaaaa-0000-4000-8000-0000000000bb"
CUSTOMER_TENANT_C = "aaaaaaaa-0000-4000-8000-0000000000a7"  # "cdmsfirst"

# The label map the tests run against (the shape the portal realm mints).
# ⚠️ The LABELS stay as the estate spells them (a label is not an id, and the
# suite reads the same with them); the UUIDs behind them are fabricated.
TEST_TENANT_UUIDS = {
    "n8n-dev": SYSTEM_TENANT,
    "george-dev": EXECUTION_TENANT,
    "fr-test": FR_TEST_TENANT,
    "cdmsfirst": CUSTOMER_TENANT_C,
}

# The gitops ``00000000-…-0001`` "system" row of the COMMERCIAL set. Fabricated,
# not a provisioned workspace, and deliberately absent from the platform set.
SYSTEM_ZERO_TENANT = "00000000-0000-0000-0000-000000000001"
COMMERCIAL_PLATFORM_TENANTS = PLATFORM_TENANTS | {SYSTEM_ZERO_TENANT}


def tenant_env() -> dict[str, str]:
    """The env values that make the service behave as the production estate does."""
    return {
        "PLATFORM_TENANT_IDS": ",".join(sorted(PLATFORM_TENANTS)),
        "RELEASE_TENANT_IDS": ",".join(sorted(RELEASE_TENANTS)),
        "TENANT_UUIDS": ",".join(f"{k}={v}" for k, v in TEST_TENANT_UUIDS.items()),
        "SYSTEM_TENANT_ID": SYSTEM_TENANT,
        "DEFAULT_TENANT_ID": SYSTEM_TENANT,
        "COMMERCIAL_PLATFORM_TENANT_IDS": ",".join(sorted(COMMERCIAL_PLATFORM_TENANTS)),
    }


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
