"""Managed-BYO (``owner_provided``) rows are owner-locked.

Owner ruling 2026-10-01: no customer tenant may use the owner's commercial
keys. The one exception is managed BYO, where the owner sets a key on ONE
tenant's ``tenant_llm_config`` / ``tenant_modality_config`` row and marks it
``owner_provided``. That key is the OWNER's credential, so where it is sent
(``base_url``), which vendor receives it (``provider``), the key itself and the
``owner_provided`` mark are the owner's decisions, never the tenant's. Before
this, ``PUT /tenant-config/{llm,modality/<m>}`` let the tenant change any of
them on a managed row, so a tenant could send the owner's key to an endpoint
or vendor of its choice. It could also clear the mark, which unlocks the row,
or set the mark on its own row.

Who the owner is: a caller carrying :data:`MANAGED_BYO_ADMIN_ROLE` in the
caller-roles ContextVar. ``AuthenticationMiddleware`` fills that ContextVar ONLY
from a signature-verified, azp-allowlisted ``X-Caller-Authorization``
secure-apps SA JWT (``src/auth/keycloak_roles.py``), never from a request
header. The tenant's own licence token, the tokenless trusted-network branch
(any pod asserting ``X-Tenant-Id``) and a delegating service account do not
carry it. Fail closed: until the role is granted to an owner SA client and that
client is in ``CALLER_ALLOWED_CLIENTS``, nobody can create, change or unmark a
managed row through these routes.

A tenant may still change what is not the owner's to decide: ``model``,
``params``, and for LLM rows ``temperature`` and ``max_tokens``. It may also
DELETE the row, which removes the owner's key rather than redirecting it.
"""

from __future__ import annotations

from typing import Any

from src.api.error_handlers import APIError
from src.api.identity_context import get_caller_roles_from_context

MANAGED_BYO_ADMIN_ROLE = "managed-byo-admin"
MANAGED_BYO_ROW_LOCKED = "managed_byo_row_locked"


def is_managed_byo_owner_caller() -> bool:
    """True when the request carries the verified owner role (see module docstring)."""
    return MANAGED_BYO_ADMIN_ROLE in get_caller_roles_from_context()


def _same_url(a: str | None, b: str | None) -> bool:
    return (a or "").strip().rstrip("/") == (b or "").strip().rstrip("/")


def managed_row_changes(
    existing: dict[str, Any] | None,
    *,
    provider: str,
    base_url_in_body: str | None,
    key_in_body: bool,
    owner_provided_final: bool,
) -> list[str]:
    """Owner-only fields this PUT would change; ``[]`` when the tenant may make it.

    ``base_url_in_body`` is the body's value (``None`` = absent = preserved).
    ``key_in_body`` is True when the body set OR cleared ``apiKey``.
    """
    managed = bool(existing and existing.get("owner_provided"))
    if not managed:
        return ["owner_provided"] if owner_provided_final else []
    assert existing is not None  # noqa: S101 -- narrowed by `managed`
    changed: list[str] = []
    if provider != existing.get("provider"):
        changed.append("provider")
    if base_url_in_body is not None and not _same_url(base_url_in_body, existing.get("base_url")):
        changed.append("base_url")
    if key_in_body:
        changed.append("api_key")
    if not owner_provided_final:
        changed.append("owner_provided")
    return changed


def assert_managed_row_change_allowed(
    existing: dict[str, Any] | None,
    *,
    provider: str,
    base_url_in_body: str | None,
    key_in_body: bool,
    owner_provided_final: bool,
) -> None:
    """Raise 403 ``managed_byo_row_locked`` when a non-owner changes an owner-only field."""
    changed = managed_row_changes(
        existing,
        provider=provider,
        base_url_in_body=base_url_in_body,
        key_in_body=key_in_body,
        owner_provided_final=owner_provided_final,
    )
    if not changed or is_managed_byo_owner_caller():
        return
    if existing and existing.get("owner_provided"):
        detail = (
            "This provider is managed by the platform owner: "
            f"{', '.join(changed)} cannot be changed by the tenant. "
            "Only model and parameters may be edited; ask the platform owner to change the rest."
        )
    else:
        detail = "Only the platform owner may mark a provider as owner-provided (managed BYO)."
    raise APIError(403, detail, error_code=MANAGED_BYO_ROW_LOCKED)
