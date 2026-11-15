"""Who moved a prompt version: the actor stamped on ``prompt_version_transitions``.

⛔ The actor comes from a VERIFIED principal only — never from a request header
or body. In precedence order:

1. ``service_account`` — the ``azp`` of a signature-verified
   ``X-Caller-Authorization`` SA JWT. ``AuthenticationMiddleware`` verifies it
   (``src/auth/keycloak_roles.verified_caller_identity``) and stores it in the
   caller-identity ContextVar; this is the SAME principal ``require_role`` reads
   its roles from, so the ``service-internal`` moves (DISCARDED, activate,
   ``initial_create``) always record the service account that was authorised.
2. ``licence_subject`` — ``subject_id`` from the licence INTROSPECTION result
   (``request.state.license``), and only when the middleware stamped the
   identity proven. ⚠️ Deliberately NOT ``request.state.subject_id``: on the
   delegation branch that is overwritten from the caller's ``X-Subject-Id``
   header.
3. ``unverified`` — nothing verified (tokenless trusted-network branch, bypass,
   no middleware). ``actor_sub`` is NULL: the tokenless branch's ``sub`` is the
   tenant a bare ``X-Tenant-Id`` header asserted, i.e. a header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.api.identity_context import get_caller_client_id_from_context

if TYPE_CHECKING:
    from fastapi import Request

ACTOR_SERVICE_ACCOUNT = "service_account"
ACTOR_LICENCE_SUBJECT = "licence_subject"
ACTOR_UNVERIFIED = "unverified"

# Column bound (migration 033 CHECK); a longer principal is truncated, not refused.
_MAX_ACTOR_SUB = 255


@dataclass(frozen=True)
class Actor:
    """``actor_sub`` (may be None) and ``actor_kind`` for one ledger row."""

    sub: str | None
    kind: str


def resolve_actor(request: Request) -> Actor:
    """The verified principal behind ``request`` (see module docstring)."""
    sa = (get_caller_client_id_from_context() or "").strip()
    if sa:
        return Actor(sub=sa[:_MAX_ACTOR_SUB], kind=ACTOR_SERVICE_ACCOUNT)

    state = request.state
    if getattr(state, "identity_proven", False) is True:
        licence: Any = getattr(state, "license", None)
        subject = licence.get("subject_id") if isinstance(licence, dict) else None
        if subject is not None and str(subject).strip():
            return Actor(sub=str(subject).strip()[:_MAX_ACTOR_SUB], kind=ACTOR_LICENCE_SUBJECT)

    return Actor(sub=None, kind=ACTOR_UNVERIFIED)
