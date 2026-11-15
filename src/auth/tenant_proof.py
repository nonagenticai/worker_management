"""Did THIS request PROVE its tenant? The one predicate (rm#187 class).

A privilege keyed on a tenant is open if the tenant can be asserted. Two
privileges in this service are keyed on a tenant and must therefore be granted
only to a tenant the request PROVED — never to one named in a bare
``X-Tenant-Id`` / ``X-Effective-Tenant-Id`` header, and never to the default
tenant a tokenless request falls back to:

* minting an inference-router bundle token for the tenant
  (``src.services.ir_bundle_tokens``, worker_management#325);
* using the OWNER'S gitops-registered commercial LLM keys (xAI, …) as a
  platform workspace (owner ruling 2026-10-03, fr#1385):
  :func:`owner_commercial_keys_permitted`.

Both call :func:`tenant_proof_refusal`, so the two cannot disagree about what
"proven" means.
"""

from __future__ import annotations

# Why a tenant was refused. A CLOSED set (log field, never a metric label).
_NOT_PROVEN_CONTEXT_MISMATCH = "not_the_request_tenant"
_NOT_PROVEN_DELEGATION_UNVERIFIED = "delegated_without_verified_end_user_token"
NOT_A_COMMERCIAL_PLATFORM_TENANT = "not_a_commercial_platform_tenant"


def ir_delegating_subjects() -> frozenset[str]:
    """``WM_DELEGATING_SUBJECTS`` (mirror of the router's ``AUTH_DELEGATING_SUBJECTS``)."""
    from src.settings import get_settings

    raw = get_settings().wm_delegating_subjects or ""
    return frozenset(s.strip().lower() for s in raw.split(",") if s.strip())


def tenant_proof_refusal(tenant_id: str, *, ir_token: bool = False) -> str | None:
    """``None`` when THIS request PROVED ``tenant_id``; else why it did not.

    ⛔ A privilege keyed on a tenant is open if the tenant can be asserted. A
    bundle token lets WM act at inference-router AS that tenant, so it is minted
    only for a tenant the caller proved with a credential WM (or licenses, on
    WM's behalf) verified. Every value of ``tenant_provenance``
    (``src/api/identity_context.py:102-116``) is handled; anything not listed as
    proven below is refused, including values added later (fail closed).

    What "verified" means, per kind:

    * ``license`` — PROVEN. The caller presented a licence/service token
      (``X-Tenant-Id: Bearer <token>``, ``auth_middleware.py:1013``); licenses
      ``/license/introspect`` returned ``active`` for it
      (``auth_middleware.py:1019``) and the tenant is the one that introspection
      returned (``auth_middleware.py:1040``), never a header. The stamp is
      ``license`` only when no ``X-Effective-Tenant-Id`` re-scope happened
      (``auth_middleware.py:1281-1283``). Covers both an end-user licence and a
      service account's own licence.
    * ``delegated`` — PROVEN ONLY WITH A VERIFIED END-USER TOKEN. The licence
      proves WHO (a subject in ``AUTH_DELEGATING_SUBJECTS``) but the tenant came
      from the ``X-Effective-Tenant-Id`` HEADER (``auth_middleware.py:1122``),
      checked for spelling only. That alone is an assertion and is refused. It
      counts as proven only when the forwarded ``X-End-User-Token`` verified
      cryptographically in ``src/auth/end_user_token.py``: signature against the
      portal JWKS + issuer pin + expiry (``_verified_payload``, :160-197), an
      ACCESS token from an allowed portal client (``typ``/``azp``,
      ``_token_shape``), and its claims proving this tenant: ``tenant_id``
      present and this tenant equal to it or listed in the admin-only
      ``member_tenant_ids`` claim (``token_proves_tenant``, fr#1385 Option 1);
      recorded only on ``PROOF_OK``. The forwarding subject must STILL be a configured delegator.
      Same predicate as ``llm_config_resolver.tenant_key_entitled``.
    * ``asserted`` — NEVER. Tokenless trusted-network caller naming a tenant in a
      bare ``X-Tenant-Id`` (``auth_middleware.py:1390-1399``).
    * ``default`` — NEVER. Tokenless, no header: ``DEFAULT_TENANT_ID``.
    * ``bypass`` — NEVER. ``bypass_auth`` (test affordance).
    * ``unknown`` — NEVER. No request context stated a provenance (background
      task, a path that never ran the middleware).

    ``ir_token=True`` (ONLY the inference-router bundle-token mint passes it):
    a ``delegated`` request WITHOUT an end-user token is also proven when its
    VERIFIED delegating subject (introspected licence subject, recorded by the
    middleware) is on ``WM_DELEGATING_SUBJECTS`` -- the same delegating-service
    allowlist inference-router itself honours (``AUTH_DELEGATING_SUBJECTS``), so
    WM mints the NAMED END tenant's token and billing goes to that tenant. The
    owner-key gate never passes it and is unchanged.

    And for every kind: ``tenant_id`` must be the REQUEST's tenant. A licence
    proves one tenant; a different tenant taken from graph state, a body field
    or a worker-config fallback proves nothing.
    """
    from src.api.identity_context import (
        PROVEN_TENANT_PROVENANCES,
        TENANT_PROVENANCE_DELEGATED,
        get_delegating_subject_from_context,
        get_end_user_proof,
        get_tenant_id_from_context,
        get_tenant_provenance_from_context,
    )

    tenant = tenant_id.strip().lower()
    if (get_tenant_id_from_context() or "").strip().lower() != tenant:
        return _NOT_PROVEN_CONTEXT_MISMATCH
    provenance = get_tenant_provenance_from_context()
    if provenance in PROVEN_TENANT_PROVENANCES:
        return None
    if provenance == TENANT_PROVENANCE_DELEGATED:
        from src.auth.end_user_token import delegating_subjects

        proven_tenant, delegator = get_end_user_proof()
        if (
            proven_tenant
            and proven_tenant.strip().lower() == tenant
            and delegator
            and delegator in delegating_subjects()
        ):
            return None
        if ir_token:
            subject = get_delegating_subject_from_context().strip().lower()
            if subject and subject in ir_delegating_subjects():
                return None
        return _NOT_PROVEN_DELEGATION_UNVERIFIED
    return f"provenance_{provenance}" if provenance else "provenance_unknown"


def owner_commercial_keys_refusal(tenant_id: object) -> str | None:
    """``None`` when ``tenant_id`` may use the owner's commercial LLM keys; else why not.

    Owner ruling 2026-10-03: "allow all gitops registered commercial apis for
    platform workspaces only". Two conditions, BOTH required:

    1. the tenant is in the gitops-defined commercial-platform set
       (:func:`src.api.tenant_constants.is_commercial_platform_tenant`);
    2. THIS request proved that tenant (:func:`tenant_proof_refusal`) — a
       licence introspected by licenses, or a verified end-user token whose
       ``tenant_id`` claim is this tenant. An asserted header or the default
       tenant never qualifies.

    Whether a key for the provider is actually REGISTERED (mounted from gitops
    secrets) is the caller's separate check. A guard error is a refusal.
    """
    from src.api.tenant_constants import is_commercial_platform_tenant

    if not isinstance(tenant_id, str) or not is_commercial_platform_tenant(tenant_id):
        return NOT_A_COMMERCIAL_PLATFORM_TENANT
    try:
        return tenant_proof_refusal(tenant_id)
    except Exception as exc:  # the guard failing is a refusal, never a grant
        return f"guard_error:{type(exc).__name__}"


def owner_commercial_keys_permitted(tenant_id: object) -> bool:
    """True only when :func:`owner_commercial_keys_refusal` finds no reason to refuse."""
    return owner_commercial_keys_refusal(tenant_id) is None
