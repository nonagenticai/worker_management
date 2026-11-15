"""DB access for per-tenant BYO LLM config (``tenant_llm_config`` table).

One row per (tenant_id, role). Phase 1 only ever touches role='__default__'.
``api_key_encrypted`` holds AES-256-GCM ciphertext (src/security/secret_box.py)
or NULL — plaintext is never stored here.

See migrations/021_tenant_llm_config.sql. Every write the API makes is also
recorded in ``tenant_llm_config_audit`` (migrations/037), in the SAME
transaction as the change, so the customer can see who set, cleared, changed,
removed or tested its key.
"""

from __future__ import annotations

import contextlib
import hashlib
from typing import Any, Optional

from src.db.db_config import get_pg_connection
from src.db.workers_state_db import with_retry
from src.worker.agents.provider_catalog import assert_local_preset_stays_on_hardware

DEFAULT_ROLE = "__default__"

# Closed sets, mirrored by the CHECK constraints in migrations/037.
# ``test_refused`` since 038 (WM-4): a key test refused before any live call.
AUDIT_ACTIONS = frozenset({"set_key", "clear_key", "update", "remove", "test", "test_refused"})
AUDIT_ACTOR_KINDS = frozenset({"platform_operator", "workspace_admin", "service"})
AUDIT_LIMIT_DEFAULT = 50
AUDIT_LIMIT_MAX = 200

_AUDIT_INSERT = """
    INSERT INTO tenant_llm_config_audit
        (tenant_id, role, action, actor_kind, provider, key_id)
    VALUES ($1, $2, $3, $4, $5, $6)
"""


def key_fingerprint(encrypted: Optional[str | bytes]) -> str:
    """Non-reversible KEY ID: 12 hex of SHA-256 of the STORED (encrypted) value.

    Never of the plaintext key. ``none`` when no key is involved. Used by the
    operator ``owner_action`` log line and the customer-visible audit row.
    """
    if not encrypted:
        return "none"
    raw = encrypted if isinstance(encrypted, bytes) else str(encrypted).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _check_audit(action: str, actor_kind: str) -> None:
    if action not in AUDIT_ACTIONS:
        raise ValueError(f"unknown audit action {action!r}")
    if actor_kind not in AUDIT_ACTOR_KINDS:
        raise ValueError(f"unknown audit actor_kind {actor_kind!r}")


class TenantLLMConfigDB:
    """CRUD for the ``tenant_llm_config`` table, scoped by tenant_id."""

    @with_retry()
    async def get_config(
        self, tenant_id: str, role: str = DEFAULT_ROLE
    ) -> Optional[dict[str, Any]]:
        """Return the config row for (tenant_id, role), or None."""
        async with get_pg_connection() as conn:
            return await conn.fetchrow(
                """
                SELECT tenant_id, role, provider, model, base_url, temperature,
                       max_tokens, params, api_key_encrypted, cost_source,
                       owner_provided, created_at, updated_at
                  FROM tenant_llm_config
                 WHERE tenant_id = $1 AND role = $2
                """,
                tenant_id,
                role,
            )

    @with_retry()
    async def list_configs(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return every configured row for a tenant (all roles/tiers).

        Ordered for stable display: the ``__default__`` row first, then tier
        rows. Used by the role-aware ``GET /tenant-config/llm/all`` endpoint that
        the multi-tier settings UI consumes.
        """
        async with get_pg_connection() as conn:
            return await conn.fetch(
                """
                SELECT tenant_id, role, provider, model, base_url, temperature,
                       max_tokens, params, api_key_encrypted, cost_source,
                       owner_provided, created_at, updated_at
                  FROM tenant_llm_config
                 WHERE tenant_id = $1
                 ORDER BY (role <> $2), role
                """,
                tenant_id,
                DEFAULT_ROLE,
            )

    @with_retry()
    async def upsert_config(  # noqa: PLR0913 -- SQL upsert with one arg per config column
        self,
        *,
        tenant_id: str,
        role: str = DEFAULT_ROLE,
        provider: str,
        model: str,
        base_url: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        params: dict[str, Any],
        api_key_encrypted: Optional[str],
        cost_source: str = "worker-management",
        owner_provided: bool = False,
        audit_action: Optional[str] = None,
        audit_actor_kind: Optional[str] = None,
    ) -> None:
        """Insert or update the (tenant_id, role) row.

        When ``audit_action``/``audit_actor_kind`` are given, the
        ``tenant_llm_config_audit`` row is inserted in the SAME transaction: if
        the audit insert fails the config change rolls back with it, so a change
        the customer cannot see never commits.

        ``api_key_encrypted`` is written verbatim: the caller decides whether it
        is a freshly-encrypted value, the preserved prior value, or None (clear).
        ``owner_provided`` marks a managed-BYO key (platform supplied it); the
        caller preserves the prior value when the PUT body omits the flag.

        Raises:
            PaidUpstreamForLocalPresetError: ``provider``/``model`` name the
                platform (non-external) preset while the model is served by a
                PAID upstream. The API layer checks this too and turns it into a
                422; enforcing it HERE as well is what makes the rule
                unbypassable — this is the only function in the process that can
                write the table, so no script, shell, or future caller can route
                around it. See the $140.31 measured in
                migrations/025_purge_local_preset_paid_tier_rows.sql.
        """
        assert_local_preset_stays_on_hardware(provider, model)
        audited = audit_action is not None or audit_actor_kind is not None
        if audited:
            _check_audit(audit_action or "", audit_actor_kind or "")
        async with (
            get_pg_connection() as conn,
            conn.transaction() if audited else contextlib.nullcontext(),
        ):
            await conn.execute(
                """
                INSERT INTO tenant_llm_config
                    (tenant_id, role, provider, model, base_url, temperature,
                     max_tokens, params, api_key_encrypted, cost_source,
                     owner_provided, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now())
                ON CONFLICT (tenant_id, role) DO UPDATE SET
                    provider          = EXCLUDED.provider,
                    model             = EXCLUDED.model,
                    base_url          = EXCLUDED.base_url,
                    temperature       = EXCLUDED.temperature,
                    max_tokens        = EXCLUDED.max_tokens,
                    params            = EXCLUDED.params,
                    api_key_encrypted = EXCLUDED.api_key_encrypted,
                    cost_source       = EXCLUDED.cost_source,
                    owner_provided    = EXCLUDED.owner_provided,
                    updated_at        = now()
                """,
                tenant_id,
                role,
                provider,
                model,
                base_url,
                temperature,
                max_tokens,
                params,
                api_key_encrypted,
                cost_source,
                owner_provided,
            )
            if audited:
                await conn.execute(
                    _AUDIT_INSERT,
                    tenant_id,
                    role,
                    audit_action,
                    audit_actor_kind,
                    provider,
                    key_fingerprint(api_key_encrypted),
                )

    @with_retry()
    async def delete_config(
        self,
        tenant_id: str,
        role: str = DEFAULT_ROLE,
        *,
        audit_actor_kind: Optional[str] = None,
    ) -> bool:
        """Delete the (tenant_id, role) row; True when a row was deleted.

        With ``audit_actor_kind``, a ``remove`` audit row (provider and key id
        of the DELETED row) is inserted in the same transaction. Deleting an
        absent row is a no-op and records nothing: nothing was removed.
        """
        if audit_actor_kind is not None:
            _check_audit("remove", audit_actor_kind)
        async with get_pg_connection() as conn, conn.transaction():
            gone = await conn.fetchrow(
                "DELETE FROM tenant_llm_config WHERE tenant_id = $1 AND role = $2 "
                "RETURNING provider, api_key_encrypted",
                tenant_id,
                role,
            )
            if gone is not None and audit_actor_kind is not None:
                await conn.execute(
                    _AUDIT_INSERT,
                    tenant_id,
                    role,
                    "remove",
                    audit_actor_kind,
                    gone.get("provider"),
                    key_fingerprint(gone.get("api_key_encrypted")),
                )
        return gone is not None

    @with_retry()
    async def record_audit(
        self,
        *,
        tenant_id: str,
        role: str,
        action: str,
        actor_kind: str,
        provider: Optional[str],
        key_id: str,
    ) -> None:
        """Insert one audit row on its own (the key TEST, which changes no config)."""
        _check_audit(action, actor_kind)
        async with get_pg_connection() as conn:
            await conn.execute(_AUDIT_INSERT, tenant_id, role, action, actor_kind, provider, key_id)

    @with_retry()
    async def list_audit(
        self, tenant_id: str, limit: int = AUDIT_LIMIT_DEFAULT
    ) -> list[dict[str, Any]]:
        """One tenant's audit rows, newest first, at most ``AUDIT_LIMIT_MAX``.

        Never selects anything identifying the actor beyond ``actor_kind``: the
        table holds no ``sub`` to select.
        """
        limit = max(1, min(int(limit), AUDIT_LIMIT_MAX))
        async with get_pg_connection() as conn:
            return await conn.fetch(
                """
                SELECT role, action, actor_kind, provider, key_id, created_at
                  FROM tenant_llm_config_audit
                 WHERE tenant_id = $1
                 ORDER BY created_at DESC, id DESC
                 LIMIT $2
                """,
                tenant_id,
                limit,
            )
