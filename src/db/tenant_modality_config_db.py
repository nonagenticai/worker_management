"""DB access for per-tenant BYO modality config (``tenant_modality_config``).

One row per (tenant_id, modality, role) where modality ∈ {tts, asr, image,
embedding, vlm}. Mirrors :mod:`src.db.tenant_llm_config_db` with a ``modality``
dimension and no LLM-only columns (temperature/max_tokens); provider-specific
knobs live in ``params``. ``api_key_encrypted`` holds AES-256-GCM ciphertext
(src/security/secret_box.py) or NULL — plaintext is never stored here.

See migrations/022_tenant_modality_config.sql.
"""

from __future__ import annotations

from typing import Any, Optional

from src.db.db_config import get_pg_connection
from src.db.workers_state_db import with_retry

DEFAULT_ROLE = "__default__"

# Modalities this config table covers. Kept here (not an enum column) so adding
# one never needs a migration — validation lives at the API/catalog layer.
MODALITIES: frozenset[str] = frozenset({"tts", "asr", "image", "embedding", "vlm"})


class TenantModalityConfigDB:
    """CRUD for ``tenant_modality_config``, scoped by tenant_id + modality."""

    @with_retry()
    async def get_config(
        self, tenant_id: str, modality: str, role: str = DEFAULT_ROLE
    ) -> Optional[dict[str, Any]]:
        """Return the config row for (tenant_id, modality, role), or None."""
        async with get_pg_connection() as conn:
            return await conn.fetchrow(
                """
                SELECT tenant_id, modality, role, provider, model, base_url,
                       params, api_key_encrypted, cost_source, owner_provided,
                       created_at, updated_at
                  FROM tenant_modality_config
                 WHERE tenant_id = $1 AND modality = $2 AND role = $3
                """,
                tenant_id,
                modality,
                role,
            )

    @with_retry()
    async def list_configs(
        self, tenant_id: str, modality: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Return configured rows for a tenant, optionally filtered by modality.

        Ordered for stable display: ``__default__`` first, then tier rows,
        grouped by modality.
        """
        async with get_pg_connection() as conn:
            if modality is None:
                return await conn.fetch(
                    """
                    SELECT tenant_id, modality, role, provider, model, base_url,
                           params, api_key_encrypted, cost_source, owner_provided,
                           created_at, updated_at
                      FROM tenant_modality_config
                     WHERE tenant_id = $1
                     ORDER BY modality, (role <> $2), role
                    """,
                    tenant_id,
                    DEFAULT_ROLE,
                )
            return await conn.fetch(
                """
                SELECT tenant_id, modality, role, provider, model, base_url,
                       params, api_key_encrypted, cost_source, owner_provided,
                       created_at, updated_at
                  FROM tenant_modality_config
                 WHERE tenant_id = $1 AND modality = $2
                 ORDER BY (role <> $3), role
                """,
                tenant_id,
                modality,
                DEFAULT_ROLE,
            )

    @with_retry()
    async def upsert_config(
        self,
        *,
        tenant_id: str,
        modality: str,
        role: str = DEFAULT_ROLE,
        provider: str,
        model: str,
        base_url: Optional[str],
        params: dict[str, Any],
        api_key_encrypted: Optional[str],
        cost_source: str = "worker-management",
        owner_provided: bool = False,
    ) -> None:
        """Insert or update the (tenant_id, modality, role) row.

        ``api_key_encrypted`` is written verbatim: the caller decides whether it
        is a freshly-encrypted value, the preserved prior value, or None (clear).
        ``owner_provided`` marks a managed-BYO key (platform supplied it); the
        caller preserves the prior value when the PUT body omits the flag.
        """
        async with get_pg_connection() as conn:
            await conn.execute(
                """
                INSERT INTO tenant_modality_config
                    (tenant_id, modality, role, provider, model, base_url,
                     params, api_key_encrypted, cost_source, owner_provided,
                     updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, now())
                ON CONFLICT (tenant_id, modality, role) DO UPDATE SET
                    provider          = EXCLUDED.provider,
                    model             = EXCLUDED.model,
                    base_url          = EXCLUDED.base_url,
                    params            = EXCLUDED.params,
                    api_key_encrypted = EXCLUDED.api_key_encrypted,
                    cost_source       = EXCLUDED.cost_source,
                    owner_provided    = EXCLUDED.owner_provided,
                    updated_at        = now()
                """,
                tenant_id,
                modality,
                role,
                provider,
                model,
                base_url,
                params,
                api_key_encrypted,
                cost_source,
                owner_provided,
            )

    @with_retry()
    async def delete_config(self, tenant_id: str, modality: str, role: str = DEFAULT_ROLE) -> None:
        """Delete the (tenant_id, modality, role) row (no-op if absent)."""
        async with get_pg_connection() as conn:
            await conn.execute(
                """
                DELETE FROM tenant_modality_config
                 WHERE tenant_id = $1 AND modality = $2 AND role = $3
                """,
                tenant_id,
                modality,
                role,
            )
