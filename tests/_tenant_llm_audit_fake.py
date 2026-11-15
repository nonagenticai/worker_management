"""In-memory half of ``TenantLLMConfigDB``'s audit surface, for router fakes.

``/tenant-config/llm`` PUT/DELETE/TEST write a ``tenant_llm_config_audit`` row
(migration 037) and ``GET /llm/audit`` reads them. Every in-memory stand-in for
``TenantLLMConfigDB`` mixes this in so it mirrors the real semantics:

* ``upsert_config(..., audit_action=, audit_actor_kind=)`` records one row
  (call :meth:`AuditFakeMixin.take_audit` from the fake's ``upsert_config``);
* ``delete_config(..., audit_actor_kind=)`` records ``remove`` only when a row
  was actually deleted (:meth:`AuditFakeMixin.audit_delete`);
* ``record_audit`` / ``list_audit`` behave like the SQL (tenant-scoped, newest
  first, capped at ``AUDIT_LIMIT_MAX``).
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
from typing import Any, Optional

# Mirrors src.db.tenant_llm_config_db (AUDIT_LIMIT_MAX / key_fingerprint). Kept
# local so a fake never imports the code under test: a regression there must
# show up as a failing assertion, not as a collection error in every file.
AUDIT_LIMIT_MAX = 200
_SEQ = itertools.count(1)


def key_fingerprint(encrypted: Optional[str | bytes]) -> str:
    if not encrypted:
        return "none"
    raw = encrypted if isinstance(encrypted, bytes) else str(encrypted).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


class AuditFakeMixin:
    """Mix into a fake ``TenantLLMConfigDB``; rows land in ``self.audit_rows``."""

    @property
    def audit_rows(self) -> list[dict[str, Any]]:
        rows = self.__dict__.get("_audit_rows")
        if rows is None:
            rows = self.__dict__["_audit_rows"] = []
        return rows

    def _append(self, **row: Any) -> None:
        row["id"] = next(_SEQ)
        row["created_at"] = datetime.datetime.now(datetime.UTC)
        self.audit_rows.append(row)

    def take_audit(self, tenant_id: str, role: str, cols: dict[str, Any]) -> None:
        """Pop the upsert's audit kwargs out of ``cols`` and record them."""
        action = cols.pop("audit_action", None)
        actor_kind = cols.pop("audit_actor_kind", None)
        if action is None and actor_kind is None:
            return
        self._append(
            tenant_id=str(tenant_id),
            role=role,
            action=action,
            actor_kind=actor_kind,
            provider=cols.get("provider"),
            key_id=key_fingerprint(cols.get("api_key_encrypted")),
        )

    def audit_delete(
        self, tenant_id: str, role: str, gone: Optional[dict[str, Any]], actor_kind: Optional[str]
    ) -> None:
        """Record ``remove`` for a row that was actually deleted."""
        if gone is None or actor_kind is None:
            return
        self._append(
            tenant_id=str(tenant_id),
            role=role,
            action="remove",
            actor_kind=actor_kind,
            provider=gone.get("provider"),
            key_id=key_fingerprint(gone.get("api_key_encrypted")),
        )

    async def record_audit(self, **row: Any) -> None:
        self._append(**{**row, "tenant_id": str(row["tenant_id"])})

    async def list_audit(self, tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), AUDIT_LIMIT_MAX))
        mine = [dict(r) for r in self.audit_rows if r["tenant_id"] == str(tenant_id)]
        mine.sort(key=lambda r: (r["created_at"], r["id"]), reverse=True)
        return mine[:limit]
