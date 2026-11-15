"""Authentication module for worker management service.

Uses license token authentication only. No Keycloak/JWT.
"""

from src.auth.dependencies import (
    TenantContext,
    get_byo_mode,
    get_identity_proven,
    get_subject_id,
    get_tenant_context,
    get_tenant_name,
    get_tenant_proven,
    get_visibility,
)
from src.auth.tenant import get_resource_with_tenant_check, get_tenant_id

__all__ = [
    "TenantContext",
    "get_byo_mode",
    "get_identity_proven",
    "get_resource_with_tenant_check",
    "get_subject_id",
    "get_tenant_context",
    # Tenant Isolation
    "get_tenant_id",
    "get_tenant_name",
    "get_tenant_proven",
    "get_visibility",
]
