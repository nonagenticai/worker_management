"""Centralized configuration for worker_management.

All environment reads go through a single pydantic ``Settings`` class.
Environment-specific values (URLs, DSNs, hostnames) carry empty-string
defaults; a ``model_post_init`` validator fails fast in production when
infra fields are missing.

No ``env_prefix`` is used so existing deployed env var names (``DATABASE_URL``,
``MINIO_ENDPOINT``, ``LICENSE_INTROSPECT_URL`` etc.) bind directly.
"""

from __future__ import annotations

import json
import re
import uuid
from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Hosts that are GPU inference backends behind inference-router's licence
# gate. No worker_management URL may point at one of these directly — every
# model call goes through inference-router (owner ruling 2026-10-06).
DIRECT_GPU_HOSTS: frozenset[str] = frozenset({"vllm.example.com"})


def is_direct_gpu_url(url: str) -> bool:
    """Return True when ``url``'s host is in :data:`DIRECT_GPU_HOSTS`."""
    if not url:
        return False
    host = urlsplit(url if "://" in url else f"//{url}").hostname or ""
    return host.lower().rstrip(".") in DIRECT_GPU_HOSTS


# The ONE place worker_management names the estate's self-hosted model. It is
# the pgx vLLM ``--served-model-name`` (the vllm.service unit), NOT a concrete
# HF model id: pgx serves exactly one model and swapping it must not require a
# code change here. Every default that means "the platform model" reads
# ``Settings.local_llm_model`` (which defaults to this) or, for the static
# import-time catalogs, this constant. Provenance of what ACTUALLY served a
# call comes from read-back (IR's ``served_model`` body field), never from
# this configured name.
LOCAL_LLM_MODEL = "local-llm"

_CALLER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_CANONICAL_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def parse_tenant_id_csv(raw: str, *, env_name: str) -> frozenset[str]:
    """Canonical, non-nil tenant UUIDs from the comma-separated ``raw``.

    Raises ``ValueError`` naming the offending entry. ⛔ Nothing is silently
    dropped here: every one of these env values GATES ACCESS, so a typo narrows
    the set — the safe direction — but it does so quietly, and the operator who
    meant to admit a tenant would never learn they failed. A loud startup
    failure is the only outcome that cannot be misread as "configured".
    """
    ids: set[str] = set()
    for part in (raw or "").split(","):
        candidate = part.strip()
        if not candidate:
            continue
        try:
            parsed = uuid.UUID(candidate)
        except ValueError:
            raise ValueError(f"{env_name}: {candidate!r} is not a uuid") from None
        if parsed.int == 0:
            raise ValueError(f"{env_name}: the nil uuid is never a tenant")
        if candidate.lower() != str(parsed):
            raise ValueError(
                f"{env_name}: {candidate!r} is not a canonical (8-4-4-4-12) uuid; "
                "braced, urn: and undashed spellings are refused so the value a "
                "header carries and the value a log shows are the same string"
            )
        ids.add(str(parsed))
    return frozenset(ids)


def parse_tenant_label_map(raw: str, *, env_name: str) -> dict[str, str]:
    """``"<label>=<uuid>,..."`` → ``{label: canonical uuid}``. Raises on a bad entry."""
    mapping: dict[str, str] = {}
    for entry in (raw or "").split(","):
        part = entry.strip()
        if not part:
            continue
        label, sep, value = part.partition("=")
        label, value = label.strip(), value.strip()
        if not sep or not label or not value:
            raise ValueError(f"{env_name}: {part!r} is not <label>=<uuid>")
        (uuid_value,) = parse_tenant_id_csv(value, env_name=f"{env_name} label {label!r}")
        if label in mapping:
            raise ValueError(f"{env_name}: duplicate label {label!r}")
        mapping[label] = uuid_value
    return mapping


def parse_project_id_caller_licences(raw: str) -> dict[str, str]:
    """Parse ``"<license_id>=<name>,..."`` into ``{license_id(lower): name}``.

    Raises ``ValueError`` naming the bad entry: a typo here would silently
    leave a caller on the shared ``licence`` label, which is exactly the
    misread ("zero missing for n8n") this map exists to prevent.
    """
    mapping: dict[str, str] = {}
    for entry in (e.strip() for e in (raw or "").split(",")):
        if not entry:
            continue
        lic, sep, name = (part.strip() for part in entry.partition("="))
        if not sep or not _CANONICAL_UUID_RE.match(lic) or not _CALLER_NAME_RE.match(name):
            raise ValueError(
                f"malformed entry {entry!r}: expected <license_id uuid>=<name>, "
                "name matching ^[a-z0-9][a-z0-9._-]{0,63}$"
            )
        if lic.lower() in mapping:
            raise ValueError(f"license_id {lic!r} mapped twice")
        mapping[lic.lower()] = name
    return mapping


_BUNDLE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def parse_ir_bundle_by_app(raw: str) -> dict[str, str]:
    """Parse ``WM_IR_BUNDLE_BY_APP`` (JSON object ``{app_key: bundle_key}``).

    Empty / blank ⇒ ``{}``. Raises ``ValueError`` naming the problem: a typo
    here would silently leave a tenant's inference-router calls tokenless on a
    409 ``ambiguous_bundle`` instead of minting under the intended bundle.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"is not valid JSON ({exc.msg})") from None
    entries = list(data.items()) if isinstance(data, dict) else None
    if entries is None:
        raise ValueError("must be a JSON object mapping app key -> bundle key")
    mapping: dict[str, str] = {}
    for app, bundle in entries:
        if not (
            isinstance(app, str)
            and isinstance(bundle, str)
            and _BUNDLE_KEY_RE.match(app)
            and _BUNDLE_KEY_RE.match(bundle)
        ):
            raise ValueError(
                f"malformed entry {app!r}: {bundle!r}: keys must be strings matching "
                "^[a-z0-9][a-z0-9._-]{0,63}$"
            )
        mapping[app] = bundle
    return mapping


class Settings(BaseSettings):
    """Worker management service settings.

    Values bind to bare env var names (no prefix) to preserve back-compat
    with the deployed manifest at
    ``clusters/production/services/workers/deployment.yaml``.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Environment / observability
    # ------------------------------------------------------------------
    environment: str = Field(default="development", description="Deployment environment")
    log_level: str = Field(default="INFO", description="Root log level")
    log_format: str = Field(
        default="text",
        description=(
            "Log output format: 'json' or 'text'. The deployment has set "
            "LOG_FORMAT=json since long before anything read it -- see "
            "api/app_config.configure_logging()."
        ),
    )
    otel_service_name: str = Field(
        default="worker-management-service",
        description="OTEL service.name resource attribute",
    )

    # ------------------------------------------------------------------
    # Database (infra — required in production)
    # ------------------------------------------------------------------
    database_url: str = Field(default="", description="PostgreSQL DSN")
    postgres_pool_min: int = Field(default=10, description="psycopg pool min_size")
    postgres_pool_max: int = Field(default=20, description="psycopg pool max_size")
    postgres_connect_timeout: int = Field(default=10, description="Connect timeout in seconds")
    postgres_command_timeout: int = Field(default=30, description="Command timeout in seconds")
    postgres_sslmode: str = Field(default="require", description="libpq sslmode")
    client_cert_path: str = Field(
        default="/app/certs/worker_mgmt_user-client-cert.pem",
        description="mTLS client cert path",
    )
    # The CA the database connection trusts. Defaulted to the image's mount so the
    # estate deployment is unchanged, but NOT required: when the file is absent the
    # pool falls back to the system trust store and logs it, so a clone with a plain
    # PostgreSQL (or sslmode=disable) runs without inventing a certificate.
    db_ca_cert_path: str = Field(
        default="/app/certs/ca-cert.pem",
        validation_alias=AliasChoices("DB_CA_CERT_PATH"),
        description="CA bundle used to verify the PostgreSQL server",
    )
    client_key_path: str = Field(
        default="/app/certs/worker_mgmt_user-client-key.pem",
        description="mTLS client key path",
    )

    # ------------------------------------------------------------------
    # License introspection (infra)
    # ------------------------------------------------------------------
    license_introspect_url: str = Field(
        default="",
        description="License introspection endpoint (licenses service)",
    )

    # ------------------------------------------------------------------
    # Tenant assertion on the tokenless trusted-network branch
    # ------------------------------------------------------------------
    # ⛔ THE DEFAULT MUST STAY ``False``. This is the "observe now, refuse
    # later" half of closing the tokenless trusted-network hole in
    # ``api/auth_middleware.dispatch`` (mirrors inference-router #125).
    #
    # On that branch a caller inside TRUSTED_CIDRS presents NO licence token
    # and still selects its own tenant with a bare ``X-Tenant-Id`` header,
    # which is then stamped as the request identity with a NULL licence. The
    # only check today is syntactic (``resolve_tenant_id``): any canonical
    # uuid is accepted, there is no allowlist and no subject check. Measured
    # in billing on 2026-08-31, worker-management emits cost events for 9
    # distinct tenant_ids across 27 worker types on this path, two of which
    # exist in NO licences registry, and the `n8n` namespace — which runs
    # user-authored workflows — is inside the Istio-admitted blast radius.
    #
    # Ten-plus live workloads depend on this path RIGHT NOW. Shipping this
    # default-closed would 401 production the moment the pod rolls, so the
    # gate ships open and gets closed per-caller once the WARNING below has
    # named every peer and each has been moved onto a licence token.
    #
    # ``False`` is NOT "off". The WARNING and the
    # ``worker_management_unauthenticated_tenant_assertion_total`` counter fire in BOTH
    # states — that is the entire point of the flag existing before it is
    # enabled. False changes the OUTCOME (served vs 401), never the
    # observability.
    require_authenticated_tenant: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT",
            "REQUIRE_AUTHENTICATED_TENANT",
        ),
        description=(
            "When True, a tokenless trusted-network request that asserts a tenant "
            "via the X-Tenant-Id header without a licence credential is refused 401. "
            "Default False = served exactly as today; the warning + counter fire either way."
        ),
    )

    # X-Project-Id gate (owner ruling 2026-10-08: every conveyor call carries
    # its fr projectId). Same shape as the flag above: default OFF ships
    # count-only — `worker_management_project_id_missing_total{caller,reason}`
    # fires in BOTH states — and gitops flips it once that counter reads zero
    # for the callers that matter. True changes the OUTCOME only (served → 400).
    require_project_id_header: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "WORKER_MANAGEMENT_REQUIRE_PROJECT_ID_HEADER",
            "REQUIRE_PROJECT_ID_HEADER",
        ),
        description=(
            "When True, an authenticated non-public request whose X-Project-Id is absent "
            "or not a canonical non-nil UUID is refused 400 PROJECT_ID_REQUIRED. "
            "Default False = served; the counter fires either way."
        ),
    )
    # Tenant×app licence check (George 2026-10-15): there are no app-to-app
    # licences, only tenant×app ones, so a licence presented here must be a
    # `workers` licence (inference-router's EXPECTED_APP_KEY pattern). observe
    # (default) refuses nothing and counts `wm_licence_app_total`; enforce 401s
    # `wrong_app` / `no_app`. Delegation (AUTH_DELEGATING_SUBJECTS +
    # X-Effective-Tenant-Id) is allowed in BOTH modes until a later step.
    # ⛔ A Literal, not a str: an unknown value ("Enforce", "on", "") fails
    # Settings validation, so the pod refuses to start instead of parsing a typo
    # into observe (no fail-open parsing, 10-15 lesson).
    wm_licence_app_mode: Literal["observe", "enforce"] = Field(
        default="observe",
        validation_alias=AliasChoices("WM_LICENCE_APP_MODE"),
        description=(
            "observe: count wm_licence_app_total{app,path,outcome}, refuse nothing. "
            "enforce: a non-delegated licence whose app is not `workers` (wrong_app) "
            "or that introspects with no app (no_app) is refused 401 wrong_app. "
            "Delegated callers are allowed in both modes. Any other value refuses startup."
        ),
    )
    # Bounded `caller` names for licence-authenticated requests on the counter
    # above, keyed by the INTROSPECTED (verified) license_id — never a forwarded
    # X-License-Id or X-App-Id. Several ids may share one name; a duplicate id
    # is malformed. Unmapped licences read `licence:<license_id>`. Malformed =
    # startup failure only while enforcing (`_validate_project_id_caller_licences`).
    project_id_caller_licences: str = Field(
        default="",
        validation_alias=AliasChoices(
            "WORKER_MANAGEMENT_PROJECT_ID_CALLER_LICENCES",
            "PROJECT_ID_CALLER_LICENCES",
        ),
        description=(
            "CSV '<license_id>=<name>' mapping verified licences to caller labels on "
            "worker_management_project_id_*_total. Unmapped = 'licence:<license_id>'."
        ),
    )

    # ------------------------------------------------------------------
    # BYO LLM — per-tenant provider key encryption (secret_box)
    # ------------------------------------------------------------------
    wm_secret_key: str = Field(
        default="",
        description=(
            "Master key for AES-256-GCM encryption of per-tenant BYO LLM API "
            "keys (64-char hex / base64-32 / passphrase). SOPS-encrypted in "
            "gitops. Empty disables BYO key storage."
        ),
    )

    # ------------------------------------------------------------------
    # Caller-identity — secure-apps SA tokens for outbound calls (M3 of
    # caller-identity-rollout). When all three are set, outbound httpx calls
    # to private-DNS hosts get an X-Caller-Authorization header minted from a
    # client_credentials grant. Empty client_id disables the feature.
    # ------------------------------------------------------------------
    keycloak_token_url: str = Field(
        default="",
        description="Keycloak secure-apps token endpoint (client_credentials grant)",
    )
    keycloak_caller_client_id: str = Field(
        default="",
        description="Keycloak SA client_id for this service (caller-identity)",
    )
    keycloak_caller_client_secret: str = Field(
        default="",
        description="Keycloak SA client_secret (SOPS-encrypted in gitops)",
    )
    # INBOUND half of caller-identity (the outbound half is the three fields
    # above). Left empty, the JWKS endpoint is derived from
    # keycloak_token_url — same realm, standard OIDC layout — so no manifest
    # change is needed for the common case.
    keycloak_jwks_url: str = Field(
        default="",
        description="Keycloak secure-apps JWKS endpoint; derived from keycloak_token_url if empty",
    )
    # END-USER proof on delegated requests (owner ruling 2026-10-01): the
    # portal-realm JWT a delegating caller forwards in ``X-End-User-Token``.
    # Left empty, the JWKS endpoint is derived from ``keycloak_jwks_url`` (else
    # ``keycloak_token_url``) by swapping its realm for ``portal`` — the same
    # in-cluster Keycloak, standard OIDC layout — so no manifest change is
    # needed. The issuer is pinned in FULL: it is the portal realm's frontend
    # URL, the exact value fr-backend pins (``KEYCLOAK_ISSUER``).
    end_user_jwks_url: str = Field(
        default="",
        description="Keycloak portal-realm JWKS endpoint; derived from keycloak_jwks_url if empty",
    )
    end_user_token_issuer: str = Field(
        default="https://auth.nonagentic.ai/realms/portal",
        description="Required `iss` of a forwarded X-End-User-Token (portal realm frontend URL)",
    )
    # Realm-per-app (G1, 2026-10-22): fr's USERS moved from realm `portal` to
    # realm `fr`. A single pinned realm therefore rejects every fr end-user
    # token with UnknownKidError (measured live 2026-11-06: fr→WM
    # GET /providers 401, ``end_user_token_invalid upstream_reason=invalid:
    # UnknownKidError``), which breaks the whole super-admin workspace switch
    # for every tenant — not one tenant.
    #
    # This is a CSV of the realms whose access tokens may prove an end user's
    # tenant. Each realm is verified against ITS OWN JWKS and pinned to
    # ``<issuer-origin>/realms/<realm>``; the signature step is what makes a
    # cross-realm token fail, so adding a realm widens the trust to exactly
    # that realm and no further. ``end_user_token_issuer`` still supplies the
    # FRONTEND origin (Keycloak's ``iss`` is the frontend URL, never the
    # in-cluster JWKS origin) and remains an exact pin for the first realm.
    #
    # Default is ``portal,fr``: portal is every other delegating service,
    # ``fr`` is fr after its realm move. Both are the same Keycloak.
    end_user_token_realms: str = Field(
        default="portal,fr",
        description="CSV of Keycloak realms whose access tokens may prove an end user's tenant",
    )
    # WM-3: the end-user proof accepts only an ACCESS token (Keycloak ``typ`` =
    # ``Bearer``; an ID token carries ``typ`` = ``ID``) issued to one of these
    # portal-realm clients (``azp``). Default = the user-facing clients that
    # actually mint the tokens fr forwards, verified live 2026-10-02 against
    # auth.nonagentic.ai/realms/portal: ``fr-frontend`` (fr.nonagentic.ai,
    # fr ci-cd ``VITE_KEYCLOAK_CLIENT_ID``), ``kag-frontend`` (fr's SPA on the
    # kag host, ``frontend/src/lib/product.ts``) and ``portal-bff``
    # (portal.nonagentic.ai). An empty value refuses every token (deny-all).
    end_user_token_allowed_clients: str = Field(
        default="fr-frontend,kag-frontend,portal-bff",
        description="CSV of portal-realm azp values whose access tokens may prove an end user's tenant",
    )
    # `azp` values allowed to assert internal-service standing via a verified
    # X-Caller-Authorization JWT. Empty means DENY-ALL — deliberately unlike
    # license-authz's SERVICE_PRINCIPAL_ALLOWED_CLIENTS (where empty means
    # allow-any), because `secure-apps` also authenticates human dashboard
    # users who carry license-admin, one of the internal-service roles.
    caller_allowed_clients: str = Field(
        default="",
        description="CSV of Keycloak azp values permitted internal-service standing (empty = built-in default)",
    )
    # D5: the verified `azp` values that may receive the DECRYPTED key of a
    # managed-BYO (owner_provided) modality row from
    # GET /tenant-config/modality/{m}/resolve. That key is the OWNER's
    # credential: it is released ONLY to the gateway that spends it on the
    # tenant's behalf, never on the tenant's own licence token. The caller must
    # ALSO pass X-Caller-Authorization verification (signature + azp in
    # CALLER_ALLOWED_CLIENTS); this list narrows that further. Empty = deny-all.
    managed_byo_key_caller_clients: str = Field(
        default="modality-gateway",
        description="CSV of verified caller azp values allowed to receive an owner_provided modality key (empty = nobody)",
    )

    # ------------------------------------------------------------------
    # Kubernetes executor + MinIO (infra)
    # ------------------------------------------------------------------
    executor_namespace: str = Field(
        default="agent-jobs", description="k8s namespace for spawned agent Jobs"
    )
    executor_service_account: str = Field(
        default="default", description="ServiceAccount for spawned agent Jobs"
    )
    k8s_executor_image: str = Field(
        default="",
        description="Container image for claude-executor Jobs",
    )
    claude_session_secret_name: str = Field(
        default="claude-secrets",
        description="Secret holding Claude session credentials mounted into Jobs",
    )
    minio_endpoint: str = Field(default="", description="MinIO/S3 endpoint host:port")
    minio_bucket: str = Field(default="agent-artifacts", description="Bucket for agent artifacts")
    minio_access_key: str = Field(default="", description="MinIO access key")
    minio_secret_key: str = Field(default="", description="MinIO secret key")
    minio_secure: bool = Field(default=False, description="Use TLS for MinIO connection")

    # ------------------------------------------------------------------
    # Inference / model-routing backends (infra)
    # ------------------------------------------------------------------
    inference_router_url: str = Field(
        default="",
        description="Base URL for inference-router (model discovery + routing)",
    )
    local_llm_model: str = Field(
        default=LOCAL_LLM_MODEL,
        description=(
            "Served-model-name of the platform's self-hosted model (pgx vLLM "
            "--served-model-name). The single default for every 'use the platform "
            "model' path: auto-created workers, the pydantic-ai safe wrapper, the "
            "local-preset tier-alias coercion and the agent factory fallback."
        ),
    )
    # ------------------------------------------------------------------
    # Per-tenant inference-router bundle tokens + tier aliases (tenant
    # inference provisioning plan §2.4). Owner ruling: "use short-lived
    # tokens, no pasting". ⛔ BOTH FLAGS DEFAULT OFF: merging with the gitops
    # env unchanged must change nothing on the wire.
    # ------------------------------------------------------------------
    wm_ir_bundle_tokens: bool = Field(
        default=False,
        description=(
            "Send 'Authorization: Bearer <tenant token>' to the INFERENCE_ROUTER_URL "
            "host only. The token is minted per tenant from licenses "
            "POST /internal/tenants/{t}/bundle-token (≤60 min), cached in memory "
            "until expiry-5 min, never logged, never persisted."
        ),
    )
    wm_tier_alias_passthrough: bool = Field(
        default=False,
        description=(
            "Let a customer's tier-small|medium|high|flash / default reach "
            "inference-router unchanged (IR resolves it against the tenant's policy), "
            "and map local-llm/default + work_tier to tier-<work_tier>."
        ),
    )
    wm_ir_bundle_by_app: str = Field(
        default="",
        description=(
            "JSON object {calling app key (X-App-Id): licences bundle key}. Consulted "
            "ONLY when the bundle-token mint answers 409 ambiguous_bundle: if the "
            "mapped bundle is exactly one of the bundles licences returned, the mint "
            "is retried once with &bundle=<key>. An app absent from the map falls "
            "back to an exact app-key == bundle-key match. Empty = no mapping. "
            "Malformed refuses startup."
        ),
    )
    licenses_internal_url: str = Field(
        default="",
        description="licenses base URL for /internal routes (bundle-token mint)",
    )
    bundle_minter_client_id: str = Field(
        default="",
        description=(
            "Keycloak secure-apps client holding ONLY realm role bundle-token-minter "
            "(worker-management-bundle-minter)"
        ),
    )
    bundle_minter_client_secret: str = Field(
        default="",
        description="Secret of bundle_minter_client_id (SOPS-encrypted in gitops)",
    )
    # Actor trust (D4): WM's OWN credential for licences
    # ``GET /internal/subjects/{sub}`` (needs service-internal). Empty ⇒ falls
    # back to the caller-identity client; neither set ⇒ a non-delegating
    # licence caller can assert only its own service-account sub as X-Actor-Id.
    actor_lookup_client_id: str = Field(
        default="",
        description="Keycloak secure-apps client with service-internal, for actor membership lookups",
    )
    actor_lookup_client_secret: str = Field(
        default="",
        description="Secret of actor_lookup_client_id (SOPS-encrypted in gitops)",
    )
    # ⛔ There is NO "WM's own inference-router identity" (the retired
    # ``WM_IR_SERVICE_TENANT_ID`` / platform-core fallback). Owner ruling
    # 2026-10-07: WM never calls inference-router on its own behalf; a call
    # without a real end tenant is refused (``IRTenantRequiredError``).
    # Mirror of inference-router's ``AUTH_DELEGATING_SUBJECTS`` (gitops
    # inference-router deployment): licence SUBJECTS of delegating services
    # (e.g. kag) that call WM on behalf of an END tenant they name in
    # ``X-Effective-Tenant-Id`` without an ``X-End-User-Token``. For these, and
    # only these, WM mints/uses the NAMED END tenant's inference-router token
    # (billing goes to the end tenant). CSV of subject uuids; empty ⇒ nobody.
    # ⛔ Scoped to the IR-token decision only; it does NOT widen the owner
    # commercial-key gate or BYO-key entitlement.
    wm_delegating_subjects: str = Field(
        default="",
        validation_alias=AliasChoices("WM_DELEGATING_SUBJECTS"),
        description=(
            "CSV of licence subjects allowed to have WM mint the named end tenant's "
            "inference-router token (mirror of inference-router AUTH_DELEGATING_SUBJECTS)"
        ),
    )
    # worker_management's OWN Keycloak service-account `sub`. Sent as
    # X-Actor-Id on WM-originated work (no inbound actor) — owner ruling: every
    # cost event carries a real actor, the acting service's own sub when no
    # human acted. Must be a canonical non-nil UUID; empty/malformed ⇒ no
    # fallback (the call goes out without X-Actor-Id, as before).
    service_actor_id: str = Field(
        default="",
        validation_alias=AliasChoices("SERVICE_ACTOR_ID"),
        description="worker_management's own service-account sub (X-Actor-Id fallback)",
    )
    inference_router_cache_ttl: int = Field(
        default=300, description="TTL (seconds) for inference-router model cache"
    )
    # Allowlist for a request-supplied ``base_url_override`` (POST/PUT /workers).
    # Comma-separated ``host:port`` (either http or https) or
    # ``scheme://host:port`` entries. ``inference_router_url``'s origin is
    # always added on top. See ``src/security/request_overrides.py``.
    base_url_override_allowlist: str = Field(
        default="inference-router.inference-router.svc.cluster.local:8150",
        validation_alias=AliasChoices(
            "WORKER_MANAGEMENT_BASE_URL_OVERRIDE_ALLOWLIST",
            "BASE_URL_OVERRIDE_ALLOWLIST",
        ),
        description="Origins a worker's base_url_override may point at (comma-separated).",
    )
    # Origins a worker's ``executor_endpoint`` (remote_claude / remote_codex) may
    # name. RemoteContainerExecutor POSTs there WITH this service's OAuth2
    # bearer, so it is an allowlist, never free text. ``coding_service_url``'s
    # origin is always added on top. Measured on prod 2026-09-22: 445 worker rows
    # carry the key and every value is null (all resolve CODING_SERVICE_URL =
    # the coding-gateway Service below).
    executor_endpoint_allowlist: str = Field(
        default="coding-gateway.coding-gateway.svc.cluster.local:8687",
        validation_alias=AliasChoices(
            "WORKER_MANAGEMENT_EXECUTOR_ENDPOINT_ALLOWLIST",
            "EXECUTOR_ENDPOINT_ALLOWLIST",
        ),
        description="Origins a worker's executor_endpoint may point at (comma-separated).",
    )
    # Per-tenant BYO endpoints for ``PUT /tenant-config/llm`` ``baseUrl``, on top
    # of ``base_url_override_allowlist``. Comma-separated
    # ``<tenant_uuid>=<scheme://host:port>`` entries; operator-set, never by the
    # tenant. See ``check_tenant_base_url`` in src/security/request_overrides.py.
    tenant_base_url_allowlist: str = Field(
        default="",
        validation_alias=AliasChoices(
            "WORKER_MANAGEMENT_TENANT_BASE_URL_ALLOWLIST",
            "TENANT_BASE_URL_ALLOWLIST",
        ),
        description="Per-tenant BYO LLM endpoints (tenant_uuid=origin, comma-separated).",
    )
    # Owner-infrastructure hosts a tenant BYO endpoint may NEVER name, on top of
    # the built-in rules in ``owner_infra_reason`` (non-global IPs incl. the
    # the tailnet CGNAT range, single-label names, the Kubernetes Service, private
    # internal, local and tailnet DNS namespaces, every worker/executor allowlist
    # host). Comma-separated
    # host names (suffix match) or IP literals: the deployment's own public domain
    # and public host addresses, which DNS reports as global. The addresses are NOT
    # defaulted here — a deployment that owns public IPs lists them itself, via
    # TENANT_BYO_OWNER_INFRA_DENYLIST, so no operator's addresses ship in this source.
    tenant_byo_owner_infra_denylist: str = Field(
        default="nonagentic.ai",
        validation_alias=AliasChoices(
            "WORKER_MANAGEMENT_TENANT_BYO_OWNER_INFRA_DENYLIST",
            "TENANT_BYO_OWNER_INFRA_DENYLIST",
        ),
        description="Owner-infra hosts/IPs a tenant BYO base_url may never name.",
    )
    # ------------------------------------------------------------------
    # TENANT IDENTIFIERS — DEFINED IN GITOPS, NEVER IN THIS SOURCE.
    #
    # ⛔ These are real tenant UUIDs of the production estate. This repository is
    # PUBLIC, so a literal here publishes the tenant map of every workspace we
    # run — who is a platform tenant, which label maps to which id. That is not a
    # credential, but it is estate-internal addressing and it does not belong in
    # a public tree. The values live in the Deployment env (gitops), beside the
    # keycloak realm config that mints them, exactly like
    # ``COMMERCIAL_PLATFORM_TENANT_IDS`` below.
    #
    # Every one is EMPTY by default and every consumer FAILS CLOSED on empty:
    #   * empty ``PLATFORM_TENANT_IDS``   → no tenant is platform (system data
    #     is writable by none of them; the verified SA roles still work)
    #   * empty ``RELEASE_TENANT_IDS``    → no tenant may release workflows
    #   * empty ``TENANT_UUIDS``          → no label resolves; a caller sending
    #     ``X-Tenant-Id: <name>`` gets a clean 400 (never a guessed tenant)
    #   * empty ``DEFAULT_TENANT_ID``     → the trusted-network default tenant
    #     path refuses instead of falling back to a tenant nobody chose
    #
    # ⚠️ Rollout order: set these BEFORE scaling the Deployment above 0. With
    # them unset the service starts and every tenant-scoped surface refuses —
    # restrictive, never permissive.
    platform_tenant_ids: str = Field(
        default="",
        validation_alias="PLATFORM_TENANT_IDS",
        description=(
            "Comma-separated canonical tenant UUIDs of OUR platform workspaces (gitops-managed). "
            "Mirrors fr's PLATFORM_TENANT_IDS: a tenant platform in one service and customer in the "
            "other is the leak this set closes. Empty (default) = none is platform."
        ),
    )
    release_tenant_ids: str = Field(
        default="",
        validation_alias="RELEASE_TENANT_IDS",
        description=(
            "Comma-separated canonical tenant UUIDs allowed to release workflows "
            "(/seed-active, /release, /promote-canary, /discard-canary). MUST be a subset of "
            "PLATFORM_TENANT_IDS; validated at startup. Empty (default) = nobody may release."
        ),
    )
    tenant_uuids: str = Field(
        default="",
        validation_alias="TENANT_UUIDS",
        description=(
            "Comma-separated <label>=<uuid> pairs mirroring what the portal realm mints "
            "(the Keycloak tenant_id user attribute). Empty (default) = no label resolves."
        ),
    )
    system_tenant_id: str = Field(
        default="",
        validation_alias="SYSTEM_TENANT_ID",
        description=(
            "The canonical uuid of the SYSTEM tenant — owner of every system-written prompt. "
            "Empty (default) = system-owned prompt surfaces refuse."
        ),
    )
    default_tenant_id: str = Field(
        default="",
        validation_alias="DEFAULT_TENANT_ID",
        description=(
            "Canonical uuid (or a label from TENANT_UUIDS) the trusted-network path uses when a "
            "caller sends no X-Tenant-Id. Empty (default) = that path refuses."
        ),
    )
    # ------------------------------------------------------------------
    # LOCAL-ONLY LLM POLICY — fail closed, PER TENANT (owner ruling 2026-10-03).
    #
    # ``ModelFactory`` constructs vendor SDK clients DIRECTLY (openai, deepseek,
    # groq, cohere, anthropic, google, xai), so disabling paid providers inside
    # inference-router does not cover this service: ``_create_xai_model`` talks
    # to api.x.ai and never touches IR. ``worker-management-secrets`` mounts a
    # live ``xai-api-key``.
    #
    # ⛔ REPLACED 2026-10-03 (fr#1385): the deployment-wide
    # ``ALLOW_EXTERNAL_LLM_PROVIDERS`` flag is GONE and is no longer read. It was
    # the wrong shape for the owner's ruling ("allow all gitops registered
    # commercial apis for platform workspaces only"): one bit for the whole
    # deployment cannot say "platform yes, customer no". The owner's commercial
    # keys are now reachable ONLY by a tenant that is BOTH in this set AND proven
    # by the request (licence introspection or a verified end-user token —
    # ``src.auth.tenant_proof``). A customer reaches a vendor only with its OWN
    # BYO key, as before.
    #
    # The SET is defined ONCE, in gitops
    # (``clusters/strato/services/workers/deployment.yaml``), never in code.
    # Default EMPTY = fail closed: every tenant is a customer (local-llm + own
    # BYO only). EMPTYING it is therefore also the kill switch for the owner's
    # keys — note it also withdraws the other platform-only privileges (any
    # inference-router model, the X-Provider pin) from those tenants.
    #
    # ONE env name only: ``COMMERCIAL_PLATFORM_TENANT_IDS`` (the name gitops CI
    # guards). No ``WORKER_MANAGEMENT_``-prefixed alias: as the first
    # AliasChoices entry it won even when EMPTY, so a stray prefixed copy would
    # silently override the guarded value.
    commercial_platform_tenant_ids: str = Field(
        default="",
        validation_alias="COMMERCIAL_PLATFORM_TENANT_IDS",
        description=(
            "Comma-separated canonical tenant UUIDs of the owner's PLATFORM workspaces "
            "(gitops-managed). Only these, and only when the request proves the tenant, "
            "may use the platform's commercial LLM keys. Empty (default) = none."
        ),
    )
    # worker_management#336: the fr card lookup a prompt release must pass
    # (src/api/prompts/release_card.py). Both are set by gitops#4411 on the
    # Deployment. Either empty => every CANARY/PUBLISHED move is 503
    # prompt_release_card_unverifiable (fail-closed), never a silent pass.
    fr_card_lookup_url: str = Field(
        default="",
        validation_alias="FR_CARD_LOOKUP_URL",
        description=(
            "fr backend BASE url (in-cluster, e.g. http://fr-backend.example.com:8888); "
            "WM appends /api/internal/cards/{card_id}/platform-owner"
        ),
    )
    fr_wm_card_lookup_token: str = Field(
        default="",
        validation_alias="FR_WM_CARD_LOOKUP_TOKEN",
        description="Caller token sent as X-Internal-Token to fr's card lookup (SOPS-encrypted in gitops)",
    )
    # worker_management#338: the workflowAutoRollback event fr records
    # (src/api/workflow_routing/routes.py ``_emit_rollback_webhook``). Both are
    # set by gitops#4416 on the Deployment. Either empty => the event is NOT
    # sent and ``wm_workflow_rollback_webhook_total{outcome="unconfigured"}``
    # counts it (the rollback itself still succeeds — the DB swap is the truth).
    fr_workflow_rollback_url: str = Field(
        default="",
        validation_alias="FR_WORKFLOW_ROLLBACK_URL",
        description=(
            "FULL fr receiver url (in-cluster, e.g. "
            "http://fr-backend.example.com:8888/api/internal/workflow-rollback); "
            "used verbatim, nothing is appended"
        ),
    )
    fr_wm_rollback_token: str = Field(
        default="",
        validation_alias="FR_WM_ROLLBACK_TOKEN",
        description="Caller token sent as X-Internal-Token to fr's rollback receiver (SOPS-encrypted in gitops)",
    )
    # Shared metering cost-event stream. BYO LLM calls go direct-to-provider and
    # bypass inference-router's metering, so worker_management emits its own
    # cost_event here (see src/telemetry/cost_events.py). Empty URL => no-op.
    # Must point at the SAME Redis that metering_service consumes from.
    metering_redis_url: str = Field(
        default="",
        description="Redis DSN for the shared metering cost-event stream (empty disables BYO metering)",
    )
    metering_costs_stream: str = Field(
        default="billing:costs",
        description=(
            "Redis stream name for metering cost events. The default must name a "
            "stream metering_service actually consumes: a pod that omits "
            "METERING_COSTS_STREAM still emits, and XADD creates the key on "
            "first write, so a wrong default routes cost events into a stream "
            "nothing reads and loses them silently rather than failing."
        ),
    )
    # httpx transport hardening for the inference-router AsyncOpenAI client.
    # A bare AsyncClient with no retries surfaced transient ambient-mesh
    # handshake blips ("Connection error") as terminal agent-run failures
    # (see 2026-05-16 trust.pdf ingest postmortem). The transport-level
    # retries cover RST / handshake-reset / immediate-close cases without
    # needing application-level retry logic.
    ir_http_connect_timeout_s: float = Field(
        default=10.0, description="Connect timeout for the IR client (seconds)"
    )
    ir_http_read_timeout_s: float = Field(
        default=600.0,
        description=(
            "Read timeout for the IR client (seconds) — must equal or exceed "
            "the slowest provider's worst-case TTFB. Non-stream chat completions "
            "from vllm buffer the entire response, so TTFB = generation time. "
            "At 9 tok/s on Nemotron-3-Super-120B-NVFP4, 4000-token responses "
            "need ~440s; with adapter-load (30s) + queue (60s) headroom, 600s "
            "is the floor. Matches IR's own vllm_timeout."
        ),
    )
    ir_http_write_timeout_s: float = Field(
        default=10.0, description="Write timeout for the IR client (seconds)"
    )
    ir_http_pool_timeout_s: float = Field(
        default=5.0, description="Pool-acquire timeout for the IR client (seconds)"
    )
    ir_http_transport_retries: int = Field(
        default=3,
        description="httpx AsyncHTTPTransport retries on connect/RST for the IR client",
    )
    # ------------------------------------------------------------------
    # CONNECTION-POOL CEILING — stated, not inherited.
    #
    # ⚠️ These three defaults are httpx's own defaults, reproduced ON PURPOSE.
    # This block changes NO behaviour, and that is the point. The IR client was
    # built with no `limits=` at all, so the ceiling was whatever httpx happened
    # to default to, invisible at the call site and absent from every review.
    #
    # It is also the leading suspect for the 2026-08 "Connection error" burst:
    # ~17-24 requests are in flight for a single agent name, straddling
    # max_keepalive_connections=20, and ztunnel counted ~2 L4 connections per
    # routed request -- the signature of a pool evicting and re-dialling
    # constantly. A socket the server has already closed, handed back out of
    # that pool, dies as an EOF-before-response-headers ~1.6s after issue,
    # which is precisely the measured distribution (p50 1.64s, 83.8% under 5s,
    # and zero mass at any configured timeout value).
    #
    # ⛔ DO NOT RAISE THESE AS A FIX YET, and do not read the numbers as an
    # endorsement of them. Raising the ceiling now would mask the symptom
    # before `connection_state` (src/worker/agents/http_diagnostics.py) has
    # said whether the failing sockets were reused at all -- and if the answer
    # turns out to be `new`, the pool was never the cause and a raised ceiling
    # would be permanent unexplained config. Ship the instrument, read it,
    # then change the number with evidence. The trade being made deliberately
    # here is a known-unchanged production behaviour in exchange for one
    # release of ambiguity.
    ir_http_max_connections: int = Field(
        default=100,
        description=(
            "Max total connections in the IR client pool. httpx's own default, "
            "stated explicitly so the ceiling is reviewable rather than inherited."
        ),
    )
    ir_http_max_keepalive_connections: int = Field(
        default=20,
        description=(
            "Max idle keep-alive connections retained by the IR client pool. "
            "httpx's own default; measured in-flight concurrency is 17-24, i.e. "
            "straddling this value, which is why it must be visible."
        ),
    )
    ir_http_keepalive_expiry_s: float = Field(
        default=5.0,
        description=(
            "Seconds an idle keep-alive connection is retained by the IR client. "
            "httpx's own default; a socket the upstream closed first fails as an "
            "EOF before response headers when it is re-offered."
        ),
    )

    # ------------------------------------------------------------------
    # SELF-HOSTED TOKEN CEILING — the time budget, expressed in tokens.
    #
    # ``SAFE_MAX_TOKENS`` (16384, pydantic_ai_agent.py) is a KERNEL-CRASH guard:
    # it keeps max_tokens away from the model's context window so the FP4-MoE
    # crash of 2026-05-13 cannot re-trigger. It says nothing about whether the
    # request can FINISH. These three fields answer that second question, and
    # the two clamps must stay separate — collapsing them into one number means
    # the next person who raises a deadline silently re-arms the kernel crash.
    #
    # ⚠️ WHY THIS EXISTS AT ALL. The ceiling was already DERIVED, in a comment,
    # and nothing enforced it. ``DEFAULT_SELF_HOSTED_MAX_TOKENS``'s block says:
    # "Anything above ~9500 tokens (~825 s) starts crowding 1100 again;
    # re-derive here, not at the call site." That warning was correct and inert.
    #
    # ⛔ CORRECTED 2026-08-25, SAME DAY, BY ME. The text here used to say SEVEN
    # registered workers sit at or above 10000 (redesign-html 16000,
    # ks-conflict-detector/ks-merge-executor 15000,
    # ks-conflict-analyzer/ks-evidence-aggregator/ks-consistency 12000,
    # ks-duplicate-checker 10000) and let that stand next to the incident below
    # as though it were the cause. It is not. I checked the CONFIG and never
    # checked whether those workers RUN:
    #     the six ks-* -> 0.0 executions in 7 DAYS. One dormant tenant
    #       (82d552da…), every row status=created, untouched since 2026-04-29.
    #       The ACTIVE tenants carry the same workers at 8000/4000.
    #     redesign-html -> ZERO executions in the 12h incident window; 7d mean
    #       2485 completion tokens against its 16000 cap; its 5 cap hits were on
    #       08-21, three days early.
    # The configs are real and every one of them did pass
    # ``min(config.max_tokens, SAFE_MAX_TOKENS)`` unchanged, which is the gap
    # this ceiling closes. They are NOT the explanation for the empties below.
    #
    # ⚠️ Two query traps that produced the wrong picture, both worth knowing:
    #   ``workers`` holds ONE ROW PER TENANT per worker_name, so counting rows
    #   inflates "how many workers"; select DISTINCT ON (tenant_id, worker_name)
    #   ordered by updated_at. And ``operational_status`` reads "offline" on
    #   rows whose agent executes 35x/hour — it is not a liveness signal. Only
    #   ``agent_executions_total`` is.
    #
    # THE COST, measured. Over 2026-08-24T16:10Z -> 2026-08-25T04:01Z the
    # inference-router logged 83 ``upstream_empty_content`` rows on pgx-local:
    # ``finish_reason: "length"`` on all 83, ``content_len: 0``,
    # ``completion_tokens`` 12000 on 59 of them, and ``latency_ms`` up to
    # 1,050,715 — 17.5 minutes for one call that returned nothing. Mean upstream
    # latency for ``outcome="empty"`` was 752.6 s against 30.0 s for
    # ``outcome="ok"``: the empties cost roughly as much GPU wall-clock as every
    # real answer combined.
    #
    # ⛔ AND NOTHING COUNTED IT. ``agent_max_tokens_cap_hit_total`` — the counter
    # built for exactly this event (#172, "a truncated completion must be
    # countable, not grep-able") — measured ZERO increase over the same 24 h.
    # ⚠️ Precision, because the first version of this line overstated it: that
    # counter is not globally dead. Over 7 DAYS it shows 5 hits, all
    # redesign-html, all on 08-21. It is blind to THIS class of event
    # specifically — the router-absorbed one — not to truncation in general.
    # It cannot see this case by construction: the router *raises* on a blank
    # completion and fails the chain over (``router_service.py:1612-1650``), so
    # this service never receives the truncated response its detector inspects.
    #
    # THE ARITHMETIC. The binding rung is the INNERMOST one, and it is not in
    # this repo: pgx-local's per-provider httpx timeout, 1100 s
    # (gitops-production/.../inference-router/providers-configmap.yaml). The
    # chain deadline (1200 s) and this service's read timeout (1200 s) both sit
    # OUTSIDE it, so neither binds first.
    #
    #   ceiling = floor(provider_timeout_s x decode_tokens_per_s x headroom)
    #           = floor(1100 x 11.55 x 0.95) = 12069  (12069.75 truncated)
    #
    # ⛔ HEADROOM WAS 0.75 (ceiling 9528) FOR ABOUT AN HOUR AND THAT WAS WRONG.
    # 0.75 was an a-priori guess for prefill + queue + network. The evidence
    # says it is too tight, and the evidence was sitting in the incident rows
    # the whole time: the runaway calls carry latency_ms up to 1,050,715 for
    # completion_tokens=12000, i.e. a full 12000-token decode completed
    # END-TO-END in ~1042s — INSIDE the 1100s rung, prefill and queue included.
    # It was not cut off by any timeout; it returned finish_reason="length".
    # So 12000 is affordable on today's pgx and 9528 blocked something that
    # demonstrably fits.
    #
    # ⛔ AND IT WAS NOT A HARMLESS BLOCK. The caller is
    # ``requirements_management``'s ``principle_consistency`` agent, which sets
    # 12000 as a PER-MESSAGE override from its own
    # ``src/agents/config.py`` — a value that repo sized against measurement:
    #     p99 of completion_tokens actually generated = 5,540 (20-run sweep,
    #       4 distinct production prompts). 12000 = 2.17x that tail.
    #     cap 6000 (1.08x) MEASURED TO FAIL in production: 87 of 346 replies
    #       cut mid-JSON (25.1%) + 21 empty (6.1%).
    #     cap 12000: 0 cut, 0 empty over n=62.
    # A run whose reasoning overruns the cap emits NOTHING usable, so cutting
    # their cap re-creates the exact defect this ceiling exists to reduce.
    # Their file also states "12000 stays clear of both [16384 ceilings], so the
    # value configured here is the value that actually runs" — a 9528 ceiling
    # silently made that sentence false, from a repo they do not read.
    #
    # 0.95 blocks what is genuinely impossible and permits what is merely
    # expensive:
    #     12000 -> ~1039s   PERMITTED (measured to complete in 1042s)
    #     15000 -> ~1299s   blocked, exceeds the 1100s rung
    #     16000 -> ~1385s   blocked, exceeds it by 26%
    # No live caller uses 15000/16000 — the only configs carrying them have 0.0
    # executions in 7 days — so this clamps nothing that currently runs. That is
    # the intended state: a guard sized to catch the impossible, not to
    # second-guess another team's measured tuning.
    #
    # ⚠️ THE REAL CONFLICT IS NOT RESOLVED HERE AND IS NOT OURS TO RESOLVE.
    # requirements_management sized 12000 against a MEDIAN 63.2 tok/s measured
    # on a different provider. pgx today decodes at 11.55 tok/s — 5.5x slower.
    # At that rate a full 12000-token decode costs ~17 minutes of shared GPU and
    # returns nothing usable when it truncates. Their number is right for the
    # output tail and expensive for this fleet, and BOTH are true. The owner
    # decision (lower the cap, split the batch further, or serve a faster model)
    # belongs to requirements_management; this ceiling only stops the
    # arithmetically impossible and makes the rest countable.
    self_hosted_decode_tokens_per_s: float = Field(
        default=11.55,
        gt=0,
        description=(
            "Measured TOTAL wall-clock decode rate of the self-hosted backend "
            "(tokens/second), including prefill and queue. 11.55 = "
            "request_generation_tokens_sum / e2e_request_latency_seconds_sum "
            "over n=12,933 requests on pgx's Qwen3.8-27B (2026-08-24). Use the "
            "TOTAL rate, not the inter-token rate — anything measuring elapsed "
            "time sees this one."
        ),
    )
    self_hosted_provider_timeout_s: float = Field(
        default=1100.0,
        gt=0,
        description=(
            "The innermost timeout rung a self-hosted request must finish "
            "inside (seconds). NOT a timeout this service applies — it mirrors "
            "inference-router's per-provider httpx timeout for pgx-local, which "
            "is smaller than both the chain deadline (1200s) and this service's "
            "own read timeout (1200s), and therefore binds first. Keep in step "
            "with providers-configmap.yaml."
        ),
    )
    self_hosted_token_budget_headroom: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of the provider timeout a generation may spend decoding. "
            "The remainder absorbs prefill, the router's concurrency-slot wait "
            "(pgx-local sets max_concurrent: 16) and network time, none of "
            "which are decode but all of which are inside the same timeout. "
            "0.95, not a rounder 0.75: the incident's own rows show a full "
            "12000-token decode completing end-to-end in ~1042s inside the "
            "1100s rung, so the non-decode share is small in practice and a "
            "tighter value blocks budgets that demonstrably fit."
        ),
    )

    @property
    def self_hosted_max_tokens_ceiling(self) -> int:
        """Largest max_tokens a self-hosted request can plausibly finish within.

        Derived, never configured directly: a hand-set ceiling drifts silently
        away from the three quantities that determine it. See the block comment
        above for the measurement behind each one.
        """
        return int(
            self.self_hosted_provider_timeout_s
            * self.self_hosted_decode_tokens_per_s
            * self.self_hosted_token_budget_headroom
        )

    # ------------------------------------------------------------------
    # Agent-run wall-clock budget (asyncio.timeout around agent.run())
    # ------------------------------------------------------------------
    # These were module-level constants in pydantic_ai_agent.py; hoisted here so a
    # re-baseline is an env-var change rather than an image rebuild. VALUES ARE
    # UNCHANGED (600.0 / 900.0) — this change only makes them configurable.
    #
    # LOAD-BEARING INVARIANT, enforced in model_post_init below:
    #     ir_http_read_timeout_s <= min(agent_run_timeout_s, agent_run_timeout_mcp_s)
    #
    # The agent cap starts its clock BEFORE connect, so when it is >= the SDK's
    # read budget it always fires first and the SDK never gets to spend a retry.
    # That is the ONLY reason the OpenAI SDK's default max_retries=2 has not been
    # multiplying our GPU load: today 600.0 == 600.0 and the retry window is
    # zero-width. Invert the ordering and every timeout silently becomes three
    # decodes. The explicit max_retries=0 on the clients is the durable fix; this
    # invariant is defence in depth for any client we add and forget to pin.
    agent_run_timeout_s: float = Field(
        default=600.0,
        description=(
            "Wall-clock budget (seconds) for a single agent.run() without MCP "
            "servers. Must be >= ir_http_read_timeout_s. Sized in 2026-05 for "
            "Nemotron-3 at 9 tok/s; the current Gemma-4-26B-A4B FP8 runs ~56 "
            "tok/s and worker-expand p99 is 152.6s / max 252.2s over 45 days, "
            "so this is ~4x the observed max and is a re-baseline candidate — "
            "NOT re-baselined here."
        ),
    )
    agent_run_timeout_mcp_s: float = Field(
        default=900.0,
        description=(
            "Wall-clock budget (seconds) for a single agent.run() WITH MCP "
            "servers — extra headroom for tool round-trips. Must be >= "
            "ir_http_read_timeout_s."
        ),
    )
    # Application-level bounded retry for the WM LLM message stage. Sits ON TOP
    # of the httpx transport retries (ir_http_transport_retries — connect/RST only).
    #
    # CORRECTED 2026-08-08: this comment used to claim the OpenAI SDK's
    # max_retries=2 covered "429/5xx only". That was FALSE. openai 2.41.1
    # `_base_client.py:1653` catches `httpx.TimeoutException` and, with
    # `remaining_retries > 0`, sleeps and `continue`s — i.e. it re-sends a
    # request that already burned its full read budget. The bare `except
    # Exception` at :1670 does the same for everything else. So the SDK default
    # turns ONE timed-out decode into THREE against the very provider whose
    # saturation caused the timeout — positive feedback on our own GPUs.
    # WM therefore constructs every self-hosted-provider client with an explicit
    # `max_retries=0` (see pydantic_ai_agent.py); retry policy is the caller's,
    # and lives in the bounded exp-backoff + jitter recreate loop these fields gate.
    # That loop remains the only thing covering a fully-established-then-dropped
    # "Connection error." on the recreate path.
    llm_retry_max_attempts: int = Field(
        default=3,
        description=(
            "Total worker.run attempts for the WM LLM stage on a transient "
            "connection/MCP error, INCLUDING the primary call (3 = 1 primary + 2 "
            "recreate-retries). 1 disables app-level retry. Each attempt is bounded "
            "by ir_http_read_timeout_s (600s)."
        ),
    )
    llm_retry_base_delay_s: float = Field(
        default=0.5,
        description=(
            "Base backoff (seconds) before the first LLM recreate-retry; doubles "
            "each attempt (0.5s, 1.0s, ...). Negligible vs the 600s read budget."
        ),
    )
    llm_retry_max_delay_s: float = Field(
        default=8.0,
        description="Ceiling (seconds) for a single LLM retry backoff after growth+jitter.",
    )
    llm_retry_jitter_ratio: float = Field(
        default=0.25,
        description=(
            "Symmetric +/- jitter fraction on each LLM retry backoff (0.25 = +/-25%) "
            "to de-correlate concurrent workers. 0.0 disables jitter."
        ),
    )
    llm_retry_deadline_s: float = Field(
        default=600.0,
        gt=0,
        description=(
            "Wall-clock budget (seconds) for the LLM recreate-retry PHASE, measured "
            "from the moment the ladder is entered. A new attempt only STARTS while "
            "elapsed < this; an attempt already running is bounded separately by "
            "agent_run_timeout_s / agent_run_timeout_mcp_s. Exists because "
            "llm_retry_max_attempts alone bounds the COUNT, not the TIME: at the "
            "values live in clusters/strato/services/workers/deployment.yaml "
            "(LLM_RETRY_MAX_ATTEMPTS=7, BASE_DELAY_S=2.0, MAX_DELAY_S=30.0) the "
            "backoffs are 2+4+8+16+30+30 = 90s and the ladder's ceiling is "
            "7 x 600s + 90s = 4290s (non-MCP), or 7 x 900s + 90s = 6390s for an MCP "
            "worker, and nothing in the loop "
            "could stop it short of that. With this deadline the ceiling for the "
            "whole request becomes: primary attempt (<= agent cap) + deadline + one "
            "straggler attempt that started just under it (<= agent cap) = "
            "600 + 600 + 600 = 1800s for a non-MCP worker at the defaults. "
            "NOTE the ceiling is NOT independent of the agent cap: an MCP worker "
            "(agent_run_timeout_mcp_s=900) still reaches 900 + 600 + 900 = 2400s, so "
            "a caller with a tighter budget must lower the agent cap too — no value "
            "of this field alone can make two 900s attempts fit under 1800s."
        ),
    )
    # inference-router answers ``X-IR-Error-Classification: queue_rejected`` when
    # no provider slot is free. That is BUSY, not broken: the recreate ladder must
    # not run (a recreated worker hits the same full queue) and the caller gets a
    # 503 + Retry-After. This is the fallback Retry-After when IR sends none.
    queue_rejected_retry_after_s: int = Field(
        default=30,
        ge=1,
        description=(
            "Retry-After (seconds) returned to the caller on an IR queue_rejected "
            "failure when inference-router's own response carries no Retry-After."
        ),
    )
    # Default per-node timeout for the pydantic-graph InferenceProviderNode
    # when ``framework_config.timeout`` is absent. The legacy default of 30s
    # killed slow-but-valid responses on Nemotron-120B (~9 tok/s) while IR
    # was still mid-keepalive-retry on a working upstream. 240s gives IR's
    # ~210s worst-case retry budget (read 120s × 1 retry + queue 60s +
    # adapter-load 30s) headroom before WM declares the provider failed.
    inference_provider_node_default_timeout_s: int = Field(
        default=600,
        description="Default timeout (seconds) for an InferenceProviderNode "
        "when framework_config.timeout is unset. Aligned with the LLM timeout "
        "ladder: >= the inference-router read budget (600s) so a slow-but-working "
        "vLLM call is not cut off at this graph node before the router/pgx finish "
        "(was 240s — an inverted hop below the downstream budget). A node that "
        "needs a shorter cap sets framework_config.timeout explicitly.",
    )
    mlx_instance_mini1_url: str = Field(
        default="", description="MLX inference instance URL for mini1 routing label"
    )
    mlx_instance_mini2_url: str = Field(
        default="", description="MLX inference instance URL for mini2 routing label"
    )
    mlx_instance_ring_url: str = Field(
        default="", description="MLX inference instance URL for ring routing label"
    )
    mlx_model_cache_ttl: int = Field(default=300, description="TTL (seconds) for MLX model cache")
    vllm_url: str = Field(default="", description="Base URL for in-cluster vLLM (OpenAI-compat)")
    vlm_local_url: str = Field(default="", description="Base URL for the vision-only VLM backend")
    wm_adapter_manager_url: str = Field(
        default="",
        description=(
            "Base URL for the LoRA adapter-manager sidecar. Optional: when empty the "
            "load-adapter node reports ADAPTER_MANAGER_NOT_CONFIGURED without any "
            "HTTP call. Must never be a DIRECT_GPU_HOSTS host (e.g. a raw vLLM host)."
        ),
    )
    coding_service_url: str = Field(
        default="",
        description="Base URL for coding-gateway (RemoteContainerExecutor)",
    )

    # ------------------------------------------------------------------
    # Agent-host launchd services (infra)
    # ------------------------------------------------------------------
    agent_host_mini1_url: str = Field(
        default="", description="agent-host launchd instance on mini1"
    )
    agent_host_mini2_url: str = Field(
        default="", description="agent-host launchd instance on mini2"
    )
    agent_host_cache_ttl: int = Field(
        default=60, description="TTL (seconds) for agent-host health cache"
    )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    otlp_metrics_endpoint: str = Field(
        default="",
        description="OTLP HTTP metrics endpoint (e.g. http://grafana-agent:4318/v1/metrics)",
    )

    # The DEDICATED Prometheus scrape port. 8887 (the API port) + 1000, the same
    # offset inference-router (8150→9150) and coding-security (8693→9693) use.
    #
    # ⛔ NEVER THE API PORT. The scrape relax — a port-level PeerAuthentication
    # PERMISSIVE plus the allow-prometheus-scrape NetworkPolicy ipBlock — is
    # PORT-scoped, not path-scoped. Putting /metrics on 8887 and relaxing 8887
    # would expose the whole Worker Management API (including
    # POST /workers/{worker_name}/message, which AUTO-CREATES workers) to every
    # pod in the cluster CIDR. Metrics get their own listener and 8887 stays
    # STRICT and closed. See src/telemetry/metrics_server.py.
    #
    # ⚠️ Changing this value is a THREE-PLACE change and they must land
    # together: this default, the Deployment containerPort + Service port, and
    # BOTH the port-level PeerAuthentication and the allow-prometheus-scrape
    # NetworkPolicy in gitops. Change one and the target simply goes down.
    metrics_port: int = Field(
        default=9887,
        gt=0,
        description=(
            "Port for the DEDICATED plaintext Prometheus listener. Must never be "
            "the API port: the scrape relax is port-scoped and the API port "
            "serves authenticated, worker-creating traffic."
        ),
    )

    # ------------------------------------------------------------------
    # worker_states retention
    # ------------------------------------------------------------------
    # See src/services/worker_states_retention.py for the measurements these
    # defaults were derived from. Summary: the table is 3088 MB / 988,948 rows
    # and 73% of it is TOASTed JSONB; the bytes sit in the 30-60 day band, not
    # in the oldest rows, and `processing` alone carries 2.1 GB.
    worker_state_retention_enabled: bool = Field(
        default=True,
        description=(
            "Run the periodic worker_states retention sweep. Set false to "
            "disarm it without a redeploy of the code."
        ),
    )
    worker_state_retention_days: int = Field(
        default=30,
        ge=1,
        description=(
            "Keep every worker_states row younger than this. 30d is the "
            "shortest window that outlives Loki's ~16d retention (the stated "
            "reason this table is the durable record) and matches the 30d "
            "window the _DETERMINISTIC_FAILURE_MARKERS analysis used."
        ),
    )
    worker_state_analytics_retention_days: int = Field(
        default=180,
        ge=1,
        description=(
            "Keep terminal rows (completed/error) this long -- they are the "
            "numerator and denominator of the rescue-rate query. The 90-180d "
            "band costs only 458 MB, so six months is nearly free."
        ),
    )
    worker_state_retention_min_per_worker: int = Field(
        default=10,
        ge=1,
        description=(
            "Never delete the newest N rows of a (tenant_id, worker_id), at "
            "any age. Protects get_latest_worker_state, which derives a "
            "worker's CURRENT status from its newest row of any type."
        ),
    )
    worker_state_retention_batch_size: int = Field(
        default=1000,
        ge=1,
        description="Rows per DELETE statement, so no batch holds a long lock.",
    )
    worker_state_retention_max_batches_per_pass: int = Field(
        default=200,
        ge=1,
        description=(
            "Upper bound on one pass. A backlog larger than this is carried to "
            "the next pass instead of extending one transaction indefinitely."
        ),
    )
    worker_state_retention_interval_seconds: int = Field(
        default=3600,
        ge=1,
        description="Seconds between retention passes.",
    )

    # ------------------------------------------------------------------
    # tenant_llm_config_audit retention (WM-5) — src/services/tenant_llm_audit_retention.py
    # ------------------------------------------------------------------
    tenant_llm_audit_retention_enabled: bool = Field(
        default=True,
        description="Run the daily tenant_llm_config_audit prune. False disarms it.",
    )
    tenant_llm_audit_retention_days: int = Field(
        default=400,
        ge=30,
        description=(
            "Delete customer-visible BYO-config audit rows older than this. 400d "
            "keeps a full year plus slack; the floor stops a typo wiping the history."
        ),
    )
    tenant_llm_audit_retention_batch_size: int = Field(
        default=1000, ge=1, description="Audit rows per DELETE batch."
    )
    tenant_llm_audit_retention_max_batches_per_pass: int = Field(
        default=100, ge=1, description="Upper bound on batches per audit prune pass."
    )
    tenant_llm_audit_retention_interval_seconds: int = Field(
        default=86400, ge=60, description="Seconds between audit prune passes."
    )

    # ------------------------------------------------------------------
    # Prompt version lifecycle (migration 031, P4 contract — twin of
    # procedure_management D4 R9/R10)
    # ------------------------------------------------------------------
    prompts_read_requires_published: bool = Field(
        default=False,
        description=(
            "ONE flag for two dark paths. (1) An unpinned GET /prompts/{id} "
            "resolves to the HIGHEST PUBLISHED version (off: the most recently "
            "activated PUBLISHED version — today's single 'active' row). "
            "(2) A /workers/{name}/message request carrying a prompt_versions "
            "stamp that names the worker's own config_revision prompt runs that "
            "exact version (off: the stamp is ignored). Default false; flag-off "
            "is byte-identical to main. Exposed on /health."
        ),
    )

    # ------------------------------------------------------------------
    # Runtime prompt source (migration 034, U6): which text a PLATFORM worker
    # runs — its config.system_prompt, or its bound PUBLISHED prompt version.
    # ------------------------------------------------------------------
    wm_prompt_source: Literal["legacy", "published"] = Field(
        default="legacy",
        description=(
            "legacy: a worker runs workers.config.system_prompt (today). published: "
            "a PLATFORM-tenant worker with a worker_prompt_binding row runs the "
            "HIGHEST PUBLISHED version of its bound prompt, overriding the config "
            "text (which n8n ensure-nodes keep rewriting). Customer tenants always "
            "run legacy, and their prompt_versions pins are ignored in either mode "
            "(metadata.prompt_pin='tenant_forbidden'). Precedence per call: "
            "request.system_prompt > pin > "
            "published > config. Any resolver failure falls back to config."
        ),
    )
    wm_prompt_canary_routable: bool = Field(
        default=False,
        description=(
            "With wm_prompt_source=published: whether a prompt_versions pin naming a "
            "CANARY version may run it. False: the pin is held, the resolved "
            "PUBLISHED version runs, metadata.prompt_pin='canary_held' and "
            "wm_prompt_canary_held_total counts it. CANARY is never auto-selected."
        ),
    )
    wm_prompt_resolve_ttl_s: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Seconds a resolved (tenant, worker) -> published prompt is cached per "
            "pod. A release reaches every pod within this bound. 0 disables the cache."
        ),
    )
    wm_prompt_drift_interval_s: float = Field(
        default=300.0,
        ge=0.0,
        description=(
            "Seconds between refreshes of wm_prompt_binding_drift (#314): whether a "
            "bound platform worker's config.system_prompt still matches its highest "
            "PUBLISHED version. Computed in a background task, never per message, and "
            "in EITHER wm_prompt_source mode so the flip is observable before it "
            "happens. 0 disables the refresher."
        ),
    )

    # ------------------------------------------------------------------
    # Workflow routing pointers (worker_management#230 follow-up)
    # ------------------------------------------------------------------
    workflow_routing_pointers_enabled: bool = Field(
        default=False,
        description=(
            "Gates the routes that CREATE workflow_routing_pointer state: "
            "POST /api/workflow-routing/promote (legacy), /seed-active and "
            "/release (G-15; release arms a canary). /promote-canary and "
            "/discard-canary only mutate an existing row and are NOT gated. "
            "Default false: the 5-min "
            "n8n timer that used to watch promoted rows for rollback "
            "(canary-workflow-rollback-watch) is retired, and no new pointer "
            "may be created until an event-driven rollback verdict routine "
            "exists to replace it. /rollback, /extend and /clear-prior only "
            "mutate rows that already exist, so they are NOT gated by this "
            "flag — with 0 rows live in production, they already 404 and "
            "gating them would risk 403ing the still-active "
            "improve::approval-callback workflow (calls /extend and "
            "rollback) ahead of its need. Set WORKFLOW_ROUTING_POINTERS_ENABLED "
            "only together with that verdict routine."
        ),
    )

    # ------------------------------------------------------------------
    # Workflow release (migration 036, G-15): the RELEASABLE SET.
    # ------------------------------------------------------------------
    # ⛔ FAIL CLOSED: unset, empty or unparsable ⇒ NOTHING is releasable and
    # /seed-active, /release and /promote-canary answer 403
    # {"error":"asset_not_releasable"}. Derived from movies-09's frozen prereg
    # list (sha256), the same list as ncn's AA_WF_PROMOTE_ALLOWLIST — never
    # hand-edited (phase-10 additions, G-15 DESIGN ruling).
    wm_wf_release_allowlist: str = Field(
        default="",
        description=(
            "JSON list of n8n workflow NAMES whose pointer may be seeded, released "
            "and promoted. An asset_key is releasable only as 'activity:<name>' with "
            "<name> in this list. Empty/unset/invalid = nothing releasable."
        ),
    )

    def _validate_timeout_ordering(self) -> None:
        """Refuse a config where the IR read budget outlives the agent cap.

        The agent's ``asyncio.timeout`` starts before connect, so while it is
        >= the downstream read budget it always fires first and the OpenAI
        SDK's retry loop never gets a turn. Inverting that ordering re-opens
        the SDK retry window on read timeouts (``openai/_base_client.py:1653``
        retries ``httpx.TimeoutException``), turning one timed-out decode into
        several against an already-saturated provider. Fail at startup rather
        than discover it as a GPU-load spiral under production load.
        """
        cap = min(self.agent_run_timeout_s, self.agent_run_timeout_mcp_s)
        if self.ir_http_read_timeout_s > cap:
            raise RuntimeError(
                "worker_management settings violate the agent-timeout invariant: "
                f"ir_http_read_timeout_s ({self.ir_http_read_timeout_s}) must be <= "
                f"min(agent_run_timeout_s={self.agent_run_timeout_s}, "
                f"agent_run_timeout_mcp_s={self.agent_run_timeout_mcp_s}) = {cap}. "
                "A read budget that outlives the agent cap re-opens the OpenAI SDK's "
                "retry-on-timeout path and multiplies GPU load on a saturated provider. "
                "Raise the agent cap or lower the read timeout."
            )

    def _validate_no_direct_gpu_hosts(self) -> None:
        """Refuse an adapter-manager URL that targets a GPU backend directly.

        Enforced in every environment: a direct vLLM URL is a policy
        violation, not an environment-specific misconfiguration.
        """
        if is_direct_gpu_url(self.wm_adapter_manager_url):
            raise RuntimeError(
                "worker_management settings violate the no-direct-GPU policy: "
                f"wm_adapter_manager_url={self.wm_adapter_manager_url!r} targets a "
                f"direct GPU host ({sorted(DIRECT_GPU_HOSTS)}). Applications must not "
                "call vLLM directly — route model traffic through inference-router "
                "(INFERENCE_ROUTER_URL), or leave WM_ADAPTER_MANAGER_URL unset."
            )

    def _validate_project_id_caller_licences(self) -> None:
        """Refuse a malformed licence→caller map, but ONLY while enforcing.

        With ``require_project_id_header`` off the map only names counter
        series, and a typo must never take WM down: the middleware warns once
        and treats it as empty (every licence then reads ``licence:<id>``).
        With the flag on, a wrong label hides which caller is being refused,
        so the config is refused at startup instead.
        """
        if not self.require_project_id_header:
            return
        try:
            parse_project_id_caller_licences(self.project_id_caller_licences)
        except ValueError as exc:
            raise RuntimeError(
                f"worker_management settings: WORKER_MANAGEMENT_PROJECT_ID_CALLER_LICENCES {exc}"
            ) from None

    def _validate_ir_bundle_by_app(self) -> None:
        """Refuse a malformed ``WM_IR_BUNDLE_BY_APP`` at startup (never at call time)."""
        try:
            parse_ir_bundle_by_app(self.wm_ir_bundle_by_app)
        except ValueError as exc:
            raise RuntimeError(f"worker_management settings: WM_IR_BUNDLE_BY_APP {exc}") from None

    def _validate_tenant_sets(self) -> None:
        """Refuse malformed tenant env values, and a release set outside the platform set.

        ⛔ REPLACES the import-time ``RuntimeError`` that used to guard
        ``RELEASE_TENANT_IDS <= PLATFORM_TENANT_IDS`` in
        ``src/api/workflow_routing/routes.py``. Both sets are now env-supplied, so
        an import-time assertion cannot see them — the invariant has to be checked
        where the values are actually read, at startup, and refuse to boot.
        """
        try:
            platform = parse_tenant_id_csv(self.platform_tenant_ids, env_name="PLATFORM_TENANT_IDS")
            release = parse_tenant_id_csv(self.release_tenant_ids, env_name="RELEASE_TENANT_IDS")
            label_map = parse_tenant_label_map(self.tenant_uuids, env_name="TENANT_UUIDS")
            if self.system_tenant_id:
                parse_tenant_id_csv(self.system_tenant_id, env_name="SYSTEM_TENANT_ID")
            if self.default_tenant_id and self.default_tenant_id not in label_map:
                parse_tenant_id_csv(self.default_tenant_id, env_name="DEFAULT_TENANT_ID")
        except ValueError as exc:
            raise RuntimeError(f"worker_management settings: {exc}") from None
        outside = sorted(release - platform)
        if outside:
            raise RuntimeError(
                "worker_management settings: RELEASE_TENANT_IDS must be a subset of "
                f"PLATFORM_TENANT_IDS (narrower, never wider); not in the platform set: {outside}"
            )

    # ------------------------------------------------------------------
    # Production fail-fast
    # ------------------------------------------------------------------
    _REQUIRED_IN_PRODUCTION: tuple[str, ...] = (
        "database_url",
        "license_introspect_url",
        "minio_endpoint",
        "inference_router_url",
        "mlx_instance_mini1_url",
        "mlx_instance_mini2_url",
        "mlx_instance_ring_url",
        "vllm_url",
        "vlm_local_url",
        "coding_service_url",
        "agent_host_mini1_url",
        "agent_host_mini2_url",
        "otlp_metrics_endpoint",
    )

    def model_post_init(self, __context: object) -> None:
        """Fail fast at startup on an unsatisfiable config.

        Two separate checks:
          1. the timeout-ordering invariant, enforced in EVERY environment — it
             is an internal-consistency property of the numbers themselves, and
             a dev/staging value that violates it is exactly what gets promoted;
          2. required infra env vars, production-only.
        """
        self._validate_timeout_ordering()
        self._validate_no_direct_gpu_hosts()
        self._validate_project_id_caller_licences()
        self._validate_ir_bundle_by_app()
        self._validate_tenant_sets()
        if self.environment != "production":
            return
        missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                "worker_management settings missing required fields in production: "
                f"{missing}. Set them via env vars on the Deployment "
                "(see gitops-production/clusters/production/services/workers/deployment.yaml)."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear cached Settings — used by tests that mutate env vars."""
    get_settings.cache_clear()
