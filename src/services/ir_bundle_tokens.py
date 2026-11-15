"""Short-lived per-tenant inference-router bundle tokens (plan §2.4, ruling 1.0.3).

Owner ruling: *use short-lived tokens, no pasting.* Instead of a long-lived
licence token pasted into a Secret, worker_management mints a ≤60-min token for
the tenant it is calling inference-router on behalf of, from licenses::

    POST {LICENSES_INTERNAL_URL}/internal/tenants/{t}/bundle-token
         ?app=inference-router&ttl_minutes=60
    Authorization: Bearer <SA JWT of worker-management-bundle-minter>
    201 {token, expires_at, tenant_id, bundle, app, license_id, token_id}

The route admits realm role ``bundle-token-minter`` ONLY (licenses
``require_bundle_token_minter``), so the SA comes from a DEDICATED client that
holds that role and nothing else — never the shared caller-identity client.

Rules, each pinned by ``tests/test_ir_bundle_tokens.py``:

* **In memory only**, per tenant, until ``expires_at - 5 min``. Never persisted,
  never shared between instances.
* **Never logged.** Log lines name the tenant, the outcome and the HTTP status —
  never the token, the SA JWT or the client secret.
* **Single-flight per tenant**: N concurrent calls for a cold tenant cause ONE
  mint; different tenants do not wait on each other.
* **409 ``ambiguous_bundle``** (several held bundles reach inference-router with
  different licences): the ``bundles`` list is logged; if the calling app's
  bundle (``WM_IR_BUNDLE_BY_APP[X-App-Id]``, else the app key itself) is exactly
  one of them, the mint is retried ONCE with ``&bundle=<key>`` and cached per
  ``(tenant, bundle)``. Otherwise no header, as for any refusal. Never a guess.
* **Fail soft**: 404/409/5xx/transport/SA failure ⇒ ``None`` (no header), counted
  in ``wm_ir_bundle_token_total{outcome}``. A refusal is negative-cached for
  :data:`NEGATIVE_TTL_S` so a tenant without a bundle cannot turn every IR call
  into a licenses call (licenses logs every refusal at WARNING).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NoReturn

import httpx
from prometheus_client import Counter

from src.api.error_handlers import APIError
from src.settings import get_settings, parse_ir_bundle_by_app

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping

logger = logging.getLogger(__name__)

IR_APP_KEY = "inference-router"
TTL_MINUTES = 60
# Re-mint this long before the token's own expiry (ruling: expiry - 5 min).
REFRESH_LEAD_S = 300
# How long a refusal / failure is remembered before the next mint attempt.
NEGATIVE_TTL_S = 60
_SA_REFRESH_LEAD_S = 30
_SA_DEFAULT_EXPIRY_S = 300
_HTTP_TIMEOUT_S = 5.0

BUNDLE_TOKEN_TOTAL = Counter(
    "wm_ir_bundle_token_total",
    "Per-tenant inference-router bundle-token mint attempts by outcome "
    "(ok|not_found|conflict|forbidden|server_error|transport_error|bad_response|"
    "sa_error|not_configured|error|tenant_not_proven|placeholder_dropped|"
    "ok_disambiguated|ambiguous_unresolved). Cache hits "
    "are not counted; tenant_not_proven and placeholder_dropped are counted on every "
    "affected call (no licenses call is made).",
    ["outcome"],
)

TENANTLESS_IR_CALL_REFUSED_TOTAL = Counter(
    "wm_ir_tenantless_call_refused_total",
    "worker_management -> inference-router calls REFUSED before sending because no "
    "real end tenant was known (call=completion|listing). worker_management never "
    "calls inference-router on its own behalf; any increment is a caller/defect to fix.",
    ["call", "worker"],
)


def preinit_tenantless_refusal_metrics() -> None:
    """Materialise ``wm_ir_tenantless_call_refused_total`` at 0 for both calls.

    The label legs are the ones the code actually emits: a completion refusal
    with no known worker, and the listing refusal (always WM's own name).
    Called at module import AND from the app lifespan, which imports this
    module eagerly -- it used to be imported lazily on first use, so the series
    did not exist until the first refusal and an ``increase()`` rule saw nothing.
    """
    TENANTLESS_IR_CALL_REFUSED_TOTAL.labels(call="completion", worker="<unknown>")
    TENANTLESS_IR_CALL_REFUSED_TOTAL.labels(call="listing", worker=WM_SERVICE_WORKER_NAME)


class IRTenantRequiredError(APIError):
    """A WM -> inference-router call had no real end tenant, so it was not made.

    OWNER RULING (2026-10-07): worker_management NEVER calls inference-router on
    its own behalf -- every call acts for a calling app that has a real end
    tenant. There is no WM / platform service tenant to fall back to (the old
    ``WM_IR_SERVICE_TENANT_ID`` fallback billed tenantless listings to
    ``platform-core`` and was removed). Refused loudly: typed error, ERROR log,
    ``wm_ir_tenantless_call_refused_total``.
    """

    def __init__(self, call: str, worker_name: str = "") -> None:
        self.call = call
        self.worker_name = worker_name
        super().__init__(
            status_code=403,
            detail=(
                f"inference-router {call} refused: no end tenant for this request. "
                "worker_management only calls inference-router on behalf of a caller "
                "with a real tenant."
            ),
            error_code="END_TENANT_REQUIRED",
        )


class IRTenantNotProvenError(APIError):
    """The request NAMED a tenant it did not prove; the IR call was not made."""

    def __init__(self, tenant_id: str, reason: str) -> None:
        self.tenant_id = tenant_id
        self.reason = reason
        super().__init__(
            status_code=403,
            detail=(
                f"inference-router call refused: tenant {tenant_id} was not proven by this "
                f"request ({reason})"
            ),
            error_code="END_TENANT_NOT_PROVEN",
        )


def refuse_tenantless_ir_call(call: str, *, worker_name: str = "", url: str = "") -> NoReturn:
    """Count, log at ERROR and raise :class:`IRTenantRequiredError`. Never sends."""
    worker = worker_name or "<unknown>"
    TENANTLESS_IR_CALL_REFUSED_TOTAL.labels(call=call, worker=worker).inc()
    logger.error(
        "wm_ir_tenantless_call_refused call=%s worker=%s url=%s -- no real end tenant; "
        "the inference-router call was NOT made. There is no WM/platform tenant "
        "fallback: fix the caller so it presents its end tenant.",
        call,
        worker,
        url or "<none>",
    )
    raise IRTenantRequiredError(call, worker_name)


@dataclass
class _Entry:
    token: str | None
    valid_until: float  # wall-clock seconds; the entry is used while now < this
    # Set ONLY on a tenant entry recorded from a 409 ``ambiguous_bundle``: the
    # bundle keys licences said reach inference-router for this tenant.
    bundles: tuple[str, ...] | None = None


def _parse_expires_at(raw: str) -> float:
    """licenses returns ``expires_at.isoformat()``; a naive value is UTC."""
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _bundle_target(app_key: str, by_app: Mapping[str, str]) -> str:
    """The ONE bundle key ``app_key`` may mint under on an ambiguous tenant.

    ``WM_IR_BUNDLE_BY_APP[app_key]`` when configured, else ``app_key`` itself
    (an exact app-key == bundle-key match). ``""`` when no app is known. This is
    a deterministic lookup, never a guess: it only picks a bundle if licences
    lists it, exactly once, among the ambiguous ones (:func:`_pick_bundle`).
    """
    app = (app_key or "").strip()
    if not app:
        return ""
    return by_app.get(app, app)


def _pick_bundle(target: str, bundles: tuple[str, ...]) -> str | None:
    if target and bundles.count(target) == 1:
        return target
    return None


def _entry_key(tenant_id: str, bundle: str | None) -> str:
    return tenant_id if bundle is None else f"{tenant_id}|bundle={bundle}"


class BundleTokenCache:
    """Per-tenant bundle-token cache with per-tenant single-flight minting.

    A tenant whose held bundles reach inference-router with DIFFERENT licences
    gets a 409 ``ambiguous_bundle`` from licences. That 409 is remembered for
    :data:`NEGATIVE_TTL_S` with its ``bundles`` list; a calling app whose bundle
    (``WM_IR_BUNDLE_BY_APP``, else its own key) is exactly one of them re-mints
    once with ``&bundle=<key>``, cached per ``(tenant, bundle)`` like any success.
    No match ⇒ ``None`` (today's behaviour). Never a guess: billing attributes
    on the licence the bundle resolves to.
    """

    def __init__(
        self,
        *,
        licenses_url: str,
        token_url: str,
        client_id: str,
        client_secret: str,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
        bundle_by_app: Mapping[str, str] | None = None,
    ) -> None:
        self._licenses_url = licenses_url.rstrip("/")
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._transport = transport
        self._clock = clock
        self._bundle_by_app: dict[str, str] = dict(bundle_by_app or {})
        self._entries: dict[str, _Entry] = {}
        self._sa: _Entry | None = None
        # Keyed by (event loop, tenant): an asyncio.Lock binds to one loop.
        self._locks: dict[tuple[int, str], asyncio.Lock] = {}

    def _fresh(self, key: str) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is not None and self._clock() < entry.valid_until:
            return entry
        return None

    def _lookup(self, tenant_id: str, target: str) -> tuple[bool, str | None]:
        """``(hit, token)`` from the cache alone; ``hit=False`` means a mint is due."""
        entry = self._fresh(tenant_id)
        if entry is not None and entry.bundles is None:
            return True, entry.token
        if target:
            # A token already minted under this app's bundle stays usable until its
            # own horizon, even once the 60 s ambiguity record has lapsed.
            pinned = self._fresh(_entry_key(tenant_id, target))
            if pinned is not None and pinned.token:
                return True, pinned.token
        if entry is None:
            return False, None
        bundle = _pick_bundle(target, entry.bundles or ())
        if bundle is None:
            return True, None
        pinned = self._fresh(_entry_key(tenant_id, bundle))
        if pinned is not None:
            return True, pinned.token
        return False, None

    async def get(self, tenant_id: str, app_key: str = "") -> str | None:
        """Return a valid tenant token, or ``None`` when none can be had.

        ``app_key`` is the calling app (inbound ``X-App-Id``); it matters ONLY
        when licences answers 409 ``ambiguous_bundle`` for this tenant.
        """
        target = _bundle_target(app_key, self._bundle_by_app)
        hit, token = self._lookup(tenant_id, target)
        if hit:
            return token
        key = (id(asyncio.get_running_loop()), tenant_id)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            hit, token = self._lookup(tenant_id, target)
            if hit:
                return token
            entry = self._fresh(tenant_id)
            if entry is None:
                token = await self._mint(tenant_id, None)
                entry = self._fresh(tenant_id)
                if entry is None or entry.bundles is None:
                    return token
                bundle = _pick_bundle(target, entry.bundles)
                if bundle is None:
                    BUNDLE_TOKEN_TOTAL.labels(outcome="ambiguous_unresolved").inc()
                    logger.warning(
                        "wm_ir_bundle_token outcome=ambiguous_unresolved tenant=%s app=%s "
                        "target_bundle=%s bundles=%s -- the calling app maps to none of "
                        "the held bundles (set WM_IR_BUNDLE_BY_APP); no tenant "
                        "Authorization header on this inference-router call",
                        tenant_id,
                        app_key or "<none>",
                        target or "<none>",
                        ",".join(entry.bundles),
                    )
                    return None
            else:
                bundle = _pick_bundle(target, entry.bundles or ())
                if bundle is None:
                    return None
            return await self._mint(tenant_id, bundle, app_key=app_key)

    def _client(self) -> httpx.AsyncClient:
        # A short-lived client per mint (≈ once per tenant per 55 min): never
        # bound to one event loop, nothing to close at shutdown.
        return httpx.AsyncClient(transport=self._transport, timeout=_HTTP_TIMEOUT_S)

    async def _sa_token(self, client: httpx.AsyncClient) -> str | None:
        now = self._clock()
        if self._sa is not None and self._sa.token and now < self._sa.valid_until:
            return self._sa.token
        try:
            resp = await client.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            )
        except httpx.HTTPError as e:
            logger.warning("wm_ir_bundle_token sa_mint_failed error=%s", type(e).__name__)
            return None
        if resp.status_code != httpx.codes.OK:
            logger.warning("wm_ir_bundle_token sa_mint_failed status=%s", resp.status_code)
            return None
        try:
            payload = resp.json()
            access = payload.get("access_token")
            expires_in = int(payload.get("expires_in") or _SA_DEFAULT_EXPIRY_S)
        except ValueError, TypeError, AttributeError:
            access = None
            expires_in = 0
        if not access:
            logger.warning("wm_ir_bundle_token sa_mint_failed reason=no_access_token")
            return None
        self._sa = _Entry(token=access, valid_until=now + max(expires_in - _SA_REFRESH_LEAD_S, 1))
        return access

    def _refuse(
        self,
        tenant_id: str,
        outcome: str,
        status: int | None,
        error: str,
        *,
        bundle: str | None = None,
        bundles: tuple[str, ...] | None = None,
    ) -> None:
        BUNDLE_TOKEN_TOTAL.labels(outcome=outcome).inc()
        self._entries[_entry_key(tenant_id, bundle)] = _Entry(
            token=None, valid_until=self._clock() + NEGATIVE_TTL_S, bundles=bundles
        )
        logger.warning(
            "wm_ir_bundle_token outcome=%s tenant=%s status=%s error=%s bundle=%s bundles=%s "
            "-- no tenant Authorization header on inference-router calls for this tenant "
            "for %ss",
            outcome,
            tenant_id,
            status,
            error,
            bundle or "-",
            ",".join(bundles) if bundles else "-",
            NEGATIVE_TTL_S,
        )

    async def _mint(self, tenant_id: str, bundle: str | None, *, app_key: str = "") -> str | None:
        params = {"app": IR_APP_KEY, "ttl_minutes": str(TTL_MINUTES)}
        if bundle is not None:
            params["bundle"] = bundle
        async with self._client() as client:
            sa = await self._sa_token(client)
            if sa is None:
                self._refuse(tenant_id, "sa_error", None, "sa_token_unavailable", bundle=bundle)
                return None
            url = f"{self._licenses_url}/internal/tenants/{tenant_id}/bundle-token"
            try:
                resp = await client.post(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {sa}"},
                )
            except httpx.HTTPError as e:
                self._refuse(tenant_id, "transport_error", None, type(e).__name__, bundle=bundle)
                return None
        status = resp.status_code
        if status not in (httpx.codes.CREATED, httpx.codes.OK):
            error = ""
            ambiguous: tuple[str, ...] | None = None
            try:
                detail = resp.json().get("detail")
                if isinstance(detail, dict):
                    error = str(detail.get("error") or "")
                    raw = detail.get("bundles")
                    if (
                        error == "ambiguous_bundle"
                        and bundle is None
                        and isinstance(raw, list)
                        and all(isinstance(b, str) for b in raw)
                    ):
                        ambiguous = tuple(raw)
            except ValueError, AttributeError:
                pass
            if status == httpx.codes.NOT_FOUND:
                outcome = "not_found"
            elif status == httpx.codes.CONFLICT:
                outcome = "conflict"
            elif status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
                outcome = "forbidden"
                # A 401/403 is OUR SA, not the tenant: drop it so the next try re-mints.
                self._sa = None
            elif status >= httpx.codes.INTERNAL_SERVER_ERROR:
                outcome = "server_error"
            else:
                outcome = "bad_response"
            if outcome != "conflict":
                ambiguous = None
            self._refuse(tenant_id, outcome, status, error, bundle=bundle, bundles=ambiguous)
            return None
        try:
            body = resp.json()
            token = body["token"]
            expires_at = _parse_expires_at(body["expires_at"])
            token_id = body.get("token_id")
        except ValueError, KeyError, TypeError:
            self._refuse(tenant_id, "bad_response", status, "unparseable_body", bundle=bundle)
            return None
        if not isinstance(token, str) or not token:
            self._refuse(tenant_id, "bad_response", status, "empty_token", bundle=bundle)
            return None
        BUNDLE_TOKEN_TOTAL.labels(outcome="ok" if bundle is None else "ok_disambiguated").inc()
        key = _entry_key(tenant_id, bundle)
        valid_until = expires_at - REFRESH_LEAD_S
        if valid_until > self._clock():
            self._entries[key] = _Entry(token=token, valid_until=valid_until)
        else:
            # licenses caps expiry at the licence's own: still valid, not cacheable.
            self._entries.pop(key, None)
        logger.info(
            "wm_ir_bundle_token outcome=%s tenant=%s bundle=%s app=%s token_id=%s expires_at=%s",
            "ok" if bundle is None else "ok_disambiguated",
            tenant_id,
            bundle or "-",
            app_key or "-",
            token_id,
            datetime.fromtimestamp(expires_at, tz=UTC).isoformat(),
        )
        return token


# Process-wide state in one mutable holder (no ``global`` rebinding).
_state: dict[str, object] = {
    "cache": None,
    "not_configured_logged": False,
}


def reset_bundle_token_cache() -> None:
    """Drop the process-wide cache (tests; settings changes)."""
    _state["cache"] = None
    _state["not_configured_logged"] = False


def _get_cache() -> BundleTokenCache | None:
    cache = _state["cache"]
    if isinstance(cache, BundleTokenCache):
        return cache
    s = get_settings()
    if not (
        s.licenses_internal_url
        and s.keycloak_token_url
        and s.bundle_minter_client_id
        and s.bundle_minter_client_secret
    ):
        return None
    cache = BundleTokenCache(
        licenses_url=s.licenses_internal_url,
        token_url=s.keycloak_token_url,
        client_id=s.bundle_minter_client_id,
        client_secret=s.bundle_minter_client_secret,
        bundle_by_app=parse_ir_bundle_by_app(s.wm_ir_bundle_by_app),
    )
    _state["cache"] = cache
    return cache


async def tenant_bearer_token(tenant_id: str) -> str | None:
    """The tenant's IR bundle token, or ``None`` (flag off, unconfigured, refused)."""
    if not get_settings().wm_ir_bundle_tokens:
        return None
    cache = _get_cache()
    if cache is None:
        BUNDLE_TOKEN_TOTAL.labels(outcome="not_configured").inc()
        if not _state["not_configured_logged"]:
            _state["not_configured_logged"] = True
            logger.error(
                "wm_ir_bundle_token outcome=not_configured -- WM_IR_BUNDLE_TOKENS is on "
                "but LICENSES_INTERNAL_URL / KEYCLOAK_TOKEN_URL / BUNDLE_MINTER_CLIENT_ID "
                "/ BUNDLE_MINTER_CLIENT_SECRET is missing; no tenant tokens are sent"
            )
        return None
    # The calling app (inbound X-App-Id, relayed verbatim, never defaulted) only
    # matters when licences answers 409 ambiguous_bundle for this tenant.
    from src.api.identity_context import get_app_id_from_context

    return await cache.get(tenant_id, get_app_id_from_context())


_DEFAULT_PORTS = {"http": 80, "https": 443}
# inference-router's coercion sentinel for an unresolvable tenant. Never a
# tenant a token can be minted for.
_NIL_TENANT_ID = "00000000-0000-0000-0000-000000000000"


def is_ir_origin(url: str) -> bool:
    """True only when ``url`` is on the ``INFERENCE_ROUTER_URL`` ORIGIN.

    Scheme, host and effective port must all match: ``http://ir:8150`` is not
    ``https://ir:9999``. Hostname alone would hand the tenant token to any
    other listener on the IR host. A malformed URL never matches.
    """
    from urllib.parse import urlsplit

    try:
        ir = urlsplit(get_settings().inference_router_url or "")
        ir_scheme = (ir.scheme or "").lower()
        ir_host = (ir.hostname or "").lower()
        ir_port = ir.port or _DEFAULT_PORTS.get(ir_scheme)
        req = urlsplit(url or "")
        req_scheme = (req.scheme or "").lower()
        req_host = (req.hostname or "").lower()
        req_port = req.port or _DEFAULT_PORTS.get(req_scheme)
    except ValueError:
        return False
    if not ir_host or ir_port is None:
        return False
    return req_scheme == ir_scheme and req_host == ir_host and req_port == ir_port


def _mint_refusal_reason(tenant_id: str) -> str | None:
    """``None`` when THIS request PROVED ``tenant_id``; else why it did not.

    The predicate itself moved to :func:`src.auth.tenant_proof.tenant_proof_refusal`
    (fr#1385) so the owner-commercial-key gate on ``/providers`` and
    ``/chat/completions`` uses the SAME proof as this mint, rather than a copy
    that could drift. Behaviour is unchanged.
    """
    from src.auth.tenant_proof import tenant_proof_refusal

    # ir_token=True: a delegating service on WM_DELEGATING_SUBJECTS proves the
    # END tenant it names (mirror of the router's AUTH_DELEGATING_SUBJECTS).
    return tenant_proof_refusal(tenant_id, ir_token=True)


async def ir_authorization(url: str, tenant_id: str | None) -> str | None:
    """``"Bearer <tenant bundle token>"`` for a WM→IR call to ``url``, else ``None``.

    The ONE decision every WM→inference-router call takes (completions via the
    identity hook, ``/api/models`` listing via the registry):

    * flag ``WM_IR_BUNDLE_TOKENS`` off ⇒ ``None`` (today's wire, unchanged);
    * no tenant or the nil sentinel ⇒ ``None``;
    * a commercial-PLATFORM tenant is treated like every other tenant (owner
      ruling 2026-10-06: every tenant that calls gets its own licence and token,
      so every call is billed to its own tenant). It used to be skipped, which
      sent the SDK placeholder ``Bearer internal`` — a 401 once IR enforces. The
      skip only ever WITHHELD a token; the proof check below still gates the
      mint, so an asserted/default platform tenant still gets nothing;
    * ``url`` not on the exact IR origin ⇒ ``None`` (never leak the token);
    * the tenant was not PROVEN by this request (:func:`_mint_refusal_reason`;
      an allowlisted delegating service naming its END tenant counts as proven)
      ⇒ RAISES :class:`IRTenantNotProvenError` (the call is not made), counted
      as ``outcome="tenant_not_proven"``, ERROR-logged; licenses is never called. Checked BEFORE the cache, so a token minted earlier for a
      proven request is never handed to a later asserting one;
    * any error while getting the token ⇒ ``None``, counted as
      ``outcome="error"``, logged by exception TYPE only (its message could
      embed a token). The token step never fails the call.
    """
    if not get_settings().wm_ir_bundle_tokens:
        return None
    if not tenant_id or tenant_id == _NIL_TENANT_ID:
        return None
    if not is_ir_origin(url):
        return None
    try:
        refusal = _mint_refusal_reason(tenant_id)
    except Exception as exc:  # the guard itself failing is a refusal, never a mint
        refusal = f"guard_error:{type(exc).__name__}"
    if refusal is not None:
        BUNDLE_TOKEN_TOTAL.labels(outcome="tenant_not_proven").inc()
        # ⛔ REFUSED, not forwarded tokenless. A caller naming a tenant it did
        # not prove (and that is not an allowlisted delegating service) gets no
        # inference-router call at all: typed error + ERROR + metric.
        logger.error(
            "wm_ir_bundle_token outcome=tenant_not_proven tenant=%s reason=%s -- "
            "inference-router call REFUSED; the caller must prove the tenant with a "
            "licence token, a verified X-End-User-Token, or be a delegating service "
            "on WM_DELEGATING_SUBJECTS",
            tenant_id,
            refusal,
        )
        raise IRTenantNotProvenError(tenant_id, refusal)
    try:
        token = await tenant_bearer_token(tenant_id)
    except Exception as exc:  # fail soft by contract
        BUNDLE_TOKEN_TOTAL.labels(outcome="error").inc()
        logger.warning(
            "wm_ir_bundle_token outcome=error tenant=%s error_type=%s -- no tenant token sent",
            tenant_id,
            type(exc).__name__,
        )
        return None
    return f"Bearer {token}" if token else None


#: The ``X-Worker-Name`` WM stamps on inference-router calls that belong to no
#: worker (model listing / validation). Same value as the cost-event
#: ``service_key`` and the ``cost_source`` this service writes, so IR, billing
#: and WM name the caller identically. IR buckets it as ``other`` until it is on
#: inference-router's ``_ALLOWED_WORKERS``.
WM_SERVICE_WORKER_NAME = "worker-management"

#: worker_management's own key in the licences app registry. Sent as
#: ``X-App-Id`` on calls WM makes on its own behalf (model listing).
WM_APP_KEY = "workers"

preinit_tenantless_refusal_metrics()


# ---------------------------------------------------------------------------
# No placeholder bearer ever reaches inference-router.
#
# The OpenAI SDK refuses to build without an api_key, so every WM→IR client is
# constructed with ``IR_CLIENT_API_KEY = "internal"`` and the identity hook  # pragma: allowlist secret
# swaps in the tenant's bundle token. When no token can be had (unproven tenant,
# mint refused, flag off) the placeholder used to go out as-is: a credential
# shaped header with no authority, which an enforcing router 401s while making
# the call LOOK licensed in any header dump. Dropping it makes the call honestly
# tokenless; the reason was already logged at ERROR by whoever withheld it.
# ---------------------------------------------------------------------------

#: Bearer values that are SDK placeholders, never a licence (case-insensitive).
PLACEHOLDER_BEARER_KEYS: frozenset[str] = frozenset({"", "internal", "empty", "none", "null"})


def is_placeholder_bearer(value: str | None) -> bool:
    """True for ``Bearer internal`` / ``Bearer EMPTY`` / a bare ``Bearer``."""
    if value is None:
        return False
    scheme, _, credential = value.strip().partition(" ")
    if scheme.lower() != "bearer":
        return False
    return credential.strip().lower() in PLACEHOLDER_BEARER_KEYS


def drop_placeholder_bearer(
    url: str, headers: MutableMapping[str, str], *, worker_name: str = ""
) -> bool:
    """Remove a placeholder ``Authorization`` from a request to the IR origin.

    Returns True when one was dropped (counted ``placeholder_dropped``, logged at
    ERROR). Other origins are untouched: the placeholder is only meaningless —
    and only ours to police — on inference-router.
    """
    if not is_ir_origin(url):
        return False
    if not is_placeholder_bearer(headers.get("Authorization")):
        return False
    del headers["Authorization"]
    BUNDLE_TOKEN_TOTAL.labels(outcome="placeholder_dropped").inc()
    logger.error(
        "wm_ir_bundle_token outcome=placeholder_dropped worker=%s -- no licence token for "
        "this inference-router call; the SDK placeholder bearer was removed and the call "
        "goes out tokenless (refused once IR enforces introspection)",
        worker_name or "<unknown>",
    )
    return True


async def ir_listing_headers(url: str, tenant_id: str | None) -> dict[str, str]:
    """Identity + licence headers for a WM→IR model-listing GET to ``url``.

    The ONE builder for every listing (``/api/models`` via the registry,
    ``/v1/models`` via the providers ``vllm`` fetcher, whose ``VLLM_URL`` is the
    IR origin in prod). The CALLER's tenant presents ITS token
    (:func:`ir_authorization`). ⛔ No tenant (or the nil sentinel) ⇒ REFUSED via
    :func:`refuse_tenantless_ir_call` -- there is no WM/platform service tenant
    to list as. ``X-Worker-Name`` is always set.
    """
    if not tenant_id or tenant_id == _NIL_TENANT_ID:
        refuse_tenantless_ir_call("listing", worker_name=WM_SERVICE_WORKER_NAME, url=url)
    # A model listing is WM's OWN call (no upstream LLM intent), so it carries
    # WM's own licences-registry app key, not the inbound caller's.
    headers: dict[str, str] = {
        "X-Worker-Name": WM_SERVICE_WORKER_NAME,
        "X-Job-Id": WM_SERVICE_WORKER_NAME,
        "X-App-Id": WM_APP_KEY,
        "X-Tenant-Id": tenant_id,
    }
    # Relay the inbound project verbatim when present; never invented. (A
    # listing is not metered spend, so a miss is not counted here.)
    from src.api.identity_context import get_effective_actor_id, get_project_id_from_context

    if project_id := get_project_id_from_context():
        headers["X-Project-Id"] = project_id
    # WHO ACTED: the inbound actor unchanged, else WM's own service account.
    if actor_id := get_effective_actor_id():
        headers["X-Actor-Id"] = actor_id
    auth = await ir_authorization(url, tenant_id)
    if auth:
        headers["Authorization"] = auth
    return headers
