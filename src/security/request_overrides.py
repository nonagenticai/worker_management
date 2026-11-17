"""Request-supplied overrides that must never reach code execution or an open relay.

Two fields on the worker API used to be trusted verbatim:

* ``output_validator`` — Python source that ``PydanticAIAgent`` ran through
  ``exec()`` with full builtins. The only "validation" was ``compile()`` plus
  the substring ``"def validate"``, so any caller that could reach
  ``POST /workers``, ``PUT /workers/{name}`` or ``POST /workers/{name}/message``
  had remote code execution in this pod. Measured on prod 2026-09-22: 0 of 793
  worker rows carry a non-empty validator, so the capability is removed
  outright rather than sandboxed. There is no flag to turn it back on.

* ``base_url_override`` — the provider client's base URL, checked only for
  being non-empty. It made this service a relay to any host a caller named,
  carrying this service's provider credentials. Measured on prod 2026-09-22:
  every non-empty value is the inference-router Service URL (288 rows), so an
  allowlist keyed on scheme+host+port refuses nothing that is live.

Both refusals are TYPED (``PydanticCustomError`` type / ``code``) so a caller's
error-reader can branch on them instead of parsing prose.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx2 as httpx
from pydantic_core import PydanticCustomError

logger = logging.getLogger(__name__)

OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED = "output_validator_code_not_supported"
BASE_URL_OVERRIDE_NOT_ALLOWED = "base_url_override_not_allowed"

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


class OutputValidatorCodeNotSupportedError(ValueError):
    """A non-empty ``output_validator`` reached a place that would have run it."""

    code = OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED


# ── output_validator ──────────────────────────────────────────────────────────


def has_validator_code(value: Any) -> bool:
    """True when ``value`` carries validator source (anything but None / blank)."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def reject_output_validator(value: Any) -> None:
    """Pydantic field-validator body: ``None`` / ``""`` pass (as None), code is a 422."""
    if has_validator_code(value):
        raise PydanticCustomError(
            OUTPUT_VALIDATOR_CODE_NOT_SUPPORTED,
            "output_validator code is not supported: request-supplied Python is never "
            "executed. Constrain output with output_schema instead.",
        )


def stored_validator_code(worker_data: dict[str, Any] | None) -> bool:
    """True when a resolved worker row carries validator source (column or config)."""
    if not isinstance(worker_data, dict):
        return False
    if has_validator_code(worker_data.get("output_validator")):
        return True
    config = worker_data.get("config")
    return isinstance(config, dict) and has_validator_code(config.get("output_validator"))


# ── base_url_override ─────────────────────────────────────────────────────────


def _not_allowed(reason: str) -> PydanticCustomError:
    return PydanticCustomError(
        BASE_URL_OVERRIDE_NOT_ALLOWED,
        "base_url_override is not allowed: {reason}",
        {"reason": reason},
    )


def _entry_key(entry: str) -> tuple[frozenset[str], str, int] | None:
    """Parse one allowlist entry (``host:port`` or ``scheme://host:port``)."""
    entry = entry.strip()
    if not entry:
        return None
    has_scheme = "://" in entry
    parts = urlsplit(entry if has_scheme else f"http://{entry}")
    host = (parts.hostname or "").lower()
    if not host:
        return None
    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:
        return None
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
        if port is None:
            return None
    schemes = frozenset({scheme}) if has_scheme else _ALLOWED_SCHEMES
    return schemes, host, port


def allowed_base_url_origins() -> list[tuple[frozenset[str], str, int]]:
    """The effective allowlist: the setting plus the configured inference-router URL."""
    from src.settings import get_settings

    settings = get_settings()
    raw = [e for e in (settings.base_url_override_allowlist or "").split(",")]
    if settings.inference_router_url:
        raw.append(settings.inference_router_url)
    keys = []
    for entry in raw:
        key = _entry_key(entry)
        if key is not None:
            keys.append(key)
    return keys


def _check_allowlisted_url(
    value: str,
    not_allowed: Any,
    allowed: list[tuple[frozenset[str], str, int]],
) -> str:
    """Parse ``value`` and return it normalised, or raise ``not_allowed(reason)``.

    ``value`` must parse to an ``http``/``https`` URL with no userinfo, no
    query/fragment, and a (scheme, lowercased host, port) that is in ``allowed``.
    IP literals are refused unless the literal itself is listed. Shared by every
    caller-supplied URL that this service would itself connect to.
    """
    if value != value.strip() or any(ord(c) < 0x21 or c == "\\" for c in value):
        raise not_allowed("whitespace, control characters and backslashes are refused")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise not_allowed("not a parseable URL") from exc
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise not_allowed("scheme must be http or https")
    if "@" in parts.netloc or parts.username is not None or parts.password is not None:
        raise not_allowed("userinfo (user@host) is refused")
    if parts.query or parts.fragment:
        raise not_allowed("query strings and fragments are refused")
    host = (parts.hostname or "").lower()
    if not host:
        raise not_allowed("no host")
    if port is None:
        port = _DEFAULT_PORTS[scheme]

    listed = any(scheme in schemes and host == h and port == p for schemes, h, p in allowed)
    if not listed:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            raise not_allowed(f"{scheme}://{host}:{port} is not in the allowlist") from None
        raise not_allowed("IP-literal hosts are refused unless listed")

    netloc = host if parts.port is None else f"{host}:{parts.port}"
    if ":" in host:  # IPv6 literal that was explicitly listed
        netloc = f"[{host}]" if parts.port is None else f"[{host}]:{parts.port}"
    return urlunsplit((scheme, netloc, parts.path, "", ""))


def normalize_base_url_override(value: str | None) -> str | None:
    """Return ``value`` normalised, or raise a typed 422 if it is not allowlisted.

    ``None`` passes (no override). The empty string keeps its long-standing
    refusal. Everything else must pass :func:`_check_allowlisted_url` against
    :func:`allowed_base_url_origins`.
    """
    if value is None:
        return None
    if value == "":
        raise ValueError("Base URL override cannot be empty string")
    return _check_allowlisted_url(value, _not_allowed, allowed_base_url_origins())


# The in-cluster inference-router Service. Live 2026-10-01: all 187 customer
# ``inference_router`` worker rows carry ``base_url_override`` on this origin
# (``…:8150/api`` or bare ``…:8150``), and gitops ``workers/deployment.yaml``
# sets ``INFERENCE_ROUTER_URL`` to it as well.
INFERENCE_ROUTER_SERVICE_ORIGIN = "inference-router.inference-router.svc.cluster.local:8150"


class _NotInferenceRouter(ValueError):
    pass


def inference_router_base_for(url: str | None) -> str | None:
    """``scheme://host:port/api`` when ``url`` points at THE inference-router, else ``None``.

    "THE inference-router" is deliberately narrow, never the operator-widenable
    ``base_url_override_allowlist``: the in-cluster IR Service origin
    (:data:`INFERENCE_ROUTER_SERVICE_ORIGIN`, either scheme) plus the origin of
    ``settings.inference_router_url``. The same parser as the PUT allowlist
    applies, so userinfo, a query/fragment, control characters or an unlisted
    port/host all return ``None``. The path is discarded and replaced by IR's
    ``/api`` base — the exact URL ``ModelFactory._create_inference_router_model``
    already sends for the live ``…:8150`` and ``…:8150/api`` shapes.
    """
    if not url:
        return None
    from src.settings import get_settings

    raw = [INFERENCE_ROUTER_SERVICE_ORIGIN]
    configured = get_settings().inference_router_url
    if configured:
        raw.append(configured)
    allowed = [k for k in (_entry_key(e) for e in raw) if k is not None]
    try:
        normalised = _check_allowlisted_url(str(url), _NotInferenceRouter, allowed)
    except _NotInferenceRouter:
        return None
    parts = urlsplit(normalised)
    return urlunsplit((parts.scheme, parts.netloc, "/api", "", ""))


# ── graph function_code ───────────────────────────────────────────────────────
#
# A graph ``function`` node's ``data.function_code`` was exec()'d with
# ``__import__`` in its builtins (graph_factory FunctionNode.run and
# nodes/function_node.py), reachable by POST /graphs + an execute call: ``import
# os`` worked. Measured on prod 2026-09-22: 0 of 42 graphs contain a function
# node at all (every node is ``worker``), and the one graph executed in the last
# 30 days (2414 sessions) has none. So code execution is removed, not sandboxed.

FUNCTION_CODE_NOT_SUPPORTED = "function_code_not_supported"


class FunctionCodeNotSupportedError(ValueError):
    """A graph node carrying ``function_code`` reached a place that would have run it."""

    code = FUNCTION_CODE_NOT_SUPPORTED


def has_function_code(data: Any) -> bool:
    """True when a node's ``data`` dict carries function source (anything but None/blank)."""
    if not isinstance(data, dict) or "function_code" not in data:
        return False
    return has_validator_code(data.get("function_code"))


def function_code_error() -> PydanticCustomError:
    """The typed 422 for a node carrying ``function_code``."""
    return PydanticCustomError(
        FUNCTION_CODE_NOT_SUPPORTED,
        "function_code is not supported: request-supplied Python is never executed. "
        "Use a worker node instead.",
    )


# ── mcp_config ────────────────────────────────────────────────────────────────
#
# A request-supplied ``mcp_config`` used to be handed straight to
# ``create_mcp_servers_from_config``: an entry with ``command`` spawned that
# process with this pod's full ``os.environ`` (every secret), an entry with
# ``url`` made this service connect anywhere, and a *string* value was opened as
# a file path. Measured on prod 2026-09-22: 0 of 793 worker rows carry a
# non-null ``mcp_config`` (761 store JSON null, 32 omit it), 0 carry a
# ``command`` or ``url``, and every ``mcp_config_preset`` is null. So:
#
# * stdio (``command``) servers come ONLY from the server-side registry
#   (``mcp_configs/`` via ``mcp_config_preset`` / ``mcp_servers``);
# * a request ``url`` server must be on the same origin allowlist as
#   ``base_url_override`` (#282), and so must an auth ``token_url``;
# * a string value (file path) is refused; presets are named, not pathed.

MCP_CONFIG_NOT_ALLOWED = "mcp_config_not_allowed"

_PRESET_NAME_RE = r"^[A-Za-z0-9_-]{1,64}$"


class McpConfigNotAllowedError(ValueError):
    """A request-supplied MCP config reached a place that would have acted on it."""

    code = MCP_CONFIG_NOT_ALLOWED


def _mcp_not_allowed(reason: str) -> PydanticCustomError:
    return PydanticCustomError(
        MCP_CONFIG_NOT_ALLOWED,
        "mcp_config is not allowed: {reason}",
        {"reason": reason},
    )


def check_request_mcp_config(value: Any) -> dict[str, Any] | None:
    """Validate a request-supplied ``mcp_config``; return it (URLs normalised) or raise.

    ``None`` and ``{}`` pass. Raises :class:`PydanticCustomError`
    (``mcp_config_not_allowed``) for anything that would spawn a process, open a
    file, or connect to a host outside the allowlist.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raise _mcp_not_allowed(
            "a string (file path) is refused; name a server-side preset with "
            "mcp_config_preset or select servers with mcp_servers"
        )
    if not isinstance(value, dict):
        raise _mcp_not_allowed("must be an object")
    if not value:
        return value
    servers = value.get("mcpServers")
    if set(value) != {"mcpServers"} or not isinstance(servers, dict):
        raise _mcp_not_allowed("must be exactly {'mcpServers': {name: {url: ...}}}")
    allowed = allowed_base_url_origins()
    out: dict[str, Any] = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            raise _mcp_not_allowed(f"server {name!r} must be an object")
        for key in ("command", "args", "cwd", "env"):
            if key in cfg:
                raise _mcp_not_allowed(
                    f"server {name!r}: {key!r} (stdio/process) servers are only "
                    "available from the server-side registry"
                )
        url = cfg.get("url")
        if not isinstance(url, str) or not url:
            raise _mcp_not_allowed(f"server {name!r} must have a 'url'")

        def _bad(reason: str, _n: Any = name) -> PydanticCustomError:
            return _mcp_not_allowed(f"server {_n!r}: {reason}")

        new_cfg = dict(cfg)
        new_cfg["url"] = _check_allowlisted_url(url, _bad, allowed)
        auth = cfg.get("auth")
        if isinstance(auth, dict) and auth.get("token_url") is not None:
            token_url = auth.get("token_url")
            if not isinstance(token_url, str):
                raise _bad("auth.token_url must be a string")
            new_cfg["auth"] = {
                **auth,
                "token_url": _check_allowlisted_url(token_url, _bad, allowed),
            }
        out[name] = new_cfg
    return {"mcpServers": out}


def check_mcp_config_preset(value: str | None) -> str | None:
    """A preset is a registry NAME, never a path."""
    import re

    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.match(_PRESET_NAME_RE, value):
        raise _mcp_not_allowed("mcp_config_preset must be a registry name ([A-Za-z0-9_-])")
    return value


def assert_mcp_config_allowed(mcp_config: Any, mcp_config_preset: Any = None) -> None:
    """Run-time fail-closed twin of the request validators (stored rows, direct callers)."""
    try:
        check_request_mcp_config(mcp_config)
        check_mcp_config_preset(mcp_config_preset)
    except PydanticCustomError as exc:
        raise McpConfigNotAllowedError(exc.message()) from None


# ── tenant BYO base_url ───────────────────────────────────────────────────────
#
# ``PUT /tenant-config/llm`` stored ``baseUrl`` unvalidated and
# ``llm_config_resolver`` copied it into ``base_url_override`` alongside the
# tenant's decrypted key — or a platform-owned key when ``owner_provided`` —
# bypassing the #282 allowlist and letting a tenant point this service at any
# host (internal ones included). BYO endpoints ARE a product feature (the
# ``openai_compatible`` / ``custom`` presets require one), but measured on prod
# 2026-09-22 the only ``tenant_llm_config`` row (1 of 1) has an empty base_url.
# So an endpoint is allowed when it is on the global allowlist OR on that
# tenant's own entry in ``tenant_base_url_allowlist`` (operator-set,
# ``<tenant_uuid>=<scheme://host:port>`` comma-separated).

TENANT_BASE_URL_NOT_ALLOWED = "tenant_base_url_not_allowed"


class TenantBaseUrlNotAllowedError(ValueError):
    """A tenant BYO ``base_url`` is not on the global or that tenant's allowlist."""

    code = TENANT_BASE_URL_NOT_ALLOWED


# ── owner infrastructure ──────────────────────────────────────────────────────
#
# Validator 3 D3 (2026-10-01): the tenant BYO allowlist used to START from the
# worker ``base_url_override`` allowlist, which always carries the
# inference-router Service. So a customer could PUT ``{provider:
# openai_compatible, model: grok-4-fast, baseUrl:
# an in-cluster inference-router Service URL ending in /v1}`` and the
# BYO OpenAI client (no identity hook) reached inference-router with NO tenant —
# owner hardware and, upstream, owner commercial keys, unattributed and booked
# as the tenant's own BYO spend. A tenant endpoint is by definition OFF the
# platform, so no owner-infra origin is ever a valid one, whoever listed it.

OWNER_INFRA_SUFFIXES: tuple[str, ...] = (
    ".svc.cluster.local",
    ".cluster.local",
    ".svc",
    ".internal",
    ".local",
    ".localhost",
    ".ts.net",
)
_DNS_CACHE_TTL_S = 30.0
_dns_cache: dict[str, tuple[float, tuple[str, ...]]] = {}


def _resolve_host_ips(host: str) -> tuple[str, ...]:
    """Every address ``host`` resolves to (A and AAAA); ``()`` when it does not.

    Module-level so tests stub DNS in one place.
    """
    import socket

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError, UnicodeError:
        return ()
    return tuple(sorted({str(info[4][0]).split("%", 1)[0] for info in infos}))


def _resolved_ips(host: str, *, use_cache: bool) -> tuple[str, ...]:
    import time

    now = time.monotonic()
    if use_cache:
        hit = _dns_cache.get(host)
        if hit is not None and now - hit[0] < _DNS_CACHE_TTL_S:
            return hit[1]
    ips = _resolve_host_ips(host)
    if use_cache:
        _dns_cache[host] = (now, ips)
    return ips


# IPv6 forms that EMBED an IPv4 address (validator 4 N2). Each is checked as
# the IPv6 address AND as the IPv4 it carries, so an estate public IP or a
# private IPv4 cannot be smuggled past the denylist/non-global rules by
# spelling it ``::ffff:203.0.113.11`` (IPv4-mapped), ``64:ff9b::a00:1``
# (NAT64 well-known prefix) or ``::a00:1`` (deprecated IPv4-compatible).
_NAT64_WKP = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")

_NON_GLOBAL_REASON = "a private, loopback, link-local, tailnet or reserved address"


def _embedded_ipv4(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 address carries (mapped / NAT64 / compatible / 6to4)."""
    if not isinstance(ip, ipaddress.IPv6Address):
        return None
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip in _NAT64_WKP or ip in _IPV4_COMPATIBLE:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.sixtofour


def _ip_owner_infra_reason(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    denied: frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address] = frozenset(),
) -> str | None:
    """Why ``ip`` is owner infra, or ``None``.

    The text never contains the address itself: a caller that RESOLVED a name
    must not echo the internal address back to the tenant (validator 4 N5).
    """
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    embedded = _embedded_ipv4(ip)
    if embedded is not None:
        candidates.append(embedded)
    for cand in candidates:
        if cand in denied:
            return "platform infrastructure"
        if not cand.is_global or cand.is_multicast:
            return _NON_GLOBAL_REASON
    if isinstance(ip, ipaddress.IPv6Address) and ip in _IPV4_COMPATIBLE:
        # Deprecated (RFC 4291 §2.5.5.1); never a legitimate vendor endpoint.
        return _NON_GLOBAL_REASON
    return None


def _denied_ips(exact: set[str]) -> frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    out: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for entry in exact:
        try:
            out.add(ipaddress.ip_address(entry.strip("[]")))
        except ValueError:
            continue
    return frozenset(out)


def _configured_owner_hosts() -> tuple[set[str], tuple[str, ...]]:
    """(exact hosts, name suffixes) the operator config names as owner infra."""
    from src.settings import get_settings

    settings = get_settings()
    exact: set[str] = set()
    for key in (*allowed_base_url_origins(), *allowed_executor_origins()):
        exact.add(key[1])
    suffixes: list[str] = []
    for raw in str(getattr(settings, "tenant_byo_owner_infra_denylist", "") or "").split(","):
        entry = raw.strip().lower().rstrip(".")
        if not entry:
            continue
        exact.add(entry)
        try:
            ipaddress.ip_address(entry)
        except ValueError:
            suffixes.append("." + entry)
    return exact, tuple(suffixes)


def owner_infra_reason(host: str, *, resolve: bool = True, use_cache: bool = True) -> str | None:
    """Why ``host`` is owner infrastructure, or ``None`` when it is not.

    Owner infra = the inference-router / coding-gateway / any worker or executor
    allowlist host, ``localhost``, a single-label (in-cluster short) name, the
    in-cluster Kubernetes Service, private-internal, local and tailnet DNS
    namespaces, the operator denylist, or ANY non-global address (private,
    loopback, link-local, the CGNAT/tailnet range, reserved, multicast) —
    as a literal or, with ``resolve``, among the addresses the name resolves
    to. A name that does not resolve is refused (fail-closed): nothing can be
    proven about where it would land.
    """
    host = (host or "").strip().lower().rstrip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host:
        return "no host"
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    exact, suffixes = _configured_owner_hosts()
    if host in exact:
        return f"{host} is platform infrastructure"
    denied = _denied_ips(exact)
    if literal is not None:
        # A literal is the caller's own text, so naming it discloses nothing.
        reason = _ip_owner_infra_reason(literal, denied)
        return f"{host} is {reason}" if reason is not None else None
    if host == "localhost" or "." not in host:
        return f"{host} is a local or in-cluster short name"
    if host.endswith(OWNER_INFRA_SUFFIXES) or host.endswith(suffixes):
        return f"{host} is in a platform-internal namespace"
    if not resolve:
        return None
    ips = _resolved_ips(host, use_cache=use_cache)
    if not ips:
        return f"{host} does not resolve"
    for raw in ips:
        try:
            reason = _ip_owner_infra_reason(ipaddress.ip_address(raw), denied)
        except ValueError:
            return f"{host} resolves to an unparseable address"
        if reason is not None:
            # N5: the internal address a name resolved to is logged for the
            # operator, never returned in the refusal text the tenant sees.
            logger.warning("owner_infra: %s resolves to %s (%s); refused", host, raw, reason)
            return f"{host} resolves to {reason}"
    return None


def _host_of(url_or_host: str | None) -> str:
    value = (url_or_host or "").strip()
    if not value:
        return ""
    if "://" in value:
        try:
            return (urlsplit(value).hostname or "").lower().rstrip(".")
        except ValueError:
            return ""
    return value.lower().rstrip(".")


def _prewarm_one(host: str) -> None:
    import time

    hit = _dns_cache.get(host)
    if hit is not None and time.monotonic() - hit[0] < _DNS_CACHE_TTL_S / 2:
        return
    _dns_cache[host] = (time.monotonic(), _resolve_host_ips(host))


async def prewarm_owner_infra_dns(*urls_or_hosts: str | None) -> None:
    """Resolve the hosts the sync owner-infra checks will need, OFF the event loop.

    Validator 4 N4: :func:`check_tenant_base_url` / :func:`is_owner_infra_url`
    are sync and called from async handlers (PUT, LLM + modality resolve, the
    worker routing belt); their ``socket.getaddrinfo`` used to run ON the event
    loop, so one slow resolver stalled every request in the process. Async
    callers await this first: the blocking lookup runs in a worker thread and
    lands in the 30 s cache, which the sync check then reads. A host refused by
    name/literal (no DNS needed) is skipped. Entries older than half the TTL
    are refreshed so the following sync check cannot find them expired.
    """
    import asyncio

    hosts: list[str] = []
    for item in urls_or_hosts:
        host = _host_of(item)
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if not host or host in hosts:
            continue
        if owner_infra_reason(host, resolve=False) is not None:
            continue
        hosts.append(host)
    for host in hosts:
        await asyncio.to_thread(_prewarm_one, host)


def is_owner_infra_url(url: str | None, *, resolve: bool = True) -> bool:
    """True when ``url``'s host is owner infrastructure (see :func:`owner_infra_reason`)."""
    if not url:
        return False
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return True
    return owner_infra_reason(host, resolve=resolve) is not None


async def refuse_owner_infra_request(request: Any) -> None:
    """httpx ``request`` event hook for clients built on a TENANT BYO base_url.

    httpx runs request hooks on EVERY hop, redirects included (the OpenAI and
    Anthropic SDKs follow redirects), so a tenant endpoint that 307s to an
    internal Service — or a name re-resolved to one after the PUT check — is
    refused before anything is sent. Resolution here is uncached.
    """
    import asyncio

    host = request.url.host
    reason = await asyncio.to_thread(owner_infra_reason, host, use_cache=False)
    if reason is not None:
        raise TenantBaseUrlNotAllowedError(
            f"tenant base_url is not allowed: refusing request to platform infrastructure "
            f"({reason})"
        )


def pinned_tenant_endpoint_ip(host: str) -> str:
    """Resolve ``host`` ONCE, validate every address, return the one to dial.

    WM-6 (validator 4 N1): the request hook resolved the name and httpx then
    resolved it AGAIN at connect, so a name re-pointed between the two (DNS
    rebinding, TTL 0) reached owner infrastructure after passing the check.
    Here the address that is checked is the address that is dialled. Name and
    literal rules are :func:`owner_infra_reason`'s; resolution is uncached; ANY
    non-global or denied address refuses (a mixed answer is not "pick the good
    one"). IPv4 is preferred when both families resolve.

    Raises :class:`TenantBaseUrlNotAllowedError`.
    """

    def _bad(reason: str) -> TenantBaseUrlNotAllowedError:
        return TenantBaseUrlNotAllowedError(
            "tenant base_url is not allowed: refusing request to platform infrastructure "
            f"({reason})"
        )

    host = (host or "").strip().lower().rstrip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    reason = owner_infra_reason(host, resolve=False)
    if reason is not None:
        raise _bad(reason)
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return host  # a literal that passed the literal rules above
    ips = _resolve_host_ips(host)
    if not ips:
        raise _bad(f"{host} does not resolve")
    exact, _suffixes = _configured_owner_hosts()
    denied = _denied_ips(exact)
    parsed: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for raw in ips:
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            raise _bad(f"{host} resolves to an unparseable address") from None
        why = _ip_owner_infra_reason(ip, denied)
        if why is not None:
            logger.warning(
                "owner_infra: %s resolves to %s (%s); refused at connect", host, raw, why
            )
            raise _bad(f"{host} resolves to {why}")
        parsed.append(ip)
    v4 = [ip for ip in parsed if isinstance(ip, ipaddress.IPv4Address)]
    return str((v4 or parsed)[0])


class PinnedTenantEndpointTransport(httpx.AsyncBaseTransport):
    """httpx transport for TENANT endpoints: dial the validated IP, keep the name.

    Every request (redirect hops included — httpx sends each through the
    transport) is resolved once by :func:`pinned_tenant_endpoint_ip`, and the
    connection is made to THAT address. The ``Host`` header is the original
    one, and TLS uses the original name for SNI and certificate verification
    (httpcore's ``sni_hostname`` extension), so a vendor endpoint behaves
    exactly as before while a rebinding answer can never be the one dialled.
    The caller's request object is not mutated (httpx builds redirects from it).
    """

    def __init__(self, **transport_kwargs: Any) -> None:
        self._inner = httpx.AsyncHTTPTransport(**transport_kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Validate-and-pin the host, then send through the inner transport."""
        import asyncio

        host = request.url.host
        ip = await asyncio.to_thread(pinned_tenant_endpoint_ip, host)
        if ip == host.strip("[]").lower():
            return await self._inner.handle_async_request(request)
        extensions = dict(request.extensions)
        if request.url.scheme == "https":
            extensions.setdefault("sni_hostname", host)
        pinned = httpx.Request(
            request.method,
            request.url.copy_with(host=ip),
            headers=request.headers,  # carries the original Host
            stream=request.stream,
            extensions=extensions,
        )
        return await self._inner.handle_async_request(pinned)

    async def aclose(self) -> None:
        """Close the inner connection pool."""
        await self._inner.aclose()


def pinned_tenant_transport(**transport_kwargs: Any) -> PinnedTenantEndpointTransport:
    """A :class:`PinnedTenantEndpointTransport` for ``httpx.AsyncClient(transport=)``."""
    return PinnedTenantEndpointTransport(**transport_kwargs)


def allowed_tenant_base_url_origins(tenant_id: str | None) -> list[tuple[frozenset[str], str, int]]:
    """Operator-allowlisted origins for ``tenant_id``, MINUS every owner-infra origin.

    Global ``base_url_override_allowlist`` entries plus the operator-set
    per-tenant entries — but never the inference-router Service (always on the
    worker allowlist), nor anything else :func:`owner_infra_reason` names
    (validator 3 D3). Filtering is by name and literal only; the resolved-address
    check runs on the concrete URL in :func:`check_tenant_base_url`.
    """
    from src.settings import get_settings

    keys = list(allowed_base_url_origins())
    raw = getattr(get_settings(), "tenant_base_url_allowlist", "") or ""
    want = str(tenant_id or "").strip().lower()
    if want:
        for entry in raw.split(","):
            tid, sep, origin = entry.partition("=")
            if not sep or tid.strip().lower() != want:
                continue
            key = _entry_key(origin)
            if key is not None:
                keys.append(key)
    return [key for key in keys if owner_infra_reason(key[1], resolve=False) is None]


def check_tenant_base_url(tenant_id: str | None, value: str | None) -> str | None:
    """Return a tenant BYO ``base_url`` normalised, or raise :class:`TenantBaseUrlNotAllowedError`.

    ``None``/``""`` pass as ``None`` (provider default endpoint). The URL must be
    allowlisted AND must not be owner infrastructure, by name or by any address
    it resolves to.
    """
    if value is None or value == "":
        return None

    def _bad(reason: str) -> TenantBaseUrlNotAllowedError:
        return TenantBaseUrlNotAllowedError(f"tenant base_url is not allowed: {reason}")

    try:
        host = (urlsplit(value).hostname or "") if isinstance(value, str) else ""
    except ValueError:
        host = ""  # _check_allowlisted_url refuses it with the parse reason
    if host:
        reason = owner_infra_reason(host, resolve=False)
        if reason is not None:
            raise _bad(f"a tenant endpoint may not point at platform infrastructure ({reason})")
    normalised = _check_allowlisted_url(value, _bad, allowed_tenant_base_url_origins(tenant_id))
    reason = owner_infra_reason(urlsplit(normalised).hostname or "")
    if reason is not None:
        raise _bad(f"a tenant endpoint may not point at platform infrastructure ({reason})")
    return normalised


# ── executor_endpoint ─────────────────────────────────────────────────────────
#
# ``WorkerCreateRequest.executor_endpoint`` was free text. RemoteContainerExecutor
# POSTs the agent prompt to it (and GETs /health and /api/containers/...) carrying
# this service's OAuth2 service-to-service bearer, with ``verify=False``: SSRF
# plus service-token exfiltration to any host a caller named. Measured on prod
# 2026-09-22: 445 worker rows carry the key, every value null, so every live run
# resolves CODING_SERVICE_URL (the coding-gateway Service). The allowlist is that
# Service plus the configured ``coding_service_url`` origin.

EXECUTOR_ENDPOINT_NOT_ALLOWED = "executor_endpoint_not_allowed"


class ExecutorEndpointNotAllowedError(ValueError):
    """An executor endpoint is not allowlisted; nothing (and no bearer) is sent."""

    code = EXECUTOR_ENDPOINT_NOT_ALLOWED


def allowed_executor_origins() -> list[tuple[frozenset[str], str, int]]:
    """``executor_endpoint_allowlist`` plus the configured ``coding_service_url`` origin."""
    from src.settings import get_settings

    settings = get_settings()
    raw: list[str] = list(str(settings.executor_endpoint_allowlist or "").split(","))
    if settings.coding_service_url:
        raw.append(settings.coding_service_url)
    keys: list[tuple[frozenset[str], str, int]] = []
    for entry in raw:
        key = _entry_key(entry)
        if key is not None:
            keys.append(key)
    return keys


def normalize_executor_endpoint(value: str | None) -> str | None:
    """Request validator: ``None``/``""`` pass as None; anything else must be allowlisted."""
    if value is None or value == "":
        return None

    def _bad(reason: str) -> PydanticCustomError:
        return PydanticCustomError(
            EXECUTOR_ENDPOINT_NOT_ALLOWED,
            "executor_endpoint is not allowed: {reason}",
            {"reason": reason},
        )

    return _check_allowlisted_url(value, _bad, allowed_executor_origins()).rstrip("/")


def check_executor_endpoint(value: str) -> str:
    """Run-time twin: raise :class:`ExecutorEndpointNotAllowedError` for an unlisted endpoint."""

    def _bad(reason: str) -> ExecutorEndpointNotAllowedError:
        return ExecutorEndpointNotAllowedError(f"executor_endpoint is not allowed: {reason}")

    if not value:
        raise _bad("no executor endpoint configured")
    return _check_allowlisted_url(value, _bad, allowed_executor_origins()).rstrip("/")


def tls_verify_for(url: str) -> bool:
    """Verify TLS everywhere except in-cluster Kubernetes Service hostnames.

    In-cluster traffic is secured by the Istio ambient mesh (mTLS via ztunnel),
    and those Services are plain http anyway; anything else is verified.
    """
    host = (urlsplit(url).hostname or "").lower()
    return not host.endswith(".svc.cluster.local")


# ── inference_provider graph node ─────────────────────────────────────────────
#
# An ``inference_provider`` node's ``framework_config`` named ``base_url`` (any
# host) and ``api_key_env`` (ANY environment variable) — so POST /graphs could
# send e.g. WM_SECRET_KEY or DATABASE_URL as a Bearer to a host of the caller's
# choosing. Measured on prod 2026-09-22: 0 of 42 graphs have such a node. Now
# ``base_url`` must be on the #282 origin allowlist and ``api_key_env`` must be
# an ``*_API_KEY`` variable name.

INFERENCE_PROVIDER_CONFIG_NOT_ALLOWED = "inference_provider_config_not_allowed"

_API_KEY_ENV_RE = r"^[A-Z][A-Z0-9_]*_API_KEY$"


def check_inference_provider_config(framework_config: Any) -> None:
    """Raise ``ValueError`` (a PydanticCustomError) if the config could leak a secret."""
    import re

    if not isinstance(framework_config, dict):
        return

    def _bad(reason: str) -> PydanticCustomError:
        return PydanticCustomError(
            INFERENCE_PROVIDER_CONFIG_NOT_ALLOWED,
            "inference_provider framework_config is not allowed: {reason}",
            {"reason": reason},
        )

    env = framework_config.get("api_key_env")
    if env is not None and (not isinstance(env, str) or not re.match(_API_KEY_ENV_RE, env)):
        raise _bad("api_key_env must name an *_API_KEY variable")
    base_url = framework_config.get("base_url")
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url:
            raise _bad("base_url must be a non-empty string")
        _check_allowlisted_url(base_url, _bad, allowed_base_url_origins())
