"""Refuse to aim a test run at production unless explicitly asked to.

Live-service integration suites across the estate used to default their target
to production (``https://<svc>.internal``), so a bare
``pytest tests/integration -m integration`` wrote real rows: 939 procedure rows
and 153 workers were left on prod by hand runs. A teardown ``DELETE`` is not a
guarantee — any crash between create and delete leaves the row behind.

Rules enforced here:

* no target configured  -> the caller SKIPS (nothing defaults to production);
* a production host      -> refused, unless ``ALLOW_PROD_INTEGRATION=1``;
* an unparseable target  -> refused, never assumed to be local.

"Production" is decided from the literal host, never by resolving it:

* DNS zones: ``*.internal`` (tailnet split-DNS), ``*.nonagentic.ai`` (public),
  ``*.svc`` / ``*.svc.cluster.local`` / ``*.cluster.local`` (in-cluster);
* short in-cluster names: a single label other than ``localhost``
  (``postgres``, ``pgbouncer``) and ``<svc>.<ns>`` (two labels, e.g.
  ``pgbouncer.pgbouncer``) unless the last label is a reserved non-cluster TLD
  (``localhost``, ``local``, ``test``, ``example``, ``invalid``). Inside a pod
  those resolve through the cluster search path; a docker-compose service name
  is indistinguishable, so it must be published on ``localhost`` instead;
* IPs: k3s pod CIDR ``10.42.0.0/16``, service CIDR ``10.43.0.0/16``, the
  tailnet ``100.64.0.0/10`` (+ ``fd7a:115c:a1e0::/48``) and every node's
  public IP.
"""

from __future__ import annotations

import ipaddress
import os
import re
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping

ALLOW_PROD_VAR = "ALLOW_PROD_INTEGRATION"

_PROD_SUFFIXES = (".internal", ".nonagentic.ai", ".svc", ".svc.cluster.local", ".cluster.local")
_NON_CLUSTER_TLDS = frozenset({"localhost", "local", "test", "example", "invalid"})
_PROD_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "10.42.0.0/16",  # k3s pod CIDR
        "10.43.0.0/16",  # k3s service CIDR
        "100.64.0.0/10",  # tailnet (CGNAT shared range)
        "fd7a:115c:a1e0::/48",  # tailnet IPv6
        "203.0.113.10/32",  # documentation-range stand-in
        "203.0.113.11/32",  # stand-in
        "203.0.113.12/32",  # stand-in
        "203.0.113.13/32",  # stand-in
        "203.0.113.14/32",  # stand-in
        "203.0.113.15/32",  # stand-in
    )
)
_KEYWORD_HOST = re.compile(r"(?:^|\s)host\s*=\s*'?([^\s']+)")


class ProdTargetRefused(RuntimeError):
    """Raised when a test run would write to a production host."""


def is_prod_host(host: str) -> bool:
    h = host.strip().lower().rstrip(".")
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        ip = ipaddress.ip_address(h.split("%", 1)[0])
    except ValueError:
        ip = None
    if ip is not None:
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        return any(ip in net for net in _PROD_NETWORKS)
    if h == "nonagentic.ai" or h.endswith(_PROD_SUFFIXES):
        return True
    labels = h.split(".")
    if len(labels) == 1:
        return h != "localhost"  # short in-cluster Service name: postgres, pgbouncer
    if len(labels) == 2:
        return labels[1] not in _NON_CLUSTER_TLDS  # <svc>.<ns>
    return False


def _hosts_of(target: str) -> list[str] | None:
    """Network hosts named by a URL or libpq keyword DSN; ``None`` if none is parseable.

    A path-valued host (``host=/var/run/postgresql``) is a local unix socket and
    is dropped; a target naming only sockets returns ``[]`` (parseable, local).
    """
    parts = urlsplit(target)
    hosts: list[str] = []
    if parts.scheme and parts.netloc:
        for hostport in parts.netloc.rpartition("@")[2].split(","):
            if hostport.startswith("["):
                hosts.append(hostport[1 : hostport.find("]")])
            elif hostport:
                hosts.append(hostport.split(":", 1)[0])
        for value in parse_qs(parts.query).get("host", []):
            hosts.extend(value.split(","))
    else:
        for value in _KEYWORD_HOST.findall(target):  # libpq keyword DSN: "host=... dbname=..."
            hosts.extend(value.split(","))
    hosts = [h for h in hosts if h]
    if not hosts:
        return None
    return [h for h in hosts if not h.startswith("/")]


def refusal(name: str, target: str, env: Mapping[str, str] | None = None) -> str | None:
    """Why ``target`` must not be used, or ``None`` when it is safe."""
    env = os.environ if env is None else env
    if env.get(ALLOW_PROD_VAR) == "1":
        return None
    hosts = _hosts_of(target)
    if hosts is None:
        return f"{name} has no parseable host — refusing rather than guessing it is local"
    for host in hosts:
        if is_prod_host(host):
            return (
                f"{name} points at production host {host!r}; set {ALLOW_PROD_VAR}=1 only if "
                "writing test data to production is really intended"
            )
    return None


def assert_not_prod(name: str, target: str, env: Mapping[str, str] | None = None) -> None:
    reason = refusal(name, target, env)
    if reason is not None:
        raise ProdTargetRefused(reason)


def integration_target(var: str, env: Mapping[str, str] | None = None) -> str | None:
    """The URL in ``var``; ``None`` when unset (caller skips). Never a prod default."""
    env = os.environ if env is None else env
    url = (env.get(var) or "").strip()
    if not url:
        return None
    assert_not_prod(var, url, env)
    return url
