"""Redaction helpers for values that end up in application logs.

Two call sites needed the same shape of protection independently
(``crud_endpoints.register_worker`` logging a whole config dict at INFO, and
``mcp_utils._create_managed_docker_server`` logging raw MCP server argv at
INFO) and neither had a shared place to put it — the only existing masking
was the ad hoc env-var loop already in ``mcp_utils.py``. This module is that
shared place so a third call site does not reinvent it again.

Nothing here is a security boundary by itself: it is a best-effort filter on
*names* (does this key/flag look like it holds a secret?), not a guarantee
that every secret shape is caught. Prefer not logging the value at all over
relying on this to scrub it.
"""

from __future__ import annotations

import re

# Substrings that mark a key/flag name as likely to hold a credential.
# Matched case-insensitively as a substring, e.g. "GITHUB_TOKEN", "--api-key",
# "Authorization", "x-bearer-token", "db-credential", "postgres_dsn".
SENSITIVE_NAME_MARKERS: tuple[str, ...] = (
    "token",
    "key",
    "secret",
    "password",
    "pass",
    "pwd",
    "auth",
    "bearer",
    "credential",
    "dsn",
)

_MASK = "***REDACTED***"

# --flag=value  /  --flag VALUE (value consumed separately, see redact_args)
_FLAG_EQUALS_RE = re.compile(r"^(?P<flag>--?[A-Za-z0-9][A-Za-z0-9_-]*)=(?P<value>.*)$")
# --flag:value  (colon separator, e.g. `--api-key:abc123`) — a leading `--?`
# means this never matches a bare positional value like an image reference
# (`ghcr.io/org/img:1.2`), which has no leading dash.
_FLAG_COLON_RE = re.compile(r"^(?P<flag>--?[A-Za-z0-9][A-Za-z0-9_-]*):(?P<value>.*)$")
# KEY=value (env-style positional arg, e.g. docker `-e KEY=value`)
_ENV_EQUALS_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.*)$")
# URLs of the form scheme://<userinfo>@host, where userinfo is user:password
# — mask only the password half, before the @.
_URL_USERINFO_RE = re.compile(
    r"^(?P<prefix>[A-Za-z][A-Za-z0-9+.-]*://[^/@:]+:)(?P<pw>[^@]+)(?P<suffix>@.*)$"
)
# mysql/curl-style attached short flag: -pSECRET (no separator). Deliberately
# only "-p" — this is the one short flag that collides with Docker's port
# publish (`-p 8080:80`), see _is_port_shaped below.
_SHORT_P_ATTACHED_RE = re.compile(r"^-p(.+)$")
# Attached short flags for the OTHER sensitive shorts: `-tghp_token`,
# `-uadmin:hunter2`. Valid curl/docker syntax, so a scan that only reads the
# separate-token form (`-t ghp_token`) leaks the attached one. Case-insensitive
# for the same reason the name test is: `-U` is the same flag as `-u`.
_SHORT_T_ATTACHED_RE = re.compile(r"(?i)^-t(.+)$")
_SHORT_U_ATTACHED_RE = re.compile(r"(?i)^-u(.+)$")
# A value that is plausibly a port / port-mapping, not a secret:
# "8080", "8080:80", "8080:80/tcp", "127.0.0.1:8080:80".
_PORT_SHAPE_RE = re.compile(r"^(?:\d{1,3}(?:\.\d{1,3}){3}:)?\d+(?::\d+)*(?:/(?:tcp|udp))?$")
# A bare `user:secret` pair (curl/docker `-u`/`--user`), e.g. "admin:hunter2".
_USER_SECRET_RE = re.compile(r"^(?P<user>[^:]+):(?P<secret>.+)$")
# Docker's `-u`/`--user` doubles as a numeric UID:GID pair (`-u 1000:1000`),
# which is not a secret and must stay readable.
_UID_GID_RE = re.compile(r"^\d+:\d+$")
# Short flags whose separate-token value is always sensitive, no shape
# exception (contrast "-p", which has the Docker port-publish exception via
# _is_port_shaped below).
# ⚠️ Compared CASE-FOLDED (see _is_user_flag / the membership tests below):
# `-T` is the same flag as `-t`, and a case-sensitive test leaked it.
_ALWAYS_SENSITIVE_SHORT_FLAGS = frozenset({"-p", "-t"})
# `-u`/`--user`: unlike the flags above, only the secret half of a
# `user:secret` pair is masked — the username stays readable — and a numeric
# `UID:GID` pair is left untouched entirely. See _redact_user_colon_secret.
_USER_FLAG_NAMES = frozenset({"-u", "--user"})


def _is_user_flag(name: str) -> bool:
    """`-u` / `--user`, case-insensitively.

    ⛔ This used to be a bare `name in _USER_FLAG_NAMES`, which contradicted
    this module's own documented rule that flag NAMES are matched
    case-insensitively: `--User admin:secret`, `-U admin:secret` and
    `--USER=admin:secret` all reached the log fully unmasked.
    """
    return name.casefold() in _USER_FLAG_NAMES


def _is_always_sensitive_short_flag(name: str) -> bool:
    """`-p` / `-t`, case-insensitively — same reason as :func:`_is_user_flag`."""
    return name.casefold() in _ALWAYS_SENSITIVE_SHORT_FLAGS


# HTTP auth credential shapes that must never reach a log regardless of which
# flag (if any) carried them — e.g. a raw "Authorization: Bearer xyz" header,
# a "--header=Authorization: Bearer xyz" value, or a bare "Bearer xyz" token.
_AUTH_HEADER_RE = re.compile(r"(?i)\b(authorization|proxy-authorization)(\s*:\s*)(\S+(?:\s+\S+)?)")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+\S+")
_BASIC_RE = re.compile(r"(?i)\bbasic\s+[A-Za-z0-9+/=]+")


def is_sensitive_name(name: str) -> bool:
    """True if ``name`` (a dict key or CLI flag) looks like it holds a secret."""
    lowered = name.lower()
    return any(marker in lowered for marker in SENSITIVE_NAME_MARKERS)


def _is_port_shaped(value: str) -> bool:
    """True if ``value`` looks like a port or Docker port-mapping, not a secret."""
    return bool(_PORT_SHAPE_RE.match(value))


def _redact_user_colon_secret(value: str) -> str:
    """Mask the secret half of a bare ``user:secret`` pair (``-u``/``--user``).

    Keeps the username readable (``admin:hunter2`` -> ``admin:***REDACTED***``).
    Leaves a numeric ``UID:GID`` pair untouched (Docker's `-u 1000:1000`,
    which is not a secret) and leaves a bare username with no ``:`` untouched
    (nothing to mask).
    """
    if _UID_GID_RE.match(value):
        return value
    match = _USER_SECRET_RE.match(value)
    if not match:
        return value
    return f"{match.group('user')}:{_MASK}"


def redact_url_userinfo(value: str) -> str:
    """Mask a URL's userinfo password.

    ``user:pass`` before the ``@`` in a ``scheme://`` URL becomes
    ``user:***@`` in the output.
    """
    match = _URL_USERINFO_RE.match(value)
    if not match:
        return value
    return f"{match.group('prefix')}{_MASK}{match.group('suffix')}"


def redact_auth_header_content(value: str) -> str:
    """Mask an HTTP Authorization/Proxy-Authorization credential.

    Applies anywhere in ``value`` — a header line (``Authorization: Bearer xyz``), a
    flag value (``--header=Authorization: Bearer xyz``), or a bare
    ``Bearer``/``Basic`` token — independent of the flag name (if any) that
    carried it. Applied to every arg/value, not only ones behind a
    sensitive-named flag: a non-sensitive flag name (``--header``, ``-H``)
    tells you nothing about whether its value is a credential.
    """
    value = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASK}", value)
    value = _BEARER_RE.sub(f"Bearer {_MASK}", value)
    value = _BASIC_RE.sub(f"Basic {_MASK}", value)
    return value


def _scrub_value(value: str) -> str:
    """Full content scan applied to any arg/value that isn't outright masked.

    Scrubs URL userinfo, then any HTTP auth credential shape.
    """
    return redact_auth_header_content(redact_url_userinfo(value))


def redact_args(args: list) -> list[str]:
    """Redact a CLI argv list for logging.

    Handles, case-insensitively on the flag/key name:
      - ``--flag=value`` / ``-f=value`` where ``flag`` looks sensitive.
      - ``--flag:value`` (colon separator, e.g. ``--api-key:abc123``) — same
        sensitivity check as the ``=`` form.
      - ``KEY=value`` (env-style positional, e.g. after a bare ``-e``).
      - ``--flag`` followed by a separate value token (``--api-key``, ``VALUE``).
      - ``-p``/``-t SECRET`` or attached ``-pSECRET`` (mysql/curl-style
        password/token flags) — ``-p`` alone has a port-shaped exception
        (``-p 8080:80``, Docker's port publish flag), which is left readable.
      - ``-u``/``--user user:secret`` (curl/docker basic-auth-style flag):
        masks ONLY the secret half, keeping the username readable
        (``admin:hunter2`` -> ``admin:***REDACTED***``) — a numeric
        ``UID:GID`` pair (Docker's `-u 1000:1000`) and a bare username with
        no ``:`` are left untouched.
      - ``pass``/``pwd`` join ``token``/``key``/``secret``/``password``/etc.
        as sensitive-name markers, so ``--pass``/``--pwd VALUE`` are masked
        by the generic separate-value path above.
      - a bare value that is itself a URL with ``user:pass`` userinfo before
        the ``@`` in ``scheme://host``, or is one after ``=``/``:`` splitting
        above.
      - an HTTP ``Authorization``/``Bearer``/``Basic`` credential appearing in
        *any* value, regardless of whether the flag carrying it has a
        sensitive-looking name (``--header=Authorization: Bearer xyz``,
        ``-H``/``Authorization: Bearer xyz`` as two args, or a single bare
        ``Authorization: Bearer xyz`` arg).

    Non-sensitive args (``--port=8080``, ``-i``, ``--rm``, an image name like
    ``ghcr.io/org/img:1.2``, ``-p 8080:80``, ``-p 127.0.0.1:5432:5432``,
    ``-u 1000:1000``) pass through unchanged.
    """
    str_args = [str(raw) for raw in args]
    n = len(str_args)
    out: list[str] = []
    i = 0
    while i < n:
        arg = str_args[i]

        u_attached = _SHORT_U_ATTACHED_RE.match(arg)
        if u_attached:
            # `-uadmin:hunter2` — same rule as the separate-token form: the
            # username half stays readable, a numeric UID:GID is left alone.
            out.append(f"{arg[:2]}{_redact_user_colon_secret(u_attached.group(1))}")
            i += 1
            continue

        t_attached = _SHORT_T_ATTACHED_RE.match(arg)
        if t_attached:
            out.append(f"{arg[:2]}{_MASK}")
            i += 1
            continue

        eq_match = _FLAG_EQUALS_RE.match(arg) or _ENV_EQUALS_RE.match(arg)
        colon_match = None if eq_match else _FLAG_COLON_RE.match(arg)
        match = eq_match or colon_match
        if match:
            groups = match.groupdict()
            name = groups["flag"] if "flag" in groups else groups["key"]
            sep = "=" if eq_match else ":"
            prefix = f"{name}{sep}"
            value = arg[len(prefix) :]
            if _is_user_flag(name):
                out.append(f"{prefix}{_redact_user_colon_secret(value)}")
            elif is_sensitive_name(name):
                out.append(f"{prefix}{_MASK}")
            else:
                out.append(f"{prefix}{_scrub_value(value)}")
            i += 1
            continue

        p_attached = _SHORT_P_ATTACHED_RE.match(arg)
        if p_attached:
            value = p_attached.group(1)
            out.append(arg if _is_port_shaped(value) else f"-p{_MASK}")
            i += 1
            continue

        if _is_user_flag(arg):
            next_value = str_args[i + 1] if i + 1 < n else None
            out.append(arg)
            if next_value is not None:
                out.append(_redact_user_colon_secret(next_value))
                i += 2
            else:
                i += 1
            continue

        if _is_always_sensitive_short_flag(arg) or (arg.startswith("-") and is_sensitive_name(arg)):
            next_value = str_args[i + 1] if i + 1 < n else None
            if arg.casefold() == "-p" and next_value is not None and _is_port_shaped(next_value):
                # Docker port publish, e.g. `-p 8080:80` — leave both readable.
                out.append(arg)
                i += 1
                continue
            out.append(arg)
            if next_value is not None:
                out.append(_MASK)
                i += 2
            else:
                i += 1
            continue

        out.append(_scrub_value(arg))
        i += 1

    return out


# Credential-bearing URL query parameters. Google's REST API accepts its key as
# ``?key=...`` and every httpx/SDK error string echoes the full request URL, so
# an upstream failure carries the key into whatever renders the exception
# (issue #327). Masked even when the exact key is unknown to this process.
_URL_SECRET_PARAM_RE = re.compile(
    r"([?&](?:key|api_key|apikey|api-key|access_token|token|secret|password)=)[^&\s'\"#]+",
    re.IGNORECASE,
)
# Known key SHAPES, masked wherever they appear (not only in a query string).
# Google API keys: ``AIza`` + 35 chars of [0-9A-Za-z_-].
_KNOWN_KEY_SHAPE_RES: tuple[re.Pattern[str], ...] = (re.compile(r"AIza[0-9A-Za-z_\-]{35}"),)
_TEXT_MASK = "***"

# Every pattern below runs synchronously on the event loop over UNBOUNDED upstream
# text (a BYO endpoint's error body reaches us via ``str(exc)``), so each one must
# be linear: no unbounded quantifier may be re-scanned from overlapping start
# positions. The scheme is bounded ({0,31}, RFC 3986 schemes are short) because an
# unbounded ``[A-Za-z][A-Za-z0-9+.-]*://`` re-scans to the end of a letter run from
# EVERY start position: 200 KB of letters took ~37 s.
_TEXT_URL_USERINFO_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]{0,31}://[^/@:\s]+:)[^@\s/]+@")
# Free-text auth shapes require a TOKEN-SHAPED value (>= 16 chars of token68-ish
# characters) so prose such as "Basic tier" or "Authorization: denied" is left
# alone. Narrower than the CLI-arg patterns above on purpose: an arg/env value is
# a credential slot where over-masking is harmless, free text is not.
_CREDENTIAL_VALUE_PATTERN = r"[A-Za-z0-9._~+/=-]{16,}"
_TEXT_AUTH_HEADER_RE = re.compile(
    rf"(?i)\b(authorization|proxy-authorization)(\s*:\s*)(?:([A-Za-z]+)\s+)?{_CREDENTIAL_VALUE_PATTERN}"
)
_TEXT_BEARER_RE = re.compile(rf"(?i)\bbearer\s+{_CREDENTIAL_VALUE_PATTERN}")
_TEXT_BASIC_RE = re.compile(r"(?i)\bbasic\s+[A-Za-z0-9+/=]{16,}")
# Input cap applied BEFORE the patterns run (defence in depth on top of linear
# patterns). Cap generously, scrub, then drop a guard tail: a secret the cut split
# in half would no longer match its pattern, so its prefix must not survive.
_MAX_SCRUB_CHARS = 64 * 1024
_CUT_GUARD_CHARS = 1024
_TRUNCATED_MARK = "...[truncated]"


def redact_secrets_in_text(text: str, *secrets: str | None) -> str:
    """Scrub credentials from free text bound for an error detail or a log line.

    In order: every exact ``secrets`` value given (the key this call used), any
    credential-bearing URL query parameter (``?key=``, ``&api_key=``, ...),
    URL userinfo passwords, token-shaped HTTP auth credentials, and known key
    shapes (Google ``AIza...``). Use it on ``str(exc)`` from any upstream call
    before the text reaches an ``HTTPException`` detail, a relayed error or a
    logger. Text longer than ``_MAX_SCRUB_CHARS`` is cut first and returned
    shortened by a further ``_CUT_GUARD_CHARS`` with a ``[truncated]`` marker.
    """
    truncated = len(text) > _MAX_SCRUB_CHARS
    if truncated:
        text = text[:_MAX_SCRUB_CHARS]
    for secret in secrets:
        if secret:
            text = text.replace(secret, _TEXT_MASK)
    text = _URL_SECRET_PARAM_RE.sub(rf"\1{_TEXT_MASK}", text)
    text = _TEXT_URL_USERINFO_RE.sub(rf"\1{_TEXT_MASK}@", text)
    text = _TEXT_AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{_TEXT_MASK}", text)
    text = _TEXT_BEARER_RE.sub(f"Bearer {_TEXT_MASK}", text)
    text = _TEXT_BASIC_RE.sub(f"Basic {_TEXT_MASK}", text)
    for shape in _KNOWN_KEY_SHAPE_RES:
        text = shape.sub(_TEXT_MASK, text)
    if truncated:
        text = text[:-_CUT_GUARD_CHARS] + _TRUNCATED_MARK
    return text
