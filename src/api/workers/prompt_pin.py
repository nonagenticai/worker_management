"""Per-call prompt-version pin for ``POST /workers/{name}/message`` (P4 contract §4).

A card carries a stamp ``prompt_versions: {"<prompt_id>": N}``. When — and only
when — ALL of these hold, the call runs ledger version ``<prompt_id>@N`` instead
of the worker's own ``config.system_prompt``:

1. ``PROMPTS_READ_REQUIRES_PUBLISHED`` is on (the SAME flag as the unpinned
   read; dark by default);
2. the request carries the stamp;
3. the worker's ``config_revision`` is a ``<prompt_id>@<M>`` label whose
   ``prompt_id`` is a key of the stamp. A stamp naming some OTHER prompt says
   nothing about this worker and is ignored.

Anything else returns ``None`` and the call is byte-identical to one without the
stamp: no ledger read, no config change, no cache-key change.

The pinned version must be runnable: PUBLISHED or CANARY. DRAFT, ARCHIVED and
DISCARDED are 409 with the stable top-level body (amendment A4)
``{"error": "prompt_version_not_runnable", "status", "prompt_id", "version"}``
— see :class:`PromptVersionNotRunnable`. A version that is not in the ledger is
404 with the equally stable top-level body
``{"error": "prompt_version_not_found", "prompt_id", "version"}`` — see
:class:`PromptVersionNotFound`.

A per-message ``system_prompt`` is the caller's explicit override and is NEVER
refused (movies-70 review): when it arrives alongside a stamp that names this
worker's prompt, the override runs, the pin is not applied (no ledger read), and
the response says ``prompt_pin = "ignored:message_override"`` with the truthful
``worker_config_revision = "unversioned:<sha12 of what ran>"`` (flag on). With
the flag off the flag-off rule wins: ``ignored:flag_off``, revision untouched.
A multi-entry card stamp naming prompts this worker does not run is ignored
(amendment A1).

PIN VISIBILITY + TRUTHFUL REVISION (amendments A2/A3), :func:`pin_verdict`.
Every execution response carries ``metadata.prompt_pin``; with the flag on the
reported ``worker_config_revision`` is the ``<prompt_id>@<N>`` whose CONTENT
ran — the label is trusted only when the sha256 of the prompt that ran equals
the ledger's ``system_prompt_sha256`` for it, else ``unversioned:<sha12>``.

TENANT GATE (U6, :func:`pin_forbidden`; owner ruling 2026-09-30, strict tenant
separation). ``prompt_versions`` has no tenant column, so EVERY ledger version is
a SYSTEM asset, and system data is never visible to a customer. A NON-platform
tenant's stamp is therefore never applied — PUBLISHED or CANARY, under
``WM_PROMPT_SOURCE=legacy`` and ``published`` alike — and is decided BEFORE any
ledger read: no :func:`resolve_prompt_pin`, no ``get_version``, no label
verification in :func:`pin_verdict`. The call runs exactly what it would have
run without the stamp (its own config prompt, or the per-message
``system_prompt``), reports ``metadata.prompt_pin = "tenant_forbidden"`` with
``worker_config_revision = "unversioned:<sha12>"`` of what ran, and is counted
in ``wm_prompt_pin_forbidden_total``. Never a 403: the customer's worker keeps
working; only the pin is dropped. Flag off keeps the flag-off rule
(``ignored:flag_off``) — no pin runs there for anyone.

CANARY HOLD (U6, :func:`canary_held`). With ``WM_PROMPT_SOURCE=published`` a
pin naming a CANARY version runs it ONLY when ``WM_PROMPT_CANARY_ROUTABLE`` is
true AND the caller is a PLATFORM tenant — a customer tenant never runs a CANARY
(owner ruling 2026-09-30). Otherwise the pin is HELD: the call runs the worker's resolved PUBLISHED
version (``prompt_source.py``) — or its config prompt when it has none — and
reports ``metadata.prompt_pin = "canary_held"``, counted in
``wm_prompt_canary_held_total``. With ``WM_PROMPT_SOURCE=legacy`` (default) a
CANARY pin runs exactly as before: the hold is part of the published rollout,
not a change to the existing pin contract.

⛔ NO CACHE MUTATION. The pin is applied to the per-request ``worker_data`` (a
fresh registry read) and its ``<prompt_id>@<N>`` joins the worker-cache
override fingerprint, so the pinned agent lives in its OWN slot. The worker's
un-pinned slot is never evicted or rebuilt, and the next un-stamped call runs
the worker's own prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from prometheus_client import Counter

from src.api.prompts import repository as prompt_repository
from src.api.prompts.hashing import format_prompt_revision, parse_prompt_revision
from src.api.tenant_constants import is_platform_tenant
from src.settings import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "PROMPT_PIN_NOT_FOUND",
    "PROMPT_PIN_NOT_RUNNABLE",
    "PROMPT_VERSION_NOT_FOUND",
    "PROMPT_VERSION_NOT_RUNNABLE",
    "PinVerdict",
    "PromptPin",
    "PromptVersionNotFound",
    "PromptVersionNotRunnable",
    "apply_pin_verdict",
    "apply_prompt_pin",
    "canary_held",
    "pin_forbidden",
    "pin_verdict",
    "resolve_prompt_pin",
    "unversioned_revision",
]

PROMPT_PIN_NOT_FOUND = "PROMPT_PIN_NOT_FOUND"
PROMPT_PIN_NOT_RUNNABLE = "PROMPT_PIN_NOT_RUNNABLE"
# A4: the stable machine-readable ``error`` value of the 409 body.
PROMPT_VERSION_NOT_RUNNABLE = "prompt_version_not_runnable"
# movies-70: the stable machine-readable ``error`` value of the pinned-404 body.
PROMPT_VERSION_NOT_FOUND = "prompt_version_not_found"

# A2 (+ movies-70 review, + U6): the values of ``metadata.prompt_pin``.
PIN_HONOURED = "honoured"
PIN_IGNORED_FLAG_OFF = "ignored:flag_off"
PIN_IGNORED_MESSAGE_OVERRIDE = "ignored:message_override"
PIN_IGNORED_NO_MATCHING_PROMPT = "ignored:no_matching_prompt"
PIN_IGNORED_NO_STAMP = "ignored:no_stamp"
PIN_LABEL_CONTENT_MISMATCH = "label_content_mismatch"
# U6: a CANARY pin that was NOT run (WM_PROMPT_CANARY_ROUTABLE off).
PIN_CANARY_HELD = "canary_held"
# U6: a NON-platform tenant's stamp — never applied, never read from the ledger.
PIN_TENANT_FORBIDDEN = "tenant_forbidden"

CANARY_HELD = Counter(
    "wm_prompt_canary_held_total",
    "prompt_versions pins naming a CANARY version that were HELD (not run) because "
    "WM_PROMPT_CANARY_ROUTABLE is off or the tenant is not a platform tenant; the "
    "resolved PUBLISHED version (or the config prompt) ran instead.",
)

PIN_FORBIDDEN = Counter(
    "wm_prompt_pin_forbidden_total",
    "prompt_versions pins sent by a NON-platform tenant that were IGNORED (never "
    "resolved, no ledger read): prompt versions are system data, never shown to or "
    "run for a customer; the worker's own prompt ran instead.",
)


@dataclass(frozen=True)
class PromptPin:
    """A resolved, runnable pin: the ledger row this call runs instead."""

    prompt_id: str
    version: int
    status: str
    system: str

    @property
    def revision(self) -> str:
        """The ``<prompt_id>@<N>`` stamp reported as ``worker_config_revision``."""
        return format_prompt_revision(self.prompt_id, self.version)


class PromptVersionNotRunnable(HTTPException):
    """409 for a DRAFT/ARCHIVED/DISCARDED version asked to run (amendment A4).

    ``raw_body`` is rendered AS the response body by ``http_exception_handler``
    — top level, not wrapped in the ``{"error": {...}}`` envelope and not under
    ``detail`` — so a consumer can branch on ``body["error"]`` directly.
    """

    def __init__(self, prompt_id: str, version: int, status: str) -> None:
        self.raw_body: dict[str, Any] = {
            "error": PROMPT_VERSION_NOT_RUNNABLE,
            "status": status,
            "prompt_id": prompt_id,
            "version": version,
        }
        super().__init__(
            status_code=409,
            detail=(
                f"{PROMPT_PIN_NOT_RUNNABLE}: prompt version "
                f"{format_prompt_revision(prompt_id, version)} is {status}; only "
                f"{' or '.join(sorted(prompt_repository.RUNNABLE_STATUSES))} can run."
            ),
        )


class PromptVersionNotFound(HTTPException):
    """404 for a pinned ``prompt_id@version`` that is not in the ledger.

    Same ``raw_body`` mechanism as :class:`PromptVersionNotRunnable`: the body
    is top level, so a consumer tells "that version does not exist" apart from
    "that worker does not exist" (whose 404 keeps the ordinary error envelope)
    by ``body["error"]`` alone.
    """

    def __init__(self, prompt_id: str, version: int) -> None:
        self.raw_body: dict[str, Any] = {
            "error": PROMPT_VERSION_NOT_FOUND,
            "prompt_id": prompt_id,
            "version": version,
        }
        super().__init__(
            status_code=404,
            detail=(
                f"{PROMPT_PIN_NOT_FOUND}: prompt version "
                f"{format_prompt_revision(prompt_id, version)} is not in the ledger."
            ),
        )


async def load_runnable_version(prompt_id: str, version: int) -> dict[str, Any]:
    """The ledger row for ``prompt_id@version``, which may run.

    Raises :class:`PromptVersionNotFound` (404) when the version is absent and
    :class:`PromptVersionNotRunnable` (409, A4 body) for any status outside
    PUBLISHED/CANARY. Shared by the per-call pin and the swap endpoint's
    ``prompt_version`` so all pin errors carry ONE top-level shape,
    ``{"error", "prompt_id", "version"}`` (+ ``"status"`` on the 409).
    """
    row = await prompt_repository.get_version(prompt_id, version)
    if row is None:
        raise PromptVersionNotFound(prompt_id, version)
    status = str(row["status"])
    if status not in prompt_repository.RUNNABLE_STATUSES:
        raise PromptVersionNotRunnable(prompt_id, version, status)
    return row


def unversioned_revision(content_sha256: str) -> str:
    """``unversioned:<first 12 hex>`` — a revision that claims no ledger version."""
    return f"unversioned:{content_sha256[:12]}"


def _stamped_prompt_id(
    stamp: dict[str, int] | None, worker_data: dict[str, Any] | None
) -> str | None:
    """The worker's labelled ``prompt_id`` when the stamp names it, else ``None``."""
    if not stamp:
        return None
    revision = (worker_data or {}).get("config_revision")
    prompt_id, _ = parse_prompt_revision(revision if isinstance(revision, str) else None)
    if prompt_id is None or prompt_id not in stamp:
        return None
    return prompt_id


async def resolve_prompt_pin(
    stamp: dict[str, int] | None,
    worker_data: dict[str, Any] | None,
    *,
    request_system_prompt: str | None = None,
) -> PromptPin | None:
    """The pin this call must run, or ``None`` when the stamp does not apply.

    A per-message ``system_prompt`` wins over a matching stamp: ``None`` is
    returned WITHOUT a ledger read (the override runs; :func:`pin_verdict`
    reports ``ignored:message_override``). Never a 422.
    """
    if not get_settings().prompts_read_requires_published:
        return None
    prompt_id = _stamped_prompt_id(stamp, worker_data)
    if prompt_id is None or stamp is None:
        return None
    if request_system_prompt is not None:
        return None
    version = int(stamp[prompt_id])

    row = await load_runnable_version(prompt_id, version)
    return PromptPin(
        prompt_id=prompt_id, version=version, status=str(row["status"]), system=row["system"]
    )


def pin_forbidden(stamp: dict[str, int] | None, tenant_id: Any = None) -> bool:
    """Whether this caller's stamp must be IGNORED as a tenant breach (U6).

    True — and ``wm_prompt_pin_forbidden_total`` is incremented — when
    ``PROMPTS_READ_REQUIRES_PUBLISHED`` is on, the request carries a stamp, and
    ``tenant_id`` is NOT a platform tenant (omitted counts as non-platform: fail
    closed). Pure: it reads no ledger, so the caller must ask it BEFORE
    :func:`resolve_prompt_pin` and skip that call entirely when it is True. Any
    stamp counts, matching or not, so a refused customer learns nothing about
    which prompt ids exist. Independent of ``WM_PROMPT_SOURCE``.
    """
    if not stamp or not get_settings().prompts_read_requires_published:
        return False
    tenant = str(tenant_id).strip().lower() if tenant_id is not None else None
    if is_platform_tenant(tenant):
        return False
    PIN_FORBIDDEN.inc()
    return True


def canary_held(pin: PromptPin | None, tenant_id: Any = None) -> bool:
    """Whether this resolved pin must be HELD rather than run (U6 canary hold).

    True — and ``wm_prompt_canary_held_total`` is incremented — only for a CANARY
    pin under ``WM_PROMPT_SOURCE=published`` when EITHER
    ``WM_PROMPT_CANARY_ROUTABLE`` is off OR ``tenant_id`` is not a platform
    tenant. ⛔ A customer tenant never runs a CANARY, even with routable on: a
    canary is unreleased system data (owner ruling 2026-09-30). ``tenant_id``
    omitted counts as non-platform (fail closed). Legacy source: always False,
    so the existing pin contract is unchanged.
    """
    if pin is None or pin.status != prompt_repository.CANARY:
        return False
    settings = get_settings()
    if settings.wm_prompt_source != "published":
        return False
    tenant = str(tenant_id).strip().lower() if tenant_id is not None else None
    if settings.wm_prompt_canary_routable and is_platform_tenant(tenant):
        return False
    CANARY_HELD.inc()
    return True


def apply_prompt_pin(worker_data: dict[str, Any], pin: PromptPin) -> None:
    """Fold the pin into THIS request's ``worker_data``.

    ``config`` is COPIED before the write so the dict object a caller (or a
    registry cache) handed us is never mutated in place.
    """
    config = worker_data.get("config")
    config = dict(config) if isinstance(config, dict) else {}
    config["system_prompt"] = pin.system
    worker_data["config"] = config
    worker_data["config_revision"] = pin.revision


@dataclass(frozen=True)
class PinVerdict:
    """What this call reports about its prompt (A2 + A3).

    ``prompt_pin`` is ``None`` only when nothing may be added (flag off and no
    stamp: today's bytes). ``revision`` is ``None`` when the verdict does not
    override the label echo (flag off); otherwise it is the truthful
    ``worker_config_revision``.
    """

    prompt_pin: str | None
    revision: str | None = None


async def pin_verdict(
    stamp: dict[str, int] | None,
    worker_data: dict[str, Any] | None,
    pin: PromptPin | None,
    ran_sha256: str | None,
    *,
    message_override: bool = False,
) -> PinVerdict:
    """Decide ``metadata.prompt_pin`` and the truthful revision for a finished run.

    Flag OFF (byte-identity): no stamp → nothing at all; a stamp → only
    ``prompt_pin="ignored:flag_off"``, the revision echo is left as today.

    Flag ON:
      * a resolved pin → ``honoured`` + ``<prompt_id>@<N>``;
      * a stamp naming this worker's prompt that lost to a per-message
        ``system_prompt`` (``message_override``) → ``ignored:message_override``
        + ``unversioned:<sha12>`` of the content that ran;
      * else the label is verified against the ledger: it is reported only if
        ``ran_sha256`` equals that version's ``system_prompt_sha256``, with
        ``ignored:no_stamp`` / ``ignored:no_matching_prompt``;
      * anything unverifiable (no/unparsable label, missing row, sha mismatch,
        ledger read failure) → ``unversioned:<sha12>`` + ``label_content_mismatch``.
        A mismatch outranks the ``ignored:*`` values: a false revision is the
        fact a consumer most needs.
    """
    if not get_settings().prompts_read_requires_published:
        return PinVerdict(PIN_IGNORED_FLAG_OFF if stamp else None)
    if pin is not None:
        return PinVerdict(PIN_HONOURED, pin.revision)
    if message_override and _stamped_prompt_id(stamp, worker_data) is not None:
        return PinVerdict(
            PIN_IGNORED_MESSAGE_OVERRIDE,
            unversioned_revision(ran_sha256) if ran_sha256 is not None else None,
        )

    label = (worker_data or {}).get("config_revision")
    prompt_id, version = parse_prompt_revision(label if isinstance(label, str) else None)
    if ran_sha256 is not None and prompt_id is not None and version is not None:
        try:
            row = await prompt_repository.get_version(prompt_id, version)
        except Exception:
            logger.warning(
                "prompt label %s@%s could not be verified against the ledger",
                prompt_id,
                version,
                exc_info=True,
            )
            row = None
        if row is not None and str(row.get("system_prompt_sha256", "")).strip() == ran_sha256:
            return PinVerdict(
                PIN_IGNORED_NO_MATCHING_PROMPT if stamp else PIN_IGNORED_NO_STAMP,
                format_prompt_revision(prompt_id, version),
            )
    return PinVerdict(
        PIN_LABEL_CONTENT_MISMATCH,
        unversioned_revision(ran_sha256) if ran_sha256 is not None else None,
    )


def apply_pin_verdict(metadata: dict[str, Any], verdict: PinVerdict) -> dict[str, Any]:
    """Fold a verdict into a ``response_metadata`` dict (in place; returned)."""
    if verdict.prompt_pin is None:
        return metadata
    if verdict.revision is not None or verdict.prompt_pin in (
        PIN_LABEL_CONTENT_MISMATCH,
        PIN_IGNORED_MESSAGE_OVERRIDE,
        PIN_TENANT_FORBIDDEN,
    ):
        prompt_id, version = parse_prompt_revision(verdict.revision)
        metadata["worker_config_revision"] = verdict.revision
        metadata["prompt_id"] = prompt_id
        metadata["prompt_version"] = version
    metadata["prompt_pin"] = verdict.prompt_pin
    return metadata
