"""Request / response models for ``/prompts*``.

Field sizes are bounded on purpose. These routes write a FLEET-GLOBAL table (no
tenant column — prompts are git assets, not tenant data) and, like every other
bare route on this service, they are reachable through the public
``api.nonagentic.ai/workers/`` prefix rewrite with nothing but a valid licence
token. Authorization is a known, deliberately-sequenced gap for this whole
service (see ``routes.py``'s header); what bounds the damage MEANWHILE is that
a published version can never be rewritten and that every field has a ceiling.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.prompts.hashing import PROMPT_ID_RE

# 1 MB, matching ``PromptSwapRequest.new_system_prompt`` — the same text ends up
# in the same place, so a limit the swap accepts and the ledger refuses would be
# a trap.
_MAX_PROMPT_CHARS = 1_000_000
_MAX_WORKER_NAMES = 64
_MAX_TAGS = 32

# REALIGN R7 (coordinator ruling 2026-10-03): a move INTO CANARY or PUBLISHED
# names the Planka card that authorised it — a 19-digit decimal id. Checked by
# the ROUTE after the role gate (422 ``prompt_release_card_required``), not by a
# field constraint here: a pydantic constraint would reject before the role
# gate and turn a role-less caller's 403 into a 422. ASCII digits only
# (``\d`` would admit other Unicode digits); ``fullmatch`` so no trailing
# newline slips through. Migration 039 enforces the same shape in a CHECK.
ACTOR_CARD_ID_RE = re.compile(r"[0-9]{19}")
_ACTOR_CARD_ID_DESCRIPTION = (
    "Planka card id (^[0-9]{19}$) of the board card authorising this release. "
    "REQUIRED for a move into CANARY or PUBLISHED (REALIGN R7): missing or "
    'malformed is 422 {"error":"prompt_release_card_required"} (checked after '
    "the role gate, before any write) and recorded on the ledger row. Neither "
    "required nor recorded for ARCHIVED, DISCARDED or DRAFT targets."
)


def is_valid_actor_card_id(value: Any) -> bool:
    """True when ``value`` is a Planka card id: exactly 19 ASCII decimal digits."""
    return isinstance(value, str) and ACTOR_CARD_ID_RE.fullmatch(value) is not None


# A caller-supplied transition ``reason`` is a short token, not prose.
TRANSITION_REASON_RE = re.compile(r"^[a-z0-9_:.-]{1,200}$")
# Every ``reason`` the SYSTEM writes to prompt_version_transitions itself. A
# caller may not forge one: a ledger row reading ``activate`` must mean the
# activate route wrote it. Sources (keep in sync; tests/test_prompt_routes_unit.py
# asserts it): repository.REASON_ACTIVATE / REASON_ACTIVATE_SUPERSEDED, the 033
# backfill's 'backfill_033_creation', and 'migration' (the backfill actor_kind).
# ``initial_create`` is deliberately NOT here: it is caller-supplied by design and
# guarded by its own path (service-internal role + no publish history).
# ``adopt_live`` IS reserved: only ``scripts/adopt_live_worker_prompts.py`` writes
# it (repository.REASON_ADOPT_LIVE), adopting a worker's LIVE DB prompt as v1.
RESERVED_TRANSITION_REASONS = frozenset(
    {"activate", "activate_superseded", "adopt_live", "backfill_033_creation", "migration"}
)
# Any future backfill reason is reserved too.
_RESERVED_REASON_PREFIXES = ("backfill_",)


class PromptVersionUpsert(BaseModel):
    """Body for ``PUT /prompts/{prompt_id}/versions/{version}``."""

    model_config = ConfigDict(extra="forbid")

    system: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_PROMPT_CHARS,
        description="The system prompt body. These exact bytes are what a worker's "
        "system_prompt is converged to, and what system_prompt_sha256 digests.",
    )
    question: str | None = Field(
        default=None,
        max_length=2_000,
        description="Short human description; what a retrieval shortlist shows.",
    )
    template: str | None = Field(
        default=None, max_length=_MAX_PROMPT_CHARS, description="User-message template."
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema the worker's output is validated against."
    )
    worker_names: list[str] = Field(
        default_factory=list,
        max_length=_MAX_WORKER_NAMES,
        description="Workers this version is published to.",
    )
    purpose_tags: list[str] = Field(
        default_factory=list, max_length=_MAX_TAGS, description="Free-text tags, searchable."
    )
    git_sha: str | None = Field(
        default=None,
        max_length=64,
        description="Commit the content came from. Provenance only — NOT part of sha256, "
        "so re-publishing identical content from a later commit is 'unchanged'.",
    )
    expected_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="Optional assertion: the payload digest the caller believes it is "
        "publishing. 422 when it does not match what the server computes — which catches "
        "a publisher whose canonicalisation has drifted from the server's.",
    )

    @field_validator("worker_names", "purpose_tags")
    @classmethod
    def _non_empty_entries(cls, v: list[str]) -> list[str]:
        for entry in v:
            if not entry or not entry.strip():
                raise ValueError("entries must be non-empty")
            if len(entry) > 255:
                raise ValueError("entries must be at most 255 characters")
        return v

    @field_validator("expected_sha256")
    @classmethod
    def _hex_digest(cls, v: str | None) -> str | None:
        if v is None:
            return None
        lowered = v.strip().lower()
        if len(lowered) != 64 or any(c not in "0123456789abcdef" for c in lowered):
            raise ValueError("expected_sha256 must be 64 lowercase hex characters")
        return lowered


class PromptVersionResponse(BaseModel):
    """One row of ``prompt_versions``."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str
    version: int
    sha256: str
    system_prompt_sha256: str
    question: str | None = None
    system: str
    template: str | None = None
    output_schema: dict[str, Any] | None = None
    worker_names: list[str] = Field(default_factory=list)
    purpose_tags: list[str] = Field(default_factory=list)
    git_sha: str | None = None
    status: str
    published_at: str | None = None
    activated_at: str | None = None
    config_revision: str = Field(
        ...,
        description="The '<prompt_id>@<version>' stamp a worker carries for this version — "
        "the exact string to compare against MessageResponse.metadata.worker_config_revision.",
    )


class PromptVersionWriteResponse(BaseModel):
    """Result of a ``PUT``. ``written`` is the publisher's idempotency signal."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str
    version: int
    sha256: str
    system_prompt_sha256: str
    status: str
    config_revision: str
    written: bool = Field(
        ...,
        description="True when this PUT inserted the row; False when it already existed "
        "with an identical payload digest (idempotent no-op).",
    )
    unchanged: bool = Field(
        ..., description="True when the row already existed with this exact sha256."
    )


class PromptSearchHit(BaseModel):
    """A search result. Carries no prompt body — use the version route for that."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str
    version: int
    sha256: str
    system_prompt_sha256: str
    question: str | None = None
    purpose_tags: list[str] = Field(default_factory=list)
    worker_names: list[str] = Field(default_factory=list)
    status: str
    published_at: str | None = None
    config_revision: str
    rank: float


class PromptSearchResponse(BaseModel):
    """A page of search hits plus the filters that produced it."""

    model_config = ConfigDict(extra="ignore")

    query: str | None = None
    tag: str | None = None
    status: str
    # Echo of the ``prompt_id`` filter the server APPLIED (null when none). A
    # server that predates the filter omits this key, so a client can tell
    # "honoured, and empty" from "silently ignored".
    prompt_id: str | None = None
    count: int
    hits: list[PromptSearchHit] = Field(default_factory=list)


class PromptStatusChange(BaseModel):
    """Body for ``POST /prompts/{prompt_id}/versions/{version}/status``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["DRAFT", "CANARY", "PUBLISHED", "ARCHIVED", "DISCARDED"] = Field(
        ...,
        description="Target status. Permitted moves: DRAFT->CANARY|DISCARDED, "
        "CANARY->PUBLISHED|DISCARDED, PUBLISHED->ARCHIVED|DISCARDED, "
        "ARCHIVED->PUBLISHED|DISCARDED; anything else is 409.",
    )
    reason: str | None = Field(
        default=None,
        max_length=200,
        description="Token (^[a-z0-9_:.-]{1,200}$) recorded on the transition ledger "
        "row; system-written reasons (activate, activate_superseded, adopt_live, "
        "backfill_*, migration) are refused 422. The value "
        "'initial_create' additionally permits DRAFT->PUBLISHED for a prompt that "
        "has never been PUBLISHED or ARCHIVED (service-internal or prompt-asset-bootstrap "
        "role required).",
    )
    actor_card_id: str | None = Field(
        default=None,
        description=_ACTOR_CARD_ID_DESCRIPTION,
        json_schema_extra={"pattern": "^[0-9]{19}$"},
    )

    @field_validator("reason")
    @classmethod
    def _reason_token(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not TRANSITION_REASON_RE.fullmatch(v):
            raise ValueError("reason must match ^[a-z0-9_:.-]{1,200}$")
        if v in RESERVED_TRANSITION_REASONS or v.startswith(_RESERVED_REASON_PREFIXES):
            raise ValueError(f"reason {v!r} is reserved for system-written ledger rows")
        return v


class PromptStatusResponse(BaseModel):
    """Result of a status transition."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str
    version: int
    status: str
    previous_status: str
    config_revision: str
    activated_at: str | None = None


class PromptTransition(BaseModel):
    """One ``prompt_version_transitions`` row (migration 033)."""

    model_config = ConfigDict(extra="ignore")

    id: int
    prompt_id: str
    version: int
    from_status: str | None = Field(
        default=None, description="Null on the creation row (the PUT that inserted the version)."
    )
    to_status: str
    reason: str | None = None
    actor_sub: str | None = Field(
        default=None,
        description="Verified principal: SA client id (azp) or licence subject; null if none.",
    )
    actor_kind: str = Field(
        ..., description="service_account | licence_subject | unverified | migration"
    )
    actor_card_id: str | None = Field(
        default=None,
        description="Planka card id that authorised this transition (migration 039, "
        "REALIGN R7). Set on every CANARY/PUBLISHED row since 039 except "
        "actor_kind=migration rows (in-process adoption); null on older rows.",
    )
    at: str


class PromptTransitionsResponse(BaseModel):
    """The ledger of one prompt version, oldest first."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str
    version: int
    count: int
    transitions: list[PromptTransition] = Field(default_factory=list)


class PromptActivateRequest(BaseModel):
    """Body for the deprecated ``POST …/activate``: the release's board card."""

    model_config = ConfigDict(extra="forbid")

    actor_card_id: str | None = Field(
        default=None,
        description=_ACTOR_CARD_ID_DESCRIPTION,
        json_schema_extra={"pattern": "^[0-9]{19}$"},
    )


class PromptActivateResponse(BaseModel):
    """Result of an activate: what is active now and what it replaced."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str
    version: int
    status: str
    config_revision: str
    previous_active_version: int | None = Field(
        default=None,
        description="The version that was active before this call, or null if none was.",
    )
    changed: bool = Field(..., description="False when this version was already the active one.")


def validate_prompt_id(prompt_id: str) -> str:
    """Return ``prompt_id`` or raise ``ValueError``.

    Shared by every route so the path parameter cannot differ from what the
    ledger key accepts.
    """
    if not PROMPT_ID_RE.match(prompt_id):
        raise ValueError(
            f"prompt_id must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}} (got {prompt_id[:64]!r})"
        )
    return prompt_id
