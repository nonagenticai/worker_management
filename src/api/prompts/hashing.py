"""The digest definitions shared by the prompt ledger and the live-call echo.

⛔ THIS MODULE IS THE SINGLE DEFINITION OF BOTH DIGESTS. Do not re-implement
either one at a call site.

``prompt_versions`` stores what a prompt version IS; ``MessageResponse.metadata``
reports what a call actually RAN. The whole point of roadmap item A5 is that
those two can be compared — so the moment the two sides compute their digest
differently, every comparison answers "drifted" and the attribution silently
becomes noise. Both sides import from here.

Two digests, two jobs (see ``migrations/030_prompt_versions.sql`` for the long
form):

* :func:`version_payload_sha256` — over the whole version payload. The
  version's identity: idempotency and the immutability refusal key on it.
* :func:`sha256_text` — over one text body. Applied to the ``system`` column it
  yields ``system_prompt_sha256``; applied to the prompt an agent instance is
  actually carrying it yields the value a live response echoes. Same function,
  so the two are comparable by construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

__all__ = [
    "PROMPT_ID_RE",
    "PROMPT_REVISION_RE",
    "format_prompt_revision",
    "parse_prompt_revision",
    "sha256_text",
    "version_payload_sha256",
]

# Registry keys come from filenames under ``AA_IMPLEMENTATION/infer/`` and from
# the ``configRevision`` an Ensure node stamps. Bounded and punctuation-limited
# so a caller-chosen id can never become a path, a wildcard or an unbounded
# metric label.
PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# ``<prompt_id>@<version>`` — what S4's generated Ensure nodes write into
# ``workers.config_revision``.
PROMPT_REVISION_RE = re.compile(
    r"^(?P<prompt_id>[A-Za-z0-9][A-Za-z0-9._-]{0,127})@(?P<version>\d{1,9})$"
)


def sha256_text(text: str) -> str:
    """Hex sha256 of ``text`` encoded UTF-8.

    Used for BOTH ``prompt_versions.system_prompt_sha256`` and the
    ``system_prompt_sha256`` a live call echoes. Deliberately no normalisation —
    no strip, no newline folding, no unicode NFC: the publisher converges a
    worker's prompt to these exact bytes, so a digest that forgave whitespace
    would call a worker "in sync" while it ran different text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def version_payload_sha256(
    *,
    question: str | None,
    system: str,
    template: str | None,
    output_schema: dict[str, Any] | None,
    worker_names: list[str] | None,
    purpose_tags: list[str] | None,
) -> str:
    """Hex sha256 over the canonical form of a whole prompt version.

    ⚠️ ``git_sha``, ``status``, ``published_at`` and ``version`` are NOT in the
    payload, and each exclusion is load-bearing:

    * ``git_sha`` is provenance. Re-publishing byte-identical content from a
      later commit must be *unchanged*, not a conflict.
    * ``status``/``published_at`` are ledger bookkeeping. Activating a version
      must not change its identity.
    * ``version`` is the key the digest is compared UNDER. Including it would
      make every version's digest unique by construction and the immutability
      check vacuous.

    Canonicalisation is ``json.dumps(..., sort_keys=True, separators=(",", ":"),
    ensure_ascii=False)`` over a fixed key set, so key order and insignificant
    whitespace in the caller's JSON cannot change the digest. ``None`` and an
    omitted field are the same thing; an empty list and ``None`` are the same
    thing — a publisher that stops sending an empty ``worker_names`` must not
    trip the immutability refusal.
    """
    payload = {
        "question": question or None,
        "system": system,
        "template": template or None,
        "output_schema": output_schema or None,
        "worker_names": list(worker_names or []),
        "purpose_tags": list(purpose_tags or []),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_prompt_revision(revision: str | None) -> tuple[str | None, int | None]:
    """Split a ``<prompt_id>@<version>`` stamp into its two parts.

    Returns ``(None, None)`` for anything that is not that shape — which today
    is almost everything: ``config_revision`` is free text (migration 016) and
    ``swap_worker_prompt`` writes a canary worker NAME into it. Measured on prod
    2026-09-22: 2 of 793 workers carry a revision and neither parses here.

    ⛔ Never re-derive this with ``int(revision)``. The improvement-loop ETL
    does exactly that, gets ``NaN`` for every real value, and silently records
    asset_version 1 — which is why 7,033 of 7,033 ``exec_quality`` rows read
    ``legacy_default``. A value that does not parse must surface as ``None``,
    never as a plausible-looking default.
    """
    if not revision or not isinstance(revision, str):
        return None, None
    match = PROMPT_REVISION_RE.match(revision.strip())
    if match is None:
        return None, None
    return match.group("prompt_id"), int(match.group("version"))


def format_prompt_revision(prompt_id: str, version: int) -> str:
    """The canonical ``<prompt_id>@<version>`` stamp."""
    return f"{prompt_id}@{version}"
