"""What prompt did this call actually run? — the ``MessageResponse.metadata`` echo.

Roadmap A5 / improvement-design §2.3: *"``MessageResponse.metadata`` gains
``worker_config_revision`` and ``system_prompt_sha256``, meaning the prompt
**actually used**, including a per-message ``systemPrompt``. This is what makes
the G1 stamp truthful rather than asserted."*

THE DISTINCTION THIS MODULE EXISTS FOR. A ``config_revision`` is a LABEL
somebody wrote on a worker row. It can be stale, hand-edited, or a canary worker
name (``swap_worker_prompt`` writes one there). On its own it proves nothing:
measured on prod 2026-09-22, 2 of 793 workers carry a revision, NEITHER is an S4
``<prompt_id>@<version>`` stamp, and all 7,033 ``exec_quality`` rows read
``legacy_default``. Echoing the label alone would reproduce exactly that — a
number that looks like attribution and is not.

So this module reports the label AND a digest of the prompt text the agent
instance was actually carrying when it ran. A consumer joins the digest against
``prompt_versions.system_prompt_sha256`` (migration 030) and gets a yes/no
answer instead of a claim:

    revision says   foo@3
    digest says     a1b2…                     <- from THIS call
    ledger says     foo@3 -> a1b2…            <- from the published version
    => the stamp is TRUE for this call.

⛔ PREFER THE RAN INSTANCE OVER THE RECORD, and never silently substitute one
for the other. The agent's ``instructions=`` are baked once at construction, so
a cached slot can be carrying text that differs from the row we just read — that
divergence is precisely what an attribution must be able to expose, not paper
over. ``system_prompt_source`` says which one the digest came from, so a
consumer can discard a ``config``-sourced digest if it wants instance-grade
evidence only.

⚠️ WHAT THIS DIGEST DOES **NOT** COVER: ``config.instruction``. The agent runs
the JOIN of both prompt fields — ``pydantic_ai_agent.py`` builds
``combined_instructions`` by joining ``config.system_prompt`` and
``config.instruction`` with a blank line and passing the result as
``instructions=`` — and ``MessageRequest.instruction`` is a per-message override
with its own worker-cache slot. So two calls that differ ONLY in ``instruction``
really do run different text and receive the SAME ``system_prompt_sha256``. That
is a deliberate scope choice, not an oversight, and it was made on two grounds:

* The field is named ``system_prompt_sha256``. Digesting system_prompt +
  instruction would make the NAME untrue — the same class of defect as an echo
  that claims to describe the ran instance and does not.
* It is the join key with ``prompt_versions.system_prompt_sha256`` (migration
  030), which digests the ``system`` text ALONE. Folding ``instruction`` in
  would make every worker that carries one fail to match its own correctly
  published ledger row, converting true stamps into false mismatches. A false
  negative is still a false stamp.

So the claim made here is narrow and true — *this is the digest of the system
prompt that ran* — and it does not pretend to be a digest of everything the
model saw. ``tests/test_message_prompt_attribution.py`` pins the gap with
``test_a_per_message_instruction_does_not_move_the_digest`` so it stays a stated
limitation rather than an assumed coverage. If per-message ``instruction``
attribution is ever needed it wants its OWN key, not this one.

⚠️ A worker with no prompt on its row is NOT unattributable. ``create_pydantic_ai_agent``
substitutes ``"You are a helpful assistant."`` for a falsy ``system_prompt``, so
that literal is what the model actually sees and what gets digested, with
``source="instance"``. ``null`` is reserved for a call whose prompt text is
genuinely unknown; reporting it for a call whose text is known exactly would be
a false negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.api.prompts.hashing import parse_prompt_revision, sha256_text

__all__ = [
    "METADATA_KEYS",
    "PROMPT_SOURCES",
    "SELECTION_KEYS",
    "PromptSelection",
    "prompt_attribution",
    "selection_attribution",
]

# Every key this module contributes to ``MessageResponse.metadata``. Named so a
# test can assert the exact contract rather than a hand-copied list.
METADATA_KEYS = (
    "worker_config_revision",
    "prompt_id",
    "prompt_version",
    "system_prompt_sha256",
    "system_prompt_source",
)

# U6: WHICH SOURCE selected the system prompt for this call, and which ledger
# version that was. Always present on every response AND every completed
# worker_states row (``metadata``), whatever WM_PROMPT_SOURCE says.
#
# ⚠️ ``prompt_source``, NOT ``system_prompt_source``. That key already exists
# (above) and answers a DIFFERENT question — whether the DIGEST came from the
# ran instance or the record ("instance"/"config"/"none"), which is what exposes
# a cached slot running stale text. Re-using the name for the selection would
# silently change the meaning of a shipped key.
SELECTION_KEYS = (
    "prompt_version_id",
    "prompt_body_sha256",
    "prompt_source",
    "canary_selected_by",
)

CANARY_SELECTED_BY_PIN = "pin"
CANARY_SELECTED_BY_ROUTER = "router"  # reserved for the G-14 router; not emitted yet

PROMPT_SOURCE_PUBLISHED = "published"  # resolved PUBLISHED version (prompt_source.py)
PROMPT_SOURCE_PIN = "pin"  # a per-call pin to a PUBLISHED version
PROMPT_SOURCE_CANARY = "canary"  # a per-call pin to a CANARY version, routable
PROMPT_SOURCE_REQUEST = "request"  # the per-message request.system_prompt
PROMPT_SOURCE_LEGACY = "legacy_config"  # workers.config.system_prompt (today)
PROMPT_SOURCES = (
    PROMPT_SOURCE_PUBLISHED,
    PROMPT_SOURCE_PIN,
    PROMPT_SOURCE_CANARY,
    PROMPT_SOURCE_REQUEST,
    PROMPT_SOURCE_LEGACY,
)


@dataclass(frozen=True)
class PromptSelection:
    """How ``send_message`` chose this call's system prompt.

    ``prompt_version_id`` is ``<prompt_id>@<N>`` for a ledger-backed selection
    (published / pin / canary) and ``None`` otherwise.
    """

    source: str
    prompt_version_id: str | None = None
    # Which mechanism put a CANARY version on this call: "pin" (an explicit
    # per-call pin) today. "router" is RESERVED for phase-10's G-14 router
    # (owner ruling 10-01 "Router behind flag": with WM_PROMPT_CANARY_ROUTABLE
    # a router may send a hash-keyed fraction of platform calls to CANARY).
    # It is not built here. None for every non-canary selection.
    canary_selected_by: str | None = None


def selection_attribution(
    selection: PromptSelection,
    worker_data: dict[str, Any] | None,
    instance: Any = None,
) -> dict[str, Any]:
    """The U6 selection keys, with the digest of the prompt the instance carries.

    ``prompt_body_sha256`` is computed by the SAME instance-first read as
    ``system_prompt_sha256`` (:func:`prompt_attribution`), so the two can never
    describe different text; it is what the outbound X-AA-Prompt-Sha256 header
    carries. ``None`` only where the prompt text is genuinely unknown.
    """
    return {
        "prompt_version_id": selection.prompt_version_id,
        "prompt_body_sha256": prompt_attribution(worker_data, instance)["system_prompt_sha256"],
        "prompt_source": selection.source,
        "canary_selected_by": selection.canary_selected_by,
    }


def _instance_prompt(instance: Any) -> str | None:
    """The system prompt an agent instance is carrying, if it exposes a real one.

    ⛔ WHERE THE PROMPT ACTUALLY LIVES: ``instance.config.system_prompt``, NOT
    ``instance.system_prompt``. This read used to look only at the top-level
    attribute, and **no agent class in this service has one** — measured against
    the three that ``create_agent`` can return:

        PydanticAIAgent | class-attr system_prompt: False | self.config: True
        CLIAgent        | class-attr system_prompt: False | self.config: True
        RemoteCLIAgent  | class-attr system_prompt: False | self.config: True

    Every one of them takes a ``*AgentConfig`` (all carrying a ``system_prompt``
    field, defaulted on ``BaseAgentConfig``) and assigns it to ``self.config``.
    So the old read returned ``None`` on every production call, the echo always
    fell through to the record, and ``system_prompt_source`` could only ever say
    ``"config"``. The ``"instance"`` branch — the one this whole module exists
    for, because it is the only thing that can expose a cached slot running text
    the row no longer holds — was unreachable dead code, and the suite was green
    because its double set the attribute the real objects do not have.

    The config read is tried FIRST because it is where the running text is; the
    top-level read is kept as a fallback so a future agent class that genuinely
    publishes a flat ``system_prompt`` is still honoured.

    ⚠️ ``isinstance`` is load-bearing on BOTH reads, not decoration: a
    ``MagicMock`` answers a mock to any attribute, so an untyped return would
    make ``sha256_text`` digest a mock's repr and the echo would be confident
    nonsense. ``instance.config.system_prompt`` is two attribute hops, so it is
    also two chances to be handed a mock.
    """
    if instance is None:
        return None
    config = getattr(instance, "config", None)
    if config is not None:
        value = getattr(config, "system_prompt", None)
        if isinstance(value, str):
            return value
    value = getattr(instance, "system_prompt", None)
    return value if isinstance(value, str) else None


def _config_prompt(worker_data: dict[str, Any] | None) -> str | None:
    """The system prompt on the per-request worker record.

    This is the value ``_globals._create_worker_instance`` reads
    (``config.get("system_prompt")``) and the one a per-message override has
    already been folded into by the time the echo is built, so it is the right
    fallback — not ``worker_data["system_prompt"]``, which nothing populates.
    """
    if not isinstance(worker_data, dict):
        return None
    config = worker_data.get("config")
    if not isinstance(config, dict):
        return None
    value = config.get("system_prompt")
    return value if isinstance(value, str) else None


def prompt_attribution(
    worker_data: dict[str, Any] | None,
    instance: Any = None,
) -> dict[str, Any]:
    """Build the prompt-attribution block of ``MessageResponse.metadata``.

    Pure: no I/O, no DB read. The hot path must not take a database round trip
    to describe itself, and reconciliation against the ledger is the consumer's
    job — it is the one that knows which version it expected.

    Every key is always present, with ``None`` where the fact is unknown. An
    absent key would be indistinguishable from an older server; an explicit
    ``null`` says "this call could not be attributed", which is a finding.
    """
    prompt_text = _instance_prompt(instance)
    source = "instance"
    if prompt_text is None:
        prompt_text = _config_prompt(worker_data)
        source = "config" if prompt_text is not None else "none"

    revision = (worker_data or {}).get("config_revision")
    if not isinstance(revision, str) or not revision.strip():
        revision = None
    prompt_id, prompt_version = parse_prompt_revision(revision)

    return {
        "worker_config_revision": revision,
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "system_prompt_sha256": sha256_text(prompt_text) if prompt_text is not None else None,
        "system_prompt_source": source,
    }


def response_metadata(
    worker_data: dict[str, Any] | None,
    instance: Any = None,
    context: Any = None,
    selection: PromptSelection | None = None,
) -> dict[str, Any]:
    """``metadata`` for a ``MessageResponse``: the caller's context plus attribution.

    ⚠️ ``metadata`` used to be ``None`` whenever the request carried no
    ``context``; it is now always a dict. ``context`` keeps its exact previous
    shape and position, and is still omitted entirely when absent, so a consumer
    reading ``metadata.context`` is unaffected.

    ``selection`` (U6) adds :data:`SELECTION_KEYS`; ``send_message`` always
    passes one.
    """
    meta: dict[str, Any] = {}
    if context:
        meta["context"] = context
    meta.update(prompt_attribution(worker_data, instance))
    if selection is not None:
        meta.update(selection_attribution(selection, worker_data, instance))
    return meta
