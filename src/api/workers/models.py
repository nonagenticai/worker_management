"""Pydantic models for Workers API requests and responses."""

import os
import re
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails

from src.api.prompts.hashing import PROMPT_ID_RE
from src.api.security_validation import (
    SecurityValidationError,
    is_safe_json_data,
    validate_identifier,
    validate_string_security,
)
from src.security.request_overrides import (
    check_mcp_config_preset,
    check_request_mcp_config,
    normalize_base_url_override,
    normalize_executor_endpoint,
    reject_output_validator,
)

# Single source of truth for provider names / credential requirements. This is
# the dependency-free registry module, NOT src.worker.agents.pydantic_ai_agent —
# importing it here costs nothing and keeps this schema module free of the heavy
# pydantic_ai/xai_sdk chain.
from src.worker.provider_registry import (
    API_KEY_PROVIDERS,
    PROVIDER_API_KEY_ENV_VARS,
    PROVIDER_NAMES,
)


# Worker State Models
class WorkerStateBase(BaseModel):
    """Base model for worker states."""

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields for backward compatibility
        json_schema_extra={
            "examples": [
                {
                    "worker_id": "worker_123",
                    "worker_name": "data_processor",
                    "is_agent": True,
                    "state_type": "processing",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "metadata": {
                        "task_id": "task_456",
                        "progress": 75,
                        "status": "running",
                    },
                    "worker_state": {
                        "current_operation": "analyzing_data",
                        "processed_records": 1500,
                        "total_records": 2000,
                        "errors": 0,
                    },
                }
            ]
        },
    )

    worker_id: str
    worker_name: str
    is_agent: bool = False
    state_type: str
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None
    worker_state: dict


class AgentStateCreate(WorkerStateBase):
    """Model for creating agent states."""

    is_agent: bool = True
    memory: dict[str, Any] | None = None
    conversation_history: list[dict[str, Any]] | None = None
    current_task: str | None = None


class AgentStateResponse(AgentStateCreate):
    """Response model for agent states."""

    id: int


def derive_agent_type(config: dict[str, Any]) -> str | None:
    """``agent_type`` for the effective worker. Pure — never raises.

    DERIVED, not merely defaulted: ``agent_role`` + ``model_provider`` without
    an explicit ``agent_type`` means ``pydantic_ai``. A create must STORE that
    — ``_globals.py`` reads ``config.get("agent_type")`` straight from the row,
    so dropping it would silently downgrade a new pydantic-ai worker to a basic
    one.

    Separate from :func:`validate_effective_worker` because the two are asked
    of different things: validation is asked of what the CLIENT sent (so that a
    caller which passes today cannot start failing), while the derived value is
    taken from the merged row that will actually be written.
    """
    agent_role = config.get("agent_role")
    agent_type = config.get("agent_type")
    if not agent_role:
        # Normalise "" -> None.
        return agent_type or None
    if not agent_type and config.get("model_provider"):
        return "pydantic_ai"
    return agent_type or None


def validate_effective_worker(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the EFFECTIVE worker and return it with ``agent_type`` derived.

    The effective worker is the stored row merged with the fields this request
    actually set — i.e. what the row will HOLD once the write lands, not what
    the caller happened to type.

    These rules used to be ``@model_validator``s on :class:`WorkerCreateRequest`,
    which run at FastAPI parse time against the request body ALONE. That was
    correct while a write replaced the whole config: the body *was* the worker.
    Since #205 a write MERGES only the keys the caller sent, so the body is a
    SUBSET of the resulting worker, and every "X is required when Y" rule asked
    of it rejects a legitimate partial update.

    Measured consequence, 2026-08-29: ``knowledge/kag`` seeds three workers on
    every pod start. Once it correctly stopped re-asserting operator-owned
    fields, every seed 422'd with "model is required when model_provider is
    specified" — it still sent ``model_provider`` while omitting ``model``,
    which the stored row already had. No client-side fix existed: dropping
    ``model_provider`` too trips "model_provider is required for pydantic_ai
    agent", and dropping ``agent_type`` as well trips the derivation below. The
    rules were not wrong; they were being asked of the wrong object. The same
    422 hit any operator editing one field by hand, and the AA
    prompt-improvement loop.

    A CREATE is not weakened: with no stored row the effective config IS the
    request, so the same rules raise on the same bodies.

    Raises ``ValueError`` — callers map it to 422 ``VALIDATION_ERROR``.
    """
    effective = dict(config)
    agent_role = effective.get("agent_role")
    agent_type = derive_agent_type(effective)
    model_provider = effective.get("model_provider")

    if effective.get("agent_type") and not agent_role:
        raise ValueError("agent_role is required when agent_type is provided")

    if agent_role and agent_type:
        try:
            import src.worker.agents  # noqa: F401, RUF100
            from src.worker.agents.agent_registry import list_available_agent_types

            available_types = list_available_agent_types()
            # Allow 'pydantic_ai' even if the optional dependency is absent.
            if agent_type not in available_types and agent_type != "pydantic_ai":
                raise ValueError(
                    f"Unsupported agent type: {agent_type}. "
                    f"Available types: {', '.join(available_types)}"
                )
        except ImportError:
            if agent_type not in [
                "pydantic_ai",
                "claude_code",
                "codex",
                "remote_claude",
                "remote_codex",
            ]:
                raise ValueError(f"Unsupported agent type: {agent_type}") from None

    effective["agent_type"] = agent_type

    if agent_type == "pydantic_ai":
        if not model_provider:
            raise ValueError("model_provider is required for pydantic_ai agent")
        # Single source of truth: the providers ModelFactory can build, exposed
        # as PROVIDER_NAMES. Checked here rather than as a field validator
        # because it is conditional on the EFFECTIVE agent_type — a partial
        # update that omits agent_type would have skipped it entirely.
        if model_provider not in PROVIDER_NAMES:
            raise ValueError(f"Unsupported model provider for pydantic_ai: {model_provider}")
        if not effective.get("model"):
            raise ValueError("model is required when model_provider is specified")

        # Membership is DERIVED from the provider registry's requires_api_key
        # flag, never hand-listed: the old hardcoded list omitted cohere and
        # google, so those two silently skipped this check and only exploded
        # later inside _create_cohere_model / _create_google_model.
        if model_provider in API_KEY_PROVIDERS:
            api_key_provided = bool(effective.get("api_key_override"))
            if not api_key_provided:
                env_var = PROVIDER_API_KEY_ENV_VARS.get(model_provider)
                if env_var:
                    api_key_provided = bool(os.environ.get(env_var))

            # Only require API keys when explicitly configured to do so.
            require_api_keys = os.environ.get("REQUIRE_API_KEYS", "false").lower() == "true"
            if not api_key_provided and require_api_keys:
                raise ValueError(
                    "API key required for the specified model provider. "
                    "Please provide an API key or set the appropriate environment variable."
                )

    return effective


class WorkerCreateRequest(BaseModel):
    """Request model for worker creation."""

    model_config = ConfigDict(
        protected_namespaces=(),
        extra="ignore",  # Allow extra fields for backward compatibility
        json_schema_extra={
            "examples": [
                {
                    "worker_name": "my_agent",
                    "agent_role": "General Purpose Agent",
                    "agent_type": "pydantic_ai",
                    "model_provider": "inference_router",
                    "model": "local-llm",
                    "system_prompt": "You are a helpful assistant.",
                },
                {
                    "worker_name": "code-reviewer",
                    "agent_role": "Claude Code agent for code-reviewer",
                    "agent_type": "claude_code",
                    "system_prompt": "You are a code reviewer.",
                },
                {
                    "worker_name": "repo-analyzer",
                    "agent_role": "Remote Claude agent for repo-analyzer",
                    "agent_type": "remote_claude",
                    "container_id": "abc123",
                    "system_prompt": "You are a code analyst.",
                },
            ]
        },
    )

    worker_name: str = Field(..., min_length=1)
    tool_names: list[str] = []  # Default to empty list
    agent_role: str | None = None
    agent_type: str | None = None
    model_provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    ir_provider: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        description=(
            "inference-router provider to pin, forwarded as the X-Provider header "
            "(IR's single-provider path, no failover), e.g. `pgx-local`. None ⇒ no "
            "header. A caller's inbound X-Provider on the message wins. Persisted "
            "like seed."
        ),
    )
    seed: StrictInt | None = Field(
        default=None,
        description=(
            "Sampling seed, sent to the model as the OpenAI-compatible top-level "
            "`seed`. None/omitted ⇒ no seed key is sent. Persisted in the worker "
            "config like temperature: omit it to keep the stored value, send an "
            "explicit null to clear it. Echoed per call in usage_metrics.seed."
        ),
    )
    repetition_penalty: float | None = Field(
        default=None,
        gt=0.0,
        le=2.0,
        description=(
            "vLLM repetition penalty (1.0 = off). Sent only to self-hosted backends "
            "(as a top-level body key via extra_body); None/omitted ⇒ nothing is "
            "sent. Persisted like seed: omit to keep, explicit null to clear. Echoed "
            "per call in usage_metrics.repetition_penalty."
        ),
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "OpenAI-compatible frequency_penalty. None/omitted ⇒ nothing is sent. "
            "Persisted like seed. Echoed per call in usage_metrics.frequency_penalty."
        ),
    )
    enforce_schema_bounds: StrictBool | None = Field(
        default=None,
        description=(
            "Retry then fail: when the output breaks a minItems/maxItems/minLength/"
            "maxLength bound in output_schema, re-ask the model with feedback naming the "
            "JSON path and the bound (within output_retries / the provider default), and "
            "fail the run with HTTP 422 SCHEMA_BOUND_VIOLATION if the last retry still "
            "violates it. Those bounds cannot be compiled into the decoding grammar, so "
            "this post-hoc check is their only enforcement. None/false ⇒ advisory "
            "(violating paths are still reported in usage_metrics.bound_violations). A "
            "per-message value wins."
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens in the model response. None = use provider default.",
        json_schema_extra={"example": 6000},
    )
    system_prompt: str | None = None
    instruction: str | None = None
    user_prompt: str | None = None
    description: str | None = None
    deps: list[str] | None = None

    # Fields for MCP integration and overrides
    mcp_config: str | dict[str, Any] | None = None  # Path to config file or config dict
    mcp_config_preset: str | None = None  # Preset config name
    mcp_servers: list[str] | None = None  # List of specific MCP server names to enable
    base_url_override: str | None = None  # Override base URL for model provider
    api_key_override: str | None = None  # Override API key
    tools_dir_override: str | None = None  # Override tools directory

    # Structured output configuration
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema to enforce model output format",
    )
    output_validator: str | None = Field(
        default=None,
        description=(
            "UNSUPPORTED. Must be null or empty: request-supplied validator code is "
            "refused with 422 output_validator_code_not_supported. Use output_schema."
        ),
    )
    output_retries: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description=(
            "Per-worker override for pydantic-ai output-validation retry "
            "budget. None preserves the provider default (mlx=10, "
            "inference_router=0, other=3). Bump above 0 for short-prompt "
            "structured-output workers on inference_router where a single "
            "guided-decoding hiccup must not fail the whole request."
        ),
    )

    # Remote execution fields (for remote_claude, remote_codex)
    container_id: str | None = Field(
        default=None,
        description="Container ID for remote execution in code_graph_helper",
    )
    executor_endpoint: str | None = Field(
        default=None,
        description="Base URL of coding service (resolved from CODING_SERVICE_URL env when omitted)",
    )
    workspace: str | None = Field(
        default=None,
        description="Working directory in remote container (defaults to /workspace)",
    )

    # State storage configuration
    disable_state_storage: bool = Field(
        default=False,
        description="Skip all database operations for ephemeral workers",
    )

    # Free-text marker stamped by the seeder/bootstrap so operators can
    # diff "what revision did each tenant land on?" without inspecting the
    # full config blob. Round-tripped untouched; never interpreted by WM.
    # Mirrors FR_AGENT_CONFIG_REVISION (scripts/seed_fr_agent.py:49) and
    # WORKER_CONFIG_REVISION (fr/backend/src/agent/wm-bootstrap.ts:36).
    config_revision: str | None = Field(
        default=None,
        max_length=64,
        description="Free-text revision marker stamped by the seeder/bootstrap; round-tripped untouched.",
    )

    # Default work-size tier for this worker. A per-message ``work_tier`` on
    # MessageRequest overrides it. Selects the tenant's per-tier BYO LLM row
    # (role ``tier:<tier>``) at run time; None ⇒ tenant default row.
    work_tier: Literal["small", "medium", "high"] | None = Field(
        default=None,
        description="Default work-size tier (small/medium/high) for this worker.",
    )

    @field_validator("worker_name")
    @classmethod
    def validate_worker_name(cls, v: str) -> str:
        """Validate worker name for security and format compliance."""
        if not v or not v.strip():
            raise ValueError("Worker name cannot be empty")

        try:
            # Use comprehensive security validation
            validated_name = validate_identifier(
                v.strip(), "worker_name", max_length=100, allow_unicode=True
            )

            # Additional validation for malicious content
            validated_name = validate_string_security(
                validated_name,
                "worker_name",
                max_length=100,
                allow_html=False,
                strict_mode=True,
            )

            return validated_name

        except SecurityValidationError as e:
            raise ValueError(str(e)) from e

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, v: list[str] | None) -> list[str]:
        """Validate and filter tool names list."""
        if v is None:
            return []

        # Filter out empty tool names and validate remaining ones
        valid_tools = []
        for tool_name in v:
            if tool_name and tool_name.strip():
                tool_name = tool_name.strip()  # noqa: PLW2901 -- intentional inline normalize
                # Validate tool name format
                if not re.match(r"^[a-zA-Z0-9_-]+$", tool_name):
                    raise ValueError(
                        f"Tool name '{tool_name}' contains invalid characters. Only letters, numbers, underscores, and hyphens are allowed"
                    )
                valid_tools.append(tool_name)
            # Skip empty tool names silently

        return valid_tools

    @field_validator("mcp_config")
    @classmethod
    def validate_mcp_config(cls, v: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
        """Refuse stdio servers, file paths and non-allowlisted URLs (typed 422).

        See ``check_request_mcp_config`` in src/security/request_overrides.py.
        """
        return check_request_mcp_config(v)

    @field_validator("executor_endpoint")
    @classmethod
    def validate_executor_endpoint(cls, v: str | None) -> str | None:
        """The remote executor receives this service's bearer: allowlisted origins only."""
        return normalize_executor_endpoint(v)

    @field_validator("mcp_config_preset")
    @classmethod
    def validate_mcp_config_preset(cls, v: str | None) -> str | None:
        """A preset is a server-side registry name, never a path."""
        return check_mcp_config_preset(v)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float | None) -> float | None:
        """Validate temperature is within acceptable range (0.0-2.0)."""
        if v is not None and (v < 0.0 or v > 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int | None) -> int | None:
        """Validate max_tokens is within a safe range (1-16384).

        Upper bound 16384 protects self-hosted vLLM (Nemotron-3 on SM120/GB10)
        from a CUTLASS TMA-WS FP4-MoE kernel crash that triggers when
        max_tokens approaches the model's full 131k context under structured
        output (incident 2026-05-13). 16384 covers all real-world worker
        outputs observed in production (peak: 15000); longer completions
        should stream or chunk.
        """
        if v is None:
            return v
        if v < 1 or v > 16384:
            raise ValueError("max_tokens must be between 1 and 16384")
        return v

    @field_validator("base_url_override")
    @classmethod
    def validate_base_url_override(cls, v: str | None) -> str | None:
        # None = no override. Otherwise scheme+host+port must be allowlisted
        # (``base_url_override_allowlist``); see src/security/request_overrides.py.
        return normalize_base_url_override(v)

    @field_validator("api_key_override")
    @classmethod
    def validate_api_key_override(cls, v: str | None) -> str | None:
        # Reject empty strings but allow None for optional override fields
        if v == "":
            raise ValueError("API key override cannot be empty string")
        return v

    @field_validator("tools_dir_override")
    @classmethod
    def validate_tools_dir_override(cls, v: str | None) -> str | None:
        # Reject empty strings but allow None for optional override fields
        if v == "":
            raise ValueError("Tools directory override cannot be empty string")
        return v

    @field_validator("deps")
    @classmethod
    def validate_deps(cls, v: list[str] | None) -> list[str] | None:
        """Validate Python dependencies list for security and format."""
        if v is None:
            return v

        # Filter out empty deps and validate remaining ones
        valid_deps = []
        for dep in v:
            if dep and dep.strip():
                dep = dep.strip()  # noqa: PLW2901 -- intentional inline normalize

                # Check for null bytes and other control characters
                if "\x00" in dep:
                    raise ValueError("Dependency name cannot contain null bytes")

                # Check for other dangerous control characters (except newlines which might be in descriptions)
                if any(ord(char) < 32 and char not in "\n\r\t" for char in dep):
                    raise ValueError("Dependency name contains control characters")

                # Use byte length for validation to properly handle Unicode with combining characters
                # This allows legitimate Unicode strings while preventing excessively long inputs
                dep_bytes = dep.encode("utf-8")
                if (
                    len(dep_bytes) > 500
                ):  # Increased from 200 chars to 500 bytes to accommodate Unicode
                    raise ValueError(
                        f"Dependency '{dep[:50]}...' is too long (max 500 bytes when UTF-8 encoded)"
                    )

                # Additional validation: check if it's a reasonable dependency name
                # Reject strings that are likely malicious (e.g., very long strings of repeated characters)
                if len(dep) > 200:  # Character count check for extremely long strings
                    # Check for repetitive patterns that might indicate fuzzing attempts
                    unique_chars = len(set(dep))
                    if unique_chars < 10 and len(dep) > 100:  # Very repetitive string
                        raise ValueError("Dependency name appears to be malformed (too repetitive)")

                valid_deps.append(dep)

        return valid_deps

    @field_validator("mcp_servers")
    @classmethod
    def validate_mcp_servers(cls, v: list[str] | None) -> list[str] | None:
        """Validate MCP server names list."""
        if v is None:
            return v

        # Filter out empty server names and validate remaining ones
        valid_servers = []
        for server_name in v:
            if server_name and server_name.strip():
                server_name = server_name.strip()  # noqa: PLW2901 -- intentional inline normalize

                # Define invalid characters that should not be allowed
                # Allow Unicode letters, numbers, underscore, hyphen, and period
                # Disallow: spaces, slashes, control chars, quotes, etc.
                invalid_chars = r'[\s/\\|;:?*<>"\'\n\t,@#$%&+=]'

                if re.search(invalid_chars, server_name):
                    # Find the first invalid character for better error message
                    match = re.search(invalid_chars, server_name)
                    invalid_char = match.group() if match else "unknown"
                    raise ValueError(
                        f"MCP server name '{server_name}' contains invalid character: '{invalid_char}'. "
                        f"Server names can only contain letters (including Unicode), numbers, underscores, hyphens, and periods."
                    )

                # Check for control characters
                if any(ord(char) < 32 for char in server_name):
                    raise ValueError(f"MCP server name '{server_name}' contains control characters")

                # Length check (count actual characters, not bytes)
                if len(server_name) > 100:
                    raise ValueError(f"MCP server name '{server_name}' is too long (max 100 chars)")

                valid_servers.append(server_name)

        return valid_servers

    @field_validator("agent_role")
    @classmethod
    def validate_agent_role(cls, v: str | None) -> str | None:
        """Validate agent role description."""
        if not v:
            return None

        # Convert to string and normalize
        v = v.strip() if isinstance(v, str) else str(v).strip()

        # Handle invalid/empty values
        if v in ["", "0", "false", "null", "none", "None"]:
            return None

        try:
            # Apply security validation
            v = validate_string_security(
                v, "agent_role", max_length=500, allow_html=False, strict_mode=True
            )
        except SecurityValidationError as e:
            raise ValueError(str(e)) from e

        return v

    # The fields the CLIENT sent, frozen at parse time. Convenience endpoints
    # ASSIGN ``agent_type`` afterwards (``convenience_endpoints.py:128``), and
    # assigning adds to ``model_fields_set`` — so that set can no longer answer
    # "what did the caller ask for?" once a request has been through one. The
    # cross-field rules have always been asked of the parsed body only, and
    # this is what lets them keep being asked of exactly that.
    _client_sent_fields: frozenset[str] = PrivateAttr(default=frozenset())

    @model_validator(mode="after")
    def _snapshot_client_sent_fields(self) -> WorkerCreateRequest:
        if not self._client_sent_fields:
            self._client_sent_fields = frozenset(self.model_fields_set)
        return self

    @model_validator(mode="after")
    def validate_remote_agent_container(self) -> WorkerCreateRequest:
        """``remote_*`` agents need a ``container_id``.

        Deliberately NOT moved to ``validate_effective_worker`` with the other
        cross-field rules. The convenience endpoints assign ``agent_type``
        AFTER parse-time validation (``convenience_endpoints.py:128``), so this
        rule has never applied to a remote worker created through them.
        Evaluating it on the merged row would newly 422 that path — a
        behaviour change this fix does not want.
        """
        if self.agent_type in ("remote_claude", "remote_codex") and not self.container_id:
            raise ValueError(f"container_id is required for {self.agent_type} agent type")
        return self

    @model_validator(mode="after")
    def validate_model_fields(self) -> WorkerCreateRequest:
        """Validate model-related fields together."""
        # Non-pydantic_ai agent types don't need model_provider/model
        non_pydantic_agent_types = {
            "claude_code",
            "codex",
            "remote_claude",
            "remote_codex",
        }
        if self.agent_type in non_pydantic_agent_types:
            return self

        # Validate text fields with security checks. The four prompt-class
        # fields (description, system_prompt, instruction, user_prompt) are
        # free-text LLM prompts where inline backticks and ${...} are
        # legitimate code/schema references, not shell metacharacters —
        # opt them out of the command-injection detector only.
        text_fields: list[tuple[str | None, str, int, bool]] = [
            (self.description, "description", 100_000, True),
            (self.system_prompt, "system_prompt", 1_000_000, True),
            (self.instruction, "instruction", 1_000_000, True),
            (self.user_prompt, "user_prompt", 1_000_000, True),
        ]

        for field_value, field_name, max_length, allow_prompt in text_fields:
            if field_value:
                try:
                    validate_string_security(
                        field_value,
                        field_name,
                        max_length=max_length,
                        allow_html=False,
                        strict_mode=True,
                        allow_prompt_constructs=allow_prompt,
                        prompt_field=True,
                    )
                except SecurityValidationError as e:
                    raise ValueError(str(e)) from e

        # Additional validation to catch schema-violating requests
        # Reject requests with suspicious field combinations or invalid structures

        # Check for completely empty or minimal configurations that might be schema violations
        total_meaningful_fields = sum(
            [
                1 if self.worker_name else 0,
                1 if self.tool_names else 0,
                1 if self.agent_role else 0,
                1 if self.description else 0,
                1 if self.system_prompt else 0,
                1 if self.instruction else 0,
                1 if self.user_prompt else 0,
            ]
        )

        # If only worker_name is provided with minimal other fields, ensure it's a valid simple worker
        if total_meaningful_fields == 1 and not self.agent_role:
            # This is a minimal worker config - allow it if worker_name is valid
            pass

        # Additional validation for specific field patterns that indicate invalid requests
        if self.temperature is not None and (self.temperature < 0.0 or self.temperature > 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")

        # Ensure tool_names list is coherent if provided
        if self.tool_names and len(self.tool_names) > 50:
            raise ValueError("Too many tools specified (maximum 50 allowed)")

        return self

    @field_validator("output_schema")
    @classmethod
    def validate_output_schema(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate that output_schema is a valid JSON schema."""
        if v is None:
            return v

        # Basic JSON schema validation
        if not isinstance(v, dict):
            raise ValueError("output_schema must be a dictionary")  # noqa: RUF100, TRY004

        if "type" not in v:
            raise ValueError("output_schema must have a 'type' field")

        return v

    @field_validator("output_validator")
    @classmethod
    def validate_output_validator_code(cls, v: str | None) -> None:
        """Refuse validator CODE; ``None`` and ``""`` are accepted and stored as None.

        Request-supplied Python used to be ``exec()``'d with full builtins. That
        path is deleted, not gated — see ``src/security/request_overrides.py``.
        """
        reject_output_validator(v)


class WorkerResponse(BaseModel):
    """Worker state returned by GET /workers and POST/PUT /workers."""

    model_config = ConfigDict(protected_namespaces=(), extra="ignore")

    id: int | str | None = Field(default=None, description="Database id for the worker row")
    name: str = Field(description="Worker name (unique per tenant)")
    status: str = Field(description="Current lifecycle status")
    db_path: str | None = Field(default=None, description="Legacy SQLite path")
    is_agent: bool = Field(description="Whether this is an in-process agent worker")
    agent_type: str | None = Field(default=None, description="Agent runtime")
    model_provider: str | None = Field(default=None, description="LLM provider")
    model: str | None = Field(default=None, description="Model name sent upstream")
    system_prompt: str | None = Field(default=None, description="System prompt")
    instruction: str | None = Field(default=None, description="Instruction prompt")

    # Round-trip fields — populated from the persisted `config` JSONB column.
    temperature: float | None = Field(default=None, description="Sampling temperature")
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens in the model response. None = use provider default.",
    )
    tool_names: list[str] = Field(default_factory=list, description="Tool function names")
    output_schema: dict | None = Field(default=None, description="JSON Schema for outputs")
    output_validator: str | None = Field(default=None, description="Python validator source")
    output_retries: int | None = Field(
        default=None,
        description="Per-worker pydantic-ai output-validation retry override (None = provider default).",
    )
    mcp_config: str | dict | None = Field(default=None, description="MCP config preset or dict")
    base_url_override: str | None = Field(default=None, description="OpenAI-compat base URL")
    api_key_override: str | None = Field(default=None, description="Per-worker API key")
    config_revision: str | None = Field(
        default=None,
        description="Revision marker last stamped by the seeder/bootstrap. Null if the row predates the column or the caller didn't stamp one.",
    )


class TextBlock(BaseModel):
    """Text content block for multimodal messages."""

    type: Literal["text"] = Field("text", description="Block discriminator.")
    text: str = Field(..., min_length=1, max_length=5_000_000)


class ImageBlock(BaseModel):
    """Image content block for multimodal messages.

    ``data`` is the base64-encoded image bytes. ``media_type`` is the
    MIME type (image/jpeg, image/png, image/webp, image/gif).
    Decoded payload is capped at ~10 MB per image to prevent OOM
    on the worker pod.
    """

    type: Literal["image"] = Field("image", description="Block discriminator.")
    data: str = Field(
        ..., min_length=1, max_length=14_000_000, description="Base64-encoded image bytes"
    )
    media_type: Literal["image/jpeg", "image/png", "image/webp", "image/gif"] = Field(
        "image/jpeg", description="MIME type of the image."
    )

    @field_validator("data")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Verify the data is valid base64 and within size limits."""
        import base64
        import binascii

        try:
            decoded = base64.b64decode(v, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"Invalid base64 image data: {e}") from e
        if len(decoded) > 10 * 1024 * 1024:
            raise ValueError(f"Decoded image is {len(decoded)} bytes; max 10 MB per image.")
        return v


ContentBlock = Annotated[TextBlock | ImageBlock, Discriminator("type")]


class MessageRequest(BaseModel):
    """Request model for sending messages to agents.

    This model supports both sending messages to existing workers and creating
    new workers on-the-fly. If a worker doesn't exist it is automatically created
    with default settings (temperature=0.1, provider/model from
    ``AUTO_CREATE_MODEL_PROVIDER`` / ``Settings.local_llm_model``) and, notably, with NO
    system prompt and NO instruction — so a mistyped or renamed worker name
    returns 200 from a promptless generic worker rather than an error. Set
    ``require_existing=true`` to turn that into a 404.

    The output_schema and output_validator fields allow per-message structured
    output configuration, overriding the worker's default schema for this specific
    message.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "message": "Analyze the sales data from Q4 2023 and identify the top performing products",
                    "context": {
                        "data_source": "sales_db",
                        "time_period": "Q4-2023",
                        "metrics": ["revenue", "units_sold", "profit_margin"],
                    },
                    "system_prompt": "You are a data analyst specializing in sales performance metrics",
                    "instruction": "Focus on year-over-year growth and seasonal trends",
                },
                {
                    "message": "Extract contact information from: John Doe, john@example.com, 555-1234",
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string", "format": "email"},
                            "phone": {"type": "string"},
                        },
                        "required": ["name", "email"],
                    },
                },
            ]
        },
    )

    # NOTE: trust_content MUST be declared before `message` and the prompt
    # fields. Pydantic v2 field_validators run in declaration order; the
    # message/prompt validators read trust_content from ValidationInfo.data,
    # which only contains fields validated BEFORE the current one.
    trust_content: bool = Field(
        default=False,
        description=(
            "Skip command-injection pattern detection on message + prompt "
            "fields. For trusted callers submitting document-as-text content "
            "(markdown with code snippets, scraped pages with shell examples, "
            "LLM ingestion pipelines) where curl/bash/$(...)/&& are content, "
            "not adversarial. All other security checks (null bytes, control "
            "chars, length cap, SQL/XSS/path-traversal/LDAP) remain active. "
            "License-authz is the trust boundary that gates who can set this."
        ),
    )

    message: str | list[ContentBlock] = Field(
        ...,
        description=(
            "Message to send to the agent. Either a plain string (~1M tokens at "
            "5 chars/token, max 5M chars) or a list of typed content blocks for "
            "multimodal input (text + image). Multimodal lists are only supported "
            "by agent_type='pydantic_ai' workers configured against a "
            "vision-capable model (claude-sonnet-4-5, gpt-4o, gemini-2-pro, gemma-4)."
        ),
    )
    context: dict[str, Any] | None = Field(None, description="Additional context for the message")
    system_prompt: str | None = Field(
        None, max_length=1_000_000, description="Override system prompt"
    )
    instruction: str | None = Field(None, max_length=1_000_000, description="Override instruction")
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Per-message output schema override",
    )
    output_validator: str | None = Field(
        default=None,
        description=(
            "UNSUPPORTED. Must be null or empty: request-supplied validator code is "
            "refused with 422 output_validator_code_not_supported. Use output_schema."
        ),
    )
    disable_state_storage: bool = Field(
        default=False,
        description="Skip database operations for auto-created workers",
    )
    require_existing: bool = Field(
        default=False,
        description=(
            "Refuse to auto-create. When true, a send to a worker name that does "
            "not exist returns 404 instead of silently provisioning a PROMPTLESS "
            "generic worker and answering 200. For callers that own a closed "
            "worker vocabulary (the n8n Workers node, the AA conveyor's step "
            "dispatchers): a typo or a renamed worker then arrives as an error "
            "their error-reader can actually read, instead of as a degraded "
            "success. Default false — auto-create stays the default behaviour and "
            "nothing existing changes."
        ),
        json_schema_extra={"example": False},
    )
    model_override: str | None = Field(
        default=None,
        description="Override model for this message, routes to correct MLX instance",
    )
    work_tier: Literal["small", "medium", "high"] | None = Field(
        default=None,
        description=(
            "Work-size tier for this message. Selects the tenant's per-tier BYO "
            "LLM row (small/medium/high → flash/reasoning/frontier model classes), "
            "falling back to the tenant default when that tier is unconfigured. "
            "Omit to use the tenant default (Phase-1 behavior)."
        ),
    )
    product: str | None = Field(
        default=None,
        description=(
            "Product/app key (licenses App.key). When set, the LLM BYO posture "
            "(byo_required / platform_only / local_allowed) is resolved "
            "per-(tenant, product): a tenant_app_byo_mode override for this product "
            "wins over the tenant-wide byo_mode; absent an override the tenant-wide "
            "value is used (backward compatible). Omit for tenant-wide posture "
            "(Phase-1 behavior). Mirrors the modality resolve endpoint's product param."
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        description=(
            "Per-message max output tokens. Folded into the worker config for THIS "
            "message (mirrors work_tier), so callers whose worker config lives under "
            "a different tenant than the runtime tenant (e.g. requirements_management "
            "forwards the end-user tenant, whose per-tenant worker row is bare "
            "auto-created) can still apply their curated budget. None ⇒ keep the "
            "worker record's value (or the provider default). Clamped to 16384 for "
            "self-hosted backends downstream."
        ),
        json_schema_extra={"example": 4000},
    )
    enable_reasoning: bool | None = Field(
        default=None,
        description=(
            "Per-message chain-of-thought switch for reasoning-capable self-hosted "
            "models. Folded into the worker config for THIS message (mirrors "
            "max_tokens/work_tier) and joined into the worker-cache key, so an "
            "opting-out caller never poisons the cached agent a later caller reuses. "
            "None ⇒ defer to the provider default: NOTHING is sent downstream for "
            "inference_router/mlx/vlm_local, but `vllm` (direct pgx) is sent "
            "enable_thinking=false, because the model it serves defaults "
            "chain-of-thought ON and an absent switch there costs ~12-15x the "
            "completion tokens for the same answer. False ⇒ suppress reasoning; True ⇒ "
            "request it. Only reaches self-hosted backends: inference_router receives "
            "extra_body.enable_reasoning (which IR rewrites to "
            "chat_template_kwargs.enable_thinking), vllm/mlx/vlm_local receive "
            "extra_body.chat_template_kwargs.enable_thinking directly, and external "
            "providers (openai/anthropic/groq/…) get nothing — no verified shape "
            "exists for them. Set False when a structured-output schema must not "
            "have its max_tokens budget eaten by a <think> block."
        ),
        json_schema_extra={"example": False},
    )

    seed: StrictInt | None = Field(
        default=None,
        description=(
            "Per-message sampling seed, applied to THIS run only (passed to the "
            "model call's settings, not baked into the cached agent, so any number "
            "of distinct seeds share one worker-cache slot). Sent to the model as "
            "the OpenAI-compatible top-level `seed` and wins over the worker "
            "record's seed. None ⇒ the record's seed, if any; if none, NO seed key "
            "is sent. Only pydantic_ai workers accept it (422 SEED_NOT_SUPPORTED "
            "otherwise). The effective value is echoed back in usage_metrics.seed."
        ),
        json_schema_extra={"example": 42},
    )

    repetition_penalty: float | None = Field(
        default=None,
        gt=0.0,
        le=2.0,
        description=(
            "Per-message vLLM repetition penalty, applied to THIS run only (same "
            "rules as seed: per run, one cache slot, wins over the worker record's "
            "value, None ⇒ the record's value or nothing). Sent only to self-hosted "
            "backends. Only pydantic_ai workers accept it (422 "
            "SAMPLING_OVERRIDE_NOT_SUPPORTED otherwise). Echoed in "
            "usage_metrics.repetition_penalty."
        ),
        json_schema_extra={"example": 1.05},
    )

    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Per-message OpenAI-compatible frequency_penalty, applied to THIS run "
            "only (same rules as seed). Echoed in usage_metrics.frequency_penalty."
        ),
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Per-message sampling temperature, applied to THIS run only (same rules "
            "as seed: passed to the model call's settings, not baked into the cached "
            "agent, wins over the worker record's temperature). None ⇒ the worker "
            "record's temperature exactly as before (auto-created workers: 0.1). "
            "For callers that need a deterministic verdict (judges) regardless of how "
            "the worker was provisioned. Only pydantic_ai workers accept it (422 "
            "SAMPLING_OVERRIDE_NOT_SUPPORTED otherwise). Echoed in "
            "usage_metrics.temperature; when the model refuses sampling settings "
            "(OpenAI reasoning models, Anthropic models that disallow them) the "
            "value is NOT sent, usage_metrics.temperature is null and "
            "usage_metrics.temperature_not_sent says why."
        ),
        json_schema_extra={"example": 0.0},
    )

    enforce_schema_bounds: StrictBool | None = Field(
        default=None,
        description=(
            "Per-message opt-in, retry then fail: when the output breaks a "
            "minItems/maxItems/minLength/maxLength bound (bounds the decoding grammar "
            "cannot carry), re-ask the model with feedback naming the path and bound "
            "within the worker's output_retries, then fail THIS run with HTTP 422 "
            "SCHEMA_BOUND_VIOLATION if it still violates. Wins over the worker "
            "record's enforce_schema_bounds; None ⇒ the record's value (default "
            "false ⇒ advisory, paths reported in usage_metrics.bound_violations)."
        ),
    )

    correlation_id: UUID | None = Field(
        default=None,
        description=(
            "End-to-end join key (card / execution correlation). Forwarded to "
            "inference-router as X-Correlation-Id and echoed in "
            "usage_metrics.correlation_id, so a worker_states row joins IR's "
            "inference_call ledger. Resolved against the inbound X-Correlation-Id "
            "header exactly like graph execution does "
            "(first_valid_correlation_id): a valid UUID header wins, a malformed "
            "header does not beat a valid body value. Omitted ⇒ the header alone "
            "decides, as before."
        ),
    )

    prompt_versions: dict[str, StrictInt] | None = Field(
        default=None,
        max_length=64,
        description=(
            "Card prompt-version stamp, {prompt_id: version} (P4 contract §4). DARK: "
            "honoured only when PROMPTS_READ_REQUIRES_PUBLISHED is on AND the worker's "
            "config_revision names one of these prompt_ids; the call then runs that "
            "exact ledger version (PUBLISHED or CANARY; any other status is 409 "
            "{error: prompt_version_not_runnable, status, prompt_id, version}; a "
            "version not in the ledger is 404 {error: prompt_version_not_found, "
            "prompt_id, version}) as a per-call override that never touches the "
            "worker's cached instance, and metadata.worker_config_revision reports "
            "'<prompt_id>@<version>'. A per-message system_prompt is never refused: it "
            "runs, the pin is not applied, and metadata.worker_config_revision is "
            "'unversioned:<sha12>' of what ran. metadata.prompt_pin is one of: "
            "honoured, ignored:flag_off, ignored:message_override, "
            "ignored:no_matching_prompt, ignored:no_stamp, label_content_mismatch, "
            "canary_held, tenant_forbidden (flag off: only when a stamp was sent, and "
            "always ignored:flag_off). A NON-platform tenant's stamp is never applied "
            "and never read from the ledger (prompt versions are system data): the "
            "worker's own prompt runs and prompt_pin is tenant_forbidden."
        ),
        json_schema_extra={"example": {"keyword-picker": 3}},
    )

    @field_validator("prompt_versions")
    @classmethod
    def validate_prompt_versions(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        """Same key charset as the ledger (``PROMPT_ID_RE``); versions are >= 1."""
        if v is None:
            return None
        for prompt_id, version in v.items():
            if not PROMPT_ID_RE.match(prompt_id):
                raise ValueError(
                    f"prompt_versions key {prompt_id[:64]!r} must match "
                    "[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
                )
            if version < 1:
                raise ValueError(f"prompt_versions[{prompt_id!r}] must be an integer >= 1")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_message_max_tokens(cls, v: int | None) -> int | None:
        """Bound per-message max_tokens to the same 1..16384 range as registration.

        Upper bound protects self-hosted vLLM from the CUTLASS FP4-MoE kernel
        crash near full context under structured output (see WorkerCreateRequest).
        """
        if v is None:
            return v
        if v < 1 or v > 16384:
            raise ValueError("max_tokens must be between 1 and 16384")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(
        cls, v: str | list[ContentBlock], info: ValidationInfo
    ) -> str | list[ContentBlock]:
        """Validate message content for security.

        For string messages, runs the standard security validator. For
        multimodal block lists, validates each text block's content and
        ensures total non-image payload doesn't exceed the string-mode cap.
        Image blocks are validated by ImageBlock's own validator.
        """
        trust_content = bool(info.data.get("trust_content", False))
        if isinstance(v, list):
            if not v:
                raise ValueError("Message content blocks cannot be empty")
            total_text = 0
            for block in v:
                if isinstance(block, TextBlock):
                    total_text += len(block.text)
                    try:
                        validate_string_security(
                            block.text,
                            "message.text",
                            max_length=5_000_000,
                            allow_html=False,
                            strict_mode=True,
                            allow_prompt_constructs=trust_content,
                            prompt_field=True,
                        )
                    except SecurityValidationError as e:
                        raise ValueError(str(e)) from e
            if total_text > 5_000_000:
                raise ValueError("Aggregate text across content blocks exceeds 5M chars")
            return v

        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        try:
            validated_message = validate_string_security(
                v.strip(),
                "message",
                max_length=5_000_000,
                allow_html=False,
                strict_mode=True,
                allow_prompt_constructs=trust_content,
                prompt_field=True,
            )
            return validated_message
        except SecurityValidationError as e:
            raise ValueError(str(e)) from e

    @field_validator("context")
    @classmethod
    def validate_context(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate context dictionary for JSON safety."""
        if v:
            # Check JSON safety
            if not is_safe_json_data(v, max_depth=10):
                raise ValueError("Context contains unsafe JSON structure (too deep or too large)")

            # Check serialized size
            if len(str(v)) > 10000:
                raise ValueError("Context is too large (max 10KB when serialized)")
        return v

    @field_validator("system_prompt", "instruction")
    @classmethod
    def validate_prompts(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate prompt fields for security."""
        if v is not None:
            trust_content = bool(info.data.get("trust_content", False))
            try:
                return validate_string_security(
                    v,
                    "prompt_field",
                    max_length=1_000_000,
                    allow_html=False,
                    strict_mode=True,
                    allow_prompt_constructs=trust_content,
                    prompt_field=True,
                )
            except SecurityValidationError as e:
                raise ValueError(str(e)) from e
        return v

    @field_validator("output_schema")
    @classmethod
    def validate_output_schema(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate that output_schema is a valid JSON schema."""
        if v is None:
            return v

        # Basic JSON schema validation
        if not isinstance(v, dict):
            raise ValueError("output_schema must be a dictionary")  # noqa: RUF100, TRY004

        if "type" not in v:
            raise ValueError("output_schema must have a 'type' field")

        return v

    @field_validator("output_validator")
    @classmethod
    def validate_output_validator_code(cls, v: str | None) -> None:
        """Refuse validator CODE; ``None`` and ``""`` are accepted and stored as None.

        Request-supplied Python used to be ``exec()``'d with full builtins. That
        path is deleted, not gated — see ``src/security/request_overrides.py``.
        """
        reject_output_validator(v)


class MessageResponse(BaseModel):
    """Response model for agent messages with structured output support."""

    model_config = ConfigDict(extra="ignore")

    output: Any = Field(
        ...,
        description="The agent's output - can be string, dict, list, or any structured data",
    )
    output_type: Literal["string", "object", "array", "boolean", "enum", "other"] = Field(
        ..., description="Type of the output returned"
    )
    output_schema_used: dict[str, Any] | None = Field(
        default=None,
        description="The JSON schema that was used for this output, if any",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-call metadata. Always carries the prompt-attribution block "
            "(roadmap A5): `worker_config_revision` (the raw label on the worker "
            "row), `prompt_id` + `prompt_version` (parsed from it when it has the "
            "`<prompt_id>@<version>` shape S4 stamps, else null), "
            "`system_prompt_sha256` (sha256 of the system prompt the agent "
            "instance ACTUALLY ran with — join it against "
            "`prompt_versions.system_prompt_sha256` to verify the label), and "
            "`system_prompt_source` (`instance` | `config` | `none`). "
            "`system_prompt_sha256` digests the system prompt ALONE and excludes "
            "any per-message `instruction`, so two calls differing only in "
            "`instruction` run different text and return the same digest. "
            "Also carries the caller's `context` verbatim when the request had one. "
            "See src/api/workers/prompt_attribution.py."
        ),
    )
    usage_metrics: dict | None = None
    # What ACTUALLY served the call, read back from inference-router and passed
    # through unchanged (also inside usage_metrics). None when not reported.
    served_provider: str | None = Field(
        default=None, description="Provider that served the call (inference-router read-back)."
    )
    served_model: str | None = Field(
        default=None, description="Served model name (inference-router read-back)."
    )
    served_model_root: str | None = Field(
        default=None,
        description="Model root behind the served name (inference-router read-back).",
    )

    @model_validator(mode="after")
    def _lift_served_fields(self) -> MessageResponse:
        um = self.usage_metrics or {}
        for key in ("served_provider", "served_model", "served_model_root"):
            if getattr(self, key) is None and isinstance(um.get(key), str) and um[key]:
                setattr(self, key, um[key])
        return self


class PromptSwapRequest(BaseModel):
    """Body for ``PUT /workers/{worker_name}/prompt`` — atomic prompt swap.

    Implements AA design decision OD-4: an atomic compare-and-swap that
    changes ONLY the worker's system prompt, guarded by the immutable
    output schema (P-MP-01) and an optional canary version id.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="ignore")

    new_system_prompt: str | None = Field(
        default=None,
        min_length=1,
        max_length=1_000_000,
        description="Replacement system prompt body (free text). Exactly one of this and "
        "prompt_version is required.",
    )
    prompt_version: str | None = Field(
        default=None,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}@\d{1,9}$",
        description="P4 A3: '<prompt_id>@<N>' — swap in that ledger version's content "
        "(PUBLISHED or CANARY; else 409 prompt_version_not_runnable, missing → 404) and set "
        "config_revision to exactly '<prompt_id>@<N>'. Exactly one of this and "
        "new_system_prompt is required.",
    )
    expected_prompt_sha256: str | None = Field(
        default=None,
        description="If set, the swap only applies when the live prompt's sha256 matches "
        "(compare-and-swap; 409 on mismatch).",
    )
    expected_output_schema: dict[str, Any] | None = Field(
        default=None,
        description="If set, asserted byte-equal to the live output_schema (P-MP-01 "
        "defense-in-depth; 422 on mismatch). The swap never mutates output_schema regardless.",
    )
    canary_version: str | None = Field(
        default=None,
        max_length=255,
        description="Audit marker — the candidate (canary) worker name the prompt came from. "
        "Stored as config_revision + in the worker_states audit row.",
    )
    reason: str | None = Field(default=None, max_length=2000, description="Audit reason")

    @model_validator(mode="before")
    @classmethod
    def _neither_source_is_main_s_missing_field(cls, data: Any) -> Any:
        """A body naming NEITHER source gets main's exact 422.

        Before P4 ``new_system_prompt`` was simply required, so a caller that
        sends neither must keep receiving ``body -> new_system_prompt: Field
        required`` (type ``missing``), byte-identical, not the new "exactly
        one of" wording. That wording is reserved for a body naming BOTH.
        """
        if (
            isinstance(data, dict)
            and "new_system_prompt" not in data
            and data.get("prompt_version") is None
        ):
            raise ValidationError.from_exception_data(
                cls.__name__,
                [InitErrorDetails(type="missing", loc=("new_system_prompt",), input=data)],
            )
        return data

    @model_validator(mode="after")
    def _exactly_one_source(self) -> PromptSwapRequest:
        if (self.new_system_prompt is None) == (self.prompt_version is None):
            raise ValueError("send exactly one of new_system_prompt and prompt_version")
        return self

    @field_validator("new_system_prompt")
    @classmethod
    def _validate_prompt(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            validate_string_security(
                v,
                "new_system_prompt",
                max_length=1_000_000,
                allow_html=False,
                strict_mode=True,
                allow_prompt_constructs=True,
                prompt_field=True,
            )
        except SecurityValidationError as e:
            raise ValueError(str(e)) from e
        return v


class PromptSwapResponse(BaseModel):
    """Result of an atomic prompt swap."""

    model_config = ConfigDict(extra="ignore")

    swapped: bool = Field(description="True if the live worker's prompt was replaced")
    worker_name: str = Field(description="The base worker whose prompt was swapped")
    output_schema_preserved: bool = Field(
        default=True, description="Always true — the swap never touches output_schema (P-MP-01)"
    )
    canary_version: str | None = Field(default=None, description="Audit marker from the request")
    prompt_sha256: str | None = Field(
        default=None, description="sha256 of the newly-applied system prompt"
    )
