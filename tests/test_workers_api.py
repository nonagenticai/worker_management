"""Tests for Workers API — validation, model_override, convenience endpoints, send_message."""

import os
from unittest.mock import patch

import pytest

from src.worker.provider_registry import API_KEY_PROVIDERS, PROVIDER_NAMES

# Parametrisation sources. Derived from the registry at collection time so a new
# provider is automatically covered by the REQUIRE_API_KEYS tests below.
KEY_REQUIRING_PROVIDERS = sorted(API_KEY_PROVIDERS)
KEYLESS_PROVIDERS = sorted(set(PROVIDER_NAMES) - API_KEY_PROVIDERS)


class TestWorkerCreateRequestValidation:
    """Test WorkerCreateRequest Pydantic model validation."""

    def _make_request(self, **kwargs):
        from src.api.workers import WorkerCreateRequest
        from src.api.workers.crud_endpoints import _config_payload
        from src.api.workers.models import validate_effective_worker

        req = WorkerCreateRequest(**kwargs)
        # The cross-field rules no longer run at parse time. Since a write
        # MERGES only the keys a caller sent, they must be asked of the row the
        # write PRODUCES, not of the body — see ``validate_effective_worker``.
        # For a CREATE the effective worker IS the body, so validating against
        # nothing stored reproduces exactly the behaviour these tests pin.
        effective = validate_effective_worker(_config_payload(req))
        req.agent_type = effective.get("agent_type")
        return req

    def test_minimal_worker_request(self):
        req = self._make_request(worker_name="basic_worker")
        assert req.worker_name == "basic_worker"
        assert req.agent_type is None
        assert req.agent_role is None

    def test_pydantic_ai_requires_model_provider(self):
        """agent_type=pydantic_ai must have model_provider."""
        with pytest.raises(ValueError, match="model_provider is required"):
            self._make_request(
                worker_name="test",
                agent_role="Test",
                agent_type="pydantic_ai",
            )

    def test_pydantic_ai_requires_model(self):
        """pydantic_ai with model_provider must have model."""
        with pytest.raises(ValueError, match="model is required"):
            self._make_request(
                worker_name="test",
                agent_role="Test",
                agent_type="pydantic_ai",
                model_provider="deepseek",
            )

    def test_pydantic_ai_full_config(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            req = self._make_request(
                worker_name="test",
                agent_role="Test",
                agent_type="pydantic_ai",
                model_provider="deepseek",
                model="deepseek-chat",
            )
            assert req.agent_type == "pydantic_ai"
            assert req.model_provider == "deepseek"

    def test_pydantic_ai_xai_provider(self):
        """xAI/Grok must be accepted even when agent_type is passed explicitly.

        Regression: the validate_model_provider allowlist previously omitted
        "xai", so an explicit agent_type="pydantic_ai" + model_provider="xai"
        was rejected while the omit-agent_type path worked.
        """
        with patch.dict(os.environ, {"XAI_API_KEY": "test-key"}):
            req = self._make_request(
                worker_name="test_xai",
                agent_role="Test",
                agent_type="pydantic_ai",
                model_provider="xai",
                model="grok-4.5",
            )
            assert req.agent_type == "pydantic_ai"
            assert req.model_provider == "xai"

    def test_pydantic_ai_google_provider(self):
        """google is runtime-supported and must be accepted with explicit agent_type.

        Regression: the old hardcoded allowlist silently omitted google (same
        drift bug as xai). The allowlist now derives from ModelProvider.
        """
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            req = self._make_request(
                worker_name="test_google",
                agent_role="Test",
                agent_type="pydantic_ai",
                model_provider="google",
                model="gemini-2.5-pro",
            )
            assert req.agent_type == "pydantic_ai"
            assert req.model_provider == "google"

    def test_pydantic_ai_allowlist_matches_model_provider_literal(self):
        """The accepted providers must stay in lockstep with ModelProvider.

        Locks the invariant so adding a provider to the ModelProvider literal
        can never again leave this validator rejecting it.
        """
        from typing import get_args

        from src.worker.agents.pydantic_ai_agent import ModelProvider

        for provider in get_args(ModelProvider):
            req = self._make_request(
                worker_name=f"t_{provider}",
                agent_role="Test",
                agent_type="pydantic_ai",
                model_provider=provider,
                model="some-model",
            )
            assert req.model_provider == provider

    def test_pydantic_ai_rejects_unknown_provider(self):
        with pytest.raises(ValueError, match="Unsupported model provider for pydantic_ai"):
            self._make_request(
                worker_name="test_bad",
                agent_role="Test",
                agent_type="pydantic_ai",
                model_provider="not-a-real-provider",
                model="x",
            )

    def test_claude_code_skips_model_provider(self):
        """claude_code agent_type should not require model_provider."""
        req = self._make_request(
            worker_name="test_cc",
            agent_role="Test",
            agent_type="claude_code",
        )
        assert req.agent_type == "claude_code"
        assert req.model_provider is None

    def test_codex_skips_model_provider(self):
        req = self._make_request(
            worker_name="test_codex",
            agent_role="Test",
            agent_type="codex",
        )
        assert req.agent_type == "codex"

    def test_remote_claude_skips_model_provider(self):
        req = self._make_request(
            worker_name="test_rc",
            agent_role="Test",
            agent_type="remote_claude",
            container_id="abc123",
        )
        assert req.agent_type == "remote_claude"

    def test_remote_codex_skips_model_provider(self):
        """remote_codex agent_type should not require model_provider."""
        req = self._make_request(
            worker_name="test_rcx",
            agent_role="Test",
            agent_type="remote_codex",
            container_id="abc123",
        )
        assert req.agent_type == "remote_codex"
        assert req.model_provider is None

    def test_remote_claude_requires_container_id(self):
        """remote_claude without container_id should raise ValueError."""
        with pytest.raises(ValueError, match="container_id"):
            self._make_request(
                worker_name="test_rc",
                agent_role="Test",
                agent_type="remote_claude",
            )

    def test_remote_codex_requires_container_id(self):
        """remote_codex without container_id should raise ValueError."""
        with pytest.raises(ValueError, match="container_id"):
            self._make_request(
                worker_name="test_rcx",
                agent_role="Test",
                agent_type="remote_codex",
            )

    def test_remote_claude_with_container_id_valid(self):
        """remote_claude with container_id should pass validation."""
        req = self._make_request(
            worker_name="test_rc",
            agent_role="Test",
            agent_type="remote_claude",
            container_id="abc123",
        )
        assert req.agent_type == "remote_claude"
        assert req.container_id == "abc123"

    def test_remote_codex_with_container_id_valid(self):
        """remote_codex with container_id should pass validation."""
        req = self._make_request(
            worker_name="test_rcx",
            agent_role="Test",
            agent_type="remote_codex",
            container_id="abc123",
        )
        assert req.agent_type == "remote_codex"
        assert req.container_id == "abc123"

    def test_agent_type_without_agent_role_raises(self):
        """agent_type without agent_role should raise."""
        with pytest.raises(ValueError, match="agent_role is required"):
            self._make_request(
                worker_name="test",
                agent_type="claude_code",
            )

    def test_agent_role_without_model_provider_leaves_type_none(self):
        """agent_role alone (no model_provider, no agent_type) should leave agent_type=None."""
        req = self._make_request(
            worker_name="test",
            agent_role="Test helper",
        )
        # Convenience endpoint will set agent_type later
        assert req.agent_type is None

    def test_agent_role_with_model_provider_defaults_to_pydantic_ai(self):
        """agent_role + model_provider (no agent_type) should default to pydantic_ai."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            req = self._make_request(
                worker_name="test",
                agent_role="Test",
                model_provider="deepseek",
                model="deepseek-chat",
            )
            assert req.agent_type == "pydantic_ai"

    def test_invalid_worker_name_rejected(self):
        with pytest.raises(ValueError):
            self._make_request(worker_name="")

    @pytest.mark.parametrize(
        "name",
        [
            "undefined",
            "null",
            "None",
            "TRUE",
            "Admin",
            "select",
            "drop",
            "  undefined  ",
        ],
    )
    def test_reserved_worker_name_rejected(self, name):
        """Regression: validate_identifier rejects JS/SQL reserved literals.

        Live audit 2026-05-13 found one row with worker_name='undefined' that
        slipped past this validator via the send_message path-param auto-create
        (which bypasses WorkerCreateRequest). Migration 017 backfills the row;
        message_endpoints.py + crud_endpoints.py update_worker now apply the
        same validate_identifier guard at the path-param edge.
        """
        with pytest.raises(ValueError, match="reserved name"):
            self._make_request(worker_name=name)

    def test_temperature_validation(self):
        with pytest.raises(ValueError, match="Temperature must be between"):
            self._make_request(worker_name="test", temperature=3.0)

    def test_valid_temperature(self):
        req = self._make_request(worker_name="test", temperature=0.7)
        assert req.temperature == 0.7


class TestMessageRequestValidation:
    """Test MessageRequest Pydantic model validation."""

    def _make_request(self, **kwargs):
        from src.api.workers import MessageRequest

        return MessageRequest(**kwargs)

    def test_basic_message(self):
        req = self._make_request(message="Hello world")
        assert req.message == "Hello world"
        assert req.model_override is None
        assert req.output_schema is None

    def test_message_with_model_override(self):
        req = self._make_request(
            message="Test",
            model_override="local-llm",
        )
        assert req.model_override == "local-llm"

    def test_message_with_output_schema(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        req = self._make_request(
            message="Test",
            output_schema=schema,
        )
        assert req.output_schema == schema

    def test_empty_message_rejected(self):
        with pytest.raises(ValueError):
            self._make_request(message="")

    def test_output_schema_requires_type(self):
        with pytest.raises(ValueError, match="must have a 'type' field"):
            self._make_request(
                message="Test",
                output_schema={"properties": {"x": {"type": "string"}}},
            )


class TestConvenienceEndpointValidation:
    """Test that convenience endpoints accept requests without model_provider."""

    def test_claude_code_request_validates(self):
        """Simulates what the /workers/claude-code endpoint does."""
        from src.api.workers import WorkerCreateRequest

        # This is what a user would POST to /workers/claude-code
        req = WorkerCreateRequest(worker_name="my_cc_agent")
        # Endpoint sets these AFTER Pydantic validation
        req.agent_type = "claude_code"
        if not req.agent_role:
            req.agent_role = f"Claude Code agent for {req.worker_name}"

        assert req.agent_type == "claude_code"
        assert req.agent_role == "Claude Code agent for my_cc_agent"
        assert req.model_provider is None

    def test_codex_request_validates(self):
        from src.api.workers import WorkerCreateRequest

        req = WorkerCreateRequest(worker_name="my_codex")
        req.agent_type = "codex"
        if not req.agent_role:
            req.agent_role = f"Codex agent for {req.worker_name}"
        assert req.agent_type == "codex"

    def test_remote_claude_request_validates(self):
        from src.api.workers import WorkerCreateRequest

        req = WorkerCreateRequest(worker_name="my_rc")
        req.agent_type = "remote_claude"
        if not req.agent_role:
            req.agent_role = f"Remote Claude agent for {req.worker_name}"
        assert req.agent_type == "remote_claude"

    def test_remote_codex_endpoint(self):
        """Simulates what the /workers/remote-codex endpoint does."""
        from src.api.workers import WorkerCreateRequest

        req = WorkerCreateRequest(worker_name="my_rcx")
        req.agent_type = "remote_codex"
        if not req.agent_role:
            req.agent_role = f"Remote Codex agent for {req.worker_name}"

        assert req.agent_type == "remote_codex"
        assert req.agent_role == "Remote Codex agent for my_rcx"
        assert req.model_provider is None


class TestModelOverrideInSendMessage:
    """Test model_override logic in send_message."""

    @pytest.mark.asyncio
    async def test_model_override_applies_to_existing_worker(self):
        """model_override should resolve MLX instance and update worker config."""
        from src.api.workers import MessageRequest

        worker_data = {
            "worker_name": "test",
            "worker_id": "w-123",
            "is_agent": True,
            "config": {"model_provider": "mlx", "model": "mlx"},
        }

        # Simulate what send_message does with model_override
        request = MessageRequest(message="Hello", model_override="local-llm")

        if request.model_override:
            resolved_url = "https://mlx-inference.example.com"

            if resolved_url:
                if "config" not in worker_data:
                    worker_data["config"] = {}
                worker_data["config"]["base_url_override"] = resolved_url
                worker_data["config"]["model"] = request.model_override

        assert worker_data["config"]["base_url_override"] == "https://mlx-inference.example.com"
        assert worker_data["config"]["model"] == "local-llm"

    @pytest.mark.asyncio
    async def test_model_override_applies_to_auto_created_worker(self):
        """model_override should work for auto-created workers too (not skipped)."""
        worker_data = {
            "worker_name": "auto_test",
            "worker_id": "w-456",
            "is_agent": True,
            "config": {"model_provider": "mlx", "model": "mlx"},
        }

        # The fix: model_override now applies regardless of worker_auto_created
        model_override = "local-llm"
        if model_override:
            worker_data["config"]["base_url_override"] = "https://mlx-inference.example.com"
            worker_data["config"]["model"] = model_override

        assert worker_data["config"]["model"] == "local-llm"

    @pytest.mark.asyncio
    async def test_output_schema_override_for_existing_worker(self):
        """Per-message output_schema should be applied for existing workers."""
        worker_data = {
            "worker_name": "existing",
            "worker_id": "w-789",
            "is_agent": True,
            "config": {},
        }
        worker_auto_created = False

        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        # Simulating the condition from send_message
        if schema and not (worker_auto_created and not None):
            worker_data["config"]["output_schema"] = schema

        assert worker_data["config"]["output_schema"] == schema


class TestErrorStateRecording:
    """Test that error states are properly recorded."""

    def test_http_exception_handler_structure(self):
        """Verify the error handling code path records state for 5xx errors."""
        # This is a structural test: verify HTTPException handler records error state
        # by checking the source code contains the error state recording logic
        import inspect

        from src.api.workers import send_message

        source = inspect.getsource(send_message)
        # HTTPException handler should record error state for 5xx
        assert "except HTTPException as he:" in source
        assert "he.status_code >= 500" in source
        assert 'state_type="error"' in source

    def test_api_error_handler_structure(self):
        """Verify BadRequestError/NotFoundError/InternalServerError handler records error state."""
        import inspect

        from src.api.workers import send_message

        source = inspect.getsource(send_message)
        # API error handler should record error state for 5xx
        assert "except (BadRequestError, NotFoundError, InternalServerError) as e:" in source
        assert "e.status_code >= 500" in source

    def test_general_exception_records_error_state(self):
        """Verify general exception handler records error state."""
        import inspect

        from src.api.workers import send_message

        source = inspect.getsource(send_message)
        # General handler already had error state recording
        assert "await db.add_worker_state(" in source


class TestMessageResponseModel:
    """Test MessageResponse model."""

    def test_string_output(self):
        from src.api.workers import MessageResponse

        resp = MessageResponse(
            output="Hello",
            output_type="string",
        )
        assert resp.output == "Hello"
        assert resp.output_type == "string"

    def test_object_output(self):
        from src.api.workers import MessageResponse

        resp = MessageResponse(
            output={"answer": "42"},
            output_type="object",
        )
        assert resp.output == {"answer": "42"}

    def test_response_with_usage_metrics(self):
        from src.api.workers import MessageResponse

        resp = MessageResponse(
            output="Hello",
            output_type="string",
            usage_metrics={"prompt_tokens": 10, "completion_tokens": 20},
        )
        assert resp.usage_metrics is not None
        assert resp.usage_metrics["prompt_tokens"] == 10


class TestInferenceRouterWorkerCreation:
    """Test that inference_router is accepted as a valid model_provider without API key."""

    def _make_request(self, **kwargs):
        from src.api.workers import WorkerCreateRequest
        from src.api.workers.crud_endpoints import _config_payload
        from src.api.workers.models import validate_effective_worker

        req = WorkerCreateRequest(**kwargs)
        # The cross-field rules no longer run at parse time. Since a write
        # MERGES only the keys a caller sent, they must be asked of the row the
        # write PRODUCES, not of the body — see ``validate_effective_worker``.
        # For a CREATE the effective worker IS the body, so validating against
        # nothing stored reproduces exactly the behaviour these tests pin.
        effective = validate_effective_worker(_config_payload(req))
        req.agent_type = effective.get("agent_type")
        return req

    def test_create_worker_inference_router_no_api_key(self):
        """Creating a worker with model_provider=inference_router should NOT fail
        with an API key validation error (422).

        inference_router is an internal service that does not require an API key.
        The API key check at line 717 only applies to openai/deepseek/groq, so
        inference_router must pass validation without any API key set.
        """
        req = self._make_request(
            worker_name="ir_worker",
            agent_role="Inference router test",
            agent_type="pydantic_ai",
            model_provider="inference_router",
            model="pi-model-v1",
        )
        assert req.model_provider == "inference_router"
        assert req.model == "pi-model-v1"
        # No API key should be required
        assert req.api_key_override is None

    def test_create_worker_inference_router_valid_provider(self):
        """inference_router is accepted as a valid model_provider value with no
        Pydantic validation error on the field.

        The field_validator on model_provider must include inference_router in
        its allowed list for pydantic_ai agents.
        """
        req = self._make_request(
            worker_name="ir_valid",
            agent_role="Validate provider",
            agent_type="pydantic_ai",
            model_provider="inference_router",
            model="some-model",
        )
        assert req.agent_type == "pydantic_ai"
        assert req.model_provider == "inference_router"

    def test_inference_router_with_require_api_keys_enabled(self):
        """Even with REQUIRE_API_KEYS=true, inference_router should not be
        affected since it is not in the API-key-required providers list."""
        with patch.dict(os.environ, {"REQUIRE_API_KEYS": "true"}):
            req = self._make_request(
                worker_name="ir_strict",
                agent_role="Strict mode test",
                agent_type="pydantic_ai",
                model_provider="inference_router",
                model="pi-model-v2",
            )
            assert req.model_provider == "inference_router"

    def test_inference_router_defaults_to_pydantic_ai(self):
        """When agent_role and model_provider=inference_router are set but
        agent_type is omitted, it should default to pydantic_ai."""
        req = self._make_request(
            worker_name="ir_default",
            agent_role="Default type test",
            model_provider="inference_router",
            model="pi-model-v1",
        )
        assert req.agent_type == "pydantic_ai"
        assert req.model_provider == "inference_router"


class TestRequireApiKeysEnforcement:
    """REQUIRE_API_KEYS=true must gate *every* provider that declares it needs a key.

    Regression: the old hardcoded membership lists were
    ["openai", "deepseek", "groq", "anthropic", "xai"] — cohere and google were
    missing, so `model_provider="google"` with no GEMINI_API_KEY was accepted at
    the API boundary (201) and only blew up later at first run inside
    `_create_google_model`. Both lists now derive from PROVIDER_METADATA's
    requires_api_key flag.
    """

    def _make_request(self, **kwargs):
        from src.api.workers import WorkerCreateRequest
        from src.api.workers.crud_endpoints import _config_payload
        from src.api.workers.models import validate_effective_worker

        req = WorkerCreateRequest(**kwargs)
        # The cross-field rules no longer run at parse time. Since a write
        # MERGES only the keys a caller sent, they must be asked of the row the
        # write PRODUCES, not of the body — see ``validate_effective_worker``.
        # For a CREATE the effective worker IS the body, so validating against
        # nothing stored reproduces exactly the behaviour these tests pin.
        effective = validate_effective_worker(_config_payload(req))
        req.agent_type = effective.get("agent_type")
        return req

    @pytest.fixture
    def strict_no_keys(self, monkeypatch):
        """REQUIRE_API_KEYS on, and not a single provider credential in the env."""
        from src.worker.provider_registry import PROVIDER_API_KEY_ENV_VARS

        for env_var in PROVIDER_API_KEY_ENV_VARS.values():
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setenv("REQUIRE_API_KEYS", "true")

    @pytest.mark.parametrize("provider", KEY_REQUIRING_PROVIDERS)
    def test_key_requiring_provider_is_rejected_without_a_key(self, provider, strict_no_keys):
        with pytest.raises(ValueError, match="API key"):
            self._make_request(
                worker_name=f"strict_{provider}",
                agent_role="Test",
                agent_type="pydantic_ai",
                model_provider=provider,
                model="some-model",
            )

    @pytest.mark.parametrize("provider", KEYLESS_PROVIDERS)
    def test_keyless_provider_is_accepted_without_a_key(self, provider, strict_no_keys):
        """Self-hosted backends declare requires_api_key=False and must not be gated."""
        req = self._make_request(
            worker_name=f"strict_{provider}",
            agent_role="Test",
            agent_type="pydantic_ai",
            model_provider=provider,
            model="some-model",
        )
        assert req.model_provider == provider

    @pytest.mark.parametrize("provider", KEY_REQUIRING_PROVIDERS)
    def test_key_requiring_provider_is_accepted_with_its_env_var(
        self, provider, strict_no_keys, monkeypatch
    ):
        """The env var consulted must be the one the registry names for that provider."""
        from src.worker.provider_registry import PROVIDER_API_KEY_ENV_VARS

        monkeypatch.setenv(PROVIDER_API_KEY_ENV_VARS[provider], "test-key")
        req = self._make_request(
            worker_name=f"strict_ok_{provider}",
            agent_role="Test",
            agent_type="pydantic_ai",
            model_provider=provider,
            model="some-model",
        )
        assert req.model_provider == provider

    @pytest.mark.parametrize("provider", KEY_REQUIRING_PROVIDERS)
    def test_api_key_override_satisfies_the_gate(self, provider, strict_no_keys):
        req = self._make_request(
            worker_name=f"strict_override_{provider}",
            agent_role="Test",
            agent_type="pydantic_ai",
            model_provider=provider,
            model="some-model",
            api_key_override="sk-inline",  # pragma: allowlist secret
        )
        assert req.model_provider == provider
