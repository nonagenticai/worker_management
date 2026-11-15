"""Drift guards for the LLM provider registry.

`worker_management` used to keep six parallel hardcoded copies of the provider
list. They drifted: `cohere`/`google` silently skipped API-key validation, and
`xai`/`google`/`vlm_local` were rejected by `AgentTemplate` and by
`safe_create_pydantic_ai_agent` despite being fully supported by ModelFactory.

Every test here asserts the lists are *derived* from
`src.worker.provider_registry`, not that they happen to agree today. The
parametrised `TestEveryProviderReachesEveryList` is the guard that would have
caught all three bugs.
"""

import typing

import pytest

from src.worker.provider_registry import (
    API_KEY_PROVIDERS,
    PROVIDER_API_KEY_ENV_VARS,
    PROVIDER_METADATA,
    PROVIDER_NAMES,
    ModelProvider,
)

ALL_PROVIDERS = list(typing.get_args(ModelProvider))

# Providers that run on our own hardware and take no credential. Spelled out
# once, here, so a change to requires_api_key in the registry has to be a
# deliberate edit in two places rather than an accident in one.
KEYLESS_PROVIDERS = {"vllm", "inference_router", "mlx", "vlm_local"}


class TestRegistryInvariants:
    def test_provider_names_is_the_literal(self):
        assert tuple(ALL_PROVIDERS) == PROVIDER_NAMES

    def test_metadata_covers_exactly_the_literal(self):
        assert set(PROVIDER_METADATA) == set(ALL_PROVIDERS)

    def test_api_key_providers_is_derived_from_requires_api_key(self):
        """API_KEY_PROVIDERS must be a projection of PROVIDER_METADATA, not a copy."""
        expected = {n for n, m in PROVIDER_METADATA.items() if m["requires_api_key"]}
        assert set(API_KEY_PROVIDERS) == expected

    def test_env_var_map_agrees_with_requires_api_key(self):
        """The two halves of the same fact must never disagree.

        This is the consistency test that replaces hand-auditing the old
        `env_var_map` against the old membership list.
        """
        assert set(PROVIDER_API_KEY_ENV_VARS) == set(API_KEY_PROVIDERS)

    def test_keyless_providers_are_not_swept_into_key_validation(self):
        assert set(API_KEY_PROVIDERS) & KEYLESS_PROVIDERS == set()
        assert set(ALL_PROVIDERS) - set(API_KEY_PROVIDERS) == KEYLESS_PROVIDERS

    def test_google_reads_gemini_api_key(self):
        """Regression: the env var is GEMINI_API_KEY, not GOOGLE_API_KEY.

        `_create_google_model` and `_fetch_google_models` both read
        GEMINI_API_KEY; a mismatch here would make the 422 fire on a key that
        is actually present.
        """
        assert PROVIDER_API_KEY_ENV_VARS["google"] == "GEMINI_API_KEY"

    def test_registry_self_validation_rejects_inconsistent_rows(self):
        """The import-time guard must actually fire, not just exist."""
        from src.worker import provider_registry

        original = provider_registry.PROVIDER_METADATA["google"]["api_key_env_var"]
        provider_registry.PROVIDER_METADATA["google"]["api_key_env_var"] = None
        try:
            with pytest.raises(RuntimeError, match="requires_api_key and api_key_env_var"):
                provider_registry._validate_registry()
        finally:
            provider_registry.PROVIDER_METADATA["google"]["api_key_env_var"] = original


class TestPydanticAiAgentReExports:
    """`from ...pydantic_ai_agent import ModelProvider` is the historical path."""

    def test_reexports_are_the_registry_objects(self):
        from src.worker.agents import pydantic_ai_agent

        assert pydantic_ai_agent.ModelProvider is ModelProvider
        assert pydantic_ai_agent.PROVIDER_METADATA is PROVIDER_METADATA


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
class TestEveryProviderReachesEveryList:
    """Every ModelProvider member must appear in every list that should contain it."""

    def test_has_metadata_row(self, provider):
        meta = PROVIDER_METADATA[provider]
        assert meta["label"]
        assert meta["description"]
        assert isinstance(meta["requires_api_key"], bool)

    def test_has_a_model_fetcher(self, provider):
        from src.api.providers._router import PROVIDER_FETCHERS

        assert provider in PROVIDER_FETCHERS

    def test_accepted_by_worker_create_request_allowlist(self, provider):
        from src.api.workers import WorkerCreateRequest

        req = WorkerCreateRequest(
            worker_name=f"w_{provider}",
            agent_role="Test",
            agent_type="pydantic_ai",
            model_provider=provider,
            model="some-model",
        )
        assert req.model_provider == provider

    def test_accepted_by_agent_template_literal(self, provider):
        """Regression: AgentTemplate's Literal omitted xai, google and vlm_local."""
        from src.worker.worker_template import AgentTemplate

        tpl = AgentTemplate(
            name=f"t_{provider}",
            description="d",
            tools=[],
            db_path="/tmp/t.db",
            agent_type="pydantic_ai",
            agent_role="Test",
            model_provider=provider,
            model="some-model",
            user_prompt="hi",
        )
        assert tpl.model_provider == provider

    def test_accepted_by_safe_wrapper_signature(self, provider):
        """Regression: safe_create_pydantic_ai_agent's Literal omitted the same three."""
        from src.worker.agents import pydantic_ai_agent_wrapper

        hints = typing.get_type_hints(pydantic_ai_agent_wrapper.safe_create_pydantic_ai_agent)
        assert provider in typing.get_args(hints["model_provider"])

    def test_key_requirement_is_declared_consistently(self, provider):
        requires = PROVIDER_METADATA[provider]["requires_api_key"]
        assert (provider in API_KEY_PROVIDERS) is bool(requires)
        assert (provider in PROVIDER_API_KEY_ENV_VARS) is bool(requires)


class TestAgentTemplateHasOneProviderList:
    def test_literal_annotation_is_model_provider_itself(self):
        """The runtime check and the annotation must not be two separate lists."""
        from src.worker.worker_template import AgentTemplate

        annotation = AgentTemplate.model_fields["model_provider"].annotation
        assert set(typing.get_args(annotation)) == set(ALL_PROVIDERS)

    def test_rejects_unknown_provider(self):
        from pydantic import ValidationError

        from src.worker.worker_template import AgentTemplate

        with pytest.raises(ValidationError):
            AgentTemplate(
                name="t",
                description="d",
                tools=[],
                db_path="/tmp/t.db",
                agent_type="pydantic_ai",
                agent_role="Test",
                model_provider="not-a-real-provider",
                model="m",
                user_prompt="hi",
            )
