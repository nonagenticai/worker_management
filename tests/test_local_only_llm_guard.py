"""The fail-closed local-only LLM guard.

The estate runs local-only inference. Disabling the paid providers inside
inference-router does NOT cover worker_management: ``ModelFactory`` constructs
vendor SDK clients directly, and ``_create_xai_model`` in particular speaks the
native xAI SDK to api.x.ai with a key that is mounted live in
``worker-management-secrets``. ``ModelFactory.enforce_local_only_provider`` is
what actually closes that path: an external provider needs a tenant BYO key, or
a PROVEN commercial-platform tenant (owner ruling 2026-10-03, fr#1385 — the
deployment-wide ``ALLOW_EXTERNAL_LLM_PROVIDERS`` flag was removed).

Every test here asserts the guard is STRUCTURAL — the permitted set is read from
``provider_registry``, and the refusal reaches a name the guard has never heard
of — rather than asserting today's seven vendor names are on a blocklist. A
blocklist is bypassed by the eighth provider on the day it is added, which is
the failure this file exists to prevent.

⛔ Do not add an autouse fixture that sets up a proven platform tenant here or
in conftest. The no-proven-tenant state IS the subject.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any
from unittest.mock import patch

import pytest
from pydantic_ai.exceptions import UserError

from src.settings import get_settings, reset_settings_cache
from src.worker.agents.pydantic_ai_agent import ModelFactory, PydanticAIAgentConfig
from src.worker.provider_registry import (
    EXTERNAL_PROVIDERS,
    LOCAL_PROVIDERS,
    PROVIDER_METADATA,
    PROVIDER_NAMES,
)

MODULE = "src.worker.agents.pydantic_ai_agent"

# Base URLs for the local backends, so their creators build a real client
# instead of failing on "no URL configured" — which would make a refusal and a
# misconfiguration indistinguishable.
_LOCAL_BASE_URLS = {
    "vllm": "http://vllm.invalid:8001/v1",
    "inference_router": "http://inference-router.invalid:8150/api",
    "mlx": "http://mlx.invalid:8080/v1",
    "vlm_local": "http://vlm.invalid:9203",
}

# Vendor SDK entry points reachable from ModelFactory, by module attribute name.
# Patched wholesale so a refusal can be shown to happen BEFORE any of them is
# touched — "raised an error" and "did not call out" are different claims.
_VENDOR_SDK_ATTRS = (
    "OpenAIProvider",
    "DeepSeekProvider",
    "GroqProvider",
    "GroqModel",
    "CohereProvider",
    "CohereModel",
    "AnthropicProvider",
    "AnthropicModel",
    "XaiProvider",
    "PinnedContextXaiModel",
    "AsyncOpenAI",
    "AsyncGroq",
    "AsyncCohereClient",
)


@pytest.fixture(autouse=True)
def _local_only_by_default(monkeypatch: pytest.MonkeyPatch):
    """Pin the environment to the shipped default and prove it took effect.

    An env var leaking in from the developer's shell or another test would make
    every refusal assertion below pass or fail for the wrong reason, so the
    state is asserted, not assumed.
    """
    from src.api.identity_context import get_tenant_id_from_context

    reset_settings_cache()
    # No request context: no tenant, so nothing is proven.
    assert get_tenant_id_from_context() in (None, "")
    # The removed flag must not be resurrected as a settings field.
    assert not hasattr(get_settings(), "allow_external_llm_providers")
    yield
    reset_settings_cache()


@contextlib.contextmanager
def _external_allowed():
    """The one ground that admits the owner's keys: a PROVEN platform tenant.

    Owner ruling 2026-10-03 — a gitops commercial-platform tenant whose request
    proved it (here: licence provenance).
    """
    from tests._platform_tenant import platform_tenant_context

    with platform_tenant_context():
        yield


@contextlib.contextmanager
def _vendor_sdks_watched():
    """Patch every vendor SDK entry point and yield the mocks for inspection."""
    with contextlib.ExitStack() as stack:
        yield {attr: stack.enter_context(patch(f"{MODULE}.{attr}")) for attr in _VENDOR_SDK_ATTRS}


def _cfg(provider: str, **overrides: Any) -> PydanticAIAgentConfig:
    kwargs: dict[str, Any] = {
        "name": "guard_test",
        "model_provider": provider,
        "model": "some-model",
        "db_path": "",
        "base_url": _LOCAL_BASE_URLS.get(provider),
        "api_key": "test-key",  # pragma: allowlist secret
    }
    kwargs.update(overrides)
    return PydanticAIAgentConfig(**kwargs)


def _assert_names_provider_and_flag(message: str, provider: str) -> None:
    """The error has to be actionable: who was refused, and what admits it."""
    assert provider in message, f"error does not name the provider: {message}"
    assert "COMMERCIAL_PLATFORM_TENANT_IDS" in message, (
        f"error does not name the set that admits it: {message}"
    )
    assert "PROVED" in message, f"error does not say the tenant must be proven: {message}"
    assert "BYO" in message, f"error does not name the BYO ground: {message}"


# ---------------------------------------------------------------------------
# The seven external providers are refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", sorted(EXTERNAL_PROVIDERS))
def test_external_provider_is_refused_at_the_dispatch(provider: str) -> None:
    """Parametrised off the REGISTRY, so an eighth external provider is covered
    the moment its row is added — never off a literal list of today's seven."""
    with pytest.raises(UserError) as exc:
        ModelFactory.initialize_model(_cfg(provider))
    _assert_names_provider_and_flag(str(exc.value), provider)


@pytest.mark.parametrize("provider", sorted(EXTERNAL_PROVIDERS))
def test_refusal_happens_before_any_vendor_sdk_is_touched(provider: str) -> None:
    """Refusing is not enough — nothing may reach the SDK that would dial out."""
    with _vendor_sdks_watched() as sdks:
        with pytest.raises(UserError):
            ModelFactory.initialize_model(_cfg(provider))
    for attr, mock in sdks.items():
        assert mock.call_count == 0, f"{attr} was constructed for a refused provider {provider}"


@pytest.mark.parametrize("provider", sorted(EXTERNAL_PROVIDERS))
def test_calling_a_vendor_creator_directly_is_also_refused(provider: str) -> None:
    """THE SECOND BARRIER, and the reason the guard is not only at the dispatch.

    Every ``_create_<vendor>_model`` calls ``get_api_key`` as its first
    statement and no local creator calls it at all, so a future creator invoked
    directly — bypassing ``initialize_model`` entirely — still cannot obtain the
    credential it needs. Reaching the creator by name off the registry keeps
    this parametrisation honest as providers are added.
    """
    creator = getattr(ModelFactory, f"_create_{provider}_model")
    with pytest.raises(UserError) as exc:
        creator(_cfg(provider))
    _assert_names_provider_and_flag(str(exc.value), provider)


@pytest.mark.parametrize("provider", sorted(EXTERNAL_PROVIDERS))
def test_refusal_precedes_the_missing_api_key_error(provider: str) -> None:
    """With no credential present at all, the POLICY error is what surfaces.

    Ordering matters for the operator: "you have no key" invites someone to go
    and find one, which is the opposite of the intended outcome.
    """
    env_var = PROVIDER_METADATA[provider]["api_key_env_var"]
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(env_var, None)
        with pytest.raises(UserError) as exc:
            ModelFactory.initialize_model(_cfg(provider, api_key=None))
    assert "API key required" not in str(exc.value)
    _assert_names_provider_and_flag(str(exc.value), provider)


# ---------------------------------------------------------------------------
# The four local providers are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", sorted(LOCAL_PROVIDERS))
def test_local_provider_still_constructs_with_the_flag_off(provider: str) -> None:
    """No behaviour change for anything that runs on our own hardware.

    These construct REAL provider objects (no SDK patching) so the test would
    notice the guard refusing them, rather than a mock absorbing the call.
    """
    model = ModelFactory.initialize_model(_cfg(provider))
    assert model is not None


@pytest.mark.parametrize("provider", sorted(LOCAL_PROVIDERS))
def test_local_provider_never_consults_the_guard_s_escape_hatch(provider: str) -> None:
    """A local provider is admitted by the allowlist, not by the opt-in flag —
    so it behaves identically in both flag states."""
    off = ModelFactory.initialize_model(_cfg(provider))
    with _external_allowed():
        on = ModelFactory.initialize_model(_cfg(provider))
    assert type(off) is type(on)


# ---------------------------------------------------------------------------
# The explicit opt-in restores the previous behaviour
# ---------------------------------------------------------------------------


def test_external_provider_constructs_when_explicitly_allowed() -> None:
    """A proven platform tenant is a real grant, not a softer refusal: the SDK is reached."""
    with _external_allowed(), _vendor_sdks_watched() as sdks:
        with patch(f"{MODULE}.OpenAIChatModel") as chat_model:
            model = ModelFactory.initialize_model(_cfg("openai", base_url=None))
    assert model is chat_model.return_value
    assert sdks["OpenAIProvider"].call_count == 1


def test_opt_in_restores_the_missing_key_error() -> None:
    """For a proven platform tenant the pre-guard behaviour is back byte for byte
    — including the error you get when the credential is not registered."""
    with _external_allowed(), patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(UserError, match="API key required for Anthropic"):
            ModelFactory.initialize_model(_cfg("anthropic", api_key=None, base_url=None))


# ---------------------------------------------------------------------------
# ⭐ The guard is structural: a provider it has never heard of is refused too
# ---------------------------------------------------------------------------


def test_a_provider_name_that_does_not_exist_yet_is_refused() -> None:
    """THE POINT OF THE WHOLE FILE.

    A blocklist of the seven known vendors would let the eighth through on the
    day someone adds ``_create_mistral_model`` and its dispatch branch. The
    guard is a positive allowlist, so a name nobody has classified is refused
    by construction — it does not have to be predicted first.
    """
    with pytest.raises(UserError) as exc:
        ModelFactory.enforce_local_only_provider("mistral", call_site="test")
    _assert_names_provider_and_flag(str(exc.value), "mistral")


def test_an_unknown_provider_is_refused_before_the_dispatch_can_reach_a_creator() -> None:
    """Same claim, through the real entry point.

    ``model_construct`` bypasses the ``ModelProvider`` Literal deliberately:
    that Literal is a SCHEMA check on an API request, not a runtime barrier, and
    an internal caller constructing a config by hand does not pass through it.
    The refusal must be the policy error, not the pre-existing
    "Unsupported model provider" fallthrough at the end of the dispatch — the
    latter would mean the guard let the name travel all the way down.
    """
    config = PydanticAIAgentConfig.model_construct(
        name="guard_test",
        model_provider="mistral",
        model="mistral-large",
        db_path="",
    )
    with _vendor_sdks_watched() as sdks:
        with pytest.raises(UserError) as exc:
            ModelFactory.initialize_model(config)
    assert "Unsupported model provider" not in str(exc.value)
    _assert_names_provider_and_flag(str(exc.value), "mistral")
    assert all(mock.call_count == 0 for mock in sdks.values())


def test_none_is_refused_rather_than_read_as_no_provider_configured() -> None:
    """A missing provider is an unclassified one. It must not fail open."""
    with pytest.raises(UserError):
        ModelFactory.enforce_local_only_provider(None, call_site="test")


def test_the_permitted_set_is_the_registry_s_local_set_not_a_copy() -> None:
    """The guard must not carry its own list — that is how the two drift."""
    admitted = set()
    for provider in PROVIDER_NAMES:
        try:
            ModelFactory.enforce_local_only_provider(provider, call_site="test")
        except UserError:
            continue
        admitted.add(provider)
    assert admitted == set(LOCAL_PROVIDERS)
    assert admitted == {n for n, m in PROVIDER_METADATA.items() if not m["external"]}
    assert admitted.isdisjoint(EXTERNAL_PROVIDERS)


def test_every_registered_provider_is_classified_as_local_or_external() -> None:
    """No third state. A provider with no declared egress is the hole."""
    assert set(PROVIDER_NAMES) == LOCAL_PROVIDERS | EXTERNAL_PROVIDERS
    assert LOCAL_PROVIDERS.isdisjoint(EXTERNAL_PROVIDERS)


def test_a_provider_row_without_an_egress_declaration_fails_at_import() -> None:
    """Adding a provider cannot silently grant it egress.

    The import-time guard must actually FIRE, not merely exist, so this removes
    the declaration from a real row and calls the validator — the same shape as
    the existing requires_api_key/api_key_env_var drift test, and the same
    RuntimeError a newly-added row with no ``external`` key would produce.
    """
    from src.worker import provider_registry

    original = provider_registry.PROVIDER_METADATA["google"].pop("external")
    try:
        with pytest.raises(RuntimeError, match="external"):
            provider_registry._validate_registry()
    finally:
        provider_registry.PROVIDER_METADATA["google"]["external"] = original
    # The mutation must leave the registry exactly as it was found.
    provider_registry._validate_registry()


def test_external_flag_is_declared_per_row_as_a_bool() -> None:
    """`external` answers "does this leave our hardware?" — it is not derived
    from `requires_api_key`, which is a different fact that happens to agree
    today and would not if a self-hosted backend ever grew an auth token."""
    for name, meta in PROVIDER_METADATA.items():
        assert isinstance(meta["external"], bool), f"{name} does not declare `external`"


# ---------------------------------------------------------------------------
# fr#1385 (owner ruling 2026-10-03): the deployment-wide flag is dead, and only a
# PROVEN commercial-platform tenant is admitted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "1"])
def test_the_removed_flag_admits_nothing(value: str) -> None:
    """``ALLOW_EXTERNAL_LLM_PROVIDERS=true`` used to open the owner's keys; it is
    no longer read, so setting it changes nothing."""
    with patch.dict(os.environ, {"ALLOW_EXTERNAL_LLM_PROVIDERS": value}):
        reset_settings_cache()
        with pytest.raises(UserError):
            ModelFactory.initialize_model(_cfg("xai"))


@pytest.mark.parametrize("provenance", ["asserted", "default", "bypass", "unknown"])
def test_a_platform_tenant_that_was_not_proven_is_refused(provenance: str) -> None:
    """rm#187 class: a platform tenant NAMED in a header (or the default tenant)
    never reaches the owner's keys."""
    from tests._platform_tenant import platform_tenant_context

    with platform_tenant_context(provenance=provenance), _vendor_sdks_watched() as sdks:
        with pytest.raises(UserError) as exc:
            ModelFactory.initialize_model(_cfg("xai"))
    assert f"provenance_{provenance}" in str(exc.value)
    assert all(m.call_count == 0 for m in sdks.values())


def test_a_proven_tenant_outside_the_gitops_set_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Membership comes from the gitops env, never a code default: with the set
    empty, even a licence-proven ex-platform tenant is a customer."""
    from tests._platform_tenant import platform_tenant_context

    monkeypatch.setenv("COMMERCIAL_PLATFORM_TENANT_IDS", "")
    reset_settings_cache()
    with platform_tenant_context(), pytest.raises(UserError) as exc:
        ModelFactory.initialize_model(_cfg("xai"))
    assert "not_a_commercial_platform_tenant" in str(exc.value)
