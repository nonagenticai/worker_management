"""F3 — inference-router AsyncOpenAI client transport hardening.

Asserts the InferenceRouterOpenAIProvider builds its underlying
httpx.AsyncClient with:
  - AsyncHTTPTransport(retries=N) so a single transient ambient-mesh
    handshake reset between workers and inference-router doesn't kill
    the agent run.
  - explicit Timeout(connect/read/write/pool) — the bare-client default
    of 5s read timeout was guaranteed-fail for 120B-model completions.

Root cause documented in the 2026-05-16 trust.pdf ingest postmortem:
PydanticAI agent ks-narrative-structure raised "Connection error" at
08:02:51 while the same path was reachable in <200ms five minutes later.
"""

from __future__ import annotations

import httpx2 as httpx
import pytest
from openai._constants import DEFAULT_MAX_RETRIES

from src.settings import get_settings
from src.worker.agents.pydantic_ai_agent import (
    InferenceRouterOpenAIProvider,
    MLXOpenAIProvider,
)


@pytest.fixture
def provider() -> InferenceRouterOpenAIProvider:
    return InferenceRouterOpenAIProvider(
        base_url="http://inference-router.inference-router.svc.cluster.local:8150/v1",
        api_key="test",  # pragma: allowlist secret
    )


def test_provider_installs_async_http_transport_with_retries(provider):
    """Transport has retries > 0 — bare AsyncClient defaults to 0."""
    http_client = provider._client._client  # AsyncOpenAI -> httpx.AsyncClient
    assert isinstance(http_client, httpx.AsyncClient)
    transport = http_client._transport
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    pool = transport._pool
    assert pool._retries == get_settings().ir_http_transport_retries
    assert pool._retries >= 1


def test_provider_uses_explicit_timeout_matching_settings(provider):
    """Read timeout is large enough for 120B-model completions, not the
    httpx default."""
    s = get_settings()
    timeout = provider._client._client.timeout
    assert timeout.connect == s.ir_http_connect_timeout_s
    assert timeout.read == s.ir_http_read_timeout_s
    assert timeout.write == s.ir_http_write_timeout_s
    assert timeout.pool == s.ir_http_pool_timeout_s
    # Defensive: assert the read timeout is "real-completion friendly".
    assert timeout.read >= 60.0


class TestSdkRetriesDisabledOnSelfHostedClients:
    """The OpenAI SDK must never retry on our own GPUs.

    openai 2.41.1 defaults ``max_retries`` to 2 (``openai/_constants.py:10``)
    and its retry loop is NOT limited to 429/5xx — ``_base_client.py:1653``
    catches ``httpx.TimeoutException`` and re-sends while retries remain::

        except httpx.TimeoutException as err:
            log.debug("Encountered httpx.TimeoutException", exc_info=True)
            if remaining_retries > 0:
                await self._sleep_for_retry(...)
                continue

    A read timeout against a self-hosted backend means a decode that already
    burned its full budget on hardware we own. Re-sending it buys a second and
    third full decode on the provider whose saturation caused the timeout.
    These assertions are made against the REAL constructed client object.
    """

    def test_sdk_default_is_actually_nonzero(self) -> None:
        """Anchor the premise. If the SDK ever defaults to 0, the explicit
        pins below become redundant and this test says so loudly rather than
        letting them rot as cargo cult."""
        assert DEFAULT_MAX_RETRIES == 2

    def test_ir_client_pins_max_retries_zero(self, provider) -> None:
        assert provider._client.max_retries == 0

    def test_mlx_client_pins_max_retries_zero(self) -> None:
        """mlx runs on our own Apple-silicon boxes."""
        mlx = MLXOpenAIProvider(base_url="http://mlx-mini2.example.com:8150/v1")
        assert mlx._client.max_retries == 0

    @pytest.mark.parametrize(
        ("name", "build"),
        [
            (
                "inference_router",
                lambda: InferenceRouterOpenAIProvider(
                    base_url="http://ir:8150/v1",
                    api_key="test",  # pragma: allowlist secret
                ),
            ),
            ("mlx", lambda: MLXOpenAIProvider(base_url="http://mlx:8150/v1")),
        ],
    )
    def test_no_self_hosted_provider_takes_the_sdk_default(self, name, build) -> None:
        """Covers the whole SELF_HOSTED_PROVIDERS surface in one place, so a
        newly-added self-hosted provider that forgets the pin fails here."""
        client = build()._client
        assert client.max_retries != DEFAULT_MAX_RETRIES, (
            f"{name} inherited the SDK default max_retries={DEFAULT_MAX_RETRIES}; "
            "a retried timeout is a repeat decode on a provider we own"
        )
        assert client.max_retries == 0


def test_provider_identity_event_hook_still_installed(provider):
    """Transport hardening must NOT remove the identity ContextVar hook —
    that's what makes per-user metering work for the cached worker path."""
    http_client = provider._client._client
    hooks = http_client._event_hooks.get("request", [])
    assert len(hooks) >= 1, "request event hook should still be installed"


def test_provider_default_headers_include_worker_name() -> None:
    """X-Worker-Name is set at construction time and flows through the
    AsyncOpenAI default_headers. IR binds it into structlog contextvars
    (routers/completions.py:42, 89), so every downstream log line — most
    importantly `provider_marked_degraded` — carries the originating
    worker. Without this the 2026-05-19 trust.pdf incident showed
    `worker_name: null` in every IR log line and forced timestamp-based
    correlation.
    """
    provider = InferenceRouterOpenAIProvider(
        base_url="http://inference-router.inference-router.svc.cluster.local:8150/v1",
        api_key="test",  # pragma: allowlist secret
        worker_name="ks-narrative-structure",
    )
    headers = provider._client._custom_headers
    assert headers.get("X-Worker-Name") == "ks-narrative-structure"
    # ⛔ ASSERTION INVERTED 2026-08-27, deliberately. This used to require the
    # identity triple to "continue to be present". That contract is exactly
    # what produced `X-Tenant-Id: ""` on the wire: an empty header is a PRESENT
    # header, so inference-router's `Header(default=...)` never fired and IR
    # coerced the empty string to the nil tenant sentinel. Measured live:
    # 1,400 of 13,893 routed calls in 24h dead-lettered in billing as
    # `nil_tenant_id`. X-Worker-Name is unaffected -- it is not identity, and
    # its purpose (naming the worker in IR's logs, per the docstring above) is
    # what made that investigation possible in the first place.
    assert "X-Tenant-Id" not in headers
    assert "X-License-Id" not in headers
    assert "X-Subject-Id" not in headers


def test_provider_omits_worker_name_defaults_to_empty_string() -> None:
    """Empty string keeps the header contract — the value is always
    present, never the literal "None" or absent.

    NB this convention is now specific to X-Worker-Name. The identity triple
    used to follow it and no longer does (see the test above): an always-present
    identity header cannot express "unknown", so it expressed "nobody" instead.
    """
    provider = InferenceRouterOpenAIProvider(
        base_url="http://inference-router.inference-router.svc.cluster.local:8150/v1",
        api_key="test",  # pragma: allowlist secret
    )
    headers = provider._client._custom_headers
    assert headers.get("X-Worker-Name") == ""
