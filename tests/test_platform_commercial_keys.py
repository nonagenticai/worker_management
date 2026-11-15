"""fr#1385 — the owner's commercial keys serve PROVEN platform workspaces only.

Owner ruling 2026-10-03: "allow all gitops registered commercial apis for
platform workspaces only. so anthropic, xai, etc." Customer tenants stay on
local-llm plus their OWN BYO keys.

The gate is applied on EVERY layer of the call path, with ONE predicate
(``src.auth.tenant_proof.owner_commercial_keys_refusal``), so the catalog can
never claim a provider the next call refuses:

* ``GET /providers``                  — ``available``
* ``GET /providers/{name}/models``    — 403 ``PROVIDER_NOT_ENABLED_FOR_TENANT``
* ``POST /chat/completions``          — 403 ``PROVIDER_NOT_ENABLED_FOR_TENANT``
  (and ``ModelFactory`` refuses again at build — tests/test_local_only_llm_guard.py)

Controls (coordinator conditions a + b):

* POSITIVE — ``…0002`` (platform-core, in the gitops set), PROVEN: available, 200.
* NEGATIVE — ``isolation-probe-test`` ``01a0fdb8…`` (a customer, no BYO rows):
  xai/anthropic not offered, 403 on models and on chat.
* SPOOF — ``…0002`` merely ASSERTED (header / delegation without a verified
  end-user token): refused exactly like a customer on the owner's keys.
* BYO — a customer with its own xai key still gets xai.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import pytest
from fastapi.testclient import TestClient

from src.api.worker_management_api import app
from src.auth import get_tenant_id
from tests._platform_tenant import caller_tenant

PLATFORM_0002 = "0195f48b-d000-7000-8000-000000000002"  # platform-core
CUSTOMER_PROBE = "01a0fdb8-f785-7785-843d-0799f306cbbd"  # isolation-probe-test
OWNER_XAI_KEY = "xai-owner-key-not-real"  # pragma: allowlist secret
TENANT_XAI_KEY = "xai-tenant-own-key-not-real"  # pragma: allowlist secret
CODE = "PROVIDER_NOT_ENABLED_FOR_TENANT"


async def _fake_tenant_id(request=None):
    return "test-tenant"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    from src.settings import reset_settings_cache

    # The registered key, as mounted from gitops on the live pod. anthropic's key
    # is registered only in inference-router, never in this Deployment.
    monkeypatch.setenv("XAI_API_KEY", OWNER_XAI_KEY)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings_cache()
    app.dependency_overrides[get_tenant_id] = _fake_tenant_id
    with (
        patch("src.api.worker_management_api.init_pg_pool", new_callable=AsyncMock),
        patch("src.api.worker_management_api.close_pg_pool", new_callable=AsyncMock),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def vendor():
    """The outbound vendor model-list call; must never be reached on a refusal."""
    with patch(
        "src.api.providers._router._httpx_json_get",
        new=AsyncMock(return_value={"data": [{"id": "grok-4"}]}),
    ) as m:
        yield m


@pytest.fixture
def no_byo_rows():
    """The customer has no tenant_llm_config rows (measured live for 01a0fdb8)."""
    with patch(
        "src.api.providers._router.tenant_byo_provider_status", new=AsyncMock(return_value={})
    ):
        yield


def _fake_agent():
    async def text(delta=True):
        yield "ok"

    stream = MagicMock()
    stream.stream_text = lambda delta=True: text(delta=delta)
    stream.usage = MagicMock(
        return_value=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
    )
    stream.all_messages = MagicMock(return_value=[])
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=stream)
    ctx.__aexit__ = AsyncMock(return_value=False)
    agent = MagicMock()
    agent.run_stream = MagicMock(return_value=ctx)
    return agent


def _chat(client: TestClient, provider: str, model: str = "grok-4"):
    return client.post(
        "/chat/completions",
        json={
            "provider": provider,
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )


def _rows(client: TestClient) -> dict[str, dict]:
    resp = client.get("/providers")
    assert resp.status_code == 200, resp.text
    return {r["name"]: r for r in resp.json()["data"]}


# ----------------------------------------------------------------- positive control


class TestPositiveControlProvenPlatform0002:
    def test_catalog_offers_the_registered_commercial_provider(self, client):
        with caller_tenant(PLATFORM_0002):
            rows = _rows(client)
        assert rows["xai"]["available"] is True
        # Registered in gitops for inference-router only, not here: not offered
        # DIRECTLY (reached via inference_router instead).
        assert rows["anthropic"]["available"] is False
        assert rows["inference_router"]["available"] is True

    def test_models_list_is_200_with_the_owner_key(self, client, vendor):
        with caller_tenant(PLATFORM_0002):
            resp = client.get("/providers/xai/models")
        assert resp.status_code == 200, resp.text
        assert [m["id"] for m in resp.json()["data"]] == ["grok-4"]
        assert vendor.await_count == 1

    def test_chat_is_200(self, client):
        with (
            caller_tenant(PLATFORM_0002),
            patch("src.api.providers._chat_router._build_agent", return_value=_fake_agent()),
        ):
            resp = _chat(client, "xai")
        assert resp.status_code == 200, resp.text


# ----------------------------------------------------------------- negative control


class TestNegativeControlCustomerProbe:
    def test_catalog_does_not_offer_owner_providers(self, client, no_byo_rows):
        with caller_tenant(CUSTOMER_PROBE):
            rows = _rows(client)
        for name in ("xai", "anthropic"):
            assert not rows.get(name, {}).get("available", False), (name, rows)

    @pytest.mark.parametrize("provider", ["xai", "anthropic"])
    def test_models_list_is_403(self, client, vendor, no_byo_rows, provider):
        with caller_tenant(CUSTOMER_PROBE):
            resp = client.get(f"/providers/{provider}/models")
        assert resp.status_code == 403, resp.text
        assert vendor.await_count == 0

    @pytest.mark.parametrize("provider", ["xai", "anthropic"])
    def test_chat_is_403_before_a_model_is_built(self, client, no_byo_rows, provider):
        with (
            caller_tenant(CUSTOMER_PROBE),
            patch("src.api.providers._chat_router._build_agent") as build,
        ):
            resp = _chat(client, provider)
        assert resp.status_code == 403, resp.text
        build.assert_not_called()


# ----------------------------------------------------------------- spoof (condition a)


class TestAssertedPlatformTenantIsRefused:
    """0002 named in a header but NOT proven: no owner keys, on every layer."""

    def test_catalog_marks_external_unavailable(self, client):
        with caller_tenant(PLATFORM_0002, proven=False):
            rows = _rows(client)
        assert rows["xai"]["available"] is False
        # Keyless platform rows are untouched (the gate is not a blanket refusal).
        assert rows["inference_router"]["available"] is True

    def test_models_list_is_typed_403_and_no_vendor_call(self, client, vendor):
        with caller_tenant(PLATFORM_0002, proven=False):
            resp = client.get("/providers/xai/models")
        assert resp.status_code == 403, resp.text
        err = resp.json()["error"]
        assert err["code"] == CODE
        assert err["details"]["provider"] == "xai"
        assert OWNER_XAI_KEY not in resp.text and "XAI_API_KEY" not in resp.text
        assert vendor.await_count == 0

    def test_chat_is_typed_403_before_a_model_is_built(self, client):
        with (
            caller_tenant(PLATFORM_0002, proven=False),
            patch("src.api.providers._chat_router._build_agent") as build,
        ):
            resp = _chat(client, "xai")
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == CODE
        build.assert_not_called()


class TestTheRealProofPredicate:
    """The HTTP tests above stub only the PROOF step. This drives the real
    predicate with the real identity contextvars the auth middleware stamps."""

    FR_SVC = "019ffe9f-31fd-70da-b3c5-f921f7290b01"  # svc-fr-backend, a delegator

    @staticmethod
    def _refusal(tenant: str, provenance: str, *, proof_tenant: str = "", delegator: str = ""):
        import contextvars

        from src.api.identity_context import set_end_user_proof, set_identity_context
        from src.auth.tenant_proof import owner_commercial_keys_refusal

        def run():
            set_identity_context(tenant, None, None, tenant_provenance=provenance)
            set_end_user_proof(
                tenant_id=proof_tenant,
                delegator=delegator,
                reason="ok" if proof_tenant else "absent",
            )
            return owner_commercial_keys_refusal(tenant)

        return contextvars.copy_context().run(run)

    def test_licence_proven_0002_is_admitted(self):
        assert self._refusal(PLATFORM_0002, "license") is None

    @pytest.mark.parametrize("provenance", ["asserted", "default", "bypass", "unknown"])
    def test_asserted_0002_is_refused(self, provenance):
        assert self._refusal(PLATFORM_0002, provenance) == f"provenance_{provenance}"

    def test_header_only_delegation_of_0002_is_refused(self, monkeypatch):
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self.FR_SVC)
        assert self._refusal(PLATFORM_0002, "delegated") == (
            "delegated_without_verified_end_user_token"
        )

    def test_delegation_with_a_verified_end_user_token_is_admitted(self, monkeypatch):
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self.FR_SVC)
        assert (
            self._refusal(
                PLATFORM_0002, "delegated", proof_tenant=PLATFORM_0002, delegator=self.FR_SVC
            )
            is None
        )

    def test_end_user_token_for_another_tenant_is_refused(self, monkeypatch):
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self.FR_SVC)
        assert (
            self._refusal(
                PLATFORM_0002, "delegated", proof_tenant=CUSTOMER_PROBE, delegator=self.FR_SVC
            )
            == "delegated_without_verified_end_user_token"
        )

    def test_licence_proven_customer_is_refused(self):
        assert self._refusal(CUSTOMER_PROBE, "license") == "not_a_commercial_platform_tenant"

    def test_a_tenant_other_than_the_requests_is_refused(self):
        import contextvars

        from src.api.identity_context import set_identity_context
        from src.auth.tenant_proof import owner_commercial_keys_refusal

        def run():
            set_identity_context(CUSTOMER_PROBE, None, None, tenant_provenance="license")
            return owner_commercial_keys_refusal(PLATFORM_0002)

        assert contextvars.copy_context().run(run) == "not_the_request_tenant"


# ----------------------------------------------------------------- BYO unchanged


class TestCustomerWithOwnByoKey:
    def test_customer_byo_xai_is_still_available(self, client):
        entry = {"api_key": TENANT_XAI_KEY, "owner_provided": False}
        status = {"xai": {"status": "usable", "entry": entry}}
        with (
            caller_tenant(CUSTOMER_PROBE),
            patch(
                "src.api.providers._router.tenant_byo_provider_status",
                new=AsyncMock(return_value=status),
            ),
            patch(
                "src.api.providers._router.live_row_models",
                new=AsyncMock(return_value=["grok-4"]),
            ),
        ):
            rows = _rows(client)
            models = client.get("/providers/xai/models")
        assert rows["xai"]["available"] is True
        assert rows["xai"]["source"] == "byo"
        assert models.status_code == 200, models.text
