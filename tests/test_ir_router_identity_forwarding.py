"""WM -> inference-router: caller identity + job id on every router call.

1. ``X-Job-Id`` = the WM worker name (job type) on every completion call.
2. ``X-App-Id`` / ``X-Project-Id`` are relayed verbatim from the inbound
   request; when absent they are NOT invented -- the call goes out without the
   header and ``wm_ir_identity_missing_total{header, worker}`` is incremented.
3. ``wm_ir_tenantless_call_refused_total`` and ``wm_ir_identity_missing_total``
   exist at 0 from app startup (the lifespan), not only after first use.

Requests go through a real ``httpx.AsyncClient`` with a ``MockTransport``
standing in for the router, so headers are asserted at the wire.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

from src.api.identity_context import (
    set_app_id_context,
    set_caller_context,
    set_identity_context,
    set_project_id_context,
)
from src.settings import reset_settings_cache
from src.worker.agents.pydantic_ai_agent import (
    IR_IDENTITY_MISSING_TOTAL,
    make_identity_event_hook,
)

IR_URL = "http://inference-router.inference-router.svc.cluster.local:8150"
COMPLETIONS = f"{IR_URL}/v1/chat/completions"
TENANT = "0195f48b-d000-7000-8000-000000000100"
PROJECT = "0195f48b-d000-7000-8000-0000000000a1"
WORKER = "ks-relationship-extractor"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_ROUTER_URL", IR_URL)
    monkeypatch.setenv("WM_IR_BUNDLE_TOKENS", "false")
    reset_settings_cache()
    set_identity_context(TENANT, None, None)
    set_project_id_context("")
    set_app_id_context("")
    set_caller_context("", ())
    yield
    set_identity_context(None, None, None)
    set_project_id_context("")
    set_app_id_context("")
    set_caller_context("", ())
    reset_settings_cache()


def _hook(worker_name: str = WORKER):
    return make_identity_event_hook(
        worker_name=worker_name,
        worker_version="1",
        fallback_tenant_id="",
        fallback_license_id="",
        fallback_subject_id="",
    )


async def _send(hook, default_headers: dict[str, str] | None = None) -> httpx.Request:
    reached: list[httpx.Request] = []

    def _router(request: httpx.Request) -> httpx.Response:
        reached.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_router),
        event_hooks={"request": [hook]},
        headers=default_headers or {},
    ) as client:
        await client.post(COMPLETIONS, json={})
    assert len(reached) == 1
    return reached[0]


def _value(header: str, worker: str) -> float:
    return IR_IDENTITY_MISSING_TOTAL.labels(header=header, worker=worker)._value.get()


@pytest.mark.asyncio
async def test_job_id_is_the_worker_name_and_inbound_identity_is_relayed() -> None:
    set_app_id_context("kag")
    set_project_id_context(PROJECT)
    app_before, proj_before = _value("app_id", WORKER), _value("project_id", WORKER)
    req = await _send(_hook())
    assert req.headers["X-Job-Id"] == WORKER
    assert req.headers["X-App-Id"] == "kag"
    assert req.headers["X-Project-Id"] == PROJECT
    # Present identity is not a miss.
    assert _value("app_id", WORKER) == app_before
    assert _value("project_id", WORKER) == proj_before


@pytest.mark.asyncio
async def test_missing_app_id_is_never_invented_and_is_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_project_id_context(PROJECT)
    set_caller_context("svc-merqi-sales", ())
    before = _value("app_id", WORKER)
    # A stale client-level default must not ride along either.
    req = await _send(_hook(), default_headers={"X-App-Id": ""})
    assert "X-App-Id" not in req.headers
    assert _value("app_id", WORKER) == before + 1
    assert req.headers["X-Job-Id"] == WORKER
    warn = [r for r in caplog.records if "wm_ir_identity_missing" in r.getMessage()]
    assert warn, "no WARN for the missing app id"
    assert warn[0].levelname == "WARNING"
    assert "header=app_id" in warn[0].getMessage()
    assert "caller=svc-merqi-sales" in warn[0].getMessage()


@pytest.mark.asyncio
async def test_missing_project_is_never_invented_and_is_counted() -> None:
    set_app_id_context("w2v")
    before = _value("project_id", WORKER)
    req = await _send(_hook())
    assert "X-Project-Id" not in req.headers
    assert req.headers["X-App-Id"] == "w2v"
    assert _value("project_id", WORKER) == before + 1


@pytest.mark.asyncio
async def test_missing_both_counts_both_and_unnamed_worker_lands_on_unknown() -> None:
    app_before = _value("app_id", "<unknown>")
    proj_before = _value("project_id", "<unknown>")
    req = await _send(_hook(worker_name=""))
    assert "X-Job-Id" not in req.headers  # never an empty job id
    assert _value("app_id", "<unknown>") == app_before + 1
    assert _value("project_id", "<unknown>") == proj_before + 1


@pytest.mark.asyncio
async def test_listing_is_wms_own_call_and_sends_app_id_workers() -> None:
    """Model listing is WM's own call: X-App-Id is WM's licences key
    ``workers`` even when the inbound request named another app."""
    from src.services.ir_bundle_tokens import WM_SERVICE_WORKER_NAME, ir_listing_headers

    set_app_id_context("n8n")
    headers = await ir_listing_headers(f"{IR_URL}/api/models", TENANT)
    assert headers["X-Job-Id"] == WM_SERVICE_WORKER_NAME
    assert headers["X-App-Id"] == "workers"
    assert "X-Project-Id" not in headers
    set_app_id_context("")
    headers = await ir_listing_headers(f"{IR_URL}/v1/models", TENANT)
    assert headers["X-App-Id"] == "workers"


def test_both_metrics_are_in_metrics_output_at_startup() -> None:
    """Fresh interpreter: import the app, run ONLY the lifespan's startup up to
    the DB (which is made to fail), then render the prometheus registry -- the
    same exposition the metrics listener serves. ir_bundle_tokens is NOT
    imported by importing the app (that was the bug); the lifespan imports it."""
    script = textwrap.dedent(
        """
        import asyncio, sys
        from unittest import mock
        from prometheus_client import REGISTRY, generate_latest
        import src.api.worker_management_api as api
        assert "src.services.ir_bundle_tokens" not in sys.modules, "precondition"

        async def boom():
            raise RuntimeError("no db in this test")

        async def main():
            with mock.patch.object(api, "start_metrics_server"), \\
                 mock.patch.object(api, "init_pg_pool", boom):
                try:
                    async with api.lifespan(api.app):
                        pass
                except RuntimeError:
                    pass

        asyncio.run(main())
        print(generate_latest(REGISTRY).decode())
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    text = out.stdout
    for line in (
        'wm_ir_tenantless_call_refused_total{call="completion",worker="<unknown>"} 0.0',
        'wm_ir_tenantless_call_refused_total{call="listing",worker="worker-management"} 0.0',
        'wm_ir_identity_missing_total{header="app_id",worker="<unknown>"} 0.0',
        'wm_ir_identity_missing_total{header="project_id",worker="<unknown>"} 0.0',
        'worker_management_actor_missing_total{caller="unknown"} 0.0',
        'worker_management_actor_missing_total{caller="improvement-loop-service"} 0.0',
    ):
        assert line in text, line
