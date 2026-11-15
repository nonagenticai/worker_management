"""X-Project-Id: always-on per-caller miss counter + default-OFF 400 gate.

Owner ruling 2026-10-08: every conveyor call carries its fr projectId. The
counter ``worker_management_project_id_missing_total{caller,reason}`` fires in
BOTH flag states; ``REQUIRE_PROJECT_ID_HEADER`` changes only the outcome.

⛔ ``caller`` is what the middleware AUTHENTICATED (verified azp, else the auth
branch). A forged ``X-App-Id`` must never move a miss onto another label — that
is what would let "zero missing for caller X" be faked before enforcement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.api.auth_middleware import (
    PROJECT_ID_MISSING_TOTAL,
    PROJECT_ID_REFUSED_TOTAL,
    AuthenticationMiddleware,
)
from src.api.identity_context import get_project_id_from_context
from src.settings import reset_settings_cache

if TYPE_CHECKING:
    from collections.abc import Iterator

TENANT = "0195f48b-d000-7000-8000-000000000001"
SUBJECT = "0195f48b-d000-7000-8000-0000000000b2"
PROJECT_ID = "0195f48b-d000-7000-8000-0000000000a1"
VERIFIED_AZP = "improvement-loop-service"  # in the built-in caller allowlist
LICENSE_ID = "0195f48b-d000-7000-8000-0000000000f1"
LICENSE_ID_2 = "0195f48b-d000-7000-8000-0000000000f2"
LIC = f"licence:{LICENSE_ID}"  # label of an UNMAPPED verified licence


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch) -> Iterator[None]:
    monkeypatch.delenv("REQUIRE_PROJECT_ID_HEADER", raising=False)
    monkeypatch.delenv("WORKER_MANAGEMENT_REQUIRE_PROJECT_ID_HEADER", raising=False)
    monkeypatch.delenv("CALLER_ALLOWED_CLIENTS", raising=False)
    monkeypatch.delenv("WORKER_MANAGEMENT_PROJECT_ID_CALLER_LICENCES", raising=False)
    monkeypatch.delenv("PROJECT_ID_CALLER_LICENCES", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def enforce(monkeypatch) -> None:
    monkeypatch.setenv("REQUIRE_PROJECT_ID_HEADER", "true")
    reset_settings_cache()


def _app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, str]:
        return {"project_id": get_project_id_from_context()}

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


def _get(
    path: str,
    headers: dict[str, str],
    *,
    trusted: bool = True,
    azp: str = "",
    license_id: str = LICENSE_ID,
):
    """One request through the real middleware; licence path when not ``trusted``."""
    client = TestClient(_app(), raise_server_exceptions=False)
    with (
        patch("src.api.auth_middleware._is_trusted_network", return_value=trusted),
        patch(
            "src.api.auth_middleware.verified_caller_identity",
            new_callable=AsyncMock,
            return_value=(azp, ()),
        ),
        patch(
            "src.api.auth_middleware._lookup_tenant_byo_mode",
            new_callable=AsyncMock,
            return_value="local_allowed",
        ),
        patch(
            "src.api.auth_middleware._introspect_license",
            new_callable=AsyncMock,
            return_value={
                "active": True,
                "tenant_id": TENANT,
                "subject_id": SUBJECT,
                "license_id": license_id,
                "byo_mode": "local_allowed",
            },
        ),
    ):
        return client.get(path, headers=headers)


def _missing(caller: str, reason: str) -> float:
    return PROJECT_ID_MISSING_TOTAL.labels(caller=caller, reason=reason)._value.get()


def _refused(caller: str, reason: str) -> float:
    return PROJECT_ID_REFUSED_TOTAL.labels(caller=caller, reason=reason)._value.get()


def _all_missing() -> float:
    return sum(
        s.value
        for m in REGISTRY.collect()
        if m.name == "worker_management_project_id_missing"
        for s in m.samples
        if s.name.endswith("_total")
    )


TRUSTED = {"X-Tenant-Id": TENANT}
LICENCE = {"X-Tenant-Id": "Bearer tok"}


def test_zero_seeded_series_exist_at_import() -> None:
    for counter in ("missing", "refused"):
        for caller in ("trusted_network", "unknown", VERIFIED_AZP):
            for reason in ("absent", "invalid"):
                assert (
                    REGISTRY.get_sample_value(
                        f"worker_management_project_id_{counter}_total",
                        {"caller": caller, "reason": reason},
                    )
                    is not None
                ), (counter, caller, reason)


def test_flag_off_absent_proceeds_and_counts_by_auth_branch() -> None:
    before_t, before_l = _missing("trusted_network", "absent"), _missing(LIC, "absent")
    assert _get("/api/probe", TRUSTED).status_code == 200
    assert _get("/api/probe", LICENCE, trusted=False).status_code == 200
    assert _missing("trusted_network", "absent") == before_t + 1
    assert _missing(LIC, "absent") == before_l + 1


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-uuid",
        "00000000-0000-0000-0000-000000000000",
        "{0195f48b-d000-7000-8000-0000000000a1}",
        "0195f48bd00070008000" + "0000000000a1",
    ],
)
def test_flag_off_invalid_proceeds_counted_invalid_and_still_relayed(bad: str) -> None:
    before = _missing(LIC, "invalid")
    resp = _get("/api/probe", {**LICENCE, "X-Project-Id": bad}, trusted=False)
    assert resp.status_code == 200
    # Flag off changes nothing about the relay: the value goes on as sent.
    assert resp.json()["project_id"] == bad
    assert _missing(LIC, "invalid") == before + 1


def test_flag_on_absent_is_400_typed_body_counted_and_refused(enforce) -> None:
    before_m, before_r = _missing(LIC, "absent"), _refused(LIC, "absent")
    resp = _get("/api/probe", LICENCE, trusted=False)
    assert resp.status_code == 400
    assert resp.json() == {
        "error": {
            "code": "PROJECT_ID_REQUIRED",
            "reason": "absent",
            "message": "X-Project-Id must be the fr project UUID this call belongs to",
            "status": 400,
        }
    }
    assert _missing(LIC, "absent") == before_m + 1
    assert _refused(LIC, "absent") == before_r + 1


def test_flag_on_invalid_is_400_with_reason_invalid(enforce) -> None:
    before = _refused("trusted_network", "invalid")
    resp = _get("/api/probe", {**TRUSTED, "X-Project-Id": "nope"})
    assert resp.status_code == 400
    assert resp.json()["error"]["reason"] == "invalid"
    assert _refused("trusted_network", "invalid") == before + 1


@pytest.mark.parametrize("value", [PROJECT_ID, PROJECT_ID.upper()])
def test_flag_on_valid_proceeds_unchanged_and_is_not_counted(enforce, value: str) -> None:
    before = _all_missing()
    resp = _get("/api/probe", {**LICENCE, "X-Project-Id": value}, trusted=False)
    assert resp.status_code == 200
    assert resp.json()["project_id"] == value  # relayed verbatim, not lowercased
    assert _all_missing() == before


@pytest.mark.parametrize("flag", [False, True])
def test_public_endpoint_never_counted_or_refused(monkeypatch, flag: bool) -> None:
    monkeypatch.setenv("REQUIRE_PROJECT_ID_HEADER", str(flag).lower())
    reset_settings_cache()
    before = _all_missing()
    assert _get("/health", {}, trusted=False).status_code == 200
    assert _all_missing() == before


def test_unauthenticated_request_is_401_not_400_and_not_counted(enforce) -> None:
    before = _all_missing()
    assert _get("/api/probe", {}, trusted=False).status_code == 401
    assert _all_missing() == before


def test_caller_is_verified_azp_and_forged_app_id_does_not_move_it() -> None:
    before_azp = _missing(VERIFIED_AZP, "absent")
    before_forged = _missing("n8n", "absent")
    resp = _get("/api/probe", {**LICENCE, "X-App-Id": "n8n"}, trusted=False, azp=VERIFIED_AZP)
    assert resp.status_code == 200
    assert _missing(VERIFIED_AZP, "absent") == before_azp + 1
    assert _missing("n8n", "absent") == before_forged


def test_forged_app_id_without_verified_azp_stays_on_auth_branch() -> None:
    before_l = _missing(LIC, "absent")
    before_forged = _missing("n8n", "absent")
    _get("/api/probe", {**LICENCE, "X-App-Id": "n8n"}, trusted=False)
    assert _missing(LIC, "absent") == before_l + 1
    assert _missing("n8n", "absent") == before_forged


# ------------------------------------------------ verified licence -> caller name


def _map(monkeypatch, raw: str, *, enforce_flag: bool = False) -> None:
    monkeypatch.setenv("WORKER_MANAGEMENT_PROJECT_ID_CALLER_LICENCES", raw)
    monkeypatch.setenv("REQUIRE_PROJECT_ID_HEADER", str(enforce_flag).lower())
    reset_settings_cache()


def test_several_mapped_licences_share_one_name_and_forged_app_id_is_ignored(
    monkeypatch,
) -> None:
    _map(monkeypatch, f"{LICENSE_ID}=n8n, {LICENSE_ID_2.upper()}=n8n")
    before = _missing("n8n", "absent")
    _get("/api/probe", {**LICENCE, "X-App-Id": "requirements"}, trusted=False)
    _get("/api/probe", LICENCE, trusted=False, license_id=LICENSE_ID_2)
    assert _missing("n8n", "absent") == before + 2
    assert _missing("requirements", "absent") == 0


def test_mapped_names_are_zero_seeded(monkeypatch) -> None:
    from src.api.auth_middleware import preinit_project_id_metrics

    _map(monkeypatch, f"{LICENSE_ID}=seeded-name")
    preinit_project_id_metrics()
    assert (
        REGISTRY.get_sample_value(
            "worker_management_project_id_missing_total",
            {"caller": "seeded-name", "reason": "absent"},
        )
        == 0.0
    )


def test_unmapped_licence_is_labelled_by_its_id(monkeypatch) -> None:
    _map(monkeypatch, f"{LICENSE_ID_2}=n8n")
    before = _missing(LIC, "absent")
    _get("/api/probe", LICENCE, trusted=False)
    assert _missing(LIC, "absent") == before + 1


def test_verified_azp_outranks_a_mapped_licence(monkeypatch) -> None:
    _map(monkeypatch, f"{LICENSE_ID}=n8n")
    before = _missing(VERIFIED_AZP, "absent")
    _get("/api/probe", LICENCE, trusted=False, azp=VERIFIED_AZP)
    assert _missing(VERIFIED_AZP, "absent") == before + 1


def test_licence_label_is_sanitised_and_capped() -> None:
    before = _missing("licence:abc-def", "absent")
    _get("/api/probe", LICENCE, trusted=False, license_id="ABC {!} -DEF")
    assert _missing("licence:abc-def", "absent") == before + 1
    capped = "licence:" + "a" * 36
    before = _missing(capped, "absent")
    _get("/api/probe", LICENCE, trusted=False, license_id="a" * 500)
    assert _missing(capped, "absent") == before + 1


BAD_MAPS = [
    f"{LICENSE_ID}=n8n,{LICENSE_ID}=other",  # duplicate id
    f"{LICENSE_ID}=N8N",  # bad name
    f"{LICENSE_ID}:n8n",  # bad syntax
    "not-a-uuid=n8n",
]


@pytest.mark.parametrize("raw", BAD_MAPS)
def test_bad_map_with_flag_on_fails_settings_validation(monkeypatch, raw: str) -> None:
    from src.settings import Settings

    _map(monkeypatch, raw, enforce_flag=True)
    with pytest.raises(RuntimeError, match="PROJECT_ID_CALLER_LICENCES"):
        Settings()


@pytest.mark.parametrize("raw", BAD_MAPS)
def test_bad_map_with_flag_off_starts_warns_and_labels_by_id(monkeypatch, caplog, raw: str) -> None:
    from src.api.auth_middleware import _parsed_caller_licences
    from src.settings import Settings

    _map(monkeypatch, raw)
    Settings()  # starts
    _parsed_caller_licences.cache_clear()
    before = _missing(LIC, "absent")
    with caplog.at_level("WARNING", logger="src.api.auth_middleware"):
        assert _get("/api/probe", LICENCE, trusted=False).status_code == 200
        _get("/api/probe", LICENCE, trusted=False)
    assert _missing(LIC, "absent") == before + 2
    warned = [r for r in caplog.records if "PROJECT_ID_CALLER_LICENCES ignored" in r.message]
    assert len(warned) == 1  # once, not per request


def test_forged_x_license_id_never_moves_the_label(monkeypatch) -> None:
    # The label comes from the INTROSPECTED licence, never the forwarded header: a caller
    # holding licence F1 that sends X-License-Id: <F2> must still count as F1 — and must
    # not be promoted to F2's mapped name.
    _map(monkeypatch, f"{LICENSE_ID_2}=n8n")
    before_own, before_n8n = _missing(LIC, "absent"), _missing("n8n", "absent")
    _get("/api/probe", {**LICENCE, "X-License-Id": LICENSE_ID_2}, trusted=False)
    assert _missing(LIC, "absent") == before_own + 1
    assert _missing("n8n", "absent") == before_n8n


def test_a_failing_counter_never_fails_the_request_with_the_flag_off() -> None:
    # Observation must not become an outage: if the metric itself blows up, the request
    # still proceeds (flag off) with its project relayed.
    with patch.object(PROJECT_ID_MISSING_TOTAL, "labels", side_effect=RuntimeError("boom")):
        absent = _get("/api/probe", TRUSTED)
        invalid = _get("/api/probe", {**TRUSTED, "X-Project-Id": "not-a-uuid"})
    assert absent.status_code == 200
    assert invalid.status_code == 200
