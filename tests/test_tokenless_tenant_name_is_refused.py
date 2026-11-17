"""A tokenless ``X-Tenant-Id`` carrying a NAME must not become the seed tenant.

⛔ WHAT WAS ACTUALLY WRONG

On the tokenless trusted-network branch, ``X-Tenant-Id: n8n-dev`` resolved to
``aaaaaaaa-0000-4000-8000-000000000001`` and the request was SERVED. Measured on
worker4 2026-09-16 against the authoritative ``licenses.tenants`` registry::

    aaaaaaaa-0000-4000-8000-000000000001 | n8n-dev-portal-realm   <- the SEED tenant
    bbbbbbbb-0000-4000-8000-0000000000a5 | n8n-dev                <- what the caller meant

So a caller naming production got a real, valid, DIFFERENT tenant, with a 200.
Not a 404, not a 500 — a plausible wrong answer, which is the failure mode that
survives review. Reads returned that tenant's rows; writes landed on it.

⛔ AND IT IS NOT A ONE-ROW TYPO. ``fr-test`` collides identically: this
deployment's map says ``…-000000000099`` (registered as ``fr-test-user``) while
a tenant literally named ``fr-test`` exists at
``aaaaaaaa-0000-4000-8000-0000000000a8``. Two of the six names in
``tenant_uuids()`` name a different live tenant in the other registry.

THE CHOSEN FAILURE DIRECTION, AND WHY IT IS NOT "RESOLVE IT PROPERLY"

Resolving the name to "its real id" is not implementable: the two registries
disagree about which id that is, and this service cannot know which one a
tokenless caller meant. Defaulting-but-loudly does not help either — the bill
and the rows still land on the wrong tenant, just with a log line next to them.
A UUID means one thing in both registries, so requiring one is the only option
under which silent misattribution is impossible.

⚠️ THE ABSENT-HEADER CASE IS DELIBERATELY UNCHANGED. No header still means
``default_tenant_uuid()``. That is this deployment's own configuration rather than a
caller's claim, it is already stamped ``tenant_provenance=default`` and counted
by ``_audit_identity_omission``, and it carries the live traffic. Changing it
belongs to the ``require_authenticated_tenant`` flag, not here.

⛔ HOW MUCH LIVE TRAFFIC — AND WHY THE OBVIOUS NUMBER IS THE WRONG ONE. An
earlier version of this docstring said "2,524 tokenless-asserted requests over
14d", taken from
``increase(worker_management_unauthenticated_tenant_assertion_total[14d])``.
That counter RESETS on every pod roll, so ``increase()`` extrapolates across the
reset and returns a non-integer — the live query reads ``2504.008``, and the
``.008`` is the tell that it is an estimate, not a count. Do not quote it as one.

The exact figure comes from Loki, which stores the log line rather than a
counter: 1,501 ``unauthenticated tenant assertion`` lines over 29 days (its full
retention).

⭐ AND THE NAME-VALUED SHARE OF THOSE IS **ONE**, WHICH IS THIS ISSUE'S OWN
PROBE. Filtering those 1,501 for a tenant NAME leaves a single line:
``peer=127.0.0.1``, ``user_agent='Python-urllib/3.14'``, ``path=/workers_states``
at 2026-09-15T05:33:40.491Z — 204 ms after a sibling line from the same peer and
UA carrying the production UUID, and 3 m 24 s before issue #264 was filed. Those
are the first two rows of the table in #264's own description, which says it was
"live-probed from inside the worker-management pod". The third row (no header)
cannot appear, because ``_audit_unauthenticated_tenant_assertion`` returns early
with no header — two lines for three probes is what the code predicts.

So the boundary this file pins has, in 29 days of production, refused exactly one
request that was not already refused, and that request was the diagnostic that
discovered the defect. ⚠️ Bounded, not absolute: Loki retains 30 days, and
``_assertion_log_seen`` dedups on (tenant, peer, shape, caller) for 300 s — which
cannot conceal a RECURRING caller (it would re-emit every 5 minutes) but can
merge distinct callers inside one 300 s window.
"""

from __future__ import annotations

import logging
import uuid

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.auth_middleware import AuthenticationMiddleware
from src.api.tenant_constants import default_tenant_uuid, tenant_uuids

# The two rows the whole defect rests on, re-derived from worker4's
# ``licenses.tenants`` on 2026-09-16 rather than copied from another module.
SEED_TENANT_UUID = "aaaaaaaa-0000-4000-8000-000000000001"  # licences: n8n-dev-portal-realm
PRODUCTION_N8N_DEV_UUID = "bbbbbbbb-0000-4000-8000-0000000000a5"  # licences: n8n-dev


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthenticationMiddleware, bypass_auth=False)

    @application.get("/api/probe")
    async def probe(request: Request) -> dict[str, str]:
        return {"tenant_id": request.state.tenant_id}

    return application


def _get(app: FastAPI, tenant_header: str | None) -> tuple[int, dict]:
    """Drive the tokenless branch from a peer inside TRUSTED_CIDRS."""
    headers = {"X-Tenant-Id": tenant_header} if tenant_header is not None else {}
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        resp = client.get("/api/probe", headers=headers)
    body = (
        resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    )
    return resp.status_code, body


# ---------------------------------------------------------------- drive 1: UUID


def test_a_valid_uuid_header_resolves_to_itself(app: FastAPI) -> None:
    """A canonical UUID is the one spelling that means one thing in both registries."""
    status, body = _get(app, PRODUCTION_N8N_DEV_UUID)
    assert status == 200
    assert body["tenant_id"] == PRODUCTION_N8N_DEV_UUID


def test_a_uuid_header_is_still_case_folded(app: FastAPI) -> None:
    """Accepting UUIDs must not regress the existing case-folding contract."""
    status, body = _get(app, PRODUCTION_N8N_DEV_UUID.upper())
    assert status == 200
    assert body["tenant_id"] == PRODUCTION_N8N_DEV_UUID


# ---------------------------------------------------------------- drive 2: NAME


def test_the_name_n8n_dev_is_refused_and_never_becomes_the_seed_tenant(app: FastAPI) -> None:
    """⭐ THE DEFECT, DRIVEN DIRECTLY. Before the fix this was 200 + the seed tenant."""
    status, body = _get(app, "n8n-dev")

    assert status == 400, "a tenant NAME must not be accepted on a tokenless request"
    # The load-bearing half. A refusal that still handed back the seed tenant
    # would pass a bare status check, so assert the misattribution is absent.
    assert body.get("tenant_id") != SEED_TENANT_UUID
    assert body.get("tenant_id") is None
    assert body["error"]["code"] == "INVALID_TENANT"


def test_the_refusal_says_why_a_name_cannot_identify_a_tenant(app: FastAPI) -> None:
    """A 400 an operator cannot action just moves the silence downstream."""
    _, body = _get(app, "n8n-dev")
    message = body["error"]["message"]
    assert "UUID" in message
    assert "NAME" in message


@pytest.mark.parametrize("name", sorted(tenant_uuids()))
def test_every_known_tenant_name_is_refused(app: FastAPI, name: str) -> None:
    """Not just ``n8n-dev``: the whole crosswalk is off the inbound boundary.

    Parametrised over the live map rather than a hand-written list, so a name
    added to ``tenant_uuids()`` later cannot quietly reopen the hole — the guard
    cannot be blinded by a list I forgot to extend.
    """
    status, body = _get(app, name)
    assert status == 400
    assert body.get("tenant_id") != tenant_uuids()[name]


@pytest.mark.parametrize("padded", ["  n8n-dev", "n8n-dev  ", "  n8n-dev  ", "\tn8n-dev\n"])
def test_a_whitespace_padded_name_is_refused_too(app: FastAPI, padded: str) -> None:
    """⛔ THE GUARD MUST NORMALISE BEFORE IT COMPARES, AND THIS IS WHY.

    Found by mutation testing, not by review: deleting the ``.strip()`` in
    ``resolve_asserted_tenant_id`` left every other test in this file green.
    The bypass is total, because the two resolvers strip at different moments —
    an unstripped ``" n8n-dev "`` misses the ``in tenant_uuids()`` guard, falls
    through to the name-TOLERANT ``resolve_tenant_id``, which strips it itself
    and maps the name to the seed tenant. Measured with the guard's strip
    removed: ``resolve_tenant_id("  n8n-dev  ")`` returns
    ``aaaaaaaa-0000-4000-8000-000000000001``.

    So a single space in a caller's config would have restored the entire
    defect, past a guard that looks correct. A membership test against raw
    caller input is only as good as the normalisation in front of it.
    """
    status, body = _get(app, padded)
    assert status == 400, f"{padded!r} bypassed the name guard"
    assert body.get("tenant_id") != SEED_TENANT_UUID
    assert body.get("tenant_id") is None


def test_a_whitespace_padded_uuid_is_still_accepted(app: FastAPI) -> None:
    """The strip must normalise, not become a reason to refuse valid callers."""
    status, body = _get(app, f"  {PRODUCTION_N8N_DEV_UUID}  ")
    assert status == 200
    assert body["tenant_id"] == PRODUCTION_N8N_DEV_UUID


def test_the_two_colliding_names_do_not_resolve_to_the_other_registrys_tenant(
    app: FastAPI,
) -> None:
    """``n8n-dev`` and ``fr-test`` each name a DIFFERENT live tenant in licences."""
    for name in ("n8n-dev", "fr-test"):
        status, body = _get(app, name)
        assert status == 400, f"{name} is ambiguous across registries and must be refused"
        assert "tenant_id" not in body


# ----------------------------------------------------------- drive 3: MALFORMED


@pytest.mark.parametrize(
    "bogus",
    [
        "sales",  # a real production literal, and a real licences tenant name
        "__system_workflow__",
        "1769980520185202330",
        "gerard-dev",  # retired from the realm 2026-08-12
        "00000000-0000-0000-0000-000000000000",  # nil uuid, refused BY VALUE
        "urn:uuid:bbbbbbbb-0000-4000-8000-0000000000a5",
        "{bbbbbbbb-0000-4000-8000-0000000000a5}",
        # A tenant UUID with its dashes stripped, NOT a credential — detect-secrets
        # scores any 32-char hex run as a high-entropy string. It is in this list
        # precisely because `uuid.UUID()` accepts the spelling and the boundary
        # must not.
        "aaaaaaaac92a7e378fddeb0d8c5396f9",  # pragma: allowlist secret
        "' OR 1=1 --",
        "   ",
    ],
)
def test_a_malformed_header_is_a_clean_400_and_not_the_default(app: FastAPI, bogus: str) -> None:
    """Unchanged behaviour, pinned so the name change did not widen or narrow it."""
    status, body = _get(app, bogus)
    assert status == 400
    # A silent fallback here would be the same defect wearing a different hat.
    assert body.get("tenant_id") != default_tenant_uuid()


# ------------------------------------------------------------ drive 4: NO HEADER


def test_no_header_still_uses_the_default_tenant(app: FastAPI) -> None:
    """⚠️ DELIBERATELY UNCHANGED — see the module docstring.

    This is the test that keeps the fix honest about its own scope. Tightening
    the absent-header case would 400 the live tokenless traffic measured on this
    service, which is a different decision behind a different flag.
    """
    status, body = _get(app, None)
    assert status == 200
    assert body["tenant_id"] == default_tenant_uuid()
    assert uuid.UUID(body["tenant_id"])


# -------------------------------------------------------- drive 5: LICENSED PATH


def test_the_licensed_path_is_unchanged_and_still_accepts_a_bearer_token(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token branch must not be collateral damage of an inbound-header rule.

    A ``Bearer`` value never reaches ``resolve_asserted_tenant_id`` at all — it
    is extracted and introspected first — so the tenant here comes from the
    introspected licence, and a NAME rule on the tokenless branch cannot touch
    it. Asserted rather than assumed, because the two paths share one header.
    """
    import src.api.auth_middleware as mw

    async def _fake_introspect(token: str) -> dict:
        return {
            "active": True,
            "tenant_id": PRODUCTION_N8N_DEV_UUID,
            "license_id": "lic-1",
            "subject_id": "sub-1",
        }

    monkeypatch.setattr(mw, "_introspect_license", _fake_introspect)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        resp = client.get("/api/probe", headers={"X-Tenant-Id": "Bearer tok-abc"})

    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == PRODUCTION_N8N_DEV_UUID


# --------------------------------------- drive 6: X-Effective-Tenant-Id (door 3)
#
# ⛔ THIS WHOLE SECTION EXISTS BECAUSE THE FIRST VERSION OF THIS FIX MISSED IT.
# `X-Effective-Tenant-Id` is the THIRD caller-asserted door, and #264's defect
# went through it verbatim: before the rewire, on this branch,
# `_effective_tenant("n8n-dev")` returned `aaaaaaaa-…0001` — the seed tenant —
# and the request was served 200. A licence proves WHO; this header then
# re-scopes WHICH on the caller's say-so, which is why the call site stamps
# `tenant_proven=not delegated`. Caller-asserted is caller-asserted whether or
# not a token was presented.


DELEGATING_SUBJECT = "01a0138e-e7b9-712c-b140-cf20d7b9e644"  # live AUTH_DELEGATING_SUBJECTS[0]


def _delegated(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, effective: str
) -> tuple[int, dict, str]:
    """Drive the delegation branch as a configured delegating service account."""
    import src.api.auth_middleware as mw

    async def _fake_introspect(token: str) -> dict:
        return {
            "active": True,
            "tenant_id": PRODUCTION_N8N_DEV_UUID,
            "license_id": "lic-d",
            "subject_id": DELEGATING_SUBJECT,
        }

    monkeypatch.setattr(mw, "_introspect_license", _fake_introspect)
    monkeypatch.setattr(mw, "_delegating_subjects", lambda: frozenset({DELEGATING_SUBJECT}))

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        resp = client.get(
            "/api/probe",
            headers={"X-Tenant-Id": "Bearer tok-d", "X-Effective-Tenant-Id": effective},
        )
    body = (
        resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    )
    return resp.status_code, body, resp.text


def test_delegating_to_a_uuid_still_works(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive control. Without it, a refusal test proves only that something broke."""
    target = "aaaaaaaa-0000-4000-8000-0000000000a7"  # cdmsfirst, agreed in both registries
    status, body, _ = _delegated(app, monkeypatch, target)
    assert status == 200
    assert body["tenant_id"] == target


def test_delegating_by_NAME_is_still_ACCEPTED_and_this_is_a_deliberate_deferral(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⛔ THIS PINS A KNOWN DEFECT AS CURRENT BEHAVIOUR. READ WHY BEFORE "FIXING" IT.

    A name here misattributes exactly as it did on the tokenless door: the
    request below resolves to the SEED tenant and is served 200. That is the same
    bug as issue #264, and it is deliberately still open on this header.

    ⛔ IT IS DEFERRED BECAUSE IT CANNOT BE SHOWN SAFE, NOT BECAUSE IT IS SAFE.
    The tokenless door was closed on evidence: its log line carries the RAW
    header, so Loki could prove only 1 of 1,501 assertions over 30d was
    name-valued (and that one was #264's own probe). This door had no such line —
    it logged only the resolved uuid, so a name-sender and a uuid-sender were
    byte-identical. Measured over the same window: 72,351 delegated requests, of
    which 58,710 (81.1%) resolved to a uuid a NAME could equally have produced.
    Refusing names here would 400 an unknown share of 81% of delegated traffic.

    ⚠️ AND THE CALLER ENUMERATION CANNOT CLOSE IT. hirag, discovery-engine and
    kag_orchestrator each resolve to a uuid before forwarding — but the logs carry
    five distinct delegating subjects, and the second-largest
    (``8382f0b3-…``, 9,270 requests) is not in the live
    ``AUTH_DELEGATING_SUBJECTS`` at all.

    ⭐ THE EXIT CONDITION IS MECHANICAL, NOT A JUDGEMENT CALL. The delegation log
    now records ``raw=``. Once a full retention window has passed, group those
    lines by the raw value; if the name count is 0, change ``_effective_tenant``
    to ``resolve_asserted_tenant_id`` and invert this test. Until then this
    assertion is what stops the door being closed on an argument instead of a
    measurement.
    """
    status, body, _ = _delegated(app, monkeypatch, "n8n-dev")
    assert status == 200, "deferred: closing this door needs the raw-value data first"
    assert body["tenant_id"] == SEED_TENANT_UUID  # the defect, pinned as still-open


def test_the_delegation_log_records_the_RAW_header_not_only_the_resolved_uuid(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """⭐ THE INSTRUMENT THAT MAKES THE DEFERRED DECISION MEASURABLE.

    Without ``raw=`` the two spellings are indistinguishable in Loki, which is the
    only reason this door is still open. This test is what stops the field being
    "tidied away" later — deleting it would silently restore the blindness and no
    other test would notice.
    """
    with caplog.at_level(logging.INFO, logger="src.api.auth_middleware"):
        _delegated(app, monkeypatch, "n8n-dev")
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "effective-tenant delegation" in line, line
    assert "raw='n8n-dev'" in line, line
    assert "resolved_from_name=True" in line, line


def test_the_delegation_log_marks_a_uuid_sender_as_not_name_resolved(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The discriminator must actually discriminate — a flag that is always True is not one."""
    target = "aaaaaaaa-0000-4000-8000-000000000100"
    with caplog.at_level(logging.INFO, logger="src.api.auth_middleware"):
        _delegated(app, monkeypatch, target)
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "resolved_from_name=False" in line, line
    assert f"raw='{target}'" in line, line


def test_a_malformed_delegation_value_is_still_refused(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged behaviour, pinned so the deferral did not widen what IS refused."""
    for bogus in ("sales", "' OR 1=1 --", "00000000-0000-0000-0000-000000000000"):
        status, _, _ = _delegated(app, monkeypatch, bogus)
        assert status == 400, f"{bogus!r} was admitted"


def test_a_licence_introspecting_to_a_name_is_refused(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⚠️ REWRITTEN — the first version of this test was VACUOUS, twice over.

    It asserted a name from introspection is *honoured*, on the rationale that
    "licences picked which registry it meant". That is backwards: resolution
    goes through ``tenant_uuids()``, the PORTAL REALM's map, so an introspected
    ``n8n-dev`` resolved to the seed tenant while licences' own registry has
    ``n8n-dev`` as ``aaaaaaaa-…``. Honouring it misattributes a licence to a
    different tenant than the authority that issued it.

    ⛔ AND THE OLD FIXTURE COULD NOT HAVE CAUGHT THAT, because it used
    ``george-dev`` — the ONE name in ``tenant_uuids()`` that licences does not
    contain at all. For that value the right answer and the wrong answer are the
    same uuid, so the test passed either way. It now uses ``n8n-dev``, where the
    two registries genuinely disagree, so the fixture can tell them apart.

    Costs nothing in production: ``licenses.tenants.id`` and
    ``license.tenant_id`` are both ``uuid NOT NULL`` (measured worker4
    2026-09-16), so a name can only arrive from a broken upstream.
    """
    import src.api.auth_middleware as mw

    async def _fake_introspect(token: str) -> dict:
        return {
            "active": True,
            "tenant_id": "n8n-dev",
            "license_id": "lic-2",
            "subject_id": "sub-2",
        }

    monkeypatch.setattr(mw, "_introspect_license", _fake_introspect)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        resp = client.get("/api/probe", headers={"X-Tenant-Id": "Bearer tok-xyz"})

    assert resp.status_code == 401, "a name from introspection must not be adopted"
    # The load-bearing half: it must not have silently become the seed tenant.
    assert SEED_TENANT_UUID not in resp.text
