"""fr#1385 Option 1: a verified end-user token proves ``tenant_id`` OR a member tenant.

``token_proves_tenant`` is pinned to the cross-repo vector file shared with fr
(``tests/fixtures/tenant_membership_vectors.json``). The file's sha256 is asserted
here so a one-sided edit in either repo goes red.

MUTATIONS (each verified red): drop the ``effective in members`` arm; replace the
exact match with a case-insensitive one; drop the ``tenant_id`` required check.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from src.api.identity_context import TENANT_PROVENANCE_DELEGATED, set_identity_context
from src.auth import end_user_token, keycloak_roles
from src.auth.end_user_token import token_proves_tenant
from src.auth.tenant_proof import tenant_proof_refusal

FIXTURE = Path(__file__).parent / "fixtures" / "tenant_membership_vectors.json"
FIXTURE_SHA256 = (
    "65f15aa451bf31c966453c0eddf2a04efc3c1a247f6536631607b049eb74a478"  # pragma: allowlist secret
)

TENANT_A = "aaaaaaaa-0000-4000-8000-000000000001"
TENANT_B = "bbbbbbbb-0000-4000-8000-000000000002"
TENANT_C = "cccccccc-0000-4000-8000-000000000003"
SA_SUBJECT = "019ffe9f-0000-7000-8000-00000000f00d"
ISSUER = "https://auth.nonagentic.ai/realms/portal"
KID = "portal-key-1"


def _vectors() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_bytes())["vectors"]


def test_fixture_is_byte_identical_to_the_shared_contract():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


@pytest.mark.parametrize("vec", _vectors(), ids=lambda v: v["name"])
def test_predicate_vector(vec):
    assert token_proves_tenant(vec["claims"], vec["effective"]) is vec["proven"]


def test_vectors_cover_both_outcomes():
    outcomes = {v["proven"] for v in _vectors()}
    assert outcomes == {True, False}


# ------------------------------------------------------------ end to end


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    from src.settings import reset_settings_cache

    monkeypatch.setenv("WM_SECRET_KEY", secrets.token_hex(32))
    monkeypatch.setenv(
        "KEYCLOAK_JWKS_URL",
        "http://keycloak.keycloak.svc.cluster.local:8080/realms/secure-apps/protocol/openid-connect/certs",
    )
    monkeypatch.delenv("END_USER_JWKS_URL", raising=False)
    monkeypatch.delenv("END_USER_TOKEN_ISSUER", raising=False)
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    reset_settings_cache()
    end_user_token._jwks_cache.clear()
    yield
    end_user_token._jwks_cache.clear()
    set_identity_context(None, None, None)
    reset_settings_cache()


@pytest.fixture
def portal_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def portal_jwks(portal_key, monkeypatch: pytest.MonkeyPatch):
    pub = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(portal_key.public_key()))
    pub.update({"kid": KID, "alg": "RS256", "use": "sig"})

    async def fake_fetch(url: str) -> dict:
        return {"keys": [pub]}

    monkeypatch.setattr(keycloak_roles, "_fetch_jwks", fake_fetch)


def _mint(key, *, members: Any = None, alg: str = "RS256") -> str:
    claims: dict[str, Any] = {
        "sub": "end-user-1",
        "iss": ISSUER,
        "exp": int(time.time()) + 600,
        "typ": "Bearer",
        "azp": "fr-frontend",
        "tenant_id": TENANT_A,
    }
    if members is not None:
        claims["member_tenant_ids"] = members
    return jwt.encode(claims, key, algorithm=alg, headers={"kid": KID})


def _verify(token: str, effective: str) -> tuple[str, str]:
    return asyncio.run(end_user_token.verify_end_user_tenant(f"Bearer {token}", effective))


class TestVerifiedTokenMembership:
    def test_member_proves_b_and_not_c(self, portal_key, portal_jwks):
        token = _mint(portal_key, members=[TENANT_B])
        assert _verify(token, TENANT_B) == (TENANT_B, "ok")
        assert _verify(token, TENANT_A) == (TENANT_A, "ok")
        assert _verify(token, TENANT_C) == ("", "tenant_mismatch")

    def test_forged_member_claim_proves_nothing(self, portal_key, portal_jwks):
        forger = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        tenant, reason = _verify(_mint(forger, members=[TENANT_B]), TENANT_B)
        assert tenant == ""
        assert reason == "invalid:InvalidSignatureError"

    def test_unsigned_member_claim_proves_nothing(self, portal_jwks):
        token = jwt.encode(
            {
                "iss": ISSUER,
                "exp": int(time.time()) + 600,
                "typ": "Bearer",
                "azp": "fr-frontend",
                "tenant_id": TENANT_A,
                "member_tenant_ids": [TENANT_B],
            },
            key="",
            algorithm="none",
            headers={"kid": KID},
        )
        tenant, reason = _verify(token, TENANT_B)
        assert tenant == ""
        assert reason.startswith("invalid:")

    def test_tampered_member_claim_breaks_the_signature(self, portal_key, portal_jwks):
        # Re-encode the payload of a genuinely signed token with B added, keeping
        # the original signature: the claim set is forged, the signature is real.
        header, _payload, sig = _mint(portal_key, members=[]).split(".")
        claims = jwt.decode(_mint(portal_key, members=[]), options={"verify_signature": False})
        claims["member_tenant_ids"] = [TENANT_B]
        forged_payload = jwt.utils.base64url_encode(json.dumps(claims).encode()).decode()
        tenant, reason = _verify(f"{header}.{forged_payload}.{sig}", TENANT_B)
        assert tenant == ""
        assert reason == "invalid:InvalidSignatureError"


class TestDelegatedPathStillNeedsADelegator:
    def _stamp_and_refusal(self, token: str, tenant: str, delegator: str) -> str | None:
        async def go() -> str | None:
            set_identity_context(
                tenant, None, delegator, tenant_provenance=TENANT_PROVENANCE_DELEGATED
            )
            request = SimpleNamespace(
                headers={end_user_token.END_USER_TOKEN_HEADER: f"Bearer {token}"},
                state=SimpleNamespace(),
            )
            await end_user_token.stamp_end_user_proof(request, tenant, delegator)
            return tenant_proof_refusal(tenant)

        return asyncio.run(go())

    def test_member_tenant_is_proven_via_a_listed_delegator(self, portal_key, portal_jwks):
        token = _mint(portal_key, members=[TENANT_B])
        assert self._stamp_and_refusal(token, TENANT_B, SA_SUBJECT) is None

    def test_member_tenant_unproven_when_delegator_not_listed(
        self, portal_key, portal_jwks, monkeypatch
    ):
        token = _mint(portal_key, members=[TENANT_B])
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "someone-else")
        assert (
            self._stamp_and_refusal(token, TENANT_B, SA_SUBJECT)
            == "delegated_without_verified_end_user_token"
        )

    def test_non_member_tenant_is_refused(self, portal_key, portal_jwks):
        token = _mint(portal_key, members=[TENANT_B])
        assert (
            self._stamp_and_refusal(token, TENANT_C, SA_SUBJECT)
            == "delegated_without_verified_end_user_token"
        )
