"""Unit tests for worker/service_authenticator.py."""

import os
from unittest.mock import patch

import pytest

import src.worker.service_authenticator as sa_module
from src.worker.service_authenticator import ServiceAuthenticator, get_service_authenticator


class TestServiceAuthenticator:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_token_from_coding_license(self):
        auth = ServiceAuthenticator()
        with patch.dict(os.environ, {"CODING_LICENSE_TOKEN": "token-123"}):
            token = await auth.get_token()
            assert token == "token-123"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_token_from_service_license(self):
        auth = ServiceAuthenticator()
        with patch.dict(os.environ, {"SERVICE_LICENSE_TOKEN": "svc-token"}, clear=False):
            os.environ.pop("CODING_LICENSE_TOKEN", None)
            token = await auth.get_token()
            assert token == "svc-token"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_token_from_license_token(self):
        auth = ServiceAuthenticator()
        with patch.dict(os.environ, {"LICENSE_TOKEN": "lic-token"}, clear=False):
            os.environ.pop("CODING_LICENSE_TOKEN", None)
            os.environ.pop("SERVICE_LICENSE_TOKEN", None)
            token = await auth.get_token()
            assert token == "lic-token"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_token_no_env_raises(self):
        auth = ServiceAuthenticator()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CODING_LICENSE_TOKEN", None)
            os.environ.pop("SERVICE_LICENSE_TOKEN", None)
            os.environ.pop("LICENSE_TOKEN", None)
            with pytest.raises(RuntimeError, match="No license token"):
                await auth.get_token()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_cached_token(self):
        auth = ServiceAuthenticator()
        auth._token = "cached"
        token = await auth.get_token()
        assert token == "cached"


class TestGetServiceAuthenticator:
    def test_singleton(self):
        sa_module._authenticator = None
        a1 = get_service_authenticator()
        a2 = get_service_authenticator()
        assert a1 is a2
        sa_module._authenticator = None
