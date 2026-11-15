"""Tests for WorkerInstanceCache, RateLimiter internals, and workers endpoint edge cases.

Targets coverage gaps NOT handled by existing test files:
- test_rate_limiting_unit.py: covers enum, dataclass, fail-open, middleware basics
- test_workers_api.py / test_api_endpoints_coverage.py: cover CRUD happy paths
- test_worker_creation_edge_cases.py: covers model validation edge cases

This file covers:
- _globals.py: WorkerInstanceCache lifecycle (get_or_create, health check,
  cleanup, remove) with real caching/locking logic
- rate_limiting.py: RedisRateLimiter strategy dispatch, get_rate_limit_status
  with Redis, _get_applicable_rules, _build_rate_limit_key, get_rate_limit_status
  module function
- models.py: uncovered Pydantic validator paths
- error_handlers: general_exception_handler debug/dev mode
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# Helpers
# ============================================================================


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# WorkerInstanceCache tests
# ============================================================================


class TestWorkerInstanceCache:
    """Test the real caching, locking, and health logic of WorkerInstanceCache."""

    @pytest.fixture
    def cache(self):
        from src.api.workers._globals import WorkerInstanceCache

        return WorkerInstanceCache()

    @pytest.mark.asyncio
    async def test_is_worker_healthy_returns_true_by_default(self, cache):
        """Worker with no check_health method should be considered healthy."""
        worker = MagicMock(spec=[])  # No attributes
        result = await cache._is_worker_healthy(worker)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_worker_healthy_false_on_check_health_false(self, cache):
        worker = MagicMock()
        worker.check_health = AsyncMock(return_value=False)
        result = await cache._is_worker_healthy(worker)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_worker_healthy_false_on_exception(self, cache):
        worker = MagicMock()
        worker.check_health = AsyncMock(side_effect=RuntimeError("boom"))
        result = await cache._is_worker_healthy(worker)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_worker_healthy_checks_mcp_servers_ready(self, cache):
        """Worker with mcp_servers config should check mcp_servers_ready."""
        worker = MagicMock()
        # Remove check_health so it falls through to MCP check
        del worker.check_health
        worker.config = MagicMock()
        worker.config.mcp_servers = ["server1"]
        worker.mcp_servers_ready = False
        result = await cache._is_worker_healthy(worker)
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_worker_handles_cleanup_error(self, cache):
        worker = MagicMock()
        worker.cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        cache._workers["test_worker"] = worker

        # Should not raise, and should still remove the worker
        await cache._cleanup_worker("test_worker")
        assert "test_worker" not in cache._workers

    @pytest.mark.asyncio
    async def test_cleanup_worker_without_cleanup_method(self, cache):
        worker = MagicMock(spec=[])  # No cleanup method
        cache._workers["test_worker"] = worker

        await cache._cleanup_worker("test_worker")
        assert "test_worker" not in cache._workers

    @pytest.mark.asyncio
    async def test_remove_worker_evicts_and_leaves_no_lock_behind(self, cache):
        """Eviction no longer depends on a pre-seeded lock entry.

        This test used to seed ``cache._worker_locks["test_worker"] =
        asyncio.Lock()`` and was named ``test_remove_worker_with_lock``, because
        ``remove_worker`` skipped any key that had no entry. ``_worker_locks`` is
        now reference counted and reaped at zero users, so the resting state of
        every key is "no entry" and that seeding would have made the test assert
        the one case that no longer occurs in production. See
        ``tests/test_worker_instance_cache_lock_lifecycle.py``.
        """
        worker = MagicMock()
        worker.cleanup = AsyncMock()
        cache._workers["test_worker"] = worker
        assert cache._worker_locks == {}

        await cache.remove_worker("test_worker")
        assert "test_worker" not in cache._workers
        assert cache._worker_locks == {}, "remove_worker leaked a lock entry"

    @pytest.mark.asyncio
    async def test_remove_worker_for_an_uncached_name_is_a_noop(self, cache):
        """A name with no cached instance is skipped -- and mints no lock.

        Formerly ``test_remove_worker_without_lock_is_noop``, whose name claimed
        the no-op came from the missing LOCK. It does not: ``remove_worker``
        builds its key list from ``_workers``, so an uncached name yields no
        keys and the lock dict is never consulted.
        """
        await cache.remove_worker("nonexistent")  # Should not raise
        assert cache._worker_locks == {}


# ============================================================================
# RateLimiter - strategy dispatch and module-level function
# ============================================================================


class TestRateLimiterStrategyDispatch:
    """Test RedisRateLimiter.check_rate_limit routes to correct strategy."""

    @pytest.mark.asyncio
    async def test_routes_to_sliding_window(self):
        from src.api.rate_limiting import (
            RateLimitRule,
            RateLimitStrategy,
            RedisRateLimiter,
        )

        limiter = RedisRateLimiter()
        limiter.redis_client = None  # Fail-open

        rule = RateLimitRule(limit=10, window=60, strategy=RateLimitStrategy.SLIDING_WINDOW)
        allowed, _info = await limiter.check_rate_limit("key", rule)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_routes_to_token_bucket(self):
        from src.api.rate_limiting import (
            RateLimitRule,
            RateLimitStrategy,
            RedisRateLimiter,
        )

        limiter = RedisRateLimiter()
        limiter.redis_client = None

        rule = RateLimitRule(limit=10, window=60, strategy=RateLimitStrategy.TOKEN_BUCKET)
        allowed, _info = await limiter.check_rate_limit("key", rule)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_routes_to_fixed_window(self):
        from src.api.rate_limiting import (
            RateLimitRule,
            RateLimitStrategy,
            RedisRateLimiter,
        )

        limiter = RedisRateLimiter()
        limiter.redis_client = None

        rule = RateLimitRule(limit=10, window=60, strategy=RateLimitStrategy.FIXED_WINDOW)
        allowed, _info = await limiter.check_rate_limit("key", rule)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_leaky_bucket_falls_back_to_sliding_window(self):
        from src.api.rate_limiting import (
            RateLimitRule,
            RateLimitStrategy,
            RedisRateLimiter,
        )

        limiter = RedisRateLimiter()
        limiter.redis_client = None

        rule = RateLimitRule(limit=10, window=60, strategy=RateLimitStrategy.LEAKY_BUCKET)
        allowed, _info = await limiter.check_rate_limit("key", rule)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_get_rate_limit_status_non_sliding_returns_estimate(self):
        from src.api.rate_limiting import (
            RateLimitRule,
            RateLimitStrategy,
            RedisRateLimiter,
        )

        limiter = RedisRateLimiter()
        # Mock redis client with a pipeline that works
        mock_redis = AsyncMock()
        limiter.redis_client = mock_redis

        rule = RateLimitRule(limit=100, window=60, strategy=RateLimitStrategy.TOKEN_BUCKET)
        status = await limiter.get_rate_limit_status("key", rule)
        assert status["remaining"] == rule.limit
        assert status["total"] == rule.limit

    @pytest.mark.asyncio
    async def test_get_rate_limit_status_handles_exception(self):
        from src.api.rate_limiting import (
            RateLimitRule,
            RedisRateLimiter,
        )

        limiter = RedisRateLimiter()
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = RuntimeError("redis down")
        limiter.redis_client = mock_redis

        rule = RateLimitRule(limit=100, window=60)
        status = await limiter.get_rate_limit_status("key", rule)
        assert status["remaining"] == rule.limit


class TestGetRateLimitStatusFunction:
    """Test the module-level get_rate_limit_status function."""

    @pytest.mark.asyncio
    async def test_invalid_rule_name_raises(self):
        from src.api.rate_limiting import get_rate_limit_status

        with pytest.raises(ValueError, match="Unknown rate limit rule"):
            await get_rate_limit_status("client-1", "invalid_rule")

    @pytest.mark.asyncio
    async def test_global_rule(self):
        from src.api.rate_limiting import get_rate_limit_status

        # rate_limiter has no Redis, so it returns defaults
        status = await get_rate_limit_status("client-1", "global")
        assert "remaining" in status

    @pytest.mark.asyncio
    async def test_auth_rule(self):
        from src.api.rate_limiting import get_rate_limit_status

        status = await get_rate_limit_status("client-1", "auth")
        assert "remaining" in status

    @pytest.mark.asyncio
    async def test_api_rule(self):
        from src.api.rate_limiting import get_rate_limit_status

        status = await get_rate_limit_status("client-1", "api")
        assert "remaining" in status


class TestRateLimitMiddlewareRules:
    """Test _get_applicable_rules and _build_rate_limit_key."""

    @pytest.fixture
    def middleware(self):
        from src.api.rate_limiting import RateLimitMiddleware

        async def fake_app(scope, receive, send):
            pass

        return RateLimitMiddleware(fake_app)

    def _make_request(self, path="/api/workers", user=None, ip="10.0.0.1"):
        request = MagicMock()
        request.url = MagicMock()
        request.url.path = path
        request.headers = {}
        request.client = MagicMock()
        request.client.host = ip
        request.state = MagicMock()
        if user is not None:
            request.state.user = user
        else:
            del request.state.user
        return request

    def test_get_applicable_rules_always_includes_global(self, middleware):
        request = self._make_request()
        rules = middleware._get_applicable_rules(request)
        rule_names = [name for name, _ in rules]
        assert "global" in rule_names

    def test_get_applicable_rules_auth_path(self, middleware):
        request = self._make_request(path="/auth/login")
        rules = middleware._get_applicable_rules(request)
        rule_names = [name for name, _ in rules]
        assert "auth" in rule_names

    def test_get_applicable_rules_api_for_authenticated(self, middleware):
        request = self._make_request(user={"user_id": "u123"})
        rules = middleware._get_applicable_rules(request)
        rule_names = [name for name, _ in rules]
        assert "api" in rule_names

    def test_build_rate_limit_key_anonymous(self, middleware):
        from src.api.rate_limiting import RATE_LIMIT_PREFIX

        request = self._make_request()
        rule = middleware.rules["global"]
        key = middleware._build_rate_limit_key(rule, request)
        assert key.startswith(RATE_LIMIT_PREFIX)
        assert "10.0.0.1" in key

    def test_build_rate_limit_key_authenticated(self, middleware):
        from src.api.rate_limiting import RATE_LIMIT_PREFIX

        request = self._make_request(user={"user_id": "u123"})
        rule = middleware.rules["api"]
        key = middleware._build_rate_limit_key(rule, request)
        assert key.startswith(RATE_LIMIT_PREFIX)
        assert "u123" in key


class TestRedisRateLimiterConnect:
    @pytest.mark.asyncio
    async def test_connect_failure_sets_client_none(self):
        from src.api.rate_limiting import RedisRateLimiter

        limiter = RedisRateLimiter(redis_url="redis://nonexistent:6379/0")
        # This should fail but not raise (it logs and sets redis_client=None)
        await limiter.connect()
        assert limiter.redis_client is None


# ============================================================================
# Workers models - uncovered validators
# ============================================================================


class TestWorkerModelsValidation:
    """Test model validation edge cases not covered elsewhere."""

    def test_worker_name_reserved_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="reserved"):
            WorkerCreateRequest(worker_name="admin")

    def test_temperature_below_zero_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="temperature"):
            WorkerCreateRequest(worker_name="test_worker", temperature=-0.5)

    def test_temperature_above_two_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="temperature"):
            WorkerCreateRequest(worker_name="test_worker", temperature=2.5)

    def test_worker_create_request_accepts_max_tokens(self):
        from src.api.workers.models import WorkerCreateRequest

        req = WorkerCreateRequest(worker_name="test_worker", max_tokens=6000)
        assert req.max_tokens == 6000

    def test_worker_create_request_max_tokens_defaults_to_none(self):
        from src.api.workers.models import WorkerCreateRequest

        req = WorkerCreateRequest(worker_name="test_worker")
        assert req.max_tokens is None

    def test_worker_create_request_max_tokens_rejects_out_of_range(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="max_tokens"):
            WorkerCreateRequest(worker_name="test_worker", max_tokens=0)
        with pytest.raises(ValidationError, match="max_tokens"):
            WorkerCreateRequest(worker_name="test_worker", max_tokens=-1)
        with pytest.raises(ValidationError, match="max_tokens"):
            WorkerCreateRequest(worker_name="test_worker", max_tokens=16385)

    def test_worker_create_request_max_tokens_accepts_boundaries(self):
        from src.api.workers.models import WorkerCreateRequest

        req_low = WorkerCreateRequest(worker_name="test_worker", max_tokens=1)
        req_high = WorkerCreateRequest(worker_name="test_worker", max_tokens=16384)
        assert req_low.max_tokens == 1
        assert req_high.max_tokens == 16384

    def test_tool_names_with_empty_strings_filtered(self):
        from src.api.workers.models import WorkerCreateRequest

        req = WorkerCreateRequest(worker_name="test_worker", tool_names=["", "  ", "valid_tool"])
        assert req.tool_names == ["valid_tool"]

    def test_tool_names_invalid_chars_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="tool_names"):
            WorkerCreateRequest(worker_name="test_worker", tool_names=["bad tool name!"])

    def test_mcp_config_null_byte_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="mcp_config is not allowed"):
            WorkerCreateRequest(worker_name="test_worker", mcp_config="/path\x00/evil")

    def test_base_url_override_empty_string_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="base_url_override"):
            WorkerCreateRequest(worker_name="test_worker", base_url_override="")

    def test_api_key_override_empty_string_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="api_key_override"):
            WorkerCreateRequest(worker_name="test_worker", api_key_override="")

    def test_tools_dir_override_empty_string_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="tools_dir_override"):
            WorkerCreateRequest(worker_name="test_worker", tools_dir_override="")

    def test_output_schema_missing_type_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="output_schema"):
            WorkerCreateRequest(
                worker_name="test_worker",
                output_schema={"properties": {}},
            )

    def test_output_validator_invalid_syntax_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="output_validator"):
            WorkerCreateRequest(
                worker_name="test_worker",
                output_validator="def validate( broken",
            )

    def test_output_validator_missing_validate_func_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="output_validator"):
            WorkerCreateRequest(
                worker_name="test_worker",
                output_validator="def process(ctx, output): pass",
            )

    def test_agent_type_without_agent_role_raises(self):
        """Moved off the request model: the rule must see the MERGED row.

        A write now merges only the keys a caller sent, so a body carrying
        ``agent_type`` alone may be a legitimate partial update of a row that
        already has an ``agent_role``. The rule is therefore asked of the
        effective worker (``validate_effective_worker``) rather than at parse
        time. For a create — nothing stored — it rejects the same body it
        always did.
        """
        from src.api.workers.models import validate_effective_worker

        with pytest.raises(ValueError, match="agent_role"):
            validate_effective_worker({"agent_type": "claude_code"})

    def test_agent_role_with_falsy_values_returns_none(self):
        from src.api.workers.models import WorkerCreateRequest

        req = WorkerCreateRequest(worker_name="test_worker", agent_role="0")
        assert req.agent_role is None

    def test_deps_null_byte_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="null"):
            WorkerCreateRequest(worker_name="test_worker", deps=["package\x00evil"])

    def test_deps_with_empty_strings_filtered(self):
        from src.api.workers.models import WorkerCreateRequest

        req = WorkerCreateRequest(worker_name="test_worker", deps=["", "  ", "numpy"])
        assert req.deps == ["numpy"]

    def test_mcp_servers_invalid_chars_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="mcp_servers"):
            WorkerCreateRequest(
                worker_name="test_worker",
                mcp_servers=["server/with/slashes"],
            )

    def test_mcp_servers_control_chars_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import WorkerCreateRequest

        with pytest.raises(ValidationError, match="mcp_servers"):
            WorkerCreateRequest(
                worker_name="test_worker",
                mcp_servers=["server\x01name"],
            )

    def test_message_request_empty_message_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import MessageRequest

        with pytest.raises(ValidationError, match="message"):
            MessageRequest(message="")

    def test_message_request_unsafe_context_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import MessageRequest

        # Create deeply nested structure exceeding max_depth
        deep = {"a": "b"}
        for _ in range(15):
            deep = {"nested": deep}

        with pytest.raises(ValidationError, match="context"):
            MessageRequest(message="hello", context=deep)

    def test_message_request_valid(self):
        from src.api.workers.models import MessageRequest

        req = MessageRequest(message="What is Python?")
        assert req.message == "What is Python?"
        assert req.context is None

    def test_message_request_output_schema_missing_type_raises(self):
        from pydantic import ValidationError

        from src.api.workers.models import MessageRequest

        with pytest.raises(ValidationError, match="output_schema"):
            MessageRequest(
                message="hello",
                output_schema={"properties": {}},
            )

    def test_message_response_types(self):
        from src.api.workers.models import MessageResponse

        # String output
        resp = MessageResponse(output="hello", output_type="string")
        assert resp.output == "hello"
        assert resp.output_type == "string"

        # Dict output
        resp = MessageResponse(output={"key": "val"}, output_type="object")
        assert resp.output == {"key": "val"}
        assert resp.output_type == "object"


# ============================================================================
# error_handlers: general_exception_handler in debug/dev mode
# ============================================================================


class TestGeneralExceptionHandlerDebugMode:
    def _make_request(self, headers=None):
        from fastapi import Request

        mock = MagicMock(spec=Request)
        mock.method = "GET"
        mock.url = MagicMock()
        mock.url.path = "/test"
        mock.client = MagicMock()
        mock.client.host = "127.0.0.1"
        real_headers = headers or {}
        headers_mock = MagicMock()
        headers_mock.get = real_headers.get
        mock.headers = headers_mock
        return mock

    def test_development_debug_includes_error_id(self):
        from src.api.error_handlers import general_exception_handler

        req = self._make_request()
        exc = RuntimeError("something /secret/path/here broke")

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            import logging

            logger = logging.getLogger("src.api.error_handlers")
            old_level = logger.level
            logger.setLevel(logging.DEBUG)
            try:
                resp = _run(general_exception_handler(req, exc))
            finally:
                logger.setLevel(old_level)

        import json as j

        body = j.loads(resp.body.decode())
        assert resp.status_code == 500
        # Should have sanitized path
        assert "/secret/path/here" not in body["error"]["message"]
        # Should include error ID prefix
        assert "Internal error (ID:" in body["error"]["message"]

    def test_production_mode_hides_details(self):
        from src.api.error_handlers import general_exception_handler

        req = self._make_request()
        exc = RuntimeError("password=s3cret key=abc")

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            resp = _run(general_exception_handler(req, exc))

        import json as j

        body = j.loads(resp.body.decode())
        assert "s3cret" not in body["error"]["message"]
        assert "password" not in body["error"]["message"]


# ============================================================================
# _helpers: _is_connection_error
# ============================================================================


class TestIsConnectionError:
    def test_connection_error_detected(self):
        from src.api.workers._helpers import _is_connection_error

        assert _is_connection_error(Exception("Connection error occurred")) is True

    def test_dns_error_detected(self):
        from src.api.workers._helpers import _is_connection_error

        assert _is_connection_error(Exception("Name or service not known")) is True

    def test_connection_refused_detected(self):
        from src.api.workers._helpers import _is_connection_error

        assert _is_connection_error(Exception("Connection refused")) is True

    def test_normal_error_not_detected(self):
        from src.api.workers._helpers import _is_connection_error

        assert _is_connection_error(Exception("Something else")) is False

    def test_no_route_detected(self):
        from src.api.workers._helpers import _is_connection_error

        assert _is_connection_error(Exception("No route to host")) is True

    def test_connect_call_failed_detected(self):
        from src.api.workers._helpers import _is_connection_error

        assert _is_connection_error(Exception("Connect call failed")) is True
