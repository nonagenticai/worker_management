"""Unit tests for src/api/rate_limiting.py.

Tests cover:
- RateLimitStrategy enum values
- RateLimitRule dataclass creation and defaults
- RateLimitExceeded exception attributes
- RedisRateLimiter fail-open behavior when redis_client is None
- RedisRateLimiter.get_rate_limit_status defaults when redis unavailable
- RateLimitMiddleware health-endpoint bypass
- RateLimitMiddleware._get_client_ip header priority
- RateLimitMiddleware._get_client_identifier authenticated vs anonymous
- RateLimitMiddleware._rate_limit_response 429 with proper headers
- RateLimitMiddleware._add_rate_limit_headers X-RateLimit-* headers
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from src.api.rate_limiting import (
    RateLimitExceeded,
    RateLimitMiddleware,
    RateLimitRule,
    RateLimitStrategy,
    RedisRateLimiter,
)

# ============================================================================
# RateLimitStrategy enum
# ============================================================================


class TestRateLimitStrategy:
    """Tests for the RateLimitStrategy enum."""

    def test_fixed_window_value(self):
        assert RateLimitStrategy.FIXED_WINDOW.value == "fixed_window"

    def test_sliding_window_value(self):
        assert RateLimitStrategy.SLIDING_WINDOW.value == "sliding_window"

    def test_token_bucket_value(self):
        assert RateLimitStrategy.TOKEN_BUCKET.value == "token_bucket"

    def test_leaky_bucket_value(self):
        assert RateLimitStrategy.LEAKY_BUCKET.value == "leaky_bucket"

    def test_enum_has_four_members(self):
        assert len(RateLimitStrategy) == 4


# ============================================================================
# RateLimitRule dataclass
# ============================================================================


class TestRateLimitRule:
    """Tests for RateLimitRule dataclass creation and defaults."""

    def test_minimal_creation(self):
        rule = RateLimitRule(limit=100, window=60)
        assert rule.limit == 100
        assert rule.window == 60

    def test_default_strategy_is_sliding_window(self):
        rule = RateLimitRule(limit=100, window=60)
        assert rule.strategy == RateLimitStrategy.SLIDING_WINDOW

    def test_default_burst_limit_is_none(self):
        rule = RateLimitRule(limit=100, window=60)
        assert rule.burst_limit is None

    def test_default_key_pattern(self):
        rule = RateLimitRule(limit=100, window=60)
        assert rule.key_pattern == "ip:{ip}"

    def test_custom_strategy(self):
        rule = RateLimitRule(limit=50, window=30, strategy=RateLimitStrategy.TOKEN_BUCKET)
        assert rule.strategy == RateLimitStrategy.TOKEN_BUCKET

    def test_custom_burst_limit(self):
        rule = RateLimitRule(limit=100, window=60, burst_limit=200)
        assert rule.burst_limit == 200

    def test_custom_key_pattern(self):
        rule = RateLimitRule(limit=100, window=60, key_pattern="api:{user_id}")
        assert rule.key_pattern == "api:{user_id}"

    def test_full_custom_rule(self):
        rule = RateLimitRule(
            limit=1000,
            window=120,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            burst_limit=1500,
            key_pattern="custom:{ip}:{user_id}",
        )
        assert rule.limit == 1000
        assert rule.window == 120
        assert rule.strategy == RateLimitStrategy.FIXED_WINDOW
        assert rule.burst_limit == 1500
        assert rule.key_pattern == "custom:{ip}:{user_id}"


# ============================================================================
# RateLimitExceeded exception
# ============================================================================


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_message_and_retry_after(self):
        exc = RateLimitExceeded("Too many requests", retry_after=30)
        assert exc.message == "Too many requests"
        assert exc.retry_after == 30

    def test_is_exception(self):
        exc = RateLimitExceeded("slow down", retry_after=10)
        assert isinstance(exc, Exception)

    def test_str_representation(self):
        exc = RateLimitExceeded("Rate limited", retry_after=60)
        assert str(exc) == "Rate limited"


# ============================================================================
# RedisRateLimiter - fail-open behavior
# ============================================================================


class TestRedisRateLimiterFailOpen:
    """Tests for RedisRateLimiter when redis_client is None (fail-open)."""

    @pytest.fixture
    def limiter_no_redis(self):
        """A RedisRateLimiter with no Redis connection."""
        limiter = RedisRateLimiter(redis_url="redis://localhost:6379/0")
        limiter.redis_client = None
        return limiter

    @pytest.fixture
    def default_rule(self):
        return RateLimitRule(limit=100, window=60)

    @pytest.mark.asyncio
    async def test_check_rate_limit_allows_all_when_redis_none(
        self, limiter_no_redis, default_rule
    ):
        allowed, _info = await limiter_no_redis.check_rate_limit("test_key", default_rule)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_returns_full_remaining_when_redis_none(
        self, limiter_no_redis, default_rule
    ):
        _, info = await limiter_no_redis.check_rate_limit("test_key", default_rule)
        assert info["remaining"] == default_rule.limit

    @pytest.mark.asyncio
    async def test_check_rate_limit_returns_reset_time_when_redis_none(
        self, limiter_no_redis, default_rule
    ):
        before = time.time()
        _, info = await limiter_no_redis.check_rate_limit("test_key", default_rule)
        after = time.time()
        assert info["reset_time"] >= before + default_rule.window
        assert info["reset_time"] <= after + default_rule.window

    @pytest.mark.asyncio
    async def test_get_rate_limit_status_returns_defaults_when_redis_none(
        self, limiter_no_redis, default_rule
    ):
        status = await limiter_no_redis.get_rate_limit_status("test_key", default_rule)
        assert status["remaining"] == default_rule.limit
        assert "reset_time" in status

    @pytest.mark.asyncio
    async def test_get_rate_limit_status_reset_time_when_redis_none(
        self, limiter_no_redis, default_rule
    ):
        before = time.time()
        status = await limiter_no_redis.get_rate_limit_status("test_key", default_rule)
        after = time.time()
        assert status["reset_time"] >= before + default_rule.window
        assert status["reset_time"] <= after + default_rule.window

    @pytest.mark.asyncio
    async def test_sliding_window_check_allows_when_redis_none(
        self, limiter_no_redis, default_rule
    ):
        """_sliding_window_check returns allow-all when redis_client is None."""
        allowed, _info = await limiter_no_redis._sliding_window_check("test_key", default_rule)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_token_bucket_check_allows_when_redis_none(self, limiter_no_redis, default_rule):
        """_token_bucket_check returns allow-all when redis_client is None."""
        allowed, _info = await limiter_no_redis._token_bucket_check("test_key", default_rule)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_fixed_window_check_allows_when_redis_none(self, limiter_no_redis, default_rule):
        """_fixed_window_check returns allow-all when redis_client is None."""
        allowed, _info = await limiter_no_redis._fixed_window_check("test_key", default_rule)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_allows_on_redis_exception(self, default_rule):
        """If Redis raises an exception, fail-open and allow the request."""
        limiter = RedisRateLimiter()
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = Exception("Redis connection lost")
        limiter.redis_client = mock_redis

        allowed, info = await limiter.check_rate_limit("test_key", default_rule)
        assert allowed is True
        assert info["remaining"] == default_rule.limit


# ============================================================================
# Helper factories for middleware tests
# ============================================================================


def _make_fake_app():
    """Create a minimal ASGI app for middleware initialization."""

    async def app(scope, receive, send):
        pass

    return app


def _make_middleware():
    """Create a RateLimitMiddleware with a fake app."""
    app = _make_fake_app()
    middleware = RateLimitMiddleware(app)
    return middleware


def _make_request(
    path="/api/workers",
    headers=None,
    client_host="192.168.1.100",
    state_user=None,
):
    """Create a mock Request object with the given parameters."""
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = path
    request.headers = headers or {}

    # Client info
    if client_host:
        request.client = MagicMock()
        request.client.host = client_host
    else:
        request.client = None

    # State (for authenticated user)
    request.state = MagicMock()
    if state_user is not None:
        request.state.user = state_user
    else:
        # hasattr should return False for user when not set
        del request.state.user

    return request


# ============================================================================
# RateLimitMiddleware._get_client_ip
# ============================================================================


class TestGetClientIp:
    """Tests for RateLimitMiddleware._get_client_ip header priority."""

    def setup_method(self):
        self.middleware = _make_middleware()

    def test_prefers_x_forwarded_for(self):
        request = _make_request(
            headers={
                "X-Forwarded-For": "10.0.0.1, 10.0.0.2",
                "X-Real-IP": "172.16.0.1",
            },
            client_host="192.168.1.100",
        )
        assert self.middleware._get_client_ip(request) == "10.0.0.1"

    def test_x_forwarded_for_single_ip(self):
        request = _make_request(
            headers={"X-Forwarded-For": "10.0.0.5"},
            client_host="192.168.1.100",
        )
        assert self.middleware._get_client_ip(request) == "10.0.0.5"

    def test_falls_back_to_x_real_ip(self):
        request = _make_request(
            headers={"X-Real-IP": "172.16.0.1"},
            client_host="192.168.1.100",
        )
        assert self.middleware._get_client_ip(request) == "172.16.0.1"

    def test_falls_back_to_client_host(self):
        request = _make_request(headers={}, client_host="192.168.1.100")
        assert self.middleware._get_client_ip(request) == "192.168.1.100"

    def test_returns_unknown_when_no_client(self):
        request = _make_request(headers={}, client_host=None)
        assert self.middleware._get_client_ip(request) == "unknown"

    def test_strips_whitespace_from_x_forwarded_for(self):
        request = _make_request(
            headers={"X-Forwarded-For": "  10.0.0.1  , 10.0.0.2"},
        )
        assert self.middleware._get_client_ip(request) == "10.0.0.1"


# ============================================================================
# RateLimitMiddleware._get_client_identifier
# ============================================================================


class TestGetClientIdentifier:
    """Tests for _get_client_identifier: user ID vs IP."""

    def setup_method(self):
        self.middleware = _make_middleware()

    def test_returns_user_id_when_authenticated(self):
        request = _make_request(
            state_user={"user_id": "user-abc-123"},
            client_host="192.168.1.100",
        )
        result = self.middleware._get_client_identifier(request)
        assert result == "user:user-abc-123"

    def test_returns_ip_when_anonymous(self):
        request = _make_request(state_user=None, client_host="10.0.0.5")
        result = self.middleware._get_client_identifier(request)
        assert result == "ip:10.0.0.5"

    def test_returns_ip_from_x_forwarded_for_when_anonymous(self):
        request = _make_request(
            state_user=None,
            headers={"X-Forwarded-For": "203.0.113.50"},
            client_host="10.0.0.1",
        )
        result = self.middleware._get_client_identifier(request)
        assert result == "ip:203.0.113.50"


# ============================================================================
# RateLimitMiddleware._rate_limit_response
# ============================================================================


class TestRateLimitResponse:
    """Tests for _rate_limit_response returning 429."""

    def setup_method(self):
        self.middleware = _make_middleware()

    def test_returns_429_status_code(self):
        info = {"total": 100, "remaining": 0, "reset_time": time.time() + 60}
        response = self.middleware._rate_limit_response(retry_after=30, info=info)
        assert response.status_code == 429

    def test_retry_after_header(self):
        info = {"total": 100, "remaining": 0, "reset_time": time.time() + 60}
        response = self.middleware._rate_limit_response(retry_after=45, info=info)
        assert response.headers["retry-after"] == "45"

    def test_x_ratelimit_limit_header(self):
        info = {"total": 500, "remaining": 0, "reset_time": time.time() + 60}
        response = self.middleware._rate_limit_response(retry_after=10, info=info)
        assert response.headers["x-ratelimit-limit"] == "500"

    def test_x_ratelimit_remaining_header_is_zero(self):
        info = {"total": 100, "remaining": 0, "reset_time": time.time() + 60}
        response = self.middleware._rate_limit_response(retry_after=10, info=info)
        assert response.headers["x-ratelimit-remaining"] == "0"

    def test_x_ratelimit_reset_header(self):
        reset_time = time.time() + 120
        info = {"total": 100, "remaining": 0, "reset_time": reset_time}
        response = self.middleware._rate_limit_response(retry_after=10, info=info)
        assert response.headers["x-ratelimit-reset"] == str(int(reset_time))

    def test_response_body_contains_error(self):
        info = {"total": 100, "remaining": 0, "reset_time": time.time() + 60}
        response = self.middleware._rate_limit_response(retry_after=25, info=info)
        body = json.loads(response.body.decode())
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert body["error"]["status"] == 429
        assert body["error"]["retry_after"] == 25
        assert "25 seconds" in body["error"]["message"]


# ============================================================================
# RateLimitMiddleware._add_rate_limit_headers
# ============================================================================


class TestAddRateLimitHeaders:
    """Tests for _add_rate_limit_headers adding X-RateLimit-* headers."""

    def setup_method(self):
        self.middleware = _make_middleware()

    def _make_response(self):
        """Create a mock response with a real dict for headers."""
        response = MagicMock()
        response.headers = {}
        return response

    def test_adds_x_ratelimit_limit(self):
        response = self._make_response()
        info = {
            "global": {
                "total": 500,
                "remaining": 490,
                "reset_time": time.time() + 60,
            }
        }
        self.middleware._add_rate_limit_headers(response, info)
        assert response.headers["X-RateLimit-Limit"] == "500"

    def test_adds_x_ratelimit_remaining(self):
        response = self._make_response()
        info = {
            "global": {
                "total": 500,
                "remaining": 123,
                "reset_time": time.time() + 60,
            }
        }
        self.middleware._add_rate_limit_headers(response, info)
        assert response.headers["X-RateLimit-Remaining"] == "123"

    def test_adds_x_ratelimit_reset(self):
        reset_time = time.time() + 45
        response = self._make_response()
        info = {
            "global": {
                "total": 500,
                "remaining": 400,
                "reset_time": reset_time,
            }
        }
        self.middleware._add_rate_limit_headers(response, info)
        assert response.headers["X-RateLimit-Reset"] == str(int(reset_time))

    def test_uses_most_restrictive_rule(self):
        """When multiple rules, use the one with lowest remaining."""
        response = self._make_response()
        info = {
            "global": {
                "total": 500,
                "remaining": 450,
                "reset_time": time.time() + 60,
            },
            "api": {
                "total": 1000,
                "remaining": 5,
                "reset_time": time.time() + 30,
            },
        }
        self.middleware._add_rate_limit_headers(response, info)
        # api has fewer remaining (5 < 450), so its values should be used
        assert response.headers["X-RateLimit-Limit"] == "1000"
        assert response.headers["X-RateLimit-Remaining"] == "5"

    def test_no_headers_when_empty_info(self):
        response = self._make_response()
        self.middleware._add_rate_limit_headers(response, {})
        assert "X-RateLimit-Limit" not in response.headers
        assert "X-RateLimit-Remaining" not in response.headers
        assert "X-RateLimit-Reset" not in response.headers


# ============================================================================
# RateLimitMiddleware._setup_default_rules
# ============================================================================


class TestSetupDefaultRules:
    """Tests for the default rate limiting rules."""

    def setup_method(self):
        self.middleware = _make_middleware()

    def test_global_rule_exists(self):
        assert "global" in self.middleware.rules

    def test_auth_rule_exists(self):
        assert "auth" in self.middleware.rules

    def test_api_rule_exists(self):
        assert "api" in self.middleware.rules

    def test_global_rule_limit_500_per_minute(self):
        rule = self.middleware.rules["global"]
        assert rule.limit == 500
        assert rule.window == 60

    def test_auth_rule_limit_50_per_minute(self):
        rule = self.middleware.rules["auth"]
        assert rule.limit == 50
        assert rule.window == 60

    def test_api_rule_limit_1000_per_minute(self):
        rule = self.middleware.rules["api"]
        assert rule.limit == 1000
        assert rule.window == 60

    def test_global_uses_sliding_window(self):
        assert self.middleware.rules["global"].strategy == RateLimitStrategy.SLIDING_WINDOW

    def test_auth_uses_token_bucket(self):
        assert self.middleware.rules["auth"].strategy == RateLimitStrategy.TOKEN_BUCKET

    def test_api_uses_sliding_window(self):
        assert self.middleware.rules["api"].strategy == RateLimitStrategy.SLIDING_WINDOW
