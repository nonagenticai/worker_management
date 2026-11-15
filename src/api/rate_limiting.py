"""Distributed Rate Limiting with Redis for Worker Management API.

This module provides advanced rate limiting capabilities using Redis as the backend
for distributed rate limiting across multiple API instances.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import redis.asyncio as redis
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request, Response
    from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Rate limiting configuration
REDIS_URL = "redis://localhost:6379/0"  # Should be from environment
RATE_LIMIT_PREFIX = "rate_limit:"
DEFAULT_RATE_LIMIT = 100  # requests per minute
BURST_LIMIT = 200  # burst capacity
RATE_LIMIT_WINDOW = 60  # seconds


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"  # nosec B105 -- enum strategy name, not a credential  # noqa: S105
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""

    limit: int  # Number of requests
    window: int  # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_limit: int | None = None
    key_pattern: str = "ip:{ip}"  # Key pattern for rate limiting


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int) -> None:
        self.message = message
        self.retry_after = retry_after
        super().__init__(message)


class RedisRateLimiter:
    """Redis-based distributed rate limiter."""

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self.redis_url = redis_url
        self.redis_client: redis.Redis | None = None
        self._connection_pool: redis.ConnectionPool | None = None

    async def connect(self) -> None:
        """Initialize Redis connection."""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                decode_responses=True,
            )
            self.redis_client = redis.Redis(connection_pool=self._connection_pool)

            # Test connection
            await self.redis_client.ping()
            logger.info("Connected to Redis for rate limiting")

        except Exception:
            logger.exception("Failed to connect to Redis")
            # Fallback to in-memory rate limiting
            self.redis_client = None

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()

    async def check_rate_limit(
        self,
        key: str,
        rule: RateLimitRule,
        client_id: str | None = None,  # noqa: ARG002, RUF100
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request is within rate limits.

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        if not self.redis_client:
            # Fallback to allow all requests if Redis is unavailable
            logger.warning("Redis unavailable, allowing request")
            return True, {
                "remaining": rule.limit,
                "reset_time": time.time() + rule.window,
            }

        try:
            if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._sliding_window_check(key, rule)
            elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._token_bucket_check(key, rule)
            elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._fixed_window_check(key, rule)
            else:
                # Default to sliding window
                return await self._sliding_window_check(key, rule)

        except Exception:
            logger.exception("Rate limiting error")
            # Fail open - allow request if rate limiting fails
            return True, {
                "remaining": rule.limit,
                "reset_time": time.time() + rule.window,
            }

    async def _sliding_window_check(
        self, key: str, rule: RateLimitRule
    ) -> tuple[bool, dict[str, Any]]:
        """Sliding window rate limiting implementation."""
        if self.redis_client is None:
            return True, {"remaining": 0, "reset_time": 0, "total": 0, "window": 0}
        now = time.time()
        window_start = now - rule.window

        pipe = self.redis_client.pipeline()

        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests in window
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(now): now})

        # Set expiration
        pipe.expire(key, rule.window + 1)

        results = await pipe.execute()
        current_count = results[1] + 1  # +1 for the current request we just added

        if current_count <= rule.limit:
            remaining = rule.limit - current_count
            reset_time = now + rule.window
            return True, {
                "remaining": remaining,
                "reset_time": reset_time,
                "total": rule.limit,
                "window": rule.window,
            }
        else:
            # Remove the request we just added since it exceeded the limit
            await self.redis_client.zrem(key, str(now))

            # Calculate retry after time
            oldest_request = await self.redis_client.zrange(key, 0, 0, withscores=True)
            if oldest_request:
                retry_after = int(float(oldest_request[0][1]) + rule.window - now)
            else:
                retry_after = rule.window

            return False, {
                "remaining": 0,
                "reset_time": now + retry_after,
                "total": rule.limit,
                "window": rule.window,
                "retry_after": retry_after,
            }

    async def _token_bucket_check(
        self, key: str, rule: RateLimitRule
    ) -> tuple[bool, dict[str, Any]]:
        """Token bucket rate limiting implementation."""
        if self.redis_client is None:
            return True, {"remaining": 0, "reset_time": 0, "total": 0, "window": 0}
        now = time.time()
        bucket_key = f"{key}:bucket"

        # Lua script for atomic token bucket operation
        lua_script = """
        local bucket_key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local burst_limit = tonumber(ARGV[4]) or limit

        local bucket = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or burst_limit
        local last_refill = tonumber(bucket[2]) or now

        -- Calculate tokens to add based on time elapsed
        local elapsed = now - last_refill
        local tokens_to_add = math.floor(elapsed * limit / window)
        tokens = math.min(burst_limit, tokens + tokens_to_add)

        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', bucket_key, window * 2)
            return {1, tokens, now + (burst_limit - tokens) * window / limit}
        else
            redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', bucket_key, window * 2)
            return {0, tokens, now + (1 - tokens) * window / limit}
        end
        """

        burst_limit = rule.burst_limit or rule.limit
        result = await self.redis_client.eval(
            lua_script, 1, bucket_key, rule.limit, rule.window, now, burst_limit
        )

        allowed = bool(result[0])
        tokens_remaining = int(result[1])
        reset_time = float(result[2])

        if allowed:
            return True, {
                "remaining": tokens_remaining,
                "reset_time": reset_time,
                "total": rule.limit,
                "window": rule.window,
            }
        else:
            retry_after = int(reset_time - now)
            return False, {
                "remaining": 0,
                "reset_time": reset_time,
                "total": rule.limit,
                "window": rule.window,
                "retry_after": retry_after,
            }

    async def _fixed_window_check(
        self, key: str, rule: RateLimitRule
    ) -> tuple[bool, dict[str, Any]]:
        """Fixed window rate limiting implementation."""
        if self.redis_client is None:
            return True, {"remaining": 0, "reset_time": 0, "total": 0, "window": 0}
        now = time.time()
        window_key = f"{key}:{int(now // rule.window)}"

        pipe = self.redis_client.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, rule.window)
        results = await pipe.execute()

        current_count = results[0]

        if current_count <= rule.limit:
            remaining = rule.limit - current_count
            reset_time = (int(now // rule.window) + 1) * rule.window
            return True, {
                "remaining": remaining,
                "reset_time": reset_time,
                "total": rule.limit,
                "window": rule.window,
            }
        else:
            reset_time = (int(now // rule.window) + 1) * rule.window
            retry_after = int(reset_time - now)
            return False, {
                "remaining": 0,
                "reset_time": reset_time,
                "total": rule.limit,
                "window": rule.window,
                "retry_after": retry_after,
            }

    async def get_rate_limit_status(self, key: str, rule: RateLimitRule) -> dict[str, Any]:
        """Get current rate limit status without consuming a token."""
        if not self.redis_client:
            return {"remaining": rule.limit, "reset_time": time.time() + rule.window}

        try:
            if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                now = time.time()
                window_start = now - rule.window

                # Remove expired entries and count
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)
                results = await pipe.execute()

                current_count = results[1]
                remaining = max(0, rule.limit - current_count)
                reset_time = now + rule.window

                return {
                    "remaining": remaining,
                    "reset_time": reset_time,
                    "total": rule.limit,
                    "window": rule.window,
                }
            else:
                # For other strategies, return estimated values
                return {
                    "remaining": rule.limit,
                    "reset_time": time.time() + rule.window,
                    "total": rule.limit,
                    "window": rule.window,
                }

        except Exception:
            logger.exception("Error getting rate limit status")
            return {"remaining": rule.limit, "reset_time": time.time() + rule.window}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Distributed rate limiting middleware."""

    def __init__(self, app: ASGIApp, redis_url: str = REDIS_URL) -> None:
        super().__init__(app)
        self.rate_limiter = RedisRateLimiter(redis_url)
        self.rules: dict[str, RateLimitRule] = {}
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default rate limiting rules."""
        # Global rate limit per IP - increased for testing
        self.rules["global"] = RateLimitRule(
            limit=500,  # 500 requests per minute (increased for testing)
            window=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            burst_limit=750,  # Higher burst for testing scenarios
            key_pattern="global:{ip}",
        )

        # Authentication endpoints - more lenient for testing
        self.rules["auth"] = RateLimitRule(
            limit=50,  # 50 requests per minute (increased from 10)
            window=60,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_limit=75,
            key_pattern="auth:{ip}",
        )

        # API endpoints - per user limits - increased
        self.rules["api"] = RateLimitRule(
            limit=1000,  # 1000 requests per minute (increased)
            window=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            burst_limit=1500,  # Higher burst for testing
            key_pattern="api:{user_id}",
        )

    async def startup(self) -> None:
        """Initialize rate limiter on startup."""
        await self.rate_limiter.connect()

    async def shutdown(self) -> None:
        """Cleanup rate limiter on shutdown."""
        await self.rate_limiter.disconnect()

    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from authenticated request
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user['user_id']}"

        # Fall back to IP address
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check X-Forwarded-For header (from proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return str(forwarded_for.split(",")[0].strip())

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return str(real_ip)

        # Fall back to direct client IP
        if request.client:
            return str(request.client.host)

        return "unknown"

    def _get_applicable_rules(self, request: Request) -> list[tuple[str, RateLimitRule]]:
        """Get applicable rate limiting rules for request."""
        path = request.url.path
        rules_to_apply = []

        # Always apply global rate limit
        rules_to_apply.append(("global", self.rules["global"]))

        # Authentication endpoints
        if path.startswith("/auth/"):
            rules_to_apply.append(("auth", self.rules["auth"]))

        # API endpoints (for authenticated users)
        elif hasattr(request.state, "user") and request.state.user:
            rules_to_apply.append(("api", self.rules["api"]))

        return rules_to_apply

    def _build_rate_limit_key(self, rule: RateLimitRule, request: Request) -> str:
        """Build rate limiting key from rule pattern."""
        client_id = self._get_client_identifier(request)
        client_ip = self._get_client_ip(request)

        user_id = "anonymous"
        if hasattr(request.state, "user") and request.state.user:
            user_id = request.state.user["user_id"]

        key = rule.key_pattern.format(ip=client_ip, user_id=user_id, client_id=client_id)

        return f"{RATE_LIMIT_PREFIX}{key}"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Apply rate limiting to incoming requests."""
        # Skip rate limiting for health check endpoints
        path = request.url.path
        if path in ("/health", "/healthz", "/ready", "/readyz", "/metrics"):
            return await call_next(request)

        # Get applicable rules
        rules = self._get_applicable_rules(request)

        rate_limit_info: dict[str, dict[str, Any]] = {}

        for rule_name, rule in rules:
            key = self._build_rate_limit_key(rule, request)
            allowed, info = await self.rate_limiter.check_rate_limit(key, rule)
            rate_limit_info[rule_name] = info

            if not allowed:
                retry_after = info.get("retry_after", rule.window)
                return self._rate_limit_response(retry_after, info)

        # All checks passed - proceed with request
        response = await call_next(request)

        # Add rate limit headers to response
        self._add_rate_limit_headers(response, rate_limit_info)

        return response

    def _rate_limit_response(self, retry_after: int, info: dict[str, Any]) -> Response:
        """Create a 429 Too Many Requests response."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please retry after %d seconds." % retry_after,
                    "status": 429,
                    "retry_after": retry_after,
                }
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(info.get("total", 0)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(info.get("reset_time", 0))),
            },
        )

    def _add_rate_limit_headers(
        self, response: Response, rate_limit_info: dict[str, dict[str, Any]]
    ) -> None:
        """Add rate limiting headers to response."""
        # Use the most restrictive rate limit for headers
        if not rate_limit_info:
            return

        # Find the rule with the lowest remaining count
        most_restrictive = min(
            rate_limit_info.values(), key=lambda x: x.get("remaining", float("inf"))
        )

        response.headers["X-RateLimit-Limit"] = str(most_restrictive.get("total", 0))
        response.headers["X-RateLimit-Remaining"] = str(most_restrictive.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(
            int(most_restrictive.get("reset_time", time.time()))
        )


# Global rate limiter instance
rate_limiter = RedisRateLimiter()


async def get_rate_limit_status(client_id: str, rule_name: str = "global") -> dict[str, Any]:
    """Get current rate limit status for a client."""
    if rule_name not in ["global", "auth", "api"]:
        raise ValueError(f"Unknown rate limit rule: {rule_name}")

    # Define rules matching the middleware defaults
    rules = {
        "global": RateLimitRule(limit=500, window=60, strategy=RateLimitStrategy.SLIDING_WINDOW),
        "auth": RateLimitRule(limit=50, window=60, strategy=RateLimitStrategy.TOKEN_BUCKET),
        "api": RateLimitRule(limit=1000, window=60, strategy=RateLimitStrategy.SLIDING_WINDOW),
    }

    rule = rules[rule_name]
    key = f"{RATE_LIMIT_PREFIX}{rule_name}:{client_id}"

    return await rate_limiter.get_rate_limit_status(key, rule)
