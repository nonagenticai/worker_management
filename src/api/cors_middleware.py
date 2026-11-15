"""CORS (Cross-Origin Resource Sharing) Middleware for Worker Management API.

This module provides secure CORS configuration with strict origin validation
and proper security headers for cross-origin requests.
"""

import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Production CORS configuration
ALLOWED_ORIGINS_PRODUCTION = [
    "https://api.example.com",
    "https://app.example.com",
    "https://dashboard.example.com",
]

# Development CORS configuration (more permissive)
ALLOWED_ORIGINS_DEVELOPMENT = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:8080",
]

# Default allowed methods
ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]

# Default allowed headers
ALLOWED_HEADERS = [
    "Accept",
    "Accept-Language",
    "Content-Language",
    "Content-Type",
    "Authorization",
    "X-API-Key",
    "X-Requested-With",
    "X-Client-ID",
    "X-Request-ID",
]

# Headers to expose to the browser
EXPOSE_HEADERS = [
    "X-Request-ID",
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
    "Content-Length",
    "Content-Range",
]


class SecureCORSMiddleware(BaseHTTPMiddleware):
    """Secure CORS middleware with strict origin validation."""

    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: list[str] | None = None,
        allowed_origin_regex: str | None = None,
        allowed_methods: list[str] | None = None,
        allowed_headers: list[str] | None = None,
        expose_headers: list[str] | None = None,
        allow_credentials: bool = True,
        allow_origin_regex: str | None = None,
        max_age: int = 600,
        is_production: bool = False,
    ) -> None:
        super().__init__(app)

        # Set default origins based on environment
        if allowed_origins is None:
            if is_production:
                allowed_origins = ALLOWED_ORIGINS_PRODUCTION
            else:
                allowed_origins = ALLOWED_ORIGINS_DEVELOPMENT

        self.allowed_origins = set(allowed_origins or [])
        self.allowed_origin_regex = allowed_origin_regex
        self.allowed_methods = set(allowed_methods or ALLOWED_METHODS)
        self.allowed_headers = set(h.lower() for h in (allowed_headers or ALLOWED_HEADERS))
        self.expose_headers = expose_headers or EXPOSE_HEADERS
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        self.is_production = is_production

        # Compile regex pattern if provided
        self.origin_regex = None
        if allowed_origin_regex:
            try:
                self.origin_regex = re.compile(allowed_origin_regex)
            except re.error:
                logger.exception("Invalid origin regex pattern")

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Handle CORS for incoming requests."""
        origin = request.headers.get("origin")

        # No origin header means this is not a CORS request; skip CORS processing
        if not origin:
            response = await call_next(request)
            return response

        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._handle_preflight(request, origin)

        # Process actual request
        response = await call_next(request)

        # Add CORS headers to response
        self._add_cors_headers(response, request, origin)

        return response

    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed based on configuration."""
        if not origin:
            return False

        # Check exact matches
        if origin in self.allowed_origins:
            return True

        # Check regex pattern
        if self.origin_regex and self.origin_regex.match(origin):
            return True

        # In development, allow localhost variations
        if not self.is_production:
            parsed = urlparse(origin)
            if parsed.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):  # nosec B104 — container requires binding to all interfaces  # noqa: S104
                return True

        return False

    def _handle_preflight(self, request: Request, origin: str) -> Response:
        """Handle CORS preflight OPTIONS requests."""
        # Check if origin is allowed
        if not self._is_origin_allowed(origin):
            logger.warning(f"CORS preflight denied for origin: {origin}")
            return Response(
                content="CORS request not allowed",
                status_code=403,
                headers={"Vary": "Origin"},
            )

        # Check requested method
        requested_method = request.headers.get("access-control-request-method")
        if requested_method and requested_method.upper() not in self.allowed_methods:
            logger.warning(f"CORS method not allowed: {requested_method}")
            return Response(content="Method not allowed", status_code=405)

        # Check requested headers
        requested_headers = request.headers.get("access-control-request-headers")
        if requested_headers:
            requested_headers_list = [h.strip().lower() for h in requested_headers.split(",")]
            if not all(h in self.allowed_headers for h in requested_headers_list):
                disallowed = [h for h in requested_headers_list if h not in self.allowed_headers]
                logger.warning(f"CORS headers not allowed: {disallowed}")
                return Response(content="Headers not allowed", status_code=403)

        # Create preflight response
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": ", ".join(sorted(self.allowed_methods)),
            "Access-Control-Allow-Headers": ", ".join(sorted(self.allowed_headers)),
            "Access-Control-Max-Age": str(self.max_age),
            "Vary": "Origin, Access-Control-Request-Method, Access-Control-Request-Headers",
        }

        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"

        if self.expose_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)

        return Response(status_code=200, headers=headers)

    def _add_cors_headers(self, response: Response, request: Request, origin: str) -> None:
        """Add CORS headers to actual response."""
        # Only add CORS headers if origin is allowed
        if not self._is_origin_allowed(origin):
            # Add Vary header to indicate origin affects response
            response.headers["Vary"] = "Origin"
            return

        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"

        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        if self.expose_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)


class CORSSecurityValidator:
    """Additional CORS security validation."""

    @staticmethod
    def validate_origin_security(origin: str) -> bool:
        """Validate origin for common security issues."""
        if not origin:
            return False

        try:
            parsed = urlparse(origin)

            # Reject non-HTTP(S) schemes
            if parsed.scheme not in ("http", "https"):
                logger.warning(f"Non-HTTP(S) origin rejected: {origin}")
                return False

            # Reject IP addresses in production (potential security risk)
            import ipaddress

            hostname = parsed.hostname
            if hostname is None:
                logger.warning(f"Origin with no hostname rejected: {origin}")
                return False

            try:
                ipaddress.ip_address(hostname)
                logger.warning(f"IP-based origin rejected: {origin}")
                return False
            except ValueError:
                # Not an IP address, which is good
                pass

            # Reject suspicious TLDs
            suspicious_tlds = [".tk", ".ml", ".ga", ".cf"]
            if any(hostname.endswith(tld) for tld in suspicious_tlds):
                logger.warning(f"Suspicious TLD origin rejected: {origin}")
                return False

            # Reject very long hostnames (potential attack)
            if len(hostname) > 253:
                logger.warning(f"Overly long hostname rejected: {origin}")
                return False

            return True

        except Exception as e:
            logger.warning(f"Origin validation error for {origin}: {e}")
            return False


def setup_cors_middleware(
    app: FastAPI, is_production: bool = False, custom_origins: list[str] | None = None
) -> None:
    """Setup CORS middleware with appropriate configuration."""
    if custom_origins:
        allowed_origins = custom_origins
    elif is_production:
        allowed_origins = ALLOWED_ORIGINS_PRODUCTION
    else:
        allowed_origins = ALLOWED_ORIGINS_DEVELOPMENT

    # Add secure CORS middleware
    app.add_middleware(
        SecureCORSMiddleware,
        allowed_origins=allowed_origins,
        allowed_methods=ALLOWED_METHODS,
        allowed_headers=ALLOWED_HEADERS,
        expose_headers=EXPOSE_HEADERS,
        allow_credentials=True,
        max_age=600,  # 10 minutes
        is_production=is_production,
    )

    logger.info(
        f"CORS middleware configured for {'production' if is_production else 'development'}"
    )
    logger.info(f"Allowed origins: {allowed_origins}")


# For backwards compatibility with FastAPI's built-in CORS
def get_cors_middleware_config(is_production: bool = False) -> dict:
    """Get CORS middleware configuration for FastAPI's CORSMiddleware."""
    origins = ALLOWED_ORIGINS_PRODUCTION if is_production else ALLOWED_ORIGINS_DEVELOPMENT

    return {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": ALLOWED_METHODS,
        "allow_headers": ALLOWED_HEADERS,
        "expose_headers": EXPOSE_HEADERS,
        "max_age": 600,
    }
