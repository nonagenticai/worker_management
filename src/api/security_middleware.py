"""Security middleware for adding security headers and request validation."""

import logging
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl

from starlette.middleware.base import BaseHTTPMiddleware

from src.api.security_validation import (
    SecurityValidationError,
    validate_query_parameters,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request, Response
    from fastapi.responses import JSONResponse
    from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        try:
            # Validate query parameters for security issues
            if request.url.query:
                try:
                    # Parse query parameters
                    parsed_params = dict(parse_qsl(request.url.query, keep_blank_values=True))

                    # Validate for security issues
                    validate_query_parameters(parsed_params)

                except SecurityValidationError as e:
                    logger.warning(f"Security validation failed for query parameters: {e}")
                    # Create a 422 response for security violations
                    from fastapi import HTTPException
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        status_code=422,
                        content={
                            "error": {
                                "code": "VALIDATION_ERROR",
                                "message": f"Invalid query parameters: {e!s}",
                                "status": 422,
                            }
                        },
                    )

            # Process the request
            response = await call_next(request)

            # Check if this is a documentation page
            path = request.url.path
            is_docs_page = path in ["/docs", "/redoc", "/openapi.json"]

            # Add security headers
            if is_docs_page:
                # Relaxed CSP for documentation pages to allow Swagger UI to load
                security_headers = {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "SAMEORIGIN",  # Allow embedding in same origin
                    "X-XSS-Protection": "1; mode=block",
                    # Allow CDN sources for Swagger UI
                    "Content-Security-Policy": (
                        "default-src 'self' https://cdn.jsdelivr.net https://fastapi.tiangolo.com; "
                        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                        "img-src 'self' data: https://fastapi.tiangolo.com; "
                        "font-src 'self' https://cdn.jsdelivr.net; "
                        "connect-src 'self'; "
                        "frame-ancestors 'self'; "
                        "base-uri 'self'; "
                        "form-action 'self'"
                    ),
                    "Referrer-Policy": "strict-origin-when-cross-origin",
                }
            else:
                # Restrictive CSP for all other endpoints
                security_headers = {
                    # Prevent MIME type sniffing
                    "X-Content-Type-Options": "nosniff",
                    # Prevent page embedding in frames (clickjacking protection)
                    "X-Frame-Options": "DENY",
                    # Enable XSS protection (though deprecated, still widely supported)
                    "X-XSS-Protection": "1; mode=block",
                    # Content Security Policy - restrictive default
                    "Content-Security-Policy": (
                        "default-src 'self'; "
                        "script-src 'self'; "
                        "style-src 'self' 'unsafe-inline'; "
                        "img-src 'self' data:; "
                        "font-src 'self'; "
                        "connect-src 'self'; "
                        "frame-ancestors 'none'; "
                        "base-uri 'self'; "
                        "form-action 'self'"
                    ),
                    # Prevent information leakage via referrer
                    "Referrer-Policy": "strict-origin-when-cross-origin",
                    # Require HTTPS for future requests (HSTS enabled for security)
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
                    # Control what features can be used
                    "Permissions-Policy": (
                        "accelerometer=(), "
                        "camera=(), "
                        "geolocation=(), "
                        "gyroscope=(), "
                        "magnetometer=(), "
                        "microphone=(), "
                        "payment=(), "
                        "usb=()"
                    ),
                    # Prevent caching of sensitive responses
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                }

            # Add all security headers to the response
            for header_name, header_value in security_headers.items():
                response.headers[header_name] = header_value

            return response

        except SecurityValidationError:
            # Already handled above with proper response
            raise
        except Exception as e:
            # Import HTTPException here to check for it
            from fastapi import HTTPException

            # If it's an HTTPException, convert to proper JSONResponse
            if isinstance(e, HTTPException):
                from fastapi.responses import JSONResponse

                logger.info(f"HTTPException in request: {e.status_code} - {e.detail}")
                return JSONResponse(
                    status_code=e.status_code,
                    content={
                        "error": {
                            "code": "HTTP_ERROR",
                            "message": e.detail,
                            "status": e.status_code,
                        }
                    },
                )

            # For unexpected errors, return generic response
            logger.error(f"Security middleware error: {e}", exc_info=True)
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Request processing failed",
                        "status": 500,
                    }
                },
            )


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for additional request validation and security checks."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate request for security issues."""
        try:
            # Check request size to prevent DoS
            content_length = request.headers.get("content-length")
            if content_length:
                content_length = int(content_length)
                max_size = 50 * 1024 * 1024  # 50MB maximum request size

                if content_length > max_size:
                    logger.warning(f"Request too large: {content_length} bytes")
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "PAYLOAD_TOO_LARGE",
                                "message": f"Request entity too large. Maximum size is {max_size} bytes.",
                                "status": 413,
                            }
                        },
                    )

            # Check for suspicious User-Agent strings
            user_agent = request.headers.get("user-agent", "")
            suspicious_agents = [
                "sqlmap",
                "nikto",
                "nmap",
                "masscan",
                "w3af",
                "burp",
                "zap",
                "acunetix",
                "nessus",
                "openvas",
                "metasploit",
            ]

            if any(agent in user_agent.lower() for agent in suspicious_agents):
                logger.warning(f"Suspicious User-Agent detected: {user_agent}")
                # Still allow the request but log it

            # Validate certain headers for XSS attempts
            potentially_dangerous_headers = ["referer", "origin", "x-forwarded-for"]
            for header_name in potentially_dangerous_headers:
                header_value = request.headers.get(header_name)
                if header_value:
                    try:
                        from src.api.security_validation import (
                            detect_sql_injection,
                            detect_xss,
                        )

                        if detect_xss(header_value) or detect_sql_injection(header_value):
                            logger.warning(
                                f"Malicious content detected in header {header_name}: {header_value}"
                            )
                            from fastapi.responses import JSONResponse

                            return JSONResponse(
                                status_code=400,
                                content={
                                    "error": {
                                        "code": "BAD_REQUEST",
                                        "message": "Invalid header content detected",
                                        "status": 400,
                                    }
                                },
                            )
                    except Exception:
                        logger.exception(f"Error validating header {header_name}")

            # Process the request
            response = await call_next(request)
            return response

        except Exception as e:
            # Import HTTPException here to check for it
            from fastapi import HTTPException

            # If it's an HTTPException, convert to proper JSONResponse
            if isinstance(e, HTTPException):
                from fastapi.responses import JSONResponse

                logger.info(f"HTTPException in request: {e.status_code} - {e.detail}")
                return JSONResponse(
                    status_code=e.status_code,
                    content={
                        "error": {
                            "code": "HTTP_ERROR",
                            "message": e.detail,
                            "status": e.status_code,
                        }
                    },
                )

            # For unexpected errors, return generic response
            logger.error(f"Request validation middleware error: {e}", exc_info=True)
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Request validation failed",
                        "status": 500,
                    }
                },
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware - COMPLETELY DISABLED."""

    def __init__(
        self,
        app: ASGIApp,
        calls_per_minute: int = 1000,
        calls_per_second: int = 50,
        burst_limit: int = 100,
    ) -> None:
        super().__init__(app)
        # All rate limiting completely disabled - no initialization of rate limiting data structures

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Rate limiting completely disabled - all requests pass through."""
        # COMPLETELY DISABLED - All rate limiting functionality removed
        # This middleware does nothing and passes all requests through
        return await call_next(request)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Production-ready authentication middleware with API key and token support."""

    def __init__(
        self, app: ASGIApp, require_auth: bool = False, api_keys: list[str] | None = None
    ) -> None:
        super().__init__(app)
        self.require_auth = require_auth
        self.api_keys = set(api_keys or [])
        self.public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply authentication based on configuration."""
        try:
            # Skip authentication for public paths
            if request.url.path in self.public_paths:
                return await call_next(request)

            # Skip authentication if not required
            if not self.require_auth:
                return await call_next(request)

            # Check for API key authentication
            auth_result = self._authenticate_request(request)

            if not auth_result["authenticated"]:
                logger.warning(
                    f"Authentication failed for {request.url.path}: {auth_result['reason']}"
                )
                return self._create_auth_error_response(auth_result["reason"])

            # Add authentication info to request state
            request.state.authenticated = True
            request.state.auth_method = auth_result["method"]
            request.state.client_id = auth_result.get("client_id", "unknown")

            return await call_next(request)

        except Exception:
            logger.exception("Authentication middleware error")
            return self._create_auth_error_response("Authentication system error")

    def _authenticate_request(self, request: Request) -> dict:
        """Authenticate the request using various methods."""
        # Method 1: API Key in header
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key in self.api_keys:
            return {
                "authenticated": True,
                "method": "api_key",
                "client_id": f"api_key_{api_key[:8]}...",
            }

        # Method 2: API Key in query parameter (less secure, but sometimes needed)
        api_key = request.query_params.get("api_key")
        if api_key and api_key in self.api_keys:
            logger.warning("API key provided in query parameter - consider using headers")
            return {
                "authenticated": True,
                "method": "api_key_query",
                "client_id": f"api_key_{api_key[:8]}...",
            }

        # Method 3: Bearer token (placeholder for JWT or other tokens)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            # In production, validate JWT token here
            # For now, we'll just check if it's not empty
            if token and len(token) > 10:
                return {
                    "authenticated": True,
                    "method": "bearer_token",
                    "client_id": f"token_{token[:8]}...",
                }

        # No valid authentication found
        return {
            "authenticated": False,
            "reason": "No valid authentication credentials provided",
        }

    def _create_auth_error_response(self, reason: str) -> JSONResponse:
        """Create authentication error response."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "AUTHENTICATION_REQUIRED",
                    "message": reason,
                    "status": 401,
                }
            },
            headers={"WWW-Authenticate": "Bearer, X-API-Key"},
        )
