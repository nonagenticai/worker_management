"""Centralized error handling middleware and utilities for the Worker Management API.

This module provides consistent error response formatting, validation error handling,
and middleware for catching and transforming exceptions across all API endpoints.
"""

import contextlib
import logging
import traceback
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi.exceptions import RequestValidationError

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors with consistent formatting."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        self.headers = headers
        super().__init__(detail)


class APIValidationError(APIError):
    """Validation error with field-level details."""

    def __init__(self, detail: str, errors: dict[str, Any] | None = None) -> None:
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=detail,
            error_code="VALIDATION_ERROR",
        )
        self.errors = errors or {}


# Alias for backwards compatibility
ValidationError = APIValidationError


class NotFoundError(APIError):
    """Resource not found error."""

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} not found: {identifier}",
            error_code="NOT_FOUND",
        )


class BadRequestError(APIError):
    """Bad request error for malformed requests."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="BAD_REQUEST",
        )


class InternalServerError(APIError):
    """Internal server error."""

    def __init__(self, detail: str = "Internal server error") -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="INTERNAL_ERROR",
        )


class MethodNotAllowedError(APIError):
    """Method not allowed error."""

    def __init__(self, method: str, allowed_methods: list[str]) -> None:
        super().__init__(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail=f"Method {method} not allowed",
            error_code="METHOD_NOT_ALLOWED",
            headers={"Allow": ", ".join(allowed_methods)},
        )


class ConflictError(APIError):
    """Conflict error for duplicate resources."""

    def __init__(self, resource_type: str, identifier: str, detail: str | None = None) -> None:
        if detail is None:
            detail = f"{resource_type} '{identifier}' already exists"
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="CONFLICT",
        )


def format_validation_errors(errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Format Pydantic validation errors into a consistent structure.

    Args:
        errors: List of validation errors from Pydantic

    Returns:
        Dict containing formatted error details
    """
    formatted_errors = []

    for error in errors:
        loc = " -> ".join(str(x) for x in error.get("loc", []))
        formatted_errors.append(
            {
                "field": loc,
                "message": error.get("msg", "Invalid value"),
                "type": error.get("type", "value_error"),
                # No "value": pydantic's `input` is the caller's submitted body,
                # which can hold credentials (mcp_config auth, api_key_override);
                # echoing it copies a secret into every log that stores the 422.
            }
        )

    return {
        "validation_errors": formatted_errors,
        "error_count": len(formatted_errors),
    }


def create_error_response(
    status_code: int,
    detail: str,
    error_code: str | None = None,
    errors: dict[str, Any] | None = None,
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Create a consistent error response.

    Args:
        status_code: HTTP status code
        detail: Human-readable error message
        error_code: Machine-readable error code
        errors: Additional error details (e.g., validation errors)
        request_id: Request ID for tracking
        headers: Additional HTTP headers to include in response

    Returns:
        JSONResponse with consistent error format
    """
    error_body: dict[str, Any] = {
        "code": error_code or "ERROR",
        "message": detail,
        "status": status_code,
    }

    if errors:
        error_body["details"] = errors

    if request_id:
        error_body["request_id"] = request_id

    content: dict[str, Any] = {"error": error_body}

    return JSONResponse(status_code=status_code, content=content, headers=headers)


def _typed_dict_detail(detail: Any) -> tuple[str, str] | None:
    """``(code, message)`` when ``detail`` is exactly ``{code: str, message: str}``."""
    if not isinstance(detail, dict) or set(detail) != {"code", "message"}:
        return None
    code, message = detail.get("code"), detail.get("message")
    if isinstance(code, str) and code and isinstance(message, str):
        return code, message
    return None


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with consistent formatting.

    Args:
        request: The incoming request
        exc: The HTTP exception

    Returns:
        JSONResponse with consistent error format
    """
    # Debug logging for all exceptions
    logger.info(f"HTTPException handler: {exc.status_code} {request.method} {request.url.path}")

    # Special handling for 405 Method Not Allowed
    if exc.status_code == 405:
        return await handle_405_method_not_allowed(request, exc)

    # Map status codes to error codes
    error_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        # WM-2: 402 was unmapped, so a 402 rendered as code "ERROR".
        402: "PAYMENT_REQUIRED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        # 429: "RATE_LIMIT_EXCEEDED",  # DISABLED - No rate limiting
        500: "INTERNAL_ERROR",
        501: "NOT_IMPLEMENTED",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
    }

    # A contract-fixed body (e.g. P4 amendment A4's
    # ``{"error": "prompt_version_not_runnable", ...}``) is rendered AS the
    # response, at the top level — not wrapped in the envelope, not under
    # ``detail``. Only exceptions that set ``raw_body`` take this branch.
    raw_body = getattr(exc, "raw_body", None)
    if isinstance(raw_body, dict):
        return JSONResponse(
            status_code=exc.status_code, content=raw_body, headers=exc.headers or None
        )

    # Always use comprehensive error format for consistency
    error_code = getattr(exc, "error_code", None) or error_code_map.get(exc.status_code, "ERROR")
    detail = exc.detail
    # WM-2: an HTTPException raised with a typed dict detail ``{code, message}``
    # (the 402 ``byo_llm_not_configured``) renders its code AS ``error.code``
    # and its message AS ``error.message`` (a string), instead of the dict
    # nested under ``message`` with ``code`` "ERROR".
    lifted = _typed_dict_detail(detail)
    if lifted is not None and not getattr(exc, "error_code", None):
        error_code, detail = lifted

    response = create_error_response(
        status_code=exc.status_code,
        detail=detail,
        error_code=error_code,
        # Additive: only exceptions that carry ``error_details`` (a failed worker
        # run's sent-sampling read-back) render ``error.details``.
        errors=getattr(exc, "error_details", None),
        request_id=request.headers.get("X-Request-ID"),
    )

    # Preserve headers from the original exception
    if hasattr(exc, "headers") and exc.headers:
        for key, value in exc.headers.items():
            response.headers[key] = value

    return response


async def handle_405_method_not_allowed(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle 405 Method Not Allowed with proper Allow headers and error format.

    Dynamically derives allowed methods by introspecting FastAPI routes to avoid
    drift from hard-coded maps.

    Args:
        request: The incoming request
        exc: The 405 HTTP exception

    Returns:
        JSONResponse with consistent 405 error format and proper Allow header
    """
    path = request.url.path

    # Try to dynamically compute allowed methods from the application's routes
    allowed_methods = _get_allowed_methods_from_app(request, path)

    # Debug logging
    logger.info(f"405 handler: path={path}, allowed_methods={allowed_methods}")
    logger.info(f"Original exception headers: {getattr(exc, 'headers', None)}")

    # Always use comprehensive error format for consistency
    response_content = {
        "error": {
            "code": "METHOD_NOT_ALLOWED",
            "message": "Method Not Allowed",
            "status": 405,
        }
    }

    response = JSONResponse(status_code=405, content=response_content)

    # Set the proper Allow header
    if allowed_methods:
        response.headers["Allow"] = ", ".join(sorted(set(allowed_methods)))
        logger.info(f"Set Allow header from app routes: {response.headers['Allow']}")
    # Fallback to original header if we can't determine allowed methods
    elif hasattr(exc, "headers") and exc.headers and "Allow" in exc.headers:
        response.headers["Allow"] = exc.headers["Allow"]
        logger.info(f"Set Allow header from exception: {response.headers['Allow']}")
    else:
        logger.warning(f"No Allow header could be determined for path: {path}")

    return response


def _get_allowed_methods_for_path(path: str, endpoint_methods: dict) -> list[Any]:
    """Get allowed methods for a given path."""
    # Try exact match first
    if path in endpoint_methods:
        return list(endpoint_methods[path])

    # Try pattern matching for dynamic segments
    for pattern, methods in endpoint_methods.items():
        if _path_matches_pattern(path, pattern):
            return list(methods)

    return []


def _path_matches_pattern(path: str, pattern: str) -> bool:
    """Check if a path matches a pattern with dynamic segments."""
    path_parts = path.strip("/").split("/")
    pattern_parts = pattern.strip("/").split("/")

    if len(path_parts) != len(pattern_parts):
        return False

    for path_part, pattern_part in zip(path_parts, pattern_parts, strict=False):
        # Skip dynamic segments (those in curly braces)
        if pattern_part.startswith("{") and pattern_part.endswith("}"):
            continue
        # Exact match required for static segments
        if path_part != pattern_part:
            return False

    return True


def _get_allowed_methods_from_app(request: Request, path: str) -> list:
    """Derive allowed HTTP methods for a path by inspecting FastAPI routes.

    This avoids maintaining a hard-coded map that can drift from reality.
    """
    try:
        routes = getattr(request.app, "routes", [])
    except Exception:
        routes = []

    allowed: list[str] = []
    for route in routes:
        # Only consider API routes that have path and methods
        route_path = getattr(route, "path", None) or getattr(route, "path_format", None)
        route_methods = getattr(route, "methods", None)
        if not route_path or not route_methods:
            continue

        if _path_matches_pattern(path, route_path):
            allowed.extend(list(route_methods))

    # Remove method used in the failing request if present (optional).
    # Wrapped in suppress() because request.method may be unset in synthetic
    # test contexts; an empty allowed list is the safe fallback.
    with contextlib.suppress(AttributeError):
        method = request.method.upper()
        allowed = [m for m in allowed if m != method]

    # Starlette includes HEAD automatically for GET; ensure HEAD present when GET allowed
    if "GET" in allowed and "HEAD" not in allowed:
        allowed.append("HEAD")

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for m in allowed:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors with secure field information.

    Args:
        request: The incoming request
        exc: The validation exception

    Returns:
        JSONResponse with validation error details (sanitized)
    """
    errors = format_validation_errors(list(exc.errors()))

    # Create a human-readable summary but sanitize sensitive information
    error_messages = []
    for error in errors["validation_errors"]:
        field_name = error["field"]
        message = error["message"]

        # Sanitize error messages that might contain sensitive info
        if "password" in message.lower() or "key" in message.lower() or "token" in message.lower():
            message = "Invalid value for sensitive field"

        # Limit error message length to prevent information leakage
        if len(message) > 200:
            message = message[:197] + "..."

        error_messages.append(f"{field_name}: {message}")

    # Limit number of error messages to prevent DoS
    if len(error_messages) > 10:
        error_messages = [*error_messages[:10], "... and more validation errors"]

    detail = "Validation failed: " + "; ".join(error_messages)

    # Log validation errors for security monitoring
    logger.warning(
        f"Validation error from {getattr(request.client, 'host', 'unknown') if request.client else 'unknown'}: {detail}",
        extra={
            "request_id": request.headers.get("X-Request-ID"),
            "method": request.method,
            "path": request.url.path,
            "error_count": len(errors["validation_errors"]),
        },
    )

    # For 422 validation errors, return validation_errors at root level
    # to match the ValidationErrorResponse schema
    content = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": detail,
            "status": status.HTTP_422_UNPROCESSABLE_CONTENT,
        },
        "validation_errors": errors["validation_errors"],
    }

    request_id = request.headers.get("X-Request-ID")
    if request_id:
        content["error"]["request_id"] = request_id

    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=content)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API errors.

    Args:
        request: The incoming request
        exc: The API error

    Returns:
        JSONResponse with error details including custom headers
    """
    return create_error_response(
        status_code=exc.status_code,
        detail=exc.detail,
        error_code=exc.error_code,
        errors=getattr(exc, "errors", None),
        request_id=request.headers.get("X-Request-ID"),
        headers=exc.headers,
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with logging and secure error responses.

    Args:
        request: The incoming request
        exc: The unexpected exception

    Returns:
        JSONResponse with generic error message (no sensitive info leaked)
    """
    import os

    # Generate a unique error ID for tracking
    from uuid import uuid7

    error_id = str(uuid7())

    # Log the full exception with traceback (but don't expose it)
    logger.error(
        f"[{error_id}] Unexpected error handling request {request.method} {request.url.path}: {exc}",
        exc_info=True,
        extra={
            "error_id": error_id,
            "request_id": request.headers.get("X-Request-ID"),
            "method": request.method,
            "path": request.url.path,
            "client_ip": (
                getattr(request.client, "host", "unknown") if request.client else "unknown"
            ),
            "user_agent": request.headers.get("user-agent", "unknown"),
            "traceback": traceback.format_exc(),
        },
    )

    # NEVER expose internal details in production
    # Only show generic error message to prevent information disclosure
    detail = "An internal server error occurred. Please try again later."

    # Only in debug mode AND development environment, show more details
    is_development = os.getenv("ENVIRONMENT", "production").lower() in [
        "development",
        "dev",
        "local",
    ]
    if logger.isEnabledFor(logging.DEBUG) and is_development:
        # Even in debug mode, sanitize the error message
        error_str = str(exc)
        # Remove potential sensitive information patterns
        import re

        # Remove file paths
        error_str = re.sub(r"/[^\s]+/", "[PATH]/", error_str)
        # Remove potential connection strings or credentials
        error_str = re.sub(
            r"password=[^\s]+", "password=[REDACTED]", error_str, flags=re.IGNORECASE
        )
        error_str = re.sub(r"key=[^\s]+", "key=[REDACTED]", error_str, flags=re.IGNORECASE)
        error_str = re.sub(r"token=[^\s]+", "token=[REDACTED]", error_str, flags=re.IGNORECASE)

        detail = f"Internal error (ID: {error_id[:8]}): {error_str[:200]}"

    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
        error_code="INTERNAL_ERROR",
        request_id=request.headers.get("X-Request-ID"),
    )


def validate_string_field(
    value: str | None,
    field_name: str,
    min_length: int = 1,
    max_length: int = 1000,
    pattern: str | None = None,
    allow_null: bool = False,
) -> str | None:
    """Validate a string field with common constraints.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        pattern: Optional regex pattern to match
        allow_null: Whether to allow None values

    Returns:
        Validated string value

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if allow_null:
            return value
        raise ValidationError(f"{field_name} cannot be null")

    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")

    # Check for null bytes
    if "\x00" in value:
        raise ValidationError(f"{field_name} cannot contain null bytes")

    # Strip whitespace for length checks
    stripped = value.strip()

    if len(stripped) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters long")

    if len(stripped) > max_length:
        raise ValidationError(f"{field_name} cannot exceed {max_length} characters")

    if pattern:
        import re

        if not re.match(pattern, stripped):
            raise ValidationError(f"{field_name} contains invalid characters or format")

    return stripped


def validate_list_field(
    value: list[Any] | None,
    field_name: str,
    min_items: int = 0,
    max_items: int = 100,
    item_validator: Callable[[Any], Any] | None = None,
    allow_null: bool = False,
) -> list[Any] | None:
    """Validate a list field with common constraints.

    Args:
        value: The list to validate
        field_name: Name of the field for error messages
        min_items: Minimum number of items
        max_items: Maximum number of items
        item_validator: Optional function to validate each item
        allow_null: Whether to allow None values

    Returns:
        Validated list

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if allow_null:
            return value
        raise ValidationError(f"{field_name} cannot be null")

    if not isinstance(value, list):
        raise ValidationError(f"{field_name} must be a list")

    if len(value) < min_items:
        raise ValidationError(f"{field_name} must contain at least {min_items} items")

    if len(value) > max_items:
        raise ValidationError(f"{field_name} cannot contain more than {max_items} items")

    if item_validator:
        validated_items = []
        for i, item in enumerate(value):
            try:
                validated_item = item_validator(item)
                if validated_item is not None:  # Skip None items
                    validated_items.append(validated_item)
            except ValidationError as e:
                raise ValidationError(f"{field_name}[{i}]: {e.detail}") from e
        return validated_items

    return value


def validate_dict_field(
    value: dict[str, Any] | None,
    field_name: str,
    max_size_bytes: int = 10000,
    allow_null: bool = False,
) -> dict[str, Any] | None:
    """Validate a dictionary field with size constraints.

    Args:
        value: The dictionary to validate
        field_name: Name of the field for error messages
        max_size_bytes: Maximum size when serialized
        allow_null: Whether to allow None values

    Returns:
        Validated dictionary

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if allow_null:
            return value
        raise ValidationError(f"{field_name} cannot be null")

    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be a dictionary")

    # Check serialized size
    import json

    try:
        serialized = json.dumps(value)
        if len(serialized.encode("utf-8")) > max_size_bytes:
            raise ValidationError(
                f"{field_name} is too large (max {max_size_bytes} bytes when serialized)"
            )
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{field_name} contains non-serializable data: {e!s}") from e

    return value


def sanitize_input(value: str) -> str:
    """Sanitize user input to prevent injection attacks.

    Args:
        value: The input to sanitize

    Returns:
        Sanitized string
    """
    if not value:
        return value

    try:
        from src.api.security_validation import validate_string_security

        return validate_string_security(value, "input", max_length=10000, strict_mode=True)
    except Exception:
        # Fallback to basic sanitization if security validation fails
        # Remove null bytes
        value = value.replace("\x00", "")

        # Limit length to prevent DoS
        max_length = 10000
        if len(value) > max_length:
            value = value[:max_length]

        return value
