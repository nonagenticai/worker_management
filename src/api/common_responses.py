"""Common response definitions for API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Standard error detail structure."""

    code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    status: int = Field(..., description="HTTP status code")


class ErrorResponse(BaseModel):
    """Standard error response structure."""

    error: ErrorDetail = Field(..., description="Error details")


class ValidationErrorDetail(BaseModel):
    """Individual validation error detail."""

    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    type: str = Field(..., description="Type of validation error")
    # No ``value`` field: the 422 handler never echoes the submitted input
    # (it can hold credentials — mcp_config auth, api_key_override), so the
    # contract must not advertise one. See error_handlers.format_validation_errors.


class ValidationErrorResponse(BaseModel):
    """Validation error response structure."""

    error: ErrorDetail = Field(..., description="Error details")
    validation_errors: list[ValidationErrorDetail] = Field(
        ..., description="List of validation errors"
    )


# Response model constants for easy reference
STANDARD_ERROR_MODELS = {
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    405: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ValidationErrorResponse},
    429: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}

# Convenience response dictionaries for endpoints
# Use these when documenting endpoints with responses={} parameter
AUTH_RESPONSES = {401: {"description": "Authentication required"}}
NOT_FOUND_RESPONSES = {404: {"description": "Resource not found"}}
AUTH_AND_NOT_FOUND = {**AUTH_RESPONSES, **NOT_FOUND_RESPONSES}

# Common response schemas
# RATE LIMITING RESPONSES REMOVED - No longer needed

STANDARD_ERROR_RESPONSES = {
    400: {
        "description": "Bad request",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "bad_request": {
                        "summary": "Bad request example",
                        "value": {
                            "error": {
                                "code": "BAD_REQUEST",
                                "message": "Bad request",
                                "status": 400,
                            }
                        },
                    }
                }
            }
        },
    },
    401: {
        "description": "Unauthorized",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "unauthorized": {
                        "summary": "Unauthorized example",
                        "value": {
                            "error": {
                                "code": "UNAUTHORIZED",
                                "message": "Authentication required",
                                "status": 401,
                            }
                        },
                    }
                }
            }
        },
    },
    403: {
        "description": "Forbidden",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "forbidden": {
                        "summary": "Forbidden example",
                        "value": {
                            "error": {
                                "code": "FORBIDDEN",
                                "message": "Access denied",
                                "status": 403,
                            }
                        },
                    }
                }
            }
        },
    },
    404: {
        "description": "Resource not found",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "not_found": {
                        "summary": "Resource not found example",
                        "value": {
                            "error": {
                                "code": "NOT_FOUND",
                                "message": "Resource not found",
                                "status": 404,
                            }
                        },
                    }
                }
            }
        },
    },
    405: {
        "description": "Method not allowed",
        "model": ErrorResponse,
        "headers": {
            "Allow": {
                "description": "Allowed HTTP methods for this endpoint",
                "schema": {"type": "string", "example": "GET, POST"},
            }
        },
        "content": {
            "application/json": {
                "examples": {
                    "method_not_allowed": {
                        "summary": "Method not allowed example",
                        "value": {
                            "error": {
                                "code": "METHOD_NOT_ALLOWED",
                                "message": "Method Not Allowed",
                                "status": 405,
                            }
                        },
                    }
                }
            }
        },
    },
    409: {
        "description": "Conflict",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "conflict": {
                        "summary": "Conflict example",
                        "value": {
                            "error": {
                                "code": "CONFLICT",
                                "message": "Resource already exists",
                                "status": 409,
                            }
                        },
                    }
                }
            }
        },
    },
    422: {
        "description": "Validation error",
        "model": ValidationErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "validation_error": {
                        "summary": "Validation error example",
                        "value": {
                            "error": {
                                "code": "VALIDATION_ERROR",
                                "message": "Validation failed",
                                "status": 422,
                            },
                            "validation_errors": [
                                {
                                    "field": "name",
                                    "message": "Field is required",
                                    "type": "missing",
                                }
                            ],
                        },
                    }
                }
            }
        },
    },
    429: {
        "description": "Rate limit exceeded",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "rate_limit": {
                        "summary": "Rate limit exceeded example",
                        "value": {
                            "error": {
                                "code": "RATE_LIMIT_EXCEEDED",
                                "message": "Rate limit exceeded",
                                "status": 429,
                            }
                        },
                    }
                }
            }
        },
    },
    500: {
        "description": "Internal server error",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "internal_error": {
                        "summary": "Internal server error example",
                        "value": {
                            "error": {
                                "code": "INTERNAL_ERROR",
                                "message": "Internal server error",
                                "status": 500,
                            }
                        },
                    }
                }
            }
        },
    },
}


def get_standard_responses(*status_codes: int) -> dict[int, dict[str, Any]]:
    """Get standard responses for the given status codes.

    Always includes 405 (Method Not Allowed) for all endpoints
    since this middleware check applies to all requests.
    Rate limiting (429) responses have been removed.

    Args:
        *status_codes: HTTP status codes to include in response (400, 401, 403, 404, etc.)

    Returns:
        Dict mapping status codes to their response configurations including model, description, and examples
    """
    if not status_codes:
        # Return all standard responses including 405
        return STANDARD_ERROR_RESPONSES

    result = {}
    for code in status_codes:
        if code in STANDARD_ERROR_RESPONSES:
            result[code] = STANDARD_ERROR_RESPONSES[code]

    # Always include 405 for method not allowed since it applies to all endpoints
    result[405] = STANDARD_ERROR_RESPONSES[405]

    # Rate limiting (429) responses removed - no longer used

    return result
