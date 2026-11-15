"""Tests for src.api.error_handlers module.

Covers exception classes, formatting helpers, validation functions,
sanitize_input, and the async handler functions.
"""

import asyncio
import json
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.error_handlers import (
    APIError,
    APIValidationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    MethodNotAllowedError,
    NotFoundError,
    ValidationError,
    api_error_handler,
    create_error_response,
    format_validation_errors,
    general_exception_handler,
    http_exception_handler,
    sanitize_input,
    validate_dict_field,
    validate_list_field,
    validate_string_field,
    validation_exception_handler,
)

# ============================================================================
# Helper to run async handlers in sync tests
# ============================================================================


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_request(method: str = "GET", path: str = "/test", headers: dict | None = None):
    """Build a minimal mock Request suitable for handler functions."""
    from fastapi import Request

    mock = MagicMock(spec=Request)
    mock.method = method
    mock.url = MagicMock()
    mock.url.path = path
    mock.client = MagicMock()
    mock.client.host = "127.0.0.1"

    real_headers = headers or {}
    # Use a MagicMock for headers so .get() can be assigned
    headers_mock = MagicMock()
    headers_mock.get = real_headers.get
    mock.headers = headers_mock
    mock.app = MagicMock()
    mock.app.routes = []
    return mock


# ============================================================================
# 1. Exception classes - status_code and error_code
# ============================================================================


class TestExceptionClasses:
    """Each custom exception must carry correct status_code and error_code."""

    def test_api_error_defaults(self):
        exc = APIError(status_code=418, detail="teapot", error_code="TEAPOT")
        assert exc.status_code == 418
        assert exc.detail == "teapot"
        assert exc.error_code == "TEAPOT"
        assert exc.headers is None
        assert str(exc) == "teapot"

    def test_api_error_with_headers(self):
        exc = APIError(status_code=400, detail="bad", headers={"X-Custom": "val"})
        assert exc.headers == {"X-Custom": "val"}
        assert exc.error_code is None

    def test_api_validation_error(self):
        exc = APIValidationError("bad input", errors={"field": "err"})
        assert exc.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.errors == {"field": "err"}

    def test_api_validation_error_default_errors(self):
        exc = APIValidationError("bad input")
        assert exc.errors == {}

    def test_validation_error_alias(self):
        assert ValidationError is APIValidationError

    def test_not_found_error(self):
        exc = NotFoundError("Worker", "abc-123")
        assert exc.status_code == 404
        assert exc.error_code == "NOT_FOUND"
        assert "Worker" in exc.detail
        assert "abc-123" in exc.detail

    def test_bad_request_error(self):
        exc = BadRequestError("missing field")
        assert exc.status_code == 400
        assert exc.error_code == "BAD_REQUEST"
        assert exc.detail == "missing field"

    def test_internal_server_error_default_message(self):
        exc = InternalServerError()
        assert exc.status_code == 500
        assert exc.error_code == "INTERNAL_ERROR"
        assert exc.detail == "Internal server error"

    def test_internal_server_error_custom_message(self):
        exc = InternalServerError("db down")
        assert exc.detail == "db down"

    def test_method_not_allowed_error(self):
        exc = MethodNotAllowedError("DELETE", ["GET", "POST"])
        assert exc.status_code == 405
        assert exc.error_code == "METHOD_NOT_ALLOWED"
        assert "DELETE" in exc.detail
        assert exc.headers == {"Allow": "GET, POST"}

    def test_conflict_error_default_detail(self):
        exc = ConflictError("Worker", "worker-1")
        assert exc.status_code == 409
        assert exc.error_code == "CONFLICT"
        assert "Worker" in exc.detail
        assert "worker-1" in exc.detail
        assert "already exists" in exc.detail

    def test_conflict_error_custom_detail(self):
        exc = ConflictError("Worker", "w1", detail="custom conflict msg")
        assert exc.detail == "custom conflict msg"

    def test_all_exceptions_are_subclass_of_api_error(self):
        classes = [
            APIValidationError,
            NotFoundError,
            BadRequestError,
            InternalServerError,
            MethodNotAllowedError,
            ConflictError,
        ]
        for cls in classes:
            assert issubclass(cls, APIError), f"{cls.__name__} should extend APIError"

    def test_all_exceptions_are_subclass_of_exception(self):
        classes = [
            APIError,
            APIValidationError,
            NotFoundError,
            BadRequestError,
            InternalServerError,
            MethodNotAllowedError,
            ConflictError,
        ]
        for cls in classes:
            assert issubclass(cls, Exception), f"{cls.__name__} should extend Exception"


# ============================================================================
# 2. format_validation_errors
# ============================================================================


class TestFormatValidationErrors:
    def test_basic_structure(self):
        errors = [
            {
                "loc": ["body", "name"],
                "msg": "field required",
                "type": "missing",
                "input": None,
            },
        ]
        result = format_validation_errors(errors)
        assert "validation_errors" in result
        assert "error_count" in result
        assert result["error_count"] == 1

    def test_multiple_errors(self):
        errors = [
            {"loc": ["body", "name"], "msg": "required", "type": "missing"},
            {"loc": ["body", "age"], "msg": "not a number", "type": "type_error"},
        ]
        result = format_validation_errors(errors)
        assert result["error_count"] == 2
        assert len(result["validation_errors"]) == 2

    def test_field_location_formatting(self):
        errors = [{"loc": ["body", "config", "timeout"], "msg": "bad", "type": "err"}]
        result = format_validation_errors(errors)
        assert result["validation_errors"][0]["field"] == "body -> config -> timeout"

    def test_defaults_for_missing_keys(self):
        errors = [{}]
        result = format_validation_errors(errors)
        ve = result["validation_errors"][0]
        assert ve["field"] == ""
        assert ve["message"] == "Invalid value"
        assert ve["type"] == "value_error"
        assert "value" not in ve

    def test_empty_list(self):
        result = format_validation_errors([])
        assert result["error_count"] == 0
        assert result["validation_errors"] == []

    def test_input_value_never_echoed(self):
        canary = "input-canary-" + "value"
        errors = [
            {
                "loc": ["body", "mcp_config"],
                "msg": "m",
                "type": "t",
                "input": {"auth": {"token": canary}},
            }
        ]
        result = format_validation_errors(errors)
        assert "value" not in result["validation_errors"][0]
        assert canary not in repr(result)

    def test_openapi_contract_matches_emitted_keys(self):
        """The published 422 schema advertises exactly the keys the handler emits.

        #296 stopped echoing ``value``; a schema that still advertised it told
        clients to expect (and log) submitted input the service never sends.
        """
        from fastapi import FastAPI

        from src.api.common_responses import (
            STANDARD_ERROR_MODELS,
            STANDARD_ERROR_RESPONSES,
            ValidationErrorDetail,
        )

        emitted = format_validation_errors(
            [{"loc": ["body", "x"], "msg": "m", "type": "t", "input": 1}]
        )
        assert set(emitted["validation_errors"][0]) == set(ValidationErrorDetail.model_fields)
        assert "value" not in ValidationErrorDetail.model_fields

        app = FastAPI()

        @app.get("/probe", responses={422: STANDARD_ERROR_MODELS[422]})
        def _probe(q: int) -> dict:  # pragma: no cover - schema only
            return {}

        schema = app.openapi()["components"]["schemas"]["ValidationErrorDetail"]
        assert set(schema["properties"]) == {"field", "message", "type"}

        example = STANDARD_ERROR_RESPONSES[422]["content"]["application/json"]["examples"]
        for ex in example.values():
            for ve in ex["value"]["validation_errors"]:
                assert "value" not in ve

    def test_single_location_element(self):
        errors = [{"loc": ["body"], "msg": "bad", "type": "err"}]
        result = format_validation_errors(errors)
        assert result["validation_errors"][0]["field"] == "body"


# ============================================================================
# 3. create_error_response
# ============================================================================


class TestCreateErrorResponse:
    def test_basic_response(self):
        resp = create_error_response(400, "bad request")
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "ERROR"
        assert body["error"]["message"] == "bad request"
        assert body["error"]["status"] == 400

    def test_custom_error_code(self):
        resp = create_error_response(404, "not found", error_code="NOT_FOUND")
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "NOT_FOUND"

    def test_includes_details_when_provided(self):
        details = {"validation_errors": [{"field": "name"}]}
        resp = create_error_response(422, "invalid", errors=details)
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["details"] == details

    def test_no_details_key_when_not_provided(self):
        resp = create_error_response(500, "oops")
        body = json.loads(bytes(resp.body).decode())
        assert "details" not in body["error"]

    def test_request_id_included(self):
        resp = create_error_response(500, "err", request_id="req-123")
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["request_id"] == "req-123"

    def test_no_request_id_key_when_not_provided(self):
        resp = create_error_response(500, "err")
        body = json.loads(bytes(resp.body).decode())
        assert "request_id" not in body["error"]

    def test_custom_headers(self):
        resp = create_error_response(405, "nope", headers={"Allow": "GET"})
        assert resp.headers.get("Allow") == "GET"

    def test_empty_errors_dict_not_included(self):
        resp = create_error_response(400, "err", errors={})
        body = json.loads(bytes(resp.body).decode())
        # Empty dict is falsy, so details should not be included
        assert "details" not in body["error"]


# ============================================================================
# 4. validate_string_field
# ============================================================================


class TestValidateStringField:
    def test_valid_string(self):
        assert validate_string_field("hello", "name") == "hello"

    def test_strips_whitespace(self):
        assert validate_string_field("  hello  ", "name") == "hello"

    def test_null_not_allowed(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_string_field(None, "name")
        assert "cannot be null" in exc_info.value.detail

    def test_null_allowed(self):
        assert validate_string_field(None, "name", allow_null=True) is None

    def test_not_a_string(self):
        with pytest.raises(ValidationError) as exc_info:
            # Intentionally invalid type — verifies runtime rejection.
            validate_string_field(cast("Any", 123), "name")
        assert "must be a string" in exc_info.value.detail

    def test_null_byte_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_string_field("abc\x00def", "name")
        assert "null bytes" in exc_info.value.detail

    def test_min_length_violation(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_string_field("", "name", min_length=1)
        assert "at least 1 characters" in exc_info.value.detail

    def test_min_length_custom(self):
        with pytest.raises(ValidationError):
            validate_string_field("ab", "name", min_length=5)

    def test_max_length_violation(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_string_field("a" * 20, "name", max_length=10)
        assert "cannot exceed 10 characters" in exc_info.value.detail

    def test_pattern_match_success(self):
        result = validate_string_field("abc123", "code", pattern=r"^[a-z0-9]+$")
        assert result == "abc123"

    def test_pattern_match_failure(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_string_field("ABC!!!", "code", pattern=r"^[a-z0-9]+$")
        assert "invalid characters" in exc_info.value.detail

    def test_whitespace_only_fails_min_length(self):
        with pytest.raises(ValidationError):
            validate_string_field("   ", "name", min_length=1)

    def test_field_name_in_error_message(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_string_field(None, "my_field")
        assert "my_field" in exc_info.value.detail


# ============================================================================
# 5. validate_list_field
# ============================================================================


class TestValidateListField:
    def test_valid_list(self):
        result = validate_list_field(["a", "b"], "items")
        assert result == ["a", "b"]

    def test_null_not_allowed(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_list_field(None, "items")
        assert "cannot be null" in exc_info.value.detail

    def test_null_allowed(self):
        assert validate_list_field(None, "items", allow_null=True) is None

    def test_not_a_list(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_list_field(cast("Any", "not-a-list"), "items")
        assert "must be a list" in exc_info.value.detail

    def test_min_items_violation(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_list_field([], "items", min_items=1)
        assert "at least 1 items" in exc_info.value.detail

    def test_max_items_violation(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_list_field([1, 2, 3], "items", max_items=2)
        assert "more than 2 items" in exc_info.value.detail

    def test_item_validator_success(self):
        def upper(item: Any) -> str:
            return str(item).upper()

        result = validate_list_field(["a", "b"], "items", item_validator=upper)
        assert result == ["A", "B"]

    def test_item_validator_skips_none_returns(self):
        def only_even(item: int):
            if item % 2 == 0:
                return item
            return None

        result = validate_list_field([1, 2, 3, 4], "nums", item_validator=only_even)
        assert result == [2, 4]

    def test_item_validator_propagates_validation_error(self):
        def strict_validator(item: Any) -> str:
            raise ValidationError(f"bad item: {item}")

        with pytest.raises(ValidationError) as exc_info:
            validate_list_field(["x"], "items", item_validator=strict_validator)
        assert "items[0]" in exc_info.value.detail

    def test_empty_list_passes_with_min_zero(self):
        result = validate_list_field([], "items", min_items=0)
        assert result == []

    def test_field_name_in_error_message(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_list_field(None, "my_list")
        assert "my_list" in exc_info.value.detail


# ============================================================================
# 6. validate_dict_field
# ============================================================================


class TestValidateDictField:
    def test_valid_dict(self):
        result = validate_dict_field({"key": "val"}, "config")
        assert result == {"key": "val"}

    def test_null_not_allowed(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_dict_field(None, "config")
        assert "cannot be null" in exc_info.value.detail

    def test_null_allowed(self):
        assert validate_dict_field(None, "config", allow_null=True) is None

    def test_not_a_dict(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_dict_field(cast("Any", [1, 2]), "config")
        assert "must be a dictionary" in exc_info.value.detail

    def test_size_limit_exceeded(self):
        big = {"data": "x" * 20000}
        with pytest.raises(ValidationError) as exc_info:
            validate_dict_field(big, "config", max_size_bytes=100)
        assert "too large" in exc_info.value.detail

    def test_size_limit_within_bounds(self):
        small = {"a": 1}
        result = validate_dict_field(small, "config", max_size_bytes=10000)
        assert result == small

    def test_non_serializable_value(self):
        bad = {"fn": lambda x: x}
        with pytest.raises(ValidationError) as exc_info:
            validate_dict_field(bad, "config")
        assert "non-serializable" in exc_info.value.detail

    def test_field_name_in_error_message(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_dict_field(None, "my_config")
        assert "my_config" in exc_info.value.detail


# ============================================================================
# 7. sanitize_input
# ============================================================================


class TestSanitizeInput:
    def test_empty_string_passthrough(self):
        assert sanitize_input("") == ""

    def test_basic_string_returned(self):
        result = sanitize_input("hello world")
        assert isinstance(result, str)
        assert "hello" in result

    def test_null_bytes_removed_via_fallback(self):
        """Ensure null bytes are removed when the fallback path is used."""
        with patch(
            "src.api.security_validation.validate_string_security",
            side_effect=Exception("forced fallback"),
        ):
            result = sanitize_input("foo\x00bar")
            assert "\x00" not in result
            assert "foobar" in result

    def test_length_limiting_fallback(self):
        """Ensure the fallback path truncates to 10000 chars."""
        with patch(
            "src.api.security_validation.validate_string_security",
            side_effect=Exception("forced fallback"),
        ):
            long_input = "a" * 20000
            result = sanitize_input(long_input)
            assert len(result) <= 10000

    def test_short_string_not_truncated_fallback(self):
        """A string under the limit should not be truncated."""
        with patch(
            "src.api.security_validation.validate_string_security",
            side_effect=Exception("forced fallback"),
        ):
            result = sanitize_input("short")
            assert result == "short"

    def test_none_like_empty_passthrough(self):
        """Empty string is falsy and should be returned as-is."""
        result = sanitize_input("")
        assert result == ""


# ============================================================================
# 8. Handler functions (async)
# ============================================================================


class TestHttpExceptionHandler:
    def test_basic_http_exception(self):
        req = _make_request()
        exc = HTTPException(status_code=404, detail="not found")
        resp = _run(http_exception_handler(req, exc))
        assert resp.status_code == 404
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "NOT_FOUND"
        assert body["error"]["message"] == "not found"

    def test_maps_status_codes_correctly(self):
        mapping = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            409: "CONFLICT",
            422: "VALIDATION_ERROR",
            500: "INTERNAL_ERROR",
        }
        for code, expected in mapping.items():
            req = _make_request()
            exc = HTTPException(status_code=code, detail="test")
            resp = _run(http_exception_handler(req, exc))
            body = json.loads(bytes(resp.body).decode())
            assert body["error"]["code"] == expected, f"Failed for status {code}"

    def test_402_is_typed_payment_required(self):
        """WM-2. MUTATION: drop 402 from the map ⇒ code "ERROR" ⇒ red."""
        req = _make_request()
        resp = _run(http_exception_handler(req, HTTPException(status_code=402, detail="pay")))
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "PAYMENT_REQUIRED"
        assert body["error"]["message"] == "pay"

    def test_typed_dict_detail_is_lifted_into_the_envelope(self):
        """WM-2: ``{code, message}`` detail renders as error.code / error.message
        (a string), not as a dict nested under message with code "ERROR"."""
        req = _make_request()
        exc = HTTPException(
            status_code=402,
            detail={"code": "byo_llm_not_configured", "message": "Configure it."},
        )
        body = json.loads(bytes(_run(http_exception_handler(req, exc)).body).decode())
        assert body["error"] == {
            "code": "byo_llm_not_configured",
            "message": "Configure it.",
            "status": 402,
        }

    def test_other_dict_details_are_not_lifted(self):
        """Only the exact ``{code, message}`` shape is lifted; anything else keeps
        its existing rendering."""
        req = _make_request()
        detail = {"code": "x", "message": "m", "extra": 1}
        exc = HTTPException(status_code=409, detail=detail)
        body = json.loads(bytes(_run(http_exception_handler(req, exc)).body).decode())
        assert body["error"]["code"] == "CONFLICT"
        assert body["error"]["message"] == detail

    def test_unknown_status_code_defaults_to_error(self):
        req = _make_request()
        exc = HTTPException(status_code=418, detail="teapot")
        resp = _run(http_exception_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "ERROR"

    def test_preserves_exception_headers(self):
        req = _make_request()
        exc = HTTPException(
            status_code=401, detail="unauth", headers={"WWW-Authenticate": "Bearer"}
        )
        resp = _run(http_exception_handler(req, exc))
        assert resp.headers.get("WWW-Authenticate") == "Bearer"

    def test_includes_request_id(self):
        req = _make_request(headers={"X-Request-ID": "req-xyz"})
        exc = HTTPException(status_code=400, detail="bad")
        resp = _run(http_exception_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["request_id"] == "req-xyz"


class TestValidationExceptionHandler:
    def test_formats_pydantic_errors(self):
        req = _make_request()
        pydantic_errors = [
            {
                "loc": ("body", "name"),
                "msg": "field required",
                "type": "missing",
                "input": None,
                "url": "...",
            }
        ]
        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = pydantic_errors

        resp = _run(validation_exception_handler(req, exc))
        assert resp.status_code == 422
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert "validation_errors" in body
        assert len(body["validation_errors"]) >= 1

    def test_sanitizes_sensitive_field_messages(self):
        req = _make_request()
        pydantic_errors = [
            {"loc": ("body", "password"), "msg": "password too short", "type": "err"},
        ]
        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = pydantic_errors

        resp = _run(validation_exception_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        # The summary message should have sanitized the "password" reference
        assert "Invalid value for sensitive field" in body["error"]["message"]

    def test_limits_error_count(self):
        req = _make_request()
        pydantic_errors = [
            {"loc": ("body", f"field_{i}"), "msg": f"error {i}", "type": "err"} for i in range(15)
        ]
        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = pydantic_errors

        resp = _run(validation_exception_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        # The message should indicate truncation
        assert "... and more" in body["error"]["message"]

    def test_includes_request_id(self):
        req = _make_request(headers={"X-Request-ID": "val-req-1"})
        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = [{"loc": ("body", "x"), "msg": "bad", "type": "err"}]

        resp = _run(validation_exception_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["request_id"] == "val-req-1"

    def test_no_request_id_when_absent(self):
        req = _make_request()
        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = [{"loc": ("body", "x"), "msg": "bad", "type": "err"}]

        resp = _run(validation_exception_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        assert "request_id" not in body["error"]

    def test_long_error_message_truncated(self):
        req = _make_request()
        long_msg = "x" * 300
        pydantic_errors = [
            {"loc": ("body", "field"), "msg": long_msg, "type": "err"},
        ]
        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = pydantic_errors

        resp = _run(validation_exception_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        # The message in the summary should be truncated (original was 300 chars,
        # limit is 200 chars per message with "..." appended)
        assert "..." in body["error"]["message"]


class TestApiErrorHandler:
    def test_handles_bad_request_error(self):
        req = _make_request()
        exc = BadRequestError("invalid input")
        resp = _run(api_error_handler(req, exc))
        assert resp.status_code == 400
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "BAD_REQUEST"
        assert body["error"]["message"] == "invalid input"

    def test_handles_not_found_error(self):
        req = _make_request()
        exc = NotFoundError("Worker", "w-1")
        resp = _run(api_error_handler(req, exc))
        assert resp.status_code == 404
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "NOT_FOUND"

    def test_includes_custom_headers(self):
        req = _make_request()
        exc = MethodNotAllowedError("DELETE", ["GET", "POST"])
        resp = _run(api_error_handler(req, exc))
        assert resp.headers.get("Allow") == "GET, POST"

    def test_includes_validation_errors(self):
        req = _make_request()
        exc = APIValidationError("bad", errors={"field": "err"})
        resp = _run(api_error_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["details"] == {"field": "err"}

    def test_includes_request_id(self):
        req = _make_request(headers={"X-Request-ID": "api-req-1"})
        exc = NotFoundError("Worker", "w-1")
        resp = _run(api_error_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["request_id"] == "api-req-1"

    def test_conflict_error_handler(self):
        req = _make_request()
        exc = ConflictError("Worker", "w-1")
        resp = _run(api_error_handler(req, exc))
        assert resp.status_code == 409
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "CONFLICT"

    def test_internal_server_error_handler(self):
        req = _make_request()
        exc = InternalServerError("db failure")
        resp = _run(api_error_handler(req, exc))
        assert resp.status_code == 500
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "INTERNAL_ERROR"


class TestGeneralExceptionHandler:
    def test_returns_500_with_generic_message(self):
        req = _make_request()
        exc = RuntimeError("something broke internally")
        resp = _run(general_exception_handler(req, exc))
        assert resp.status_code == 500
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["code"] == "INTERNAL_ERROR"
        # Should NOT leak the actual error message in production mode
        assert "something broke internally" not in body["error"]["message"]
        assert "internal server error" in body["error"]["message"].lower()

    def test_includes_request_id(self):
        req = _make_request(headers={"X-Request-ID": "gen-req-1"})
        exc = RuntimeError("boom")
        resp = _run(general_exception_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        assert body["error"]["request_id"] == "gen-req-1"

    def test_no_request_id_when_absent(self):
        req = _make_request()
        exc = RuntimeError("boom")
        resp = _run(general_exception_handler(req, exc))
        body = json.loads(bytes(resp.body).decode())
        assert "request_id" not in body["error"]

    def test_different_exception_types(self):
        """Handler should work for any Exception subclass."""
        for exc_cls in [ValueError, TypeError, KeyError, IOError]:
            req = _make_request()
            exc = exc_cls("test error")
            resp = _run(general_exception_handler(req, exc))
            assert resp.status_code == 500
            body = json.loads(bytes(resp.body).decode())
            assert body["error"]["code"] == "INTERNAL_ERROR"
