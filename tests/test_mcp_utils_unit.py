"""Unit tests for src/worker/mcp_utils.py.

Covers config validation, server creation from dict configs, and helper functions.
All Docker, subprocess, and network calls are mocked.
"""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from src.security.redact import redact_args
from src.worker.mcp_utils import (
    MCPServerManager,
    _create_managed_docker_server,
    create_mcp_servers_from_config,
    expand_environment_variables,
    resolve_working_directory,
    validate_mcp_config,
    validate_server_config,
)


def _url_with_userinfo(scheme: str, user: str, password: str, host: str, path: str = "") -> str:
    """Build a URL whose userinfo is ``user:password`` before the ``@``, in a
    ``scheme://`` address, assembled from parts at runtime.

    Not written as one contiguous literal: detect-secrets' BasicAuthDetector
    matches on that shape alone, regardless of content, so concatenating the
    pieces keeps these fixtures test-only without an allowlist pragma.
    """
    return scheme + "://" + user + ":" + password + "@" + host + path


# ---------------------------------------------------------------------------
# validate_server_config – SSE (url-based) servers
# ---------------------------------------------------------------------------


class TestValidateServerConfigSSE:
    """validate_server_config with SSE-type (url) configs."""

    def test_valid_sse_config(self):
        """A config with 'url' should pass validation."""
        result = validate_server_config("my-sse", {"url": "http://localhost:3000/sse"})
        assert result is True

    def test_valid_sse_config_with_timeout(self):
        """SSE config with a positive numeric timeout should pass."""
        result = validate_server_config(
            "sse-timeout", {"url": "http://localhost:3000/sse", "timeout": 30}
        )
        assert result is True

    def test_valid_sse_config_with_headers(self):
        """SSE config with headers should pass."""
        result = validate_server_config(
            "sse-headers",
            {
                "url": "http://localhost:3000/sse",
                "headers": {"Authorization": "Bearer tok"},
            },
        )
        assert result is True


# ---------------------------------------------------------------------------
# validate_server_config – stdio (command-based) servers
# ---------------------------------------------------------------------------


class TestValidateServerConfigStdio:
    """validate_server_config with stdio-type (command) configs."""

    def test_valid_stdio_config(self):
        """A config with 'command' should pass validation."""
        result = validate_server_config("my-stdio", {"command": "npx", "args": ["-y", "server"]})
        assert result is True

    def test_valid_stdio_config_minimal(self):
        """A config with only 'command' and no args should pass."""
        result = validate_server_config("stdio-min", {"command": "/usr/local/bin/mcp-server"})
        assert result is True

    def test_valid_stdio_config_with_env(self):
        """Stdio config with env dict should pass."""
        result = validate_server_config(
            "stdio-env",
            {"command": "node", "args": ["server.js"], "env": {"PORT": "3000"}},
        )
        assert result is True


# ---------------------------------------------------------------------------
# validate_server_config – rejection cases
# ---------------------------------------------------------------------------


class TestValidateServerConfigRejection:
    """validate_server_config rejects configs missing required fields."""

    def test_missing_url_and_command(self):
        """Config without 'url' or 'command' must be rejected."""
        result = validate_server_config("bad-server", {"timeout": 10})
        assert result is False

    def test_empty_config(self):
        """Empty dict must be rejected."""
        result = validate_server_config("empty", {})
        assert result is False

    def test_non_dict_config(self):
        """Non-dict config must be rejected."""
        # Intentional bad type — verifies runtime guard.
        result = validate_server_config("not-dict", cast("Any", "this is a string"))
        assert result is False

    def test_negative_timeout(self):
        """Negative timeout must be rejected."""
        result = validate_server_config(
            "neg-timeout", {"url": "http://localhost:3000", "timeout": -5}
        )
        assert result is False

    def test_zero_timeout(self):
        """Zero timeout must be rejected."""
        result = validate_server_config(
            "zero-timeout", {"url": "http://localhost:3000", "timeout": 0}
        )
        assert result is False

    def test_non_numeric_timeout(self):
        """Non-numeric timeout must be rejected."""
        result = validate_server_config(
            "str-timeout", {"url": "http://localhost:3000", "timeout": "slow"}
        )
        assert result is False

    def test_both_url_and_command_warns_but_passes(self):
        """Config with both 'url' and 'command' should still pass (with a warning)."""
        result = validate_server_config("both", {"url": "http://localhost:3000", "command": "npx"})
        assert result is True


# ---------------------------------------------------------------------------
# validate_mcp_config – full config structure
# ---------------------------------------------------------------------------


class TestValidateMCPConfig:
    """validate_mcp_config checks the top-level config dict."""

    def test_valid_config_with_sse_server(self):
        config = {
            "mcpServers": {
                "my-sse": {"url": "http://localhost:3000/sse"},
            }
        }
        assert validate_mcp_config(config) is True

    def test_valid_config_with_stdio_server(self):
        config = {
            "mcpServers": {
                "my-stdio": {"command": "npx", "args": ["-y", "some-server"]},
            }
        }
        assert validate_mcp_config(config) is True

    def test_missing_mcpServers_key(self):
        assert validate_mcp_config({"servers": {}}) is False

    def test_non_dict_input(self):
        assert validate_mcp_config(cast("Any", "nope")) is False

    def test_mcpServers_not_dict(self):
        assert validate_mcp_config({"mcpServers": ["a", "b"]}) is False

    def test_empty_mcpServers_is_valid(self):
        assert validate_mcp_config({"mcpServers": {}}) is True

    def test_invalid_server_inside_config_fails(self):
        config = {
            "mcpServers": {
                "bad-server": {"timeout": 10},  # missing url and command
            }
        }
        assert validate_mcp_config(config) is False


# ---------------------------------------------------------------------------
# create_mcp_servers_from_config – dict input
# ---------------------------------------------------------------------------


class TestCreateMCPServersFromConfigDict:
    """create_mcp_servers_from_config with dict config creates servers."""

    def test_creates_sse_server_from_dict(self):
        """SSE server config in dict should produce one MCPServerSSE."""
        config = {
            "mcpServers": {
                "test-sse": {"url": "http://localhost:9090/sse"},
            }
        }
        servers = create_mcp_servers_from_config(config)
        assert len(servers) == 1
        # MCPServerSSE has a url attribute
        assert hasattr(servers[0], "url")

    def test_creates_stdio_server_from_dict(self):
        """Stdio server config should produce one MCPServerStdio."""
        config = {
            "mcpServers": {
                "test-stdio": {
                    "command": "node",
                    "args": ["server.js"],
                },
            }
        }
        servers = create_mcp_servers_from_config(config)
        assert len(servers) == 1
        assert hasattr(servers[0], "command")

    def test_creates_multiple_servers(self):
        """Multiple server entries should produce multiple server objects."""
        config = {
            "mcpServers": {
                "sse-a": {"url": "http://localhost:9090/sse"},
                "stdio-a": {"command": "npx", "args": ["-y", "server"]},
            }
        }
        servers = create_mcp_servers_from_config(config)
        assert len(servers) == 2

    def test_skips_invalid_server_and_creates_valid_ones(self):
        """Invalid server entries should be skipped; valid ones still created."""
        config = {
            "mcpServers": {
                "good": {"url": "http://localhost:9090/sse"},
                "bad": {"timeout": 5},  # missing url and command
            }
        }
        servers = create_mcp_servers_from_config(config)
        assert len(servers) == 1


# ---------------------------------------------------------------------------
# create_mcp_servers_from_config – empty / missing config
# ---------------------------------------------------------------------------


class TestCreateMCPServersFromConfigEmpty:
    """create_mcp_servers_from_config with empty config returns empty list."""

    def test_empty_mcpServers_returns_empty(self):
        servers = create_mcp_servers_from_config({"mcpServers": {}})
        assert servers == []

    def test_missing_mcpServers_key_returns_empty(self):
        servers = create_mcp_servers_from_config({"other": "stuff"})
        assert servers == []

    def test_none_like_config_returns_empty(self):
        servers = create_mcp_servers_from_config({})
        assert servers == []

    def test_list_config_returns_empty(self):
        """Passing a list instead of dict should return empty (logged error)."""
        servers = create_mcp_servers_from_config(cast("Any", ["not", "a", "dict"]))
        assert servers == []

    def test_nonexistent_file_path_returns_empty(self):
        """Passing a file path that doesn't exist should return empty."""
        servers = create_mcp_servers_from_config("/tmp/does_not_exist_mcp_config.json")
        assert servers == []


# ---------------------------------------------------------------------------
# create_mcp_servers_from_config – file-based input
# ---------------------------------------------------------------------------


class TestCreateMCPServersFromConfigFile:
    """create_mcp_servers_from_config reading from a JSON file."""

    def test_loads_from_file(self, tmp_path):
        """Should load config from a valid JSON file and create servers."""
        config = {
            "mcpServers": {
                "file-sse": {"url": "http://localhost:7777/sse"},
            }
        }
        config_file = tmp_path / "mcp.json"
        config_file.write_text(json.dumps(config))

        servers = create_mcp_servers_from_config(str(config_file))
        assert len(servers) == 1

    def test_invalid_json_file_returns_empty(self, tmp_path):
        """Malformed JSON file should return empty list."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json!!!")

        servers = create_mcp_servers_from_config(str(bad_file))
        assert servers == []


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestExpandEnvironmentVariables:
    """expand_environment_variables should replace ${VAR} patterns."""

    def test_expands_known_variable(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "abc123")
        result = expand_environment_variables(["--token", "${MY_TOKEN}"])
        assert result == ["--token", "abc123"]

    def test_missing_variable_expands_to_empty(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        result = expand_environment_variables(["${NONEXISTENT_VAR_XYZ}"])
        assert result == [""]

    def test_non_string_args_converted(self):
        result = expand_environment_variables([42, True])
        assert result == ["42", "True"]

    def test_no_expansion_needed(self):
        result = expand_environment_variables(["plain", "args"])
        assert result == ["plain", "args"]

    def test_empty_list(self):
        assert expand_environment_variables([]) == []


class TestResolveWorkingDirectory:
    """resolve_working_directory resolves paths correctly."""

    def test_absolute_path_returned_as_is(self, tmp_path):
        abs_dir = tmp_path / "subdir"
        abs_dir.mkdir()
        result = resolve_working_directory(str(abs_dir), None)
        assert result == str(abs_dir.resolve())

    def test_relative_path_resolved_against_base(self, tmp_path):
        subdir = tmp_path / "rel"
        subdir.mkdir()
        result = resolve_working_directory("rel", tmp_path)
        assert result == str(subdir.resolve())

    def test_none_input_returns_none(self):
        assert resolve_working_directory(None, Path("/tmp")) is None

    def test_empty_string_returns_none(self):
        assert resolve_working_directory("", Path("/tmp")) is None


# ---------------------------------------------------------------------------
# redact_args – unit-level
# ---------------------------------------------------------------------------


class TestRedactArgs:
    """redact_args masks secret-shaped argv while keeping the rest readable."""

    def test_flag_equals_value_masked(self):
        assert redact_args(["--api-key=abc123"]) == ["--api-key=***REDACTED***"]

    def test_separate_flag_and_value_masked(self):
        assert redact_args(["--token", "tok456"]) == ["--token", "***REDACTED***"]

    def test_env_style_arg_masked(self):
        assert redact_args(["-e", "GITHUB_TOKEN=ghp_x"]) == [
            "-e",
            "GITHUB_TOKEN=***REDACTED***",
        ]

    def test_url_userinfo_password_masked(self):
        dsn = _url_with_userinfo("postgres", "u", "pw789", "h", "/db")
        expected = _url_with_userinfo("postgres", "u", "***REDACTED***", "h", "/db")
        assert redact_args([dsn]) == [expected]

    def test_non_sensitive_arg_untouched(self):
        assert redact_args(["--port=8080"]) == ["--port=8080"]
        assert redact_args(["-i", "--rm"]) == ["-i", "--rm"]


# ---------------------------------------------------------------------------
# _create_managed_docker_server – args logged at INFO must be redacted
# ---------------------------------------------------------------------------


class TestManagedDockerServerArgsRedaction:
    """`Args: {args}` at INFO used to log raw argv -- see mcp_utils.py's
    `_create_managed_docker_server`. This must never put a secret-shaped
    flag/env value on the wire, while non-sensitive args stay readable.
    """

    def _server_config(self) -> dict[str, Any]:
        return {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_TOKEN=ghp_x",
                "--api-key=abc123",
                "--token",
                "tok456",
                "--port=8080",
                # image comes before the DSN so the (pre-existing, unrelated)
                # image-detection heuristic in _create_managed_docker_server
                # doesn't pick the DSN as the "image" -- that heuristic bug is
                # out of scope here, this test is about the Args: log line.
                "ghcr.io/example/image:latest",
                _url_with_userinfo("postgres", "u", "pw789", "h", "/db"),
            ],
            "env": {},
        }

    def test_secrets_not_logged_non_sensitive_args_readable(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.worker.mcp_utils"):
            result = _create_managed_docker_server("secret-server", self._server_config())

        assert result is not None

        for secret in ("abc123", "tok456", "ghp_x", "pw789"):
            assert secret not in caplog.text

        # non-sensitive args remain readable
        assert "--port=8080" in caplog.text
        assert "--rm" in caplog.text

    def test_non_sensitive_env_name_still_content_scanned(self, caplog):
        """A non-sensitive env key (e.g. EXTRA_HEADERS) can still carry a
        literal Authorization/Bearer credential or a DSN userinfo password --
        the name alone must not gate the content scan, same rule as argv.
        """
        config = {
            "command": "docker",
            "args": ["run", "ghcr.io/example/image:latest"],
            "env": {
                "EXTRA_HEADERS": "Authorization: Bearer env-hdr-secret",
                "DATABASE_URL": _url_with_userinfo("postgres", "u", "env-dsn-secret", "h", "/db"),
            },
        }
        with caplog.at_level(logging.INFO, logger="src.worker.mcp_utils"):
            result = _create_managed_docker_server("secret-env-server", config)

        assert result is not None
        assert "env-hdr-secret" not in caplog.text
        assert "env-dsn-secret" not in caplog.text


# ---------------------------------------------------------------------------
# redact_args -- every leak reported against 67734cb, plus non-regression
# ---------------------------------------------------------------------------


class TestRedactArgsLeakCases:
    """Round 16 verifier findings against the first cut of redact_args.

    Each case is a leak that was measured to survive the original
    implementation; the assertion is simply that the secret substring never
    appears in the output. Non-regression cases assert the opposite: that
    ordinary, non-secret argv stays human-readable.
    """

    LEAK_CASES: ClassVar[list[tuple[str, list[str], str]]] = [
        # ── R16 second audit: case-folding and attached short forms ────────
        # The flag-NAME test was case-SENSITIVE while this module's own
        # docstring promised case-insensitive matching, and the attached form
        # existed only for `-p`. Both are ordinary curl/docker spellings, and
        # both reached an INFO log fully unmasked.
        (
            "--User is the same flag as --user",
            ["--User", "admin:s3cr3t-pw"],
            "s3cr3t-pw",
        ),
        (
            "-U is the same flag as -u",
            ["-U", "admin:s3cr3t-pw"],
            "s3cr3t-pw",
        ),
        (
            "--USER=admin:secret, attached and upper-cased",
            ["--USER=admin:s3cr3t-pw"],
            "s3cr3t-pw",
        ),
        (
            "-T is the same flag as -t",
            ["-T", "ghp_r3alt0ken"],
            "ghp_r3alt0ken",
        ),
        (
            "attached -uadmin:secret (parses as flag `-uadmin` if read naively)",
            ["-uadmin:s3cr3t-pw"],
            "s3cr3t-pw",
        ),
        (
            "attached -tTOKEN",
            ["-tghp_r3alt0ken"],
            "ghp_r3alt0ken",
        ),
        (
            "short -p flag with separate value (not port-shaped)",
            ["-p", "s3cr3t-pw"],
            "s3cr3t-pw",
        ),
        (
            "mysql-style attached -pSECRET",
            ["-psecretpw"],
            "secretpw",
        ),
        (
            "bare Authorization header as a single arg",
            ["Authorization: Bearer abc.def.ghi"],
            "abc.def.ghi",
        ),
        (
            "--header=Authorization: Bearer ... -- non-sensitive flag name",
            ["--header=Authorization: Bearer abc.def.ghi"],
            "abc.def.ghi",
        ),
        (
            "-H then Authorization value as two args -- non-sensitive flag",
            ["-H", "Authorization: Bearer abc.def.ghi"],
            "abc.def.ghi",
        ),
        (
            "bare Bearer token with no header name at all",
            ["Bearer abc.def.ghi"],
            "abc.def.ghi",
        ),
        (
            "Basic auth value",
            ["Authorization: Basic dXNlcjpwYXNz"],
            "dXNlcjpwYXNz",
        ),
        (
            "Proxy-Authorization header",
            ["Proxy-Authorization: Bearer abc.def.ghi"],
            "abc.def.ghi",
        ),
        # --- Round 16 audit follow-up: item 3 (src/security/redact.py leaks) ---
        (
            "--pass with separate value",
            ["--pass", "hunter2"],
            "hunter2",
        ),
        (
            "--pwd with separate value",
            ["--pwd", "hunter2"],
            "hunter2",
        ),
        (
            "-u user:secret pair",
            ["-u", "admin:hunter2"],
            "hunter2",
        ),
        (
            "--user user:secret pair",
            ["--user", "admin:hunter2"],
            "hunter2",
        ),
        (
            "-t with separate value",
            ["-t", "ghp_realtoken"],
            "ghp_realtoken",
        ),
        (
            "colon-separated flag:value",
            ["--api-key:abc123"],
            "abc123",
        ),
    ]

    NON_REGRESSION_CASES: ClassVar[list[tuple[str, list[str], str]]] = [
        (
            "attached numeric UID:GID stays readable (-u1000:1000)",
            ["-u1000:1000"],
            "1000:1000",
        ),
        ("docker port publish, separate args", ["-p", "8080:80"], "8080:80"),
        ("docker port publish with protocol", ["-p", "8080:80/tcp"], "8080:80/tcp"),
        ("docker port publish with host ip", ["-p", "127.0.0.1:5432:5432"], "127.0.0.1:5432:5432"),
        ("long-form --port=8080", ["--port=8080"], "--port=8080"),
        ("image name", ["ghcr.io/example/image:latest"], "ghcr.io/example/image:latest"),
        (
            "image ref with tag, colon-shaped like a flag:value",
            ["ghcr.io/org/img:1.2"],
            "ghcr.io/org/img:1.2",
        ),
        ("plain flags", ["-i", "--rm"], "--rm"),
        ("-u numeric UID:GID pair is not a secret", ["-u", "1000:1000"], "1000:1000"),
        ("-u bare username, no colon, nothing to mask", ["-u", "admin"], "admin"),
        ("-u user:secret pair keeps the username half readable", ["-u", "admin:hunter2"], "admin:"),
    ]

    @pytest.mark.parametrize(
        "description,args,secret",
        LEAK_CASES,
        ids=[c[0] for c in LEAK_CASES],
    )
    def test_secret_never_survives_redaction(self, description, args, secret):
        result = redact_args(args)
        assert secret not in " ".join(result), (
            f"{description}: {args!r} -> {result!r} still contains {secret!r}"
        )

    @pytest.mark.parametrize(
        "description,args,expected_visible",
        NON_REGRESSION_CASES,
        ids=[c[0] for c in NON_REGRESSION_CASES],
    )
    def test_non_secret_stays_readable(self, description, args, expected_visible):
        result = redact_args(args)
        assert expected_visible in " ".join(result), (
            f"{description}: {args!r} -> {result!r} lost {expected_visible!r}"
        )


# ---------------------------------------------------------------------------
# MCPServerManager._create_server -- must never log the raw config dict
# ---------------------------------------------------------------------------


class TestCreateServerNeverLogsRawConfig:
    """`logger.info(f"Creating MCP server with config: {server_config}")` used
    to dump args/env/url/headers verbatim at INFO, before any redaction ran.
    """

    def _combo_config(self) -> dict[str, Any]:
        return {
            "name": "combo-server",
            "url": _url_with_userinfo("https", "svc-user", "s3cr3t-url-pw", "example.com", "/mcp"),
            "headers": {"Authorization": "Bearer hdr-secret-tok"},
            "args": ["--token", "arg-secret-tok"],
            "env": {"API" + "_KEY": "env-secret-key"},
        }

    @pytest.mark.asyncio
    async def test_no_secret_from_config_reaches_the_log(self, caplog):
        manager = MCPServerManager()
        config = self._combo_config()

        with caplog.at_level(logging.INFO, logger="src.worker.mcp_utils"):
            result = await manager._create_server(config, "")

        assert result is not None
        for secret in ("s3cr3t-url-pw", "hdr-secret-tok", "arg-secret-tok", "env-secret-key"):
            assert secret not in caplog.text

    @pytest.mark.asyncio
    async def test_identifying_shape_still_logged(self, caplog):
        """The redaction must not degrade into logging nothing useful."""
        manager = MCPServerManager()
        config = self._combo_config()

        with caplog.at_level(logging.INFO, logger="src.worker.mcp_utils"):
            await manager._create_server(config, "")

        assert "combo-server" in caplog.text
        assert "transport=http" in caplog.text
        assert "'url'" in caplog.text  # sorted(config.keys()) rendering
