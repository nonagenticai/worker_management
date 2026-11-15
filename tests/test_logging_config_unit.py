"""Unit tests for logging_config.py."""

import logging
import os
from unittest.mock import MagicMock, patch

from src.logging_config import add_service_context, add_trace_context, configure_logging, get_logger


class TestAddTraceContext:
    """Tests for add_trace_context processor."""

    def test_adds_trace_ids_when_otel_available(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_ctx = MagicMock()
        mock_ctx.is_valid = True
        mock_ctx.trace_id = 0x1234567890ABCDEF
        mock_ctx.span_id = 0xFEDCBA09
        mock_ctx.trace_flags = 1
        mock_span.get_span_context.return_value = mock_ctx

        with (
            patch("src.logging_config.OTEL_AVAILABLE", True),
            patch("src.logging_config.trace") as mock_trace,
        ):
            mock_trace.get_current_span.return_value = mock_span
            event_dict: dict = {}
            result = add_trace_context(MagicMock(), "info", event_dict)
            assert "trace_id" in result
            assert "span_id" in result
            assert "trace_flags" in result

    def test_no_trace_when_otel_unavailable(self):
        with patch("src.logging_config.OTEL_AVAILABLE", False):
            event_dict: dict = {}
            result = add_trace_context(MagicMock(), "info", event_dict)
            assert "trace_id" not in result

    def test_no_trace_when_span_not_recording(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with (
            patch("src.logging_config.OTEL_AVAILABLE", True),
            patch("src.logging_config.trace") as mock_trace,
        ):
            mock_trace.get_current_span.return_value = mock_span
            event_dict: dict = {}
            result = add_trace_context(MagicMock(), "info", event_dict)
            assert "trace_id" not in result

    def test_no_trace_when_context_invalid(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_ctx = MagicMock()
        mock_ctx.is_valid = False
        mock_span.get_span_context.return_value = mock_ctx

        with (
            patch("src.logging_config.OTEL_AVAILABLE", True),
            patch("src.logging_config.trace") as mock_trace,
        ):
            mock_trace.get_current_span.return_value = mock_span
            event_dict: dict = {}
            result = add_trace_context(MagicMock(), "info", event_dict)
            assert "trace_id" not in result


class TestAddServiceContext:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_SERVICE_NAME", None)
            os.environ.pop("ENVIRONMENT", None)
            event_dict: dict = {}
            result = add_service_context(MagicMock(), "info", event_dict)
            assert result["service"] == "worker-management-service"
            assert result["environment"] == "development"

    def test_from_env(self):
        with patch.dict(
            os.environ,
            {
                "OTEL_SERVICE_NAME": "custom",
                "ENVIRONMENT": "prod",
            },
        ):
            event_dict: dict = {}
            result = add_service_context(MagicMock(), "info", event_dict)
            assert result["service"] == "custom"
            assert result["environment"] == "prod"


class TestConfigureLogging:
    def test_json_format(self):
        configure_logging(log_level="INFO", json_format=True)
        root = logging.getLogger()
        assert root.handlers  # Should have at least one handler

    def test_console_format(self):
        configure_logging(log_level="DEBUG", json_format=False)
        root = logging.getLogger()
        assert root.handlers


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert logger is not None
