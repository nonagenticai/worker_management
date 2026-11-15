"""Structured logging configuration with OpenTelemetry trace correlation.

This module configures structlog for JSON-formatted logging with automatic
trace context injection for correlation in Grafana Loki/Tempo.
"""

import logging
import sys
from typing import Any, cast

import structlog

from src.settings import get_settings

# OpenTelemetry trace context injection
try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace: Any = None


def add_trace_context(
    logger: logging.Logger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: dict[str, Any],  # noqa: ARG001, RUF100
) -> dict[str, Any]:
    """Add OpenTelemetry trace context to log records.

    This processor adds trace_id and span_id to every log message,
    enabling correlation between logs and traces in Grafana.
    """
    if OTEL_AVAILABLE and trace:
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            if ctx.is_valid:
                event_dict["trace_id"] = format(ctx.trace_id, "032x")
                event_dict["span_id"] = format(ctx.span_id, "016x")
                event_dict["trace_flags"] = ctx.trace_flags
    return event_dict


def add_service_context(
    logger: logging.Logger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: dict[str, Any],  # noqa: ARG001, RUF100
) -> dict[str, Any]:
    """Add service context to log records."""
    settings = get_settings()
    event_dict["service"] = settings.otel_service_name
    event_dict["environment"] = settings.environment
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_format: bool = True,  # noqa: FBT001, FBT002, RUF100
    include_timestamps: bool = True,  # noqa: ARG001, FBT001, FBT002, RUF100
) -> None:
    """Configure structured logging with trace correlation.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, output JSON; if False, output human-readable format
        include_timestamps: If True, include ISO-format timestamps
    """
    # Get log level from environment or parameter
    env_level = get_settings().log_level
    level = (env_level or log_level).upper()

    # Build processor chain
    processors = [
        # Add log level
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add service context
        add_service_context,
        # Add trace context for correlation
        add_trace_context,
        # Add call site info for debugging
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        # Add exception info
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Add timestamps
        structlog.processors.TimeStamper(fmt="iso"),
        # Process for stdlib integration
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # Configure structlog
    structlog.configure(
        processors=cast("Any", processors),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Choose renderer based on format preference
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )

    # Create formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        # Remove ProcessorFormatter.wrap_for_formatter from foreign_pre_chain
        foreign_pre_chain=cast(
            "Any",
            [
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                add_service_context,
                add_trace_context,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
        ),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Configure root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level))

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog bound logger
    """
    return structlog.get_logger(name)


# Example usage in other modules:
# from src.logging_config import get_logger
# logger = get_logger(__name__)
# logger.info("Processing request", graph_id="abc123", node_count=5)
