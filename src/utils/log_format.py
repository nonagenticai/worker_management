"""JSON log formatter, in a module with NO import side effects.

It lives here rather than in ``api/app_config.py`` because the container starts
``src.api.worker_management_api``, which must install this formatter at import
time — and importing ``app_config`` to get it would also run that module's
``create_secure_app()`` and its own ``configure_logging()``. A logging helper
must be importable without building an application.

Shape matches the rest of the fleet (w2v_orchestrator, requirements) so one
Loki/alloy parser covers all three.
"""

from __future__ import annotations

import datetime
import json
import logging


class JsonLogFormatter(logging.Formatter):
    """Render a record as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as a single-line JSON object."""
        payload: dict[str, object] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            # Inside the object, never as trailing unprefixed traceback lines --
            # that shape is what made this service's log unattributable.
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)
