"""Workflow routing pointer API package.

Exposes the ``router`` that registers the
``/api/workflow-routing/*`` endpoints used by n8n entry workflows and the
canary blue/green rollback monitor (Improvement Plan 4 / Lane D).
"""

from src.api.workflow_routing.routes import router

__all__ = ["router"]
