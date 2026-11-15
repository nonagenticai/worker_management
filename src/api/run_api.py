"""API server startup script.

This module serves as the entry point for running the Worker Management API server.
It configures the Python path and starts the uvicorn server with the FastAPI application.
"""

import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import uvicorn

if __name__ == "__main__":
    """Start the API server with uvicorn.

    Runs the worker management API on host 0.0.0.0:8887 with auto-reload enabled
    for development purposes.
    """
    uvicorn.run("src.api.worker_management_api:app", host="0.0.0.0", port=8887, reload=True)  # nosec B104 — container requires binding to all interfaces  # noqa: S104
