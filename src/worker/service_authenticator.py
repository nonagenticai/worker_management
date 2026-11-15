"""Service-to-service authentication using license tokens.

This module provides authentication for service-to-service calls,
using license tokens from environment variables.
"""

import logging
import os

logger = logging.getLogger(__name__)


class ServiceAuthenticator:
    """Authenticator that uses license tokens for service-to-service auth."""

    def __init__(self) -> None:
        """Initialize the authenticator."""
        self._token: str | None = None

    async def get_token(self) -> str:
        """Get the license token for service-to-service authentication.

        Returns:
            License token from environment variable.

        Raises:
            RuntimeError: If no token is configured.
        """
        if self._token:
            return self._token

        # Try different environment variable names
        token = (
            os.getenv("CODING_LICENSE_TOKEN")
            or os.getenv("SERVICE_LICENSE_TOKEN")
            or os.getenv("LICENSE_TOKEN")
        )

        if not token:
            raise RuntimeError(
                "No license token configured. Set CODING_LICENSE_TOKEN, "
                "SERVICE_LICENSE_TOKEN, or LICENSE_TOKEN environment variable."
            )

        self._token = token
        logger.debug("Using license token for service authentication")
        return token


# Singleton instance
_authenticator: ServiceAuthenticator | None = None


def get_service_authenticator() -> ServiceAuthenticator:
    """Get the service authenticator singleton.

    Returns:
        ServiceAuthenticator instance.
    """
    global _authenticator  # noqa: PLW0603 -- lazy-init singleton
    if _authenticator is None:
        _authenticator = ServiceAuthenticator()
    return _authenticator
