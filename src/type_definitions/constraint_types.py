"""String constraint types for API validation and OpenAPI documentation.

These types provide consistent string validation across the API by:
1. Documenting constraints in OpenAPI schema (pattern, min/max length)
2. Blocking dangerous control characters that could cause security issues
3. Allowing legitimate Unicode content including newlines and tabs

The SAFE_STRING_PATTERN rejects only the most dangerous control characters
(null bytes, vertical tabs, form feeds, etc.) while allowing normal text.
"""

from typing import Annotated

from pydantic import Field


def validate_identifier(value: str, field_name: str = "Field") -> str:
    """Validate and normalise a free-text identifier (name/id) string.

    Single source of truth for the "non-empty, no null bytes, stripped" check
    that was previously copy-pasted across ~10 models and endpoints. Error
    messages keep the ``"cannot be empty"`` / ``"cannot contain null bytes"``
    substrings that existing tests assert on.

    Args:
        value: The raw string to validate.
        field_name: Human-readable field label used in error messages.

    Returns:
        The stripped string.

    Raises:
        ValueError: If empty/whitespace-only or containing a null byte.
    """
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    if "\x00" in value:
        raise ValueError(f"{field_name} cannot contain null bytes")
    return value.strip()


# Pattern that blocks dangerous control characters but allows:
# - Normal printable characters
# - Newlines (\n = \x0A), carriage returns (\r = \x0D), tabs (\t = \x09)
# Blocks: \x00-\x08 (null, SOH-BS), \x0B (VT), \x0C (FF), \x0E-\x1F (SO-US), \x7F (DEL)
SAFE_STRING_PATTERN = r"^[^\x00-\x08\x0B\x0C\x0E-\x1F\x7F]*$"

# Base SafeString - no length constraints, just pattern validation
SafeString = Annotated[str, Field(pattern=SAFE_STRING_PATTERN)]

# Short strings for identifiers, names, etc. (1-256 chars)
SafeStringShort = Annotated[str, Field(min_length=1, max_length=256, pattern=SAFE_STRING_PATTERN)]

# Medium strings for descriptions, prompts, etc. (1-1000 chars)
SafeStringMedium = Annotated[str, Field(min_length=1, max_length=1000, pattern=SAFE_STRING_PATTERN)]

# Long strings for instructions, system prompts, etc. (1-10000 chars)
SafeStringLong = Annotated[str, Field(min_length=1, max_length=10000, pattern=SAFE_STRING_PATTERN)]

# Code strings - allow empty, max 50000 chars for Python code content
SafeCodeString = Annotated[str, Field(max_length=50000, pattern=SAFE_STRING_PATTERN)]
