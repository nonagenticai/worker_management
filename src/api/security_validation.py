"""Comprehensive security validation module for the Worker Management API.

This module provides robust input validation, sanitization, and security checks
to protect against various attack vectors including SQL injection, XSS, path
traversal, command injection, and other malicious inputs.
"""

import html
import re
import unicodedata
from typing import Any


class SecurityValidationError(ValueError):
    """Exception raised when security validation fails."""


def detect_sql_injection(text: str) -> bool:
    """Detect potential SQL injection patterns in text.

    Args:
        text: Text to analyze for SQL injection patterns

    Returns:
        True if potential SQL injection detected, False otherwise
    """
    if not text:
        return False

    # Convert to lowercase for pattern matching
    text_lower = text.lower()

    # Check strong SQL injection patterns first, regardless of language context
    strong_sql_patterns = [
        r"union\s+select\s+",
        r"'\s*(or|and)\s*'?\w*'?\s*=\s*'?\w*'?",
        r";\s*(drop|delete|insert|update|create|alter)\s+",
        r"exec\s*\(|execute\s*\(|sp_|xp_",
        r"information_schema\.|sysobjects\.|sys\.tables",
        r"waitfor\s+delay|benchmark\s*\(|sleep\s*\(",
        r"'\s*;\s*(select|insert|update|delete|create|drop|alter)",
    ]
    for pattern in strong_sql_patterns:
        if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
            return True

    # Check if this looks like Python code - be much more lenient
    python_indicators = [
        "def ",
        "class ",
        "import ",
        "from ",
        "return",
        "print",
        "if ",
        "for ",
        "while ",
        "# ",
        "else:",
        "elif ",
        "lambda",
        "try:",
        "except:",
        "finally:",
        "with ",
        "as ",
        "pass",
        "__name__",
        "__main__",
        "self.",
        ".get(",
        ".format(",
        "f'",
        'f"',
        '"""',
        "'''",
        "True",
        "False",
        "None",
    ]

    is_python_code = any(
        indicator in text_lower or indicator in text for indicator in python_indicators
    )

    if is_python_code:
        # For Python code, only check for the most dangerous SQL injection patterns
        # that would be clearly malicious even in code context
        sql_patterns = [
            # Only check for obvious SQL injection with command execution that's clearly not Python
            r"';\s*drop\s+table\s+\w+\s*;",
            r"';\s*delete\s+from\s+\w+\s*;",
            r"';\s*truncate\s+table\s+\w+\s*;",
            # Clear SQL injection attempts in strings with classic patterns
            r"'\s*or\s*'1'\s*=\s*'1'\s*--",
            r"'\s*union\s+select\s+.*password",
            r"'\s*union\s+select\s+.*from\s+users",
            # Only very obvious SQL injection with database schema access
            r"information_schema\.tables",
            r"sys\.tables",
        ]
    else:
        # For non-code input, use more comprehensive SQL injection detection
        sql_patterns = [
            # Classic SQL injection patterns
            r"'\s*(or|and)\s*'?\w*'?\s*=\s*'?\w*'?",
            r"'\s*(or|and)\s*\d+\s*=\s*\d+",
            r"union\s+select\s+.*from\s+",
            r";\s*(drop|delete|insert|update|create|alter)\s+",
            r"exec\s*\(|execute\s*\(|sp_|xp_",
            r"information_schema\.|sysobjects\.|sys\.tables",
            r"waitfor\s+delay|benchmark\s*\(|sleep\s*\(",
            # SQL comment patterns that indicate injection
            r"--\s*$|/\*.*\*/",
            # Hex encoding attacks
            r"0x[0-9a-f]{4,}",
            # SQL keywords in suspicious context
            r"'\s*;\s*(select|insert|update|delete|create|drop|alter)",
        ]

    for pattern in sql_patterns:
        if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
            return True

    return False


def detect_xss(text: str) -> bool:
    """Detect potential XSS (Cross-Site Scripting) patterns in text.

    Args:
        text: Text to analyze for XSS patterns

    Returns:
        True if potential XSS detected, False otherwise
    """
    if not text:
        return False

    # Convert to lowercase for pattern matching
    text_lower = text.lower()

    # Check if this looks like Python code - be lenient for XSS detection
    python_indicators = [
        "def ",
        "class ",
        "import ",
        "from ",
        "return",
        "print",
        "if ",
        "for ",
        "while ",
        "# ",
        "else:",
        "elif ",
        "lambda",
        "try:",
        "except:",
        "finally:",
        "with ",
        "as ",
        "pass",
        "__name__",
        "__main__",
        "self.",
        ".get(",
        ".format(",
        "f'",
        'f"',
        '"""',
        "'''",
        "True",
        "False",
        "None",
    ]

    is_python_code = any(
        indicator in text_lower or indicator in text for indicator in python_indicators
    )

    if is_python_code:
        # For Python code, only check for the most obvious XSS patterns
        xss_patterns = [
            # Very obvious script tags
            r"<script[^>]*>.*?</script>",
            r"<script[^>]*>",
            r"</script>",
            # Direct javascript/vbscript URLs
            r"javascript:\s*alert",
            r"vbscript:\s*msgbox",
            r"data:text/html,.*<script",
            # Obvious HTML injection
            r"<iframe[^>]*src[^>]*>",
            r"<embed[^>]*src[^>]*>",
            # URL encoded script tags
            r"%3cscript",
            r"&lt;script",
        ]
    else:
        # For non-code input, use more comprehensive XSS detection
        xss_patterns = [
            # Script tags
            r"<script[^>]*>.*?</script>",
            r"<script[^>]*>",
            r"</script>",
            # Event handlers - more specific to avoid false positives
            r"on(click|load|error|mouseover|focus|blur|submit)\s*=",
            r"javascript:",
            r"vbscript:",
            r"data:text/html",
            # HTML tags commonly used for XSS
            r"<(iframe|embed|object|applet|meta|link|style)[^>]*>",
            r"<img[^>]*onerror[^>]*>",
            r"<svg[^>]*onload[^>]*>",
            r"<body[^>]*onload[^>]*>",
            # URL encoded XSS attempts
            r"%3c%73%63%72%69%70%74",  # <script
            r"%3cscript",
            r"&lt;script",
            # Expression and eval patterns
            r"expression\s*\(",
            r"eval\s*\(",
            r"settimeout\s*\(",
            r"setinterval\s*\(",
        ]

    for pattern in xss_patterns:
        if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
            return True

    return False


def detect_path_traversal(text: str) -> bool:
    """Detect potential path traversal patterns in text.

    Args:
        text: Text to analyze for path traversal patterns

    Returns:
        True if potential path traversal detected, False otherwise
    """
    if not text:
        return False

    # Path traversal patterns
    traversal_patterns = [
        r"\.\./",
        r"\.\.\\",
        r"\.\.%2f",
        r"\.\.%5c",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
        r"file://",
        r"/etc/passwd",
        r"/etc/shadow",
        r"\\windows\\system32",
        r"boot\.ini",
        r"web\.config",
    ]

    text_lower = text.lower()
    return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in traversal_patterns)


def detect_command_injection(text: str) -> bool:
    """Detect potential command injection patterns in text.

    Args:
        text: Text to analyze for command injection patterns

    Returns:
        True if potential command injection detected, False otherwise
    """
    if not text:
        return False

    # Long multiline text is document content (e.g. markdown with code blocks,
    # LLM outputs), not injected commands. Real command injection payloads are
    # short single-line inputs.
    if len(text) > 500 and "\n" in text:
        return False

    # Command injection patterns - more specific to avoid false positives
    command_patterns = [
        # Shell metacharacters in dangerous contexts
        r"[;&]\s*(rm|del|format|cat|wget|curl|nc|netcat|python|perl|ruby|php|bash|sh|cmd|powershell)",
        r"\|\s*(rm|del|format|cat|wget|curl|nc|netcat|python|perl|ruby|php|bash|sh|cmd|powershell)",
        r"`[^`]*`",  # Backticks for command substitution
        r"\$\([^)]+\)",  # Command substitution $()
        r"\$\{[^}]+\}",  # Variable expansion that could be dangerous
        # Dangerous commands with their typical patterns
        r"(rm\s+-rf|del\s+/[qsf]|format\s+[a-z]:)",
        r"(cat\s+/etc/passwd|type\s+c:\\windows)",
        r"(wget\s+https?://|curl\s+https?://|nc\s+-[elvnup])",
        r"(python\s+-c|perl\s+-e|ruby\s+-e|php\s+-r)",
        r"(exec\s*\(|system\s*\(|shell_exec\s*\(|eval\s*\(|passthru\s*\()",
        r"(&&\s*whoami|;\s*whoami|\|\s*whoami)",
        r"(&&\s*id|;\s*id|\|\s*id)",
        # Path traversal combined with commands
        r"(\.\./){2,}",
        r"(\\\\){2,}",
    ]

    text_lower = text.lower()
    return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in command_patterns)


def detect_ldap_injection(text: str) -> bool:
    """Detect potential LDAP injection patterns in text.

    Args:
        text: Text to analyze for LDAP injection patterns

    Returns:
        True if potential LDAP injection detected, False otherwise
    """
    if not text:
        return False

    # LDAP injection patterns
    ldap_patterns = [
        r"\*\)\(.*=",
        r"\)\(.*\|.*=",
        r"\*\)\(\|",
        r"\)\(cn=",
        r"\*\)\)%00",
        r"admin\*",
        r"\*\)\(userPassword=",
        r"\)\(objectClass=",
    ]

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in ldap_patterns)


def validate_string_security(
    text: str,
    field_name: str,
    max_length: int = 1000,
    allow_html: bool = False,
    strict_mode: bool = True,
    allow_prompt_constructs: bool = False,
    prompt_field: bool = False,
) -> str:
    """Comprehensive security validation for string fields.

    Args:
        text: Text to validate
        field_name: Name of the field being validated (for error messages)
        max_length: Maximum allowed length
        allow_html: Whether to allow HTML content (default: False)
        strict_mode: Whether to use strict validation (default: True)
        allow_prompt_constructs: When True, skip the command-injection
            detector only. Use for free-text LLM prompt fields where
            inline backticks and ``${...}`` placeholders are legitimate
            (markdown code spans, schema-key references, etc). The
            web-surface detectors still run unless ``prompt_field`` is set.
        prompt_field: When True, skip the ENTIRE injection battery
            (SQL/XSS/path-traversal/LDAP/command-injection). Use for free-text
            bound for an LLM — ``message``, ``system_prompt``, ``instruction``,
            task templates — which is sent to the model as text and is never
            rendered in a browser, used in a SQL string, used as a filesystem/
            LDAP path, or executed as a shell command by this service. Those
            detectors guard surfaces the prompt never touches, so they are pure
            false positives here: they reject legitimate code/HTML/SQL/shell
            content the caller wants the model to read (a page with ``<style>``,
            a CSS ``/* */`` comment, a ``../`` URL, ``curl ... | bash``, ``$(...)``,
            ``rm -rf``, ``SELECT ... FROM``). Only the structural guards
            (null-byte, control-char, length) above still run.

    Returns:
        Validated and potentially sanitized text

    Raises:
        SecurityValidationError: If security validation fails
    """
    if not text:
        return text

    # Check for null bytes
    if "\x00" in text:
        raise SecurityValidationError(f"{field_name} contains null bytes")

    # Check length
    if len(text) > max_length:
        raise SecurityValidationError(
            f"{field_name} exceeds maximum length of {max_length} characters"
        )

    # Check for control characters (except common whitespace)
    for i, char in enumerate(text):
        if ord(char) < 32 and char not in "\t\n\r":
            raise SecurityValidationError(
                f"{field_name} contains control character at position {i}: '\\x{ord(char):02x}'"
            )

    if strict_mode:
        # SQL / XSS / path-traversal / LDAP guard surfaces an LLM prompt never
        # touches (browser DOM, SQL string, filesystem path, LDAP filter). For
        # prompt-class free-text fields they are pure false positives — e.g. any
        # page with <style>, a CSS /* */ comment, or a ../ relative URL would be
        # rejected — so skip them when prompt_field is set. The structural guards
        # above (null-byte, control-char, length) always run.
        if not prompt_field:
            # SQL injection detection
            if detect_sql_injection(text):
                raise SecurityValidationError(
                    f"{field_name} contains potential SQL injection patterns"
                )

            # XSS detection (unless HTML is explicitly allowed)
            if not allow_html and detect_xss(text):
                raise SecurityValidationError(f"{field_name} contains potential XSS patterns")

            # Path traversal detection
            if detect_path_traversal(text):
                raise SecurityValidationError(
                    f"{field_name} contains potential path traversal patterns"
                )

        # Command injection detection. An LLM prompt is never executed as a
        # shell by this service — it is passed to the model as text — so for
        # prompt-class fields this detector only blocks legitimate code/shell
        # snippets (curl, $(...), rm -rf, a bash/python one-liner) the caller
        # explicitly wants the model to read. Skip it entirely when
        # prompt_field is set. For non-prompt fields the existing narrower
        # opt-outs (allow_prompt_constructs / system_prompt by name) still
        # apply. (Sandboxing of any agent that actually executes code is the
        # responsibility of the executor, not this input regex.)
        if (
            not prompt_field
            and not allow_prompt_constructs
            and field_name != "system_prompt"
            and detect_command_injection(text)
        ):
            raise SecurityValidationError(
                f"{field_name} contains potential command injection patterns"
            )

        # LDAP injection detection — same web-surface rationale as above.
        if not prompt_field and detect_ldap_injection(text):
            raise SecurityValidationError(
                f"{field_name} contains potential LDAP injection patterns"
            )

    return text


def sanitize_html_content(text: str) -> str:
    """Sanitize HTML content to prevent XSS attacks.

    Args:
        text: HTML content to sanitize

    Returns:
        Sanitized HTML content
    """
    if not text:
        return text

    # HTML escape the content
    sanitized = html.escape(text, quote=True)

    return sanitized


def validate_unicode_string(text: str, field_name: str) -> str:
    """Validate Unicode string for security issues.

    Args:
        text: Unicode text to validate
        field_name: Name of the field being validated

    Returns:
        Validated Unicode text

    Raises:
        SecurityValidationError: If validation fails
    """
    if not text:
        return text

    # Check for dangerous Unicode categories
    for i, char in enumerate(text):
        category = unicodedata.category(char)

        # Reject control characters and format characters
        if category in ("Cc", "Cf"):  # noqa: RUF100, SIM102
            # Allow common whitespace characters
            if char not in "\t\n\r ":
                raise SecurityValidationError(
                    f"{field_name} contains dangerous Unicode character at position {i}: "
                    f"'{char}' (U+{ord(char):04X}, category: {category})"
                )

        # Reject line/paragraph separators
        if category in ("Zl", "Zp"):
            raise SecurityValidationError(
                f"{field_name} contains Unicode separator character at position {i}: "
                f"'{char}' (U+{ord(char):04X}, category: {category})"
            )

        # Reject unassigned characters
        if category == "Cn":
            raise SecurityValidationError(
                f"{field_name} contains unassigned Unicode character at position {i}: "
                f"U+{ord(char):04X}"
            )

    # Check for dangerous Unicode sequences
    dangerous_sequences = [
        "\ufeff",  # Byte Order Mark
        "\u202e",  # Right-to-Left Override
        "\u200e",  # Left-to-Right Mark
        "\u200f",  # Right-to-Left Mark
        "\ufffe",  # Noncharacter
        "\uffff",  # Noncharacter
    ]

    for seq in dangerous_sequences:
        if seq in text:
            raise SecurityValidationError(
                f"{field_name} contains dangerous Unicode sequence: U+{ord(seq):04X}"
            )

    return text


def validate_identifier(
    text: str, field_name: str, max_length: int = 100, allow_unicode: bool = True
) -> str:
    """Validate identifier strings (names, IDs, etc.) for security.

    Args:
        text: Identifier to validate
        field_name: Name of the field being validated
        max_length: Maximum allowed length
        allow_unicode: Whether to allow Unicode characters

    Returns:
        Validated identifier

    Raises:
        SecurityValidationError: If validation fails
    """
    if not text or not text.strip():
        raise SecurityValidationError(f"{field_name} cannot be empty")

    text = text.strip()

    # Check length
    if len(text) > max_length:
        raise SecurityValidationError(
            f"{field_name} exceeds maximum length of {max_length} characters"
        )

    # Check for null bytes
    if "\x00" in text:
        raise SecurityValidationError(f"{field_name} contains null bytes")

    if allow_unicode:
        # Use Unicode-aware validation
        text = validate_unicode_string(text, field_name)

        # Check first character
        if not (text[0].isalnum() or text[0] == "_"):
            raise SecurityValidationError(
                f"{field_name} must start with a letter, digit, or underscore"
            )

        # Check remaining characters
        for i, char in enumerate(text[1:], 1):
            if not (char.isalnum() or char in "_-"):
                raise SecurityValidationError(
                    f"{field_name} contains invalid character at position {i}: '{char}'. "
                    f"Only letters, digits, underscores, and hyphens are allowed"
                )
    # ASCII-only validation
    elif not re.match(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]*$", text):
        raise SecurityValidationError(
            f"{field_name} contains invalid characters. "
            f"Only ASCII letters, digits, underscores, and hyphens are allowed"
        )

    # Check for reserved identifiers
    reserved_names = {
        "null",
        "undefined",
        "none",
        "nil",
        "true",
        "false",
        "admin",
        "root",
        "system",
        "guest",
        "anonymous",
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "exec",
        "execute",
        "eval",
        "function",
        "var",
        "let",
        "const",
    }

    if text.lower() in reserved_names:
        raise SecurityValidationError(f"{field_name} cannot use reserved name: {text}")

    return text


def validate_query_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Validate query parameters for security issues.

    Args:
        params: Dictionary of query parameters

    Returns:
        Validated parameters

    Raises:
        SecurityValidationError: If validation fails
    """
    validated_params = {}

    for key, value in params.items():
        # Validate parameter name
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
            raise SecurityValidationError(f"Invalid query parameter name: {key}")

        # Validate parameter value
        if isinstance(value, str):
            # Check for XSS in query parameters
            if detect_xss(value):
                raise SecurityValidationError(
                    f"Query parameter '{key}' contains potential XSS patterns"
                )

            # Check for SQL injection in query parameters
            if detect_sql_injection(value):
                raise SecurityValidationError(
                    f"Query parameter '{key}' contains potential SQL injection patterns"
                )

            # Limit parameter value length
            if len(value) > 1000:
                raise SecurityValidationError(f"Query parameter '{key}' value is too long")

        validated_params[key] = value

    return validated_params


def is_safe_json_data(data: Any, max_depth: int = 10, current_depth: int = 0) -> bool:
    """Check if JSON data is safe (no excessive nesting or size).

    Args:
        data: JSON data to validate
        max_depth: Maximum allowed nesting depth
        current_depth: Current nesting depth (for recursion)

    Returns:
        True if data is safe, False otherwise
    """
    if current_depth > max_depth:
        return False

    if isinstance(data, dict):
        if len(data) > 1000:  # Limit number of keys
            return False
        for key, value in data.items():
            if not isinstance(key, str) or len(key) > 100:
                return False
            if not is_safe_json_data(value, max_depth, current_depth + 1):
                return False
    elif isinstance(data, list):
        if len(data) > 10000:  # Limit array size
            return False
        for item in data:
            if not is_safe_json_data(item, max_depth, current_depth + 1):
                return False
    elif isinstance(data, str) and len(data) > 100000:  # Limit string length
        return False

    return True
