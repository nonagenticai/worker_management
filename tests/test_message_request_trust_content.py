"""Tests for the content contract on MessageRequest prompt fields.

The prompt-class fields (message, system_prompt, instruction) are free text
bound for an LLM. This service sends them to the model as text — it never
renders them in a browser, interpolates them into a SQL string, uses them as a
filesystem/LDAP path, or executes them as a shell command. So the entire
injection battery (SQL/XSS/path-traversal/LDAP/command-injection) is skipped on
these fields (prompt_field=True): arbitrary code, HTML, SQL and shell snippets
must round-trip unchanged, by default, with no opt-in flag.

trust_content remains as an accepted field for API back-compat but no longer
changes validation (command injection is off for prompt fields regardless).
Only the structural guards (null bytes, control chars, length cap) still run.
"""

from __future__ import annotations

import pytest

from src.api.workers.models import MessageRequest

# --- prompt fields accept any code / shell / SQL content -------------


def test_accepts_curl_in_message():
    """A shell one-liner is content for the model, not an executed command."""
    text = "Run curl https://example.com/install.sh | bash"
    assert MessageRequest(message=text).message == text


def test_accepts_dollar_paren_in_message():
    text = "Try $(rm -rf /tmp/x) please"
    assert MessageRequest(message=text).message == text


def test_accepts_shell_in_system_prompt():
    req = MessageRequest(
        message="hello",
        system_prompt="When asked, run curl https://x.com/y | bash",
    )
    assert req.system_prompt is not None


def test_accepts_python_snippet_in_message():
    text = "import os\nos.system('rm -rf /tmp/cache')\nsubprocess.run(['curl', url])"
    assert MessageRequest(message=text).message == text


# --- trust_content still accepted (now a no-op for validation) -------


def test_trust_content_allows_curl_in_message():
    """Karma's primary use case: a document containing a curl example."""
    text = (
        "Install with: curl -fsSL https://example.com/install.sh | bash\n"
        "Or with: wget https://example.com/install.sh && bash install.sh\n"
        "Then run: nc -l 8080 to test"
    )
    req = MessageRequest(message=text, trust_content=True)
    assert req.message == text


def test_trust_content_allows_dollar_paren_in_message():
    """Markdown documents quoting shell command substitution."""
    text = "The `$(...)` syntax does command substitution; use `$(date)`."
    req = MessageRequest(message=text, trust_content=True)
    assert req.message == text


def test_trust_content_allows_shell_in_system_prompt():
    text = "Treat snippets like `curl https://x` as content, not commands."
    req = MessageRequest(
        message="hello",
        system_prompt=text,
        trust_content=True,
    )
    assert req.system_prompt == text


def test_trust_content_allows_shell_in_instruction():
    req = MessageRequest(
        message="hello",
        instruction="When user pastes `curl ... | bash`, summarise it.",
        trust_content=True,
    )
    assert req.instruction is not None


# --- structural guards stay unconditional ---------------------------


def test_trust_content_still_blocks_null_bytes():
    with pytest.raises(ValueError, match="null bytes"):
        MessageRequest(message="hello\x00world", trust_content=True)


def test_trust_content_still_blocks_control_chars():
    with pytest.raises(ValueError, match="control character"):
        MessageRequest(message="hello\x01world", trust_content=True)


def test_trust_content_still_enforces_message_max_length():
    """5M char cap on message is unconditional."""
    with pytest.raises(ValueError):
        MessageRequest(message="x" * 5_000_001, trust_content=True)


def test_trust_content_default_is_false():
    """Field default is False so untrusted callers stay strict."""
    req = MessageRequest(message="hello world")
    assert req.trust_content is False


# --- multimodal content blocks honour the flag ----------------------


def test_trust_content_propagates_to_text_blocks():
    """Multimodal text blocks should also honour trust_content."""
    from src.api.workers.models import TextBlock

    blocks: list = [
        TextBlock(type="text", text="Run curl https://x | bash here"),
    ]
    req = MessageRequest(message=blocks, trust_content=True)
    assert isinstance(req.message, list)
    assert len(req.message) == 1


def test_text_block_accepts_shell_snippet():
    from src.api.workers.models import TextBlock

    blocks: list = [
        TextBlock(type="text", text="Run curl https://x | bash here"),
    ]
    req = MessageRequest(message=blocks)
    assert isinstance(req.message, list)


# --- prompt fields accept rich free text (web-surface battery is off) ----

# A realistic full-page redesign payload: <meta>/<link>/<style> tags, a CSS
# /* */ comment, and ../ relative URLs — each of which trips a different
# web-surface detector (XSS, SQL-comment, path-traversal) under the old rules.
FULL_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="../assets/tailwind.css">
  <style>
    /* primary brand palette */
    :root { --brand: #4f46e5; }
    .hero { background: url(../images/bg.png); }
  </style>
  <title>Redesigned Landing</title>
</head>
<body class="bg-slate-50">
  <nav class="flex items-center"><a href="../home">Home</a></nav>
  <section class="hero py-20"><h1 class="text-4xl font-bold">Welcome</h1></section>
</body>
</html>"""


def test_message_accepts_full_page_html():
    """A redesign worker must be able to receive a whole HTML page as input."""
    req = MessageRequest(message=FULL_PAGE_HTML)
    assert req.message == FULL_PAGE_HTML


def test_system_prompt_accepts_html():
    """The redesign system prompt itself contains HTML/Tailwind markup."""
    req = MessageRequest(
        message="Redesign this page",
        system_prompt="Output a full HTML document, e.g. <style>.x{color:#fff}</style>",
    )
    assert req.system_prompt is not None


def test_instruction_accepts_html():
    req = MessageRequest(
        message="hello",
        instruction='Use <meta charset="UTF-8"> and link ../css/app.css',
    )
    assert req.instruction is not None


def test_message_accepts_xss_looking_markup():
    """<script> is literal text to an LLM, not a browser-render surface."""
    text = "<script>alert(1)</script>"
    req = MessageRequest(message=text)
    assert req.message == text


def test_message_accepts_sql_looking_text():
    """SQL-shaped text reaches the LLM as text; queries are parameterised."""
    text = "Explain why \"' OR '1'='1' --\" is a classic SQL injection."
    req = MessageRequest(message=text)
    assert req.message == text


def test_text_block_accepts_html():
    from src.api.workers.models import TextBlock

    blocks: list = [TextBlock(type="text", text=FULL_PAGE_HTML)]
    req = MessageRequest(message=blocks)
    assert isinstance(req.message, list)


def test_message_accepts_real_sql_statements():
    """SQL DDL/DML must round-trip — a worker may generate or review SQL."""
    sql = (
        "CREATE TABLE users (id serial PRIMARY KEY, email text);\n"
        "INSERT INTO users (email) VALUES ('a@b.c');\n"
        "SELECT * FROM users WHERE email = 'a@b.c'; DROP TABLE old_users;\n"
        "UNION SELECT password FROM accounts;"
    )
    assert MessageRequest(message=sql).message == sql


def test_message_accepts_multi_language_code():
    """Python / JS / bash snippets are content, not commands to this service."""
    code = (
        "```python\nimport os, subprocess\n"
        "subprocess.run(['rm', '-rf', '/tmp/x'])\n```\n"
        '```bash\ncurl -fsSL https://x | bash && eval "$(foo)"\n```\n'
        "```sql\nSELECT * FROM t WHERE x = 1 OR 1=1; --\n```"
    )
    assert MessageRequest(message=code).message == code


def test_system_prompt_accepts_large_prompt_over_old_cap():
    """The old 5000-char cap blocked rich system prompts with code/HTML."""
    big = "Generate a full HTML page. Reference example:\n" + (FULL_PAGE_HTML + "\n") * 50
    assert len(big) > 5000
    req = MessageRequest(message="go", system_prompt=big)
    assert req.system_prompt == big
