"""Unit tests for worker/agents/tool_conversion.py."""

import pytest
from pydantic import ValidationError

from src.worker.agents.tool_conversion import (
    load_local_tool,
    load_tools_from_directory,
    parse_json_to_tools,
)


class TestParseJsonToTools:
    def test_empty_list(self):
        result = parse_json_to_tools([])
        assert result == []

    def test_invalid_tool_definition(self):
        with pytest.raises((ValidationError, ValueError, TypeError, AttributeError)):
            parse_json_to_tools([{"invalid": "data"}])


class TestLoadLocalTool:
    def test_nonexistent_file(self):
        result = load_local_tool("/nonexistent/path/tool.py")
        assert result is None

    def test_directory_instead_of_file(self, tmp_path):
        result = load_local_tool(tmp_path)
        assert result is None

    def test_file_missing_required_attrs(self, tmp_path):
        tool_file = tmp_path / "bad_tool.py"
        tool_file.write_text("x = 1\n")
        result = load_local_tool(tool_file)
        assert result is None


class TestLoadToolsFromDirectory:
    def test_nonexistent_directory(self):
        result = load_tools_from_directory("/nonexistent/dir")
        assert result == []

    def test_empty_directory(self, tmp_path):
        result = load_tools_from_directory(tmp_path)
        assert result == []

    def test_skips_dunder_files(self, tmp_path):
        init_file = tmp_path / "__init__.py"
        init_file.write_text("")
        result = load_tools_from_directory(tmp_path)
        assert result == []

    def test_file_not_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        result = load_tools_from_directory(f)
        assert result == []
