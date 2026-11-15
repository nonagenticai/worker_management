"""Test MCP server name validation with Unicode support."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pydantic import ValidationError

from src.api.workers import WorkerCreateRequest

# Test creator ID (UUID format)
TEST_CREATOR_ID = "550e8400-e29b-41d4-a716-446655440000"


class TestMCPServerValidation:
    """Test cases for MCP server name validation."""

    def test_valid_ascii_names(self):
        """Test valid ASCII MCP server names."""
        valid_names = [
            ["filesystem"],
            ["memory", "database"],
            ["test-server"],
            ["test_server"],
            ["server123"],
            ["MCP-Server-2024"],
        ]

        for names in valid_names:
            request = WorkerCreateRequest(worker_name="test_worker", mcp_servers=names)
            assert request.mcp_servers == names

    def test_valid_unicode_names(self):
        """Test valid Unicode MCP server names."""
        valid_unicode_names = [
            ["rïùâ¡"],
            ["café-server"],
            ["データベース"],  # Japanese
            ["服务器"],  # Chinese
            ["сервер"],  # Russian
            ["서버"],  # Korean
            ["server-αβγ"],  # Greek letters
            ["test_ñ_server"],
            ["résumé-db"],
            ["Schöne-Datenbank"],  # German
        ]

        for names in valid_unicode_names:
            request = WorkerCreateRequest(worker_name="test_worker", mcp_servers=names)
            assert request.mcp_servers == names

    def test_invalid_names(self):
        """Test invalid MCP server names."""
        invalid_names = [
            [""],  # Empty string (filtered out, not invalid)
            ["server with spaces"],  # Spaces not allowed
            ["server/with/slashes"],  # Slashes not allowed
            ["server\\with\\backslashes"],  # Backslashes not allowed
            ["server|pipe"],  # Pipe not allowed
            ["server;semicolon"],  # Semicolon not allowed
            ["server:colon"],  # Colon not allowed
            ["server?question"],  # Question mark not allowed
            ["server*asterisk"],  # Asterisk not allowed
            ["server<bracket>"],  # Angle brackets not allowed
            ['server"quote'],  # Quotes not allowed
            ["server'apostrophe"],  # Apostrophes not allowed
            ["server\nwith\nnewline"],  # Newlines not allowed
            ["server\twith\ttab"],  # Tabs not allowed
            ["a" * 101],  # Too long (max 100 chars)
        ]

        for names in invalid_names:
            # Skip empty string test as it's filtered out
            if names == [""]:
                continue

            try:
                WorkerCreateRequest(worker_name="test_worker", mcp_servers=names)
                raise AssertionError(f"Expected ValidationError for {names}")
            except ValidationError as e:
                assert "MCP server name" in str(e)

    def test_empty_and_null_handling(self):
        """Test handling of empty and null values."""
        # None should be accepted
        request = WorkerCreateRequest(worker_name="test_worker", mcp_servers=None)
        assert request.mcp_servers is None

        # Empty list should be accepted
        request = WorkerCreateRequest(worker_name="test_worker", mcp_servers=[])
        assert request.mcp_servers == []

        # List with empty strings should filter them out
        request = WorkerCreateRequest(
            worker_name="test_worker",
            mcp_servers=["", "valid-server", "   ", "another-server"],
        )
        assert request.mcp_servers == ["valid-server", "another-server"]

    def test_mixed_valid_invalid(self):
        """Test mix of valid and invalid names."""
        try:
            WorkerCreateRequest(
                worker_name="test_worker",
                mcp_servers=["valid-server", "invalid server with spaces", "café"],
            )
            raise AssertionError("Expected ValidationError")
        except ValidationError as e:
            assert "invalid server with spaces" in str(e)

    def test_unicode_normalization(self):
        """Test that Unicode names are properly handled."""
        # Test different Unicode representations of the same character
        names = ["café"]  # é can be represented as single char or base + combining
        request = WorkerCreateRequest(worker_name="test_worker", mcp_servers=names)
        assert request.mcp_servers is not None
        assert len(request.mcp_servers[0]) in [4, 5]  # Depending on normalization

    def test_edge_cases(self):
        """Test edge cases for MCP server names."""
        # Valid edge cases
        valid_edge_cases = [
            ["_underscore_start"],
            ["-hyphen-start"],
            ["123numeric"],
            ["UPPERCASE"],
            ["CamelCase"],
            ["snake_case"],
            ["kebab-case"],
            ["server.with.dots"],  # Periods now allowed
        ]

        for names in valid_edge_cases:
            request = WorkerCreateRequest(worker_name="test_worker", mcp_servers=names)
            assert request.mcp_servers == names

        # Invalid edge cases
        invalid_edge_cases = [
            ["with,comma"],  # Comma not allowed
            ["with@at"],  # @ not allowed
            ["with#hash"],  # # not allowed
            ["with$dollar"],  # $ not allowed
            ["with%percent"],  # % not allowed
            ["with&ampersand"],  # & not allowed
            ["with+plus"],  # + not allowed
            ["with=equals"],  # = not allowed
        ]

        for names in invalid_edge_cases:
            try:
                WorkerCreateRequest(worker_name="test_worker", mcp_servers=names)
                raise AssertionError(f"Expected ValidationError for {names}")
            except ValidationError:
                pass  # Expected


if __name__ == "__main__":
    # Run tests
    test = TestMCPServerValidation()

    print("Testing valid ASCII names...")
    test.test_valid_ascii_names()
    print("✓ Valid ASCII names passed")

    print("\nTesting valid Unicode names...")
    test.test_valid_unicode_names()
    print("✓ Valid Unicode names passed")

    print("\nTesting invalid names...")
    test.test_invalid_names()
    print("✓ Invalid names properly rejected")

    print("\nTesting empty and null handling...")
    test.test_empty_and_null_handling()
    print("✓ Empty and null handling passed")

    print("\nTesting mixed valid/invalid...")
    test.test_mixed_valid_invalid()
    print("✓ Mixed valid/invalid handling passed")

    print("\nTesting edge cases...")
    test.test_edge_cases()
    print("✓ Edge cases passed")

    print("\n✅ All tests passed!")
