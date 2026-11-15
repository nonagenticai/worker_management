"""Tests for multimodal MessageRequest content blocks (text + image)."""

from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from src.api.workers.models import (
    ContentBlock,
    ImageBlock,
    MessageRequest,
    TextBlock,
)

# 1×1 PNG (transparent) base64-encoded.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class TestTextBlock:
    def test_minimal(self):
        block = TextBlock(type="text", text="hello")
        assert block.type == "text"
        assert block.text == "hello"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            TextBlock(type="text", text="")


class TestImageBlock:
    def test_minimal(self):
        block = ImageBlock(type="image", data=_TINY_PNG_B64, media_type="image/png")
        assert block.type == "image"
        assert block.data == _TINY_PNG_B64
        assert block.media_type == "image/png"

    def test_default_media_type_is_jpeg(self):
        block = ImageBlock(type="image", data=_TINY_PNG_B64)
        assert block.media_type == "image/jpeg"

    def test_invalid_base64_rejected(self):
        with pytest.raises(ValidationError):
            ImageBlock(type="image", data="not valid base64!!!", media_type="image/png")

    def test_unsupported_media_type_rejected(self):
        # Pass via model_validate so the Literal constraint runs at the
        # validation boundary (direct ctor call would fail static type
        # check, which itself is the right behavior).
        with pytest.raises(ValidationError):
            ImageBlock.model_validate(
                {"type": "image", "data": _TINY_PNG_B64, "media_type": "image/svg+xml"}
            )

    def test_oversize_rejected(self):
        # Build a >10MB base64 string. 14M chars b64 = ~10.5 MB raw.
        oversize = "A" * 14_000_001
        # Valid base64 first then size check should fire.
        with pytest.raises(ValidationError):
            ImageBlock(type="image", data=oversize, media_type="image/jpeg")


class TestMessageRequestStringMode:
    def test_plain_string(self):
        req = MessageRequest(message="describe this site")
        assert req.message == "describe this site"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            MessageRequest(message="")


class TestMessageRequestMultimodal:
    def test_text_only_block_list(self):
        req = MessageRequest(message=[TextBlock(type="text", text="describe this image")])
        assert isinstance(req.message, list)
        assert len(req.message) == 1
        assert req.message[0].type == "text"

    def test_text_plus_image_block_list(self):
        req = MessageRequest(
            message=[
                TextBlock(type="text", text="What's in this image?"),
                ImageBlock(type="image", data=_TINY_PNG_B64, media_type="image/png"),
            ]
        )
        assert isinstance(req.message, list)
        assert len(req.message) == 2
        assert req.message[0].type == "text"
        assert req.message[1].type == "image"

    def test_dict_form_via_pydantic_validation(self):
        """API callers send JSON dicts; validate they parse via the discriminator."""
        req = MessageRequest.model_validate(
            {
                "message": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image",
                        "data": _TINY_PNG_B64,
                        "media_type": "image/jpeg",
                    },
                ]
            }
        )
        assert isinstance(req.message, list)
        assert req.message[0].type == "text"
        image_block = req.message[1]
        assert image_block.type == "image"
        assert isinstance(image_block, ImageBlock)
        assert image_block.data == _TINY_PNG_B64

    def test_empty_list_rejected(self):
        with pytest.raises(ValidationError):
            MessageRequest(message=[])

    def test_unknown_block_type_rejected(self):
        with pytest.raises(ValidationError):
            MessageRequest.model_validate({"message": [{"type": "audio", "data": "x"}]})


class TestPydanticAIPartsConversion:
    """Tests for PydanticAIAgent._to_pydantic_ai_parts (boundary converter)."""

    def test_string_passthrough(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        result = PydanticAIAgent._to_pydantic_ai_parts("hello world")
        assert result == "hello world"

    def test_list_with_text_only(self):
        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        blocks = [TextBlock(type="text", text="describe")]
        result = PydanticAIAgent._to_pydantic_ai_parts(blocks)
        assert result == ["describe"]

    def test_list_with_image_yields_binarycontent(self):
        from pydantic_ai import BinaryContent

        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        blocks = [
            TextBlock(type="text", text="What is this?"),
            ImageBlock(type="image", data=_TINY_PNG_B64, media_type="image/png"),
        ]
        result = PydanticAIAgent._to_pydantic_ai_parts(blocks)
        assert len(result) == 2
        assert result[0] == "What is this?"
        assert isinstance(result[1], BinaryContent)
        assert result[1].media_type == "image/png"
        # BinaryContent.data holds raw decoded bytes.
        assert result[1].data == base64.b64decode(_TINY_PNG_B64)

    def test_list_with_dict_blocks(self):
        """Direct API callers may pass dicts pre-validation; converter handles them."""
        from pydantic_ai import BinaryContent

        from src.worker.agents.pydantic_ai_agent import PydanticAIAgent

        blocks = [
            {"type": "text", "text": "hi"},
            {"type": "image", "data": _TINY_PNG_B64, "media_type": "image/jpeg"},
        ]
        result = PydanticAIAgent._to_pydantic_ai_parts(blocks)
        assert result[0] == "hi"
        assert isinstance(result[1], BinaryContent)


class TestNonPydanticAgentsRejectMultimodal:
    """Other agent types raise NotImplementedError on multimodal input."""

    @pytest.mark.asyncio
    async def test_cli_subclass_rejects_list_prompt(self):
        """ClaudeCodeAgent inherits from CLIAgent — picks up the run() guard."""
        from unittest.mock import MagicMock

        from src.worker.agents.claude_code_agent import ClaudeCodeAgent

        agent = ClaudeCodeAgent.__new__(ClaudeCodeAgent)
        agent.name = "test"
        agent.config = MagicMock(use_agent_host=False, use_k8s=False, use_proxy=False)

        with pytest.raises(NotImplementedError, match="multimodal"):
            await agent.run([{"type": "text", "text": "hi"}])

    @pytest.mark.asyncio
    async def test_remote_cli_agent_rejects_list_prompt(self):
        from unittest.mock import MagicMock

        from src.worker.agents.remote_cli_agent import RemoteCLIAgent

        agent = RemoteCLIAgent.__new__(RemoteCLIAgent)
        agent.name = "test"
        agent.config = MagicMock(cli="claude-code", container_id="c1")

        with pytest.raises(NotImplementedError, match="multimodal"):
            await agent.run([{"type": "text", "text": "hi"}])


__all__ = ["ContentBlock"]  # silence ruff F401 for the type alias re-export
