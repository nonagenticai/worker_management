"""Unit tests for type_definitions/constraint_types.py."""

import pytest
from pydantic import BaseModel, ValidationError

from src.type_definitions.constraint_types import (
    SafeCodeString,
    SafeString,
    SafeStringLong,
    SafeStringMedium,
    SafeStringShort,
)


class ShortModel(BaseModel):
    val: SafeStringShort


class MediumModel(BaseModel):
    val: SafeStringMedium


class LongModel(BaseModel):
    val: SafeStringLong


class CodeModel(BaseModel):
    val: SafeCodeString


class PlainModel(BaseModel):
    val: SafeString


class TestSafeStringShort:
    def test_valid(self):
        m = ShortModel(val="hello")
        assert m.val == "hello"

    def test_too_long(self):
        with pytest.raises(ValidationError):
            ShortModel(val="x" * 257)

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            ShortModel(val="")

    def test_control_chars_rejected(self):
        with pytest.raises(ValidationError):
            ShortModel(val="bad\x00char")


class TestSafeStringMedium:
    def test_valid(self):
        m = MediumModel(val="description here")
        assert m.val == "description here"

    def test_too_long(self):
        with pytest.raises(ValidationError):
            MediumModel(val="x" * 1001)


class TestSafeStringLong:
    def test_valid(self):
        m = LongModel(val="long text " * 100)
        assert len(m.val) > 0

    def test_too_long(self):
        with pytest.raises(ValidationError):
            LongModel(val="x" * 10001)


class TestSafeCodeString:
    def test_allows_empty(self):
        m = CodeModel(val="")
        assert m.val == ""

    def test_allows_newlines_tabs(self):
        m = CodeModel(val="def f():\n\treturn 1\n")
        assert "\n" in m.val

    def test_too_long(self):
        with pytest.raises(ValidationError):
            CodeModel(val="x" * 50001)


class TestSafeString:
    def test_no_length_constraint(self):
        m = PlainModel(val="x" * 10000)
        assert len(m.val) == 10000

    def test_rejects_null_bytes(self):
        with pytest.raises(ValidationError):
            PlainModel(val="has\x00null")

    def test_allows_normal_chars(self):
        m = PlainModel(val="Hello, World! 123")
        assert m.val == "Hello, World! 123"
