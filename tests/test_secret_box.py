"""Unit tests for src/security/secret_box.py (AES-256-GCM BYO key encryption)."""

from __future__ import annotations

import base64
import secrets

import pytest

from src.security import secret_box
from src.security.secret_box import (
    SecretDecryptError,
    SecretKeyMissingError,
    decrypt_secret,
    encrypt_secret,
    has_secret_key,
)

HEX_KEY = secrets.token_hex(32)  # 64-char hex -> 32 bytes


@pytest.fixture
def hex_key(monkeypatch):
    monkeypatch.setenv("WM_SECRET_KEY", HEX_KEY)
    return HEX_KEY


def test_roundtrip(hex_key):
    blob = encrypt_secret("sk-proj-abc123")
    assert decrypt_secret(blob) == "sk-proj-abc123"


def test_wire_format_is_iv_tag_ct(hex_key):
    """base64(iv(12) || tag(16) || ciphertext) — mirrors commonpractice."""
    blob = encrypt_secret("x")
    raw = base64.b64decode(blob)
    assert len(raw) == 12 + 16 + 1  # 1-byte plaintext


def test_ciphertext_is_nondeterministic(hex_key):
    assert encrypt_secret("same") != encrypt_secret("same")  # random IV


def test_tamper_fails_closed(hex_key):
    blob = encrypt_secret("secret")
    raw = bytearray(base64.b64decode(blob))
    raw[-1] ^= 0x01
    tampered = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(SecretDecryptError):
        decrypt_secret(tampered)


def test_wrong_key_fails_closed(hex_key, monkeypatch):
    blob = encrypt_secret("secret")
    monkeypatch.setenv("WM_SECRET_KEY", secrets.token_hex(32))
    with pytest.raises(SecretDecryptError):
        decrypt_secret(blob)


def test_truncated_blob_fails_closed(hex_key):
    with pytest.raises(SecretDecryptError):
        decrypt_secret(base64.b64encode(b"short").decode())


def test_non_base64_fails_closed(hex_key):
    with pytest.raises(SecretDecryptError):
        decrypt_secret("!!!not base64!!!")


def test_base64_key_accepted(monkeypatch):
    key_b64 = base64.b64encode(secrets.token_bytes(32)).decode()
    monkeypatch.setenv("WM_SECRET_KEY", key_b64)
    assert decrypt_secret(encrypt_secret("v")) == "v"


def test_passphrase_key_is_deterministic(monkeypatch):
    monkeypatch.setenv("WM_SECRET_KEY", "a-human-passphrase-not-32-bytes")
    blob = encrypt_secret("v")
    # Same passphrase still decrypts (scrypt derivation is stable).
    assert decrypt_secret(blob) == "v"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("WM_SECRET_KEY", raising=False)
    assert has_secret_key() is False
    with pytest.raises(SecretKeyMissingError):
        encrypt_secret("v")


def test_empty_plaintext_rejected(hex_key):
    with pytest.raises(ValueError, match="empty"):
        encrypt_secret("")


def test_has_secret_key_true(hex_key):
    assert has_secret_key() is True


def test_module_exports():
    assert hasattr(secret_box, "encrypt_secret")
    assert hasattr(secret_box, "decrypt_secret")
