"""AES-256-GCM secret encryption for per-tenant BYO LLM keys.

Mirrors the wire format of commonpractice's ``backend/src/host/secret-box.ts``
so the scheme is identical across services (each service still uses its OWN
master key — there is no cross-service interop, per the per-service-secrets
invariant). The wire format is::

    base64( iv(12 bytes) || tag(16 bytes) || ciphertext )

Master key resolution from ``WM_SECRET_KEY`` (in resolution order):

1. 64-char hex string  -> 32 raw bytes (preferred for production).
2. base64 / base64url that decodes to exactly 32 bytes -> used directly.
3. anything else        -> treated as a passphrase, scrypt-derived to 32 bytes
   (deterministic; a warning is logged once so operators know it is weaker
   than a real 256-bit key).

Decryption is fail-closed: a wrong key, truncated blob, or tampered ciphertext
raises :class:`SecretDecryptError`. Callers (the resolution seam) treat that as
"no usable key" and fall back to system routing — they never surface plaintext,
crash, or put an empty key on the wire.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

_IV_LEN = 12
_TAG_LEN = 16
_KEY_LEN = 32

# Fixed salt for passphrase derivation. The master key must be STABLE for a
# given WM_SECRET_KEY (we cannot store a per-key random salt), so the salt is a
# constant. scrypt with a fixed salt is still a sound KDF for this purpose.
_SCRYPT_SALT = b"worker-management/byo-llm/secret-box/v1"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1

_ENV_VAR = "WM_SECRET_KEY"

_passphrase_warned = False


class SecretBoxError(Exception):
    """Base class for secret-box failures."""


class SecretKeyMissingError(SecretBoxError):
    """WM_SECRET_KEY is unset or empty when encryption/decryption is attempted."""


class SecretDecryptError(SecretBoxError):
    """Ciphertext could not be decrypted (wrong key, truncated, or tampered)."""


def _derive_passphrase(raw: str) -> bytes:
    global _passphrase_warned  # noqa: PLW0603 -- one-shot operator warning
    if not _passphrase_warned:
        logger.warning(
            "%s is a passphrase, not a 32-byte key; deriving via scrypt. "
            "Prefer a 64-char hex or base64-encoded 32-byte key in production.",
            _ENV_VAR,
        )
        _passphrase_warned = True
    return hashlib.scrypt(
        raw.encode("utf-8"),
        salt=_SCRYPT_SALT,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_LEN,
    )


def _master_key() -> bytes:
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        raise SecretKeyMissingError(f"{_ENV_VAR} is not set")

    # 1. 64-char hex -> 32 bytes
    if len(raw) == 64:
        try:
            return bytes.fromhex(raw)
        except ValueError:
            pass

    # 2. base64 / base64url decoding to exactly 32 bytes
    try:
        candidate = base64.b64decode(raw, validate=True)
        if len(candidate) == _KEY_LEN:
            return candidate
    except binascii.Error, ValueError:
        pass
    try:
        candidate = base64.urlsafe_b64decode(raw)
        if len(candidate) == _KEY_LEN:
            return candidate
    except binascii.Error, ValueError:
        pass

    # 3. passphrase
    return _derive_passphrase(raw)


def encrypt_secret(plaintext: str) -> str:
    """Encrypt ``plaintext`` and return a base64 ``iv||tag||ciphertext`` blob.

    Raises:
        SecretKeyMissingError: if ``WM_SECRET_KEY`` is unset.
    """
    if plaintext == "":
        raise ValueError("refusing to encrypt an empty secret")
    key = _master_key()
    iv = os.urandom(_IV_LEN)
    # cryptography returns ciphertext || tag (tag = trailing 16 bytes).
    ct_and_tag = AESGCM(key).encrypt(iv, plaintext.encode("utf-8"), None)
    ciphertext, tag = ct_and_tag[:-_TAG_LEN], ct_and_tag[-_TAG_LEN:]
    return base64.b64encode(iv + tag + ciphertext).decode("ascii")


def decrypt_secret(blob: str) -> str:
    """Decrypt a base64 ``iv||tag||ciphertext`` blob produced by :func:`encrypt_secret`.

    Raises:
        SecretKeyMissingError: if ``WM_SECRET_KEY`` is unset.
        SecretDecryptError: on wrong key, truncated blob, or tampering.
    """
    key = _master_key()
    try:
        buf = base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SecretDecryptError("ciphertext is not valid base64") from exc

    if len(buf) < _IV_LEN + _TAG_LEN:
        raise SecretDecryptError("ciphertext too short")

    iv = buf[:_IV_LEN]
    tag = buf[_IV_LEN : _IV_LEN + _TAG_LEN]
    ciphertext = buf[_IV_LEN + _TAG_LEN :]
    try:
        plaintext = AESGCM(key).decrypt(iv, ciphertext + tag, None)
    except InvalidTag as exc:
        raise SecretDecryptError("authentication failed (wrong key or tampered)") from exc
    return plaintext.decode("utf-8")


def has_secret_key() -> bool:
    """Return True if ``WM_SECRET_KEY`` is configured (non-empty)."""
    return bool(os.environ.get(_ENV_VAR, "").strip())
